"""Adaptive CE-priority PCGrad restricted to the multimodal projector."""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn


STATE_FILE = "adaptive_projector_pcgrad_state.pt"
METADATA_FILE = "adaptive_projector_pcgrad_metadata.json"
RESOLVED_CONFIG_FILE = "adaptive_projector_pcgrad_config.json"
STATE_VERSION = 1


@dataclass(frozen=True)
class AdaptiveProjectorPCGradConfig:
    stage: int
    planned_optimizer_steps: int
    max_aux_ratio: float = 0.10
    warmup_ratio: float = 0.03
    norm_ema_beta: float = 0.95
    lambda_max: float = 1.0
    ce_norm_floor: float = 1e-12
    aux_norm_floor: float = 1e-12
    residual_rtol: float = 1e-7

    def validate(self) -> "AdaptiveProjectorPCGradConfig":
        if self.stage not in (1, 2):
            raise ValueError(f"adaptive PCGrad stage must be 1 or 2, got {self.stage}.")
        if int(self.planned_optimizer_steps) <= 0:
            raise ValueError("planned_optimizer_steps must be positive.")
        finite_nonnegative = (
            "max_aux_ratio",
            "warmup_ratio",
            "lambda_max",
            "residual_rtol",
        )
        for name in finite_nonnegative:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative, got {value}.")
        if self.warmup_ratio > 1.0:
            raise ValueError("warmup_ratio must be in [0, 1].")
        if not 0.0 <= float(self.norm_ema_beta) < 1.0:
            raise ValueError("norm_ema_beta must be in [0, 1).")
        for name in ("ce_norm_floor", "aux_norm_floor"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive, got {value}.")
        return self

    @property
    def warmup_steps(self) -> int:
        return int(math.ceil(float(self.warmup_ratio) * int(self.planned_optimizer_steps)))

    def to_dict(self) -> Dict[str, object]:
        result = asdict(self)
        result["warmup_steps"] = self.warmup_steps
        return result


def _all_reduce_sum(tensor: torch.Tensor, process_group=None) -> None:
    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size(group=process_group) > 1
    ):
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=process_group)


def _sequence_stats(
    main: Sequence[Optional[torch.Tensor]],
    auxiliary: Sequence[Optional[torch.Tensor]],
    reference: torch.Tensor,
    *,
    distributed_shards: bool,
    process_group=None,
) -> torch.Tensor:
    if len(main) != len(auxiliary):
        raise ValueError("Main and auxiliary projector gradient lists must have equal length.")
    stats = torch.zeros(3, device=reference.device, dtype=torch.float64)
    with torch.no_grad():
        for g, b in zip(main, auxiliary):
            gf = g.detach().reshape(-1).float() if g is not None else None
            bf = b.detach().reshape(-1).float() if b is not None else None
            if gf is not None:
                stats[1].add_(torch.dot(gf, gf).double())
            if bf is not None:
                stats[2].add_(torch.dot(bf, bf).double())
                if gf is not None:
                    if gf.shape != bf.shape:
                        raise ValueError("Mismatched projector gradient shapes.")
                    stats[0].add_(torch.dot(gf, bf).double())
        if distributed_shards:
            _all_reduce_sum(stats, process_group)
    return stats


def _transform_auxiliary(
    main: Sequence[Optional[torch.Tensor]],
    auxiliary: Sequence[Optional[torch.Tensor]],
    *,
    auxiliary_scale: float,
    main_coefficient: float,
) -> List[Optional[torch.Tensor]]:
    transformed: List[Optional[torch.Tensor]] = []
    with torch.no_grad():
        for g, b in zip(main, auxiliary):
            if b is None and (g is None or main_coefficient == 0.0):
                transformed.append(None)
                continue
            template = b if b is not None else g
            value = torch.zeros_like(template)
            if b is not None and auxiliary_scale != 0.0:
                value.add_(b.detach(), alpha=auxiliary_scale)
            if g is not None and main_coefficient != 0.0:
                value.add_(g.detach(), alpha=main_coefficient)
            transformed.append(value)
    return transformed


def _merge_gradients(
    main: Sequence[Optional[torch.Tensor]],
    update: Sequence[Optional[torch.Tensor]],
) -> List[Optional[torch.Tensor]]:
    merged: List[Optional[torch.Tensor]] = []
    with torch.no_grad():
        for g, u in zip(main, update):
            if g is None and u is None:
                merged.append(None)
            elif g is None:
                merged.append(u.detach().clone())
            elif u is None:
                merged.append(g.detach().clone())
            else:
                value = g.detach().clone()
                value.add_(u.detach())
                merged.append(value)
    return merged


class AdaptiveProjectorPCGradController:
    """Prepare a projector-gradient proposal; commit state only after a real step."""

    def __init__(self, config: AdaptiveProjectorPCGradConfig):
        self.config = config.validate()
        self.successful_steps = 0
        self.ema_g: Optional[float] = None
        self.ema_b: Optional[float] = None
        self.valid_proposals = 0
        self.conflict_proposals = 0
        self.lambda_ceiling_hits = 0
        self.near_zero_skips = 0
        self.overflow_skips = 0
        self._pending: Optional[Dict[str, object]] = None

    @property
    def rho(self) -> float:
        warmup_steps = self.config.warmup_steps
        if warmup_steps == 0:
            return float(self.config.max_aux_ratio)
        fraction = min(1.0, float(self.successful_steps + 1) / float(warmup_steps))
        return float(self.config.max_aux_ratio) * fraction

    @staticmethod
    def _ema_candidate(old: Optional[float], value: float, beta: float) -> float:
        return value if old is None else beta * old + (1.0 - beta) * value

    def prepare(
        self,
        main_gradients: Sequence[Optional[torch.Tensor]],
        auxiliary_gradients: Sequence[Optional[torch.Tensor]],
        *,
        valid_count: float,
        reference_tensor: torch.Tensor,
        distributed_shards: bool = False,
        process_group=None,
    ) -> Tuple[List[Optional[torch.Tensor]], Dict[str, float]]:
        if self._pending is not None:
            raise RuntimeError("Adaptive projector PCGrad proposal was not finished before preparing another.")
        valid_count = float(valid_count)
        if not math.isfinite(valid_count) or valid_count < 0.0:
            raise ValueError(f"valid_count must be finite and non-negative, got {valid_count}.")

        stats = _sequence_stats(
            main_gradients,
            auxiliary_gradients,
            reference_tensor,
            distributed_shards=distributed_shards,
            process_group=process_group,
        )
        stats_cpu = stats.detach().cpu()
        if not bool(torch.isfinite(stats_cpu).all().item()):
            raise RuntimeError("Adaptive projector PCGrad received NaN/Inf gradient statistics.")
        # One tiny device-to-host transfer is enough for the scalar decision;
        # avoid issuing a separate D2H copy for every statistic.
        dot_b_g, raw_g_norm_sq, raw_b_norm_sq = stats_cpu.tolist()
        dot_b_g = float(dot_b_g)
        g_norm_sq = max(0.0, float(raw_g_norm_sq))
        b_norm_sq = max(0.0, float(raw_b_norm_sq))
        g_norm = math.sqrt(g_norm_sq)
        b_norm = math.sqrt(b_norm_sq)
        rho_t = self.rho
        cosine = 0.0
        if g_norm > self.config.ce_norm_floor and b_norm > self.config.aux_norm_floor:
            cosine = dot_b_g / (g_norm * b_norm)

        usable = (
            valid_count > 0.0
            and g_norm > self.config.ce_norm_floor
            and b_norm > self.config.aux_norm_floor
        )
        logs: Dict[str, float] = {
            "ce_norm": g_norm,
            "raw_cka_norm": b_norm,
            "raw_cosine": cosine,
            "conflict": float(dot_b_g < 0.0 and usable),
            "rho": rho_t,
            "valid_count": valid_count,
            "successful_steps": float(self.successful_steps),
            "warmup_steps": float(self.config.warmup_steps),
            "lambda": 0.0,
            "lambda_at_ceiling": 0.0,
            "projected_norm": 0.0,
            "projection_retention": 0.0,
            "effective_aux_ratio": 0.0,
            "cap_scale": 0.0,
            "near_zero_skip": float(not usable and valid_count > 0.0),
        }

        if not usable:
            self._pending = {
                "ema_g": None,
                "ema_b": None,
                "valid": False,
                "ceiling": False,
                "conflict": False,
                "near_zero": bool(valid_count > 0.0),
            }
            # CE-only window: preserve the original gradient objects so the
            # backend can skip a full projector clone and copy-back.
            return list(main_gradients), logs

        candidate_g = self._ema_candidate(self.ema_g, g_norm, self.config.norm_ema_beta)
        candidate_b = self._ema_candidate(self.ema_b, b_norm, self.config.norm_ema_beta)
        lambda_t = min(
            float(self.config.lambda_max),
            rho_t * candidate_g / max(candidate_b, float(self.config.aux_norm_floor)),
        )
        at_ceiling = bool(lambda_t >= float(self.config.lambda_max) and self.config.lambda_max > 0.0)
        dot_a_g = lambda_t * dot_b_g
        projection_coefficient = 0.0
        if dot_a_g < 0.0:
            # No epsilon here: the nonzero guard above makes the real denominator safe.
            projection_coefficient = dot_a_g / g_norm_sq

        projected = _transform_auxiliary(
            main_gradients,
            auxiliary_gradients,
            auxiliary_scale=lambda_t,
            main_coefficient=-projection_coefficient,
        )
        projected_stats = _sequence_stats(
            main_gradients,
            projected,
            reference_tensor,
            distributed_shards=distributed_shards,
            process_group=process_group,
        )
        projected_values = projected_stats.detach().cpu().tolist()
        residual_dot = float(projected_values[0])
        projected_norm = math.sqrt(max(0.0, float(projected_values[2])))
        if residual_dot < 0.0:
            # The projection coefficient and the distributed dot reduction are
            # evaluated through different FP32 operation orders. Correct every
            # finite negative residual first, then verify the corrected result;
            # rejecting before this correction produces false failures near the
            # half-space boundary.
            correction = -residual_dot / g_norm_sq
            projected = _transform_auxiliary(
                main_gradients,
                projected,
                auxiliary_scale=1.0,
                main_coefficient=correction,
            )
            corrected_stats = _sequence_stats(
                main_gradients,
                projected,
                reference_tensor,
                distributed_shards=distributed_shards,
                process_group=process_group,
            )
            corrected_values = corrected_stats.detach().cpu().tolist()
            residual_dot = float(corrected_values[0])
            projected_norm = math.sqrt(max(0.0, float(corrected_values[2])))
            corrected_scale = g_norm * projected_norm
            corrected_tolerance = max(
                float(self.config.residual_rtol) * corrected_scale,
                8.0 * torch.finfo(torch.float32).eps * corrected_scale,
                1e-30,
            )
            if residual_dot < -corrected_tolerance:
                raise RuntimeError(
                    "One-sided projection retained a material conflict after correction: "
                    f"dot={residual_dot}, tolerance={corrected_tolerance}."
                )

        allowed_norm = rho_t * g_norm
        cap_scale = 0.0 if projected_norm == 0.0 else min(1.0, allowed_norm / projected_norm)
        with torch.no_grad():
            for gradient in projected:
                if gradient is not None:
                    gradient.mul_(cap_scale)
        update_norm = cap_scale * projected_norm
        update_dot = cap_scale * residual_dot
        invariant_scale = g_norm * update_norm
        invariant_tolerance = max(
            float(self.config.residual_rtol) * invariant_scale,
            8.0 * torch.finfo(torch.float32).eps * invariant_scale,
            1e-30,
        )
        if update_dot < -invariant_tolerance:
            raise RuntimeError("Adaptive projector PCGrad violated <g,u> >= 0.")
        cap_tolerance = max(
            float(self.config.residual_rtol) * allowed_norm,
            8.0 * torch.finfo(torch.float32).eps * allowed_norm,
            1e-30,
        )
        if update_norm > allowed_norm + cap_tolerance:
            raise RuntimeError("Adaptive projector PCGrad violated its current-CE hard cap.")

        raw_scaled_norm = lambda_t * b_norm
        logs.update({
            "ema_ce_norm_proposal": candidate_g,
            "ema_cka_norm_proposal": candidate_b,
            "lambda": lambda_t,
            "lambda_at_ceiling": float(at_ceiling),
            "projected_norm": projected_norm,
            "projection_retention": (
                0.0 if raw_scaled_norm <= self.config.aux_norm_floor else projected_norm / raw_scaled_norm
            ),
            "effective_aux_ratio": update_norm / g_norm,
            "cap_scale": cap_scale,
            "post_projection_dot": update_dot,
        })
        self._pending = {
            "ema_g": candidate_g,
            "ema_b": candidate_b,
            "valid": True,
            "ceiling": at_ceiling,
            "conflict": bool(dot_b_g < 0.0),
            "near_zero": False,
        }
        return _merge_gradients(main_gradients, projected), logs

    def finish(self, *, optimizer_stepped: bool) -> None:
        if self._pending is None:
            return
        pending = self._pending
        self._pending = None
        if not optimizer_stepped:
            self.overflow_skips += 1
            return
        self.successful_steps += 1
        if pending["near_zero"]:
            self.near_zero_skips += 1
        if pending["valid"]:
            self.ema_g = float(pending["ema_g"])
            self.ema_b = float(pending["ema_b"])
            self.valid_proposals += 1
            self.conflict_proposals += int(bool(pending["conflict"]))
            self.lambda_ceiling_hits += int(bool(pending["ceiling"]))

    def state_dict(self) -> Dict[str, object]:
        if self._pending is not None:
            raise RuntimeError("Adaptive PCGrad state may only be saved at an optimizer boundary.")
        return {
            "version": STATE_VERSION,
            "config": self.config.to_dict(),
            "successful_steps": self.successful_steps,
            "ema_g": self.ema_g,
            "ema_b": self.ema_b,
            "valid_proposals": self.valid_proposals,
            "conflict_proposals": self.conflict_proposals,
            "lambda_ceiling_hits": self.lambda_ceiling_hits,
            "near_zero_skips": self.near_zero_skips,
            "overflow_skips": self.overflow_skips,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if int(state.get("version", -1)) != STATE_VERSION:
            raise RuntimeError("Unsupported adaptive projector PCGrad controller state version.")
        saved_config = state.get("config", {})
        if not isinstance(saved_config, Mapping):
            raise RuntimeError("Adaptive PCGrad checkpoint has no resolved controller config.")
        if int(saved_config.get("stage", -1)) != self.config.stage:
            raise RuntimeError("Refusing to load adaptive PCGrad controller state from another stage.")
        expected_config = self.config.to_dict()
        mismatches = {
            key: (saved_config.get(key), value)
            for key, value in expected_config.items()
            if saved_config.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                "Adaptive PCGrad resolved config changed on same-stage resume: "
                f"{mismatches}."
            )
        successful_steps = int(state.get("successful_steps", 0))
        ema_g = None if state.get("ema_g") is None else float(state["ema_g"])
        ema_b = None if state.get("ema_b") is None else float(state["ema_b"])
        counters = {
            "valid_proposals": int(state.get("valid_proposals", 0)),
            "conflict_proposals": int(state.get("conflict_proposals", 0)),
            "lambda_ceiling_hits": int(state.get("lambda_ceiling_hits", 0)),
            "near_zero_skips": int(state.get("near_zero_skips", 0)),
            "overflow_skips": int(state.get("overflow_skips", 0)),
        }
        if successful_steps < 0 or successful_steps > self.config.planned_optimizer_steps:
            raise RuntimeError("Adaptive PCGrad checkpoint has an invalid successful-step count.")
        if any(value < 0 for value in counters.values()):
            raise RuntimeError("Adaptive PCGrad checkpoint has a negative diagnostic counter.")
        if counters["valid_proposals"] > successful_steps:
            raise RuntimeError("Adaptive PCGrad checkpoint has more valid proposals than successful steps.")
        if counters["conflict_proposals"] > counters["valid_proposals"]:
            raise RuntimeError("Adaptive PCGrad checkpoint has an invalid conflict counter.")
        if counters["lambda_ceiling_hits"] > counters["valid_proposals"]:
            raise RuntimeError("Adaptive PCGrad checkpoint has an invalid lambda-ceiling counter.")
        if counters["near_zero_skips"] > successful_steps:
            raise RuntimeError("Adaptive PCGrad checkpoint has an invalid near-zero counter.")
        for name, value in (("ema_g", ema_g), ("ema_b", ema_b)):
            if value is not None and (not math.isfinite(value) or value <= 0.0):
                raise RuntimeError(f"Adaptive PCGrad checkpoint has invalid {name}={value}.")
        if (ema_g is None) != (ema_b is None):
            raise RuntimeError("Adaptive PCGrad checkpoint must restore both norm EMAs or neither.")
        if (counters["valid_proposals"] == 0) != (ema_g is None):
            raise RuntimeError(
                "Adaptive PCGrad checkpoint EMA presence is inconsistent with valid proposals."
            )

        self.successful_steps = successful_steps
        self.ema_g = ema_g
        self.ema_b = ema_b
        self.valid_proposals = counters["valid_proposals"]
        self.conflict_proposals = counters["conflict_proposals"]
        self.lambda_ceiling_hits = counters["lambda_ceiling_hits"]
        self.near_zero_skips = counters["near_zero_skips"]
        self.overflow_skips = counters["overflow_skips"]
        self._pending = None


def projector_signature(projector: nn.Module) -> List[Dict[str, object]]:
    result = []
    seen = set()
    for name, parameter in projector.named_parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        result.append({
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "requires_grad": bool(parameter.requires_grad),
        })
    return result


def projector_sha256(projector: nn.Module) -> str:
    digest = hashlib.sha256()
    seen = set()
    with torch.no_grad():
        for name, parameter in projector.named_parameters():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            digest.update(name.encode("utf-8"))
            digest.update(str(list(parameter.shape)).encode("utf-8"))
            digest.update(str(parameter.dtype).encode("utf-8"))
            raw = parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            digest.update(raw)
    return digest.hexdigest()


def write_json(path: str, payload: Mapping[str, object]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload
