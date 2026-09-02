import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch


VSP_GRAD_EPS = 1e-12
VSP_GRAD_STAT_CHUNK_SIZE = 1_048_576
VSP_AUXILIARY_NAMES = ("proj", "final")
VSP_GROUP_NAMES = ("projector", "llm")
VSP_PROJECTOR_KEYWORDS = ("mm_projector", "vision_resampler", "vision_tower")


@dataclass
class VSPParameterGroup:
    name: str
    param_names: List[str]
    params: List[torch.nn.Parameter]


def finite_float(value, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def validate_vsp_gradient_config(config) -> None:
    eps = finite_float(getattr(config, "vsp_grad_eps", VSP_GRAD_EPS), VSP_GRAD_EPS)
    if eps <= 0.0:
        raise ValueError(f"vsp_grad_eps must be finite and positive, got {eps}.")
    threshold = finite_float(getattr(config, "vsp_pcgrad_threshold", 0.05), 0.05)
    if threshold < 0.0:
        raise ValueError(f"vsp_pcgrad_threshold must be non-negative, got {threshold}.")
    for attr in ("vsp_proj_max_grad_ratio", "vsp_llm_max_grad_ratio"):
        ratio = finite_float(getattr(config, attr, 0.0), 0.0)
        if ratio < 0.0:
            raise ValueError(f"{attr} must be non-negative, got {ratio}.")
    beta = finite_float(getattr(config, "vsp_grad_ema_beta", 0.95), 0.95)
    if beta < 0.0 or beta >= 1.0:
        raise ValueError(f"vsp_grad_ema_beta must be in [0, 1), got {beta}.")
    interval = int(getattr(config, "vsp_grad_log_interval", 10) or 10)
    if interval <= 0:
        raise ValueError(f"vsp_grad_log_interval must be positive, got {interval}.")


def vsp_controller_requested(config) -> bool:
    return bool(
        getattr(config, "vsp_gradient_diagnostics", False)
        or getattr(config, "vsp_asymmetric_pcgrad", False)
        or getattr(config, "vsp_norm_cap", False)
        or getattr(config, "use_pcgrad", False)
    )


def vsp_rewrites_gradients(config) -> bool:
    return bool(
        getattr(config, "vsp_asymmetric_pcgrad", False)
        or getattr(config, "vsp_norm_cap", False)
        or getattr(config, "use_pcgrad", False)
    )


def is_projector_parameter(name: str) -> bool:
    return any(keyword in name for keyword in VSP_PROJECTOR_KEYWORDS)


def collect_vsp_parameter_groups(model) -> Dict[str, VSPParameterGroup]:
    groups = {
        "projector": VSPParameterGroup("projector", [], []),
        "llm": VSPParameterGroup("llm", [], []),
    }
    seen_param_ids = set()
    for name, param in model.named_parameters():
        if not getattr(param, "requires_grad", False):
            continue
        param_id = id(param)
        if param_id in seen_param_ids:
            continue
        seen_param_ids.add(param_id)
        group_name = "projector" if is_projector_parameter(name) else "llm"
        groups[group_name].param_names.append(name)
        groups[group_name].params.append(param)
    return groups


def _is_usable_loss(loss: Optional[torch.Tensor]) -> bool:
    return torch.is_tensor(loss) and bool(loss.requires_grad)


def validate_vsp_losses(
    lm_loss: torch.Tensor,
    vsp_proj_loss: Optional[torch.Tensor] = None,
    vsp_final_loss: Optional[torch.Tensor] = None,
) -> None:
    for name, loss in (
        ("lm_loss", lm_loss),
        ("vsp_proj_loss", vsp_proj_loss),
        ("vsp_final_loss", vsp_final_loss),
    ):
        if loss is None:
            continue
        if not torch.is_tensor(loss):
            raise TypeError(f"{name} must be a tensor or None.")
        detached = loss.detach()
        if detached.numel() != 1:
            raise ValueError(f"{name} must be scalar, got shape {tuple(detached.shape)}.")
        if not torch.isfinite(detached).all():
            raise RuntimeError(f"{name} is NaN or Inf; VSP gradient control skipped.")


def _dense_float_gradient(gradient: torch.Tensor) -> torch.Tensor:
    gradient = gradient.detach()
    if gradient.is_sparse:
        gradient = gradient.coalesce().to_dense()
    return gradient.float()


def _empty_like_gradients(params: Sequence[torch.nn.Parameter]) -> List[Optional[torch.Tensor]]:
    return [None for _ in params]


def _extract_gradients(
    loss: Optional[torch.Tensor],
    params: Sequence[torch.nn.Parameter],
    retain_graph: bool = True,
) -> List[Optional[torch.Tensor]]:
    if not params:
        return []
    if not _is_usable_loss(loss):
        return _empty_like_gradients(params)
    return list(
        torch.autograd.grad(
            loss,
            params,
            retain_graph=retain_graph,
            create_graph=False,
            allow_unused=True,
        )
    )


def _accumulate_sequence_stats(
    main_gradients: Sequence[Optional[torch.Tensor]],
    auxiliary_gradients: Sequence[Optional[torch.Tensor]],
    reference_tensor: torch.Tensor,
    chunk_size: int = VSP_GRAD_STAT_CHUNK_SIZE,
) -> torch.Tensor:
    if len(main_gradients) != len(auxiliary_gradients):
        raise ValueError("VSP gradient lists must have the same length.")
    if int(chunk_size) <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")

    stats = torch.zeros(3, device=reference_tensor.device, dtype=torch.float64)
    with torch.no_grad():
        for main_gradient, auxiliary_gradient in zip(main_gradients, auxiliary_gradients):
            main_flat = None
            auxiliary_flat = None
            if main_gradient is not None:
                main_flat = main_gradient.detach().reshape(-1)
            if auxiliary_gradient is not None:
                auxiliary_flat = auxiliary_gradient.detach().reshape(-1)
            if auxiliary_flat is None:
                continue
            if main_flat is not None and main_flat.shape != auxiliary_flat.shape:
                raise ValueError("Mismatched VSP gradient tensor shapes.")

            numel = auxiliary_flat.numel()
            for start in range(0, numel, int(chunk_size)):
                stop = min(start + int(chunk_size), numel)
                main_chunk = main_flat[start:stop].float() if main_flat is not None else None
                auxiliary_chunk = auxiliary_flat[start:stop].float() if auxiliary_flat is not None else None
                if main_chunk is not None:
                    stats[1].add_(torch.dot(main_chunk, main_chunk).double())
                stats[2].add_(torch.dot(auxiliary_chunk, auxiliary_chunk).double())
                if main_chunk is not None:
                    stats[0].add_(torch.dot(main_chunk, auxiliary_chunk).double())
    return stats


def _stats_to_values(stats: torch.Tensor, eps: float) -> Dict[str, torch.Tensor]:
    dot = stats[0]
    main_norm_sq = stats[1].clamp_min(0.0)
    auxiliary_norm_sq = stats[2].clamp_min(0.0)
    main_norm = main_norm_sq.sqrt()
    auxiliary_norm = auxiliary_norm_sq.sqrt()
    cosine = torch.where(
        (main_norm * auxiliary_norm) > float(eps),
        dot / (main_norm * auxiliary_norm + float(eps)),
        torch.zeros_like(dot),
    )
    return {
        "dot": dot,
        "main_norm_sq": main_norm_sq,
        "auxiliary_norm_sq": auxiliary_norm_sq,
        "main_norm": main_norm,
        "auxiliary_norm": auxiliary_norm,
        "cosine": cosine,
    }


def _sequence_norm(
    gradients: Sequence[Optional[torch.Tensor]],
    reference_tensor: torch.Tensor,
    process_group=None,
) -> torch.Tensor:
    stats = torch.zeros((), device=reference_tensor.device, dtype=torch.float64)
    with torch.no_grad():
        for gradient in gradients:
            if gradient is None:
                continue
            flat = gradient.detach().reshape(-1).float()
            stats.add_(torch.dot(flat, flat).double())
        _all_reduce_if_needed(stats, process_group)
    return stats.clamp_min(0.0).sqrt()


def _all_reduce_if_needed(tensor: torch.Tensor, process_group=None) -> None:
    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size(group=process_group) > 1
    ):
        torch.distributed.all_reduce(
            tensor,
            op=torch.distributed.ReduceOp.SUM,
            group=process_group,
        )


def _safe_item(value: torch.Tensor) -> float:
    result = float(value.detach().float().item())
    if not math.isfinite(result):
        raise RuntimeError("VSP gradient statistic is NaN or Inf.")
    return result


def _apply_auxiliary_transform(
    main_gradients: Sequence[Optional[torch.Tensor]],
    auxiliary_gradients: Sequence[Optional[torch.Tensor]],
    reference_tensor: torch.Tensor,
    *,
    apply_pcgrad: bool,
    apply_norm_cap: bool,
    pcgrad_threshold: float,
    max_grad_ratio: float,
    main_norm_reference: Optional[float],
    eps: float,
    process_group=None,
    chunk_size: int = VSP_GRAD_STAT_CHUNK_SIZE,
    clone_auxiliary: bool = True,
) -> Tuple[List[Optional[torch.Tensor]], Dict[str, float]]:
    stats_tensor = _accumulate_sequence_stats(
        main_gradients,
        auxiliary_gradients,
        reference_tensor=reference_tensor,
        chunk_size=chunk_size,
    )
    _all_reduce_if_needed(stats_tensor, process_group)
    values = _stats_to_values(stats_tensor, eps)

    dot = values["dot"]
    main_norm_sq = values["main_norm_sq"]
    aux_norm = values["auxiliary_norm"]
    main_norm = values["main_norm"]
    cosine = values["cosine"]
    finite_stats = torch.isfinite(dot) & torch.isfinite(main_norm_sq) & torch.isfinite(aux_norm)
    has_main = main_norm_sq > float(eps)
    conflict = (cosine < -float(pcgrad_threshold)) & has_main & finite_stats
    coefficient = torch.where(
        apply_pcgrad & conflict,
        dot / (main_norm_sq + float(eps)),
        torch.zeros_like(dot),
    )

    safe_auxiliary: List[Optional[torch.Tensor]] = []
    coeff_float = _safe_item(coefficient)
    with torch.no_grad():
        for main_gradient, auxiliary_gradient in zip(main_gradients, auxiliary_gradients):
            if auxiliary_gradient is None:
                safe_auxiliary.append(None)
                continue
            if clone_auxiliary:
                safe_gradient = auxiliary_gradient.detach().clone()
            else:
                safe_gradient = auxiliary_gradient.detach()
            if main_gradient is not None and coeff_float != 0.0:
                safe_gradient.add_(main_gradient.detach(), alpha=-coeff_float)
            safe_auxiliary.append(safe_gradient)

    safe_aux_norm = _sequence_norm(safe_auxiliary, reference_tensor, process_group=process_group)
    safe_aux_norm_float = _safe_item(safe_aux_norm)
    main_norm_float = _safe_item(main_norm)
    aux_norm_float = _safe_item(aux_norm)
    cap_reference = main_norm_float if main_norm_reference is None else float(main_norm_reference)
    cap_scale = 1.0
    if apply_norm_cap:
        allowed_aux_norm = max(0.0, float(max_grad_ratio)) * max(0.0, cap_reference)
        if safe_aux_norm_float <= float(eps):
            cap_scale = 1.0
        else:
            cap_scale = min(1.0, allowed_aux_norm / (safe_aux_norm_float + float(eps)))
        if not math.isfinite(cap_scale):
            raise RuntimeError("VSP norm-cap scale is NaN or Inf.")
        if cap_scale < 1.0:
            with torch.no_grad():
                for gradient in safe_auxiliary:
                    if gradient is not None:
                        gradient.mul_(cap_scale)

    removed_norm = abs(coeff_float) * main_norm_float
    removed_fraction = 0.0 if aux_norm_float <= float(eps) else min(1.0, removed_norm / (aux_norm_float + float(eps)))
    stats = {
        "cos": _safe_item(cosine),
        "main_norm": main_norm_float,
        "aux_norm": aux_norm_float,
        "safe_aux_norm": safe_aux_norm_float,
        "raw_aux_ratio": 0.0 if main_norm_float <= float(eps) else aux_norm_float / (main_norm_float + float(eps)),
        "safe_aux_ratio": 0.0 if main_norm_float <= float(eps) else safe_aux_norm_float / (main_norm_float + float(eps)),
        "conflict": float(bool(conflict.item())),
        "severe_conflict": float(bool((cosine < -0.5).item())),
        "projection_removed_fraction": removed_fraction if apply_pcgrad else 0.0,
        "cap_scale": cap_scale,
        "cap_active": float(cap_scale < 1.0),
    }
    return safe_auxiliary, stats


def _sum_gradient_sequences(
    first: Sequence[Optional[torch.Tensor]],
    second: Sequence[Optional[torch.Tensor]],
) -> List[Optional[torch.Tensor]]:
    if len(first) != len(second):
        raise ValueError("VSP gradient lists must have the same length.")
    summed: List[Optional[torch.Tensor]] = []
    with torch.no_grad():
        for left, right in zip(first, second):
            if left is None and right is None:
                summed.append(None)
            elif left is None:
                summed.append(right.detach().clone())
            elif right is None:
                summed.append(left.detach().clone())
            else:
                value = left.detach().clone()
                value.add_(right.detach())
                summed.append(value)
    return summed


def _scale_gradient_sequence(gradients: Sequence[Optional[torch.Tensor]], scale: float) -> None:
    if scale == 1.0:
        return
    with torch.no_grad():
        for gradient in gradients:
            if gradient is not None:
                gradient.mul_(scale)


def _summed_sequence_norm(
    first: Sequence[Optional[torch.Tensor]],
    second: Sequence[Optional[torch.Tensor]],
    reference_tensor: torch.Tensor,
    process_group=None,
    chunk_size: int = VSP_GRAD_STAT_CHUNK_SIZE,
) -> torch.Tensor:
    if len(first) != len(second):
        raise ValueError("VSP gradient lists must have the same length.")
    stats = torch.zeros((), device=reference_tensor.device, dtype=torch.float64)
    with torch.no_grad():
        for left, right in zip(first, second):
            if left is None and right is None:
                continue
            left_flat = left.detach().reshape(-1) if left is not None else None
            right_flat = right.detach().reshape(-1) if right is not None else None
            if left_flat is not None and right_flat is not None and left_flat.shape != right_flat.shape:
                raise ValueError("Mismatched VSP gradient tensor shapes.")
            numel = left_flat.numel() if left_flat is not None else right_flat.numel()
            for start in range(0, numel, int(chunk_size)):
                stop = min(start + int(chunk_size), numel)
                if left_flat is None:
                    chunk = right_flat[start:stop].float()
                elif right_flat is None:
                    chunk = left_flat[start:stop].float()
                else:
                    chunk = left_flat[start:stop].float() + right_flat[start:stop].float()
                stats.add_(torch.dot(chunk, chunk).double())
        _all_reduce_if_needed(stats, process_group)
    return stats.clamp_min(0.0).sqrt()


def _combine_final_gradients(
    main_gradients: Sequence[Optional[torch.Tensor]],
    proj_gradients: Sequence[Optional[torch.Tensor]],
    final_gradients: Sequence[Optional[torch.Tensor]],
    clone_outputs: bool = True,
) -> List[Optional[torch.Tensor]]:
    if not (len(main_gradients) == len(proj_gradients) == len(final_gradients)):
        raise ValueError("VSP gradient lists must have the same length.")
    combined: List[Optional[torch.Tensor]] = []
    with torch.no_grad():
        for main_gradient, proj_gradient, final_gradient in zip(main_gradients, proj_gradients, final_gradients):
            output = None
            for gradient in (main_gradient, proj_gradient, final_gradient):
                if gradient is None:
                    continue
                if output is None:
                    output = gradient.detach().clone() if clone_outputs else gradient.detach()
                else:
                    output.add_(gradient.detach())
            combined.append(output)
    return combined


class VSPGradientController:
    def __init__(self, model, config, process_group=None):
        validate_vsp_gradient_config(config)
        self.model = model
        self.config = config
        self.process_group = process_group
        self.ema: Dict[str, float] = {}
        self._warned_empty_groups = False

    @property
    def apply_pcgrad(self) -> bool:
        return bool(
            getattr(self.config, "vsp_asymmetric_pcgrad", False)
            or getattr(self.config, "use_pcgrad", False)
        )

    @property
    def apply_norm_cap(self) -> bool:
        return bool(getattr(self.config, "vsp_norm_cap", False))

    @property
    def eps(self) -> float:
        return finite_float(getattr(self.config, "vsp_grad_eps", VSP_GRAD_EPS), VSP_GRAD_EPS)

    @property
    def threshold(self) -> float:
        return finite_float(getattr(self.config, "vsp_pcgrad_threshold", 0.05), 0.05)

    @property
    def beta(self) -> float:
        return finite_float(getattr(self.config, "vsp_grad_ema_beta", 0.95), 0.95)

    def max_ratio(self, group_name: str) -> float:
        if group_name == "projector":
            return finite_float(getattr(self.config, "vsp_proj_max_grad_ratio", 0.5), 0.5)
        return finite_float(getattr(self.config, "vsp_llm_max_grad_ratio", 0.1), 0.1)

    def _ema_update(self, key: str, value: float) -> float:
        if not math.isfinite(value):
            raise RuntimeError(f"VSP EMA input {key} is NaN or Inf.")
        if key not in self.ema:
            self.ema[key] = value
        else:
            self.ema[key] = self.beta * self.ema[key] + (1.0 - self.beta) * value
        return self.ema[key]

    def _main_norm_reference(self, group_name: str) -> Optional[float]:
        return self.ema.get(f"grad/{group_name}/main_grad_norm_ema")

    def state_dict(self) -> Dict[str, Dict[str, float]]:
        return {"ema": dict(self.ema)}

    def load_state_dict(self, state_dict: Mapping[str, Mapping[str, float]]) -> None:
        ema_state = state_dict.get("ema", {}) if isinstance(state_dict, Mapping) else {}
        self.ema = {
            str(key): float(value)
            for key, value in ema_state.items()
            if math.isfinite(float(value))
        }

    def _record_group_logs(
        self,
        logs: MutableMapping[str, float],
        group_name: str,
        group_logs: Mapping[str, float],
    ) -> None:
        for suffix, value in group_logs.items():
            logs[f"grad/{group_name}/{suffix}"] = value
        for suffix in (
            "main_grad_norm",
            "proj_safe_aux_norm",
            "final_safe_aux_norm",
            "proj_vs_lm_cos",
            "final_vs_lm_cos",
            "conflict_rate",
            "cap_scale",
        ):
            if suffix in group_logs:
                ema = self._ema_update(f"grad/{group_name}/{suffix}_ema", group_logs[suffix])
                logs[f"grad/{group_name}/{suffix}_ema"] = ema

    def _process_group_gradients(
        self,
        group_name: str,
        main_gradients: Sequence[Optional[torch.Tensor]],
        proj_gradients: Sequence[Optional[torch.Tensor]],
        final_gradients: Sequence[Optional[torch.Tensor]],
        reference_tensor: torch.Tensor,
        clone_gradients: bool = True,
    ) -> Tuple[List[Optional[torch.Tensor]], Dict[str, float]]:
        main_norm_reference = self._main_norm_reference(group_name)
        max_ratio = self.max_ratio(group_name)

        raw_aux_norm = _summed_sequence_norm(
            proj_gradients,
            final_gradients,
            reference_tensor,
            process_group=self.process_group,
        )
        proj_final_stats = _accumulate_sequence_stats(
            proj_gradients,
            final_gradients,
            reference_tensor=reference_tensor,
        )
        _all_reduce_if_needed(proj_final_stats, self.process_group)
        proj_final_values = _stats_to_values(proj_final_stats, self.eps)

        safe_proj, proj_stats = _apply_auxiliary_transform(
            main_gradients,
            proj_gradients,
            reference_tensor,
            apply_pcgrad=self.apply_pcgrad,
            apply_norm_cap=self.apply_norm_cap,
            pcgrad_threshold=self.threshold,
            max_grad_ratio=max_ratio,
            main_norm_reference=main_norm_reference,
            eps=self.eps,
            process_group=self.process_group,
            clone_auxiliary=clone_gradients,
        )
        safe_final, final_stats = _apply_auxiliary_transform(
            main_gradients,
            final_gradients,
            reference_tensor,
            apply_pcgrad=self.apply_pcgrad,
            apply_norm_cap=self.apply_norm_cap,
            pcgrad_threshold=self.threshold,
            max_grad_ratio=max_ratio,
            main_norm_reference=main_norm_reference,
            eps=self.eps,
            process_group=self.process_group,
            clone_auxiliary=clone_gradients,
        )

        safe_aux_norm = _summed_sequence_norm(safe_proj, safe_final, reference_tensor, process_group=self.process_group)
        main_norm = max(proj_stats["main_norm"], final_stats["main_norm"])
        cap_reference = main_norm if main_norm_reference is None else float(main_norm_reference)
        total_cap_scale = 1.0
        if self.apply_norm_cap:
            allowed_aux_norm = max_ratio * max(0.0, cap_reference)
            safe_aux_norm_for_cap = _safe_item(safe_aux_norm)
            if safe_aux_norm_for_cap <= self.eps:
                total_cap_scale = 1.0
            else:
                total_cap_scale = min(1.0, allowed_aux_norm / (safe_aux_norm_for_cap + self.eps))
            if not math.isfinite(total_cap_scale):
                raise RuntimeError("VSP total auxiliary cap scale is NaN or Inf.")
            if total_cap_scale < 1.0:
                _scale_gradient_sequence(safe_proj, total_cap_scale)
                _scale_gradient_sequence(safe_final, total_cap_scale)
                safe_aux_norm = _summed_sequence_norm(safe_proj, safe_final, reference_tensor, process_group=self.process_group)

        raw_aux_norm_float = _safe_item(raw_aux_norm)
        safe_aux_norm_float = _safe_item(safe_aux_norm)
        main_norm_float = max(proj_stats["main_norm"], final_stats["main_norm"])
        raw_ratio = 0.0 if main_norm_float <= self.eps else raw_aux_norm_float / (main_norm_float + self.eps)
        safe_ratio = 0.0 if main_norm_float <= self.eps else safe_aux_norm_float / (main_norm_float + self.eps)
        cap_scale = min(proj_stats["cap_scale"], final_stats["cap_scale"], total_cap_scale)
        conflict_rate = 0.5 * (proj_stats["conflict"] + final_stats["conflict"])

        logs = {
            "proj_vs_lm_cos": proj_stats["cos"],
            "final_vs_lm_cos": final_stats["cos"],
            "proj_vs_final_cos": _safe_item(proj_final_values["cosine"]),
            "main_grad_norm": main_norm_float,
            "proj_aux_grad_norm": proj_stats["aux_norm"],
            "final_aux_grad_norm": final_stats["aux_norm"],
            "proj_safe_aux_norm": proj_stats["safe_aux_norm"] * proj_stats["cap_scale"] * total_cap_scale,
            "final_safe_aux_norm": final_stats["safe_aux_norm"] * final_stats["cap_scale"] * total_cap_scale,
            "proj_raw_aux_ratio": proj_stats["raw_aux_ratio"],
            "final_raw_aux_ratio": final_stats["raw_aux_ratio"],
            "raw_aux_ratio": raw_ratio,
            "safe_aux_ratio": safe_ratio,
            "proj_conflict": proj_stats["conflict"],
            "final_conflict": final_stats["conflict"],
            "proj_severe_conflict": proj_stats["severe_conflict"],
            "final_severe_conflict": final_stats["severe_conflict"],
            "conflict_rate": conflict_rate,
            "proj_projection_removed_fraction": proj_stats["projection_removed_fraction"],
            "final_projection_removed_fraction": final_stats["projection_removed_fraction"],
            "projection_removed_fraction": max(
                proj_stats["projection_removed_fraction"],
                final_stats["projection_removed_fraction"],
            ),
            "proj_cap_scale": proj_stats["cap_scale"],
            "final_cap_scale": final_stats["cap_scale"],
            "total_aux_cap_scale": total_cap_scale,
            "cap_scale": cap_scale,
            "cap_active": float(cap_scale < 1.0),
            "effective_aux_coeff": cap_scale,
        }
        final_sequence = _combine_final_gradients(main_gradients, safe_proj, safe_final, clone_outputs=clone_gradients)
        return final_sequence, logs

    def compute_diagnostics(
        self,
        lm_loss: torch.Tensor,
        vsp_proj_loss: Optional[torch.Tensor] = None,
        vsp_final_loss: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        return self._compute(lm_loss, vsp_proj_loss, vsp_final_loss, assign_gradients=False)

    def compute_and_assign_gradients(
        self,
        lm_loss: torch.Tensor,
        vsp_proj_loss: Optional[torch.Tensor] = None,
        vsp_final_loss: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        return self._compute(lm_loss, vsp_proj_loss, vsp_final_loss, assign_gradients=True)

    def _compute(
        self,
        lm_loss: torch.Tensor,
        vsp_proj_loss: Optional[torch.Tensor],
        vsp_final_loss: Optional[torch.Tensor],
        *,
        assign_gradients: bool,
    ) -> Dict[str, float]:
        validate_vsp_losses(lm_loss, vsp_proj_loss, vsp_final_loss)
        groups = collect_vsp_parameter_groups(self.model)
        logs: Dict[str, float] = {}

        for group_name in VSP_GROUP_NAMES:
            group = groups[group_name]
            params = group.params
            if not params:
                continue
            main_gradients = _extract_gradients(lm_loss, params, retain_graph=True)
            proj_gradients = _extract_gradients(vsp_proj_loss, params, retain_graph=True)
            final_gradients = _extract_gradients(vsp_final_loss, params, retain_graph=True)
            final_sequence, group_logs = self._process_group_gradients(
                group_name,
                main_gradients,
                proj_gradients,
                final_gradients,
                reference_tensor=lm_loss,
            )
            self._record_group_logs(logs, group_name, group_logs)
            if assign_gradients:
                with torch.no_grad():
                    for param, gradient in zip(params, final_sequence):
                        if gradient is None:
                            param.grad = None
                        else:
                            param.grad = gradient.to(device=param.device, dtype=param.dtype)

        return logs


def partitioned_group_stats(
    main_parts: Mapping[int, Sequence[Optional[torch.Tensor]]],
    auxiliary_parts: Mapping[int, Sequence[Optional[torch.Tensor]]],
    group_ids: Iterable[int],
    reference_tensor: torch.Tensor,
    *,
    process_group=None,
    chunk_size: int = VSP_GRAD_STAT_CHUNK_SIZE,
) -> torch.Tensor:
    stats = torch.zeros(3, device=reference_tensor.device, dtype=torch.float64)
    for group_id in group_ids:
        main_group = list(main_parts.get(group_id, []))
        auxiliary_group = list(auxiliary_parts.get(group_id, []))
        shard_count = max(len(main_group), len(auxiliary_group))
        if len(main_group) < shard_count:
            main_group.extend([None] * (shard_count - len(main_group)))
        if len(auxiliary_group) < shard_count:
            auxiliary_group.extend([None] * (shard_count - len(auxiliary_group)))
        stats.add_(
            _accumulate_sequence_stats(
                main_group,
                auxiliary_group,
                reference_tensor=reference_tensor,
                chunk_size=chunk_size,
            )
        )
    _all_reduce_if_needed(stats, process_group)
    return stats


def _flatten_partitioned_parts(
    main_parts: Mapping[int, Sequence[Optional[torch.Tensor]]],
    proj_parts: Mapping[int, Sequence[Optional[torch.Tensor]]],
    final_parts: Mapping[int, Sequence[Optional[torch.Tensor]]],
    group_ids: Sequence[int],
) -> Tuple[
    List[Optional[torch.Tensor]],
    List[Optional[torch.Tensor]],
    List[Optional[torch.Tensor]],
    List[Tuple[int, int]],
]:
    flat_main: List[Optional[torch.Tensor]] = []
    flat_proj: List[Optional[torch.Tensor]] = []
    flat_final: List[Optional[torch.Tensor]] = []
    layout: List[Tuple[int, int]] = []

    for group_id in group_ids:
        main_group = list(main_parts.get(group_id, []))
        proj_group = list(proj_parts.get(group_id, []))
        final_group = list(final_parts.get(group_id, []))
        present_lengths = [
            len(group)
            for group in (main_group, proj_group, final_group)
            if len(group) > 0
        ]
        if not present_lengths:
            continue
        if len(set(present_lengths)) > 1:
            raise ValueError(f"ZeRO-2 VSP gradient group {group_id} has mismatched shard counts.")
        shard_count = present_lengths[0]
        if len(main_group) == 0:
            main_group = [None] * shard_count
        if len(proj_group) == 0:
            proj_group = [None] * shard_count
        if len(final_group) == 0:
            final_group = [None] * shard_count

        for shard_index, (main_gradient, proj_gradient, final_gradient) in enumerate(
            zip(main_group, proj_group, final_group)
        ):
            tensors = [
                gradient
                for gradient in (main_gradient, proj_gradient, final_gradient)
                if gradient is not None
            ]
            if tensors:
                first_shape = tensors[0].shape
                if any(tensor.shape != first_shape for tensor in tensors[1:]):
                    raise ValueError(
                        f"ZeRO-2 VSP gradient group {group_id} shard {shard_index} "
                        "has mismatched tensor shapes."
                    )
            flat_main.append(main_gradient)
            flat_proj.append(proj_gradient)
            flat_final.append(final_gradient)
            layout.append((group_id, shard_index))

    return flat_main, flat_proj, flat_final, layout


def combine_partitioned_vsp_gradients(
    controller: VSPGradientController,
    main_parts: Mapping[int, Sequence[Optional[torch.Tensor]]],
    proj_parts: Mapping[int, Sequence[Optional[torch.Tensor]]],
    final_parts: Mapping[int, Sequence[Optional[torch.Tensor]]],
    group_id_to_name: Mapping[int, str],
    reference_tensor: torch.Tensor,
) -> Tuple[Dict[int, List[torch.Tensor]], Dict[str, float]]:
    all_group_ids = sorted(set(main_parts) | set(proj_parts) | set(final_parts))
    logs: Dict[str, float] = {}
    combined_parts: Dict[int, List[torch.Tensor]] = {}

    for group_name in VSP_GROUP_NAMES:
        group_ids = [
            group_id
            for group_id in all_group_ids
            if group_id_to_name.get(group_id, "llm") == group_name
        ]
        if not group_ids:
            continue
        flat_main, flat_proj, flat_final, layout = _flatten_partitioned_parts(
            main_parts,
            proj_parts,
            final_parts,
            group_ids,
        )
        if not layout:
            continue
        final_sequence, group_logs = controller._process_group_gradients(
            group_name,
            flat_main,
            flat_proj,
            flat_final,
            reference_tensor=reference_tensor,
            clone_gradients=False,
        )
        controller._record_group_logs(logs, group_name, group_logs)
        for (group_id, _), gradient in zip(layout, final_sequence):
            if gradient is None:
                continue
            combined_parts.setdefault(group_id, []).append(gradient)

    return combined_parts, logs

