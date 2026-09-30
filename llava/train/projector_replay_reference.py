"""Reference projector-only replay for adaptive CKA gradient control.

The sidecar is deliberately not attached to the trained model, optimizer, or any
DDP/ZeRO reducer.  It replays only the deterministic multimodal projector on
frozen vision features captured from the main forward.
"""
from __future__ import annotations

import copy
import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn.utils import parametrize


# FP32 scalar reductions can place an otherwise exact boundary score a few
# ulps outside [0, 1]. This floor only governs the score-domain check;
# material errors are still rejected rather than hidden by clamping.
_FP32_CKA_SCORE_RTOL_FLOOR = 8.0 * torch.finfo(torch.float32).eps


@contextmanager
def _strict_fp32_matmul(device: torch.device):
    """Temporarily prevent CUDA FP32 Gram products from using TF32."""
    if device.type != "cuda":
        yield
        return
    previous = bool(torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@dataclass
class CkaSumResult:
    loss_sum: torch.Tensor
    valid_count: torch.Tensor
    invalid_counts: Dict[str, torch.Tensor]


def _scalar_count(reference: torch.Tensor, value: int = 0) -> torch.Tensor:
    return torch.tensor(value, device=reference.device, dtype=torch.float64)


def projector_cka_sum(
    vision_features: torch.Tensor,
    projected_features: torch.Tensor,
    *,
    norm_floor: float = 1e-12,
    score_rtol: float = 1e-7,
) -> CkaSumResult:
    """Return the unweighted per-observation CKA loss sum and valid count.

    Observations are independent rows on the batch axis.  Patch tokens are
    centered within each observation; batches are never flattened together.
    Gram/CKA math is always FP32 with autocast disabled.
    """
    if vision_features.ndim != 3 or projected_features.ndim != 3:
        raise ValueError("Projector CKA expects [observations, patches, channels] tensors.")
    if vision_features.shape[:2] != projected_features.shape[:2]:
        raise ValueError(
            "Projector replay changed observation/patch correspondence: "
            f"{tuple(vision_features.shape[:2])} != {tuple(projected_features.shape[:2])}."
        )
    if vision_features.device != projected_features.device:
        raise ValueError("Projector CKA inputs must be on the same device.")
    if norm_floor <= 0.0 or not math.isfinite(float(norm_floor)):
        raise ValueError("norm_floor must be finite and positive.")
    if not math.isfinite(float(score_rtol)) or score_rtol < 0.0:
        raise ValueError("score_rtol must be finite and non-negative.")

    reference = projected_features
    invalid = {
        "too_few_patches": _scalar_count(reference),
        "nonfinite_input": _scalar_count(reference),
        "nonfinite_gram": _scalar_count(reference),
        "zero_gram_norm": _scalar_count(reference),
        "score_out_of_range": _scalar_count(reference),
        "score_roundoff_clamped": _scalar_count(reference),
    }
    observation_count, patch_count = vision_features.shape[:2]
    if observation_count == 0:
        return CkaSumResult(projected_features.float().sum() * 0.0, _scalar_count(reference), invalid)
    if patch_count < 2:
        invalid["too_few_patches"] = _scalar_count(reference, observation_count)
        return CkaSumResult(projected_features.float().sum() * 0.0, _scalar_count(reference), invalid)

    with torch.autocast(device_type=reference.device.type, enabled=False), _strict_fp32_matmul(
        reference.device
    ):
        x = vision_features.detach().float()
        y = projected_features.float()
        finite = torch.isfinite(x).flatten(1).all(dim=1) & torch.isfinite(y).flatten(1).all(dim=1)
        invalid["nonfinite_input"] = (~finite).to(torch.float64).sum()
        x = x[finite]
        y = y[finite]
        x = x - x.mean(dim=1, keepdim=True)
        y = y - y.mean(dim=1, keepdim=True)
        gx = torch.bmm(x, x.transpose(1, 2))
        gy = torch.bmm(y, y.transpose(1, 2))
        gx_norm_sq = (gx * gx).sum(dim=(1, 2))
        gy_norm_sq = (gy * gy).sum(dim=(1, 2))
        gx_norm = torch.sqrt(gx_norm_sq)
        gy_norm = torch.sqrt(gy_norm_sq)
        finite_gram = torch.isfinite(gx_norm_sq) & torch.isfinite(gy_norm_sq)
        invalid["nonfinite_gram"] = (~finite_gram).to(torch.float64).sum()
        nonzero = finite_gram & (gx_norm > float(norm_floor)) & (gy_norm > float(norm_floor))
        invalid["zero_gram_norm"] = (
            finite_gram & ~((gx_norm > float(norm_floor)) & (gy_norm > float(norm_floor)))
        ).to(torch.float64).sum()
        gx = gx[nonzero]
        gy = gy[nonzero]
        # Algebraically identical to sum((Gx/||Gx||) * (Gy/||Gy||)), but it
        # avoids two normalized N x N intermediates and has a much tighter
        # self-CKA rounding error.
        scores = (gx * gy).sum(dim=(1, 2)) / (
            gx_norm[nonzero] * gy_norm[nonzero]
        )
        effective_score_rtol = max(float(score_rtol), _FP32_CKA_SCORE_RTOL_FLOOR)
        grossly_invalid = (~torch.isfinite(scores)) | (scores < -effective_score_rtol) | (
            scores > 1.0 + effective_score_rtol
        )
        invalid["score_out_of_range"] = grossly_invalid.to(torch.float64).sum()
        scores = scores[~grossly_invalid]
        roundoff = (scores < 0.0) | (scores > 1.0)
        invalid["score_roundoff_clamped"] = roundoff.to(torch.float64).sum()
        # Only correct representational roundoff after recording the explicit
        # tolerance check. Materially invalid scores are excluded and make the
        # effective update fail at its distributed boundary.
        scores = scores.clamp(0.0, 1.0)
        losses = 1.0 - scores
        return CkaSumResult(
            loss_sum=losses.sum(),
            valid_count=torch.ones_like(losses, dtype=torch.float64).sum(),
            invalid_counts=invalid,
        )


def ordered_trainable_parameters(module: nn.Module) -> Tuple[List[str], List[nn.Parameter]]:
    names: List[str] = []
    params: List[nn.Parameter] = []
    seen = set()
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        names.append(name)
        params.append(parameter)
    return names, params


def validate_replayable_projector(projector: nn.Module) -> None:
    """Fail fast for stateful/stochastic/tied/hooked projectors."""
    from llava.model.multimodal_projector.builder import (
        AdditiveCouplingBlock,
        CouplingProjector,
    )

    names, params = ordered_trainable_parameters(projector)
    if not params:
        raise RuntimeError("Adaptive projector PCGrad requires trainable projector parameters.")

    all_parameters = list(projector.named_parameters(remove_duplicate=False))
    parameter_ids = [id(parameter) for _, parameter in all_parameters]
    if len(parameter_ids) != len(set(parameter_ids)):
        raise RuntimeError("Adaptive projector replay does not support tied projector parameters.")

    for module_name, module in projector.named_modules():
        label = module_name or "<root>"
        allowed_types = (
            nn.Linear,
            nn.GELU,
            nn.Sequential,
            nn.ModuleList,
            AdditiveCouplingBlock,
            CouplingProjector,
        )
        if isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.Dropout, nn.AlphaDropout)):
            raise RuntimeError(
                f"Adaptive projector replay rejects stochastic/stateful module {label}: "
                f"{type(module).__name__}."
            )
        if not isinstance(module, allowed_types):
            raise RuntimeError(
                "Adaptive projector replay only supports the repository's deterministic "
                f"Linear/GELU/coupling projectors; found {type(module).__name__} at {label}."
            )
        if any(True for _ in module.buffers(recurse=False)):
            raise RuntimeError(f"Adaptive projector replay rejects mutable buffers in {label}.")
        if parametrize.is_parametrized(module):
            raise RuntimeError(f"Adaptive projector replay rejects parametrized module {label}.")
        hook_maps = (
            module._forward_hooks,
            module._forward_pre_hooks,
            module._backward_hooks,
            getattr(module, "_backward_pre_hooks", {}),
        )
        if any(bool(hooks) for hooks in hook_maps):
            raise RuntimeError(f"Adaptive projector replay rejects hooks on projector module {label}.")
    for parameter_name, parameter in projector.named_parameters(remove_duplicate=False):
        parameter_hook_maps = (
            getattr(parameter, "_backward_hooks", {}),
            getattr(parameter, "_post_accumulate_grad_hooks", {}),
        )
        if any(bool(hooks) for hooks in parameter_hook_maps):
            raise RuntimeError(
                "Adaptive projector replay rejects hooks on projector parameter "
                f"{parameter_name}."
            )


def clone_plain_projector(projector: nn.Module) -> nn.Module:
    validate_replayable_projector(projector)
    sidecar = copy.deepcopy(projector)
    sidecar.train(projector.training)
    source = dict(projector.named_parameters())
    for name, parameter in sidecar.named_parameters():
        parameter.requires_grad_(source[name].requires_grad)
    return sidecar


def sync_projector_sidecar(projector: nn.Module, sidecar: nn.Module) -> None:
    source = dict(projector.named_parameters())
    target = dict(sidecar.named_parameters())
    if source.keys() != target.keys():
        raise RuntimeError("Projector sidecar parameter names no longer match the live projector.")
    with torch.no_grad():
        for name in source:
            if source[name].shape != target[name].shape:
                raise RuntimeError(f"Projector sidecar shape mismatch for {name}.")
            target[name].copy_(source[name].detach())
    sidecar.train(projector.training)


class ProjectorReplayAccumulator:
    """Accumulate raw CKA gradient SUMs without retaining the VLM graph."""

    def __init__(
        self,
        projector: nn.Module,
        *,
        chunk_size: int = 1,
        norm_floor: float = 1e-12,
        score_rtol: float = 1e-7,
    ):
        if int(chunk_size) <= 0:
            raise ValueError("Projector replay chunk_size must be positive.")
        self.projector = projector
        self.sidecar = clone_plain_projector(projector)
        self.chunk_size = int(chunk_size)
        self.norm_floor = float(norm_floor)
        self.score_rtol = float(score_rtol)
        self.names, self.live_params = ordered_trainable_parameters(projector)
        sidecar_named = dict(self.sidecar.named_parameters())
        self.sidecar_params = [sidecar_named[name] for name in self.names]
        self._gradient_numels = [parameter.numel() for parameter in self.sidecar_params]
        self.gradient_sums_flat = torch.zeros(
            sum(self._gradient_numels),
            device=self.sidecar_params[0].device,
            dtype=torch.float32,
        )
        self._refresh_gradient_views()
        self.cka_sum = torch.zeros((), device=self.sidecar_params[0].device, dtype=torch.float64)
        self.valid_count = torch.zeros_like(self.cka_sum)
        self.invalid_counts = {
            name: torch.zeros_like(self.cka_sum)
            for name in (
                "too_few_patches",
                "nonfinite_input",
                "nonfinite_gram",
                "zero_gram_norm",
                "score_out_of_range",
                "score_roundoff_clamped",
            )
        }
        self.snapshot_token = None
        self._verified = False
        self._has_pending_data = False

    def _refresh_gradient_views(self) -> None:
        if self.gradient_sums_flat is None:
            self.gradient_sums = []
            return
        self.gradient_sums = []
        offset = 0
        for parameter, numel in zip(self.sidecar_params, self._gradient_numels):
            self.gradient_sums.append(
                self.gradient_sums_flat.narrow(0, offset, numel).view_as(parameter)
            )
            offset += numel

    def _clear_accumulators(self) -> None:
        if self.gradient_sums_flat is None:
            self.gradient_sums_flat = torch.zeros(
                sum(self._gradient_numels),
                device=self.sidecar_params[0].device,
                dtype=torch.float32,
            )
            self._refresh_gradient_views()
        self.gradient_sums_flat.zero_()
        self.cka_sum.zero_()
        self.valid_count.zero_()
        for count in self.invalid_counts.values():
            count.zero_()
        self._has_pending_data = False

    def _ensure_sidecar_placement(self) -> None:
        target_device = self.live_params[0].device
        target_dtype = self.live_params[0].dtype
        if self.gradient_sums_flat is not None and all(
            parameter.device == target_device and parameter.dtype == target_dtype
            for parameter in self.sidecar_params
        ):
            return
        if self._has_pending_data:
            raise RuntimeError("Cannot move projector sidecar during an accumulation window.")
        self.sidecar.to(device=target_device, dtype=target_dtype)
        sidecar_named = dict(self.sidecar.named_parameters())
        self.sidecar_params = [sidecar_named[name] for name in self.names]
        self.gradient_sums_flat = torch.zeros(
            sum(self._gradient_numels), device=target_device, dtype=torch.float32
        )
        self._refresh_gradient_views()
        self.cka_sum = self.cka_sum.to(device=target_device)
        self.valid_count = self.valid_count.to(device=target_device)
        self.invalid_counts = {
            name: value.to(device=target_device)
            for name, value in self.invalid_counts.items()
        }

    def begin_window(self, stage: int, optimizer_step: int) -> None:
        token = (int(stage), int(optimizer_step))
        if self.snapshot_token == token:
            return
        if self.snapshot_token is not None and self._has_pending_data:
            raise RuntimeError("Projector replay snapshot changed before the accumulation window was consumed.")
        self._ensure_sidecar_placement()
        self._clear_accumulators()
        sync_projector_sidecar(self.projector, self.sidecar)
        self.snapshot_token = token

    def verify_equivalence(self, sample: torch.Tensor, *, atol: float, rtol: float) -> None:
        if self._verified or sample.numel() == 0:
            return
        with torch.no_grad():
            live = self.projector(sample)
            replay = self.sidecar(sample)
        torch.testing.assert_close(replay, live, atol=atol, rtol=rtol)
        self._verified = True

    def add(
        self,
        feature_batches: Sequence[torch.Tensor],
        *,
        loss_scale: float = 1.0,
    ) -> None:
        if not math.isfinite(float(loss_scale)) or float(loss_scale) <= 0.0:
            raise ValueError("Projector replay loss_scale must be finite and positive.")
        for features in feature_batches:
            if not torch.is_tensor(features):
                raise TypeError("Captured projector replay features must be tensors.")
            if features.ndim != 3:
                raise ValueError(f"Captured projector replay features must be rank 3, got {features.ndim}.")
            features = features.detach().to(
                device=self.sidecar_params[0].device,
                dtype=self.sidecar_params[0].dtype,
            )
            if features.shape[0] > 0:
                self._has_pending_data = True
            for start in range(0, features.shape[0], self.chunk_size):
                chunk = features[start:start + self.chunk_size]
                projected = self.sidecar(chunk)
                # Keep both the CKA forward and its backward in strict FP32.
                # The sidecar forward remains outside this scope so an FP32
                # projector still follows the main-forward TF32 policy.
                with _strict_fp32_matmul(chunk.device):
                    result = projector_cka_sum(
                        chunk,
                        projected,
                        norm_floor=self.norm_floor,
                        score_rtol=self.score_rtol,
                    )
                    self.valid_count.add_(result.valid_count.detach())
                    for name, count in result.invalid_counts.items():
                        self.invalid_counts[name].add_(count.detach())
                    gradients = torch.autograd.grad(
                        result.loss_sum * float(loss_scale),
                        self.sidecar_params,
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=True,
                    )
                with torch.no_grad():
                    self.cka_sum.add_(result.loss_sum.detach().double())
                    for destination, gradient in zip(self.gradient_sums, gradients):
                        if gradient is not None:
                            destination.add_(gradient.detach().float(), alpha=1.0 / float(loss_scale))
                del gradients, projected, result

    def stats_tensor(self, ce_sum: torch.Tensor, ce_count: torch.Tensor) -> torch.Tensor:
        values = [self.cka_sum, self.valid_count]
        values.extend(self.invalid_counts[name] for name in sorted(self.invalid_counts))
        values.extend((ce_sum.double(), ce_count.double()))
        return torch.stack(values)

    @property
    def stat_names(self) -> Tuple[str, ...]:
        return (
            "cka_sum",
            "valid_count",
            *(f"invalid/{name}" for name in sorted(self.invalid_counts)),
            "ce_sum",
            "ce_count",
        )

    def consume_flat(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        owned_gradient_buffer = self.gradient_sums_flat
        if owned_gradient_buffer is None:
            raise RuntimeError("Projector replay gradient buffer was already consumed.")
        cka_sum = self.cka_sum.detach().clone()
        count = self.valid_count.detach().clone()
        invalid = {name: value.detach().clone() for name, value in self.invalid_counts.items()}
        self.gradient_sums_flat = None
        self._refresh_gradient_views()
        self.cka_sum.zero_()
        self.valid_count.zero_()
        for value in self.invalid_counts.values():
            value.zero_()
        self._has_pending_data = False
        self.snapshot_token = None
        return owned_gradient_buffer, cka_sum, count, invalid

    def recycle_flat_buffer(self, buffer: torch.Tensor) -> None:
        """Return a consumed buffer after the boundary merge, avoiding a clone."""
        if self.gradient_sums_flat is not None:
            raise RuntimeError("Projector replay already owns an accumulation buffer.")
        if (
            buffer.ndim != 1
            or not buffer.is_contiguous()
            or buffer.numel() != sum(self._gradient_numels)
            or buffer.dtype != torch.float32
            or buffer.device != self.sidecar_params[0].device
        ):
            raise ValueError("Recycled projector replay buffer has an incompatible layout.")
        buffer.zero_()
        self.gradient_sums_flat = buffer
        self._refresh_gradient_views()

    def consume(self) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        owned_gradient_buffer, cka_sum, count, invalid = self.consume_flat()
        gradients = []
        offset = 0
        for parameter, numel in zip(self.sidecar_params, self._gradient_numels):
            gradients.append(owned_gradient_buffer.narrow(0, offset, numel).view_as(parameter))
            offset += numel
        # The list-returning reference API transfers gradient ownership to the
        # caller, so allocate a fresh buffer. The optimized Trainer instead
        # uses consume_flat()+recycle_flat_buffer() and performs no full clone.
        self.gradient_sums_flat = torch.zeros_like(owned_gradient_buffer)
        self._refresh_gradient_views()
        return gradients, cka_sum, count, invalid
