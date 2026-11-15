"""Loss kernel for aligned subsets supplied by a calibration token selector.

This module does not select tokens or choose target layers. Callers must supply
the same ordered token indices for the vision anchor and target state.
"""

from collections import Counter
from dataclasses import dataclass
import math
from typing import Dict, Optional, Sequence, Tuple

import torch


@dataclass
class MaskedCKAResult:
    loss: torch.Tensor
    valid_count: int
    invalid_count: int
    invalid_reasons: Dict[str, int]
    # One size per sample; zero denotes an excluded sample.
    subset_sizes: Tuple[int, ...]


def compute_selected_cka_loss(
    vision_features: torch.Tensor,
    target_features: torch.Tensor,
    selected_indices: Sequence[Optional[torch.Tensor]],
    eps: float = 1e-12,
) -> MaskedCKAResult:
    """Compute mean valid per-sample CKA on an already selected token union.

    Inputs have shape (batch, aligned_tokens, channels); channel counts may
    differ. Indices refer to the aligned token axis in both tensors. ``None``
    excludes a sample with no valid image/query. Attention and coverage indices
    must be passed as one ordered union, without duplicates.

    Only the gathered target contributes gradients. The raw vision features
    are detached, and centering happens *after* gathering each sample's subset.
    The caller is responsible for CE and for reporting these diagnostics.
    """
    if vision_features.ndim != 3 or target_features.ndim != 3:
        raise ValueError("Selected CKA expects two rank-3 feature tensors.")
    if vision_features.shape[:2] != target_features.shape[:2]:
        raise ValueError("Selected CKA requires aligned batch and token axes.")
    if vision_features.device != target_features.device:
        raise ValueError("Selected CKA features must be on the same device.")
    if len(selected_indices) != target_features.shape[0]:
        raise ValueError("Selected CKA requires one index tensor per sample.")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("Selected CKA eps must be finite and positive.")

    losses = []
    reasons = Counter()
    sizes = []
    token_count = target_features.shape[1]

    with torch.autocast(device_type=target_features.device.type, enabled=False):
        for sample, indices in enumerate(selected_indices):
            sizes.append(0)
            if indices is None:
                reasons["missing_selection"] += 1
                continue
            if (
                not torch.is_tensor(indices)
                or indices.ndim != 1
                or indices.dtype not in (torch.int32, torch.int64)
            ):
                raise ValueError("Selected indices must be 1-D integer tensors.")
            indices = indices.detach().to(device=target_features.device, dtype=torch.long)
            if bool(((indices < 0) | (indices >= token_count)).any()):
                raise ValueError("Selected CKA index is outside the aligned token axis.")
            if indices.unique().numel() != indices.numel():
                raise ValueError("Selected CKA indices must not contain duplicates.")
            if indices.numel() < 2:
                reasons["too_few_tokens"] += 1
                continue

            x = vision_features[sample].detach().index_select(0, indices).float()
            y = target_features[sample].index_select(0, indices).float()
            if not bool(torch.isfinite(x).all() & torch.isfinite(y).all()):
                reasons["nonfinite_features"] += 1
                continue
            xc = x - x.mean(dim=0, keepdim=True)
            yc = y - y.mean(dim=0, keepdim=True)
            gx = xc @ xc.T
            gy = yc @ yc.T
            denominator = torch.linalg.matrix_norm(gx) * torch.linalg.matrix_norm(gy)
            if not bool(torch.isfinite(denominator)):
                reasons["nonfinite_denominator"] += 1
                continue
            if bool(denominator <= eps):
                reasons["degenerate_denominator"] += 1
                continue
            sample_loss = 1.0 - (gx * gy).sum() / denominator
            if not bool(torch.isfinite(sample_loss)):
                reasons["nonfinite_loss"] += 1
                continue
            losses.append(sample_loss)
            sizes[-1] = indices.numel()

        # Empty-slice sum gives a graph-connected zero even if excluded samples
        # contain NaNs. Multiplying an arbitrary target value by zero would not.
        loss = (
            torch.stack(losses).mean()
            if losses
            else target_features[:, :0, :].float().sum()
        )

    return MaskedCKAResult(
        loss=loss,
        valid_count=len(losses),
        invalid_count=sum(reasons.values()),
        invalid_reasons=dict(reasons),
        subset_sizes=tuple(sizes),
    )
