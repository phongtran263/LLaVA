"""Manual or fixed random attention-head selection and image-token CKA.

Callers capture attention output immediately before ``o_proj``, reshape it
using the number of attention/query heads (not KV heads), and select heads
outside gradient-checkpointed blocks before invoking the loss below.
"""

import json
import math
from numbers import Integral

import torch


def parse_head_ids(value):
    """Parse layer (1-based) -> query/output head IDs (0-based).

    None/blank disables manual selection. A supplied mapping is authoritative:
    only its layers receive head CKA. Return canonical, sorted integer keys and
    head lists; model-dependent upper bounds are checked by the caller.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, str):
        def unique_keys(pairs):
            result = {}
            for key, heads in pairs:
                if key in result:
                    raise ValueError(f"cka_loss_head_ids contains duplicate layer key {key!r}.")
                result[key] = heads
            return result

        try:
            value = json.loads(value, object_pairs_hook=unique_keys)
        except json.JSONDecodeError as exc:
            raise ValueError(
                'cka_loss_head_ids must be a JSON object, e.g. {"1":[0,3],"4":[2,5]}.'
            ) from exc
    if not isinstance(value, dict) or not value:
        raise ValueError("cka_loss_head_ids must be a nonempty layer-to-head-list object.")
    result = {}
    for layer, heads in value.items():
        if isinstance(layer, str) and layer.isascii() and layer.isdigit():
            layer = int(layer)
        if isinstance(layer, bool) or not isinstance(layer, Integral) or layer < 1:
            raise ValueError("cka_loss_head_ids layer IDs must be positive integers (1-based).")
        layer = int(layer)
        if layer in result:
            raise ValueError(f"cka_loss_head_ids contains duplicate layer ID {layer}.")
        if not isinstance(heads, list) or not heads:
            raise ValueError(f"cka_loss_head_ids layer {layer} requires a nonempty list of head IDs.")
        if any(isinstance(head, bool) or not isinstance(head, Integral) or head < 0 for head in heads):
            raise ValueError(f"cka_loss_head_ids layer {layer} head IDs must be nonnegative integers (0-based).")
        heads = [int(head) for head in heads]
        if len(set(heads)) != len(heads):
            raise ValueError(f"cka_loss_head_ids layer {layer} contains duplicate head IDs.")
        result[layer] = sorted(heads)
    return dict(sorted(result.items()))


def validate_head_fraction(fraction):
    """Return a finite selected-head fraction in ``(0, 1]``."""
    if isinstance(fraction, bool):
        raise ValueError("cka_loss_head_fraction must be a number in (0, 1].")
    try:
        fraction = float(fraction)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("cka_loss_head_fraction must be a number in (0, 1].") from exc
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("cka_loss_head_fraction must be finite and in (0, 1].")
    return fraction


def validate_head_seed(seed):
    """Return a nonnegative integer seed supported by a CPU generator."""
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("cka_loss_head_seed must be a nonnegative integer.")
    seed = int(seed)
    if not 0 <= seed <= 2**64 - 1:
        raise ValueError("cka_loss_head_seed must be in [0, 2**64 - 1].")
    return seed


def validate_head_count(count, num_heads=None):
    """Validate an optional exact number of selected query/output heads."""
    if count is None:
        return None
    if isinstance(count, bool) or not isinstance(count, Integral):
        raise ValueError("cka_loss_num_heads must be a positive integer or None.")
    count = int(count)
    if count < 1:
        raise ValueError("cka_loss_num_heads must be a positive integer or None.")
    if num_heads is not None and count > int(num_heads):
        raise ValueError(
            f"cka_loss_num_heads={count} exceeds the model's {int(num_heads)} query/output heads."
        )
    return count


def select_head_indices(
    num_heads,
    fraction=0.25,
    base_seed=42,
    layer_idx=0,
    selected_count=None,
):
    """Select a reproducible, sorted CPU LongTensor without touching global RNG.

    Each layer uses ``base_seed + layer_idx``. A selection contains no duplicate
    heads and stays fixed across steps, workers, and checkpoint recomputation.
    Independent layer seeds do not require the selected sets to be disjoint.
    """
    if isinstance(num_heads, bool) or not isinstance(num_heads, Integral) or num_heads < 1:
        raise ValueError("num_heads must be a positive integer.")
    if isinstance(layer_idx, bool) or not isinstance(layer_idx, Integral) or layer_idx < 0:
        raise ValueError("layer_idx must be a nonnegative integer.")
    selected_count = validate_head_count(selected_count, num_heads=num_heads)
    if selected_count is None:
        fraction = validate_head_fraction(fraction)
        selected_count = max(1, math.ceil(int(num_heads) * fraction))
    seed = validate_head_seed(base_seed)
    layer_seed = validate_head_seed(seed + int(layer_idx))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(layer_seed)
    return torch.randperm(int(num_heads), generator=generator, device="cpu")[:selected_count].sort().values


def compute_head_cka_loss(head_outputs, reference, vision_mask, tau=0.0, eps=1e-8):
    """Mean per-head CKA over samples containing at least two selected tokens.

    ``head_outputs`` has shape ``(B, T, K, Dh)`` for already selected heads;
    ``reference`` has shape ``(B, T, Dv)``, and ``vision_mask`` is ``(B, T)``.
    Only marked tokens participate in centering and CKA. The reference is
    detached, while the loss preserves gradients through the target heads.
    FP32 token-token Gram math is vectorized over heads; each sample's reference
    Gram is computed once and shared by all heads. No attention map is needed.
    """
    # llava_arch is imported lazily because backbones import this helper while
    # llava_arch is itself imported during model package initialization.
    from .llava_arch import (
        cka_similarity_to_loss,
        normalize_centered_cka_features,
        validate_cka_eps,
        validate_cka_loss_tau,
    )

    tau = validate_cka_loss_tau(tau)
    eps = validate_cka_eps(eps)
    if head_outputs.ndim != 4 or reference.ndim != 3:
        raise ValueError("Head CKA expects rank-4 head outputs and a rank-3 reference.")
    if head_outputs.shape[:2] != reference.shape[:2]:
        raise ValueError("Head CKA requires aligned batch and token axes.")
    if vision_mask.ndim != 2 or vision_mask.shape != head_outputs.shape[:2]:
        raise ValueError("Head CKA vision_mask must match the batch and token axes.")
    if head_outputs.device != reference.device:
        raise ValueError("Head CKA outputs and reference must be on the same device.")
    if not head_outputs.is_floating_point() or not reference.is_floating_point():
        raise ValueError("Head CKA outputs and reference must be floating-point tensors.")
    if min(head_outputs.shape[2:]) < 1 or reference.shape[-1] < 1:
        raise ValueError("Head CKA requires at least one head and nonempty feature dimensions.")

    device = head_outputs.device
    vision_mask = vision_mask.to(device=device, dtype=torch.bool)
    token_counts = vision_mask.sum(dim=1)
    valid_samples = token_counts >= 2
    if not bool(valid_samples.any().item()):
        # An empty-slice sum is graph-connected and remains zero even when
        # excluded target positions contain NaNs or infinities.
        return head_outputs[:, :0].float().sum()

    # Exclude text-only/one-token samples before Gram math, preserving a mean
    # over valid samples rather than diluting it by the original batch size.
    valid_rows = valid_samples.nonzero(as_tuple=True)[0]
    token_counts = token_counts[valid_samples]
    mask = vision_mask[valid_samples]
    max_tokens = int(token_counts.max().item())
    indices = mask.to(torch.int64).topk(max_tokens, dim=1).indices
    compact_mask = (
        torch.arange(max_tokens, device=device).unsqueeze(0)
        < token_counts.unsqueeze(1)
    )
    num_heads, head_dim = head_outputs.shape[2:]
    batch_size = token_counts.shape[0]

    with torch.autocast(device_type=device.type, enabled=False):
        # Jointly index samples/tokens before converting to FP32. Unlike a
        # full-sequence batch slice followed by gather, advanced indexing does
        # not retain the full (B, T, K, Dh) source for gather's backward.
        sample_indices = valid_rows[:, None]
        x = head_outputs[sample_indices, indices].float()
        y = reference.detach()[sample_indices, indices].float()
        x_mask = compact_mask[:, :, None, None]
        y_mask = compact_mask[:, :, None]
        x = x.masked_fill(~x_mask, 0.0)
        y = y.masked_fill(~y_mask, 0.0)
        x_mean = x.sum(dim=1, keepdim=True) / token_counts[:, None, None, None]
        y_mean = y.sum(dim=1, keepdim=True) / token_counts[:, None, None]
        x = (x - x_mean).masked_fill(~x_mask, 0.0)
        y = (y - y_mean).masked_fill(~y_mask, 0.0)

        # Normalization is per sample/head, never across the selected heads.
        x = x.permute(0, 2, 1, 3).reshape(batch_size * num_heads, max_tokens, head_dim)
        x = normalize_centered_cka_features(x)
        y = normalize_centered_cka_features(y)
        xx = torch.bmm(x, x.transpose(1, 2)).reshape(
            batch_size, num_heads, max_tokens, max_tokens
        )
        yy = torch.bmm(y, y.transpose(1, 2))
        hsic_xy = (xx * yy[:, None]).sum(dim=(-2, -1))
        hsic_xx = xx.square().sum(dim=(-2, -1))
        hsic_yy = yy.square().sum(dim=(-2, -1))[:, None]
        denominator = torch.sqrt(torch.clamp(hsic_xx * hsic_yy, min=eps))
        cka = (hsic_xy / denominator).clamp(0.0, 1.0)
        return cka_similarity_to_loss(cka, tau=tau).mean()
