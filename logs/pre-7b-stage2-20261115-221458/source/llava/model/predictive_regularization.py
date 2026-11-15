"""Patch-level PRe (arXiv:2603.20808), independent of the CKA objectives."""

import math

import torch
from torch.nn import functional as F


def validate_pre_config(config):
    if not getattr(config, "pre_loss", False):
        return
    if getattr(config, "model_type", None) != "llava_llama":
        raise ValueError("PRe currently supports the LLaVA-1.5 LLaMA/Vicuna backbone only.")
    if getattr(config, "cka_loss", False):
        raise ValueError("PRe reproduction requires cka_loss=False.")
    layer = getattr(config, "pre_loss_layer", 16)
    depth = config.num_hidden_layers
    if isinstance(layer, bool) or not isinstance(layer, int) or not 1 <= layer <= depth:
        raise ValueError(f"pre_loss_layer must be a 1-based decoder block in [1, {depth}].")
    weight = float(getattr(config, "pre_loss_weight", 0.5))
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("pre_loss_weight must be finite and nonnegative.")


def compute_predictive_regularization_loss(predictor, hidden_states, anchor, vision_feature_mask):
    """Negative cosine over corresponding image patches; detach only the anchor.

    Gathering before the MLP excludes text and padding, and avoids evaluating a
    4096-wide predictor on the entire language sequence. Empty batches still
    exercise the predictor and connect every parameter to a differentiable zero
    for distributed training with text-only modality groups.
    """
    if hidden_states.ndim != 3 or hidden_states.shape != anchor.shape:
        raise ValueError("PRe hidden states and post-projector anchors must have matching [B,L,D] shapes.")
    if vision_feature_mask.shape != hidden_states.shape[:2]:
        raise ValueError("PRe image-token mask must match the [B,L] feature shape.")
    mask = vision_feature_mask.to(device=hidden_states.device, dtype=torch.bool)
    predictions = predictor(hidden_states[mask])
    if predictions.numel() == 0:
        return predictions.float().sum()
    targets = anchor.detach().to(hidden_states.device)[mask]
    with torch.autocast(device_type=predictions.device.type, enabled=False):
        return -F.cosine_similarity(predictions.float(), targets.float(), dim=-1, eps=1e-8).mean()
