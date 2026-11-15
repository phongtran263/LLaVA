"""FP32 math for Stage 1 protected CKA refinement.

These helpers do not own a model, optimizer, or training loop. In particular,
``protected_projector_gradient`` must receive the *effective-batch* gradients,
after any accumulation, unscaling, and distributed synchronization.
"""

import math
from numbers import Integral

import torch
import torch.nn.functional as F

from llava.constants import IGNORE_INDEX


def differentiable_zero(tensor):
    """A graph-connected FP32 zero, even if unused tensor values are nonfinite."""
    return tensor[..., :0].sum(dtype=torch.float32)


def _finite_scalar(name, value, *, minimum=None, maximum=None, strict_min=False):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    if minimum is not None and (value <= minimum if strict_min else value < minimum):
        relation = ">" if strict_min else ">="
        raise ValueError(f"{name} must be {relation} {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {value}")
    return value


class _ChunkedTokenKL(torch.autograd.Function):
    """Exact full-vocabulary KL with only chunk-sized FP32 softmax buffers.

    Retain the original logits, not every chunk's FP32 probabilities. Backward
    recomputes probabilities and writes the analytic (p_student - p_ref) / N
    gradient into the unavoidable student-logit gradient buffer. Only first
    derivatives are required by the refinement gradient surgery.
    """

    @staticmethod
    def forward(ctx, student_logits, reference_logits, positions, chunk_size):
        ctx.save_for_backward(student_logits, reference_logits, positions)
        ctx.chunk_size = chunk_size
        count = positions.shape[0]
        total = student_logits.new_zeros((), dtype=torch.float32)
        with torch.autocast(device_type=student_logits.device.type, enabled=False):
            for start in range(0, count, chunk_size):
                batch, token = positions[start:start + chunk_size].unbind(dim=1)
                log_s = F.log_softmax(student_logits[batch, token].float(), dim=-1)
                log_r = F.log_softmax(reference_logits[batch, token].float(), dim=-1)
                total += (log_r.exp() * (log_r - log_s)).sum()
        return total / count

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        student, reference, positions = ctx.saved_tensors
        gradient = torch.zeros_like(student)
        count = positions.shape[0]
        with torch.autocast(device_type=student.device.type, enabled=False):
            multiplier = grad_output.float() / count
            for start in range(0, count, ctx.chunk_size):
                batch, token = positions[start:start + ctx.chunk_size].unbind(dim=1)
                probability_s = F.softmax(student[batch, token].float(), dim=-1)
                probability_r = F.softmax(reference[batch, token].float(), dim=-1)
                gradient[batch, token] = ((probability_s - probability_r) * multiplier).to(student.dtype)
        return gradient, None, None, None


def token_kl(student_logits, reference_logits, expanded_labels, chunk_size=128):
    """Return ``(KL(reference || student), supervised_token_count)``.

    Logits have shape [batch, expanded_sequence, vocabulary]. Labels must be
    the labels *after* multimodal expansion. The causal shift uses logit t to
    predict label t+1; image, prompt, and padding labels equal IGNORE_INDEX.
    Vocabulary probabilities are never approximated or averaged over vocab.
    """
    if student_logits.ndim != 3 or student_logits.shape != reference_logits.shape:
        raise ValueError("student and reference logits must have matching [B, T, vocabulary] shapes")
    if expanded_labels.shape != student_logits.shape[:2]:
        raise ValueError("expanded_labels must match the expanded [B, T] logit dimensions")
    if student_logits.shape[-1] == 0:
        raise ValueError("logits must have a nonempty vocabulary")
    if not student_logits.is_floating_point() or not reference_logits.is_floating_point():
        raise ValueError("student and reference logits must be floating point")
    if student_logits.device != reference_logits.device or expanded_labels.device != student_logits.device:
        raise ValueError("student logits, reference logits, and expanded_labels must share a device")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    positions = (expanded_labels[:, 1:] != IGNORE_INDEX).nonzero(as_tuple=False)
    count = positions.shape[0]
    if count == 0:
        return differentiable_zero(student_logits), 0
    loss = _ChunkedTokenKL.apply(student_logits, reference_logits.detach(), positions, int(chunk_size))
    return loss, count


def target_margin_cka(vision_features, projected_features, reference_features, margin=0.02, eps=1e-8):
    """Per-image centered linear CKA hinge loss and detached aggregate metrics.

    All inputs are [images, patches, features], with matching image and patch
    axes. Images with any nonfinite or near-zero centered Gram norm are omitted
    altogether. Even if every image is invalid the returned loss has a safe
    zero gradient with respect to the student, including nonfinite inputs.
    """
    margin = _finite_scalar("margin", margin, minimum=0.0)
    eps = _finite_scalar("eps", eps, minimum=0.0, strict_min=True)
    inputs = (vision_features, projected_features, reference_features)
    if any(value.ndim != 3 for value in inputs):
        raise ValueError("CKA features must have shape [images, patches, features]")
    if any(value.shape[:2] != projected_features.shape[:2] for value in inputs):
        raise ValueError("CKA inputs must have matching image and patch dimensions")
    if any(value.device != projected_features.device for value in inputs):
        raise ValueError("CKA inputs must share a device")
    if any(not value.is_floating_point() for value in inputs):
        raise ValueError("CKA inputs must be floating point")

    losses, references, currents, met = [], [], [], []
    with torch.autocast(device_type=projected_features.device.type, enabled=False):
        for vision, student, reference in zip(*inputs):
            values = (vision.detach().float(), student.float(), reference.detach().float())
            if any(not bool(torch.isfinite(value).all()) for value in values):
                continue
            grams = []
            for value in values:
                centered = value - value.mean(dim=0, keepdim=True)
                grams.append(centered @ centered.transpose(0, 1))
            norms = [torch.linalg.vector_norm(gram) for gram in grams]
            if any(not bool(torch.isfinite(norm)) or bool(norm <= eps) for norm in norms):
                continue
            # Normalize before the dot product to avoid an overflowing product
            # of norms; all three Gram computations remain strictly FP32.
            gram_v, gram_s, gram_r = [gram / norm for gram, norm in zip(grams, norms)]
            c_ref = (gram_v * gram_r).sum().clamp(0.0, 1.0).detach()
            c_student = (gram_v * gram_s).sum().clamp(0.0, 1.0)
            target = (c_ref + margin).clamp(max=1.0).detach()
            losses.append(F.relu(target - c_student).square())
            references.append(c_ref)
            currents.append(c_student.detach())
            met.append((c_student.detach() >= target).float())

    count = len(losses)
    loss = torch.stack(losses).mean() if count else differentiable_zero(projected_features)
    ref_sum = float(torch.stack(references).sum()) if count else 0.0
    current_sum = float(torch.stack(currents).sum()) if count else 0.0
    met_sum = float(torch.stack(met).sum()) if count else 0.0
    metrics = {
        "cka_ref": ref_sum / count if count else 0.0,
        "cka_current": current_sum / count if count else 0.0,
        "target_met": met_sum / count if count else 0.0,
        "valid_images": count,
        "invalid_images": projected_features.shape[0] - count,
        "cka_ref_sum": ref_sum,
        "cka_current_sum": current_sum,
        "target_met_sum": met_sum,
    }
    return loss, metrics


@torch.no_grad()
def protected_projector_gradient(
    g_main, g_cka, ema_main=None, *, ema_decay=0.99, aux_ratio=0.10,
    ce_grad_ref, lambda_t=1.0, eps=1e-8,
):
    """Protect an unchanged main gradient using a whole-projector projection.

    Inputs are flat, complete-projector vectors. The returned final gradient and
    new EMA are FP32 and do not alias/mutate inputs. ``ce_grad_ref`` is a fixed
    calibration statistic, not a per-step main/CE gradient norm. Projection
    removes *both* positive and negative parallel components of the CKA gradient.
    """
    decay = _finite_scalar("ema_decay", ema_decay, minimum=0.0, maximum=1.0)
    ratio = _finite_scalar("aux_ratio", aux_ratio, minimum=0.0)
    reference_norm = _finite_scalar("ce_grad_ref", ce_grad_ref, minimum=0.0)
    weight = _finite_scalar("lambda_t", lambda_t, minimum=0.0)
    eps = _finite_scalar("eps", eps, minimum=0.0, strict_min=True)
    vectors = {"g_main": g_main, "g_cka": g_cka}
    if ema_main is not None:
        vectors["ema_main"] = ema_main
    for name, vector in vectors.items():
        if vector.ndim != 1 or vector.shape != g_main.shape:
            raise ValueError(f"{name} must be a matching flat whole-projector gradient")
        if vector.device != g_main.device or not vector.is_floating_point():
            raise ValueError(f"{name} must be floating point on the main gradient's device")
        if not bool(torch.isfinite(vector).all()):
            raise RuntimeError(f"{name} contains NaN or Inf; refusing a protected optimizer update")

    with torch.autocast(device_type=g_main.device.type, enabled=False):
        main, cka = g_main.detach().float(), g_cka.detach().float()
        previous = torch.zeros_like(main) if ema_main is None else ema_main.detach().float()
        ema = decay * previous + (1.0 - decay) * main
        main_norm, cka_norm, ema_norm = [torch.linalg.vector_norm(value) for value in (main, cka, ema)]
        if not all(bool(torch.isfinite(norm)) for norm in (main_norm, cka_norm, ema_norm)):
            raise RuntimeError("Protected gradient norm is NaN or Inf; refusing an optimizer update")
        small_ema = bool(ema_norm <= eps)
        if small_ema:
            auxiliary = torch.zeros_like(cka)
        else:
            direction = ema / ema_norm
            auxiliary = cka - torch.dot(cka, direction) * direction
        auxiliary_norm = torch.linalg.vector_norm(auxiliary)
        if not bool(torch.isfinite(auxiliary_norm)):
            raise RuntimeError("Protected auxiliary norm is NaN or Inf; refusing an optimizer update")
        cap = ratio * reference_norm
        if not math.isfinite(cap):
            raise ValueError("aux_ratio * ce_grad_ref must be finite")
        scale = min(weight, cap / (float(auxiliary_norm) + eps))
        scaled = scale * auxiliary
        final = main + scaled
        if not bool(torch.isfinite(final).all()):
            raise RuntimeError("Protected final gradient contains NaN or Inf; refusing an optimizer update")
        cosine = float(torch.dot(main / main_norm, cka / cka_norm)) if bool(main_norm > eps) and bool(cka_norm > eps) else 0.0
        logs = {
            "main_norm": float(main_norm),
            "cka_norm": float(cka_norm),
            "ema_norm": float(ema_norm),
            "aux_norm": float(auxiliary_norm),
            "scaled_aux_norm": float(torch.linalg.vector_norm(scaled)),
            "cosine": max(-1.0, min(1.0, cosine)),
            "scale": scale,
            "aux_cap": cap,
            "small_ema": int(small_ema),
        }
    return final, ema, logs
