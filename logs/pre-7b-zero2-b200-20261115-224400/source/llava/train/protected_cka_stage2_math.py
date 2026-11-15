"""Per-group Stage 2 gradient protection without concatenating model gradients.

The caller supplies effective-batch gradients after accumulation/unscaling and
any supported distributed reduction. A ``None`` CKA gradient is an active-set
boundary, not a zero: projection must never create an auxiliary update there.
Stage 1's math and training behavior are deliberately unchanged.
"""

import math
from numbers import Integral

import torch


STAT_CHUNK_SIZE = 1_048_576


def _scalar(name, value, *, minimum=0.0, maximum=None, strictly_positive=False):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < minimum or (strictly_positive and value == minimum):
        raise ValueError(f"{name} must be {'greater than' if strictly_positive else 'at least'} {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return value


def _layout(group_names, size, allowed_mask, chunk_size):
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    names = list(group_names)
    if len(names) != size or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("group_names must contain one nonempty group name per parameter")
    allowed = [True] * size if allowed_mask is None else list(allowed_mask)
    if len(allowed) != size or any(not isinstance(value, bool) for value in allowed):
        raise ValueError("allowed_mask must contain one boolean per parameter")
    groups = {name: [] for name in names}
    for index, name in enumerate(names):
        groups[name].append(index)
    return names, allowed, groups


def _flat(value):
    # No whole-group/model concatenation. A noncontiguous individual gradient
    # can require a same-dtype reshape copy; FP32 casts remain chunk-bounded.
    return value.detach().reshape(-1)


def _validate_tensors(sequences, names, groups, chunk_size):
    """Validate shapes/devices and reduce finite checks without per-chunk sync."""
    size = len(names)
    if any(len(values) != size for values in sequences.values()):
        raise ValueError("main, CKA, and EMA gradient lists must have matching lengths")
    devices, finite = {}, {}
    for index, group in enumerate(names):
        shape, device = None, None
        for label, values in sequences.items():
            value = values[index]
            if value is None:
                continue
            if not torch.is_tensor(value) or not value.is_floating_point() or value.layout != torch.strided:
                raise ValueError(f"{label}[{index}] must be a dense floating-point tensor or None")
            if shape is not None and (value.shape != shape or value.device != device):
                raise ValueError(f"gradient shapes/devices disagree for parameter {index}")
            shape, device = value.shape, value.device
            if group in devices and device != devices[group]:
                raise ValueError(f"all gradients in group {group!r} must share one device")
            if group not in devices:
                devices[group] = device
                finite[group] = torch.ones((), dtype=torch.bool, device=device)
            flattened = _flat(value)
            for start in range(0, flattened.numel(), chunk_size):
                # Catch values that become nonfinite in the required FP32 math.
                finite[group] = finite[group] & torch.isfinite(flattened[start:start + chunk_size].float()).all()
    for group, valid in finite.items():
        if not bool(valid):
            raise RuntimeError(f"Gradient group {group!r} contains NaN or Inf; refusing optimizer update")
    return {name: devices.get(name, torch.device("cpu")) for name in groups}


def _norm(values, indices, device, chunk_size):
    squared = torch.zeros((), dtype=torch.float32, device=device)
    for index in indices:
        if values[index] is None:
            continue
        flattened = _flat(values[index])
        for start in range(0, flattened.numel(), chunk_size):
            chunk = flattened[start:start + chunk_size].float()
            squared += torch.dot(chunk, chunk)
    result = squared.sqrt()
    if not bool(torch.isfinite(result)):
        raise RuntimeError("Protected group gradient norm is NaN or Inf; refusing optimizer update")
    return result


def _normalized_dot(left, right, indices, device, chunk_size, left_norm=1.0, right_norm=1.0):
    result = torch.zeros((), dtype=torch.float32, device=device)
    for index in indices:
        if left[index] is None or right[index] is None:
            continue
        left_flat, right_flat = _flat(left[index]), _flat(right[index])
        for start in range(0, left_flat.numel(), chunk_size):
            left_chunk = left_flat[start:start + chunk_size].float() / left_norm
            right_chunk = right_flat[start:start + chunk_size].float() / right_norm
            result += torch.dot(left_chunk, right_chunk)
    return result


def _auxiliary_chunks(cka, ema, coefficient, ema_norm, chunk_size):
    cka_flat = _flat(cka)
    ema_flat = None if ema is None else _flat(ema)
    for start in range(0, cka_flat.numel(), chunk_size):
        end = start + chunk_size
        auxiliary = cka_flat[start:end].float()
        if ema_flat is not None:
            auxiliary = auxiliary - coefficient * (ema_flat[start:end].float() / ema_norm)
        yield start, end, auxiliary


@torch.no_grad()
def group_gradient_norms(grads, group_names, allowed_mask=None, chunk_size=STAT_CHUNK_SIZE):
    """FP32 group CE-gradient norms for *Stage 2's own* fixed calibration.

    ``None`` gradients contribute no storage and zero squared norm. Groups with
    no selected gradients are included with norm zero. This function does not
    reuse Stage 1 calibration statistics or mix projector and LLM norms.
    """
    grads = list(grads)
    names, allowed, groups = _layout(group_names, len(grads), allowed_mask, chunk_size)
    devices = _validate_tensors({"gradient": grads}, names, groups, chunk_size)
    norms = {}
    for group, indices in groups.items():
        with torch.autocast(device_type=devices[group].type, enabled=False):
            norms[group] = float(_norm(grads, [i for i in indices if allowed[i]], devices[group], chunk_size))
    return norms


@torch.no_grad()
def protected_group_gradient(
    main_grads, cka_grads, ema_grads=None, *, group_names, ce_grad_refs,
    aux_ratios, allowed_cka_mask=None, ema_decay=0.99, lambda_t=1.0,
    eps=1e-8, stat_chunk_size=STAT_CHUNK_SIZE,
):
    """Return ``(final_grads, new_ema_grads, per_group_metrics)``.

    Every list position denotes the same student parameter. Statistics and caps
    are separate for each named group (normally ``projector`` and ``llm``).
    ``allowed_cka_mask`` is permanent routing, such as projector-only ablation;
    an additional, per-step active mask is ``cka_grads[i] is not None``.

    EMA is updated on allowed main gradients, then restricted to the *current*
    CKA active set for projection. Both signs of its parallel CKA component are
    removed. Missing main/previous EMA entries stay implicit zero without an
    allocation. No CKA support, a tiny active EMA, or zero auxiliary strength
    returns the exact original main gradient objects, including ``None``.

    Updated gradients and EMA are FP32. Unchanged final entries can alias the
    corresponding main input (and retain its dtype); inputs are never mutated.
    The caller owns any final parameter-dtype conversion and optimizer step.
    """
    decay = _scalar("ema_decay", ema_decay, maximum=1.0)
    weight = _scalar("lambda_t", lambda_t)
    eps = _scalar("eps", eps, strictly_positive=True)
    main, cka = list(main_grads), list(cka_grads)
    previous = [None] * len(main) if ema_grads is None else list(ema_grads)
    names, allowed, groups = _layout(group_names, len(main), allowed_cka_mask, stat_chunk_size)
    devices = _validate_tensors({"main": main, "cka": cka, "ema": previous}, names, groups, stat_chunk_size)
    caps = {}
    for group in groups:
        if group not in ce_grad_refs or group not in aux_ratios:
            raise ValueError(f"Missing Stage 2 calibrated CE norm or auxiliary ratio for group {group!r}")
        reference = _scalar(f"ce_grad_refs[{group!r}]", ce_grad_refs[group])
        ratio = _scalar(f"aux_ratios[{group!r}]", aux_ratios[group])
        caps[group] = _scalar(f"auxiliary cap for {group!r}", reference * ratio)

    final, ema, logs = list(main), [None] * len(main), {}
    for group, indices in groups.items():
        device = devices[group]
        with torch.autocast(device_type=device.type, enabled=False):
            eligible = [i for i in indices if allowed[i]]
            active = [i for i in eligible if cka[i] is not None]
            for index in eligible:
                current, old = main[index], previous[index]
                if current is None and old is None:
                    continue  # Do not allocate zeros for never-used parameters.
                template = current if current is not None else old
                updated = torch.empty_like(template, dtype=torch.float32, memory_format=torch.contiguous_format)
                updated_flat = updated.view(-1)
                current_flat = None if current is None else _flat(current)
                old_flat = None if old is None else _flat(old)
                for start in range(0, updated.numel(), stat_chunk_size):
                    end = start + stat_chunk_size
                    chunk = updated_flat[start:end]
                    if old_flat is None:
                        chunk.zero_()
                    else:
                        chunk.copy_(old_flat[start:end]).mul_(decay)
                    if current_flat is not None:
                        chunk.add_(current_flat[start:end].float(), alpha=1.0 - decay)
                ema[index] = updated

            main_norm = _norm(main, indices, device, stat_chunk_size)
            cka_norm = _norm(cka, active, device, stat_chunk_size)
            ema_norm = _norm(ema, active, device, stat_chunk_size)
            small_ema = bool(ema_norm <= eps)
            cosine = (float(_normalized_dot(main, cka, active, device, stat_chunk_size,
                                           left_norm=main_norm, right_norm=cka_norm))
                      if bool(main_norm > eps) and bool(cka_norm > eps) else 0.0)
            auxiliary_norm = 0.0
            scale = 0.0
            scaled_norm = 0.0
            if active and not small_ema:
                coefficient = _normalized_dot(cka, ema, active, device, stat_chunk_size, right_norm=ema_norm)
                squared = torch.zeros((), dtype=torch.float32, device=device)
                for index in active:
                    for _, _, chunk in _auxiliary_chunks(cka[index], ema[index], coefficient,
                                                        ema_norm, stat_chunk_size):
                        squared += torch.dot(chunk, chunk)
                auxiliary_norm = float(squared.sqrt())
                if not math.isfinite(auxiliary_norm):
                    raise RuntimeError("Protected group auxiliary norm is NaN or Inf; refusing optimizer update")
                scale = min(weight, caps[group] / (auxiliary_norm + eps))
                if scale > 0.0 and auxiliary_norm > 0.0:
                    finite = torch.ones((), dtype=torch.bool, device=device)
                    scaled_squared = torch.zeros((), dtype=torch.float32, device=device)
                    for index in active:
                        result = torch.empty_like(cka[index], dtype=torch.float32,
                                                  memory_format=torch.contiguous_format)
                        result_flat = result.view(-1)
                        main_flat = None if main[index] is None else _flat(main[index])
                        for start, end, auxiliary in _auxiliary_chunks(cka[index], ema[index], coefficient,
                                                                     ema_norm, stat_chunk_size):
                            scaled = auxiliary * scale
                            scaled_squared += torch.dot(scaled, scaled)
                            result_flat[start:end] = scaled if main_flat is None else main_flat[start:end].float() + scaled
                            finite = finite & torch.isfinite(result_flat[start:end]).all()
                        final[index] = result
                    if not bool(finite):
                        raise RuntimeError("Protected final gradient contains NaN or Inf; refusing optimizer update")
                    scaled_norm = float(scaled_squared.sqrt())
            logs[group] = {
                "main_norm": float(main_norm), "cka_norm": float(cka_norm), "ema_norm": float(ema_norm),
                "aux_norm": auxiliary_norm, "scaled_aux_norm": scaled_norm,
                "cosine": max(-1.0, min(1.0, cosine)), "scale": scale,
                "aux_cap": caps[group], "small_ema": int(small_ema),
                "allowed_tensors": len(eligible), "active_tensors": len(active),
                "active_parameters": sum(cka[i].numel() for i in active),
            }
    return final, ema, logs
