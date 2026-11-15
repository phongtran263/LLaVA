"""Scoped, deterministic Stage-1 forwards with one shared frozen language model.

This module deliberately does not change a backbone's forward or multimodal
expansion implementation. Temporary wrappers capture its normal expansion and
replay it with the *same* vision tensor and a frozen reference projector. All
wrappers, hooks, and training flags are restored even when a forward fails.
"""

from contextlib import contextmanager

import torch

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX


@contextmanager
def _temporary_attribute(instance, name, value):
    missing = object()
    previous = instance.__dict__.get(name, missing)
    setattr(instance, name, value)
    try:
        yield
    finally:
        if previous is missing:
            delattr(instance, name)
        else:
            setattr(instance, name, previous)


@contextmanager
def _deterministic_mode(model, reference_projector=None):
    # Restore each flag, not model.train(): nested modules can have deliberately
    # different train/eval settings. Eval mode does NOT disable student autograd.
    reference_modules = () if reference_projector is None else tuple(reference_projector.modules())
    modules = dict.fromkeys((*model.modules(), *reference_modules))
    flags = [(module, module.training) for module in modules]
    try:
        for module, _ in flags:
            module.training = False
        yield
    finally:
        for module, training in flags:
            module.training = training


def _same_optional_tensor(left, right):
    if left is None or right is None:
        return left is right
    return (left.shape == right.shape and left.dtype == right.dtype
            and left.device == right.device and torch.equal(left, right))


def _image_feature_indices(inputs):
    """Match flat expansion's indexing, excluding text-only dummy images."""
    input_ids = inputs["input_ids"]
    mask = inputs.get("attention_mask")
    image_tokens = input_ids.eq(IMAGE_TOKEN_INDEX)
    if mask is not None:
        if mask.ndim != 2 or mask.shape != input_ids.shape:
            raise ValueError("Protected CKA requires a 2D token attention mask.")
        image_tokens = image_tokens & mask.bool()
    indices = []
    cursor = 0
    for count in image_tokens.sum(dim=1).tolist():
        indices.extend(range(cursor, cursor + count))
        # The existing expansion consumes one dummy image for a text-only row.
        cursor += max(count, 1)
    return indices, cursor


def _validate_student_inputs(model, inputs):
    if not callable(getattr(model, "get_model", None)):
        raise ValueError("Protected CKA requires an unwrapped LLaVA model (no DDP/DeepSpeed).")
    backbone = model.get_model()
    if getattr(backbone.config, "cka_loss", False):
        raise ValueError("Protected CKA cannot be combined with the legacy cka_loss path.")
    if getattr(model, "is_gradient_checkpointing", False):
        raise ValueError("Protected CKA v1 does not support gradient checkpointing.")
    if getattr(backbone.config, "mm_patch_merge_type", "flat") != "flat":
        raise ValueError("Protected CKA v1 requires mm_patch_merge_type='flat' (no spatial/unpad).")
    projector = getattr(backbone, "mm_projector", None)
    if projector is None:
        raise ValueError("Protected CKA requires a student projector.")
    projector_ids = {id(parameter) for parameter in projector.parameters()}
    if any(parameter.requires_grad and id(parameter) not in projector_ids
           for parameter in model.parameters()):
        raise ValueError("Protected CKA requires the entire vision encoder and LLM to be frozen.")
    if any(parameter.dtype == torch.float16 for parameter in projector.parameters()):
        raise ValueError("Protected CKA v1 supports FP32/BF16, not FP16 projectors.")
    if inputs.get("inputs_embeds") is not None or inputs.get("past_key_values") is not None:
        raise ValueError("Protected CKA requires uncached teacher-forcing input_ids, not embeddings/past state.")
    input_ids, labels = inputs.get("input_ids"), inputs.get("labels")
    if (not torch.is_tensor(input_ids) or input_ids.ndim != 2
            or not torch.is_tensor(labels) or labels.shape != input_ids.shape):
        raise ValueError("Protected CKA requires aligned rank-2 input_ids and labels.")
    images = inputs.get("images")
    if images is not None and (not torch.is_tensor(images) or images.ndim != 4):
        raise ValueError("Protected CKA v1 requires rank-4 image tensors; lists/5D/multicrop are unsupported.")
    image_indices, expected_features = _image_feature_indices(inputs)
    vision_tower = model.get_vision_tower()
    if getattr(vision_tower, "select_feature", "patch") != "patch":
        raise ValueError("Protected CKA requires patch-only vision features, not CLS tokens.")
    if image_indices and (images is None or vision_tower is None or input_ids.shape[1] <= 1):
        raise ValueError("Protected CKA image tokens require images, a vision tower, and a full text prefix.")
    return projector, image_indices, expected_features, vision_tower


def _finite_student_ce(student, expanded_labels):
    # HF's mean-reduced CE is NaN when every shifted target is ignored. Return
    # a differentiable zero instead; no supervised token must mean no CE update.
    if not bool(expanded_labels[:, 1:].ne(IGNORE_INDEX).any()):
        return student.logits[..., :1].sum() * 0.0
    if student.loss is None:
        raise RuntimeError("Protected CKA requires the backbone's teacher-forcing CE loss.")
    return student.loss


def ce_only_forward(model, inputs):
    """One deterministic CE forward, with expanded labels and no reference copy.

    Used for Phase A, CE-only baseline continuation, and no-update calibration.
    Autograd remains enabled unless the caller explicitly disables it (e.g. for
    validation). The caller's module modes and batch are left unchanged.
    """
    _validate_student_inputs(model, inputs)
    batch = dict(inputs)
    batch.update(use_cache=False, return_dict=True)
    original_prepare = model.prepare_inputs_labels_for_multimodal
    expansions = []

    def capture_prepare(*args, **kwargs):
        result = original_prepare(*args, **kwargs)
        if not isinstance(result, tuple) or len(result) != 6:
            raise RuntimeError("Protected CKA requires the ordinary six-field multimodal expansion.")
        expansions.append(result)
        return result

    with _deterministic_mode(model), _temporary_attribute(
        model, "prepare_inputs_labels_for_multimodal", capture_prepare
    ):
        student = model(**batch)
    if len(expansions) != 1:
        raise RuntimeError("Protected CKA requires one multimodal expansion per CE-only batch.")
    expansion = expansions[0]
    return {
        "student_ce": _finite_student_ce(student, expansion[5]),
        "expanded_labels": expansion[5],
        "expanded_attention_mask": expansion[2],
        "expanded_position_ids": expansion[1],
    }


def protected_cka_forward(model, reference_projector, inputs, *, need_reference_logits=True):
    """Return CE, paired logits, expanded targets, and aligned patch features.

    ``inputs`` is the existing collator/prepared teacher-forcing batch. Supported
    images are a flat rank-4 tensor; list/multicrop/spatial-unpad layouts are
    intentionally rejected in v1. Only projector parameters may be trainable.
    The reference is an independently frozen projector created by the caller.

    Feature fields are lists of ``[patch, channel]`` tensors, one per real image
    (never text, padding, or dummy images). Student tensors retain their graph.
    Reference tensors are computed under ``no_grad``. Omitting reference logits
    saves the second LLM pass, but still computes reference projector features.
    This helper does not support concurrent forwards on the same model instance.
    """
    projector, image_indices, expected_features, vision_tower = _validate_student_inputs(model, inputs)
    if reference_projector is None:
        raise ValueError("Protected CKA requires a frozen reference projector.")
    projector_ids = {id(parameter) for parameter in projector.parameters()}
    if projector is reference_projector or any(
        id(parameter) in projector_ids for parameter in reference_projector.parameters()
    ):
        raise ValueError("Protected CKA reference projector must be an independent frozen copy.")
    if any(parameter.requires_grad for parameter in reference_projector.parameters()):
        raise ValueError("Protected CKA reference projector must be frozen.")

    batch = dict(inputs)
    batch.update(use_cache=False, return_dict=True)
    # Leave caller masks/positions and the backbone's causal attention untouched.
    captured = {"student": [], "reference": []}
    expansions = {"student": [], "reference": []}
    projector_autocast = None
    branch = "student"
    original_prepare = model.prepare_inputs_labels_for_multimodal

    def capture_prepare(*args, **kwargs):
        result = original_prepare(*args, **kwargs)
        if not isinstance(result, tuple) or len(result) != 6:
            raise RuntimeError("Protected CKA requires the ordinary six-field multimodal expansion.")
        expansions[branch].append(result)
        return result

    def capture_projector(module, args, output):
        nonlocal projector_autocast
        if (len(args) != 1 or not torch.is_tensor(args[0]) or args[0].ndim != 3
                or not torch.is_tensor(output) or output.ndim != 3
                or args[0].shape[:2] != output.shape[:2]):
            raise ValueError("Protected CKA requires aligned rank-3 vision/projector patch tensors.")
        captured[branch].append((args[0], output))
        if branch == "student":
            device_type = args[0].device.type
            projector_autocast = dict(device_type=device_type,
                                      enabled=torch.is_autocast_enabled(device_type),
                                      dtype=torch.get_autocast_dtype(device_type))

    handle = projector.register_forward_hook(capture_projector)
    try:
        with _deterministic_mode(model, reference_projector), _temporary_attribute(
            model, "prepare_inputs_labels_for_multimodal", capture_prepare
        ):
            # Crucially, this call is not wrapped in no_grad: CE/KL backpropagate
            # through the frozen decoder into the trainable student projector.
            student = model(**batch)
            if len(expansions["student"]) != 1 or len(captured["student"]) > 1:
                raise RuntimeError("Protected CKA requires one multimodal expansion/projector call per batch.")
            student_expansion = expansions["student"][0]
            reference_logits = None
            reference_features = None
            branch = "reference"
            if captured["student"]:
                features, projected = captured["student"][0]
                if features.requires_grad:
                    raise ValueError("Protected CKA vision features must be frozen (no input/vision gradients).")
                if features.shape[0] != expected_features:
                    raise ValueError("Protected CKA images do not match the multimodal expansion's image indexing.")

                def cached_vision(*args, **kwargs):
                    return features  # Exact same V object, not a second encoding.

                def reference_projection(value):
                    if value is not features:
                        raise RuntimeError("Protected CKA reference must use the exact student vision tensor.")
                    return reference_projector(value)

                with torch.no_grad():
                    if need_reference_logits:
                        with _temporary_attribute(vision_tower, "forward", cached_vision), _temporary_attribute(
                            projector, "forward", reference_projection
                        ):
                            reference_logits = model(**batch).logits
                        if len(captured["reference"]) != 1:
                            raise RuntimeError("Protected CKA expected one reference projector call.")
                        reference_features = captured["reference"][0][1]
                    else:
                        # Native Accelerate AMP can wrap only model.forward,
                        # not the caller's entire helper. Re-enter the student
                        # projector's exact AMP context for this direct call.
                        with torch.autocast(**projector_autocast):
                            reference_features = reference_projector(features)
            elif image_indices:
                raise RuntimeError("Protected CKA did not capture the image projector output.")
            elif need_reference_logits:
                with torch.no_grad():
                    reference_logits = model(**batch).logits

            if need_reference_logits:
                if len(expansions["reference"]) != 1:
                    raise RuntimeError("Protected CKA expected one reference multimodal expansion.")
                reference_expansion = expansions["reference"][0]
                for index, name in ((0, "input_ids"), (1, "position_ids"),
                                    (2, "attention_mask"), (5, "labels")):
                    if not _same_optional_tensor(student_expansion[index], reference_expansion[index]):
                        raise RuntimeError(f"Protected CKA student/reference expanded {name} differ.")
                if student.logits.shape != reference_logits.shape:
                    raise RuntimeError("Protected CKA student/reference logits are misaligned.")
    finally:
        handle.remove()

    expanded_labels = student_expansion[5]
    return {
        "student_ce": _finite_student_ce(student, expanded_labels),
        "student_logits": student.logits,
        "reference_logits": reference_logits,
        "expanded_labels": expanded_labels,
        "expanded_attention_mask": student_expansion[2],
        "expanded_position_ids": student_expansion[1],
        "vision_features": [features[index] for index in image_indices] if captured["student"] else [],
        "projected_features": [projected[index] for index in image_indices] if captured["student"] else [],
        "reference_projected_features": [reference_features[index] for index in image_indices]
        if captured["student"] else [],
    }
