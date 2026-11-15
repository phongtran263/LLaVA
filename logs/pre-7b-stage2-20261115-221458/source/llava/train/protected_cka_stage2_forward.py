"""Stage-2 protected forwards; Stage-1 and backbone implementations are unchanged.

The student executes its existing CKA hooks/selector. We capture the actual
arguments of that method, replace only its scalar objective with target-margin
CKA, and evaluate the immutable teacher on the student's selected patches.
"""

import copy
from contextlib import ExitStack, contextmanager, nullcontext
from functools import partial

import torch
from torch import nn

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from llava.model import llava_arch
from llava.train.protected_cka_math import differentiable_zero, target_margin_cka


@contextmanager
def _attribute(obj, name, value):
    missing = object()
    previous = obj.__dict__.get(name, missing)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if previous is missing:
            delattr(obj, name)
        else:
            setattr(obj, name, previous)


def _core(model):
    core = model.get_base_model() if hasattr(model, "peft_config") else model
    if not callable(getattr(core, "get_model", None)):
        raise ValueError("Protected Stage 2 requires an unwrapped LLaVA model or PEFT LLaVA model.")
    return core


def _checkpointing_guard(model, *, deterministic=False):
    if not getattr(model, "is_gradient_checkpointing", False):
        return
    for module in model.modules():
        if not getattr(module, "gradient_checkpointing", False):
            continue
        function = getattr(module, "_gradient_checkpointing_func", None)
        if not isinstance(function, partial) or function.keywords.get("use_reentrant") is not False:
            raise ValueError("Protected Stage 2 requires gradient checkpointing with use_reentrant=False.")
    if deterministic and not getattr(model, "_protected_stage2_deterministic", False):
        raise ValueError("Protected Stage 2 checkpointing requires stage2_deterministic_context "
                         "around BOTH forward and autograd/backward.")


@contextmanager
def stage2_deterministic_context(model):
    """Keep this scope open through both VJPs, including checkpoint recompute.

    All stochastic child modules are in eval mode. The decoder container alone
    retains its training flag, which the existing implementation uses to enable
    CKA capture and non-reentrant checkpointing; its attention/dropout children
    remain deterministic. No student parameters or activations are detached.
    """
    core = _core(model)
    _checkpointing_guard(model)
    flags = [(module, module.training) for module in model.modules()]
    with _attribute(model, "_protected_stage2_deterministic", True):
        try:
            for module, _ in flags:
                module.training = False
            core.get_model().training = True
            yield
        finally:
            for module, training in flags:
                module.training = training


def snapshot_stage2_reference(model, placement="same"):
    """Create a fully independent frozen snapshot, including LoRA and buffers.

    ``cpu`` stores the teacher on CPU between forwards and temporarily moves it
    to the student's device for execution. It does not remove the requirement
    that student + teacher weights fit during a paired forward. No shared base
    weights, adapter switching, adapter merging, or mutable aliasing is used.
    """
    _core(model)
    if placement not in ("same", "cpu"):
        raise ValueError("Protected Stage 2 reference placement supports only 'same' or 'cpu'.")
    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        raise ValueError("Protected Stage 2 reference cloning does not support quantized models.")
    memo = {}
    try:
        # Explicit tensor memo avoids first allocating a duplicate GPU model
        # when the requested storage placement is CPU. Tied parameters stay tied.
        for tensor in (*model.parameters(), *model.buffers()):
            if id(tensor) in memo:
                continue
            value = tensor.detach().to(device="cpu" if placement == "cpu" else tensor.device, copy=True)
            memo[id(tensor)] = nn.Parameter(value, requires_grad=False) if isinstance(tensor, nn.Parameter) else value
        with ExitStack() as stack:
            for module in model.modules():
                # Accelerate's forward wrapper closes over the original model;
                # copying that closure would silently execute student weights.
                original = module.__dict__.get("_original_forward")
                if original is not None:
                    stack.enter_context(_attribute(module, "forward", original))
                for name in ("last_cka_final_hidden", "last_cka_projector_output", "_aux_losses"):
                    if name in module.__dict__:
                        stack.enter_context(_attribute(module, name, None))
            reference = copy.deepcopy(model, memo)
    except (RuntimeError, TypeError) as error:
        raise RuntimeError("Cannot allocate an independent Protected Stage 2 teacher. "
                           "Use reference placement='cpu' for storage or provision enough memory; "
                           "the teacher must not share mutable student weights.") from error
    for module in reference.modules():
        module.__dict__.pop("_original_forward", None)
    reference.requires_grad_(False).eval()
    reference._protected_stage2_reference_placement = placement
    return reference


def _same(left, right):
    if left is None or right is None:
        return left is right
    return left.shape == right.shape and torch.equal(left.to(right.device), right)


def _safe_ce(output, labels):
    if not bool(labels[:, 1:].ne(IGNORE_INDEX).any()):
        return differentiable_zero(output.logits)
    if output.loss is None:
        raise RuntimeError("Protected Stage 2 requires teacher-forcing labels and native CE.")
    return output.loss


def _validate(model, inputs, reference):
    core = _core(model)
    ids, labels = inputs.get("input_ids"), inputs.get("labels")
    if not torch.is_tensor(ids) or ids.ndim != 2 or not torch.is_tensor(labels) or labels.shape != ids.shape:
        raise ValueError("Protected Stage 2 requires aligned rank-2 input_ids and labels.")
    if inputs.get("inputs_embeds") is not None or inputs.get("past_key_values") is not None:
        raise ValueError("Protected Stage 2 requires uncached teacher-forcing input_ids.")
    images = inputs.get("images")
    if images is not None and (not torch.is_tensor(images) or images.ndim != 4):
        raise ValueError("Protected Stage 2 v1 supports rank-4 images, not lists/5D/multicrop.")
    if getattr(core.config, "mm_patch_merge_type", "flat") != "flat":
        raise ValueError("Protected Stage 2 v1 requires flat patch merging, not spatial/unpad.")
    mask = inputs.get("attention_mask")
    counts = ids.eq(IMAGE_TOKEN_INDEX)
    if mask is not None:
        if mask.ndim != 2 or mask.shape != ids.shape:
            raise ValueError("Protected Stage 2 requires a rank-2 token attention mask.")
        counts = counts & mask.bool()
    counts = counts.sum(1)
    if bool((counts > 1).any()):
        raise ValueError("Protected Stage 2 v1 supports at most one image per conversation.")
    rows = counts.gt(0)
    vision = core.get_vision_tower()
    if bool(rows.any()) and (images is None or vision is None or ids.shape[1] <= 1):
        raise ValueError("Protected Stage 2 image tokens require images and a vision tower.")
    if images is not None and images.shape[0] != ids.shape[0]:
        raise ValueError("Protected Stage 2 requires one image/dummy image per batch row.")
    if vision is not None and any(parameter.requires_grad for parameter in vision.parameters()):
        raise ValueError("Protected Stage 2 currently requires the recipe's frozen vision encoder.")
    _checkpointing_guard(model, deterministic=reference is not None)
    if reference is not None:
        if reference is model or any(parameter.requires_grad for parameter in reference.parameters()):
            raise ValueError("Protected Stage 2 requires an independent frozen reference model.")
        mutable = {(parameter.device, parameter.data_ptr()) for parameter in model.parameters() if parameter.requires_grad}
        if any((parameter.device, parameter.data_ptr()) in mutable for parameter in reference.parameters()):
            raise ValueError("Protected Stage 2 teacher must not share mutable student weights.")
    return core, rows


def _term(anchor, student, reference_anchor, reference, mask, margin, eps):
    losses, observations = [], {}
    with torch.autocast(device_type=student.device.type, enabled=False):
        for index in range(student.shape[0]):
            selected = mask[index].detach().bool()
            if not bool(selected.any()):
                continue
            if reference_anchor is anchor:
                # The existing frozen-vision-anchor definition is exactly the
                # Stage-1 math. Keep the common implementation for that case.
                loss, metrics = target_margin_cka(
                    anchor[index][selected].unsqueeze(0), student[index][selected].unsqueeze(0),
                    reference[index][selected].unsqueeze(0), margin=margin, eps=eps)
                if metrics["valid_images"]:
                    losses.append(loss)
                    observations[index] = (metrics["cka_ref"], metrics["cka_current"], metrics["target_met"])
                continue
            values = (anchor[index][selected].detach().float(), student[index][selected].float(),
                      reference_anchor[index][selected].detach().float(), reference[index][selected].detach().float())
            if any(not bool(torch.isfinite(value).all()) for value in values):
                continue
            grams = [(value - value.mean(0, keepdim=True)) for value in values]
            grams = [value @ value.T for value in grams]
            norms = [torch.linalg.vector_norm(value) for value in grams]
            if any(not bool(torch.isfinite(norm)) or bool(norm <= eps) for norm in norms):
                continue
            a, s, ar, r = [gram / norm for gram, norm in zip(grams, norms)]
            current = (a * s).sum().clamp(0, 1)
            frozen = (ar * r).sum().clamp(0, 1).detach()
            target = (frozen + margin).clamp(max=1).detach()
            losses.append(torch.relu(target - current).square())
            observations[index] = (float(frozen), float(current.detach()), float(current.detach() >= target))
    return (torch.stack(losses).mean() if losses else differentiable_zero(student)), observations


def stage2_forward(model, inputs, reference=None, *, need_cka=False,
                   need_reference_logits=True, margin=0.02, eps=1e-8):
    """Native Stage-2 CE plus paired logits and the existing method's CKA terms.

    Phase A (``reference=None, need_cka=False``) preserves normal training modes.
    Phase B is deterministic. With checkpointing, the caller MUST hold
    ``stage2_deterministic_context`` through gradient extraction/backward.
    The runtime owns KL, gradient routing/protection, reference persistence, and
    optimizer steps. Concurrent forwards on the same model are unsupported.
    """
    core, image_rows = _validate(model, inputs, reference)
    if need_cka and reference is None:
        raise ValueError("Target-margin Stage-2 CKA requires a frozen Stage-2 reference.")
    if margin < 0 or eps <= 0:
        raise ValueError("Protected Stage 2 requires a nonnegative margin and positive eps.")
    batch = dict(inputs, use_cache=False, return_dict=True)
    if reference is not None and batch.get("attention_mask") is None:
        # Native CKA retains the expansion padding mask for attention selection,
        # while the eval path otherwise returns None for an omitted input mask.
        # Explicit all-valid input tokens preserve that original meaning and
        # make both branches retain identical expansion padding afterwards.
        batch["attention_mask"] = torch.ones_like(batch["input_ids"], dtype=torch.bool)
    expansions, projected, hidden_terms, projector_terms = [], [], [], []
    student_hiddens, teacher_hiddens = {}, {}
    reference_expansions, reference_projected = [], []
    amp = None
    specs = core._get_cka_layer_specs() if need_cka and hasattr(core, "_get_cka_layer_specs") else []
    anchor_spec = core._get_cka_anchor_spec(specs) if specs else None
    capture_specs = list(specs)
    if anchor_spec is not None and anchor_spec not in capture_specs:
        capture_specs.append(anchor_spec)

    def prepare_capture(original, destination):
        def capture(*args, **kwargs):
            result = original(*args, **kwargs)
            if len(result) not in (6, 9):
                raise RuntimeError("Unexpected Stage-2 multimodal expansion contract.")
            destination.append(result)
            return result
        return capture

    def capture_projector(module, arguments, output):
        nonlocal amp
        if not arguments or arguments[0].ndim != 3 or output.ndim != 3:
            raise ValueError("Protected Stage 2 requires rank-3 projector patch features.")
        projected.append((arguments[0], output))
        device = arguments[0].device.type
        amp = dict(device_type=device, enabled=torch.is_autocast_enabled(device),
                   dtype=torch.get_autocast_dtype(device))

    def capture_amp(module, arguments):
        nonlocal amp
        device = arguments[0].device.type
        amp = dict(device_type=device, enabled=torch.is_autocast_enabled(device),
                   dtype=torch.get_autocast_dtype(device))

    def capture_projector_term(anchor, value, *args, **kwargs):
        projector_terms.append((anchor, value))
        return differentiable_zero(value)

    def capture_hidden_term(projected_features, layer_hidden_states, vision_feature_mask, **kwargs):
        name = next((key for key, value in student_hiddens.items() if value is projected_features), None)
        if name is None:
            raise RuntimeError("Cannot associate existing Stage-2 CKA activation with its configured layer.")
        hidden_terms.append((name, layer_hidden_states, projected_features, vision_feature_mask.detach()))
        return differentiable_zero(projected_features)

    def selection_without_autocast_weight_cache(original):
        def select(*args, **kwargs):
            hidden = kwargs.get("hidden_states")
            if hidden is None:
                return original(*args, **kwargs)
            device = hidden.device.type
            # Existing selection deliberately runs q/k projections in no_grad.
            # Under AMP, caching those detached weight casts would poison the
            # subsequent trainable attention call and change checkpoint saved
            # tensors when the selection hook is absent during recomputation.
            # Preserve identical BF16/FP32 math, but do not cache selector casts.
            with torch.autocast(device_type=device, enabled=torch.is_autocast_enabled(device),
                                dtype=torch.get_autocast_dtype(device), cache_enabled=False):
                return original(*args, **kwargs)
        return select

    own_determinism = reference is not None and not getattr(model, "_protected_stage2_deterministic", False)
    with ExitStack() as stack:
        if own_determinism:
            stack.enter_context(stage2_deterministic_context(model))
        stack.enter_context(_attribute(core.config, "cka_loss", bool(need_cka)))
        stack.enter_context(_attribute(core, "prepare_inputs_labels_for_multimodal",
                                       prepare_capture(core.prepare_inputs_labels_for_multimodal, expansions)))
        stack.callback(core.get_model().mm_projector.register_forward_hook(capture_projector).remove)
        stack.callback(core.lm_head.register_forward_pre_hook(capture_amp).remove)
        if need_cka:
            stack.enter_context(_attribute(llava_arch, "compute_linear_cka_loss", capture_projector_term))
            selector = getattr(core, "_select_vision_feature_subset_from_attention_inputs", None)
            if selector is not None:
                stack.enter_context(_attribute(core, "_select_vision_feature_subset_from_attention_inputs",
                                               selection_without_autocast_weight_cache(selector)))
            if hasattr(core, "_compute_masked_linear_cka_loss"):
                stack.enter_context(_attribute(core, "_compute_masked_linear_cka_loss", capture_hidden_term))
            if capture_specs:
                for handle in core._register_cka_layer_hooks(capture_specs, student_hiddens):
                    stack.callback(handle.remove)
        student = model(**batch)  # Student LLM/adapters remain trainable.
    if len(expansions) != 1 or len(projected) > 1:
        raise RuntimeError("Protected Stage 2 requires one expansion/projector call per batch.")
    expansion = expansions[0]
    if projected and projected[0][0].requires_grad:
        raise ValueError("Protected Stage 2 vision features must remain frozen.")
    teacher_logits = None
    if reference is not None:
        teacher = _core(reference)
        reference.eval()
        execution_device = next(model.parameters()).device
        offload = getattr(reference, "_protected_stage2_reference_placement", "same") == "cpu"
        run_decoder = need_reference_logits or bool(hidden_terms)
        execution_module = reference if run_decoder else teacher.get_model().mm_projector
        try:
            execution_module.to(execution_device)
            with torch.no_grad(), ExitStack() as stack:
                stack.enter_context(torch.autocast(**amp) if amp is not None else nullcontext())
                stack.enter_context(_attribute(teacher, "prepare_inputs_labels_for_multimodal",
                                               prepare_capture(teacher.prepare_inputs_labels_for_multimodal,
                                                               reference_expansions)))
                stack.callback(teacher.get_model().mm_projector.register_forward_hook(
                    lambda module, arguments, output: reference_projected.append(output)).remove)
                if projected:
                    features = projected[0][0]
                    stack.enter_context(_attribute(teacher.get_vision_tower(), "forward", lambda *args, **kw: features))
                if hidden_terms:
                    for handle in teacher._register_cka_layer_hooks(capture_specs, teacher_hiddens):
                        stack.callback(handle.remove)
                if run_decoder:
                    teacher_output = reference(**batch)
                    teacher_logits = teacher_output.logits if need_reference_logits else None
                elif projected:
                    teacher.get_model().mm_projector(projected[0][0])
        finally:
            if offload:
                execution_module.to("cpu")
        if run_decoder and len(reference_expansions) != 1:
            raise RuntimeError("Protected Stage 2 expected one teacher expansion.")
        if run_decoder:
            for index, name in ((0, "input_ids"), (1, "position_ids"), (2, "attention_mask"), (5, "labels")):
                if not _same(expansion[index], reference_expansions[0][index]):
                    raise RuntimeError(f"Protected Stage 2 student/reference expanded {name} differ.")

    # Never attach an artificial zero through logits: doing so makes unrelated
    # LLM/lm_head parameters appear connected to a projector-only CKA graph.
    cka = student.logits.new_zeros((), dtype=torch.float32)
    term_logs, by_image = {}, {}
    default_weight = getattr(core.config, "cka_loss_weight", 1.0)
    weights = [getattr(core.config, key, None) for key in ("cka_loss_projector_weight", "cka_loss_final_hidden_weight")]
    projector_weight, hidden_weight = [float(default_weight if value is None else value) for value in weights]

    def add_term(name, anchor, value, frozen_anchor, frozen, mask, weight):
        nonlocal cka
        loss, observations = _term(anchor, value, frozen_anchor, frozen, mask, margin, eps)
        # A dummy/degenerate-only term must not mark unrelated parameters as
        # CKA-connected: the controller uses None gradients as active boundaries.
        if observations:
            cka = cka + weight * loss
        term_logs[name] = {"loss": loss.detach(), "mask": mask.detach(), "weight": weight}
        for index, observation in observations.items():
            by_image.setdefault(index, []).append(observation)

    if projector_terms:
        if len(projector_terms) != 1 or len(reference_projected) != 1:
            raise RuntimeError("Protected Stage 2 expected one paired projector term.")
        anchor, value = projector_terms[0]
        patch_mask = image_rows[:, None].expand(value.shape[:2])
        add_term("projector", anchor, value, anchor, reference_projected[0], patch_mask, projector_weight)
    for name, anchor, value, mask in hidden_terms:
        frozen_anchor = teacher_hiddens[anchor_spec["name"]] if anchor_spec is not None else anchor
        add_term(name, anchor, value, frozen_anchor, teacher_hiddens[name], mask & image_rows[:, None], hidden_weight)
    summaries = [(sum(row[0] for row in values) / len(values),
                  sum(row[1] for row in values) / len(values), float(all(row[2] for row in values)))
                 for values in by_image.values()]
    valid = len(summaries)
    sums = [sum(row[index] for row in summaries) for index in range(3)]
    metrics = dict(valid_images=valid, invalid_images=int(image_rows.sum()) - valid if need_cka else 0,
                   cka_ref=sums[0] / max(valid, 1), cka_current=sums[1] / max(valid, 1),
                   target_met=sums[2] / max(valid, 1), cka_ref_sum=sums[0],
                   cka_current_sum=sums[1], target_met_sum=sums[2])
    real_images = image_rows.nonzero(as_tuple=False).flatten().tolist()
    return dict(student_ce=_safe_ce(student, expansion[5]), student_logits=student.logits,
                reference_logits=teacher_logits, expanded_labels=expansion[5],
                expanded_attention_mask=expansion[2], expanded_position_ids=expansion[1],
                image_rows=image_rows, cka_loss=cka, cka_metrics=metrics, cka_terms=term_logs,
                projected_features=[projected[0][1][index] for index in real_images] if projected else [],
                reference_projected_features=[reference_projected[0][index] for index in real_images]
                if reference_projected else [])
