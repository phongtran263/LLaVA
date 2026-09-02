import math
import os
import torch
import torch.nn as nn

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from transformers.modeling_utils import unwrap_model
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from llava.train.vsp_gradient_controller import (
    VSPGradientController,
    combine_partitioned_vsp_gradients,
    is_projector_parameter,
    validate_vsp_gradient_config,
    vsp_controller_requested,
    vsp_rewrites_gradients,
)


PCGRAD_EPS = 1e-12
PCGRAD_STAT_CHUNK_SIZE = 1_048_576


def _empty_pcgrad_stats(reference_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
    zero = torch.zeros((), device=reference_tensor.device, dtype=torch.float32)
    return {
        "dot_product": zero,
        "main_grad_norm": zero,
        "auxiliary_grad_norm": zero,
        "cosine_similarity": zero,
        "conflict": zero,
        "projection_magnitude": zero,
    }


def _pcgrad_coefficient_and_stats(
    dot_product: torch.Tensor,
    main_norm_sq: torch.Tensor,
    auxiliary_norm_sq: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    usable_main_gradient = main_norm_sq > float(eps)
    finite_statistics = (
        torch.isfinite(dot_product)
        & torch.isfinite(main_norm_sq)
        & torch.isfinite(auxiliary_norm_sq)
    )
    conflict = (dot_product < 0.0) & usable_main_gradient & finite_statistics
    safe_main_norm_sq = torch.where(
        usable_main_gradient,
        main_norm_sq,
        torch.ones_like(main_norm_sq),
    )
    coefficient = torch.where(
        conflict,
        dot_product / safe_main_norm_sq,
        torch.zeros_like(dot_product),
    )

    norm_product = (main_norm_sq * auxiliary_norm_sq).clamp_min(0.0).sqrt()
    cosine_similarity = torch.where(
        norm_product > float(eps),
        dot_product / norm_product,
        torch.zeros_like(dot_product),
    )
    stats = {
        "dot_product": dot_product.detach(),
        "main_grad_norm": main_norm_sq.clamp_min(0.0).sqrt().detach(),
        "auxiliary_grad_norm": auxiliary_norm_sq.clamp_min(0.0).sqrt().detach(),
        "cosine_similarity": cosine_similarity.detach(),
        "conflict": conflict.float().detach(),
        "projection_magnitude": (-coefficient).clamp_min(0.0).detach(),
    }
    return coefficient.detach(), stats


def _dense_float_gradient(gradient: torch.Tensor) -> torch.Tensor:
    gradient = gradient.detach()
    if gradient.is_sparse:
        gradient = gradient.coalesce().to_dense()
    return gradient.float()


def compute_pcgrad_projection_coefficient(
    main_gradients: Sequence[Optional[torch.Tensor]],
    auxiliary_gradients: Sequence[Optional[torch.Tensor]],
    reference_tensor: torch.Tensor,
    eps: float = PCGRAD_EPS,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Return the coefficient for projecting only the conflicting auxiliary part."""
    if len(main_gradients) != len(auxiliary_gradients):
        raise ValueError("PCGrad main and auxiliary gradient lists must have the same length.")
    if not math.isfinite(float(eps)) or eps <= 0.0:
        raise ValueError(f"PCGrad eps must be finite and positive, got {eps}.")

    dot_product = torch.zeros((), device=reference_tensor.device, dtype=torch.float32)
    main_norm_sq = torch.zeros_like(dot_product)
    auxiliary_norm_sq = torch.zeros_like(dot_product)

    with torch.no_grad():
        for main_gradient, auxiliary_gradient in zip(main_gradients, auxiliary_gradients):
            main_float = None
            if main_gradient is not None:
                main_float = _dense_float_gradient(main_gradient)
                main_norm_sq.add_(main_float.square().sum())
            if auxiliary_gradient is not None:
                auxiliary_float = _dense_float_gradient(auxiliary_gradient)
                auxiliary_norm_sq.add_(auxiliary_float.square().sum())
                if main_float is not None:
                    dot_product.add_((main_float * auxiliary_float).sum())

        return _pcgrad_coefficient_and_stats(
            dot_product,
            main_norm_sq,
            auxiliary_norm_sq,
            eps,
        )


def build_pcgrad_surrogate_loss(
    main_loss: torch.Tensor,
    auxiliary_loss: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter],
    eps: float = PCGRAD_EPS,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Build a scalar whose gradient is main + projected auxiliary gradient.

    This fallback is used by unsharded backends. The scalar is only a backward
    surrogate; callers must continue reporting ``main_loss + auxiliary_loss``.
    """
    parameters = tuple(parameter for parameter in parameters if parameter.requires_grad)
    if not parameters or not auxiliary_loss.requires_grad:
        return main_loss + auxiliary_loss, _empty_pcgrad_stats(main_loss)

    try:
        main_gradients = torch.autograd.grad(
            main_loss,
            parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        if not any(gradient is not None for gradient in main_gradients):
            raise RuntimeError(
                "PCGrad could not observe any main-loss gradients. Reentrant gradient "
                "checkpointing is a common cause; use use_reentrant=False."
            )
        auxiliary_gradients = torch.autograd.grad(
            auxiliary_loss,
            parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "PCGrad gradient probing failed. Use non-reentrant gradient checkpointing "
            "(gradient_checkpointing_kwargs={'use_reentrant': False})."
        ) from exc

    coefficient, stats = compute_pcgrad_projection_coefficient(
        main_gradients,
        auxiliary_gradients,
        reference_tensor=main_loss,
        eps=eps,
    )
    del main_gradients, auxiliary_gradients

    coefficient = coefficient.to(device=main_loss.device, dtype=main_loss.dtype)
    return (1.0 - coefficient) * main_loss + auxiliary_loss, stats


def project_pcgrad_gradient_parts(
    main_parts: Dict[int, List[torch.Tensor]],
    auxiliary_parts: Dict[int, List[torch.Tensor]],
    reference_tensor: torch.Tensor,
    process_group=None,
    eps: float = PCGRAD_EPS,
    chunk_size: int = PCGRAD_STAT_CHUNK_SIZE,
) -> Tuple[Dict[int, List[torch.Tensor]], Dict[str, torch.Tensor]]:
    """Project accumulated ZeRO-2 gradient shards and return the final shards."""
    if not math.isfinite(float(eps)) or eps <= 0.0:
        raise ValueError(f"PCGrad eps must be finite and positive, got {eps}.")
    if int(chunk_size) <= 0:
        raise ValueError(f"PCGrad chunk_size must be positive, got {chunk_size}.")

    group_keys = sorted(set(main_parts) | set(auxiliary_parts))
    statistics = torch.zeros(3, device=reference_tensor.device, dtype=torch.float64)

    with torch.no_grad():
        for group_key in group_keys:
            group_main = main_parts.get(group_key)
            group_auxiliary = auxiliary_parts.get(group_key)
            if group_main is not None and not isinstance(group_main, (list, tuple)):
                raise TypeError(f"PCGrad main gradient group {group_key} must be a list or tuple.")
            if group_auxiliary is not None and not isinstance(group_auxiliary, (list, tuple)):
                raise TypeError(f"PCGrad auxiliary gradient group {group_key} must be a list or tuple.")
            if group_main is not None and group_auxiliary is not None and len(group_main) != len(group_auxiliary):
                raise ValueError(f"PCGrad gradient group {group_key} has mismatched shard counts.")

            shard_count = len(group_main) if group_main is not None else len(group_auxiliary or [])
            for shard_index in range(shard_count):
                main_gradient = group_main[shard_index] if group_main is not None else None
                auxiliary_gradient = group_auxiliary[shard_index] if group_auxiliary is not None else None
                if main_gradient is not None and auxiliary_gradient is not None:
                    if main_gradient.shape != auxiliary_gradient.shape:
                        raise ValueError(
                            f"PCGrad gradient group {group_key} shard {shard_index} has "
                            "mismatched shapes."
                        )
                if main_gradient is None and auxiliary_gradient is None:
                    continue

                main_flat = main_gradient.detach().reshape(-1) if main_gradient is not None else None
                auxiliary_flat = (
                    auxiliary_gradient.detach().reshape(-1)
                    if auxiliary_gradient is not None
                    else None
                )
                numel = main_flat.numel() if main_flat is not None else auxiliary_flat.numel()
                for start in range(0, numel, int(chunk_size)):
                    stop = min(start + int(chunk_size), numel)
                    main_chunk = main_flat[start:stop].float() if main_flat is not None else None
                    auxiliary_chunk = (
                        auxiliary_flat[start:stop].float()
                        if auxiliary_flat is not None
                        else None
                    )
                    if main_chunk is not None:
                        statistics[1].add_(torch.dot(main_chunk, main_chunk).double())
                    if auxiliary_chunk is not None:
                        statistics[2].add_(torch.dot(auxiliary_chunk, auxiliary_chunk).double())
                        if main_chunk is not None:
                            statistics[0].add_(torch.dot(main_chunk, auxiliary_chunk).double())

        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size(group=process_group) > 1
        ):
            torch.distributed.all_reduce(
                statistics,
                op=torch.distributed.ReduceOp.SUM,
                group=process_group,
            )

        coefficient, stats = _pcgrad_coefficient_and_stats(
            statistics[0],
            statistics[1],
            statistics[2],
            eps,
        )
        main_scale = float((1.0 - coefficient).item())

        final_parts = {}
        for group_key in group_keys:
            group_main = main_parts.get(group_key)
            group_auxiliary = auxiliary_parts.get(group_key)
            if group_main is None:
                final_parts[group_key] = list(group_auxiliary)
                continue
            if group_auxiliary is None:
                final_parts[group_key] = list(group_main)
                continue

            final_group = []
            for main_gradient, auxiliary_gradient in zip(group_main, group_auxiliary):
                if auxiliary_gradient is None:
                    final_group.append(main_gradient)
                    continue
                if main_gradient is None:
                    final_group.append(auxiliary_gradient)
                    continue
                auxiliary_gradient.add_(main_gradient, alpha=main_scale)
                final_group.append(auxiliary_gradient)
            final_parts[group_key] = final_group

    return final_parts, stats


def sanitize_generation_config_for_save(model):
    model_to_save = unwrap_model(model)
    generation_config = getattr(model_to_save, "generation_config", None)
    if generation_config is None:
        return

    if getattr(generation_config, "do_sample", None) is False:
        for attr, default in {
            "temperature": 1.0,
            "top_p": 1.0,
            "typical_p": 1.0,
            "top_k": 50,
            "epsilon_cutoff": 0.0,
            "eta_cutoff": 0.0,
        }.items():
            if hasattr(generation_config, attr):
                setattr(generation_config, attr, default)

    if getattr(generation_config, "num_beams", None) in (None, 1):
        for attr, default in {
            "num_beams": 1,
            "early_stopping": False,
            "num_beam_groups": 1,
            "diversity_penalty": 0.0,
            "length_penalty": 1.0,
            "constraints": None,
        }.items():
            if hasattr(generation_config, attr):
                setattr(generation_config, attr, default)

    if (
        getattr(generation_config, "do_sample", None) is False
        and getattr(generation_config, "num_beams", None) == 1
        and getattr(generation_config, "num_return_sequences", None) != 1
    ):
        generation_config.num_return_sequences = 1


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if not self.model.config.cka_loss:
            return (loss, outputs) if return_outputs else loss

        # Model output splits CKA into the projector term and auxiliary LLM-hidden terms
        # so they can be weighted and logged separately.
        projector_cka_loss = outputs["projector_cka_loss"]
        aux_losses = outputs["aux_losses"]
        return (loss, projector_cka_loss, aux_losses, outputs) if return_outputs else (loss, projector_cka_loss, aux_losses)

    def _should_log_gradient_norms(self):
        if not getattr(self.args, 'log_gradient_norms', False):
            return False

        interval = max(1, int(getattr(self.args, 'gradient_log_steps', 1) or 1))
        global_step = int(getattr(self.state, 'global_step', 0) or 0)
        if global_step % interval != 0:
            return False
        if getattr(self, '_last_gradient_log_step', None) == global_step:
            return False

        self._last_gradient_log_step = global_step
        return True

    def _find_model_attr(self, model, attr_name):
        queue = [model]
        visited = set()

        while queue:
            current = queue.pop(0)
            if current is None or id(current) in visited:
                continue
            visited.add(id(current))

            try:
                if hasattr(current, attr_name):
                    return getattr(current, attr_name)
            except Exception:
                pass

            for child_attr in ('module', 'base_model', 'model', 'get_model'):
                try:
                    child = getattr(current, child_attr)
                except Exception:
                    continue
                if child_attr == 'get_model' and callable(child):
                    try:
                        child = child()
                    except Exception:
                        continue
                if isinstance(child, (list, tuple)):
                    queue.extend(child)
                else:
                    queue.append(child)

        return None

    def _set_model_attr(self, model, attr_name, value):
        queue = [model]
        try:
            queue.append(unwrap_model(model))
        except Exception:
            pass
        visited = set()

        while queue:
            current = queue.pop(0)
            if current is None or id(current) in visited:
                continue
            visited.add(id(current))

            try:
                if hasattr(current, attr_name):
                    setattr(current, attr_name, value)
            except Exception:
                pass

            for child_attr in ('module', 'base_model', 'model', 'get_model'):
                try:
                    child = getattr(current, child_attr)
                except Exception:
                    continue
                if child_attr == 'get_model' and callable(child):
                    try:
                        child = child()
                    except Exception:
                        continue
                if isinstance(child, (list, tuple)):
                    queue.extend(child)
                else:
                    queue.append(child)

    def _clear_gradient_log_tensors(self, model):
        if not (
            getattr(self.args, 'log_gradient_norms', False)
            or getattr(self.model.config, 'cka_loss', False)
            or self._vsp_controller_requested()
        ):
            return
        self._set_model_attr(model, 'last_cka_final_hidden', None)
        self._set_model_attr(model, 'last_cka_projector_output', None)
        self._set_model_attr(model, '_aux_losses', [])

    def _gradient_norm(self, loss, tensors, loss_name, target_name):
        if loss is None or not torch.is_tensor(loss) or not loss.requires_grad:
            return None

        tensors = [tensor for tensor in tensors if torch.is_tensor(tensor) and tensor.requires_grad]
        if len(tensors) == 0:
            return None

        try:
            grads = torch.autograd.grad(
                loss,
                tensors,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
        except RuntimeError as exc:
            warning_key = f'{loss_name}->{target_name}'
            warned = getattr(self, '_gradient_log_warnings', set())
            if warning_key not in warned:
                logger.warning(
                    "Skipping gradient norm log for %s because autograd.grad failed: %s",
                    warning_key,
                    str(exc).split('\n')[0],
                )
                warned.add(warning_key)
                self._gradient_log_warnings = warned
            return None

        grad_norms = []
        for grad in grads:
            if grad is None:
                continue
            if grad.is_sparse:
                grad = grad.coalesce().values()
            grad_norms.append(grad.detach().float().norm(2))

        if len(grad_norms) == 0:
            # A disconnected loss-target pair has a mathematically zero gradient.
            # Logging 0.0 keeps the W&B series visible on language-only batches.
            return 0.0

        return torch.stack(grad_norms).norm(2).item()

    def _sum_losses(self, losses):
        if not losses:
            return None

        total = losses[0]
        for loss in losses[1:]:
            total = total + loss
        return total

    def _drop_zero_weighted_cka_losses(self, projector_cka_loss, aux_losses):
        config = self.model.config
        if abs(float(getattr(config, 'cka_loss_projector_weight', 1.0) or 0.0)) <= 0.0:
            projector_cka_loss = None
        if abs(float(getattr(config, 'cka_loss_final_hidden_weight', 1.0) or 0.0)) <= 0.0:
            aux_losses = []
        return projector_cka_loss, aux_losses

    def _get_cka_auxiliary_loss(self, text_loss, projector_cka_loss=None, aux_losses=None):
        cka_terms = []
        if projector_cka_loss is not None:
            cka_terms.append(projector_cka_loss)
        cka_terms.extend(aux_losses or [])
        if not cka_terms:
            return text_loss.new_zeros(())
        return self._sum_losses(cka_terms)

    def _get_deepspeed_engine(self, model):
        for candidate in (model, getattr(self, 'deepspeed', None)):
            if (
                candidate is not None
                and callable(getattr(candidate, 'backward', None))
                and callable(getattr(candidate, 'step', None))
                and callable(getattr(candidate, 'zero_optimization_stage', None))
            ):
                return candidate
        return None

    def _get_deepspeed_zero_stage(self, model):
        engine = self._get_deepspeed_engine(model)
        if engine is not None:
            try:
                return int(engine.zero_optimization_stage())
            except (TypeError, ValueError):
                pass

        accelerator = getattr(self, 'accelerator', None)
        state = getattr(accelerator, 'state', None)
        plugin = getattr(state, 'deepspeed_plugin', None)
        stage = getattr(plugin, 'zero_stage', None)
        if stage is not None:
            try:
                return int(stage)
            except (TypeError, ValueError):
                pass
        return None

    def _validate_pcgrad_backend(self, model):
        zero_stage = self._get_deepspeed_zero_stage(model)
        if zero_stage is not None and zero_stage >= 3:
            raise RuntimeError(
                "CKA PCGrad currently supports DeepSpeed ZeRO-2 or lower, but "
                f"ZeRO-{zero_stage} is configured. Use scripts/zero2.json or disable PCGrad."
            )
        if getattr(self, 'is_fsdp_enabled', False):
            raise RuntimeError(
                "CKA PCGrad does not currently support FSDP parameter sharding. "
                "Use DeepSpeed ZeRO-2 or disable PCGrad."
            )
        if zero_stage != 2:
            world_size = int(getattr(self.args, 'world_size', 1) or 1)
            distributed_is_initialized = (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
                and torch.distributed.get_world_size() > 1
            )
            if world_size > 1 or distributed_is_initialized:
                raise RuntimeError(
                    "CKA PCGrad on non-ZeRO-2 distributed backends is not supported. "
                    "Use DeepSpeed ZeRO-2 for exact global PCGrad, or run PCGrad on a "
                    "single process."
                )
        return zero_stage

    def _vsp_controller_requested(self):
        return vsp_controller_requested(self.model.config)

    def _vsp_rewrites_gradients(self):
        return vsp_rewrites_gradients(self.model.config)

    def _get_vsp_gradient_controller(self, model, process_group=None):
        controller = getattr(self, '_vsp_gradient_controller', None)
        if controller is None or controller.model is not unwrap_model(model):
            controller = VSPGradientController(
                unwrap_model(model),
                self.model.config,
                process_group=process_group,
            )
            self._vsp_gradient_controller = controller
        else:
            controller.config = self.model.config
            controller.process_group = process_group
            validate_vsp_gradient_config(controller.config)
        return controller

    def _should_log_vsp_gradient_stats(self):
        interval = max(1, int(getattr(self.model.config, 'vsp_grad_log_interval', 10) or 10))
        global_step = int(getattr(self.state, 'global_step', 0) or 0)
        if global_step % interval != 0:
            return False
        if getattr(self, '_last_vsp_gradient_log_step', None) == global_step:
            return False
        self._last_vsp_gradient_log_step = global_step
        return True

    def _store_vsp_gradient_logs(self, logs):
        if logs and self._should_log_vsp_gradient_stats():
            self._last_vsp_gradient_logs = dict(logs)

    def _validate_vsp_gradient_backend(self, model):
        zero_stage = self._get_deepspeed_zero_stage(model)
        if zero_stage is not None and zero_stage >= 3:
            raise RuntimeError(
                "VSP gradient diagnostics/controller supports DeepSpeed ZeRO-2 or lower, "
                f"but ZeRO-{zero_stage} is configured. ZeRO-3 shards parameters in a way "
                "that this controller does not gather."
            )
        if getattr(self, 'is_fsdp_enabled', False):
            raise RuntimeError(
                "VSP gradient diagnostics/controller does not support FSDP parameter sharding. "
                "Use DeepSpeed ZeRO-2 or a single process."
            )
        if zero_stage != 2:
            world_size = int(getattr(self.args, 'world_size', 1) or 1)
            distributed_is_initialized = (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
                and torch.distributed.get_world_size() > 1
            )
            if world_size > 1 or distributed_is_initialized:
                raise RuntimeError(
                    "VSP gradient statistics on non-ZeRO-2 distributed backends would use "
                    "incomplete local gradients. Use DeepSpeed ZeRO-2 or a single process."
                )
            if self._vsp_rewrites_gradients():
                gas = int(getattr(self.args, 'gradient_accumulation_steps', 1) or 1)
                if gas != 1:
                    raise RuntimeError(
                        "VSP PCGrad/norm-cap on an unsharded backend currently requires "
                        "gradient_accumulation_steps=1. Use DeepSpeed ZeRO-2 for exact "
                        "accumulated-gradient surgery."
                    )
                if getattr(self.args, 'fp16', False):
                    raise RuntimeError(
                        "VSP PCGrad/norm-cap on an unsharded fp16 backend would bypass the "
                        "AMP scaler. Use bf16/fp32, disable the controller, or use ZeRO-2."
                    )
                if getattr(self, 'use_apex', False):
                    raise RuntimeError(
                        "VSP PCGrad/norm-cap on Apex AMP is not supported. Use bf16/fp32 "
                        "or DeepSpeed ZeRO-2."
                    )
            else:
                gas = int(getattr(self.args, 'gradient_accumulation_steps', 1) or 1)
                if gas != 1 and not getattr(self, '_vsp_diag_accum_warning_emitted', False):
                    logger.warning(
                        "VSP diagnostics on an unsharded backend with gradient accumulation "
                        "reports per-microbatch gradient statistics; the original accumulated "
                        "weighted-loss update is preserved."
                    )
                    self._vsp_diag_accum_warning_emitted = True
        return zero_stage

    def _get_zero2_vsp_group_names(self, zero_optimizer):
        param_groups = getattr(zero_optimizer, 'param_groups', None)
        if param_groups is None:
            wrapped_optimizer = getattr(zero_optimizer, 'optimizer', None)
            param_groups = getattr(wrapped_optimizer, 'param_groups', None)
        if not isinstance(param_groups, (list, tuple)):
            raise RuntimeError("Could not inspect DeepSpeed optimizer parameter groups for VSP control.")

        try:
            named_parameters = dict(unwrap_model(self.model).named_parameters())
        except Exception:
            named_parameters = dict(self.model.named_parameters())
        name_by_param_id = {id(param): name for name, param in named_parameters.items()}

        group_names = {}
        for group_index, group in enumerate(param_groups):
            configured_name = group.get('vsp_group') if isinstance(group, dict) else None
            if configured_name in ('projector', 'llm'):
                group_names[group_index] = configured_name
                continue

            has_projector = False
            has_llm = False
            for param in group.get('params', []):
                param_name = name_by_param_id.get(id(param), '')
                if is_projector_parameter(param_name):
                    has_projector = True
                else:
                    has_llm = True
            if has_projector and has_llm:
                raise RuntimeError(
                    "DeepSpeed optimizer group mixes projector and LLM parameters, so VSP "
                    "group-wise ratios would be wrong. Let LLaVATrainer create the optimizer "
                    "or split projector parameters into separate optimizer groups."
                )
            group_names[group_index] = 'projector' if has_projector else 'llm'

        return group_names

    def _validate_zero2_pcgrad_engine(self, engine):
        zero_optimizer = getattr(engine, 'optimizer', None)
        if zero_optimizer is None:
            raise RuntimeError("DeepSpeed ZeRO-2 PCGrad could not access the engine optimizer.")
        if not getattr(zero_optimizer, 'partition_gradients', False):
            raise RuntimeError("DeepSpeed PCGrad expected a ZeRO-2 gradient-partition optimizer.")
        if getattr(zero_optimizer, 'cpu_offload', False):
            raise RuntimeError(
                "DeepSpeed ZeRO-2 optimizer offload is not supported by CKA PCGrad. "
                "Use the non-offloaded scripts/zero2.json configuration."
            )
        averaged_gradients = getattr(zero_optimizer, 'averaged_gradients', None)
        if not isinstance(averaged_gradients, dict):
            raise RuntimeError(
                "This DeepSpeed version does not expose the ZeRO-2 averaged-gradient "
                "dictionary required by CKA PCGrad."
            )
        all_grad_tensors = getattr(zero_optimizer, 'all_grad_tensors', None)
        if all_grad_tensors is not None and not isinstance(all_grad_tensors, dict):
            raise RuntimeError("Unsupported DeepSpeed ZeRO-2 all_grad_tensors layout.")
        if (
            getattr(engine, 'has_moe_layers', False)
            or getattr(zero_optimizer, 'has_moe_layers', False)
            or getattr(engine, 'pipeline_parallelism', False)
        ):
            raise RuntimeError(
                "DeepSpeed MoE and pipeline parallelism are not supported by CKA PCGrad."
            )
        return zero_optimizer

    @staticmethod
    def _load_zero2_gradient_parts(live_parts, owned_parts):
        """Transfer gradient-part ownership into a DeepSpeed-owned dict."""
        if not isinstance(live_parts, dict) or not isinstance(owned_parts, dict):
            raise TypeError("ZeRO-2 PCGrad gradient-part state must be dictionary-backed.")
        if live_parts is owned_parts:
            raise RuntimeError("ZeRO-2 PCGrad cannot load a gradient dictionary into itself.")
        live_parts.clear()
        live_parts.update(owned_parts)
        owned_parts.clear()

    @staticmethod
    def _take_zero2_gradient_parts(live_parts):
        """Transfer non-empty gradient parts out without replacing DeepSpeed's dict."""
        if not isinstance(live_parts, dict):
            raise TypeError("ZeRO-2 PCGrad gradient-part state must be dictionary-backed.")
        owned_parts = {
            group_id: group_parts
            for group_id, group_parts in live_parts.items()
            if group_parts is not None
        }
        live_parts.clear()
        return owned_parts

    def _project_zero2_pcgrad_parts(
        self,
        zero_optimizer,
        main_parts,
        auxiliary_parts,
        reference_tensor,
    ):
        if not auxiliary_parts:
            self._last_pcgrad_stats = _empty_pcgrad_stats(reference_tensor)
            return main_parts

        final_parts, stats = project_pcgrad_gradient_parts(
            main_parts,
            auxiliary_parts,
            reference_tensor=reference_tensor,
            process_group=getattr(zero_optimizer, 'dp_process_group', None),
        )
        self._last_pcgrad_stats = stats

        # final_parts has its own lists and points at the now-projected auxiliary
        # tensors, so the unneeded main partition can be released before step().
        main_parts.clear()
        auxiliary_parts.clear()
        return final_parts

    def _deepspeed_zero2_pcgrad_backward(
        self,
        model,
        text_loss,
        cka_auxiliary_loss,
    ):
        engine = self._get_deepspeed_engine(model)
        if engine is None:
            raise RuntimeError("DeepSpeed ZeRO-2 PCGrad could not locate the DeepSpeed engine.")
        zero_optimizer = self._validate_zero2_pcgrad_engine(engine)

        accelerator = getattr(self, 'accelerator', None)
        if accelerator is None or not hasattr(accelerator, 'sync_gradients'):
            raise RuntimeError("DeepSpeed ZeRO-2 PCGrad requires Accelerate accumulation state.")
        sync_gradients = bool(accelerator.sync_gradients)
        engine.set_gradient_accumulation_boundary(sync_gradients)

        main_parts = getattr(self, '_pcgrad_zero2_main_parts', {})
        auxiliary_parts = getattr(self, '_pcgrad_zero2_auxiliary_parts', {})
        if not isinstance(main_parts, dict) or not isinstance(auxiliary_parts, dict):
            raise RuntimeError("Corrupt ZeRO-2 PCGrad accumulation state.")

        auxiliary_requires_grad = (
            torch.is_tensor(cka_auxiliary_loss)
            and bool(cka_auxiliary_loss.requires_grad)
        )
        uses_all_grad_layout = isinstance(
            getattr(zero_optimizer, 'all_grad_tensors', None),
            dict,
        )

        if uses_all_grad_layout:
            # DeepSpeed 0.18.x: all_grad_tensors accumulates across non-boundary
            # micro-batches; averaged_gradients is materialized only at boundary.
            live_accumulated = zero_optimizer.all_grad_tensors
            live_averaged = zero_optimizer.averaged_gradients

            # If older micro-batches produced auxiliary gradients but this boundary
            # batch has a disconnected/constant auxiliary, a zero backward is needed
            # solely to materialize the stored auxiliary accumulator.
            flush_stored_auxiliary = sync_gradients and bool(auxiliary_parts)
            run_auxiliary_backward = auxiliary_requires_grad or flush_stored_auxiliary

            self._load_zero2_gradient_parts(live_accumulated, main_parts)
            live_averaged.clear()
            engine.backward(text_loss, retain_graph=run_auxiliary_backward)
            if sync_gradients:
                main_parts = self._take_zero2_gradient_parts(live_averaged)
                live_accumulated.clear()
            else:
                main_parts = self._take_zero2_gradient_parts(live_accumulated)
                live_averaged.clear()

            if run_auxiliary_backward:
                self._load_zero2_gradient_parts(live_accumulated, auxiliary_parts)
                live_averaged.clear()
                auxiliary_backward_loss = (
                    cka_auxiliary_loss
                    if auxiliary_requires_grad
                    else text_loss * 0.0
                )
                engine.backward(auxiliary_backward_loss)
                if sync_gradients:
                    auxiliary_parts = self._take_zero2_gradient_parts(live_averaged)
                    live_accumulated.clear()
                else:
                    auxiliary_parts = self._take_zero2_gradient_parts(live_accumulated)
                    live_averaged.clear()

            if not sync_gradients:
                self._pcgrad_zero2_main_parts = main_parts
                self._pcgrad_zero2_auxiliary_parts = auxiliary_parts
                return

            final_parts = self._project_zero2_pcgrad_parts(
                zero_optimizer,
                main_parts,
                auxiliary_parts,
                text_loss,
            )
            live_averaged.clear()
            self._load_zero2_gradient_parts(live_averaged, final_parts)
            live_accumulated.clear()

        else:
            # DeepSpeed 0.15.x: averaged_gradients itself accumulates a reduced
            # partition on every micro-batch.
            live_averaged = zero_optimizer.averaged_gradients

            self._load_zero2_gradient_parts(live_averaged, main_parts)
            engine.backward(text_loss, retain_graph=auxiliary_requires_grad)
            main_parts = self._take_zero2_gradient_parts(live_averaged)

            if auxiliary_requires_grad:
                self._load_zero2_gradient_parts(live_averaged, auxiliary_parts)
                engine.backward(cka_auxiliary_loss)
                auxiliary_parts = self._take_zero2_gradient_parts(live_averaged)

            if not sync_gradients:
                self._pcgrad_zero2_main_parts = main_parts
                self._pcgrad_zero2_auxiliary_parts = auxiliary_parts
                return

            final_parts = self._project_zero2_pcgrad_parts(
                zero_optimizer,
                main_parts,
                auxiliary_parts,
                text_loss,
            )
            live_averaged.clear()
            self._load_zero2_gradient_parts(live_averaged, final_parts)

        # Accelerate's DeepSpeed optimizer/scheduler wrappers are no-ops. Calling
        # the engine directly avoids a step between the two backward passes.
        self._pcgrad_zero2_main_parts = {}
        self._pcgrad_zero2_auxiliary_parts = {}
        engine.step()

        # Successful step leaves group -> None; fp16 overflow may replace the dict.
        current_averaged = getattr(zero_optimizer, 'averaged_gradients', None)
        if isinstance(current_averaged, dict):
            current_averaged.clear()
        current_all_grad = getattr(zero_optimizer, 'all_grad_tensors', None)
        if isinstance(current_all_grad, dict):
            current_all_grad.clear()

    def _combine_zero2_vsp_parts(
        self,
        model,
        zero_optimizer,
        main_parts,
        proj_parts,
        final_parts,
        reference_tensor,
    ):
        controller = self._get_vsp_gradient_controller(
            model,
            process_group=getattr(zero_optimizer, 'dp_process_group', None),
        )
        final_zero2_parts, logs = combine_partitioned_vsp_gradients(
            controller,
            main_parts,
            proj_parts,
            final_parts,
            self._get_zero2_vsp_group_names(zero_optimizer),
            reference_tensor=reference_tensor,
        )
        self._store_vsp_gradient_logs(logs)
        main_parts.clear()
        proj_parts.clear()
        final_parts.clear()
        return final_zero2_parts

    def _deepspeed_zero2_vsp_backward(
        self,
        model,
        text_loss,
        projector_cka_loss,
        final_hidden_cka_loss,
    ):
        engine = self._get_deepspeed_engine(model)
        if engine is None:
            raise RuntimeError("DeepSpeed ZeRO-2 VSP controller could not locate the DeepSpeed engine.")
        zero_optimizer = self._validate_zero2_pcgrad_engine(engine)

        accelerator = getattr(self, 'accelerator', None)
        if accelerator is None or not hasattr(accelerator, 'sync_gradients'):
            raise RuntimeError("DeepSpeed ZeRO-2 VSP controller requires Accelerate accumulation state.")
        sync_gradients = bool(accelerator.sync_gradients)
        engine.set_gradient_accumulation_boundary(sync_gradients)

        main_parts = getattr(self, '_vsp_zero2_main_parts', {})
        proj_parts = getattr(self, '_vsp_zero2_proj_parts', {})
        final_parts = getattr(self, '_vsp_zero2_final_parts', {})
        if not all(isinstance(parts, dict) for parts in (main_parts, proj_parts, final_parts)):
            raise RuntimeError("Corrupt ZeRO-2 VSP accumulation state.")

        proj_requires_grad = torch.is_tensor(projector_cka_loss) and bool(projector_cka_loss.requires_grad)
        final_requires_grad = torch.is_tensor(final_hidden_cka_loss) and bool(final_hidden_cka_loss.requires_grad)
        uses_all_grad_layout = isinstance(getattr(zero_optimizer, 'all_grad_tensors', None), dict)

        if uses_all_grad_layout:
            live_accumulated = zero_optimizer.all_grad_tensors
            live_averaged = zero_optimizer.averaged_gradients

            flush_stored_proj = sync_gradients and bool(proj_parts)
            flush_stored_final = sync_gradients and bool(final_parts)
            run_proj_backward = proj_requires_grad or flush_stored_proj
            run_final_backward = final_requires_grad or flush_stored_final
            retain_for_auxiliary = run_proj_backward or run_final_backward

            self._load_zero2_gradient_parts(live_accumulated, main_parts)
            live_averaged.clear()
            engine.backward(text_loss, retain_graph=retain_for_auxiliary)
            if sync_gradients:
                main_parts = self._take_zero2_gradient_parts(live_averaged)
                live_accumulated.clear()
            else:
                main_parts = self._take_zero2_gradient_parts(live_accumulated)
                live_averaged.clear()

            if run_proj_backward:
                self._load_zero2_gradient_parts(live_accumulated, proj_parts)
                live_averaged.clear()
                proj_backward_loss = projector_cka_loss if proj_requires_grad else text_loss * 0.0
                engine.backward(proj_backward_loss, retain_graph=run_final_backward)
                if sync_gradients:
                    proj_parts = self._take_zero2_gradient_parts(live_averaged)
                    live_accumulated.clear()
                else:
                    proj_parts = self._take_zero2_gradient_parts(live_accumulated)
                    live_averaged.clear()

            if run_final_backward:
                self._load_zero2_gradient_parts(live_accumulated, final_parts)
                live_averaged.clear()
                final_backward_loss = final_hidden_cka_loss if final_requires_grad else text_loss * 0.0
                engine.backward(final_backward_loss)
                if sync_gradients:
                    final_parts = self._take_zero2_gradient_parts(live_averaged)
                    live_accumulated.clear()
                else:
                    final_parts = self._take_zero2_gradient_parts(live_accumulated)
                    live_averaged.clear()

            if not sync_gradients:
                self._vsp_zero2_main_parts = main_parts
                self._vsp_zero2_proj_parts = proj_parts
                self._vsp_zero2_final_parts = final_parts
                return

            final_zero2_parts = self._combine_zero2_vsp_parts(
                model,
                zero_optimizer,
                main_parts,
                proj_parts,
                final_parts,
                text_loss,
            )
            live_averaged.clear()
            self._load_zero2_gradient_parts(live_averaged, final_zero2_parts)
            live_accumulated.clear()

        else:
            live_averaged = zero_optimizer.averaged_gradients

            self._load_zero2_gradient_parts(live_averaged, main_parts)
            engine.backward(text_loss, retain_graph=proj_requires_grad or final_requires_grad)
            main_parts = self._take_zero2_gradient_parts(live_averaged)

            if proj_requires_grad:
                self._load_zero2_gradient_parts(live_averaged, proj_parts)
                engine.backward(projector_cka_loss, retain_graph=final_requires_grad)
                proj_parts = self._take_zero2_gradient_parts(live_averaged)

            if final_requires_grad:
                self._load_zero2_gradient_parts(live_averaged, final_parts)
                engine.backward(final_hidden_cka_loss)
                final_parts = self._take_zero2_gradient_parts(live_averaged)

            if not sync_gradients:
                self._vsp_zero2_main_parts = main_parts
                self._vsp_zero2_proj_parts = proj_parts
                self._vsp_zero2_final_parts = final_parts
                return

            final_zero2_parts = self._combine_zero2_vsp_parts(
                model,
                zero_optimizer,
                main_parts,
                proj_parts,
                final_parts,
                text_loss,
            )
            live_averaged.clear()
            self._load_zero2_gradient_parts(live_averaged, final_zero2_parts)

        # Accelerate's DeepSpeed optimizer/scheduler wrappers are no-ops; the
        # direct engine step keeps the three backward passes in one optimizer update.
        self._vsp_zero2_main_parts = {}
        self._vsp_zero2_proj_parts = {}
        self._vsp_zero2_final_parts = {}
        engine.step()

        current_averaged = getattr(zero_optimizer, 'averaged_gradients', None)
        if isinstance(current_averaged, dict):
            current_averaged.clear()
        current_all_grad = getattr(zero_optimizer, 'all_grad_tensors', None)
        if isinstance(current_all_grad, dict):
            current_all_grad.clear()

    def _build_pcgrad_backward_loss(self, model, text_loss, cka_auxiliary_loss):
        if not getattr(self, '_pcgrad_memory_warning_emitted', False):
            logger.warning(
                "CKA PCGrad on an unsharded backend performs two retained full-parameter "
                "gradient probes per micro-batch; full-model fine-tuning can use "
                "substantially more memory."
            )
            self._pcgrad_memory_warning_emitted = True

        backward_loss, stats = build_pcgrad_surrogate_loss(
            text_loss,
            cka_auxiliary_loss,
            model.parameters(),
        )
        self._last_pcgrad_stats = stats
        return backward_loss

    def _collect_gradient_norm_logs(self, model, text_loss, projector_cka_loss=None, aux_losses=None):
        if not self._should_log_gradient_norms():
            return

        self._last_gradient_norm_logs = None
        try:
            unwrapped_model = unwrap_model(model)
        except Exception:
            unwrapped_model = model

        try:
            projector_params = [
                param for name, param in unwrapped_model.named_parameters()
                if 'mm_projector' in name and param.requires_grad
            ]
        except Exception as exc:
            logger.warning("Could not collect mm_projector parameters for gradient logging: %s", exc)
            projector_params = []

        projector_output = self._find_model_attr(unwrapped_model, 'last_cka_projector_output')
        projector_output_tensors = [
            projector_output
        ] if torch.is_tensor(projector_output) and projector_output.requires_grad else []
        final_hidden = self._find_model_attr(unwrapped_model, 'last_cka_final_hidden')
        final_hidden_tensors = [final_hidden] if torch.is_tensor(final_hidden) and final_hidden.requires_grad else []
        final_hidden_cka_loss = self._sum_losses(aux_losses or [])

        cka_loss = None
        if projector_cka_loss is not None and final_hidden_cka_loss is not None:
            cka_loss = projector_cka_loss + final_hidden_cka_loss
        elif projector_cka_loss is not None:
            cka_loss = projector_cka_loss
        elif final_hidden_cka_loss is not None:
            cka_loss = final_hidden_cka_loss

        logs = {'grad_norm/measured_global_step': float(getattr(self.state, 'global_step', 0) or 0)}

        projector_losses = (
            ('text_loss', text_loss),
            ('cka_loss', cka_loss),
            ('projector_cka_loss', projector_cka_loss),
            ('final_hidden_cka_loss', final_hidden_cka_loss),
        )
        final_hidden_losses = (
            ('text_loss', text_loss),
            ('cka_loss', cka_loss),
            ('final_hidden_cka_loss', final_hidden_cka_loss),
        )
        target_specs = (
            ('projector_output', projector_output_tensors, projector_losses),
            ('projector_params', projector_params, projector_losses),
            ('final_hidden', final_hidden_tensors, final_hidden_losses),
        )
        for target_name, target_tensors, loss_specs in target_specs:
            for loss_name, loss_value in loss_specs:
                norm = self._gradient_norm(loss_value, target_tensors, loss_name, target_name)
                if norm is not None:
                    logs[f'grad_norm/{loss_name}/{target_name}'] = norm

        if len(logs) > 1:
            self._last_gradient_norm_logs = logs

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            if not self.model.config.cka_loss:
                text_loss = self.compute_loss(model, inputs)
                projector_cka_loss = None
                aux_losses = None
            else:
                text_loss, projector_cka_loss, aux_losses = self.compute_loss(model, inputs)
                projector_cka_loss, aux_losses = self._drop_zero_weighted_cka_losses(projector_cka_loss, aux_losses)

        if self.args.n_gpu > 1:
            text_loss = text_loss.mean()
            if projector_cka_loss is not None:
                projector_cka_loss = projector_cka_loss.mean()
        if self.model.config.cka_loss and self.args.n_gpu > 1:
            aux_losses = [aux_loss.mean() for aux_loss in aux_losses]

        self._collect_gradient_norm_logs(model, text_loss, projector_cka_loss, aux_losses)

        loss = text_loss
        backward_loss = text_loss
        use_vsp_controller = False
        zero_stage = None
        final_hidden_cka_loss = None
        if self.model.config.cka_loss:
            # The model has already applied the projector/final CKA weights.
            final_hidden_cka_loss = self._sum_losses(aux_losses or [])
            cka_auxiliary_loss = self._get_cka_auxiliary_loss(
                text_loss,
                projector_cka_loss,
                aux_losses,
            )
            loss = text_loss + cka_auxiliary_loss
            backward_loss = loss
            use_vsp_controller = self._vsp_controller_requested()
            if use_vsp_controller:
                zero_stage = self._validate_vsp_gradient_backend(model)

        try:
            if use_vsp_controller and zero_stage == 2:
                self._deepspeed_zero2_vsp_backward(
                    model,
                    text_loss,
                    projector_cka_loss,
                    final_hidden_cka_loss,
                )
            elif use_vsp_controller and self._vsp_rewrites_gradients():
                controller = self._get_vsp_gradient_controller(model)
                vsp_logs = controller.compute_and_assign_gradients(
                    text_loss,
                    projector_cka_loss,
                    final_hidden_cka_loss,
                )
                self._store_vsp_gradient_logs(vsp_logs)
            else:
                if use_vsp_controller:
                    controller = self._get_vsp_gradient_controller(model)
                    vsp_logs = controller.compute_diagnostics(
                        text_loss,
                        projector_cka_loss,
                        final_hidden_cka_loss,
                    )
                    self._store_vsp_gradient_logs(vsp_logs)
                if self.use_apex:
                    with amp.scale_loss(backward_loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    self.accelerator.backward(backward_loss)
        finally:
            self._clear_gradient_log_tensors(model)

        # Report the real objective, never the gradient-controller internals.
        return loss.detach() / self.args.gradient_accumulation_steps

    def _collect_router_stats(self):
        queue = [self.model]
        visited = set()

        while queue:
            model = queue.pop(0)
            if model is None or id(model) in visited:
                continue
            visited.add(id(model))

            router_stats = getattr(model, "router_last_stats", None)
            if router_stats:
                return router_stats, model

            for attr in ("get_model", "get_vision_tower", "vision_tower", "base_model", "model", "module"):
                if not hasattr(model, attr):
                    continue

                candidate = getattr(model, attr)
                child = candidate() if attr in ("get_model", "get_vision_tower") and callable(candidate) else candidate

                if isinstance(child, (list, tuple)):
                    queue.extend(child)
                else:
                    queue.append(child)

        return None, None

    def log(self, logs):
        logs = dict(logs)
        model = self.model.module if hasattr(self.model, 'module') else self.model

        cka_loss = getattr(model, 'last_cka_loss', None)
        text_loss = getattr(model, 'last_text_loss', None)
        cka_projector_loss = getattr(model, 'last_cka_projector_loss', getattr(model, 'last_cka_pre_post_loss', None))
        cka_pre_final_loss = getattr(model, 'last_cka_pre_final_loss', None)
        cka_layers_loss = getattr(model, 'last_cka_layers_loss', None)
        cka_per_layer_losses = getattr(model, 'last_cka_per_layer_losses', None)

        if cka_loss is not None:
            logs['loss/cka_loss'] = cka_loss.item() if torch.is_tensor(cka_loss) else float(cka_loss)
        if text_loss is not None:
            logs['loss/text_loss'] = text_loss.item() if torch.is_tensor(text_loss) else float(text_loss)
        if cka_projector_loss is not None:
            logs['loss/cka_projector_loss'] = cka_projector_loss.item() if torch.is_tensor(cka_projector_loss) else float(cka_projector_loss)
        if cka_pre_final_loss is not None:
            logs['loss/cka_pre_final_loss'] = cka_pre_final_loss.item() if torch.is_tensor(cka_pre_final_loss) else float(cka_pre_final_loss)
        if cka_layers_loss is not None:
            logs['loss/cka_layers_loss'] = cka_layers_loss.item() if torch.is_tensor(cka_layers_loss) else float(cka_layers_loss)
        if isinstance(cka_per_layer_losses, dict):
            for layer_name, layer_loss in sorted(cka_per_layer_losses.items()):
                logs[f'loss/cka_layers/{layer_name}'] = layer_loss.item() if torch.is_tensor(layer_loss) else float(layer_loss)

        gradient_norm_logs = getattr(self, '_last_gradient_norm_logs', None)
        if gradient_norm_logs:
            logs.update(gradient_norm_logs)
            self._last_gradient_norm_logs = None

        pcgrad_stats = getattr(self, '_last_pcgrad_stats', None)
        if pcgrad_stats:
            for stat_name, stat_value in pcgrad_stats.items():
                logs[f'pcgrad/{stat_name}'] = (
                    stat_value.item() if torch.is_tensor(stat_value) else float(stat_value)
                )
            self._last_pcgrad_stats = None

        vsp_gradient_logs = getattr(self, '_last_vsp_gradient_logs', None)
        if vsp_gradient_logs:
            logs.update(vsp_gradient_logs)
            self._last_vsp_gradient_logs = None

        return super().log(logs)

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            split_projector_groups = self.args.mm_projector_lr is not None or vsp_controller_requested(getattr(opt_model, 'config', self.model.config))
            if split_projector_groups:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if is_projector_parameter(name)]
                projector_lr_kwargs = {"lr": self.args.mm_projector_lr} if self.args.mm_projector_lr is not None else {}
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "vsp_group": "llm",
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "vsp_group": "llm",
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "vsp_group": "projector",
                        **projector_lr_kwargs,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "vsp_group": "projector",
                        **projector_lr_kwargs,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "vsp_group": "llm",
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "vsp_group": "llm",
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            sanitize_generation_config_for_save(self.model)
            super(LLaVATrainer, self)._save(output_dir, state_dict)
