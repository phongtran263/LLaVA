import math
import os
import time
from contextlib import contextmanager
import torch
import torch.nn as nn
from packaging import version

from torch.utils.data import Sampler

from transformers import Trainer, TrainerCallback
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
from llava.train.adaptive_projector_pcgrad import (
    AdaptiveProjectorPCGradConfig,
    AdaptiveProjectorPCGradController,
    METADATA_FILE as ADAPTIVE_PCGRAD_METADATA_FILE,
    RESOLVED_CONFIG_FILE as ADAPTIVE_PCGRAD_CONFIG_FILE,
    STATE_FILE as ADAPTIVE_PCGRAD_STATE_FILE,
    projector_sha256,
    projector_signature,
    load_json,
    write_json,
)
from llava.train.projector_replay_reference import (
    ProjectorReplayAccumulator,
    ordered_trainable_parameters,
)


PCGRAD_EPS = 1e-12
PCGRAD_STAT_CHUNK_SIZE = 1_048_576


def validate_cka_loss_start_ratio(value):
    """Return a finite CKA start ratio in the closed interval [0, 1]."""
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"cka_loss_start_ratio must be a number in [0, 1], got {value!r}"
        ) from exc

    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(
            f"cka_loss_start_ratio must be finite and in [0, 1], got {value!r}"
        )
    return value


def get_cka_loss_schedule_state(enabled, start_ratio, global_step, max_steps):
    """Return whether CKA is active and its optimizer-step boundary."""
    start_ratio = validate_cka_loss_start_ratio(start_ratio)
    global_step = int(global_step)
    max_steps = int(max_steps)
    if global_step < 0:
        raise ValueError(f"global_step must be non-negative, got {global_step}")
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")

    start_step = int(math.ceil(start_ratio * max_steps))
    return bool(enabled) and global_step >= start_step, start_step


def validate_stop_after_step_ratio(value):
    """Validate an optional fractional optimizer-step stopping point."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"stop_after_step_ratio must be a number in (0, 1), got {value!r}"
        ) from exc

    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(
            f"stop_after_step_ratio must be finite and in (0, 1), got {value!r}"
        )
    return value


def get_stop_after_step(step_ratio, max_steps):
    """Return the optimizer step at which a partial full-horizon run stops."""
    step_ratio = validate_stop_after_step_ratio(step_ratio)
    max_steps = int(max_steps)
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    return int(math.ceil(step_ratio * max_steps))


def validate_save_at_step_ratio(value):
    """Validate an optional fractional optimizer-step checkpoint boundary."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"save_at_step_ratio must be a number in (0, 1), got {value!r}"
        ) from exc

    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(
            f"save_at_step_ratio must be finite and in (0, 1), got {value!r}"
        )
    return value


def get_save_at_step(step_ratio, max_steps):
    """Return the optimizer step at which the fractional checkpoint is saved."""
    step_ratio = validate_save_at_step_ratio(step_ratio)
    max_steps = int(max_steps)
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    return int(math.ceil(step_ratio * max_steps))


class StopAfterStepRatioCallback(TrainerCallback):
    """Save a resumable checkpoint and stop at a fraction of the full run."""

    def on_step_end(self, args, state, control, **kwargs):
        step_ratio = getattr(args, "stop_after_step_ratio", None)
        if step_ratio is None:
            return control

        stop_step = get_stop_after_step(step_ratio, state.max_steps)
        if state.global_step >= stop_step:
            control.should_save = True
            control.should_training_stop = True
        return control


class SaveAtStepRatioCallback(TrainerCallback):
    """Save once at a fraction of the full run without stopping training."""

    def on_step_end(self, args, state, control, **kwargs):
        step_ratio = getattr(args, "save_at_step_ratio", None)
        if step_ratio is None:
            return control

        save_step = get_save_at_step(step_ratio, state.max_steps)
        if state.global_step == save_step:
            control.should_save = True
        return control


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


def adaptive_projector_pcgrad_enabled(config) -> bool:
    return bool(getattr(config, "adaptive_projector_pcgrad", False))


def resolve_mm_projector_module(model) -> nn.Module:
    """Resolve the actual projector module without broad name matching."""
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
        projector = getattr(current, "mm_projector", None)
        if isinstance(projector, nn.Module):
            return projector
        for attribute in ("module", "base_model", "model", "get_model"):
            try:
                child = getattr(current, attribute)
            except Exception:
                continue
            if attribute == "get_model" and callable(child):
                try:
                    child = child()
                except Exception:
                    continue
            if isinstance(child, (list, tuple)):
                queue.extend(child)
            else:
                queue.append(child)
    raise RuntimeError("Adaptive projector PCGrad could not resolve model.mm_projector.")


class LLaVATrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._adaptive_pcgrad_controller = None
        self._adaptive_pcgrad_resume_state = None
        self._adaptive_pcgrad_resume_checkpoint = None
        self._adaptive_pcgrad_backend_validated = False
        self._adaptive_pcgrad_ce_sum = None
        self._adaptive_pcgrad_ce_count = None
        self._adaptive_pcgrad_window_start = None

        if not adaptive_projector_pcgrad_enabled(self.model.config):
            return
        if bool(getattr(self.args, "fp16", False)):
            raise RuntimeError(
                "Adaptive projector PCGrad currently supports bf16/fp32 ZeRO-2 only; "
                "fp16 loss-scaler integration is intentionally fail-fast."
            )
        if bool(getattr(self, "use_apex", False)):
            raise RuntimeError("Adaptive projector PCGrad does not support Apex AMP.")

        self._adaptive_pcgrad_projector = resolve_mm_projector_module(self.model)
        chunk_size = int(getattr(self.model.config, "adaptive_pcgrad_cka_chunk_size", 1) or 1)
        resolved = dict(getattr(self.model.config, "adaptive_projector_pcgrad_config", {}) or {})
        self._adaptive_pcgrad_replay = ProjectorReplayAccumulator(
            self._adaptive_pcgrad_projector,
            chunk_size=chunk_size,
            norm_floor=float(resolved.get("aux_norm_floor", 1e-12)),
            score_rtol=float(resolved.get("residual_rtol", 1e-7)),
        )
        names, parameters = ordered_trainable_parameters(self._adaptive_pcgrad_projector)
        self._adaptive_pcgrad_projector_names = tuple(names)
        self._adaptive_pcgrad_projector_parameters = tuple(parameters)
        self._adaptive_pcgrad_projector_parameter_ids = {id(parameter) for parameter in parameters}
        if not parameters:
            raise RuntimeError("Adaptive projector PCGrad found no trainable projector parameters.")

        try:
            import accelerate
            import deepspeed
            import transformers
            logger.info(
                "Adaptive projector PCGrad runtime: torch=%s transformers=%s accelerate=%s "
                "deepspeed=%s; requested backend=DeepSpeed ZeRO-2 bf16/fp32.",
                torch.__version__,
                transformers.__version__,
                accelerate.__version__,
                deepspeed.__version__,
            )
        except Exception as exc:
            logger.warning("Could not log all adaptive PCGrad package versions: %s", exc)

    def _get_adaptive_pcgrad_controller(self) -> AdaptiveProjectorPCGradController:
        controller = self._adaptive_pcgrad_controller
        if controller is not None:
            return controller

        resolved = dict(getattr(self.model.config, "adaptive_projector_pcgrad_config", {}) or {})
        resolved.pop("warmup_steps", None)
        resolved.pop("cka_chunk_size", None)
        resolved.pop("profile", None)
        resolved["planned_optimizer_steps"] = int(getattr(self.state, "max_steps", 0) or 0)
        allowed = {
            "stage",
            "planned_optimizer_steps",
            "max_aux_ratio",
            "warmup_ratio",
            "norm_ema_beta",
            "lambda_max",
            "ce_norm_floor",
            "aux_norm_floor",
            "residual_rtol",
        }
        unknown = sorted(set(resolved) - allowed)
        if unknown:
            raise RuntimeError(f"Unknown adaptive projector PCGrad config fields: {unknown}.")
        controller = AdaptiveProjectorPCGradController(
            AdaptiveProjectorPCGradConfig(**resolved)
        )
        if self._adaptive_pcgrad_resume_state is not None:
            controller.load_state_dict(self._adaptive_pcgrad_resume_state)
            self._adaptive_pcgrad_resume_state = None
        self._adaptive_pcgrad_controller = controller
        return controller

    def _validate_adaptive_pcgrad_backend(self, model):
        if self._adaptive_pcgrad_backend_validated:
            return self._get_deepspeed_engine(model), self._adaptive_pcgrad_zero_optimizer
        if not self.is_deepspeed_enabled:
            raise RuntimeError(
                "Adaptive projector PCGrad currently supports only non-offloaded "
                "DeepSpeed ZeRO-2. Use scripts/zero2.json."
            )
        if getattr(self, "is_fsdp_enabled", False):
            raise RuntimeError("Adaptive projector PCGrad does not support FSDP.")
        engine = self._get_deepspeed_engine(model)
        if engine is None or int(engine.zero_optimization_stage()) != 2:
            actual = None if engine is None else engine.zero_optimization_stage()
            raise RuntimeError(
                "Adaptive projector PCGrad requires DeepSpeed ZeRO-2; "
                f"detected stage {actual}."
            )
        zero_optimizer = self._validate_zero2_pcgrad_engine(engine)
        if getattr(zero_optimizer, "cpu_offload", False):
            raise RuntimeError("Adaptive projector PCGrad does not support ZeRO optimizer offload.")
        if getattr(engine, "has_moe_layers", False) or getattr(engine, "pipeline_parallelism", False):
            raise RuntimeError("Adaptive projector PCGrad does not support MoE or pipeline parallelism.")
        import accelerate
        import deepspeed
        import tokenizers
        import transformers
        supported_versions = {
            "torch": (torch.__version__, "2.7.1"),
            "transformers": (transformers.__version__, "4.51.3"),
            "tokenizers": (tokenizers.__version__, "0.21.2"),
            "accelerate": (accelerate.__version__, "1.6.0"),
            "deepspeed": (deepspeed.__version__, "0.18.9"),
        }
        version_mismatches = {
            package: (actual, expected)
            for package, (actual, expected) in supported_versions.items()
            if version.parse(actual).base_version != version.parse(expected).base_version
        }
        if version_mismatches:
            raise RuntimeError(
                "Adaptive projector PCGrad's ZeRO-2 storage adapter is restricted to its "
                f"pinned supported runtime stack; version mismatches: {version_mismatches}."
            )
        fp16_enabled = bool(engine.fp16_enabled())
        bf16_enabled = bool(engine.bfloat16_enabled())
        torch_autocast_enabled = bool(engine.torch_autocast_enabled())
        if (
            fp16_enabled
            or torch_autocast_enabled
            or getattr(self.accelerator, "scaler", None) is not None
        ):
            raise RuntimeError(
                "Adaptive projector PCGrad supports DeepSpeed BF16/FP32 only; "
                "an FP16/scaler/custom torch-autocast runtime was detected."
            )
        if bf16_enabled != bool(getattr(self.args, "bf16", False)):
            raise RuntimeError(
                "Adaptive projector PCGrad precision mismatch between Trainer and DeepSpeed: "
                f"args.bf16={bool(getattr(self.args, 'bf16', False))}, "
                f"engine.bfloat16_enabled()={bf16_enabled}."
            )
        self._adaptive_pcgrad_precision = "bf16" if bf16_enabled else "fp32"

        projector_ids = self._adaptive_pcgrad_projector_parameter_ids
        seen = set()
        for group_index, group in enumerate(zero_optimizer.round_robin_bit16_groups):
            group_ids = {id(parameter) for parameter in group}
            overlap = group_ids & projector_ids
            if overlap and overlap != group_ids:
                raise RuntimeError(
                    f"ZeRO optimizer group {group_index} mixes projector and decoder parameters."
                )
            seen.update(overlap)
        if seen != projector_ids:
            missing = len(projector_ids - seen)
            raise RuntimeError(
                f"ZeRO optimizer is missing {missing} trainable projector parameters."
            )
        self._adaptive_pcgrad_backend_validated = True
        self._adaptive_pcgrad_zero_optimizer = zero_optimizer
        logger.info(
            "Adaptive projector PCGrad backend validated: ZeRO-2, offload=false, "
            "precision=%s, projector_parameters=%d.",
            "bf16" if getattr(self.args, "bf16", False) else "fp32",
            len(projector_ids),
        )
        return engine, zero_optimizer

    @staticmethod
    def _adaptive_all_reduce_sum(tensor, process_group=None):
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

    def _adaptive_global_auxiliary(self, zero_optimizer):
        flat_sum, cka_sum, valid_count, invalid = self._adaptive_pcgrad_replay.consume_flat()
        process_group = getattr(zero_optimizer, "dp_process_group", None)

        ce_sum = self._adaptive_pcgrad_ce_sum
        ce_count = self._adaptive_pcgrad_ce_count
        if ce_sum is None or ce_count is None:
            ce_sum = cka_sum.new_zeros(())
            ce_count = cka_sum.new_zeros(())
        stat_names = (
            "cka_sum",
            "valid_count",
            *(f"invalid/{name}" for name in sorted(invalid)),
            "ce_sum",
            "ce_count",
        )
        stat_values = [cka_sum, valid_count]
        stat_values.extend(invalid[name] for name in sorted(invalid))
        stat_values.extend((ce_sum.double(), ce_count.double()))
        stat_names = (*stat_names, "aux_finite_ranks")
        stat_values.append(torch.isfinite(flat_sum).all().to(torch.float64))
        stats = torch.stack(stat_values)
        self._adaptive_all_reduce_sum(stats, process_group)
        # The boundary decision needs host scalars anyway. Copy this tiny
        # vector once instead of synchronizing once per logged statistic.
        global_stats = dict(zip(stat_names, stats.detach().cpu().unbind()))

        self._adaptive_pcgrad_ce_sum = None
        self._adaptive_pcgrad_ce_count = None
        global_count = float(global_stats["valid_count"].item())
        severe_invalid = (
            float(global_stats["invalid/nonfinite_input"].item())
            + float(global_stats["invalid/nonfinite_gram"].item())
            + float(global_stats["invalid/score_out_of_range"].item())
        )
        if severe_invalid > 0.0:
            self._adaptive_pcgrad_replay.recycle_flat_buffer(flat_sum)
            raise RuntimeError(
                "Adaptive projector CKA found non-finite/materially invalid observations; "
                "the whole effective update is aborted by policy."
            )
        expected_finite_ranks = float(
            torch.distributed.get_world_size(group=process_group)
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 1
        )
        if float(global_stats["aux_finite_ranks"].item()) != expected_finite_ranks:
            self._adaptive_pcgrad_replay.recycle_flat_buffer(flat_sum)
            raise RuntimeError("Adaptive projector CKA produced non-finite auxiliary gradients.")
        if global_count > 0.0:
            # The expensive projector-vector collective is unnecessary for an
            # all-text/all-invalid (but non-severe) effective batch.
            self._adaptive_all_reduce_sum(flat_sum, process_group)
            flat_sum.div_(global_count)
        else:
            flat_sum.zero_()
        return flat_sum, global_stats, process_group

    def _adaptive_zero2_projector_parts(self, zero_optimizer, flat_auxiliary):
        auxiliary_by_parameter = {}
        offset = 0
        for parameter in self._adaptive_pcgrad_projector_parameters:
            auxiliary_by_parameter[id(parameter)] = flat_auxiliary.narrow(
                0, offset, parameter.numel()
            ).view_as(parameter)
            offset += parameter.numel()
        if offset != flat_auxiliary.numel():
            raise RuntimeError("Projector replay gradient buffer no longer matches live parameters.")

        main_parts = []
        auxiliary_parts = []
        destinations = []
        process_group = None
        averaged = zero_optimizer.averaged_gradients
        projector_ids = self._adaptive_pcgrad_projector_parameter_ids
        for group_index, group in enumerate(zero_optimizer.round_robin_bit16_groups):
            group_ids = {id(parameter) for parameter in group}
            if not (group_ids & projector_ids):
                continue
            if group_ids != (group_ids & projector_ids):
                raise RuntimeError(f"Mixed ZeRO optimizer group {group_index} is unsupported.")
            group_main = averaged.get(group_index)
            if not isinstance(group_main, (list, tuple)):
                raise RuntimeError(
                    f"ZeRO-2 did not materialize projector gradient shard {group_index}."
                )
            group_process = zero_optimizer.real_dp_process_group[group_index]
            if process_group is None:
                process_group = group_process
            elif process_group is not group_process:
                raise RuntimeError("Projector optimizer groups use different data-parallel groups.")
            rank = torch.distributed.get_rank(group=group_process)
            partition_size = int(zero_optimizer.partition_size[group_index])
            partition_start = rank * partition_size
            partition_end = partition_start + partition_size
            device = group_main[0].device
            buffers = getattr(self, "_adaptive_pcgrad_local_auxiliary_buffers", None)
            if buffers is None:
                buffers = {}
                self._adaptive_pcgrad_local_auxiliary_buffers = buffers
            buffer_key = (group_index, device, partition_size)
            local_auxiliary = buffers.get(buffer_key)
            if local_auxiliary is None:
                local_auxiliary = torch.zeros(
                    partition_size, device=device, dtype=torch.float32
                )
                buffers[buffer_key] = local_auxiliary
            else:
                local_auxiliary.zero_()
            group_offset = 0
            for parameter in group:
                parameter_end = group_offset + parameter.numel()
                overlap_start = max(group_offset, partition_start)
                overlap_end = min(parameter_end, partition_end)
                if overlap_end > overlap_start:
                    source_offset = overlap_start - group_offset
                    destination_offset = overlap_start - partition_start
                    length = overlap_end - overlap_start
                    source = auxiliary_by_parameter[id(parameter)].reshape(-1).narrow(
                        0, source_offset, length
                    )
                    local_auxiliary.narrow(0, destination_offset, length).copy_(source)
                group_offset = parameter_end

            local_offset = 0
            for main_part in group_main:
                numel = main_part.numel()
                auxiliary_part = local_auxiliary.narrow(0, local_offset, numel).view_as(main_part)
                main_parts.append(main_part)
                auxiliary_parts.append(auxiliary_part)
                destinations.append(main_part)
                local_offset += numel
            if local_offset != partition_size:
                raise RuntimeError(
                    f"ZeRO projector shard {group_index} has {local_offset} elements; "
                    f"expected {partition_size}."
                )
        if not main_parts:
            raise RuntimeError("No ZeRO-2 projector gradient shards were found.")
        return main_parts, auxiliary_parts, destinations, process_group

    def _adaptive_training_step(self, model, inputs, num_items_in_batch=None):
        controller = self._get_adaptive_pcgrad_controller()
        engine, zero_optimizer = self._validate_adaptive_pcgrad_backend(model)
        sync_gradients = bool(self.accelerator.sync_gradients)
        engine.set_gradient_accumulation_boundary(sync_gradients)

        new_window = self._adaptive_pcgrad_replay.snapshot_token is None
        self._adaptive_pcgrad_replay.begin_window(
            controller.config.stage,
            controller.successful_steps,
        )
        profile = bool(getattr(self.model.config, "adaptive_pcgrad_profile", False))
        if new_window and profile:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            self._adaptive_pcgrad_window_start = time.perf_counter()

        self._set_model_attr(model, "_adaptive_projector_replay_features", [])
        with self.compute_loss_context_manager():
            text_loss = self.compute_loss(model, inputs)
        if self.args.n_gpu > 1:
            text_loss = text_loss.mean()

        captured = self._find_model_attr(model, "_adaptive_projector_replay_features")
        if captured is None:
            captured = []
        if not isinstance(captured, (list, tuple)):
            raise RuntimeError("Adaptive projector replay capture must be a list of tensors.")
        detached_loss = text_loss.detach().double()
        if self._adaptive_pcgrad_ce_sum is None:
            self._adaptive_pcgrad_ce_sum = torch.zeros_like(detached_loss)
            self._adaptive_pcgrad_ce_count = torch.zeros_like(detached_loss)
        self._adaptive_pcgrad_ce_sum.add_(detached_loss)
        self._adaptive_pcgrad_ce_count.add_(1.0)

        engine.backward(text_loss)
        # The full VLM graph is gone here. Only the plain projector is replayed.
        try:
            self._adaptive_pcgrad_replay.add(
                captured,
                loss_scale=1.0,
            )
        finally:
            # Do not keep detached vision-feature batches alive while the
            # boundary collectives, surgery, clipping, and optimizer step run.
            # After add() returns, replay owns only detached gradient sums.
            self._set_model_attr(model, "_adaptive_projector_replay_features", [])
            del captured

        if not sync_gradients:
            # DeepSpeed requires ``engine.step()`` after every backward call,
            # including non-boundary microbatches. At this point it only
            # advances its accumulation lifecycle; the optimizer/scheduler do
            # not update until the boundary explicitly set above.
            engine.step()
            if bool(engine.was_step_applied()):
                raise RuntimeError(
                    "DeepSpeed applied an optimizer update before the adaptive "
                    "PCGrad accumulation boundary."
                )
            return text_loss.detach() / self.args.gradient_accumulation_steps

        flat_auxiliary, global_stats, process_group = self._adaptive_global_auxiliary(zero_optimizer)
        main_parts, auxiliary_parts, destinations, shard_process_group = (
            self._adaptive_zero2_projector_parts(zero_optimizer, flat_auxiliary)
        )
        self._adaptive_pcgrad_replay.recycle_flat_buffer(flat_auxiliary)
        auxiliary_group_ranks = tuple(torch.distributed.get_process_group_ranks(process_group))
        shard_group_ranks = tuple(torch.distributed.get_process_group_ranks(shard_process_group))
        if auxiliary_group_ranks != shard_group_ranks:
            raise RuntimeError("Auxiliary and ZeRO projector reductions use different process groups.")
        # Non-MoE ZeRO-2 uses the same DP membership for both handles. Use the
        # shard handle for controller statistics because those coordinates are
        # disjoint according to that exact group.
        process_group = shard_process_group
        valid_count = float(global_stats["valid_count"].item())
        merged, controller_logs = controller.prepare(
            main_parts,
            auxiliary_parts,
            valid_count=valid_count,
            reference_tensor=text_loss,
            distributed_shards=True,
            process_group=process_group,
        )
        with torch.no_grad():
            for destination, value in zip(destinations, merged):
                if value is None or value is destination:
                    continue
                destination.copy_(value.to(dtype=destination.dtype))
        # Keep only DeepSpeed's live gradient shards across engine.step().
        # The merged proposal is no longer needed after the copy-back.
        del merged, value, destination

        try:
            engine.step()
        except Exception:
            controller.finish(optimizer_stepped=False)
            raise
        optimizer_stepped = bool(engine.was_step_applied())
        controller.finish(optimizer_stepped=optimizer_stepped)

        stage_prefix = f"s{controller.config.stage}/"
        cka_count = float(global_stats["valid_count"].item())
        cka_sum = float(global_stats["cka_sum"].item())
        ce_count = float(global_stats["ce_count"].item())
        ce_sum = float(global_stats["ce_sum"].item())
        invalid_total = sum(
            float(value.item())
            for key, value in global_stats.items()
            if key.startswith("invalid/") and key != "invalid/score_roundoff_clamped"
        )
        observation_total = cka_count + invalid_total
        invalid_fraction = invalid_total / observation_total if observation_total > 0.0 else 0.0
        logs = {
            f"{stage_prefix}ce_mean": ce_sum / ce_count if ce_count > 0.0 else 0.0,
            f"{stage_prefix}cka_sum": cka_sum,
            f"{stage_prefix}cka_count": cka_count,
            f"{stage_prefix}cka_mean": cka_sum / cka_count if cka_count > 0.0 else 0.0,
            f"{stage_prefix}cka_ce_only_no_valid": float(cka_count <= 0.0),
            f"{stage_prefix}cka_invalid_fraction": invalid_fraction,
            f"{stage_prefix}optimizer_stepped": float(optimizer_stepped),
            f"{stage_prefix}overflow_skips": float(controller.overflow_skips),
            f"{stage_prefix}ema_ce_norm": float(controller.ema_g or 0.0),
            f"{stage_prefix}ema_cka_norm": float(controller.ema_b or 0.0),
            f"{stage_prefix}lambda_ceiling_frequency": (
                float(controller.lambda_ceiling_hits) / float(controller.valid_proposals)
                if controller.valid_proposals > 0 else 0.0
            ),
            f"{stage_prefix}conflict_rate": (
                float(controller.conflict_proposals) / float(controller.valid_proposals)
                if controller.valid_proposals > 0 else 0.0
            ),
            f"{stage_prefix}near_zero_skip_frequency": (
                float(controller.near_zero_skips) / float(max(controller.successful_steps, 1))
            ),
        }
        logs.update({f"{stage_prefix}{key}": value for key, value in controller_logs.items()})
        for key, value in global_stats.items():
            if key.startswith("invalid/"):
                logs[f"{stage_prefix}cka_{key}"] = float(value.item())
        if (
            invalid_fraction >= 0.10
            and not getattr(self, "_adaptive_pcgrad_invalid_warning_emitted", False)
            and self.is_world_process_zero()
        ):
            logger.warning(
                "Adaptive projector CKA skipped %.1f%% of image observations in an effective "
                "batch (zero Gram norm/too few patches). Inspect s%d/cka_invalid_* logs.",
                100.0 * invalid_fraction,
                controller.config.stage,
            )
            self._adaptive_pcgrad_invalid_warning_emitted = True
        if (
            controller.valid_proposals >= 100
            and controller.lambda_ceiling_hits * 2 >= controller.valid_proposals
            and not getattr(self, "_adaptive_pcgrad_ceiling_warning_emitted", False)
            and self.is_world_process_zero()
        ):
            logger.warning(
                "Adaptive projector PCGrad lambda has reached lambda_max in at least half "
                "of valid proposals; the ceiling remains enforced. Inspect s%d/lambda and "
                "s%d/effective_aux_ratio before tuning.",
                controller.config.stage,
                controller.config.stage,
            )
            self._adaptive_pcgrad_ceiling_warning_emitted = True
        global_norm = float(getattr(zero_optimizer, "_global_grad_norm", 0.0) or 0.0)
        max_norm = float(getattr(self.args, "max_grad_norm", 0.0) or 0.0)
        logs[f"{stage_prefix}preclip_norm"] = global_norm
        logs[f"{stage_prefix}clip_factor"] = (
            1.0 / max(1.0, (global_norm + 1e-6) / max_norm) if max_norm > 0.0 else 1.0
        )
        try:
            logs[f"{stage_prefix}learning_rate"] = float(self._get_learning_rate())
        except Exception:
            pass
        if profile and self._adaptive_pcgrad_window_start is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                logs[f"{stage_prefix}peak_allocated_mb"] = float(
                    torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                )
            logs[f"{stage_prefix}optimizer_step_wall_ms"] = float(
                (time.perf_counter() - self._adaptive_pcgrad_window_start) * 1000.0
            )
            self._adaptive_pcgrad_window_start = None
        self._last_adaptive_pcgrad_logs = logs
        return text_loss.detach() / self.args.gradient_accumulation_steps

    @contextmanager
    def _cka_loss_runtime_context(self, model):
        """Enable CKA only after the configured fraction of optimizer steps.

        The configured flag is temporarily changed for the complete
        forward/backward operation so inactive steps skip CKA hooks and kernels,
        including any gradient-checkpoint recomputation. It is restored before
        checkpoint serialization or the next Trainer operation.
        """
        config = self.model.config
        configured = bool(getattr(config, 'cka_loss', False))
        start_ratio = validate_cka_loss_start_ratio(
            getattr(config, 'cka_loss_start_ratio', 0.0)
        )
        global_step = int(getattr(self.state, 'global_step', 0) or 0)
        max_steps = int(getattr(self.state, 'max_steps', 0) or 0)
        active, start_step = get_cka_loss_schedule_state(
            configured,
            start_ratio,
            global_step,
            max_steps,
        )

        self._last_cka_schedule_active = active
        self._last_cka_schedule_start_step = start_step
        self._last_cka_schedule_start_ratio = start_ratio

        if not active:
            for attr_name, value in (
                ('last_cka_loss', None),
                ('last_cka_projector_loss', None),
                ('last_cka_pre_post_loss', None),
                ('last_cka_pre_final_loss', None),
                ('last_cka_layers_loss', None),
                ('last_cka_per_layer_losses', {}),
                ('last_cka_subset_vision_feature_mask', None),
                ('last_cka_final_hidden', None),
                ('last_cka_projector_output', None),
                ('_aux_losses', []),
            ):
                self._set_model_attr(model, attr_name, value)

        config.cka_loss = active
        try:
            yield active
        finally:
            config.cka_loss = configured

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Keep the existing per-microbatch text/CKA means. The new Trainer's
        # token-count argument must not change their relative weighting.
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

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        if adaptive_projector_pcgrad_enabled(self.model.config):
            if is_sagemaker_mp_enabled():
                raise RuntimeError("Adaptive projector PCGrad does not support SageMaker model parallelism.")
            return self._adaptive_training_step(
                model,
                inputs,
                num_items_in_batch=num_items_in_batch,
            )

        with self._cka_loss_runtime_context(model):
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            return self._training_step_with_cka_runtime_state(
                model,
                inputs,
                num_items_in_batch=num_items_in_batch,
            )

    def _training_step_with_cka_runtime_state(self, model, inputs, num_items_in_batch=None):

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
                    if not self.is_deepspeed_enabled:
                        # Older Trainer versions configure accumulation in
                        # Accelerate; 4.51 leaves its factor at 1. Compensate so
                        # backward divides by the configured factor exactly once.
                        backward_loss = backward_loss * (
                            self.accelerator.gradient_accumulation_steps
                            / self.args.gradient_accumulation_steps
                        )
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

    def log(self, logs, *args, **kwargs):
        logs = dict(logs)
        model = self.model.module if hasattr(self.model, 'module') else self.model

        if hasattr(self, '_last_cka_schedule_active'):
            logs['cka/schedule_active'] = float(self._last_cka_schedule_active)
            logs['cka/schedule_start_step'] = float(self._last_cka_schedule_start_step)
            logs['cka/schedule_start_ratio'] = float(self._last_cka_schedule_start_ratio)

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

        adaptive_logs = getattr(self, '_last_adaptive_pcgrad_logs', None)
        if adaptive_logs:
            logs.update(adaptive_logs)
            self._last_adaptive_pcgrad_logs = None

        return super().log(logs, *args, **kwargs)

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
            adaptive_pcgrad = adaptive_projector_pcgrad_enabled(
                getattr(opt_model, 'config', self.model.config)
            )
            split_projector_groups = (
                self.args.mm_projector_lr is not None
                or vsp_controller_requested(getattr(opt_model, 'config', self.model.config))
                or adaptive_pcgrad
            )
            if split_projector_groups:
                if adaptive_pcgrad:
                    projector = resolve_mm_projector_module(opt_model)
                    exact_projector_ids = {
                        id(parameter) for parameter in projector.parameters() if parameter.requires_grad
                    }
                    projector_parameters = [
                        name for name, parameter in opt_model.named_parameters()
                        if id(parameter) in exact_projector_ids
                    ]
                else:
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
                if adaptive_pcgrad:
                    # Empty LLM groups are common in stage 1 and are rejected by
                    # ZeRO-2 while it inspects the first parameter dtype.
                    optimizer_grouped_parameters = [
                        group for group in optimizer_grouped_parameters if group["params"]
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

    def _save_adaptive_pcgrad_artifacts(self, output_dir):
        if not adaptive_projector_pcgrad_enabled(self.model.config):
            return
        if not self.is_world_process_zero():
            return
        controller = self._get_adaptive_pcgrad_controller()
        os.makedirs(output_dir, exist_ok=True)
        torch.save(
            controller.state_dict(),
            os.path.join(output_dir, ADAPTIVE_PCGRAD_STATE_FILE),
        )
        resolved = controller.config.to_dict()
        write_json(os.path.join(output_dir, ADAPTIVE_PCGRAD_CONFIG_FILE), resolved)
        base_identifier = (
            getattr(self.model.config, "adaptive_pcgrad_base_model_identifier", None)
            or getattr(self.model.config, "_name_or_path", None)
            or "unknown"
        )
        metadata = {
            "version": 1,
            "stage": int(controller.config.stage),
            "base_model_identifier": str(base_identifier),
            "projector_sha256": projector_sha256(self._adaptive_pcgrad_projector),
            "projector_signature": projector_signature(self._adaptive_pcgrad_projector),
            "resolved_config": resolved,
            "successful_steps": int(controller.successful_steps),
            "parent_projector_sha256": getattr(
                self.model.config,
                "adaptive_pcgrad_parent_projector_sha256",
                getattr(self.model.config, "adaptive_pcgrad_stage1_projector_sha256", None),
            ),
            "backend": {
                "name": "deepspeed_zero2",
                "offload": False,
                "precision": getattr(
                    self,
                    "_adaptive_pcgrad_precision",
                    "bf16" if getattr(self.args, "bf16", False) else "fp32",
                ),
                "deepspeed": __import__("deepspeed").__version__,
                "torch": str(torch.__version__),
            },
            "ce_reduction": (
                "repository baseline: accumulated per-microbatch loss means; "
                "projector CKA: global valid-image SUM/count"
            ),
        }
        write_json(os.path.join(output_dir, ADAPTIVE_PCGRAD_METADATA_FILE), metadata)

    def _save_adaptive_stage1_adapter(self, output_dir):
        keys_to_match = ['mm_projector', 'vision_resampler']
        if getattr(self.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])
        weight_to_save = get_mm_adapter_state_maybe_zero_3(
            self.model.named_parameters(), keys_to_match
        )
        if self.is_world_process_zero():
            os.makedirs(output_dir, exist_ok=True)
            self.model.config.save_pretrained(output_dir)
            torch.save(weight_to_save, os.path.join(output_dir, 'mm_projector.bin'))

    def _load_adaptive_pcgrad_resume_artifacts(self, resume_from_checkpoint):
        """Validate provenance and restore stage-local controller state.

        DeepSpeed model/optimizer state is loaded by Transformers without
        calling ``_load_from_checkpoint``. This helper therefore runs from
        ``_load_optimizer_and_scheduler`` as well, after the live projector
        weights have been restored by DeepSpeed.
        """
        checkpoint_dir = os.path.realpath(os.fspath(resume_from_checkpoint))
        if self._adaptive_pcgrad_resume_checkpoint == checkpoint_dir:
            return
        metadata_path = os.path.join(checkpoint_dir, ADAPTIVE_PCGRAD_METADATA_FILE)
        state_path = os.path.join(checkpoint_dir, ADAPTIVE_PCGRAD_STATE_FILE)
        if not os.path.isfile(metadata_path) or not os.path.isfile(state_path):
            raise RuntimeError(
                "Adaptive projector PCGrad resume requires its metadata and controller state; "
                f"missing files in {checkpoint_dir}."
            )
        metadata = load_json(metadata_path)
        expected_stage = int(
            getattr(self.model.config, "adaptive_projector_pcgrad_config", {}).get("stage", -1)
        )
        if int(metadata.get("stage", -1)) != expected_stage:
            raise RuntimeError("Refusing to resume adaptive PCGrad from another stage.")
        expected_base = str(
            getattr(self.model.config, "adaptive_pcgrad_base_model_identifier", None)
            or getattr(self.model.config, "_name_or_path", None)
            or "unknown"
        )
        if metadata.get("base_model_identifier") != expected_base:
            raise RuntimeError(
                "Adaptive PCGrad checkpoint base model mismatch: "
                f"{metadata.get('base_model_identifier')!r} != {expected_base!r}."
            )
        if metadata.get("projector_signature") != projector_signature(
            self._adaptive_pcgrad_projector
        ):
            raise RuntimeError("Adaptive PCGrad checkpoint projector signature mismatch.")
        if expected_stage == 2:
            expected_parent = getattr(
                self.model.config, "adaptive_pcgrad_stage1_projector_sha256", None
            )
            if not expected_parent or metadata.get("parent_projector_sha256") != expected_parent:
                raise RuntimeError(
                    "Adaptive PCGrad stage-2 checkpoint was created from a different "
                    "stage-1 projector parent."
                )
        expected_hash = metadata.get("projector_sha256")
        actual_hash = projector_sha256(self._adaptive_pcgrad_projector)
        if expected_hash != actual_hash:
            raise RuntimeError(
                "Adaptive PCGrad checkpoint projector hash mismatch after loading: "
                f"expected {expected_hash}, got {actual_hash}."
            )
        try:
            resume_state = torch.load(state_path, map_location="cpu", weights_only=True)
        except TypeError:
            resume_state = torch.load(state_path, map_location="cpu")
        if metadata.get("resolved_config") != resume_state.get("config"):
            raise RuntimeError(
                "Adaptive PCGrad checkpoint metadata/controller config mismatch."
            )
        if int(metadata.get("successful_steps", -1)) != int(
            resume_state.get("successful_steps", -2)
        ):
            raise RuntimeError(
                "Adaptive PCGrad checkpoint metadata/controller step-count mismatch."
            )

        # During DeepSpeed resume, state.max_steps has already been resolved for
        # the current run. Loading now therefore also rejects a changed planned
        # horizon. The pending path covers non-DeepSpeed's earlier model hook.
        if int(getattr(self.state, "max_steps", 0) or 0) > 0:
            controller = self._get_adaptive_pcgrad_controller()
            controller.load_state_dict(resume_state)
        else:
            self._adaptive_pcgrad_resume_state = resume_state
        self._adaptive_pcgrad_resume_checkpoint = checkpoint_dir

    def _load_optimizer_and_scheduler(self, checkpoint):
        super()._load_optimizer_and_scheduler(checkpoint)
        if checkpoint is not None and adaptive_projector_pcgrad_enabled(self.model.config):
            self._load_adaptive_pcgrad_resume_artifacts(checkpoint)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        if not adaptive_projector_pcgrad_enabled(self.model.config):
            return super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        checkpoint_dir = os.fspath(resume_from_checkpoint)
        expected_stage = int(
            getattr(self.model.config, "adaptive_projector_pcgrad_config", {}).get("stage", -1)
        )

        if expected_stage == 1 and getattr(self.args, 'tune_mm_mlp_adapter', False):
            adapter_path = os.path.join(checkpoint_dir, 'mm_projector.bin')
            if not os.path.isfile(adapter_path):
                raise RuntimeError(f"Stage-1 resume is missing {adapter_path}.")
            try:
                adapter_state = torch.load(adapter_path, map_location="cpu", weights_only=True)
            except TypeError:
                adapter_state = torch.load(adapter_path, map_location="cpu")
            projector_state = {}
            for key, value in adapter_state.items():
                marker = "mm_projector."
                if marker in key:
                    projector_state[key.split(marker, 1)[1]] = value
            if not projector_state:
                raise RuntimeError("Stage-1 checkpoint contains no mm_projector parameters.")
            self._adaptive_pcgrad_projector.load_state_dict(projector_state, strict=True)
        else:
            super()._load_from_checkpoint(checkpoint_dir, model=model)
        self._load_adaptive_pcgrad_resume_artifacts(checkpoint_dir)

    def _save_checkpoint(self, model, trial, *args, **kwargs):
        if adaptive_projector_pcgrad_enabled(self.model.config):
            super(LLaVATrainer, self)._save_checkpoint(model, trial, *args, **kwargs)
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            output_dir = os.path.join(self._get_output_dir(trial=trial), checkpoint_folder)
            if getattr(self.args, 'tune_mm_mlp_adapter', False):
                self._save_adaptive_stage1_adapter(output_dir)
            self._save_adaptive_pcgrad_artifacts(output_dir)
            return
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
            super(LLaVATrainer, self)._save_checkpoint(model, trial, *args, **kwargs)

    def save_state(self):
        super().save_state()
        if adaptive_projector_pcgrad_enabled(self.model.config):
            self._save_adaptive_pcgrad_artifacts(self.args.output_dir)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            sanitize_generation_config_for_save(self.model)
            super(LLaVATrainer, self)._save(output_dir, state_dict)
