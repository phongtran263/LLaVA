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
from typing import List, Optional


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
        ):
            return
        self._set_model_attr(model, 'last_cka_final_hidden', None)
        self._set_model_attr(model, 'last_cka_projector_output', None)
        self._set_model_attr(model, 'last_cka_projector_cka_output', None)
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

    def _build_pretrain_projector_pcgrad_loss(
        self,
        model,
        text_loss,
        projector_cka_loss,
        aux_losses,
    ):
        objective = text_loss + self._get_cka_auxiliary_loss(
            text_loss,
            projector_cka_loss,
            aux_losses,
        )
        if projector_cka_loss is None or not projector_cka_loss.requires_grad:
            return objective, objective
        if not getattr(self.args, 'tune_mm_mlp_adapter', False):
            raise RuntimeError(
                'pretrain_projector_pcgrad requires --tune_mm_mlp_adapter True.'
            )

        active_other_aux = [
            aux_loss
            for aux_loss in (aux_losses or [])
            if torch.is_tensor(aux_loss) and aux_loss.requires_grad
        ]
        if active_other_aux:
            raise RuntimeError(
                'pretrain_projector_pcgrad supports projector CKA only; '
                'set --cka_loss_layers -1.'
            )

        trainable = [
            (name, param)
            for name, param in model.named_parameters()
            if param.requires_grad
        ]
        invalid_names = [name for name, _ in trainable if 'mm_projector' not in name]
        if invalid_names:
            preview = ', '.join(invalid_names[:3])
            raise RuntimeError(
                'pretrain_projector_pcgrad requires projector-only training; '
                f'found other trainable parameters: {preview}'
            )
        params = [param for _, param in trainable]
        if not params:
            raise RuntimeError('pretrain_projector_pcgrad found no trainable projector parameters.')

        # Extract each task gradient once. autograd.grad does not populate .grad,
        # so DeepSpeed reduction is deferred to the tiny surrogate backward below.
        main_grads = torch.autograd.grad(
            text_loss,
            params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        aux_grads = torch.autograd.grad(
            projector_cka_loss,
            params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        shared = [
            (main_grad, aux_grad)
            for main_grad, aux_grad in zip(main_grads, aux_grads)
            if main_grad is not None and aux_grad is not None
        ]
        zero = text_loss.new_zeros((), dtype=torch.float32)
        dot = sum(
            (main_grad.detach().float() * aux_grad.detach().float()).sum()
            for main_grad, aux_grad in shared
        ) if shared else zero
        main_norm_sq = sum(
            main_grad.detach().float().square().sum()
            for main_grad, _ in shared
        ) if shared else zero
        coefficient = torch.where(
            dot < 0,
            dot / main_norm_sq.clamp_min(1e-12),
            dot.new_zeros(()),
        )

        injection = text_loss.new_zeros(())
        for param, main_grad, aux_grad in zip(params, main_grads, aux_grads):
            final_grad = None
            if main_grad is not None:
                final_grad = main_grad.detach().float()
            if aux_grad is not None:
                safe_aux_grad = aux_grad.detach().float()
                if main_grad is not None:
                    safe_aux_grad = safe_aux_grad - coefficient * main_grad.detach().float()
                final_grad = safe_aux_grad if final_grad is None else final_grad + safe_aux_grad
            if final_grad is not None:
                injection = injection + (
                    (param - param.detach()) * final_grad.to(param.dtype)
                ).sum()

        # Forward value equals the real objective, while its gradient is the
        # already-composed PCGrad vector. DeepSpeed still owns scaling, gradient
        # accumulation, synchronization, and optimizer-visible .grad buffers.
        backward_loss = objective.detach() + injection
        return objective, backward_loss

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

        projector_task_output = self._find_model_attr(
            unwrapped_model,
            'last_cka_projector_output',
        )
        projector_cka_output = self._find_model_attr(
            unwrapped_model,
            'last_cka_projector_cka_output',
        )
        if not torch.is_tensor(projector_cka_output):
            projector_cka_output = projector_task_output

        def connected_unique_tensors(*tensors):
            connected = []
            seen = set()
            for tensor in tensors:
                if (
                    torch.is_tensor(tensor)
                    and tensor.requires_grad
                    and id(tensor) not in seen
                ):
                    connected.append(tensor)
                    seen.add(id(tensor))
            return connected

        projector_task_output_tensors = connected_unique_tensors(
            projector_task_output
        )
        projector_cka_output_tensors = connected_unique_tensors(
            projector_cka_output
        )
        combined_projector_output_tensors = connected_unique_tensors(
            projector_task_output,
            projector_cka_output,
        )
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
        projector_output_losses = (
            ('text_loss', text_loss, projector_task_output_tensors),
            ('cka_loss', cka_loss, combined_projector_output_tensors),
            (
                'projector_cka_loss',
                projector_cka_loss,
                projector_cka_output_tensors,
            ),
            (
                'final_hidden_cka_loss',
                final_hidden_cka_loss,
                projector_task_output_tensors,
            ),
        )
        for loss_name, loss_value, target_tensors in projector_output_losses:
            norm = self._gradient_norm(
                loss_value,
                target_tensors,
                loss_name,
                'projector_output',
            )
            if norm is not None:
                logs[f'grad_norm/{loss_name}/projector_output'] = norm

        final_hidden_losses = (
            ('text_loss', text_loss),
            ('cka_loss', cka_loss),
            ('final_hidden_cka_loss', final_hidden_cka_loss),
        )
        target_specs = (
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

        if self.model.config.cka_loss:
            cka_auxiliary_loss = self._get_cka_auxiliary_loss(
                text_loss,
                projector_cka_loss,
                aux_losses,
            )
            loss = text_loss + cka_auxiliary_loss
            backward_loss = loss
            if getattr(self.args, 'pretrain_projector_pcgrad', False):
                loss, backward_loss = self._build_pretrain_projector_pcgrad_loss(
                    model,
                    text_loss,
                    projector_cka_loss,
                    aux_losses,
                )
        elif getattr(self.args, 'pretrain_projector_pcgrad', False):
            raise RuntimeError('pretrain_projector_pcgrad requires --cka_loss True.')

        try:
            if self.use_apex:
                with amp.scale_loss(backward_loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(backward_loss)
        finally:
            self._clear_gradient_log_tensors(model)

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
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                projector_lr_kwargs = {"lr": self.args.mm_projector_lr} if self.args.mm_projector_lr is not None else {}
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        **projector_lr_kwargs,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
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
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
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
