"""Opt-in instruction refinement with a fixed, independent Stage-2 teacher.

Trainer still owns the single optimizer/clipping/scheduler step. This runtime
only prepares its accumulated gradients and owns validation/transactional saves.
Single-process FP32/BF16 is intentional: unsupported sharded/scaled backends must
fail, never silently use local gradients or an incompletely frozen teacher.
"""

import copy
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import Trainer, TrainerCallback, TrainerState
from transformers.trainer_utils import TrainOutput, get_last_checkpoint

from llava.constants import IGNORE_INDEX
from llava.train.llava_trainer import LengthGroupedSampler
from llava.train.protected_cka_math import token_kl
from llava.train.protected_cka_config import _persist_or_validate
from llava.train.protected_cka_refinement import PlateauMonitor, preserve_rng, _load, _atomic_save
from llava.train.protected_cka_stage2_forward import (
    snapshot_stage2_reference, stage2_deterministic_context, stage2_forward,
)
from llava.train.protected_cka_stage2_math import group_gradient_norms, protected_group_gradient

STATE_NAME = "protected_cka_stage2_state.pt"
STUDENT_NAME = "student_model.pt"
REFERENCE_NAME = "reference_model.pt"
SPLIT_DIRECTORY = "protected_cka_stage2_split"
CHECKPOINT_MARKER = "protected_cka_stage2_checkpoint.json"


class ProtectedModalitySampler(LengthGroupedSampler):
    """The existing grouping algorithm, using a replayable per-epoch RNG."""

    def __init__(self, *args, seed, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed, self.epoch = seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        # The legacy mixed-modality helper also uses the default CPU RNG for
        # its two within-modality shuffles. Scope that RNG rather than changing
        # the grouping algorithm or perturbing the model's dropout stream.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator().manual_seed(self.seed + self.epoch).get_state())
            self.generator = None
            indices = list(super().__iter__())
        return iter(indices)


@contextmanager
def _evaluation_mode(model):
    flags = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, training in flags:
            module.training = training


def _cpu_state(model):
    return {name: value.detach().to(device="cpu", copy=True) for name, value in model.state_dict().items()}


def _grad(loss, parameters, *, retain_graph=False):
    if not loss.requires_grad or not parameters:
        return [None] * len(parameters)
    return list(torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True))


def _accumulate(previous, current):
    if previous is None:
        return [None if value is None else value.detach().to(dtype=torch.float32, copy=True) for value in current]
    for index, value in enumerate(current):
        if value is not None:
            if previous[index] is None:
                previous[index] = value.detach().to(dtype=torch.float32, copy=True)
            else:
                previous[index].add_(value.detach())
    return previous


class _Stage2Callback(TrainerCallback):
    def __init__(self, runtime):
        self.runtime = runtime

    def on_train_begin(self, args, state, control, **kwargs):
        self.runtime.begin()

    def on_step_end(self, args, state, control, **kwargs):
        self.runtime.step_end(control)


class ProtectedCKAStage2:
    def __init__(self, trainer):
        self.trainer, self.args, self.model = trainer, trainer.args, trainer.model
        if trainer.accelerator.num_processes != 1 or trainer.accelerator.scaler is not None:
            raise ValueError("Protected CKA Stage 2 supports one process, FP32/BF16, without GradScaler only.")
        if self.args.deepspeed or self.args.fsdp:
            raise ValueError("Protected CKA Stage 2 does not yet support DeepSpeed/FSDP reduction or snapshots.")
        if getattr(self.args, "tune_mm_mlp_adapter", False):
            raise ValueError("Protected CKA Stage 2 must keep the Stage-2 full-FT/LoRA recipe, not Stage-1 freezing.")
        if trainer.eval_dataset is None or not len(trainer.eval_dataset):
            raise ValueError("Protected CKA Stage 2 requires held-out instruction validation data.")
        self.named_params = [(name, p) for name, p in self.model.named_parameters() if p.requires_grad]
        self.params = [p for _, p in self.named_params]
        self.group_names = ["projector" if "mm_projector." in name else "llm" for name, _ in self.named_params]
        if not self.params or "llm" not in self.group_names:
            raise ValueError("Protected Stage 2 requires trainable student LLM/adapters; refusing a Stage-1 recipe.")
        tower = self.model.get_vision_tower()
        if tower is not None and any(p.requires_grad for p in tower.parameters()):
            raise ValueError("Protected Stage 2 currently requires the recipe's vision tower to remain frozen.")
        self.allowed = [self.args.cka_grad_scope == "existing" or group == "projector"
                        for group in self.group_names]
        if not any(self.allowed):
            raise ValueError("No trainable parameters in the requested cka_grad_scope.")
        self.allowed_indices = [i for i, allowed in enumerate(self.allowed) if allowed]
        self.allowed_params = [self.params[i] for i in self.allowed_indices]
        self.phase, self.reference, self.phase_b_start = "A", None, None
        self.transition_reason = None
        self.refinement_accepted, self.source_provenance = False, None
        self.ce_ref, self.ce_grad_refs, self.ema = None, {"projector": 0., "llm": 0.}, None
        self.aux_ratios = {"projector": self.option("proj_aux_ratio"), "llm": self.option("llm_aux_ratio")}
        self.retries, self.pending_rollback, self.loaded = 0, False, False
        self.accepted_checkpoint = self.finished_reason = self.reference_file = None
        self.monitor = PlateauMonitor(self.option("plateau_window"), self.option("plateau_patience"),
                                      self.option("plateau_min_delta"))
        self.accum_main = self.accum_cka = None
        self.accum_count, self.step_logs = 0, {}
        self.seen_cka_support = set()
        self.signature = self._signature()
        self.source_initialized = False
        with preserve_rng():
            batches = len(trainer.get_train_dataloader())
        if not batches or batches % self.args.gradient_accumulation_steps:
            raise ValueError(f"Protected Stage 2 exact replay requires complete accumulation windows: {batches} "
                             f"microbatches vs accumulation={self.args.gradient_accumulation_steps}. "
                             "Use accumulation=1 or adjust batch/split sizes.")
        self._write_scope_manifest()
        trainer.add_callback(_Stage2Callback(self))

    def option(self, name):
        return getattr(self.args, "protected_stage2_" + name)

    @property
    def use_cka(self):
        return (self.phase == "B" and self.option("continuation") == "protected"
                and bool(getattr(self.model.config, "cka_loss", False))
                and self.option("lambda_max") > 0)

    @property
    def use_kl(self):
        return self.phase == "B" and self.option("continuation") != "ce_only" and self.option("kl_weight") > 0

    @property
    def phase_b_horizon(self):
        return min(self.option("max_phase_b_steps"), self.args.max_steps - self.phase_b_start)

    def _signature(self):
        config = {k: v for k, v in vars(self.args).items() if k.startswith("protected_stage2_")}
        # Old unpickled TrainingArguments may lack this instance attribute but
        # still read the new dataclass class default. Sign the effective value,
        # never mistake omission for the old plateau-only (zero) policy.
        config["protected_stage2_min_phase_b_steps"] = self.option("min_phase_b_steps")
        for name in ("start_phase_b", "source_stage2_checkpoint", "split_manifest"):
            config.pop("protected_stage2_" + name, None)
        for name in ("cka_grad_scope", "max_steps", "seed", "data_seed", "per_device_train_batch_size",
                     "gradient_accumulation_steps", "learning_rate", "warmup_steps", "warmup_ratio",
                     "weight_decay", "adam_beta1", "adam_beta2", "adam_epsilon", "max_grad_norm", "bf16",
                     "lr_scheduler_type", "lr_scheduler_kwargs", "optim", "dataloader_drop_last",
                     "per_device_eval_batch_size", "mm_projector_lr", "group_by_modality_length",
                     "gradient_checkpointing", "gradient_checkpointing_kwargs", "lora_enable",
                     "lora_r", "lora_alpha", "lora_dropout", "lora_bias"):
            config[name] = getattr(self.args, name, None)
        data = {}
        for name in ("train_dataset", "eval_dataset"):
            dataset = getattr(self.trainer, name)
            rows = getattr(dataset, "list_data_dict", {"class": type(dataset).__qualname__, "length": len(dataset)})
            data[name] = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
        model_config = self.model.config.to_dict()
        data["model"] = {k: v for k, v in model_config.items() if k.startswith("cka_loss") or k in (
            "_name_or_path", "model_type", "hidden_size", "num_hidden_layers", "vocab_size", "mm_vision_tower",
            "mm_vision_select_layer", "mm_projector_type", "mm_patch_merge_type", "tokenizer_model_max_length",
            "tokenizer_padding_side", "llava_conversation_version")}
        data["trainable"] = [(name, list(p.shape), str(p.dtype)) for name, p in self.named_params]
        data["config"] = config
        return json.loads(json.dumps(data, default=str))

    def _write_scope_manifest(self):
        root = Path(self.args.output_dir)
        root.mkdir(parents=True, exist_ok=True)
        scope = {"cka_grad_scope": self.args.cka_grad_scope, "groups": {}}
        for group in ("projector", "llm"):
            selected = [(name, p) for (name, p), g, allowed in zip(self.named_params, self.group_names, self.allowed)
                        if g == group and allowed]
            scope["groups"][group] = {"names": [name for name, _ in selected],
                                     "parameters": sum(p.numel() for _, p in selected)}
        # This is eligibility, not a claim that every parameter is in each CKA
        # graph. Actual support is recorded below and None gradients stay None.
        _persist_or_validate(root / "protected_cka_stage2_scope.json", scope)
        for group, info in scope["groups"].items():
            print(f"[Protected Stage 2] CKA scope={self.args.cka_grad_scope}, {group}: "
                  f"{len(info['names'])} eligible tensors / {info['parameters']:,} parameters; "
                  f"names in {root / 'protected_cka_stage2_scope.json'}", flush=True)

    def _record_support(self, gradients):
        names = {name for (name, _), grad in zip(self.named_params, gradients) if grad is not None}
        if not names.issubset(self.seen_cka_support):
            self.seen_cka_support.update(names)
            path = Path(self.args.output_dir) / "protected_cka_stage2_active_parameters.json"
            with path.open("w") as handle:
                json.dump(sorted(self.seen_cka_support), handle, indent=2)

    def _aux_grad(self, loss, *, retain_graph=True):
        gradients = [None] * len(self.params)
        if self.use_cka:
            for index, value in zip(self.allowed_indices, _grad(loss, self.allowed_params, retain_graph=retain_graph)):
                gradients[index] = value
        return gradients

    def training_step(self, model, inputs):
        model.train()
        inputs = self.trainer._prepare_inputs(inputs)
        log_timing = (self.trainer.state.global_step + 1) % max(1, self.args.logging_steps) == 0
        if log_timing and self.accum_count == 0:
            if self.params[0].is_cuda:
                torch.cuda.synchronize(self.params[0].device)
            self.step_started = time.perf_counter()
        context = stage2_deterministic_context(model) if self.phase == "B" else nullcontext()
        with context:
            with self.trainer.compute_loss_context_manager():
                output = stage2_forward(model, inputs, self.reference if self.phase == "B" else None,
                                        need_cka=self.use_cka, need_reference_logits=self.use_kl,
                                        margin=self.option("margin"), eps=self.option("eps"))
                ce = output["student_ce"]
                kl = (token_kl(output["student_logits"], output["reference_logits"], output["expanded_labels"],
                               self.option("kl_chunk_size"))[0] if self.use_kl else ce.new_zeros(()))
                main = ce + self.option("kl_weight") * kl
                cka = output["cka_loss"]
            if not bool(torch.isfinite(main)) or not bool(torch.isfinite(cka)):
                raise RuntimeError("Nonfinite Protected Stage-2 objective; refusing update.")
            auxiliary = self._aux_grad(cka)
            main_grads = _grad(main, self.params)
        self._record_support(auxiliary)
        self.accum_main = _accumulate(self.accum_main, main_grads)
        self.accum_cka = _accumulate(self.accum_cka, auxiliary)
        del main_grads, auxiliary
        self.accum_count += 1
        logs = {"ce": float(ce.detach()), "text_kl": float(kl.detach()), "cka_loss": float(cka.detach()),
                "valid_tokens": int((output["expanded_labels"][:, 1:] != IGNORE_INDEX).sum()),
                **output.get("cka_metrics", {})}
        for key, value in logs.items():
            if not key.endswith("_sum"):
                self.step_logs[key] = self.step_logs.get(key, 0.) + float(value)
        reported_loss = main.detach() / self.args.gradient_accumulation_steps
        # Release logits/captured features before allocating protected gradients.
        del output, main, ce, kl, cka
        if self.trainer.accelerator.sync_gradients:
            count = self.accum_count
            for gradients in (self.accum_main, self.accum_cka):
                for gradient in gradients:
                    if gradient is not None:
                        gradient.div_(count)
            gradient_logs = {}
            if self.use_cka:
                elapsed = self.trainer.state.global_step - self.phase_b_start
                warmup = self.option("warmup_steps")
                weight = self.option("lambda_max") * (min(1., elapsed / warmup) if warmup else 1.)
                # Do not allocate a model-sized EMA for parameters that never
                # participate in CKA (e.g. lm_head or blocks after its target).
                ema_scope = [allowed and (gradient is not None or (self.ema is not None and self.ema[i] is not None))
                             for i, (allowed, gradient) in enumerate(zip(self.allowed, self.accum_cka))]
                final, self.ema, grouped = protected_group_gradient(
                    self.accum_main, self.accum_cka, self.ema, group_names=self.group_names,
                    ce_grad_refs=self.ce_grad_refs, aux_ratios=self.aux_ratios, allowed_cka_mask=ema_scope,
                    ema_decay=self.option("ema_decay"), lambda_t=weight, eps=self.option("eps"))
                gradient_logs = {f"grad/{group}/{key}": value for group, stats in grouped.items()
                                 for key, value in stats.items()}
                gradient_logs["lambda"] = weight
            else:
                final = self.accum_main
            finite = torch.ones((), dtype=torch.bool, device=self.params[0].device)
            for gradient in final:
                if gradient is not None:
                    for chunk in gradient.reshape(-1).split(1_048_576):
                        finite.logical_and_(torch.isfinite(chunk).all())
            if not bool(finite):
                raise RuntimeError("Nonfinite Protected Stage-2 gradient; refusing optimizer update.")
            for parameter, gradient in zip(self.params, final):
                parameter.grad = None if gradient is None else gradient.to(parameter.dtype)
            self.latest_logs = {"protected2/" + key: value if key in (
                "valid_tokens", "valid_images", "invalid_images") else value / count
                for key, value in self.step_logs.items()}
            self.latest_logs.update({"protected2/" + key: value for key, value in gradient_logs.items()})
            self.latest_logs.update({"protected2/phase_b": int(self.phase == "B"),
                                     "protected2/proj_aux_ratio": self.aux_ratios["projector"],
                                     "protected2/llm_aux_ratio": self.aux_ratios["llm"]})
            if log_timing:
                if self.params[0].is_cuda:
                    torch.cuda.synchronize(self.params[0].device)
                self.latest_logs["protected2/time/effective_batch_compute_seconds"] = time.perf_counter() - self.step_started
            self.accum_main = self.accum_cka = None
            self.accum_count, self.step_logs = 0, {}
        return reported_loss

    def _loader(self, dataset, *, training=False):
        return DataLoader(dataset, batch_size=(self.args.per_device_train_batch_size if training
                                              else self.args.per_device_eval_batch_size), shuffle=False,
                          num_workers=0, collate_fn=self.trainer.data_collator,
                          pin_memory=self.args.dataloader_pin_memory)

    def calibrate(self):
        """New Stage-2 CE norms, separately on the actual projector/LLM CKA graph."""
        self.ce_grad_refs = {"projector": 0., "llm": 0.}
        if not self.use_cka:
            return
        norms = []
        accumulated, support, count = None, [False] * len(self.params), 0
        with preserve_rng(), stage2_deterministic_context(self.model):
            for batch in self._loader(self.trainer.train_dataset, training=True):
                with self.trainer.compute_loss_context_manager():
                    output = stage2_forward(self.model, self.trainer._prepare_inputs(batch), self.reference,
                                            need_cka=True, need_reference_logits=False,
                                            margin=self.option("margin"), eps=self.option("eps"))
                aux = self._aux_grad(output["cka_loss"])
                self._record_support(aux)
                support = [old or grad is not None for old, grad in zip(support, aux)]
                accumulated = _accumulate(accumulated, _grad(output["student_ce"], self.params))
                del output, aux
                count += 1
                if count == self.args.gradient_accumulation_steps:
                    if any(support):
                        for gradient in accumulated:
                            if gradient is not None:
                                gradient.div_(count)
                        norms.append(group_gradient_norms(accumulated, self.group_names, allowed_mask=support))
                    accumulated, support, count = None, [False] * len(self.params), 0
                    if len(norms) >= self.option("ce_gradient_calibration_batches"):
                        break
        if norms:
            self.ce_grad_refs = {group: sum(float(item.get(group, 0.)) for item in norms) / len(norms)
                                 for group in self.ce_grad_refs}
        if not all(math.isfinite(value) for value in self.ce_grad_refs.values()):
            raise RuntimeError("Nonfinite Stage-2 calibration norm.")
        self.trainer.log({**{"protected2/ce_grad_ref/" + k: v for k, v in self.ce_grad_refs.items()},
                          "protected2/calibration_effective_batches": len(norms)})

    @torch.no_grad()
    def validate(self):
        started = time.perf_counter()
        totals = dict(ce=0., tokens=0, text_kl=0., multimodal_ce=0., multimodal_tokens=0,
                      text_only_ce=0., text_only_tokens=0, drift_num=0., drift_den=0.,
                      cka_ref_sum=0., cka_current_sum=0., target_met_sum=0., valid_images=0, invalid_images=0)
        with preserve_rng(), _evaluation_mode(self.model):
            for batch in self._loader(self.trainer.eval_dataset):
                inputs = self.trainer._prepare_inputs(batch)
                context = stage2_deterministic_context(self.model) if self.phase == "B" else nullcontext()
                with context, self.trainer.compute_loss_context_manager():
                    # Gate/diagnostics measure actual reference divergence even
                    # in CE-only controls or when training kl_weight is zero.
                    measure_cka = self.reference is not None and bool(getattr(self.model.config, "cka_loss", False))
                    output = stage2_forward(self.model, inputs, self.reference, need_cka=measure_cka,
                                            need_reference_logits=self.reference is not None,
                                            margin=self.option("margin"), eps=self.option("eps"))
                    labels = output["expanded_labels"][:, 1:]
                    tokens = int(labels.ne(IGNORE_INDEX).sum())
                    totals["ce"] += float(output["student_ce"]) * tokens
                    totals["tokens"] += tokens
                    if self.reference is not None:
                        kl, _ = token_kl(output["student_logits"], output["reference_logits"],
                                         output["expanded_labels"], self.option("kl_chunk_size"))
                        totals["text_kl"] += float(kl) * tokens
                    for row, has_image in enumerate(output["image_rows"]):
                        kind = "multimodal" if bool(has_image) else "text_only"
                        indices = labels[row].ne(IGNORE_INDEX).nonzero(as_tuple=True)[0]
                        totals[kind + "_tokens"] += indices.numel()
                        for chunk in indices.split(self.option("kl_chunk_size")):
                            if chunk.numel():
                                totals[kind + "_ce"] += float(F.cross_entropy(
                                    output["student_logits"][row, chunk].float(), labels[row, chunk], reduction="sum"))
                    metrics = output.get("cka_metrics", {})
                    for name in ("cka_ref_sum", "cka_current_sum", "target_met_sum", "valid_images", "invalid_images"):
                        totals[name] += metrics.get(name, 0.)
                    for student, reference in zip(output["projected_features"], output["reference_projected_features"]):
                        totals["drift_num"] += float((student.float() - reference.float()).square().sum())
                        totals["drift_den"] += float(reference.float().square().sum())
                del output
        if not totals["tokens"]:
            raise ValueError("Stage-2 held-out validation has no supervised assistant-answer tokens.")
        result = {"ce": totals["ce"] / totals["tokens"], "text_kl": totals["text_kl"] / totals["tokens"],
                  "drift": totals["drift_num"] / (totals["drift_den"] + self.option("eps")),
                  "cka_ref": totals["cka_ref_sum"] / max(1, totals["valid_images"]),
                  "cka_current": totals["cka_current_sum"] / max(1, totals["valid_images"]),
                  "target_met": totals["target_met_sum"] / max(1, totals["valid_images"]),
                  "valid_images": totals["valid_images"], "invalid_images": totals["invalid_images"],
                  "valid_tokens": totals["tokens"], "seconds": time.perf_counter() - started}
        for kind in ("multimodal", "text_only"):
            result[kind + "_tokens"] = totals[kind + "_tokens"]
            if totals[kind + "_tokens"]:
                result[kind + "_ce"] = totals[kind + "_ce"] / totals[kind + "_tokens"]
        return result

    def reset_refinement_optimizer(self):
        optimizer = self.trainer.optimizer
        optimizer.state.clear()
        # Preserve the recipe's projector/LLM LR ratio when applying refine_lr.
        for group in optimizer.param_groups:
            ratio = group.get("protected_stage2_lr_ratio")
            if ratio is None:
                ratio = group.get("initial_lr", self.args.learning_rate) / self.args.learning_rate
                group["protected_stage2_lr_ratio"] = ratio
            group["lr"] = group["initial_lr"] = self.option("refine_lr") * ratio
        self.trainer.lr_scheduler = None
        self.trainer.create_scheduler(max(1, self.phase_b_horizon), optimizer=optimizer)
        self.trainer.callback_handler.lr_scheduler = self.trainer.lr_scheduler

    def enter_phase_b(self, reason="explicit_continuation"):
        self.save_checkpoint(name=f"ce-only-checkpoint-{self.trainer.state.global_step}")
        self.phase, self.phase_b_start = "B", self.trainer.state.global_step
        self.transition_reason = reason
        self.refinement_accepted = False
        started = time.perf_counter()
        self.reference = snapshot_stage2_reference(self.model, placement=self.option("reference_device"))
        self.reference_file = None
        self.calibrate()
        metrics = self.validate()
        self.ce_ref = metrics["ce"]
        if not all(math.isfinite(metrics[key]) for key in ("ce", "text_kl", "drift")):
            raise RuntimeError("Nonfinite Stage-2 reference validation; refinement cannot start.")
        self.reset_refinement_optimizer()
        self.ema = None
        self.trainer.log({"protected2/transition_step": self.phase_b_start, "protected2/ce_ref": self.ce_ref,
                          "protected2/phase_b": 1, "protected2/phase_b_budget": self.phase_b_horizon,
                          "protected2/transition_reason": reason,
                          "protected2/refinement_accepted": 0,
                          "protected2/time/reference_calibration_seconds": time.perf_counter() - started})
        self.accepted_checkpoint = self.save_checkpoint()

    def begin(self):
        if self.loaded:
            self.loaded = False
            return
        if self.option("start_phase_b"):
            self.enter_phase_b(reason="explicit_continuation")
        else:
            metrics = self.validate()
            self.monitor.update(metrics["ce"], eligible=False)
            self.trainer.log({"protected2/validation/" + k: v for k, v in metrics.items()})
            self.accepted_checkpoint = self.save_checkpoint()

    def gate(self, metrics):
        valid = (all(math.isfinite(metrics[key]) for key in ("ce", "text_kl", "drift"))
                 and metrics["ce"] <= self.ce_ref * (1 + self.option("ce_tolerance"))
                 and metrics["text_kl"] <= self.option("text_kl_budget"))
        budget = self.option("drift_budget")
        return valid and (budget is None or metrics["drift"] <= budget)

    def step_end(self, control, *, force=False):
        step = self.trainer.state.global_step
        if step % max(1, self.args.logging_steps) == 0:
            self.trainer.log(getattr(self, "latest_logs", {}))
        at_limit = ((self.phase == "A" and step >= self.option("max_phase_a_steps"))
                    or (self.phase == "B" and step - self.phase_b_start >= self.phase_b_horizon))
        # Reserve the refinement budget at its exact optimizer-step boundary,
        # even when that boundary does not coincide with validation cadence.
        # Minimum CE-only training remains mandatory for either trigger.
        eligible = step >= self.option("min_phase_a_steps")
        reserve = (self.phase == "A" and eligible
                   and self.args.max_steps - step <= self.option("min_phase_b_steps"))
        if not force and step % self.option("eval_interval") and step < self.args.max_steps and not at_limit and not reserve and not control.should_training_stop:
            return
        metrics = self.validate()
        self.trainer.log({"protected2/validation/" + key: value for key, value in metrics.items()})
        if self.phase == "A":
            plateau = self.monitor.update(metrics["ce"], eligible=eligible)
            if eligible and (plateau or reserve) and step < self.args.max_steps:
                self.enter_phase_b(reason="plateau" if plateau else "budget_reservation")
            else:
                if at_limit or step >= self.args.max_steps:
                    self.finished_reason = "no_refinement_steps_remaining" if plateau else "phase_a_limit_without_plateau"
                    control.should_training_stop = True
                    self.trainer.log({"protected2/" + self.finished_reason: 1})
                self.accepted_checkpoint = self.save_checkpoint()
        else:
            gate_pass = self.gate(metrics)
            ce_control = self.option("continuation") == "ce_only"
            if ce_control and not all(math.isfinite(metrics[key]) for key in ("ce", "text_kl", "drift")):
                raise RuntimeError("Nonfinite Stage-2 CE-only validation; refusing checkpoint export.")
            # CE-only is a fixed-budget control, not protected refinement:
            # divergence from its initial checkpoint is diagnostic, not a
            # reason to rollback ordinary CE optimization.
            if not gate_pass and not ce_control:
                self.pending_rollback, control.should_training_stop = True, True
                self.trainer.log({"protected2/gate_pass": 0, "protected2/rollback": 1})
                return
            # The transition snapshot is a safe CE-only baseline, not an
            # accepted refinement. Only a later passing gate can promote it.
            self.refinement_accepted = self.use_cka and step > self.phase_b_start
            if at_limit or step >= self.args.max_steps:
                control.should_training_stop = True
                self.finished_reason = "refinement_budget_complete"
            self.trainer.log({"protected2/gate_pass": int(gate_pass), "protected2/retries": self.retries,
                              "protected2/gate_diagnostic_only": int(ce_control),
                              "protected2/refinement_accepted": int(self.refinement_accepted)})
            self.accepted_checkpoint = self.save_checkpoint()

    def completion_metrics(self):
        steps = 0 if self.phase_b_start is None else self.trainer.state.global_step - self.phase_b_start
        return dict(protected2_refinement_accepted=int(self.refinement_accepted),
                    protected2_accepted_refinement_steps=steps if self.refinement_accepted else 0,
                    protected2_phase_b_steps=steps,
                    protected2_requested_phase_b_steps=0 if self.phase_b_start is None else self.phase_b_horizon)

    def state_dict(self):
        return dict(version=1, signature=self.signature, phase=self.phase, phase_b_start=self.phase_b_start,
                    transition_reason=self.transition_reason,
                    refinement_accepted=self.refinement_accepted, source_provenance=self.source_provenance,
                    has_reference=self.reference is not None,
                    ema=None if self.ema is None else [None if value is None else value.detach().to(device="cpu", copy=True)
                                                       for value in self.ema],
                    ce_ref=self.ce_ref, ce_grad_refs=self.ce_grad_refs, aux_ratios=self.aux_ratios,
                    retries=self.retries, monitor=vars(self.monitor).copy(), finished_reason=self.finished_reason,
                    seen_cka_support=sorted(self.seen_cka_support))

    def load_checkpoint(self, directory):
        directory = Path(directory)
        required = (STATE_NAME, STUDENT_NAME, "optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json")
        missing = [name for name in required if not (directory / name).is_file()]
        if missing:
            raise ValueError("Incomplete Protected Stage-2 checkpoint: " + ", ".join(missing))
        payload = _load(directory / STATE_NAME)
        saved_signature = copy.deepcopy(payload.get("signature", {}))
        # Old runs had no budget reservation. Continuing their exact run is
        # possible only when explicitly retaining that policy (min B = 0).
        saved_signature.setdefault("config", {}).setdefault("protected_stage2_min_phase_b_steps", 0)
        if payload.get("version") != 1 or saved_signature != self.signature:
            raise ValueError("Protected Stage-2 checkpoint/config/data mismatch; exact resume needs the same setup.")
        if payload["has_reference"] and not (directory / REFERENCE_NAME).is_file():
            raise ValueError("Protected Stage-2 checkpoint is missing its immutable reference model.")
        self.model.load_state_dict(_load(directory / STUDENT_NAME), strict=True)
        for name in ("phase", "phase_b_start", "ce_ref", "ce_grad_refs", "aux_ratios", "retries", "finished_reason"):
            setattr(self, name, payload[name])
        self.transition_reason = payload.get("transition_reason")
        self.refinement_accepted = payload.get("refinement_accepted", False)
        self.source_provenance = payload.get("source_provenance")
        self.ema = (None if payload["ema"] is None else
                    [None if value is None else value.to(p.device) for value, p in zip(payload["ema"], self.params)])
        if payload["has_reference"]:
            if self.reference is None:
                self.reference = snapshot_stage2_reference(self.model, placement=self.option("reference_device"))
            self.reference.load_state_dict(_load(directory / REFERENCE_NAME), strict=True)
            self.reference_file = str(directory / REFERENCE_NAME)
        else:
            self.reference, self.reference_file = None, None
        self.monitor.__dict__.update(payload["monitor"])
        self.seen_cka_support = set(payload.get("seen_cka_support", []))
        self.accum_main = self.accum_cka = None
        self.accum_count, self.step_logs = 0, {}
        self.accepted_checkpoint, self.loaded = str(directory), True

    def save_checkpoint(self, name=None):
        if self.accum_count:
            raise RuntimeError("Protected Stage-2 checkpoints require an effective-batch boundary.")
        root = Path(self.args.output_dir)
        destination = root / (name or f"checkpoint-{self.trainer.state.global_step}")
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite existing checkpoint {destination}")
        temporary = Path(tempfile.mkdtemp(prefix=".protected-stage2-", dir=root))
        torch.save(_cpu_state(self.model), temporary / STUDENT_NAME)
        self.model.config.save_pretrained(temporary)
        torch.save(self.args, temporary / "training_args.bin")
        torch.save(self.state_dict(), temporary / STATE_NAME)
        if self.reference is not None:
            if self.reference_file is None:
                torch.save(_cpu_state(self.reference), temporary / REFERENCE_NAME)
            else:
                try:
                    os.link(self.reference_file, temporary / REFERENCE_NAME)
                except OSError:
                    shutil.copyfile(self.reference_file, temporary / REFERENCE_NAME)
        split = root / SPLIT_DIRECTORY / "split.json"
        if split.is_file():
            (temporary / SPLIT_DIRECTORY).mkdir()
            shutil.copyfile(split, temporary / SPLIT_DIRECTORY / "split.json")
        self.trainer._save_optimizer_and_scheduler(str(temporary))
        self.trainer._save_scaler(str(temporary))
        self.trainer._save_rng_state(str(temporary))
        self.trainer.state.save_to_json(str(temporary / "trainer_state.json"))
        with (temporary / CHECKPOINT_MARKER).open("w", encoding="utf-8") as handle:
            json.dump(self._checkpoint_marker(self.trainer.state.global_step), handle, sort_keys=True)
        os.replace(temporary, destination)
        if self.reference is not None:
            self.reference_file = str(destination / REFERENCE_NAME)
        self._rotate_checkpoints(destination)
        return str(destination)

    def _checkpoint_marker(self, step):
        # A tiny completion/ownership marker avoids deserializing model-sized
        # EMA tensors merely to decide whether an older snapshot is ours.
        return dict(version=1, output_dir=str(Path(self.args.output_dir).resolve()), global_step=step,
                    signature_sha256=hashlib.sha256(json.dumps(self.signature, sort_keys=True).encode()).hexdigest(),
                    has_reference=self.reference is not None)

    def _rotate_checkpoints(self, destination):
        limit = self.args.save_total_limit
        if limit is None or limit <= 0 or re.fullmatch(r"checkpoint-\d+", destination.name) is None:
            return
        root = Path(self.args.output_dir).resolve()
        keep = {destination.resolve()}
        for path in (self.accepted_checkpoint, self.option("source_stage2_checkpoint"),
                     getattr(self.args, "resume_from_checkpoint", None)):
            if isinstance(path, (str, Path)):
                keep.add(Path(path).resolve())
        identity = self._checkpoint_marker(0)
        required = (STATE_NAME, STUDENT_NAME, "optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json")
        candidates = []
        for path in root.iterdir():
            match = re.fullmatch(r"checkpoint-(\d+)", path.name)
            if not match or path.is_symlink() or not path.is_dir() or path.resolve().parent != root:
                continue
            marker_file = path / CHECKPOINT_MARKER
            if marker_file.is_symlink() or not marker_file.is_file():
                continue  # Legacy/unmarked and foreign directories are never retention targets.
            try:
                with marker_file.open(encoding="utf-8") as handle:
                    marker = json.load(handle)
            except (OSError, ValueError):
                continue
            if not isinstance(marker, dict) or any(marker.get(key) != identity[key]
                    for key in ("version", "output_dir", "signature_sha256")):
                continue
            if marker.get("global_step") != int(match.group(1)) or not isinstance(marker.get("has_reference"), bool):
                continue
            files = required + ((REFERENCE_NAME,) if marker["has_reference"] else ())
            if any(not (path / name).is_file() or (path / name).is_symlink() for name in files):
                continue
            candidates.append((int(match.group(1)), path))
        # Promotion happens after save_checkpoint returns. Keep the previous
        # accepted snapshot as well as at least two latest complete snapshots.
        candidates.sort(key=lambda item: item[0], reverse=True)
        keep.update(path for _, path in candidates[:max(2, int(limit))])
        for _, path in candidates:
            if path not in keep:
                try:
                    shutil.rmtree(path)
                    print(f"[Protected Stage 2] Removed expired checkpoint: {path}", flush=True)
                except OSError as error:
                    print(f"[Protected Stage 2] Could not remove expired checkpoint {path}: {error}", flush=True)

    def _validate_continuation_source(self, source, payload):
        """Weights-only initialization has a narrower contract than exact resume.

        The new run owns its optimizer, schedule and refinement settings, but
        may not quietly reuse a different dataset, tokenizer/model or adapter
        recipe. In particular, source phase/shape alone is not provenance.
        """
        signature = payload.get("signature", {})
        if (payload.get("version") != 1 or payload.get("phase") != "A"
                or payload.get("has_reference") is not False or payload.get("phase_b_start") is not None):
            raise ValueError("source_stage2_checkpoint must be a verified Stage-2 CE-only Phase-A snapshot.")
        for key in ("train_dataset", "eval_dataset", "trainable"):
            if key not in signature or signature[key] != self.signature[key]:
                raise ValueError(f"Stage-2 continuation source {key} mismatch; use the original data/model recipe.")
        source_model = {key: value for key, value in signature.get("model", {}).items()
                        if not key.startswith("cka_loss")}
        current_model = {key: value for key, value in self.signature["model"].items()
                         if not key.startswith("cka_loss")}
        if source_model != current_model:
            raise ValueError("Stage-2 continuation source model/preprocessing mismatch.")
        source_args, current_args = signature.get("config", {}), self.signature["config"]
        checks = ["protected_stage2_split_fingerprint", "lora_enable"]
        if source_args.get("lora_enable") or current_args.get("lora_enable"):
            checks += ["lora_r", "lora_alpha", "lora_dropout", "lora_bias"]
        for key in checks:
            if source_args.get(key) != current_args.get(key):
                raise ValueError(f"Stage-2 continuation source {key} mismatch.")
        self.source_provenance = {
            "checkpoint": str(source.resolve()), "kind": "protected_stage2_ce_only",
            "signature_sha256": hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest(),
            "source_finished_reason": payload.get("finished_reason"),
        }

    def _validate_source_model_config(self, source, *, verified_signature=None):
        config_file = source / "config.json"
        if not config_file.is_file():
            raise ValueError("Stage-2 continuation source requires config.json to verify model identity.")
        with config_file.open(encoding="utf-8") as handle:
            saved = json.load(handle)
        # Transformers 4.51 removes _name_or_path even when it is nonempty.
        # Restore it only from independently verified checkpoint metadata, not
        # from the requested/current model (which would conceal a mismatch).
        if "_name_or_path" not in saved:
            if verified_signature is not None and "_name_or_path" in verified_signature.get("model", {}):
                saved["_name_or_path"] = verified_signature["model"]["_name_or_path"]
            elif (source / "adapter_config.json").is_file():
                with (source / "adapter_config.json").open(encoding="utf-8") as handle:
                    saved["_name_or_path"] = json.load(handle).get("base_model_name_or_path")
                if not saved["_name_or_path"]:
                    raise ValueError("Legacy Stage-2 source lacks verified base-model identity in adapter_config.json.")
            else:
                raise ValueError("Legacy Stage-2 source lacks model identity: config.json omits _name_or_path "
                                 "and no protected source signature is available. Use a verified CE-only checkpoint.")
        # save_pretrained writes a config diff, omitting default-valued fields.
        # Reconstruct those defaults before comparing with the live full config.
        saved = {**type(self.model.config).from_dict(saved).to_dict(), **saved}
        if saved.get("tune_mm_mlp_adapter") or saved.get("protected_cka_refinement"):
            raise ValueError("Stage-2 continuation source must be Stage 2 CE-only, not a Stage-1 checkpoint.")
        if not (source / STATE_NAME).is_file() and saved.get("cka_loss"):
            raise ValueError("Legacy Stage-2 continuation source enables CKA and cannot be verified CE-only.")
        current = self.model.config.to_dict()
        keys = ("_name_or_path", "model_type", "hidden_size", "num_hidden_layers", "vocab_size",
                "num_attention_heads", "num_key_value_heads", "head_dim", "intermediate_size",
                "hidden_act", "rope_theta", "rope_scaling", "tie_word_embeddings",
                "bos_token_id", "eos_token_id", "pad_token_id", "mm_vision_tower", "mm_hidden_size",
                "mm_vision_select_layer", "mm_vision_select_feature", "mm_projector_type",
                "mm_patch_merge_type", "mm_use_im_start_end", "mm_use_im_patch_token",
                "tokenizer_model_max_length", "tokenizer_padding_side", "llava_conversation_version")
        different = [key for key in keys if saved.get(key) != current.get(key)]
        if different:
            raise ValueError("Stage-2 continuation source model/preprocessing mismatch: " + ", ".join(different))

    def _initialize_source(self):
        source = self.option("source_stage2_checkpoint")
        if not source or self.source_initialized:
            return
        source = Path(source)
        if (source / STATE_NAME).is_file():
            payload = _load(source / STATE_NAME)
            self._validate_continuation_source(source, payload)
            self._validate_source_model_config(source, verified_signature=payload["signature"])
            self.model.load_state_dict(_load(source / STUDENT_NAME), strict=True)
        else:
            self._validate_source_model_config(source)
            self.source_provenance = {"checkpoint": str(source.resolve()), "kind": "legacy_stage2_ce_only"}
            trainable = {name: p.requires_grad for name, p in self.model.named_parameters()}
            full_files = ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json",
                          "pytorch_model.bin.index.json")
            if any((source / name).is_file() for name in full_files):
                if any((source / name).is_file() for name in full_files[2:]):
                    from transformers.modeling_utils import load_sharded_checkpoint
                    load_sharded_checkpoint(self.model, str(source), strict=True, prefer_safe=True)
                elif (source / "model.safetensors").is_file():
                    from safetensors.torch import load_file
                    self.model.load_state_dict(load_file(str(source / "model.safetensors")), strict=True)
                else:
                    self.model.load_state_dict(_load(source / "pytorch_model.bin"), strict=True)
            else:
                if not hasattr(self.model, "peft_config"):
                    raise ValueError("A LoRA CE-only source requires the matching LoRA student recipe.")
                adapter_file = source / "adapter_config.json"
                if not adapter_file.is_file():
                    raise ValueError("LoRA CE-only source requires adapter_config.json to verify its recipe.")
                with adapter_file.open(encoding="utf-8") as handle:
                    adapter = json.load(handle)
                if len(self.model.peft_config) != 1:
                    raise ValueError("Stage-2 legacy continuation supports one verified LoRA adapter.")
                current_adapter = next(iter(self.model.peft_config.values())).to_dict()
                for key in ("r", "lora_alpha", "lora_dropout", "bias", "use_rslora", "use_dora",
                            "rank_pattern", "alpha_pattern", "modules_to_save", "target_modules"):
                    before, after = adapter.get(key), current_adapter.get(key)
                    if isinstance(before, (list, set)) and isinstance(after, (list, set)):
                        before, after = sorted(before), sorted(after)
                    if before != after:
                        raise ValueError(f"Stage-2 continuation source LoRA recipe mismatch: {key}.")
                non_lora = source / "non_lora_trainables.bin"
                if not non_lora.is_file():
                    raise ValueError("LoRA source is missing non_lora_trainables.bin (including the trained projector). "
                                     "Use a complete Stage-2 export, not an adapter-only checkpoint.")
                weights = _load(non_lora)
                # Legacy export removes the PEFT base_model prefix.
                current = self.model.state_dict()
                mapped = {key if key in current else "base_model." + key: value for key, value in weights.items()}
                required = {name for name, p in self.model.named_parameters() if p.requires_grad
                            and "lora_" not in name and ".modules_to_save." not in name}
                if not required.issubset(mapped):
                    raise ValueError("LoRA CE-only source lacks trainable non-LoRA weights: "
                                     + ", ".join(sorted(required - mapped.keys())))
                Trainer._load_from_checkpoint(self.trainer, str(source), model=self.model)
                unexpected = self.model.load_state_dict(mapped, strict=False).unexpected_keys
                if unexpected:
                    raise ValueError(f"Unexpected non-LoRA source parameters: {unexpected}")
            for name, p in self.model.named_parameters():
                if name not in trainable:
                    raise ValueError("Source loading changed adapter topology; use the original Stage-2 LoRA recipe.")
                p.requires_grad_(trainable[name])
            actual = dict(self.model.named_parameters())
            if any(actual.get(name) is not parameter for name, parameter in self.named_params):
                raise ValueError("Source loading replaced trainable parameter objects; initialize the matching LoRA recipe first.")
        self.source_initialized = True

    def run(self, parent_train, resume_from_checkpoint=None, **kwargs):
        if resume_from_checkpoint is True:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError("No Protected Stage-2 checkpoint to resume.")
        if resume_from_checkpoint:
            state_path = Path(resume_from_checkpoint) / STATE_NAME
            if not state_path.is_file():
                raise ValueError("Full Stage-2 resume needs protected state; use source_stage2_checkpoint for CE-only initialization.")
            saved = _load(state_path)
            state = TrainerState.load_from_json(str(Path(resume_from_checkpoint) / "trainer_state.json"))
            if saved.get("finished_reason") or state.global_step >= self.args.max_steps:
                raise ValueError("Protected Stage-2 run already finished; start a new continuation instead.")
        else:
            self._initialize_source()
        while True:
            self.pending_rollback = False
            result = parent_train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)
            if not self.pending_rollback:
                accepted = TrainerState.load_from_json(str(Path(self.accepted_checkpoint) / "trainer_state.json"))
                if accepted.global_step != self.trainer.state.global_step:
                    self.step_end(self.trainer.control, force=True)
            if not self.pending_rollback:
                return TrainOutput(result.global_step, result.training_loss,
                                   dict(result.metrics, **self.completion_metrics()))
            reduced = {group: value * self.option("aux_backoff") for group, value in self.aux_ratios.items()}
            failures, checkpoint = self.retries + 1, self.accepted_checkpoint
            self.load_checkpoint(checkpoint)
            self.trainer._load_optimizer_and_scheduler(checkpoint)
            self.trainer._load_scaler(checkpoint)
            self.trainer.state = TrainerState.load_from_json(str(Path(checkpoint) / "trainer_state.json"))
            self.trainer._load_rng_state(checkpoint)
            self.model.zero_grad(set_to_none=True)
            self.aux_ratios, self.retries = reduced, failures
            exhausted = failures > self.option("max_retries")
            self.finished_reason = "gate_retry_exhausted" if exhausted else None
            _atomic_save(self.state_dict(), Path(checkpoint) / STATE_NAME)
            self.trainer.log({"protected2/gate_failures": failures, "protected2/rollbacks": failures,
                              "protected2/proj_aux_ratio": reduced["projector"],
                              "protected2/llm_aux_ratio": reduced["llm"],
                              "protected2/safe_step": self.trainer.state.global_step,
                              "protected2/retries_exhausted": int(exhausted)})
            if exhausted:
                return TrainOutput(self.trainer.state.global_step, result.training_loss,
                                   dict(result.metrics, protected2_retry_exhausted=1,
                                        protected2_safe_step=self.trainer.state.global_step,
                                        **self.completion_metrics()))
            resume_from_checkpoint = checkpoint
            # Native AMP preparation wraps model.forward. Re-entering Trainer
            # with an already prepared model would nest those wrappers.
            self.trainer.optimizer = self.trainer.lr_scheduler = None
            self.trainer._created_lr_scheduler = False
            self.trainer.model = self.trainer.accelerator.unwrap_model(
                self.trainer.model, keep_fp32_wrapper=False)
            self.trainer.model_wrapped = self.trainer.model
