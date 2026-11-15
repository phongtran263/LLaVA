"""Opt-in Stage-1 lifecycle; the existing Trainer still owns optimizer steps.

The deliberately narrow backend contract (single process, FP32/BF16, no gradient
checkpointing) makes gradient accumulation and full rollback explicit. No model
or Stage-2 forward is changed when this feature is disabled.
"""

import copy
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import TrainerCallback, TrainerState
from transformers.trainer_utils import TrainOutput, get_last_checkpoint

from llava.constants import IGNORE_INDEX
from llava.train.protected_cka_forward import ce_only_forward, protected_cka_forward
from llava.train.protected_cka_math import (
    differentiable_zero, protected_projector_gradient, target_margin_cka, token_kl,
)

STATE_NAME = "protected_cka_state.pt"


@contextmanager
def preserve_rng():
    """Validation/calibration must not consume the training RNG stream."""
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
             torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    try:
        yield
    finally:
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])
        if state[3] is not None:
            torch.cuda.set_rng_state_all(state[3])


def _load(path):
    # Our own checkpoints include RNG/config state, not just tensor weights.
    return torch.load(path, map_location="cpu", weights_only=False)


def _atomic_save(value, path):
    temporary = str(path) + ".tmp"
    torch.save(value, temporary)
    os.replace(temporary, path)


class PlateauMonitor:
    """Absolute improvement in a trailing mean of validation CE."""

    def __init__(self, window, patience, min_delta):
        self.window, self.patience, self.min_delta = window, patience, min_delta
        self.history, self.best, self.bad_evaluations = [], None, 0

    def update(self, value, *, eligible=True):
        if not math.isfinite(value):
            raise RuntimeError("Phase-A validation CE is nonfinite; cannot establish alignment.")
        self.history = (self.history + [float(value)])[-self.window:]
        if len(self.history) < self.window:
            return False
        mean = sum(self.history) / len(self.history)
        if self.best is None or self.best - mean > self.min_delta:
            self.best, self.bad_evaluations = mean, 0
        elif eligible:
            self.bad_evaluations += 1
        return eligible and self.bad_evaluations >= self.patience


class _ProtectedCallback(TrainerCallback):
    def __init__(self, runtime):
        self.runtime = runtime

    def on_train_begin(self, args, state, control, **kwargs):
        self.runtime.begin()

    def on_step_end(self, args, state, control, **kwargs):
        self.runtime.step_end(control)


class ProtectedCKARefinement:
    def __init__(self, trainer):
        self.trainer, self.args = trainer, trainer.args
        self.model = trainer.model
        self.projector = self.model.get_model().mm_projector
        self.params = [p for p in self.projector.parameters() if p.requires_grad]
        ids = {id(p) for p in self.params}
        if not ids or any(p.requires_grad and id(p) not in ids for p in self.model.parameters()):
            raise ValueError("Protected CKA Stage 1 requires only the projector to be trainable.")
        if trainer.eval_dataset is None or len(trainer.eval_dataset) == 0:
            raise ValueError("Protected CKA requires a fixed held-out validation dataset.")
        if trainer.accelerator.num_processes != 1 or trainer.accelerator.scaler is not None:
            raise ValueError("Protected CKA v1 supports one process and FP32/BF16 without a scaler only.")
        self.phase = "A"
        self.reference = None
        self.phase_b_start = None
        self.ce_ref = self.ce_grad_ref = None
        self.ema = None
        self.aux_ratio = self.option("aux_ratio")
        self.retries = 0
        self.accepted_checkpoint = None
        self.pending_rollback = False
        self.loaded = False
        self.finished_reason = None
        self.transition_reason = None
        self.source_provenance = None
        self.refinement_accepted = False
        self.monitor = PlateauMonitor(self.option("plateau_window"), self.option("plateau_patience"),
                                      self.option("plateau_min_delta"))
        self.accum_main = self.accum_cka = None
        self.accum_count = 0
        self.step_logs = {}
        self.signature = self._signature()
        self.callback = _ProtectedCallback(self)

        # Transformers 4.51 computes resumed epoch/skip offsets with floor
        # division. A partial accumulation window then replays different data.
        # Reject that case instead of duplicating its entire inner loop.
        with preserve_rng():
            batches = len(trainer.get_train_dataloader())
        if not batches or batches % self.args.gradient_accumulation_steps:
            raise ValueError(
                f"Protected CKA exact rollback requires complete accumulation windows: "
                f"{batches} training microbatches are not divisible by "
                f"gradient_accumulation_steps={self.args.gradient_accumulation_steps}. "
                "Use gradient_accumulation_steps=1 or adjust the batch/dataset sizes."
            )
        trainer.add_callback(self.callback)

    def option(self, name):
        defaults = {"min_phase_b_steps": 0, "source_stage1_checkpoint": None,
                    "continuation": "protected"}
        if name in defaults:
            return getattr(self.args, "protected_cka_" + name, defaults[name])
        return getattr(self.args, "protected_cka_" + name)

    def _signature(self):
        # Paths/output directory are not identity: content and sampling order are.
        data = {}
        for name in ("train_dataset", "eval_dataset"):
            dataset = getattr(self.trainer, name)
            rows = getattr(dataset, "list_data_dict", None)
            if rows is None:
                # Primarily useful for tiny deterministic unit-test datasets.
                rows = {"class": type(dataset).__qualname__, "length": len(dataset)}
            data[name] = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
        config = {key: value for key, value in vars(self.args).items()
                  if key.startswith("protected_cka_")}
        config.pop("protected_cka_start_phase_b", None)  # initialization-only
        config.pop("protected_cka_source_stage1_checkpoint", None)  # weights-only initialization
        config.pop("protected_cka_split_manifest", None)  # content fingerprint is retained
        # Unpickled older dataclasses may inherit new class defaults without
        # storing those fields in vars(args). Sign the actual runtime policy.
        config["protected_cka_min_phase_b_steps"] = self.option("min_phase_b_steps")
        config["protected_cka_continuation"] = self.option("continuation")
        for key in ("max_steps", "seed", "data_seed", "per_device_train_batch_size",
                    "gradient_accumulation_steps", "learning_rate", "warmup_steps", "warmup_ratio",
                    "weight_decay", "adam_beta1", "adam_beta2", "adam_epsilon", "max_grad_norm",
                    "bf16", "lr_scheduler_type", "lr_scheduler_kwargs", "optim", "dataloader_drop_last",
                    "per_device_eval_batch_size", "mm_projector_lr", "label_smoothing_factor"):
            config[key] = getattr(self.args, key, None)
        data["config"] = json.loads(json.dumps(config, default=str))
        data["model"] = {key: getattr(self.model.config, key, None) for key in (
            "_name_or_path", "model_type", "hidden_size", "num_hidden_layers", "vocab_size",
            "mm_vision_tower", "mm_vision_select_layer", "mm_projector_type", "mm_patch_merge_type",
            "tokenizer_model_max_length", "tokenizer_padding_side", "_attn_implementation")}
        return data

    def _flat_grad(self, loss, *, retain_graph=False):
        if not loss.requires_grad:
            return torch.cat([torch.zeros_like(p, dtype=torch.float32).reshape(-1) for p in self.params])
        gradients = torch.autograd.grad(loss, self.params, retain_graph=retain_graph, allow_unused=True)
        return torch.cat([(g.detach().float() if g is not None else torch.zeros_like(p, dtype=torch.float32))
                          .reshape(-1) for p, g in zip(self.params, gradients)])

    def _cka(self, output):
        if not output["projected_features"]:
            return differentiable_zero(output["student_logits"]), {
                "valid_images": 0, "invalid_images": 0, "cka_ref": 0., "cka_current": 0.,
                "target_met": 0., "cka_ref_sum": 0., "cka_current_sum": 0., "target_met_sum": 0.,
            }
        return target_margin_cka(
            torch.stack(output["vision_features"]), torch.stack(output["projected_features"]),
            torch.stack(output["reference_projected_features"]),
            margin=self.option("margin"), eps=self.option("eps"),
        )

    def training_step(self, model, inputs):
        inputs = self.trainer._prepare_inputs(inputs)
        with self.trainer.compute_loss_context_manager():
            continuation = self.option("continuation")
            if self.phase == "A" or continuation == "ce_only":
                output = ce_only_forward(model, inputs)
                ce = output["student_ce"]
                main, cka, kl = ce, ce.new_zeros(()), ce.new_zeros(())
                metrics = {}
            else:
                output = protected_cka_forward(model, self.reference, inputs,
                                               need_reference_logits=self.option("kl_weight") > 0)
                ce = output["student_ce"]
                kl = (token_kl(output["student_logits"], output["reference_logits"],
                               output["expanded_labels"], self.option("kl_chunk_size"))[0]
                      if self.option("kl_weight") > 0 else ce.new_zeros(()))
                cka, metrics = (self._cka(output) if continuation == "protected"
                                else (ce.new_zeros(()), {}))
                main = ce + self.option("kl_weight") * kl
        if not bool(torch.isfinite(main)) or not bool(torch.isfinite(cka)):
            raise RuntimeError("Nonfinite Protected CKA objective; no optimizer update was performed.")
        use_aux = (self.phase == "B" and continuation == "protected"
                   and self.option("lambda_max") > 0 and self.aux_ratio > 0)
        # Two VJPs, but the auxiliary traverses only the tiny projector, not LLM.
        # Do that small branch first, allowing the large main backward to free
        # its saved decoder activations immediately.
        g_cka = self._flat_grad(cka, retain_graph=True) if use_aux else None
        g_main = self._flat_grad(main)
        if g_cka is None:
            g_cka = torch.zeros_like(g_main)
        self.accum_main = g_main if self.accum_main is None else self.accum_main.add_(g_main)
        self.accum_cka = g_cka if self.accum_cka is None else self.accum_cka.add_(g_cka)
        self.accum_count += 1
        logs = {"ce": float(ce.detach()), "text_kl": float(kl.detach()), "cka_loss": float(cka.detach()),
                "valid_tokens": int((output["expanded_labels"][:, 1:] != IGNORE_INDEX).sum()), **metrics}
        for key, value in logs.items():
            if not key.endswith("_sum"):
                self.step_logs[key] = self.step_logs.get(key, 0.) + value
        if self.trainer.accelerator.sync_gradients:
            count = self.accum_count  # includes a shorter final accumulation window
            # Match this repository's existing mean-of-microbatch-CE semantics
            # (compute_loss deliberately ignores num_items_in_batch). Surgery
            # is applied once to that accumulated effective-batch gradient.
            main_gradient, cka_gradient = self.accum_main / count, self.accum_cka / count
            gradient_logs = {}
            if self.phase == "B" and continuation == "protected":
                warmup = self.option("warmup_steps")
                step = self.trainer.state.global_step - self.phase_b_start
                weight = self.option("lambda_max") * (min(1., step / warmup) if warmup else 1.)
                final, self.ema, gradient_logs = protected_projector_gradient(
                    main_gradient, cka_gradient, self.ema, ema_decay=self.option("ema_decay"),
                    aux_ratio=self.aux_ratio, ce_grad_ref=self.ce_grad_ref, lambda_t=weight,
                    eps=self.option("eps"),
                )
                gradient_logs["lambda"] = weight
            else:
                final = main_gradient
            if not bool(torch.isfinite(final).all()):
                raise RuntimeError("Nonfinite projector gradient; refusing optimizer update.")
            offset = 0
            for parameter in self.params:
                size = parameter.numel()
                parameter.grad = final[offset:offset + size].view_as(parameter).to(parameter.dtype).clone()
                offset += size
            # HF Trainer owns clipping, the single optimizer step, LR step, zero_grad.
            self.latest_logs = {"protected/" + key: (value if key in (
                "valid_tokens", "valid_images", "invalid_images") else value / count)
                for key, value in self.step_logs.items()}
            self.latest_logs.update({"protected/grad/" + key: value for key, value in gradient_logs.items()})
            self.latest_logs.update({"protected/phase_b": int(self.phase == "B"),
                                     "protected/aux_ratio": self.aux_ratio})
            self.accum_main = self.accum_cka = None
            self.accum_count, self.step_logs = 0, {}
        # This is the main scalar objective, not a claim that surgery differentiates
        # an equivalent scalar loss. Report auxiliary and gradient scale separately.
        return main.detach() / self.args.gradient_accumulation_steps

    def _loader(self, dataset, *, training=False):
        return DataLoader(dataset, batch_size=(self.args.per_device_train_batch_size if training
                                              else self.args.per_device_eval_batch_size),
                          shuffle=False, num_workers=0, collate_fn=self.trainer.data_collator,
                          pin_memory=self.args.dataloader_pin_memory)

    def calibrate(self):
        """Mean CE norm over short effective batches, without a single update."""
        norms, accumulated, count = [], None, 0
        with preserve_rng():
            for batch in self._loader(self.trainer.train_dataset, training=True):
                with self.trainer.compute_loss_context_manager():
                    output = ce_only_forward(self.model, self.trainer._prepare_inputs(batch))
                gradient = self._flat_grad(output["student_ce"])
                accumulated = gradient if accumulated is None else accumulated.add_(gradient)
                count += 1
                if count == self.args.gradient_accumulation_steps:
                    norms.append(float((accumulated / count).norm()))
                    accumulated, count = None, 0
                    if len(norms) == self.option("ce_gradient_calibration_batches"):
                        break
            if count:
                norms.append(float((accumulated / count).norm()))
        if not norms or not all(math.isfinite(norm) for norm in norms):
            raise RuntimeError("Cannot calibrate a finite CE gradient reference.")
        self.ce_grad_ref = sum(norms) / len(norms)
        self.trainer.log({"protected/ce_grad_ref": self.ce_grad_ref,
                          "protected/calibration_effective_batches": len(norms)})

    @torch.no_grad()
    def validate(self):
        totals = dict(ce=0., text_kl=0., tokens=0, drift_num=0., drift_den=0.,
                      cka_ref_sum=0., cka_current_sum=0., target_met_sum=0.,
                      valid_images=0, invalid_images=0)
        with preserve_rng():
            for batch in self._loader(self.trainer.eval_dataset):
                inputs = self.trainer._prepare_inputs(batch)
                with self.trainer.compute_loss_context_manager():
                    output = (ce_only_forward(self.model, inputs) if self.reference is None
                              else protected_cka_forward(self.model, self.reference, inputs))
                    tokens = int((output["expanded_labels"][:, 1:] != IGNORE_INDEX).sum())
                    totals["ce"] += float(output["student_ce"]) * tokens
                    totals["tokens"] += tokens
                    if self.reference is not None:
                        kl, _ = token_kl(output["student_logits"], output["reference_logits"],
                                         output["expanded_labels"], self.option("kl_chunk_size"))
                        totals["text_kl"] += float(kl) * tokens
                        _, metrics = self._cka(output)
                        for key in ("cka_ref_sum", "cka_current_sum", "target_met_sum",
                                    "valid_images", "invalid_images"):
                            totals[key] += metrics[key]
                        for student, reference in zip(output["projected_features"],
                                                      output["reference_projected_features"]):
                            totals["drift_num"] += float((student.float() - reference.float()).square().sum())
                            totals["drift_den"] += float(reference.float().square().sum())
                del output  # release paired full-vocabulary logits before the next batch
        if totals["tokens"] == 0:
            raise ValueError("Held-out validation has no supervised caption tokens.")
        return {"ce": totals["ce"] / totals["tokens"],
                "text_kl": totals["text_kl"] / totals["tokens"],
                "drift": totals["drift_num"] / (totals["drift_den"] + self.option("eps")),
                "cka_ref": totals["cka_ref_sum"] / max(1, totals["valid_images"]),
                "cka_current": totals["cka_current_sum"] / max(1, totals["valid_images"]),
                "target_met": totals["target_met_sum"] / max(1, totals["valid_images"]),
                "valid_images": totals["valid_images"], "invalid_images": totals["invalid_images"],
                "valid_tokens": totals["tokens"]}

    def reset_refinement_optimizer(self):
        optimizer = self.trainer.optimizer
        optimizer.state.clear()
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] = self.option("refine_lr")
        self.trainer.lr_scheduler = None
        self.trainer.create_scheduler(max(1, self.args.max_steps - self.phase_b_start), optimizer=optimizer)
        self.trainer.callback_handler.lr_scheduler = self.trainer.lr_scheduler

    def enter_phase_b(self, reason="explicit_continuation"):
        self.save_checkpoint(name=f"ce-only-checkpoint-{self.trainer.state.global_step}")
        self.phase, self.phase_b_start = "B", self.trainer.state.global_step
        self.transition_reason = reason
        self.reference = copy.deepcopy(self.projector).requires_grad_(False).eval()
        self.calibrate()
        metrics = self.validate()  # student is exactly the immutable reference here
        self.ce_ref = metrics["ce"]
        if not all(math.isfinite(metrics[key]) for key in ("ce", "text_kl", "drift")):
            raise RuntimeError("Nonfinite reference validation; refinement cannot start.")
        self.reset_refinement_optimizer()
        self.ema = None
        self.trainer.log({"protected/transition_step": self.phase_b_start,
                          "protected/transition_reason": reason,
                          "protected/refinement_accepted": 0,
                          "protected/ce_ref": self.ce_ref, "protected/phase_b": 1})
        self.accepted_checkpoint = self.save_checkpoint()

    def begin(self):
        if self.loaded:
            self.loaded = False
            return
        if self.option("start_phase_b") or self.option("source_stage1_checkpoint"):
            self.enter_phase_b(reason="explicit_continuation")
        else:
            metrics = self.validate()
            self.monitor.update(metrics["ce"], eligible=False)
            self.trainer.log({"protected/validation/ce": metrics["ce"], "protected/phase_b": 0})
            self.accepted_checkpoint = self.save_checkpoint()

    def gate(self, metrics):
        return (all(math.isfinite(metrics[key]) for key in ("ce", "text_kl", "drift"))
                and metrics["ce"] <= self.ce_ref * (1 + self.option("ce_tolerance"))
                and metrics["text_kl"] <= self.option("text_kl_budget")
                and metrics["drift"] <= self.option("drift_budget"))

    def step_end(self, control, *, force=False):
        step = self.trainer.state.global_step
        if step % max(1, self.args.logging_steps) == 0:
            self.trainer.log(getattr(self, "latest_logs", {}))
        limit = self.option("max_phase_a_steps") or self.args.max_steps
        at_limit = self.phase == "A" and step >= limit
        eligible = step >= self.option("min_phase_a_steps")
        remaining = self.args.max_steps - step
        reserve_due = (self.phase == "A" and eligible
                       and 0 < remaining <= self.option("min_phase_b_steps"))
        if (not force and step % self.option("eval_interval") and step < self.args.max_steps
                and not at_limit and not reserve_due and not control.should_training_stop):
            return
        metrics = self.validate()
        self.trainer.log({"protected/validation/" + key: value for key, value in metrics.items()})
        if self.phase == "A":
            plateau = self.monitor.update(metrics["ce"], eligible=eligible)
            if eligible and (plateau or reserve_due) and remaining > 0:
                self.enter_phase_b(reason="plateau" if plateau else "budget_reservation")
            else:
                if at_limit or step >= self.args.max_steps:
                    self.finished_reason = ("no_refinement_steps_remaining" if plateau
                                            else "phase_a_limit_without_plateau")
                    control.should_training_stop = True
                    self.trainer.log({"protected/" + self.finished_reason: 1})
                self.accepted_checkpoint = self.save_checkpoint()
        else:
            passed = self.gate(metrics)
            finite = all(math.isfinite(metrics[key]) for key in ("ce", "text_kl", "drift"))
            ce_control = self.option("continuation") == "ce_only"
            # The CE-only arm measures unconstrained continuation for the same
            # fixed budget; refinement thresholds remain visible diagnostics,
            # not an extra CE-control objective or an early-stopping rule.
            self.trainer.log({"protected/gate_pass": int(passed),
                              "protected/gate_diagnostic_only": int(ce_control),
                              "protected/retries": self.retries})
            if finite and (passed or ce_control):
                self.refinement_accepted = (self.option("continuation") == "protected"
                                            and self.option("lambda_max") > 0
                                            and step > self.phase_b_start)
                self.trainer.log({"protected/refinement_accepted": int(self.refinement_accepted)})
                self.accepted_checkpoint = self.save_checkpoint()
            else:
                self.pending_rollback = True
                control.should_training_stop = True
                self.trainer.log({"protected/rollback": 1,
                                  "protected/nonfinite_validation": int(not finite)})

    def budget_metrics(self):
        start = self.phase_b_start
        return {"requested_total_steps": self.args.max_steps,
                "completed_total_steps": self.trainer.state.global_step,
                "requested_phase_b_steps": 0 if start is None else self.args.max_steps - start,
                "completed_phase_b_steps": 0 if start is None else self.trainer.state.global_step - start,
                "accepted_refinement_steps": (self.trainer.state.global_step - start
                                               if start is not None and self.refinement_accepted else 0),
                "refinement_accepted": int(self.refinement_accepted)}

    def state_dict(self):
        return dict(version=1, signature=self.signature, phase=self.phase, phase_b_start=self.phase_b_start,
                    reference=None if self.reference is None else {k: v.detach().cpu().clone()
                        for k, v in self.reference.state_dict().items()},
                    ema=None if self.ema is None else self.ema.detach().cpu().clone(),
                    ce_ref=self.ce_ref, ce_grad_ref=self.ce_grad_ref, aux_ratio=self.aux_ratio,
                    retries=self.retries, monitor=vars(self.monitor), finished_reason=self.finished_reason,
                    transition_reason=self.transition_reason, source_provenance=self.source_provenance,
                    refinement_accepted=self.refinement_accepted)

    @staticmethod
    def _resume_signature(signature):
        """Absent new fields retain legacy semantics, never newer defaults.

        Old exact resumes must request min_phase_b_steps=0; changing a finished
        horizon or phase policy is only supported as a separate continuation.
        """
        signature = copy.deepcopy(signature)
        config = signature.setdefault("config", {})
        config.setdefault("protected_cka_min_phase_b_steps", 0)
        config.setdefault("protected_cka_continuation", "protected")
        config.pop("protected_cka_source_stage1_checkpoint", None)
        return signature

    def load_source_checkpoint(self, directory):
        """Validate CE-only provenance and load weights for a NEW run.

        Source optimization/phase budgets may differ. The frozen model,
        preprocessing and exact train/held-out sample identities may not.
        """
        directory = Path(directory)
        required = (STATE_NAME, "mm_projector.bin", "config.json", "trainer_state.json")
        missing = [name for name in required if not (directory / name).is_file()]
        if missing:
            raise ValueError("Stage-1 CE-only continuation requires a verified protected Phase-A "
                             "checkpoint; missing: " + ", ".join(missing))
        payload = _load(directory / STATE_NAME)
        if (payload.get("version") != 1 or payload.get("phase") != "A"
                or payload.get("reference") is not None or payload.get("phase_b_start") is not None):
            raise ValueError("Stage-1 source must be a verified CE-only Phase-A checkpoint, not refinement weights.")
        signature = payload.get("signature", {})
        for key in ("model", "train_dataset", "eval_dataset"):
            if signature.get(key) != self.signature.get(key):
                raise ValueError(f"Stage-1 source {key} provenance mismatch; refusing continuation.")
        saved_fingerprint = signature.get("config", {}).get("protected_cka_split_fingerprint")
        current_fingerprint = self.signature["config"].get("protected_cka_split_fingerprint")
        if saved_fingerprint != current_fingerprint:
            raise ValueError("Stage-1 source held-out split/preprocessing provenance mismatch.")
        self._validate_source_model_config(directory, signature["model"])
        source_state = TrainerState.load_from_json(str(directory / "trainer_state.json"))
        weights = _load(directory / "mm_projector.bin")
        self.projector.load_state_dict({key.split("mm_projector.", 1)[-1]: value
                                        for key, value in weights.items()}, strict=True)
        self.source_provenance = {"checkpoint": str(directory.resolve()),
                                  "global_step": source_state.global_step,
                                  "signature": signature}

    def _validate_source_model_config(self, directory, signed_model=None):
        with (directory / "config.json").open(encoding="utf-8") as handle:
            serialized = json.load(handle)
        # save_pretrained serializes a diff: restore model-family defaults so
        # omitted fields do not bypass checks or falsely reject older sources.
        saved = type(self.model.config).from_dict(serialized).to_dict()
        # Some config classes replace explicit model_type during to_dict().
        # Defaults may fill missing keys, but never erase conflicting evidence.
        saved.update(serialized)
        # Transformers 4.51 save_pretrained deliberately strips _name_or_path.
        # Recover it only from the source signature already compared above;
        # never invent provenance from the current run or ignore an explicit
        # serialized identity that conflicts with the source/current model.
        if "_name_or_path" not in serialized and "name_or_path" not in serialized:
            if signed_model is None or "_name_or_path" not in signed_model:
                raise ValueError("Stage-1 source config omits model identity and has no verified signed identity.")
            saved["_name_or_path"] = signed_model["_name_or_path"]
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
            raise ValueError("Stage-1 source frozen-model/preprocessing provenance mismatch: " + ", ".join(different))

    def load_checkpoint(self, directory):
        directory = Path(directory)
        required = (STATE_NAME, "mm_projector.bin", "optimizer.pt", "scheduler.pt",
                    "rng_state.pth", "trainer_state.json", "config.json")
        missing = [name for name in required if not (directory / name).is_file()]
        if missing:
            raise ValueError("Incomplete protected checkpoint (full rollback state required): " + ", ".join(missing))
        payload = _load(directory / STATE_NAME)
        if (payload.get("version") != 1 or
                self._resume_signature(payload["signature"]) != self._resume_signature(self.signature)):
            raise ValueError("Protected checkpoint/config/data mismatch; exact resume requires the same run setup.")
        weights = _load(directory / "mm_projector.bin")
        self.projector.load_state_dict({key.split("mm_projector.", 1)[-1]: value for key, value in weights.items()})
        for name in ("phase", "phase_b_start", "ce_ref", "ce_grad_ref", "aux_ratio", "retries", "finished_reason"):
            setattr(self, name, payload[name])
        self.transition_reason = payload.get("transition_reason")
        self.source_provenance = payload.get("source_provenance")
        self.refinement_accepted = payload.get("refinement_accepted", False)
        self.ema = None if payload["ema"] is None else payload["ema"].to(self.params[0].device)
        if payload["reference"] is not None:
            if self.reference is None:
                self.reference = copy.deepcopy(self.projector).requires_grad_(False).eval()
            self.reference.load_state_dict(payload["reference"])
        else:
            self.reference = None
        self.monitor.__dict__.update(payload["monitor"])
        self.accum_main = self.accum_cka = None
        self.accum_count, self.step_logs = 0, {}
        self.accepted_checkpoint, self.loaded = str(directory), True

    def save_checkpoint(self, name=None):
        if self.accum_count:
            raise RuntimeError("Protected checkpoints must be written at effective-batch boundaries.")
        root = Path(self.args.output_dir)
        root.mkdir(parents=True, exist_ok=True)
        destination = root / (name or f"checkpoint-{self.trainer.state.global_step}")
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite existing checkpoint {destination}")
        temporary = Path(tempfile.mkdtemp(prefix=".protected-checkpoint-", dir=root))
        self.model.config.save_pretrained(temporary)
        torch.save({"model.mm_projector." + key: value.detach().cpu().clone()
                    for key, value in self.projector.state_dict().items()}, temporary / "mm_projector.bin")
        torch.save(self.state_dict(), temporary / STATE_NAME)
        torch.save(self.args, temporary / "training_args.bin")
        split_directory = root / "protected_cka_split"
        if split_directory.is_dir():
            # The index/fingerprint manifest is sufficient to reconstruct and
            # validate both splits. Do not duplicate the much larger human-
            # readable image manifests in every optimizer checkpoint.
            saved_split = temporary / "protected_cka_split"
            saved_split.mkdir()
            shutil.copyfile(split_directory / "split.json", saved_split / "split.json")
        self.trainer._save_optimizer_and_scheduler(str(temporary))
        self.trainer._save_scaler(str(temporary))
        self.trainer._save_rng_state(str(temporary))
        self.trainer.state.save_to_json(str(temporary / "trainer_state.json"))
        os.replace(temporary, destination)
        return str(destination)

    def run(self, parent_train, resume_from_checkpoint=None, **kwargs):
        source = self.option("source_stage1_checkpoint")
        if source and resume_from_checkpoint:
            raise ValueError("Choose a new Stage-1 CE-only source continuation OR exact resume, not both.")
        if source:
            # No source optimizer/RNG/plateau/finished horizon is imported.
            # Provenance checks complete before Trainer can step an optimizer.
            self.load_source_checkpoint(source)
        if resume_from_checkpoint is True:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError("No protected checkpoint to resume.")
        if resume_from_checkpoint and not os.path.isfile(os.path.join(resume_from_checkpoint, STATE_NAME)):
            raise ValueError("Protected resume requires a full protected checkpoint. For CE-only weights use "
                             "--protected_cka_source_stage1_checkpoint in a new output directory.")
        if resume_from_checkpoint:
            saved = _load(Path(resume_from_checkpoint) / STATE_NAME)
            state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, "trainer_state.json"))
            if saved.get("finished_reason") or state.global_step >= self.args.max_steps:
                raise ValueError("This protected run is already finished. Start a new run from its safe "
                                 "--protected_cka_source_stage1_checkpoint instead of extending its horizon.")
        while True:
            self.pending_rollback = False
            result = parent_train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)
            # Also guard a graceful stop requested by an external callback
            # between validation intervals: never export an unchecked student.
            if not self.pending_rollback:
                accepted_state = TrainerState.load_from_json(
                    os.path.join(self.accepted_checkpoint, "trainer_state.json"))
                if accepted_state.global_step != self.trainer.state.global_step:
                    self.step_end(self.trainer.control, force=True)
            if not self.pending_rollback:
                self.trainer.log({"protected/" + key: value for key, value in self.budget_metrics().items()})
                return result
            reduced_ratio = self.aux_ratio * self.option("aux_backoff")
            failures = self.retries + 1
            checkpoint = self.accepted_checkpoint
            # Restore everything even when there will be no retry. In particular,
            # train.py's final adapter save must never export the failed student.
            self.load_checkpoint(checkpoint)
            self.trainer._load_optimizer_and_scheduler(checkpoint)
            self.trainer._load_scaler(checkpoint)
            self.trainer.state = TrainerState.load_from_json(os.path.join(checkpoint, "trainer_state.json"))
            self.trainer._load_rng_state(checkpoint)
            self.projector.zero_grad(set_to_none=True)
            self.aux_ratio, self.retries = reduced_ratio, failures
            exhausted = failures > self.option("max_retries")
            self.finished_reason = "gate_retry_exhausted" if exhausted else None
            _atomic_save(self.state_dict(), Path(checkpoint) / STATE_NAME)
            self.trainer.log({"protected/gate_failures": failures, "protected/rollbacks": failures,
                              "protected/aux_ratio": self.aux_ratio,
                              "protected/safe_step": self.trainer.state.global_step,
                              "protected/refinement_accepted": int(self.refinement_accepted),
                              "protected/retries_exhausted": int(exhausted)})
            if exhausted:
                budget = self.budget_metrics()
                self.trainer.log({"protected/" + key: value for key, value in budget.items()})
                metrics = dict(result.metrics, protected_retry_exhausted=1,
                               protected_safe_step=self.trainer.state.global_step,
                               **{"protected_" + key: value for key, value in budget.items()})
                return TrainOutput(self.trainer.state.global_step, result.training_loss, metrics)
            resume_from_checkpoint = checkpoint
            # Recreate clean native optimizer/scheduler wrappers on the next HF
            # loop; their exact state will be loaded from the accepted snapshot.
            self.trainer.optimizer = self.trainer.lr_scheduler = None
            self.trainer._created_lr_scheduler = False
            self.trainer.model = self.trainer.accelerator.unwrap_model(
                self.trainer.model, keep_fp32_wrapper=False)
            self.trainer.model_wrapped = self.trainer.model
