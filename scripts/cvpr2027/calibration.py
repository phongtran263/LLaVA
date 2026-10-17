"""Gradient-strength calibration on fixed image-containing training batches."""
import csv
import json
import math
import os
from pathlib import Path
import random
import statistics
from contextlib import nullcontext


def calibration_autocast(trainer):
    """The save-free probe bypasses Accelerate.prepare_model's AMP wrapper."""
    import torch

    device_type = torch.device(trainer.args.device).type
    if getattr(trainer.args, "bf16", False):
        return torch.autocast(device_type=device_type, dtype=torch.bfloat16)
    if getattr(trainer.args, "fp16", False):
        return torch.autocast(device_type=device_type, dtype=torch.float16)
    return nullcontext()


def match_strength(rows, original_coefficient):
    """Match the *combined vector* at the projector, not the sum of its norms."""
    usable = [r for r in rows if r["projector_raw_norm"] > 1e-12]
    if not usable:
        raise ValueError("No nonzero projector CKA gradients; no coefficient can be calibrated.")
    coefficients = [r["full_projector_norm"] / r["projector_raw_norm"] for r in usable]
    coefficient = statistics.median(coefficients)
    if not math.isfinite(coefficient) or coefficient <= 0:
        raise ValueError("Full regularizer has zero/nonfinite projector gradient strength.")
    ratios = [coefficient * r["projector_raw_norm"] / r["full_projector_norm"]
              for r in usable if r["full_projector_norm"] > 1e-12]
    return {"recommended_lambda_proj": coefficient,
            "original_lambda_proj": original_coefficient,
            "is_stronger_than_original": coefficient > original_coefficient,
            "valid_batches": len(usable), "batchwise_matching_coefficients": coefficients,
            "achieved_to_full_projector_norm_ratios": ratios,
            "matching_scope": "trainable mm_projector parameters, at shared stage-1 initialization",
            "selection_rule": "median(norm(lambda_proj*g_proj + lambda_final*g_final) / norm(g_proj))",
            "limitation": "Characterizes projector strength at initialization; does not match decoder gradients "
                          "or guarantee equality later in training. Image-containing batches only: native projector "
                          "CKA also includes dummy image features for text-only examples. Calibration runs unsharded "
                          "without Trainer activation checkpointing. No benchmark scores used."}


def calibrate(trainer, log_dir):
    import torch
    from transformers import set_seed

    log_dir = Path(log_dir)
    seed = int(os.environ.get("CVPR_CALIBRATION_SEED", 42))
    batch_count = int(os.environ.get("CVPR_CALIBRATION_BATCHES", 8))
    batch_size = int(os.environ.get("CVPR_CALIBRATION_BATCH_SIZE", 1))
    model = trainer.model
    model.train()
    config = model.config
    p, f = float(config.cka_loss_projector_weight), float(config.cka_loss_final_hidden_weight)
    if p <= 0 or f <= 0 or getattr(config, "cka_loss_geometry", "linear") != "linear":
        raise ValueError("Calibrate the original Full linear-CKA objective with both positive coefficients.")
    dataset = trainer.train_dataset
    samples = getattr(dataset, "list_data_dict", None)
    if samples is None:
        raise ValueError("Calibration requires the native LazySupervisedDataset sample metadata.")
    candidates = [i for i, sample in enumerate(samples) if sample.get("image")]
    count = batch_count * batch_size
    if len(candidates) < count:
        raise ValueError(f"Need {count} distinct image examples; dataset has {len(candidates)}.")
    selected = random.Random(seed).sample(candidates, count)
    projector = [param for name, param in model.named_parameters()
                 if "mm_projector" in name and param.requires_grad]
    final_block = [param for param in model.get_model().layers[-1].parameters() if param.requires_grad]
    if not projector or not final_block:
        raise ValueError("Full stage-2 calibration needs trainable projector and decoder.")
    vision = list(model.get_vision_tower().parameters())
    if any(param.requires_grad for param in vision):
        raise ValueError("Vision encoder must be frozen.")
    config.use_cache = False
    rows = []

    def gradients(loss, parameters):
        if loss is None or not loss.requires_grad:
            return [None] * len(parameters)
        return list(torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True))

    def norm(values):
        squares = [value.detach().float().square().sum() for value in values if value is not None]
        value = torch.stack(squares).sum().sqrt().item() if squares else 0.0
        if not math.isfinite(value):
            raise ValueError("Nonfinite regularizer gradient")
        return value

    for index in range(batch_count):
        set_seed(seed + index)
        indices = selected[index * batch_size:(index + 1) * batch_size]
        batch = trainer._prepare_inputs(trainer.data_collator([dataset[i] for i in indices]))
        model.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        with trainer.compute_loss_context_manager(), calibration_autocast(trainer):
            output = model(**batch)
        proj_loss = output.projector_cka_loss
        final_loss = sum(output.aux_losses)
        if not all(torch.isfinite(loss).item() for loss in (output.loss, proj_loss, final_loss)):
            raise ValueError("Nonfinite CE/CKA loss")
        # Keep two relatively small projector gradient vectors to measure cancellation.
        gp = gradients(proj_loss, projector)
        gf = gradients(final_loss, projector)
        full = [(a.float() if a is not None else torch.zeros_like(param, dtype=torch.float32)) +
                (b.float() if b is not None else torch.zeros_like(param, dtype=torch.float32))
                for a, b, param in zip(gp, gf, projector)]
        np, nf, nfull = norm(gp), norm(gf), norm(full)
        dot = sum((a.detach().float() * b.detach().float()).sum().item()
                  for a, b in zip(gp, gf) if a is not None and b is not None)
        del gp, gf, full
        # Report a concrete decoder scope. Projector-only has no direct decoder gradient.
        decoder_norm = norm(gradients(final_loss, final_block))
        if nf <= 0 or decoder_norm <= 0:
            raise ValueError("Final CKA did not reach the trainable projector/final decoder block.")
        if any(param.grad is not None for param in vision):
            raise ValueError("Frozen vision encoder unexpectedly received gradients.")
        rows.append({"batch": index, "sample_indices": ";".join(map(str, indices)),
                     "ce_loss": output.loss.item(), "projector_raw_loss": proj_loss.item() / p,
                     "final_raw_loss": final_loss.item() / f,
                     "projector_raw_norm": np / p, "final_raw_projector_norm": nf / f,
                     "weighted_proj_projector_norm": np, "weighted_final_projector_norm": nf,
                     "full_projector_norm": nfull,
                     "projector_term_cosine": dot / (np * nf) if np * nf > 0 else 0,
                     "weighted_final_last_decoder_block_norm": decoder_norm,
                     "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0})
        trainer._clear_gradient_log_tensors(model)
        del output, proj_loss, final_loss, batch
    with (log_dir / "gradient_norms.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    samples_record = {"seed": seed, "sampling": "uniform without replacement among image-containing training examples",
                      "indices": selected, "sample_ids": [samples[i].get("id", "") for i in selected],
                      "images": [samples[i]["image"] for i in selected]}
    (log_dir / "calibration_samples.json").write_text(json.dumps(samples_record, indent=2) + "\n")
    result = match_strength(rows, p)
    result.update(seed=seed, lambda_final=f,
                  calibration_identity_sha256=os.environ["CVPR_CALIBRATION_ID"],
                  stage1_sha256=os.environ.get("CVPR_STAGE1_SHA256", ""),
                  data_sha256=os.environ.get("CVPR_DATA_SHA256", ""),
                  decoder_measurement_scope="last decoder block parameters (depth read from model)")
    (log_dir / "strength_match.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
