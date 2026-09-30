"""CPU-only tests for adaptive projector PCGrad's ZeRO-2 integration helpers."""

from __future__ import annotations

import os
import tempfile
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from transformers import TrainingArguments

from llava.train.adaptive_projector_pcgrad import (
    AdaptiveProjectorPCGradConfig,
    AdaptiveProjectorPCGradController,
    METADATA_FILE,
    STATE_FILE,
    projector_sha256,
    projector_signature,
    write_json,
)
from llava.train.llava_trainer import LLaVATrainer


_INVALID_REASONS = (
    "too_few_patches",
    "nonfinite_input",
    "nonfinite_gram",
    "zero_gram_norm",
    "score_out_of_range",
    "score_roundoff_clamped",
)


class _FakeReplay:
    def __init__(self, flat_sum, cka_sum, valid_count):
        self._flat_sum = flat_sum
        self._cka_sum = cka_sum
        self._valid_count = valid_count

    def consume_flat(self):
        device = self._flat_sum.device
        invalid = {
            name: torch.zeros((), dtype=torch.float64, device=device)
            for name in _INVALID_REASONS
        }
        return self._flat_sum, self._cka_sum, self._valid_count, invalid


class _FakeZeroOptimizer:
    pass


class _LifecycleReplay:
    def __init__(self):
        self.snapshot_token = None
        self.add_calls = 0

    def begin_window(self, stage, optimizer_step):
        self.snapshot_token = (stage, optimizer_step)

    def add(self, captured, *, loss_scale):
        self.add_calls += 1
        self.captured = captured
        self.loss_scale = loss_scale


class _LifecycleEngine:
    def __init__(self):
        self.boundaries = []
        self.backward_calls = 0
        self.step_calls = 0

    def set_gradient_accumulation_boundary(self, value):
        self.boundaries.append(bool(value))

    def backward(self, loss):
        self.backward_calls += 1
        loss.backward()

    def step(self):
        self.step_calls += 1

    def was_step_applied(self):
        return False


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(adaptive_projector_pcgrad=False)
        self.mm_projector = nn.Sequential(nn.Linear(3, 4), nn.GELU(), nn.Linear(4, 2))
        # Its name intentionally contains "projector". Exact identity grouping
        # must still leave it in the LLM/non-projector groups.
        self.projector_shadow = nn.Linear(2, 2)
        self.decoder = nn.Linear(2, 1)

    def forward(self, input_ids=None, **kwargs):
        del kwargs
        value = torch.zeros(1, 3) if input_ids is None else input_ids.float()
        hidden = self.mm_projector(value)
        return {"loss": self.decoder(self.projector_shadow(hidden)).sum()}


def _bare_trainer(projector):
    trainer = object.__new__(LLaVATrainer)
    parameters = tuple(parameter for parameter in projector.parameters() if parameter.requires_grad)
    trainer._adaptive_pcgrad_projector_parameters = parameters
    trainer._adaptive_pcgrad_projector_parameter_ids = {id(parameter) for parameter in parameters}
    return trainer


def _controller_config():
    return AdaptiveProjectorPCGradConfig(
        stage=2,
        planned_optimizer_steps=4,
        max_aux_ratio=0.5,
        warmup_ratio=0.0,
        norm_ema_beta=0.9,
        lambda_max=10.0,
        ce_norm_floor=1e-12,
        aux_norm_floor=1e-12,
        residual_rtol=1e-6,
    )


def _distributed_backend_worker(rank, world_size, init_file, output_dir):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        projector = nn.Linear(6, 1, bias=True)  # 7 parameters -> one padded slot.
        trainer = _bare_trainer(projector)

        global_main = torch.tensor([1.0, -2.0, 0.5, 3.0, -1.0, 2.0, 0.25, 0.0])
        rank0_aux_sum = torch.tensor([-3.0, 0.0, 1.5, -1.5, 3.0, 0.0, 0.75])
        local_aux_sum = rank0_aux_sum.clone() if rank == 0 else torch.zeros_like(rank0_aux_sum)
        local_count = 3.0 if rank == 0 else 0.0
        trainer._adaptive_pcgrad_replay = _FakeReplay(
            local_aux_sum,
            torch.tensor(1.2 if rank == 0 else 0.0, dtype=torch.float64),
            torch.tensor(local_count, dtype=torch.float64),
        )
        trainer._adaptive_pcgrad_ce_sum = torch.tensor(4.0 + rank, dtype=torch.float64)
        trainer._adaptive_pcgrad_ce_count = torch.tensor(2.0, dtype=torch.float64)

        partition_size = 4
        start = rank * partition_size
        local_main = global_main.narrow(0, start, partition_size).clone()
        zero = _FakeZeroOptimizer()
        zero.dp_process_group = dist.group.WORLD
        zero.real_dp_process_group = [dist.group.WORLD]
        zero.round_robin_bit16_groups = [list(projector.parameters())]
        zero.partition_size = [partition_size]
        # Split the local partition to exercise the list-of-gradient-parts path.
        zero.averaged_gradients = {0: [local_main[:1], local_main[1:]]}

        global_auxiliary, stats, process_group = trainer._adaptive_global_auxiliary(zero)
        main_parts, auxiliary_parts, destinations, shard_group = (
            trainer._adaptive_zero2_projector_parts(zero, global_auxiliary)
        )
        assert process_group is shard_group

        controller = AdaptiveProjectorPCGradController(_controller_config())
        merged, logs = controller.prepare(
            main_parts,
            auxiliary_parts,
            valid_count=float(stats["valid_count"].item()),
            reference_tensor=main_parts[0],
            distributed_shards=True,
            process_group=process_group,
        )
        local_merged = torch.cat([part.reshape(-1) for part in merged])
        gathered = [torch.empty_like(local_merged) for _ in range(world_size)]
        dist.all_gather(gathered, local_merged)

        payload = {
            "global_auxiliary": global_auxiliary.cpu(),
            "global_merged": torch.cat(gathered).cpu(),
            "stats": {name: float(value.item()) for name, value in stats.items()},
            "logs": logs,
        }
        torch.save(payload, os.path.join(output_dir, f"rank-{rank}.pt"))
    finally:
        dist.destroy_process_group()


class Zero2ProjectorShardMappingTests(unittest.TestCase):
    def test_full_auxiliary_vector_maps_exactly_to_rank_local_partitions(self):
        projector = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
        trainer = _bare_trainer(projector)
        flat = torch.arange(1, 10, dtype=torch.float32)
        weight_group = [projector[0].weight, projector[1].weight]  # 6 values
        bias_group = [projector[0].bias, projector[1].bias]  # 3 values + padding

        for rank, expected_weight, expected_bias in (
            (0, [1.0, 2.0, 3.0], [5.0, 6.0]),
            (1, [4.0, 7.0, 8.0], [9.0, 0.0]),
        ):
            with self.subTest(rank=rank):
                zero = _FakeZeroOptimizer()
                group = object()
                zero.round_robin_bit16_groups = [weight_group, bias_group]
                zero.real_dp_process_group = [group, group]
                zero.partition_size = [3, 2]
                zero.averaged_gradients = {
                    0: [torch.zeros(1), torch.zeros(2)],
                    1: [torch.zeros(2)],
                }
                with mock.patch("torch.distributed.get_rank", return_value=rank):
                    _, auxiliary, destinations, returned_group = (
                        trainer._adaptive_zero2_projector_parts(zero, flat)
                    )

                self.assertIs(returned_group, group)
                self.assertEqual(len(destinations), 3)
                torch.testing.assert_close(
                    torch.cat([part.reshape(-1) for part in auxiliary[:2]]),
                    torch.tensor(expected_weight),
                )
                torch.testing.assert_close(
                    auxiliary[2].reshape(-1), torch.tensor(expected_bias)
                )

    def test_mixed_projector_and_decoder_group_is_rejected(self):
        projector = nn.Linear(2, 2)
        decoder_parameter = nn.Parameter(torch.ones(1))
        trainer = _bare_trainer(projector)
        zero = _FakeZeroOptimizer()
        zero.round_robin_bit16_groups = [[projector.weight, decoder_parameter]]
        zero.real_dp_process_group = [object()]
        zero.partition_size = [3]
        zero.averaged_gradients = {0: [torch.zeros(3)]}

        with self.assertRaisesRegex(RuntimeError, "Mixed ZeRO optimizer group"):
            trainer._adaptive_zero2_projector_parts(zero, torch.zeros(6))


class DeepSpeedAccumulationLifecycleTests(unittest.TestCase):
    def test_non_boundary_microbatch_still_advances_engine_lifecycle(self):
        trainer = object.__new__(LLaVATrainer)
        trainer.model = SimpleNamespace(
            config=SimpleNamespace(adaptive_pcgrad_profile=False)
        )
        trainer.args = SimpleNamespace(n_gpu=1, gradient_accumulation_steps=4)
        trainer.accelerator = SimpleNamespace(sync_gradients=False)
        trainer._adaptive_pcgrad_ce_sum = None
        trainer._adaptive_pcgrad_ce_count = None
        trainer._adaptive_pcgrad_window_start = None
        trainer._adaptive_pcgrad_replay = _LifecycleReplay()

        controller = SimpleNamespace(
            config=SimpleNamespace(stage=2),
            successful_steps=3,
        )
        engine = _LifecycleEngine()
        parameter = torch.tensor(2.0, requires_grad=True)
        trainer._get_adaptive_pcgrad_controller = lambda: controller
        trainer._validate_adaptive_pcgrad_backend = lambda model: (engine, object())
        trainer._set_model_attr = lambda model, name, value: None
        trainer._find_model_attr = lambda model, name: []
        trainer.compute_loss_context_manager = nullcontext
        trainer.compute_loss = lambda model, inputs: parameter.square()

        returned = trainer._adaptive_training_step(object(), {})

        self.assertEqual(engine.boundaries, [False])
        self.assertEqual(engine.backward_calls, 1)
        self.assertEqual(engine.step_calls, 1)
        self.assertEqual(trainer._adaptive_pcgrad_replay.add_calls, 1)
        self.assertEqual(trainer._adaptive_pcgrad_replay.snapshot_token, (2, 3))
        torch.testing.assert_close(returned, torch.tensor(1.0))
        torch.testing.assert_close(parameter.grad, torch.tensor(4.0))


class DeepSpeedResumeLifecycleTests(unittest.TestCase):
    def test_optimizer_load_hook_restores_adaptive_artifacts(self):
        trainer = object.__new__(LLaVATrainer)
        trainer.model = SimpleNamespace(
            config=SimpleNamespace(adaptive_projector_pcgrad=True)
        )
        with mock.patch(
            "transformers.Trainer._load_optimizer_and_scheduler"
        ) as base_load, mock.patch.object(
            trainer, "_load_adaptive_pcgrad_resume_artifacts"
        ) as adaptive_load:
            trainer._load_optimizer_and_scheduler("checkpoint-3")

        base_load.assert_called_once_with("checkpoint-3")
        adaptive_load.assert_called_once_with("checkpoint-3")

    def test_resume_restores_exact_controller_state_after_model_load(self):
        projector = nn.Linear(3, 2)
        config = _controller_config()
        source = AdaptiveProjectorPCGradController(config)
        main = [torch.tensor([1.0, 2.0])]
        auxiliary = [torch.tensor([-0.5, 1.0])]
        source.prepare(
            main,
            auxiliary,
            valid_count=1.0,
            reference_tensor=main[0],
        )
        source.finish(optimizer_stepped=True)

        trainer = object.__new__(LLaVATrainer)
        trainer.model = SimpleNamespace(
            config=SimpleNamespace(
                adaptive_projector_pcgrad=True,
                adaptive_projector_pcgrad_config={
                    key: value
                    for key, value in config.to_dict().items()
                    if key not in {"planned_optimizer_steps", "warmup_steps"}
                },
                adaptive_pcgrad_base_model_identifier="unit-test-model",
                adaptive_pcgrad_stage1_projector_sha256="stage1-parent",
            )
        )
        trainer.state = SimpleNamespace(max_steps=config.planned_optimizer_steps)
        trainer._adaptive_pcgrad_projector = projector
        trainer._adaptive_pcgrad_controller = None
        trainer._adaptive_pcgrad_resume_state = None
        trainer._adaptive_pcgrad_resume_checkpoint = None

        with tempfile.TemporaryDirectory() as checkpoint:
            write_json(
                os.path.join(checkpoint, METADATA_FILE),
                {
                    "stage": config.stage,
                    "base_model_identifier": "unit-test-model",
                    "projector_signature": projector_signature(projector),
                    "projector_sha256": projector_sha256(projector),
                    "parent_projector_sha256": "stage1-parent",
                    "resolved_config": source.state_dict()["config"],
                    "successful_steps": source.successful_steps,
                },
            )
            torch.save(source.state_dict(), os.path.join(checkpoint, STATE_FILE))
            trainer._load_adaptive_pcgrad_resume_artifacts(checkpoint)

        self.assertEqual(
            trainer._adaptive_pcgrad_controller.state_dict(),
            source.state_dict(),
        )


class DistributedAdaptiveBackendTests(unittest.TestCase):
    def test_two_rank_global_mean_and_sharded_merge_match_single_process(self):
        if not dist.is_available() or not dist.is_gloo_available():
            self.skipTest("torch.distributed gloo is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            init_file = os.path.join(directory, "gloo-init")
            mp.spawn(
                _distributed_backend_worker,
                args=(2, init_file, directory),
                nprocs=2,
                join=True,
            )
            rank0 = torch.load(os.path.join(directory, "rank-0.pt"), weights_only=True)
            rank1 = torch.load(os.path.join(directory, "rank-1.pt"), weights_only=True)

        expected_auxiliary = torch.tensor([-1.0, 0.0, 0.5, -0.5, 1.0, 0.0, 0.25])
        for payload in (rank0, rank1):
            torch.testing.assert_close(payload["global_auxiliary"], expected_auxiliary)
            self.assertEqual(payload["stats"]["valid_count"], 3.0)
            self.assertAlmostEqual(payload["stats"]["cka_sum"], 1.2)
            self.assertEqual(payload["stats"]["ce_sum"], 9.0)
            self.assertEqual(payload["stats"]["ce_count"], 4.0)

        global_main = torch.tensor([1.0, -2.0, 0.5, 3.0, -1.0, 2.0, 0.25, 0.0])
        padded_auxiliary = torch.cat([expected_auxiliary, torch.zeros(1)])
        reference = AdaptiveProjectorPCGradController(_controller_config())
        expected_merged, expected_logs = reference.prepare(
            [global_main],
            [padded_auxiliary],
            valid_count=3.0,
            reference_tensor=global_main,
        )
        torch.testing.assert_close(rank0["global_merged"], expected_merged[0])
        torch.testing.assert_close(rank1["global_merged"], expected_merged[0])
        for name in (
            "ce_norm",
            "raw_cka_norm",
            "raw_cosine",
            "lambda",
            "projected_norm",
            "effective_aux_ratio",
            "post_projection_dot",
        ):
            self.assertAlmostEqual(rank0["logs"][name], expected_logs[name], places=6)
            self.assertAlmostEqual(rank1["logs"][name], expected_logs[name], places=6)


class ExactProjectorOptimizerGroupingTests(unittest.TestCase):
    def test_adaptive_groups_use_module_identity_not_name_substrings(self):
        model = _TinyModel()
        with tempfile.TemporaryDirectory() as output_dir:
            args = TrainingArguments(output_dir=output_dir, report_to=[], use_cpu=True)
            args.mm_projector_lr = None
            trainer = LLaVATrainer(model=model, args=args)
            model.config.adaptive_projector_pcgrad = True
            optimizer = trainer.create_optimizer()

        projector_ids = {id(parameter) for parameter in model.mm_projector.parameters()}
        shadow_ids = {id(parameter) for parameter in model.projector_shadow.parameters()}
        grouped_projector_ids = {
            id(parameter)
            for group in optimizer.param_groups
            if group.get("vsp_group") == "projector"
            for parameter in group["params"]
        }
        grouped_llm_ids = {
            id(parameter)
            for group in optimizer.param_groups
            if group.get("vsp_group") == "llm"
            for parameter in group["params"]
        }
        self.assertEqual(grouped_projector_ids, projector_ids)
        self.assertTrue(shadow_ids <= grouped_llm_ids)
        self.assertTrue(projector_ids.isdisjoint(grouped_llm_ids))


if __name__ == "__main__":
    unittest.main()
