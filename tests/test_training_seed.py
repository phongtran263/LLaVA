"""Seed all RNGs before model initialization, without downloading models."""
import importlib
import json
import random
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from peft import LoraConfig, get_peft_model

from llava.model.multimodal_projector.builder import build_vision_projector
from llava.train.llava_trainer import LengthGroupedSampler
from llava.train.train import DataArguments, ModelArguments, TrainingArguments
from tests.test_training_compatibility import make_model


training = importlib.import_module("llava.train.train")


class StopBeforeDownload(Exception):
    pass


class TrainingSeedTests(unittest.TestCase):
    def _initialize(self, seed=42, *, pretrain=False, subset_seed=None, capture=None):
        model_args = ModelArguments(tune_mm_mlp_adapter=pretrain)
        data_args = DataArguments(train_data_seed=subset_seed)
        args = SimpleNamespace(
            seed=seed, local_rank=-1, stop_after_step_ratio=None,
            save_at_step_ratio=None, gradient_checkpointing=False,
            fp16=False, bf16=False, cache_dir=None,
        )
        observed = {}

        def before_download(*_args, **_kwargs):
            observed['python'] = random.random()
            observed['numpy'] = np.random.rand(8)
            observed['torch'] = torch.rand(8)
            if capture is not None:
                observed['capture'] = capture()
            raise StopBeforeDownload

        with mock.patch.object(training.transformers, 'HfArgumentParser') as parser, \
             mock.patch.object(training.transformers.AutoConfig, 'from_pretrained', side_effect=before_download), \
             mock.patch('torch.cuda.manual_seed_all') as cuda_seed:
            parser.return_value.parse_args_into_dataclasses.return_value = (model_args, data_args, args)
            with self.assertRaises(StopBeforeDownload):
                training.train()
            cuda_seed.assert_any_call(seed)
        return observed, data_args

    def test_both_stages_seed_all_rngs_before_loading_model(self):
        for pretrain in (True, False):
            with self.subTest(pretrain=pretrain):
                first, _ = self._initialize(123, pretrain=pretrain)
                random.random()
                np.random.rand(50)
                torch.rand(50)
                second, _ = self._initialize(123, pretrain=pretrain)
                self.assertEqual(first['python'], second['python'])
                np.testing.assert_array_equal(first['numpy'], second['numpy'])
                torch.testing.assert_close(first['torch'], second['torch'], atol=0, rtol=0)
                different, _ = self._initialize(124, pretrain=pretrain)
                self.assertNotEqual(first['python'], different['python'])
                self.assertFalse(np.array_equal(first['numpy'], different['numpy']))
                self.assertFalse(torch.equal(first['torch'], different['torch']))

    def test_all_projector_types_have_repeatable_initial_weights(self):
        for kind in ('linear', 'mlp2x_gelu', 'coupling2x_gelu'):
            with self.subTest(projector=kind):
                def capture():
                    model = build_vision_projector(SimpleNamespace(
                        mm_projector_type=kind, mm_hidden_size=8, hidden_size=16,
                    ))
                    return torch.cat([p.detach().flatten() for p in model.parameters()])
                a, _ = self._initialize(42, capture=capture)
                b, _ = self._initialize(42, capture=capture)
                c, _ = self._initialize(43, capture=capture)
                torch.testing.assert_close(a['capture'], b['capture'], atol=0, rtol=0)
                self.assertFalse(torch.equal(a['capture'], c['capture']))

    def test_lora_initialization_is_repeatable(self):
        def capture():
            model = get_peft_model(make_model(cka=False), LoraConfig(
                r=2, lora_alpha=4, target_modules=['q_proj'], lora_dropout=0.0,
            ))
            return torch.cat([p.detach().flatten() for n, p in model.named_parameters() if 'lora_A' in n])
        a, _ = self._initialize(42, capture=capture)
        b, _ = self._initialize(42, capture=capture)
        c, _ = self._initialize(43, capture=capture)
        torch.testing.assert_close(a['capture'], b['capture'], atol=0, rtol=0)
        self.assertFalse(torch.equal(a['capture'], c['capture']))

    def test_modality_sampler_order_is_repeatable(self):
        def capture():
            return list(LengthGroupedSampler(
                batch_size=2, world_size=1, lengths=[10, -5, 20, -8] * 20,
                group_by_modality=True,
            ))
        a, _ = self._initialize(42, capture=capture)
        b, _ = self._initialize(42, capture=capture)
        c, _ = self._initialize(43, capture=capture)
        self.assertEqual(a['capture'], b['capture'])
        self.assertNotEqual(a['capture'], c['capture'])

    def test_subset_seed_inherits_run_seed_and_preserves_explicit_zero(self):
        _, inherited = self._initialize(123)
        _, explicit = self._initialize(123, subset_seed=0)
        self.assertEqual(inherited.train_data_seed, 123)
        self.assertEqual(explicit.train_data_seed, 0)

    def test_dataset_subset_uses_resolved_seed(self):
        def subset(seed):
            _, args = self._initialize(seed)
            args.train_data_fraction = 0.25
            rows = [{'id': i} for i in range(80)]
            with mock.patch('builtins.open', mock.mock_open(read_data=json.dumps(rows))):
                dataset = training.LazySupervisedDataset('unused.json', None, args)
            return [row['id'] for row in dataset.list_data_dict]
        self.assertEqual(subset(12), subset(12))
        self.assertNotEqual(subset(12), subset(13))

    def test_default_seed_and_determinism_policy_are_unchanged(self):
        self.assertEqual(TrainingArguments.__dataclass_fields__['seed'].default, 42)
        self.assertFalse(TrainingArguments.__dataclass_fields__['full_determinism'].default)
        original = torch.are_deterministic_algorithms_enabled()
        self._initialize(42)
        self.assertEqual(torch.are_deterministic_algorithms_enabled(), original)


if __name__ == '__main__':
    unittest.main()
