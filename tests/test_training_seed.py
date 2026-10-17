"""Keep original Trainer-controlled seeding after model initialization."""
import importlib
import json
import random
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from llava.model.multimodal_projector.builder import build_vision_projector
from llava.train.llava_trainer import LLaVATrainer, LengthGroupedSampler
from llava.train.train import DataArguments, ModelArguments, TrainingArguments
from tests.test_training_compatibility import make_model


training = importlib.import_module("llava.train.train")


class StopBeforeDownload(Exception):
    pass


class TrainingSeedTests(unittest.TestCase):
    @staticmethod
    def _ambient_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    @staticmethod
    def _rng_sample():
        return dict(python=random.random(), numpy=np.random.rand(8), torch=torch.rand(8))

    def _assert_same_sample(self, first, second):
        self.assertEqual(first['python'], second['python'])
        np.testing.assert_array_equal(first['numpy'], second['numpy'])
        torch.testing.assert_close(first['torch'], second['torch'], atol=0, rtol=0)

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
            observed.update(self._rng_sample())
            if capture is not None:
                observed['capture'] = capture()
            raise StopBeforeDownload

        with mock.patch.object(training.transformers, 'HfArgumentParser') as parser, \
             mock.patch.object(training.transformers.AutoConfig, 'from_pretrained', side_effect=before_download):
            parser.return_value.parse_args_into_dataclasses.return_value = (model_args, data_args, args)
            with self.assertRaises(StopBeforeDownload):
                training.train()
        return observed, data_args

    def _trainer_initialize(self, seed, ambient_seed, *, pretrain=False, capture=None):
        self._ambient_seed(ambient_seed)
        model = make_model(cka=False, pretrain=pretrain)
        model.get_model().mm_projector = build_vision_projector(SimpleNamespace(
            mm_projector_type='mlp2x_gelu', mm_hidden_size=6, hidden_size=16,
        ))
        initial = {name: value.detach().clone()
                   for name, value in model.get_model().mm_projector.named_parameters()}
        with tempfile.TemporaryDirectory() as output:
            args = TrainingArguments(output_dir=output, seed=seed, use_cpu=True,
                                     report_to=[], disable_tqdm=True)
            LLaVATrainer(model=model, args=args)
            for name, parameter in model.get_model().mm_projector.named_parameters():
                torch.testing.assert_close(parameter, initial[name], atol=0, rtol=0)
            observed = self._rng_sample()
            if capture is not None:
                observed['capture'] = capture()
        observed['initial_projector'] = torch.cat([value.flatten() for value in initial.values()])
        return observed

    def test_both_stages_keep_ambient_rng_until_model_loading(self):
        for pretrain in (True, False):
            with self.subTest(pretrain=pretrain):
                self._ambient_seed(900)
                expected = self._rng_sample()
                self._ambient_seed(900)
                first, _ = self._initialize(123, pretrain=pretrain)
                self._assert_same_sample(first, expected)
                self._ambient_seed(900)
                changed_argument, _ = self._initialize(124, pretrain=pretrain)
                self._assert_same_sample(changed_argument, expected)

    def test_projector_initialization_follows_ambient_rng_not_training_seed(self):
        for kind in ('linear', 'mlp2x_gelu', 'coupling2x_gelu'):
            with self.subTest(projector=kind):
                def capture():
                    model = build_vision_projector(SimpleNamespace(
                        mm_projector_type=kind, mm_hidden_size=8, hidden_size=16,
                    ))
                    return torch.cat([p.detach().flatten() for p in model.parameters()])
                self._ambient_seed(100)
                a, _ = self._initialize(42, capture=capture)
                self._ambient_seed(100)
                b, _ = self._initialize(43, capture=capture)
                self._ambient_seed(101)
                c, _ = self._initialize(42, capture=capture)
                torch.testing.assert_close(a['capture'], b['capture'], atol=0, rtol=0)
                self.assertFalse(torch.equal(a['capture'], c['capture']))

    def test_trainer_constructor_seeds_rng_without_reinitializing_projector(self):
        for pretrain in (True, False):
            with self.subTest(pretrain=pretrain):
                first = self._trainer_initialize(42, 100, pretrain=pretrain)
                second = self._trainer_initialize(42, 101, pretrain=pretrain)
                different = self._trainer_initialize(43, 100, pretrain=pretrain)
                self._assert_same_sample(first, second)
                self.assertFalse(torch.equal(first['initial_projector'], second['initial_projector']))
                self.assertNotEqual(first['python'], different['python'])
                self.assertFalse(np.array_equal(first['numpy'], different['numpy']))
                self.assertFalse(torch.equal(first['torch'], different['torch']))

    def test_modality_sampler_order_is_repeatable_after_trainer_creation(self):
        def capture():
            return list(LengthGroupedSampler(
                batch_size=2, world_size=1, lengths=[10, -5, 20, -8] * 20,
                group_by_modality=True,
            ))
        a = self._trainer_initialize(42, 100, capture=capture)
        b = self._trainer_initialize(42, 101, capture=capture)
        c = self._trainer_initialize(43, 100, capture=capture)
        self.assertEqual(a['capture'], b['capture'])
        self.assertNotEqual(a['capture'], c['capture'])

    def test_subset_seed_inherits_run_seed_and_preserves_explicit_zero(self):
        _, inherited = self._initialize(123)
        _, explicit = self._initialize(123, subset_seed=0)
        self.assertEqual(inherited.train_data_seed, 123)
        self.assertEqual(explicit.train_data_seed, 0)

    def test_dataset_subset_uses_resolved_seed(self):
        def subset(seed, ambient_seed):
            self._ambient_seed(ambient_seed)
            _, args = self._initialize(seed)
            args.train_data_fraction = 0.25
            rows = [{'id': i} for i in range(80)]
            with mock.patch('builtins.open', mock.mock_open(read_data=json.dumps(rows))):
                dataset = training.LazySupervisedDataset('unused.json', None, args)
            return [row['id'] for row in dataset.list_data_dict]
        self.assertEqual(subset(12, 100), subset(12, 101))
        self.assertNotEqual(subset(12, 100), subset(13, 100))

    def test_default_seed_and_determinism_policy_are_unchanged(self):
        self.assertEqual(TrainingArguments.__dataclass_fields__['seed'].default, 42)
        self.assertFalse(TrainingArguments.__dataclass_fields__['full_determinism'].default)
        original = torch.are_deterministic_algorithms_enabled()
        self._initialize(42)
        self.assertEqual(torch.are_deterministic_algorithms_enabled(), original)


if __name__ == '__main__':
    unittest.main()
