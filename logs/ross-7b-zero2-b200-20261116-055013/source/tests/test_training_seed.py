"""Keep original Trainer-controlled seeding after model initialization."""
import importlib
import json
import random
import tempfile
import unittest
from pathlib import Path
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

    def _initialize(self, seed=42, *, pretrain=False, subset_seed=None, capture=None,
                    projector_seed=None):
        model_args = ModelArguments(tune_mm_mlp_adapter=pretrain,
                                    mm_projector_init_seed=projector_seed)
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

    def _vision_setup(self, projector_seed, *, cka=False, existing=False, adapter=None):
        # make_model has completed HF post_init. Mimic a plain base checkpoint
        # whose missing projector is constructed later by vision setup.
        model = make_model(cka=cka)
        core = model.get_model()
        core.vision_tower.hidden_size = 6
        core.vision_tower.load_model = mock.Mock()
        if existing:
            core.mm_projector = build_vision_projector(SimpleNamespace(
                mm_projector_type='mlp2x_gelu', mm_hidden_size=6, hidden_size=16))
        else:
            del core.mm_projector
        args = ModelArguments(vision_tower='fake', mm_projector_type='mlp2x_gelu',
                              mm_projector_init_seed=projector_seed,
                              pretrain_mm_mlp_adapter=adapter)
        return model, args

    @staticmethod
    def _projector_weights(model):
        return torch.cat([parameter.detach().flatten().clone()
                          for parameter in model.get_model().mm_projector.parameters()])

    def test_optional_projector_seed_matches_cka_arms_and_ignores_ambient_rng(self):
        initialized = []
        for cka, ambient_seed in ((False, 100), (True, 200)):
            model, args = self._vision_setup(1001, cka=cka)
            self._ambient_seed(ambient_seed)
            model.get_model().initialize_vision_modules(args)
            initialized.append(self._projector_weights(model))
            self.assertEqual(model.config.to_dict()['mm_projector_init_seed'], 1001)
        torch.testing.assert_close(initialized[0], initialized[1], atol=0, rtol=0)
        model, args = self._vision_setup(1002)
        self._ambient_seed(100)
        model.get_model().initialize_vision_modules(args)
        self.assertFalse(torch.equal(initialized[0], self._projector_weights(model)))

    def test_optional_projector_seed_preserves_rng_and_backbone_weights(self):
        model, args = self._vision_setup(1001)
        original = {name: value.clone() for name, value in model.state_dict().items()}
        self._ambient_seed(100)
        expected = self._rng_sample()
        self._ambient_seed(100)
        with mock.patch('torch.cuda.manual_seed_all', side_effect=AssertionError('CUDA reseeded')), \
             mock.patch('torch.cuda.get_rng_state', side_effect=AssertionError('CUDA RNG read')):
            model.get_model().initialize_vision_modules(args)
        self._assert_same_sample(self._rng_sample(), expected)
        for name, value in original.items():
            torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0)

    def test_missing_projector_seed_preserves_legacy_constructor_rng(self):
        model, args = self._vision_setup(None)
        self._ambient_seed(100)
        expected = build_vision_projector(SimpleNamespace(
            mm_projector_type='mlp2x_gelu', mm_hidden_size=6, hidden_size=16))
        expected_rng = torch.get_rng_state().clone()
        self._ambient_seed(100)
        model.get_model().initialize_vision_modules(args)
        torch.testing.assert_close(
            self._projector_weights(model),
            torch.cat([parameter.detach().flatten() for parameter in expected.parameters()]),
            atol=0, rtol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        self.assertFalse(hasattr(model.config, 'mm_projector_init_seed'))

    def test_existing_projector_is_not_reset_by_optional_seed(self):
        model, args = self._vision_setup(1001, existing=True)
        projector = model.get_model().mm_projector
        expected = self._projector_weights(model)
        with mock.patch('llava.model.llava_arch.build_vision_projector',
                        side_effect=AssertionError('Existing projector rebuilt')):
            model.get_model().initialize_vision_modules(args)
        self.assertIs(model.get_model().mm_projector, projector)
        torch.testing.assert_close(self._projector_weights(model), expected, atol=0, rtol=0)

    def test_pretrained_adapter_weights_override_optional_initialization_seed(self):
        reference = build_vision_projector(SimpleNamespace(
            mm_projector_type='mlp2x_gelu', mm_hidden_size=6, hidden_size=16))
        with torch.no_grad():
            for index, parameter in enumerate(reference.parameters()):
                parameter.fill_(index + 0.25)
        with tempfile.TemporaryDirectory() as output:
            checkpoint = Path(output) / 'mm_projector.bin'
            torch.save({'model.mm_projector.' + name: value
                        for name, value in reference.state_dict().items()}, checkpoint)
            model, args = self._vision_setup(1001, adapter=str(checkpoint))
            model.get_model().initialize_vision_modules(args)
        torch.testing.assert_close(
            self._projector_weights(model),
            torch.cat([parameter.detach().flatten() for parameter in reference.parameters()]),
            atol=0, rtol=0)

    def test_trainer_seed_42_remains_independent_of_projector_initialization_seed(self):
        samples, weights = [], []
        for projector_seed in (1001, 1002):
            model, args = self._vision_setup(projector_seed)
            model.get_model().initialize_vision_modules(args)
            expected = self._projector_weights(model)
            weights.append(expected)
            with tempfile.TemporaryDirectory() as output:
                training_args = TrainingArguments(output_dir=output, seed=42, use_cpu=True,
                                                 report_to=[], disable_tqdm=True)
                trainer = LLaVATrainer(model=model, args=training_args)
                self.assertEqual(trainer.args.seed, 42)
                samples.append(self._rng_sample())
            torch.testing.assert_close(self._projector_weights(model), expected, atol=0, rtol=0)
        self._assert_same_sample(*samples)
        self.assertFalse(torch.equal(*weights))

    def test_optional_projector_seed_is_validated_before_model_loading(self):
        for value in (-1, 2 ** 63, True, 1.5, '42'):
            with self.subTest(seed=value), self.assertRaisesRegex(ValueError, 'mm_projector_init_seed'):
                self._initialize(projector_seed=value)
        for value in (0, 2 ** 63 - 1):
            with self.subTest(seed=value):
                self._ambient_seed(100)
                expected = self._rng_sample()
                self._ambient_seed(100)
                observed, _ = self._initialize(projector_seed=value)
                self._assert_same_sample(observed, expected)

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
        self.assertIsNone(ModelArguments.__dataclass_fields__['mm_projector_init_seed'].default)
        self.assertEqual(TrainingArguments.__dataclass_fields__['seed'].default, 42)
        self.assertFalse(TrainingArguments.__dataclass_fields__['full_determinism'].default)
        original = torch.are_deterministic_algorithms_enabled()
        self._initialize(42)
        self.assertEqual(torch.are_deterministic_algorithms_enabled(), original)


if __name__ == '__main__':
    unittest.main()
