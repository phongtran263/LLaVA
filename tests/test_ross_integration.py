"""CPU ROSS/LLaVA integration regressions without FLUX weights or a full DiT."""

from contextlib import nullcontext
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn

from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
from tests.test_pre_deepspeed_scaling import _DeepSpeedOutputAccelerator
from llava.train import llava_trainer as training


class _VisionTower(nn.Module):
    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(601)
        self.register_buffer('features', torch.randn(4, 6, generator=generator))

    def forward(self, images):
        return self.features.unsqueeze(0).expand(images.shape[0], -1, -1)


class _FrozenVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1), requires_grad=False)
        self.encode = mock.Mock(side_effect=AssertionError('The integration fake should not encode pixels.'))


def _config(enabled=True):
    config = LlavaConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=64,
    )
    config.ross_enable = enabled
    config.ross_loss_weight = 0.75
    config.image_embed_len = 4
    config.mm_inv_hidden_size = 4
    config.pre_loss = False
    config.cka_loss = False
    config.use_cache = False
    config.mm_patch_merge_type = 'flat'
    config.image_aspect_ratio = 'pad'
    config.tokenizer_padding_side = 'right'
    config.tokenizer_model_max_length = 64
    config.tune_mm_mlp_adapter = False
    config.mm_use_im_start_end = False
    return config


def _model(enabled=True):
    model = LlavaLlamaForCausalLM(_config(enabled)).train()
    model.get_model().vision_tower = _VisionTower()
    model.get_model().mm_projector = nn.Sequential(nn.Linear(6, 16), nn.GELU(), nn.Linear(16, 16))
    if enabled:
        model.get_model().mm_inv_projector = nn.Linear(16, 4)
        model.get_model().pixel_decoder = _FrozenVAE()
    return model


def _batch():
    ids = torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3]])
    return dict(input_ids=ids, labels=ids.clone(), images=torch.zeros(1, 3, 2, 2))


def _fake_reconstruction(head, _vae, visual_hidden, _images, valid_mask, *, config):
    # A deterministic differentiable objective with the same head/backbone routes.
    per_image = head(visual_hidden).square().mean(dim=(1, 2))
    return (per_image * valid_mask).sum() / valid_mask.sum().clamp_min(1)


class RossIntegrationTests(unittest.TestCase):
    def test_final_normalized_states_and_aligned_image_tokens_reach_reconstruction(self):
        torch.manual_seed(607)
        model = _model()
        observed = {}
        def capture_norm_input(_module, inputs):
            observed['pre_norm'] = inputs[0].detach().clone()
        def capture_norm_output(_module, _inputs, output):
            observed['final'] = output.detach().clone()
        def capture_first_block(_module, _inputs, output):
            observed['early'] = (output[0] if isinstance(output, (tuple, list)) else output).detach().clone()
        handles = [
            model.get_model().norm.register_forward_pre_hook(capture_norm_input),
            model.get_model().norm.register_forward_hook(capture_norm_output),
            model.get_model().layers[0].register_forward_hook(capture_first_block),
        ]
        batch = _batch()
        try:
            with mock.patch('llava.model.ross.compute_ross_reconstruction_loss', side_effect=_fake_reconstruction) as reconstruction:
                output = model(**batch)
        finally:
            for handle in handles:
                handle.remove()
        reconstruction.assert_called_once()
        args = reconstruction.call_args.args
        self.assertIs(args[0], model.get_model().mm_inv_projector)
        self.assertIs(args[1], model.get_model().pixel_decoder)
        self.assertIs(args[3], batch['images'])
        self.assertIs(reconstruction.call_args.kwargs['config'], model.config)
        torch.testing.assert_close(args[2], observed['final'][:, 1:5])
        self.assertFalse(torch.allclose(args[2], observed['pre_norm'][:, 1:5]))
        self.assertFalse(torch.allclose(args[2], observed['early'][:, 1:5]))
        torch.testing.assert_close(args[4], torch.tensor([True]))
        self.assertEqual(int(model.last_ross_num_images), 1)
        self.assertFalse(model.last_ross_loss.requires_grad)
        self.assertFalse(model.last_ross_text_loss.requires_grad)
        torch.testing.assert_close(output.loss.detach(), model.last_ross_text_loss + 0.75 * model.last_ross_loss)

    def test_auxiliary_toggle_preserves_logits_and_language_loss(self):
        model = _model()
        with mock.patch('llava.model.ross.compute_ross_reconstruction_loss', side_effect=_fake_reconstruction):
            enabled = model(**_batch())
        reconstruction_loss = model.last_ross_loss.clone()
        text_loss = model.last_ross_text_loss.clone()
        model.config.ross_enable = False
        disabled = model(**_batch())
        torch.testing.assert_close(enabled.logits, disabled.logits, rtol=0, atol=0)
        torch.testing.assert_close(disabled.loss.detach(), text_loss, rtol=0, atol=0)
        torch.testing.assert_close(enabled.loss.detach(), disabled.loss.detach() + 0.75 * reconstruction_loss)

    def test_auxiliary_gradient_reaches_final_norm_decoder_and_projector(self):
        torch.manual_seed(613)
        model = _model()
        captured = []
        def retain(_module, _inputs, output):
            captured.append(output)
        handle = model.get_model().norm.register_forward_hook(retain)
        try:
            with mock.patch('llava.model.ross.compute_ross_reconstruction_loss', side_effect=_fake_reconstruction) as reconstruction:
                model(**_batch())
        finally:
            handle.remove()
        # Backpropagate the auxiliary branch alone, independently of CE.
        args = reconstruction.call_args.args
        _fake_reconstruction(*args, **reconstruction.call_args.kwargs).backward()
        self.assertTrue(captured)
        for module in (model.get_model().mm_inv_projector, model.get_model().mm_projector, model.get_model().layers[-1].mlp, model.get_model().norm):
            gradients = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
            self.assertTrue(gradients)
            self.assertTrue(all(value is not None and torch.isfinite(value).all() for value in gradients))
            self.assertGreater(sum(float(value.norm()) for value in gradients), 0)
        self.assertIsNone(model.get_model().pixel_decoder.weight.grad)
        model.get_model().pixel_decoder.encode.assert_not_called()

    def test_left_right_padding_and_mixed_text_rows_preserve_image_alignment(self):
        for padding_side in ('left', 'right'):
            with self.subTest(padding_side=padding_side):
                model = _model()
                model.config.tokenizer_padding_side = padding_side
                ids = torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3], [1, 4, 0, 0]])
                images = torch.stack((torch.ones(3, 2, 2), torch.full((3, 2, 2), 2.0)))
                captured = []
                handle = model.get_model().norm.register_forward_hook(lambda _module, _inputs, output: captured.append(output.detach().clone()))
                try:
                    with mock.patch('llava.model.ross.compute_ross_reconstruction_loss', side_effect=_fake_reconstruction) as reconstruction:
                        model(input_ids=ids, labels=ids.clone(), attention_mask=torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]), images=images)
                finally:
                    handle.remove()
                visual_hidden, actual_images, valid = reconstruction.call_args.args[2:5]
                self.assertIs(actual_images, images)
                torch.testing.assert_close(valid, torch.tensor([True, False]))
                torch.testing.assert_close(visual_hidden[0], captured[0][0, 1:5])
                torch.testing.assert_close(visual_hidden[1], torch.zeros_like(visual_hidden[1]), rtol=0, atol=0)
                self.assertEqual(int(model.last_ross_num_images), 1)

    def test_truncated_patch_grids_are_rejected_before_reconstruction(self):
        model = _model()
        model.config.tokenizer_model_max_length = 3
        with mock.patch('llava.model.ross.compute_ross_reconstruction_loss') as reconstruction:
            with self.assertRaisesRegex(ValueError, 'complete visual patch grid'):
                model(**_batch())
        reconstruction.assert_not_called()

    def test_text_only_batch_retains_zero_head_gradient(self):
        model = _model()
        ids = torch.tensor([[1, 2, 3, 4]])
        with mock.patch('llava.model.ross.compute_ross_reconstruction_loss', side_effect=_fake_reconstruction) as reconstruction:
            output = model(input_ids=ids, labels=ids.clone())
        self.assertEqual(float(model.last_ross_loss), 0)
        self.assertEqual(int(model.last_ross_num_images), 0)
        self.assertFalse(bool(reconstruction.call_args.args[4].any()))
        output.loss.backward()
        for parameter in model.get_model().mm_inv_projector.parameters():
            self.assertIsNotNone(parameter.grad)
            torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter), rtol=0, atol=0)

    def test_evaluation_skips_reconstruction_and_vae(self):
        model = _model().eval()
        with mock.patch('llava.model.ross.compute_ross_reconstruction_loss') as reconstruction:
            enabled = model(**_batch())
            model.config.ross_enable = False
            disabled = model(**_batch())
        reconstruction.assert_not_called()
        model.get_model().pixel_decoder.encode.assert_not_called()
        torch.testing.assert_close(enabled.loss, disabled.loss, rtol=0, atol=0)
        torch.testing.assert_close(enabled.logits, disabled.logits, rtol=0, atol=0)

    def test_disabled_flag_preserves_backbone_weights_rng_and_state(self):
        absent_config = _config(False)
        del absent_config.ross_enable
        torch.manual_seed(617)
        absent = LlavaLlamaForCausalLM(absent_config)
        absent_rng = torch.get_rng_state().clone()
        torch.manual_seed(617)
        disabled = LlavaLlamaForCausalLM(_config(False))
        self.assertFalse(hasattr(disabled.get_model(), 'mm_inv_projector'))
        self.assertFalse(hasattr(disabled.get_model(), 'pixel_decoder'))
        self.assertTrue(torch.equal(absent_rng, torch.get_rng_state()))
        self.assertEqual(absent.state_dict().keys(), disabled.state_dict().keys())
        for name, value in absent.state_dict().items():
            torch.testing.assert_close(value, disabled.state_dict()[name], rtol=0, atol=0)

    def test_pre_and_cka_cannot_be_combined_with_ross(self):
        for flag in ('pre_loss', 'cka_loss'):
            with self.subTest(flag=flag):
                model = _model()
                setattr(model.config, flag, True)
                with self.assertRaises(ValueError):
                    model(**_batch())

    def test_fresh_head_is_built_late_and_saved_head_reconstructs_without_vae(self):
        config = _config()
        with mock.patch('llava.model.ross.build_ross_denoiser', side_effect=lambda _config: nn.Linear(16, 4)) as factory:
            model = LlavaLlamaForCausalLM(config)
            factory.assert_not_called()
            model.get_model().initialize_ross_modules(load_vae=False)
            factory.assert_called_once()
            self.assertIs(model.config.ross_head_initialized, True)
            original_head = model.get_model().mm_inv_projector
            model.get_model().initialize_ross_modules(load_vae=False)
            self.assertIs(model.get_model().mm_inv_projector, original_head)
            factory.assert_called_once()
            with torch.no_grad():
                original_head.weight.fill_(0.125)
            with tempfile.TemporaryDirectory(prefix='llava-ross-integration-') as directory:
                model.save_pretrained(directory)
                with mock.patch('llava.model.ross.FrozenFluxAutoencoder') as vae_factory:
                    restored = LlavaLlamaForCausalLM.from_pretrained(directory, local_files_only=True)
                vae_factory.assert_not_called()
            self.assertEqual(factory.call_count, 2)
        self.assertFalse(hasattr(restored.get_model(), 'pixel_decoder'))
        self.assertIs(restored.config.ross_enable, True)
        for name, value in original_head.state_dict().items():
            torch.testing.assert_close(value, restored.get_model().mm_inv_projector.state_dict()[name], rtol=0, atol=0)

    def test_tuple_output_retains_combined_loss(self):
        model = _model()
        with mock.patch('llava.model.ross.compute_ross_reconstruction_loss', side_effect=_fake_reconstruction):
            as_dict = model(**_batch())
            as_tuple = model(**_batch(), return_dict=False)
        self.assertIsInstance(as_tuple, tuple)
        torch.testing.assert_close(as_tuple[0], as_dict.loss)
        torch.testing.assert_close(as_tuple[1], as_dict.logits)

    def test_nonreentrant_checkpointing_keeps_auxiliary_gradients(self):
        model = _model()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        with mock.patch('llava.model.ross.compute_ross_reconstruction_loss', side_effect=_fake_reconstruction):
            output = model(**_batch())
        output.loss.backward()
        for module in (model.get_model().mm_inv_projector, model.get_model().mm_projector):
            gradients = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
            self.assertTrue(all(value is not None and torch.isfinite(value).all() for value in gradients))
            self.assertGreater(sum(float(value.norm()) for value in gradients), 0)


class _NestedRossLossModel(nn.Module):
    def __init__(self, accelerator=None):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))
        self.reconstruction_weight = nn.Parameter(torch.tensor(0.25))
        self.config = SimpleNamespace(cka_loss=False, pre_loss=False, ross_enable=True)
        self.output_accelerator = accelerator

    def forward(self):
        logits = 3 * self.weight
        ce = logits.square()
        reconstruction = (self.weight - 1).square() + self.reconstruction_weight.square()
        output = {'loss': ce + reconstruction, 'logits': logits}
        return self.output_accelerator.attach(output) if self.output_accelerator is not None else output


class RossDeepSpeedScalingTests(unittest.TestCase):
    def test_ross_training_step_scales_ce_and_auxiliary_once_with_actual_output_hooks(self):
        for steps in (1, 2, 32):
            with self.subTest(steps=steps):
                reference = _NestedRossLossModel()
                (reference()['loss'] / steps).backward()
                accelerator = _DeepSpeedOutputAccelerator(steps)
                model = _NestedRossLossModel(accelerator)
                trainer = object.__new__(training.LLaVATrainer)
                trainer.model = model
                trainer.args = SimpleNamespace(n_gpu=1, gradient_accumulation_steps=steps)
                trainer.accelerator = accelerator
                trainer.is_deepspeed_enabled = True
                trainer.use_apex = False
                trainer.compute_loss_context_manager = nullcontext
                trainer.compute_loss = lambda current_model, _inputs: current_model()['loss']
                trainer._collect_gradient_norm_logs = mock.Mock()
                trainer._clear_gradient_log_tensors = mock.Mock()
                trainer._vsp_controller_requested = lambda: False
                with mock.patch.object(training, 'backward_pre_deepspeed_loss', wraps=training.backward_pre_deepspeed_loss) as backward:
                    result = trainer._training_step_with_cka_runtime_state(model, {})
                backward.assert_called_once()
                self.assertEqual(backward.call_args.args[2], steps)
                self.assertEqual(accelerator.calls, [{'scale_wrt_gas': False}])
                self.assertTrue(torch.isfinite(result))
                for actual, expected in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
