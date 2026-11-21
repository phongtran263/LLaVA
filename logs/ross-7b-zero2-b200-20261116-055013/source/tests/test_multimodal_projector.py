"""Unit tests for multimodal projector construction and serialization."""

import io
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from llava.model.multimodal_projector.builder import build_vision_projector


def _config(projector_type, mm_hidden_size=6, hidden_size=8):
    return SimpleNamespace(
        mm_projector_type=projector_type,
        mm_hidden_size=mm_hidden_size,
        hidden_size=hidden_size,
    )


class MultimodalProjectorTests(unittest.TestCase):
    def test_coupling_projector_shape_and_backward_even_and_odd_hidden_size(self):
        cases = (
            ("coupling1x_gelu", 6, 8, (2, 5, 6)),
            ("coupling2x_gelu", 5, 7, (3, 4, 5)),
        )
        for projector_type, mm_hidden_size, hidden_size, input_shape in cases:
            with self.subTest(
                projector_type=projector_type,
                mm_hidden_size=mm_hidden_size,
                hidden_size=hidden_size,
            ):
                torch.manual_seed(17)
                projector = build_vision_projector(
                    _config(projector_type, mm_hidden_size, hidden_size)
                )
                inputs = torch.randn(*input_shape, requires_grad=True)

                outputs = projector(inputs)

                self.assertEqual(outputs.shape, input_shape[:-1] + (hidden_size,))
                self.assertTrue(torch.isfinite(outputs).all())

                outputs.square().mean().backward()
                self.assertIsNotNone(inputs.grad)
                self.assertTrue(torch.isfinite(inputs.grad).all())

                parameter_grads = [
                    parameter.grad for parameter in projector.parameters()
                ]
                self.assertTrue(parameter_grads)
                self.assertTrue(all(grad is not None for grad in parameter_grads))
                self.assertTrue(
                    all(torch.isfinite(grad).all() for grad in parameter_grads)
                )
                self.assertGreater(
                    sum(grad.abs().sum().item() for grad in parameter_grads), 0.0
                )

    def test_coupling_blocks_are_identity_at_initialization(self):
        projector = build_vision_projector(
            _config("coupling3x_gelu", mm_hidden_size=5, hidden_size=7)
        ).eval()
        inputs = torch.randn(2, 4, 5)

        torch.testing.assert_close(projector(inputs), projector.stem(inputs))

    def test_coupling_depth_in_type_controls_projector_capacity(self):
        one_block = build_vision_projector(_config("coupling1x_gelu"))
        three_blocks = build_vision_projector(_config("coupling3x_gelu"))

        one_block_parameters = sum(
            parameter.numel() for parameter in one_block.parameters()
        )
        three_block_parameters = sum(
            parameter.numel() for parameter in three_blocks.parameters()
        )
        self.assertGreater(three_block_parameters, one_block_parameters)

    def test_existing_mlp2x_gelu_layout_and_checkpoint_remain_compatible(self):
        config = _config("mlp2x_gelu", mm_hidden_size=5, hidden_size=7)
        projector = build_vision_projector(config)

        self.assertIsInstance(projector, nn.Sequential)
        self.assertEqual(
            [type(module) for module in projector],
            [nn.Linear, nn.GELU, nn.Linear],
        )
        self.assertEqual(
            list(projector.state_dict()),
            ["0.weight", "0.bias", "2.weight", "2.bias"],
        )

        legacy_projector = nn.Sequential(
            nn.Linear(config.mm_hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        load_result = projector.load_state_dict(legacy_projector.state_dict(), strict=True)
        self.assertEqual(load_result.missing_keys, [])
        self.assertEqual(load_result.unexpected_keys, [])

        inputs = torch.randn(2, 3, config.mm_hidden_size)
        torch.testing.assert_close(projector(inputs), legacy_projector(inputs))

    def test_legacy_mlp_checkpoint_cannot_strict_load_into_coupling_projector(self):
        config = _config("mlp2x_gelu", mm_hidden_size=5, hidden_size=7)
        legacy_state = build_vision_projector(config).state_dict()
        coupling = build_vision_projector(
            _config("coupling1x_gelu", mm_hidden_size=5, hidden_size=7)
        )

        with self.assertRaises(RuntimeError):
            coupling.load_state_dict(legacy_state, strict=True)

    def test_coupling_depth_must_be_positive(self):
        with self.assertRaises(ValueError):
            build_vision_projector(_config("coupling0x_gelu"))

    def test_malformed_coupling_projector_types_are_rejected(self):
        malformed_types = (
            "couplingx_gelu",
            "coupling-1x_gelu",
            "coupling2x_relu",
            "coupling2x_gelu_extra",
        )
        for projector_type in malformed_types:
            with self.subTest(projector_type=projector_type), self.assertRaises(ValueError):
                build_vision_projector(_config(projector_type))

    def test_coupling_projector_adapter_state_dict_roundtrip(self):
        config = _config("coupling2x_gelu", mm_hidden_size=5, hidden_size=7)
        torch.manual_seed(23)
        source = build_vision_projector(config).eval()
        inputs = torch.randn(2, 4, config.mm_hidden_size)
        expected = source(inputs)

        adapter_state = {
            f"model.mm_projector.{key}": value.detach().clone()
            for key, value in source.state_dict().items()
        }
        checkpoint = io.BytesIO()
        torch.save(adapter_state, checkpoint)
        checkpoint.seek(0)
        loaded_adapter_state = torch.load(checkpoint, map_location="cpu")
        projector_state = {
            key.split("mm_projector.", 1)[1]: value
            for key, value in loaded_adapter_state.items()
            if "mm_projector." in key
        }

        torch.manual_seed(29)
        restored = build_vision_projector(config).eval()
        load_result = restored.load_state_dict(projector_state, strict=True)
        self.assertEqual(load_result.missing_keys, [])
        self.assertEqual(load_result.unexpected_keys, [])
        torch.testing.assert_close(restored(inputs), expected)


if __name__ == "__main__":
    unittest.main()
