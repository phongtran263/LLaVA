import unittest

import transformers

from llava import conversation as conversation_lib
from llava.mm_utils import infer_conversation_mode
from llava.model.backbones import (
    as_llava_config,
    backbone_type,
    default_conversation,
    get_llava_model_class,
)


class BackboneRegistryTests(unittest.TestCase):
    def test_existing_family_dispatch(self):
        cases = [
            (transformers.LlamaConfig(vocab_size=32000), "llama", "v1"),
            (transformers.LlamaConfig(vocab_size=128256), "llama", "llama3"),
            (transformers.MistralConfig(), "mistral", "mistral_instruct"),
        ]
        if hasattr(transformers, "Qwen2Config"):
            cases.append((transformers.Qwen2Config(), "qwen2", "qwen2"))

        for config, family, conversation in cases:
            with self.subTest(family=family, conversation=conversation):
                self.assertEqual(backbone_type(config), family)
                self.assertEqual(default_conversation(config), conversation)
                model_cls = get_llava_model_class(config)
                converted = as_llava_config(config)
                self.assertIsInstance(converted, model_cls.config_class)

    @unittest.skipUnless(hasattr(transformers, "Qwen3Config"), "requires Transformers 4.51.3")
    def test_modern_family_dispatch(self):
        cases = [
            (transformers.Qwen3Config(), "qwen3", "qwen3", "LlavaQwen3ForCausalLM"),
            (transformers.Gemma3TextConfig(), "gemma3_text", "gemma3", "LlavaGemma3ForCausalLM"),
            (transformers.Phi3Config(), "phi3", "phi3", "LlavaPhi3ForCausalLM"),
        ]
        for config, family, conversation, class_name in cases:
            with self.subTest(family=family):
                self.assertEqual(backbone_type(config), family)
                self.assertEqual(default_conversation(config), conversation)
                self.assertEqual(get_llava_model_class(config).__name__, class_name)
                self.assertEqual(as_llava_config(config).model_type, {
                    "qwen3": "llava_qwen3",
                    "gemma3_text": "llava_gemma3",
                    "phi3": "llava_phi3",
                }[family])

    def test_conversation_inference_prefers_checkpoint_config(self):
        config = transformers.LlamaConfig(vocab_size=128256)
        config.llava_conversation_version = "llama3"
        self.assertEqual(infer_conversation_mode("renamed-checkpoint", config), "llama3")
        self.assertEqual(infer_conversation_mode("Qwen3-4B"), "qwen3")
        self.assertEqual(infer_conversation_mode("gemma-3-1b-it"), "gemma3")
        self.assertEqual(infer_conversation_mode("Phi-3.5-mini-instruct"), "phi3")

    def test_named_model_variants_use_their_native_chat_format(self):
        tinyllama = transformers.LlamaConfig(vocab_size=32000)
        tinyllama._name_or_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.assertEqual(default_conversation(tinyllama), "tinyllama")
        self.assertEqual(
            infer_conversation_mode(tinyllama._name_or_path, tinyllama),
            "tinyllama",
        )

        if hasattr(transformers, "Qwen3Config"):
            qwen3_instruct = transformers.Qwen3Config()
            qwen3_instruct._name_or_path = "Qwen/Qwen3-4B-Instruct-2507"
            self.assertEqual(default_conversation(qwen3_instruct), "qwen3_instruct")
            self.assertEqual(
                infer_conversation_mode(qwen3_instruct._name_or_path, qwen3_instruct),
                "qwen3_instruct",
            )

    def test_variant_prompt_tokens(self):
        qwen3 = conversation_lib.conv_templates["qwen3_instruct"].copy()
        qwen3.append_message(qwen3.roles[0], "question")
        qwen3.append_message(qwen3.roles[1], "answer")
        qwen3_prompt = qwen3.get_prompt()
        self.assertIn("<|im_start|>assistant\nanswer<|im_end|>", qwen3_prompt)
        self.assertNotIn("<think>", qwen3_prompt)

        tinyllama = conversation_lib.conv_templates["tinyllama"].copy()
        tinyllama.append_message(tinyllama.roles[0], "question")
        tinyllama.append_message(tinyllama.roles[1], "answer")
        tinyllama_prompt = tinyllama.get_prompt()
        self.assertIn("<|user|>\nquestion</s>", tinyllama_prompt)
        self.assertIn("<|assistant|>\nanswer</s>", tinyllama_prompt)


if __name__ == "__main__":
    unittest.main()

