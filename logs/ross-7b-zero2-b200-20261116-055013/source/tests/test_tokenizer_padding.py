"""End-of-turn supervision must survive padding, including PAD == EOS/UNK."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import GenerationConfig, PreTrainedTokenizerFast

from llava import conversation as conversation_lib
from llava.constants import IGNORE_INDEX
from llava.model.backbones import configure_tokenizer_padding
from llava.model.builder import _set_generation_pad_token
from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
from llava.train.train import DataCollatorForSupervisedDataset, _tokenize_fn, preprocess_qwen
from tests.test_cka_backbones import _FakeVisionTower, _configure_multimodal_cka, _projector


def make_tokenizer(eos="<|eot_id|>", pad="<|eot_id|>"):
    specials = ["<unk>", "<pad>", "<s>", "</s>", "<|eot_id|>",
                "<|finetune_right_pad_id|>", "<|endoftext|>", "<|im_start|>",
                "<|im_end|>", "<|start_header_id|>", "<|end_header_id|>",
                "<start_of_turn>", "<end_of_turn>", "<eos>", "<|end|>",
                "<|system|>", "<|user|>", "<|assistant|>"]
    words = specials + ["user", "assistant", "system", "model", "Yes", "No", "Question"]
    backend = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", bos_token="<s>",
        eos_token=eos, pad_token=pad, additional_special_tokens=specials,
        model_max_length=256, padding_side="right",
    )


class TokenizerPaddingTests(unittest.TestCase):
    def test_llama3_replaces_missing_or_boundary_padding_without_adding_tokens(self):
        for pad in (None, "<|eot_id|>", "<s>"):
            with self.subTest(pad=pad):
                tok = make_tokenizer(pad=pad)
                vocab = tok.get_vocab()
                config = SimpleNamespace(model_type="llava_llama", vocab_size=128256)
                configure_tokenizer_padding(tok, config)
                self.assertEqual(tok.pad_token, "<|finetune_right_pad_id|>")
                self.assertEqual(config.pad_token_id, tok.pad_token_id)
                self.assertEqual(tok.eos_token, "<|eot_id|>")
                self.assertEqual(tok.get_vocab(), vocab)

    def test_existing_padding_is_preserved_for_other_families(self):
        for family, pad, eos in (
            ("llama", "<unk>", "</s>"), ("mistral", "<unk>", "</s>"),
            ("qwen2", "<|endoftext|>", "<|im_end|>"),
            ("qwen3", "<|endoftext|>", "<|im_end|>"),
            ("gemma3_text", "<pad>", "<eos>"),
            ("phi3", "<|endoftext|>", "<|endoftext|>"),
            ("mpt", "<|endoftext|>", "<|endoftext|>"),
        ):
            with self.subTest(family=family):
                tok = make_tokenizer(eos=eos, pad=pad)
                config = SimpleNamespace(model_type=family, vocab_size=32000)
                configure_tokenizer_padding(tok, config)
                self.assertEqual(tok.pad_token, pad)
                self.assertEqual(tok.eos_token, eos)

    def test_custom_safe_llama3_padding_is_preserved(self):
        tok = make_tokenizer(pad="<pad>")
        configure_tokenizer_padding(tok, SimpleNamespace(model_type="llama", vocab_size=128256))
        self.assertEqual(tok.pad_token, "<pad>")

    def test_old_checkpoint_generation_uses_native_pad_and_retains_stop_ids(self):
        tok = make_tokenizer()
        eos = tok.eos_token_id
        embeddings = torch.nn.Embedding(len(tok), 8, padding_idx=eos)
        original_weights = embeddings.weight.detach().clone()
        model = SimpleNamespace(
            config=SimpleNamespace(model_type="llava_llama", vocab_size=128256, pad_token_id=eos),
            generation_config=GenerationConfig(eos_token_id=[eos, 3], pad_token_id=eos),
            get_input_embeddings=lambda: embeddings,
        )
        _set_generation_pad_token(tok, model)
        self.assertEqual(model.config.pad_token_id, tok.pad_token_id)
        self.assertEqual(model.generation_config.pad_token_id, tok.pad_token_id)
        self.assertEqual(model.generation_config.eos_token_id, [eos, 3])
        self.assertNotIn(tok.pad_token_id, model.generation_config.eos_token_id)
        self.assertEqual(embeddings.padding_idx, tok.pad_token_id)
        torch.testing.assert_close(embeddings.weight, original_weights, atol=0, rtol=0)
        embeddings(torch.tensor([eos, tok.pad_token_id])).sum().backward()
        self.assertGreater(float(embeddings.weight.grad[eos].abs().sum()), 0.)
        self.assertEqual(float(embeddings.weight.grad[tok.pad_token_id].abs().sum()), 0.)

    def test_collator_preserves_real_eos_and_unk_but_masks_added_padding(self):
        for pad in ("<|eot_id|>", "<unk>", "<pad>"):
            with self.subTest(pad=pad):
                tok = make_tokenizer(pad=pad)
                sequences = [torch.tensor([24, tok.pad_token_id, 25, tok.eos_token_id]),
                             torch.tensor([24, tok.eos_token_id])]
                rows = [dict(input_ids=ids, labels=ids.clone()) for ids in sequences]
                batch = DataCollatorForSupervisedDataset(tok)(rows)
                self.assertEqual(batch["attention_mask"].tolist(), [[True]*4, [True, True, False, False]])
                self.assertEqual(batch["labels"][1, 2:].tolist(), [IGNORE_INDEX, IGNORE_INDEX])
                torch.testing.assert_close(batch["labels"][0], sequences[0])

    def test_collator_truncation_keeps_lengths_aligned(self):
        tok = make_tokenizer()
        tok.model_max_length = 3
        ids = torch.tensor([24, tok.eos_token_id, 25, tok.eos_token_id])
        batch = DataCollatorForSupervisedDataset(tok)([dict(input_ids=ids, labels=ids.clone())])
        self.assertEqual(batch["attention_mask"].tolist(), [[True, True, True]])
        torch.testing.assert_close(batch["labels"][0], ids[:3])

    def test_legacy_length_calculation_counts_real_pad_tokens(self):
        tok = make_tokenizer(pad="<unk>")
        result = _tokenize_fn(["Question <unk> Yes <|eot_id|>"], tok)
        self.assertEqual(result["input_ids_lens"], [4])

    def test_turn_labels_survive_multimodal_expansion_for_all_chat_families(self):
        cases = [
            ("llama3", "<|eot_id|>", "<|eot_id|>", "<|eot_id|>"),
            ("qwen2", "<|im_end|>", "<|endoftext|>", "<|im_end|>"),
            ("qwen3", "<|im_end|>", "<|endoftext|>", "<|im_end|>"),
            ("gemma3", "<eos>", "<pad>", "<end_of_turn>"),
            ("phi3", "<|endoftext|>", "<|endoftext|>", "<|end|>"),
            ("tinyllama", "</s>", "</s>", "</s>"),
        ]
        source = [{"from": "human", "value": "<image>\nQuestion"},
                  {"from": "gpt", "value": "Yes"},
                  {"from": "human", "value": "Question"},
                  {"from": "gpt", "value": "No"}]
        for template, eos, pad, turn_end in cases:
            with self.subTest(template=template):
                tok = make_tokenizer(eos=eos, pad=pad)
                with mock.patch.object(conversation_lib, 'default_conversation', conversation_lib.conv_templates[template]):
                    data = preprocess_qwen([copy.deepcopy(source)], tok, has_image=True)
                turn_id = tok.convert_tokens_to_ids(turn_end)
                self.assertEqual(int((data['labels'] == turn_id).sum()), 2)
                row = dict(input_ids=data['input_ids'][0], labels=data['labels'][0], image=torch.zeros(3, 2, 2))
                batch = DataCollatorForSupervisedDataset(tok)([row])
                config = _configure_multimodal_cka(LlavaConfig(
                    vocab_size=len(tok), hidden_size=16, intermediate_size=32,
                    num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                ))
                config.cka_loss = False
                config.tokenizer_model_max_length = 256
                model = LlavaLlamaForCausalLM(config)
                model.model.vision_tower = _FakeVisionTower()
                model.model.mm_projector = _projector()
                expanded = model.prepare_inputs_labels_for_multimodal(
                    batch['input_ids'], None, batch['attention_mask'], None, batch['labels'], batch['images'],
                )
                self.assertEqual(int((expanded[5] == turn_id).sum()), 2)
                output = model(**batch)
                output.loss.backward()
                self.assertTrue(torch.isfinite(output.loss))
                self.assertTrue(torch.isfinite(model.lm_head.weight.grad).all())
                self.assertGreater(float(model.lm_head.weight.grad[turn_id].abs().sum()), 0.)

    def test_eval_mask_preserves_real_tokens_with_left_or_right_padding(self):
        # Exercise the adapter's actual mask expression without importing optional
        # benchmark dependencies or loading a GPU model.
        path = Path(__file__).resolve().parents[1] / 'lmms-eval/lmms_eval/models/simple/llava.py'
        if not path.is_file():
            self.skipTest('lmms-eval submodule is not checked out')
        cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'Llava')
        pad = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'pad_sequence')
        generate = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'generate_until')
        mask = next(n for n in ast.walk(generate) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'attention_masks' for t in n.targets))
        scope = {'torch': torch}
        exec(compile(ast.Module(body=[pad], type_ignores=[]), str(path), 'exec'), scope)
        for side in ('left', 'right'):
            tok = make_tokenizer()
            tok.padding_side = side
            adapter = SimpleNamespace(tokenizer=tok, device='cpu')
            adapter.pad_sequence = scope['pad_sequence'].__get__(adapter)
            rows = [torch.tensor([24, tok.eos_token_id, 25]), torch.tensor([tok.eos_token_id])]
            scope.update(self=adapter, input_ids_list=rows,
                         input_ids=adapter.pad_sequence(rows, True, tok.pad_token_id), pad_token_ids=tok.pad_token_id)
            exec(compile(ast.Module(body=[mask], type_ignores=[]), str(path), 'exec'), scope)
            expected = [[True]*3, [False, False, True] if side == 'left' else [True, False, False]]
            self.assertEqual(scope['attention_masks'].tolist(), expected)


if __name__ == '__main__':
    unittest.main()
