# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch

import transformers
import tokenizers

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer, sanitize_generation_config_for_save
from llava.train.vsp_gradient_controller import validate_vsp_gradient_config

from llava import conversation as conversation_lib
from llava.model import *
from llava.model.llava_arch import validate_cka_loss_tau
from llava.mm_utils import tokenizer_image_token

from PIL import Image


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def get_parameter_numel(param):
    ds_numel = getattr(param, "ds_numel", None)
    if ds_numel is not None:
        if hasattr(ds_numel, "item"):
            ds_numel = ds_numel.item()
        return int(ds_numel)
    return param.numel()


def format_parameter_shape(param):
    ds_shape = getattr(param, "ds_shape", None)
    if ds_shape is not None:
        return list(ds_shape)
    if param.numel() == 0 and getattr(param, "ds_numel", None) is not None:
        return f"partitioned(numel={get_parameter_numel(param)})"
    return list(param.shape)


from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    force_download: bool = field(default=False)
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    student_vision_tower: Optional[str] = field(default=None)
    pretrain_student_mm_mlp_adapter: Optional[str] = field(default=None)
    router_hidden_size: Optional[int] = field(default=None)
    router_dropout: Optional[float] = field(default=None)
    guided_text_select_layer: Optional[int] = field(default=None)
    mtd_topk: Optional[int] = field(default=None)
    cka_loss: bool = field(default=False)
    use_pcgrad: bool = field(default=False, metadata={"help": "Legacy alias for --vsp_asymmetric_pcgrad True."})
    vsp_gradient_diagnostics: bool = field(default=False, metadata={"help": "Log gradient-conflict diagnostics for LM vs weighted VSP/CKA losses without changing the update."})
    vsp_asymmetric_pcgrad: bool = field(default=False, metadata={"help": "Protect the LM gradient and project only conflicting weighted VSP/CKA auxiliary gradients."})
    vsp_norm_cap: bool = field(default=False, metadata={"help": "Cap weighted VSP/CKA auxiliary gradients relative to the LM gradient by parameter group."})
    vsp_pcgrad_threshold: float = field(default=0.05)
    vsp_proj_max_grad_ratio: float = field(default=0.5)
    vsp_llm_max_grad_ratio: float = field(default=0.1)
    vsp_grad_ema_beta: float = field(default=0.95)
    vsp_grad_log_interval: int = field(default=10)
    vsp_grad_eps: float = field(default=1e-12)
    cka_loss_tau: float = field(default=0.0, metadata={"help": "Tolerated raw CKA loss tau in [0, 1]. Uses max(0, 1 - CKA - tau); tau=0 preserves the legacy objective."})
    cka_loss_weight: float = field(default=1.0)
    cka_loss_projector_weight: Optional[float] = field(default=None, metadata={"help": "Weight for the projector CKA loss. Defaults to cka_loss_weight for backward compatibility."})
    cka_loss_final_hidden_weight: Optional[float] = field(default=None, metadata={"help": "Weight for the chained LLM hidden-state CKA loss. Defaults to cka_loss_weight for backward compatibility."})
    # CKA has two terms: projector CKA always follows `cka_loss`, while this
    # option controls the chained LLM-hidden CKA term.
    cka_loss_layers: Optional[str] = field(default="final", metadata={"help": "Comma-separated 1-based LLM layer indices and/or 'final' for chained LLM-hidden CKA: post_projector->first->next->...->final, e.g. '8,16,24,final'. Use 'all' for every block, 'every4' or 'interval:4' for every k-th block, and '-1' to disable this term. Hidden-chain CKA is supported by LLaMA/Qwen; Mistral/MPT use projector CKA only."})
    cka_loss_layer_decay: float = field(default=1.0, metadata={"help": "Deprecated; retained for compatibility with older consecutive-layer CKA runs."})
    # 1-based layer used only to rank/select important image tokens by
    # text-to-image attention; it is not the hidden layer used for CKA.
    cka_loss_subset_select_layer: Optional[int] = field(default=None, metadata={"help": "LLM transformer layer index (1-based) whose text-to-image attention is used to select important image tokens for later CKA layers."})
    cka_loss_subset_query_tokens: str = field(default="text", metadata={"help": "Query tokens for attention-based CKA subset selection: 'text' for all non-image text tokens, or 'instruction' for non-answer prompt tokens (labels == IGNORE_INDEX)."})
    cka_loss_subset_ratio: float = field(default=0.5, metadata={"help": "Legacy max-ratio cap for attention-selected CKA image tokens when cka_loss_subset_max_ratio is not set."})
    cka_loss_subset_min_ratio: float = field(default=0.10, metadata={"help": "Minimum fraction of image tokens kept by dynamic attention/Otsu CKA subset selection."})
    cka_loss_subset_max_ratio: Optional[float] = field(default=0.90, metadata={"help": "Maximum fraction of image tokens kept by dynamic attention/Otsu CKA subset selection. Defaults to cka_loss_subset_ratio for compatibility."})
    cka_loss_subset_fallback_mass: float = field(default=0.90, metadata={"help": "Cumulative attention mass to keep when Otsu separability is too low."})
    cka_loss_subset_otsu_min_separability: float = field(default=0.30, metadata={"help": "Minimum Otsu between-class separability before falling back to cumulative attention mass."})


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    train_data_fraction: float = field(default=1.0, metadata={"help": "Fraction of the training JSON to use. Set below 1.0 for quick debug runs."})
    train_data_seed: int = field(default=42, metadata={"help": "Seed used when sampling train_data_fraction."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    debug_compare_cka: bool = field(default=False, metadata={"help": "Run a single-batch cka_loss off/on comparison and exit."})
    debug_compare_batch_size: int = field(default=1, metadata={"help": "Number of dataset items to collate for the CKA debug comparison."})
    debug_compare_batch_start: int = field(default=0, metadata={"help": "Starting dataset index for the CKA debug comparison batch."})
    debug_compare_seed: int = field(default=1234, metadata={"help": "Random seed reused for both CKA-off and CKA-on debug passes."})
    debug_compare_grad_param: Optional[str] = field(default=None, metadata={"help": "Optional substring used to choose a reference parameter when comparing gradients."})
    log_gradient_norms: bool = field(default=False, metadata={"help": "Log per-loss gradient norms for projector parameters and final hidden states."})
    gradient_log_steps: int = field(default=50, metadata={"help": "How often, in optimizer steps, to compute gradient norm debug logs."})


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_qwen(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    def encode_prompt(prompt: str) -> List[int]:
        if has_image:
            return tokenizer_image_token(prompt, tokenizer)
        return tokenizer(prompt).input_ids

    input_ids = []
    targets = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        prompt = conv.system + conv.sep
        assistant_spans = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            value = sentence["value"]
            if type(value) is tuple:
                value, _, _ = value

            if role == conv.roles[1]:
                answer_start = len(prompt) + len(role)
                prompt += role + value + conv.sep
                assistant_spans.append((answer_start, len(prompt)))
            else:
                prompt += role + value + conv.sep

        cur_input_ids = torch.tensor(
            encode_prompt(prompt)[:tokenizer.model_max_length],
            dtype=torch.long,
        )
        cur_target = torch.full_like(cur_input_ids, IGNORE_INDEX)

        for start, end in assistant_spans:
            start_token = len(encode_prompt(prompt[:start]))
            end_token = len(encode_prompt(prompt[:end]))
            start_token = min(start_token, cur_input_ids.shape[0])
            end_token = min(end_token, cur_input_ids.shape[0])
            if end_token > start_token:
                cur_target[start_token:end_token] = cur_input_ids[start_token:end_token]

        input_ids.append(cur_input_ids)
        targets.append(cur_target)

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=pad_token_id,
    )
    targets = torch.nn.utils.rnn.pad_sequence(
        targets,
        batch_first=True,
        padding_value=IGNORE_INDEX,
    )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    # Llama 3 chat headers and <|eot_id|> are easier to mask by assistant spans
    # than by separator token counts, especially with image-token placeholders.
    return preprocess_qwen(sources, tokenizer, has_image=has_image)


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "qwen2":
        return preprocess_qwen(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "llama3":
        return preprocess_llama3(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))
        original_size = len(list_data_dict)
        train_data_fraction = float(getattr(data_args, 'train_data_fraction', 1.0) or 1.0)
        if train_data_fraction <= 0.0 or train_data_fraction > 1.0:
            raise ValueError(f"train_data_fraction must be in (0, 1], got {train_data_fraction}")
        if original_size > 0 and train_data_fraction < 1.0:
            subset_size = max(1, int(original_size * train_data_fraction))
            train_data_seed = int(getattr(data_args, 'train_data_seed', 42))
            generator = torch.Generator()
            generator.manual_seed(train_data_seed)
            selected_indices = torch.randperm(original_size, generator=generator)[:subset_size].tolist()
            selected_indices.sort()
            list_data_dict = [list_data_dict[idx] for idx in selected_indices]
            rank0_print(
                f"Using {len(list_data_dict):,}/{original_size:,} training samples "
                f"({train_data_fraction:.2%}) with train_data_seed={train_data_seed}"
            )

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def run_cka_debug_compare(trainer: LLaVATrainer, training_args: TrainingArguments) -> Dict:
    dataset = trainer.train_dataset
    if dataset is None or len(dataset) == 0:
        raise ValueError("CKA debug compare requires a non-empty train_dataset.")

    batch_size = max(1, int(training_args.debug_compare_batch_size))
    start_index = max(0, int(training_args.debug_compare_batch_start))
    indices = [(start_index + offset) % len(dataset) for offset in range(batch_size)]
    instances = [dataset[idx] for idx in indices]
    batch = trainer.data_collator(instances)

    results = trainer.debug_compare_cka(
        batch=batch,
        seed=training_args.debug_compare_seed,
        grad_param_name=training_args.debug_compare_grad_param,
    )
    results["batch_indices"] = indices
    results["debug_seed"] = training_args.debug_compare_seed
    return results


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    if model_args.use_pcgrad:
        model_args.vsp_asymmetric_pcgrad = True
    vsp_gradient_requested = bool(
        model_args.vsp_gradient_diagnostics
        or model_args.vsp_asymmetric_pcgrad
        or model_args.vsp_norm_cap
    )
    validate_vsp_gradient_config(model_args)
    if vsp_gradient_requested and not model_args.cka_loss:
        rank0_print("Warning: VSP gradient diagnostics/controller has no effect unless --cka_loss is enabled.")
    if vsp_gradient_requested and training_args.gradient_checkpointing:
        if not hasattr(training_args, "gradient_checkpointing_kwargs"):
            raise ValueError(
                "VSP gradient diagnostics/controller with gradient checkpointing requires a Transformers version "
                "that supports gradient_checkpointing_kwargs."
            )
        checkpointing_kwargs = dict(training_args.gradient_checkpointing_kwargs or {})
        if checkpointing_kwargs.get("use_reentrant", True):
            rank0_print("VSP gradient controller: setting gradient checkpointing to use_reentrant=False.")
        checkpointing_kwargs["use_reentrant"] = False
        training_args.gradient_checkpointing_kwargs = checkpointing_kwargs
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    model_name_lower = model_args.model_name_or_path.lower()
    is_llama3_model = "llama-3" in model_name_lower or "llama3" in model_name_lower

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None:
        if 'mpt' in model_name_lower:
            config = transformers.AutoConfig.from_pretrained(
                model_args.model_name_or_path,
                trust_remote_code=True,
                force_download=model_args.force_download,
            )
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMptForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                force_download=model_args.force_download,
                **bnb_model_from_pretrained_args
            )
        elif 'qwen' in model_name_lower:
            model = LlavaQwenForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                force_download=model_args.force_download,
                **bnb_model_from_pretrained_args
            )
        elif 'mistral' in model_name_lower:
            model = LlavaMistralForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                force_download=model_args.force_download,
                **bnb_model_from_pretrained_args
            )
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                force_download=model_args.force_download,
                **bnb_model_from_pretrained_args
            )
    else:
        if 'qwen' in model_name_lower:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                force_download=model_args.force_download,
                **bnb_model_from_pretrained_args
            )
        else:
            model = transformers.LlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                force_download=model_args.force_download,
                **bnb_model_from_pretrained_args
            )

    if attn_implementation == "flash_attention_2" and training_args.bits not in [4, 8]:
        if not torch.cuda.is_available():
            raise ValueError("Flash Attention 2.0 requires CUDA, but CUDA is not available.")
        model.to(training_args.device)

    model.config.use_cache = False
    model.config.guided_text_select_layer = model_args.guided_text_select_layer
    model_args.text_hidden_size = model.config.hidden_size
    model.config.cka_loss = model_args.cka_loss
    model.config.use_pcgrad = bool(model_args.use_pcgrad)
    model.config.vsp_gradient_diagnostics = bool(model_args.vsp_gradient_diagnostics)
    model.config.vsp_asymmetric_pcgrad = bool(model_args.vsp_asymmetric_pcgrad)
    model.config.vsp_norm_cap = bool(model_args.vsp_norm_cap)
    model.config.vsp_pcgrad_threshold = float(model_args.vsp_pcgrad_threshold)
    model.config.vsp_proj_max_grad_ratio = float(model_args.vsp_proj_max_grad_ratio)
    model.config.vsp_llm_max_grad_ratio = float(model_args.vsp_llm_max_grad_ratio)
    model.config.vsp_grad_ema_beta = float(model_args.vsp_grad_ema_beta)
    model.config.vsp_grad_log_interval = int(model_args.vsp_grad_log_interval)
    model.config.vsp_grad_eps = float(model_args.vsp_grad_eps)
    model.config.cka_loss_tau = validate_cka_loss_tau(model_args.cka_loss_tau)
    model.config.log_gradient_norms = training_args.log_gradient_norms
    model.config.cka_loss_weight = model_args.cka_loss_weight
    model.config.cka_loss_projector_weight = (
        float(model_args.cka_loss_projector_weight)
        if model_args.cka_loss_projector_weight is not None
        else float(model_args.cka_loss_weight)
    )
    model.config.cka_loss_final_hidden_weight = (
        float(model_args.cka_loss_final_hidden_weight)
        if model_args.cka_loss_final_hidden_weight is not None
        else float(model_args.cka_loss_weight)
    )
    model.config.cka_loss_layer_decay = max(0.0, min(1.0, float(model_args.cka_loss_layer_decay)))
    model.config.cka_loss_subset_select_layer = model_args.cka_loss_subset_select_layer
    cka_subset_query_tokens = str(model_args.cka_loss_subset_query_tokens or "text").lower().replace("_", "-")
    if cka_subset_query_tokens in ("all", "all-text", "text"):
        cka_subset_query_tokens = "text"
    elif cka_subset_query_tokens in ("instruction", "instructions", "prompt", "non-answer", "nonanswer"):
        cka_subset_query_tokens = "instruction"
    else:
        rank0_print(
            f"Warning: Invalid cka_loss_subset_query_tokens '{model_args.cka_loss_subset_query_tokens}', defaulting to text"
        )
        cka_subset_query_tokens = "text"
    model.config.cka_loss_subset_query_tokens = cka_subset_query_tokens
    model.config.cka_loss_subset_ratio = max(0.0, min(1.0, float(model_args.cka_loss_subset_ratio)))
    model.config.cka_loss_subset_min_ratio = max(0.0, min(1.0, float(model_args.cka_loss_subset_min_ratio)))
    if model_args.cka_loss_subset_max_ratio is None:
        model.config.cka_loss_subset_max_ratio = model.config.cka_loss_subset_ratio
    else:
        model.config.cka_loss_subset_max_ratio = max(0.0, min(1.0, float(model_args.cka_loss_subset_max_ratio)))
    if model.config.cka_loss_subset_max_ratio > 0.0:
        model.config.cka_loss_subset_max_ratio = max(
            model.config.cka_loss_subset_min_ratio,
            model.config.cka_loss_subset_max_ratio,
        )
    model.config.cka_loss_subset_fallback_mass = max(0.0, min(1.0, float(model_args.cka_loss_subset_fallback_mass)))
    model.config.cka_loss_subset_otsu_min_separability = max(0.0, min(1.0, float(model_args.cka_loss_subset_otsu_min_separability)))
    # cka_loss_layers accepts comma-separated 1-based decoder layer indices plus
    # the special final/pre-norm hidden state. It also accepts "all", which
    # expands to every transformer block output, and interval shorthands such as
    # "every4" or "interval:4". The requested hidden states are regularized as a
    # chain over image tokens: post_projector->first->next->... .
    if model_args.cka_loss_layers:
        cka_loss_layers_arg = model_args.cka_loss_layers.strip()
        cka_loss_layers_lower = cka_loss_layers_arg.lower()
        if cka_loss_layers_lower in ("-1", "none", "off", "false"):
            model.config.cka_loss_layers = [-1]
        elif cka_loss_layers_lower == "all":
            model.config.cka_loss_layers = "all"
        else:
            parsed_cka_layers = []
            invalid_cka_layer = None
            layers = getattr(model.get_model(), "layers", None) if hasattr(model, "get_model") else None
            num_cka_layers = len(layers) if layers is not None else int(getattr(model.config, "num_hidden_layers", 0) or 0)

            def parse_interval_token(token_lower):
                for prefix in ("every", "interval:", "interval=", "stride:", "stride=", "step:", "step="):
                    if token_lower.startswith(prefix):
                        raw_interval = token_lower[len(prefix):].lstrip("_-")
                        try:
                            interval = int(raw_interval)
                        except ValueError:
                            return None
                        return interval if interval > 0 else None
                return None

            for raw_token in cka_loss_layers_arg.split(","):
                token = raw_token.strip()
                if not token:
                    continue
                token_lower = token.lower()
                if token_lower in ("-1", "none", "off", "false"):
                    if len([part for part in cka_loss_layers_arg.split(",") if part.strip()]) == 1:
                        parsed_cka_layers = [-1]
                        break
                    continue
                if token_lower in ("final", "last"):
                    parsed_cka_layers.append("final")
                    continue
                if token_lower == "all":
                    if num_cka_layers <= 0:
                        invalid_cka_layer = token
                        break
                    parsed_cka_layers.extend(range(1, num_cka_layers + 1))
                    continue
                interval = parse_interval_token(token_lower)
                if interval is not None:
                    if num_cka_layers <= 0:
                        invalid_cka_layer = token
                        break
                    parsed_cka_layers.extend(range(interval, num_cka_layers + 1, interval))
                    continue
                try:
                    layer_idx = int(token)
                except ValueError:
                    invalid_cka_layer = token
                    break
                if layer_idx <= 0:
                    invalid_cka_layer = token
                    break
                parsed_cka_layers.append(layer_idx)

            if parsed_cka_layers == [-1]:
                model.config.cka_loss_layers = [-1]
            elif invalid_cka_layer is not None or not parsed_cka_layers:
                rank0_print(
                    f"Warning: Invalid cka_loss_layers format '{model_args.cka_loss_layers}', "
                    "defaulting to final"
                )
                model.config.cka_loss_layers = "final"
            else:
                model.config.cka_loss_layers = parsed_cka_layers
    else:
        model.config.cka_loss_layers = "final"

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if 'mpt' in model_name_lower:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            force_download=model_args.force_download,
        )
    elif 'qwen' in model_name_lower:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            force_download=model_args.force_download,
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=is_llama3_model,
            force_download=model_args.force_download,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        if 'qwen' in model_name_lower:
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
            if tokenizer.pad_token_id is not None:
                model.config.pad_token_id = tokenizer.pad_token_id
        elif is_llama3_model:
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token
            if tokenizer.pad_token_id is not None:
                model.config.pad_token_id = tokenizer.pad_token_id
        else:
            tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        mm_projector = getattr(model.get_model(), "mm_projector", None)
        if mm_projector is not None:
            mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"✅ TRAINABLE: {name} | Shape: {format_parameter_shape(param)}")
    trainable_params = sum(get_parameter_numel(p) for p in model.parameters() if p.requires_grad)
    total_params = sum(get_parameter_numel(p) for p in model.parameters())
    trainable_pct = trainable_params / total_params if total_params > 0 else 0.0
    print(f"Trainable params: {trainable_params:,} | Total params: {total_params:,} | Trainable%: {trainable_pct:.2%}")
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    if training_args.debug_compare_cka:
        if hasattr(trainer, "_move_model_to_device"):
            trainer._move_model_to_device(trainer.model, training_args.device)
        else:
            trainer.model.to(training_args.device)
        if hasattr(trainer.model, "get_model"):
            inner_model = trainer.model.get_model()
            mm_projector = getattr(inner_model, "mm_projector", None)
            if mm_projector is not None:
                if training_args.bf16:
                    mm_projector.to(dtype=torch.bfloat16, device=training_args.device)
                elif training_args.fp16:
                    mm_projector.to(dtype=torch.float16, device=training_args.device)
        debug_results = run_cka_debug_compare(trainer, training_args)
        if training_args.local_rank in (-1, 0):
            print(json.dumps(debug_results, indent=2, sort_keys=True))
        return

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            sanitize_generation_config_for_save(model)
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
