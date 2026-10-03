#    Copyright 2023 Haotian Liu
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
import warnings
import shutil
from llava.model.backbones import backbone_type, configure_tokenizer_padding, get_llava_model_class

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN




def _set_generation_pad_token(tokenizer, model, include_pad_as_eos=False):
    configure_tokenizer_padding(tokenizer, model.config)
    model.get_input_embeddings().padding_idx = tokenizer.pad_token_id

    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        return

    eos_ids = generation_config.eos_token_id
    if eos_ids is None:
        eos_ids = tokenizer.eos_token_id
    if eos_ids is not None and not isinstance(eos_ids, (list, tuple)):
        eos_ids = [eos_ids]
    eos_ids = list(eos_ids or [])
    terminal_ids = [tokenizer.eos_token_id]
    if include_pad_as_eos:
        terminal_ids.append(tokenizer.pad_token_id)
    for token_id in terminal_ids:
        if token_id is not None and token_id not in eos_ids:
            eos_ids.append(token_id)
    if eos_ids:
        generation_config.eos_token_id = eos_ids

    if tokenizer.pad_token_id is not None:
        generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.use_cache = True
    generation_config.output_attentions = False
    generation_config.output_hidden_states = False


def _checkpoint_family(model_path, model_base=None):
    from transformers import PretrainedConfig

    last_error = None
    for source in (model_path, model_base):
        if source is None:
            continue
        try:
            config_dict, _ = PretrainedConfig.get_config_dict(source, trust_remote_code=True)
            return backbone_type(config_dict["model_type"]), config_dict["model_type"], config_dict
        except (OSError, KeyError, ValueError) as exc:
            last_error = exc
    raise ValueError(f"Cannot determine backbone model_type for {model_path!r}: {last_error}")


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}
    family, checkpoint_type, config_dict = _checkpoint_family(model_path, model_base)
    is_llava = checkpoint_type.startswith("llava") or "llava" in model_name.lower()
    tokenizer_fast = family in ("qwen2", "qwen3", "gemma3_text", "phi3", "mpt") or (
        family == "llama" and int(config_dict.get("vocab_size", 0)) >= 128000
    )
    llava_cls = get_llava_model_class(family) if is_llava else None

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    if is_llava:
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            lora_cfg_pretrained = llava_cls.config_class.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=tokenizer_fast)
            print(f"Loading LLaVA-{family} from base model...")
            model = llava_cls.from_pretrained(
                model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs
            )
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            if family == "mpt" and not os.path.isfile(os.path.join(model_path, "configuration_mpt.py")):
                shutil.copyfile(
                    os.path.join(model_base, "configuration_mpt.py"),
                    os.path.join(model_path, "configuration_mpt.py"),
                )
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=tokenizer_fast)
            cfg_pretrained = llava_cls.config_class.from_pretrained(
                model_path, trust_remote_code=(family == "mpt")
            )
            model = llava_cls.from_pretrained(
                model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs
            )

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=tokenizer_fast)
            model = llava_cls.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                **kwargs
            )
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=tokenizer_fast)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=tokenizer_fast)
            model = AutoModelForCausalLM.from_pretrained(
                model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs
            )

    _set_generation_pad_token(
        tokenizer, model,
        include_pad_as_eos=family in ("qwen2", "qwen3"),
    )

    image_processor = None

    if is_llava:
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
            vision_tower.to(device=model.device, dtype=model.dtype)
        if device_map != 'auto':
            vision_tower.to(device=device_map, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
