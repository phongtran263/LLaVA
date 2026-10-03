"""Select LLaVA wrappers from checkpoint metadata, including renamed local runs."""
from importlib import import_module

_BACKBONES = {
    "llama": ("llava_llama", "LlavaLlamaForCausalLM"),
    "qwen2": ("llava_qwen", "LlavaQwenForCausalLM"),
    "qwen3": ("llava_modern", "LlavaQwen3ForCausalLM"),
    "gemma3_text": ("llava_modern", "LlavaGemma3ForCausalLM"),
    "phi3": ("llava_modern", "LlavaPhi3ForCausalLM"),
    "mistral": ("llava_mistral", "LlavaMistralForCausalLM"),
    "mpt": ("llava_mpt", "LlavaMptForCausalLM"),
}
_ALIASES = {
    "llava": "llama", "llava_llama": "llama", "llava_qwen": "qwen2",
    "llava_qwen3": "qwen3", "llava_gemma3": "gemma3_text",
    "llava_phi3": "phi3", "llava_mistral": "mistral", "llava_mpt": "mpt",
}


def backbone_type(config):
    kind = getattr(config, "model_type", config)
    return _ALIASES.get(kind, kind)


def get_llava_model_class(config):
    kind = backbone_type(config)
    if kind not in _BACKBONES:
        raise ValueError(
            f"Unsupported LLaVA backbone model_type={kind!r}. "
            f"Supported: {', '.join(_BACKBONES)}. Gemma support is the 1B text decoder."
        )
    module_name, class_name = _BACKBONES[kind]
    try:
        module = import_module(f"llava.model.language_model.{module_name}")
    except ImportError as exc:
        raise ImportError(f"{kind} requires the training dependencies in pyproject.toml "
                          "(transformers==4.51.3).") from exc
    return getattr(module, class_name)


def as_llava_config(config):
    model_cls = get_llava_model_class(config)
    if isinstance(config, model_cls.config_class):
        return config
    values = config.to_dict()
    values.pop("model_type", None)
    converted = model_cls.config_class.from_dict(values)
    converted.model_type = model_cls.config_class.model_type
    return converted


def is_llama3(config):
    return backbone_type(config) == "llama" and config.vocab_size >= 128000


def use_fast_tokenizer(config):
    return is_llama3(config) or backbone_type(config) in {
        "qwen2", "qwen3", "gemma3_text", "phi3", "mpt",
    }


def configure_tokenizer_padding(tokenizer, config):
    """Keep Llama-3 turn boundaries distinct from padding, without adding tokens."""
    if is_llama3(config) and tokenizer.pad_token_id in (
        None, tokenizer.eos_token_id, tokenizer.bos_token_id,
    ):
        native_pad = "<|finetune_right_pad_id|>"
        if native_pad in tokenizer.get_vocab():
            tokenizer.pad_token = native_pad
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or tokenizer.bos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer has no padding token or usable fallback token.")
    config.pad_token_id = tokenizer.pad_token_id


def default_conversation(config):
    family = backbone_type(config)
    model_name = str(getattr(config, "_name_or_path", "") or "").lower()
    if family == "qwen3" and "instruct" in model_name:
        return "qwen3_instruct"
    if family == "llama" and "tinyllama" in model_name:
        return "tinyllama"
    if is_llama3(config):
        return "llama3"
    return {
        "llama": "v1", "qwen2": "qwen2", "qwen3": "qwen3",
        "gemma3_text": "gemma3", "phi3": "phi3",
        "mistral": "mistral_instruct", "mpt": "mpt",
    }[family]
