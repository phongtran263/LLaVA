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


from typing import List, Optional, Tuple, Union
import inspect
import math

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         Qwen2Config, Qwen2Model, Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from .llava_llama import CausalLMOutputWithPastAux, LlavaLlamaForCausalLM


_QWEN2_FORWARD_SUPPORTS_CACHE_POSITION = (
    "cache_position" in inspect.signature(Qwen2ForCausalLM.forward).parameters
)

class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig
    _compute_masked_linear_cka_loss = LlavaLlamaForCausalLM._compute_masked_linear_cka_loss
    _get_cka_attention_subset_kwargs = LlavaLlamaForCausalLM._get_cka_attention_subset_kwargs
    _fallback_keep_count_from_attention_mass = LlavaLlamaForCausalLM._fallback_keep_count_from_attention_mass
    _otsu_keep_count_from_log_probs = LlavaLlamaForCausalLM._otsu_keep_count_from_log_probs
    _select_topk_indices_from_attention_scores = LlavaLlamaForCausalLM._select_topk_indices_from_attention_scores
    _select_vision_feature_subset_from_attention = LlavaLlamaForCausalLM._select_vision_feature_subset_from_attention
    _get_cka_attention_query_mask = LlavaLlamaForCausalLM._get_cka_attention_query_mask
    _get_cka_layer_specs = LlavaLlamaForCausalLM._get_cka_layer_specs
    _register_cka_layer_hooks = LlavaLlamaForCausalLM._register_cka_layer_hooks
    _iter_cka_layer_hiddens = LlavaLlamaForCausalLM._iter_cka_layer_hiddens
    _compute_cka_vision_reference_losses = LlavaLlamaForCausalLM._compute_cka_vision_reference_losses

    def __init__(self, config):
        config.model_type = "llava_qwen"
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = LlavaQwenModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    @torch.no_grad()
    def _select_vision_feature_subset_from_attention_inputs(
        self,
        attention_module: nn.Module,
        hidden_states: torch.FloatTensor,
        vision_feature_mask: torch.BoolTensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        past_key_value,
        min_keep_ratio: float,
        max_keep_ratio: float,
        fallback_mass: float,
        otsu_min_separability: float,
        labels: Optional[torch.LongTensor] = None,
        min_keep_tokens: int = 2,
    ) -> Optional[torch.BoolTensor]:
        if hidden_states is None or vision_feature_mask is None:
            return None

        if hidden_states.ndim != 3 or vision_feature_mask.ndim != 2:
            return None

        bsz, q_len, _ = hidden_states.size()
        if vision_feature_mask.shape != (bsz, q_len):
            return None

        if past_key_value is not None:
            try:
                cached_len = past_key_value.get_usable_length(
                    q_len,
                    getattr(attention_module, "layer_idx", None),
                )
            except AttributeError:
                return None
            if cached_len:
                return None

        if attention_mask is None:
            valid_attention_mask = torch.ones_like(vision_feature_mask, dtype=torch.bool)
        elif attention_mask.ndim == 2:
            valid_attention_mask = attention_mask.to(device=vision_feature_mask.device).bool()
        else:
            return None

        if valid_attention_mask.shape != (bsz, q_len):
            return None

        if max_keep_ratio <= 0.0:
            return vision_feature_mask & valid_attention_mask

        if position_ids is None:
            position_ids = torch.arange(q_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0)
        elif position_ids.shape[0] == 1 and bsz > 1:
            position_ids = position_ids.expand(bsz, -1)

        query_states = attention_module.q_proj(hidden_states)
        key_states = attention_module.k_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, attention_module.num_heads, attention_module.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, attention_module.num_key_value_heads, attention_module.head_dim
        ).transpose(1, 2)

        rotary_seq_len = q_len
        if position_ids is not None:
            rotary_seq_len = max(q_len, int(position_ids.max().item()) + 1)
        cos, sin = attention_module.rotary_emb(key_states, seq_len=rotary_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        key_states = repeat_kv(key_states, attention_module.num_key_value_groups)

        selected_masks = []
        scale = 1.0 / math.sqrt(attention_module.head_dim)

        for batch_idx in range(bsz):
            valid_mask = valid_attention_mask[batch_idx]
            image_mask = vision_feature_mask[batch_idx] & valid_mask
            cur_labels = labels[batch_idx] if labels is not None else None
            text_mask = self._get_cka_attention_query_mask(
                vision_feature_mask[batch_idx],
                valid_mask,
                cur_labels,
            )

            image_token_count = int(image_mask.sum().item())
            text_token_count = int(text_mask.sum().item())
            if image_token_count < min_keep_tokens or text_token_count == 0:
                selected_masks.append(image_mask.clone())
                continue

            text_positions = torch.where(text_mask)[0]
            valid_positions = torch.where(valid_mask)[0]
            image_positions = torch.where(image_mask)[0]

            q_i = query_states[batch_idx, :, text_positions, :]
            k_i = key_states[batch_idx, :, valid_positions, :]
            attn_scores = torch.matmul(q_i, k_i.transpose(-1, -2)).float() * scale

            causal_mask = valid_positions.unsqueeze(0) <= text_positions.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(
                ~causal_mask.unsqueeze(0),
                torch.finfo(attn_scores.dtype).min,
            )

            attn_probs = torch.softmax(attn_scores, dim=-1)
            image_columns = image_mask[valid_positions]
            text_to_image = attn_probs[:, :, image_columns]
            if text_to_image.numel() == 0:
                selected_masks.append(image_mask.clone())
                continue

            image_scores = text_to_image.mean(dim=(0, 1))
            topk_indices = self._select_topk_indices_from_attention_scores(
                image_scores=image_scores,
                min_keep_tokens=min_keep_tokens,
                min_keep_ratio=min_keep_ratio,
                max_keep_ratio=max_keep_ratio,
                fallback_mass=fallback_mass,
                otsu_min_separability=otsu_min_separability,
            )

            selected_mask = torch.zeros_like(image_mask)
            selected_mask[image_positions[topk_indices]] = True
            selected_masks.append(selected_mask)

        return torch.stack(selected_masks, dim=0)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        cka_enabled = self.get_model().training and getattr(self.get_model().config, 'cka_loss', False)
        vision_feature_mask = None
        subset_vision_feature_mask = None
        pre_post_cka_loss = None
        pre_projector_features = None
        self.last_cka_loss = None
        self.last_cka_projector_loss = None
        self.last_cka_pre_post_loss = None
        self.last_cka_pre_final_loss = None
        self.last_cka_layers_loss = None
        self.last_cka_per_layer_losses = {}
        self.last_cka_subset_vision_feature_mask = None
        self.last_cka_final_hidden = None
        self.last_cka_projector_output = None

        if inputs_embeds is None:
            if cka_enabled:
                (
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels,
                    vision_feature_mask,
                    pre_post_cka_loss,
                    pre_projector_features,
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    images,
                    image_sizes
                )
            else:
                (
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    images,
                    image_sizes
                )

        cka_layer_specs = self._get_cka_layer_specs() if cka_enabled else []
        llm_cka_enabled = cka_enabled and len(cka_layer_specs) > 0
        should_output_hidden_states = output_hidden_states
        should_output_attentions = output_attentions
        subset_select_layer = getattr(self.get_model().config, 'cka_loss_subset_select_layer', None)
        subset_selection_kwargs = self._get_cka_attention_subset_kwargs()
        uses_final_cka_layer = any(spec["kind"] == "final" for spec in cka_layer_specs)

        final_layer_pre_norm_hidden = None
        norm_pre_hook_handle = None
        if llm_cka_enabled and uses_final_cka_layer and hasattr(self.get_model(), "norm"):
            def capture_final_layer_pre_norm_hidden(module, module_inputs):
                nonlocal final_layer_pre_norm_hidden
                if module_inputs:
                    final_layer_pre_norm_hidden = module_inputs[0]

            norm_pre_hook_handle = self.get_model().norm.register_forward_pre_hook(
                capture_final_layer_pre_norm_hidden
            )

        captured_cka_layer_hiddens = {}
        cka_layer_hook_handles = []
        if llm_cka_enabled:
            cka_layer_hook_handles = self._register_cka_layer_hooks(
                cka_layer_specs,
                captured_cka_layer_hiddens,
            )

        attention_subset_hook_handle = None
        captured_subset_vision_feature_mask = None
        if (
            llm_cka_enabled
            and subset_select_layer is not None
            and vision_feature_mask is not None
            and inputs_embeds is not None
        ):
            layers = getattr(self.get_model(), "layers", None)
            layer_idx = int(subset_select_layer) - 1
            if layers is not None and 0 <= layer_idx < len(layers):
                selected_self_attn = layers[layer_idx].self_attn

                def capture_attention_subset_from_layer(module, module_inputs, module_kwargs):
                    nonlocal captured_subset_vision_feature_mask
                    hook_hidden_states = module_kwargs.get("hidden_states", None)
                    if hook_hidden_states is None and module_inputs:
                        hook_hidden_states = module_inputs[0]

                    captured_mask = self._select_vision_feature_subset_from_attention_inputs(
                        attention_module=module,
                        hidden_states=hook_hidden_states,
                        vision_feature_mask=vision_feature_mask,
                        attention_mask=attention_mask,
                        position_ids=module_kwargs.get("position_ids", position_ids),
                        past_key_value=module_kwargs.get("past_key_value", None),
                        labels=labels,
                        **subset_selection_kwargs,
                    )
                    if captured_mask is not None:
                        captured_subset_vision_feature_mask = captured_mask

                attention_subset_hook_handle = selected_self_attn.register_forward_pre_hook(
                    capture_attention_subset_from_layer,
                    with_kwargs=True,
                )

        forward_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=should_output_attentions,
            output_hidden_states=should_output_hidden_states,
            return_dict=return_dict,
        )
        if _QWEN2_FORWARD_SUPPORTS_CACHE_POSITION:
            forward_kwargs["cache_position"] = cache_position

        try:
            output = super().forward(**forward_kwargs)
        finally:
            if norm_pre_hook_handle is not None:
                norm_pre_hook_handle.remove()
            if attention_subset_hook_handle is not None:
                attention_subset_hook_handle.remove()
            for hook_handle in cka_layer_hook_handles:
                hook_handle.remove()

        if (
            llm_cka_enabled
            and subset_select_layer is not None
            and vision_feature_mask is not None
            and inputs_embeds is not None
        ):
            subset_vision_feature_mask = captured_subset_vision_feature_mask
            if subset_vision_feature_mask is None and output.attentions is not None:
                subset_vision_feature_mask = self._select_vision_feature_subset_from_attention(
                    attentions=output.attentions,
                    vision_feature_mask=vision_feature_mask,
                    attention_mask=attention_mask,
                    select_layer=subset_select_layer,
                    labels=labels,
                    **subset_selection_kwargs,
                )
            if subset_vision_feature_mask is None:
                subset_vision_feature_mask = vision_feature_mask

        if cka_enabled and output.loss is not None:
            projector_cka_loss = output.loss.new_zeros(())
            if pre_post_cka_loss is not None:
                projector_cka_loss = pre_post_cka_loss.to(output.loss.device)

            cka_layers_loss = output.loss.new_zeros(())
            final_hidden = final_layer_pre_norm_hidden
            if uses_final_cka_layer and final_hidden is None and output.hidden_states is not None:
                final_hidden = output.hidden_states[-1]
            if getattr(self.get_model().config, 'log_gradient_norms', False):
                self.last_cka_final_hidden = final_hidden

            if vision_feature_mask is not None and pre_projector_features is not None:
                layer_mask = subset_vision_feature_mask if subset_vision_feature_mask is not None else vision_feature_mask
                layer_losses, per_layer_losses = self._compute_cka_vision_reference_losses(
                    cka_layer_specs=cka_layer_specs,
                    captured_layer_hiddens=captured_cka_layer_hiddens,
                    final_hidden=final_hidden,
                    output_hidden_states=output.hidden_states,
                    vision_encoder_features=pre_projector_features,
                    vision_feature_mask=layer_mask,
                    output_device=output.loss.device,
                )

                if layer_losses:
                    cka_layers_loss = torch.stack(layer_losses).sum()
                self.last_cka_per_layer_losses = per_layer_losses
                self.last_cka_subset_vision_feature_mask = (
                    subset_vision_feature_mask.detach() if subset_vision_feature_mask is not None else None
                )
            else:
                self.last_cka_subset_vision_feature_mask = (
                    subset_vision_feature_mask.detach() if subset_vision_feature_mask is not None else None
                )

            cka_loss = projector_cka_loss + cka_layers_loss

            self.last_cka_loss = cka_loss.detach()
            self.last_cka_projector_loss = projector_cka_loss.detach()
            self.last_text_loss = output.loss.detach()
            self.last_cka_pre_post_loss = projector_cka_loss.detach()
            self.last_cka_pre_final_loss = cka_layers_loss.detach()
            self.last_cka_layers_loss = cka_layers_loss.detach()
            self._aux_losses = [cka_layers_loss]
            default_cka_weight = getattr(self.get_model().config, 'cka_loss_weight', 1.0)
            projector_cka_weight = getattr(self.get_model().config, 'cka_loss_projector_weight', None)
            if projector_cka_weight is None:
                projector_cka_weight = default_cka_weight
            final_hidden_cka_weight = getattr(self.get_model().config, 'cka_loss_final_hidden_weight', None)
            if final_hidden_cka_weight is None:
                final_hidden_cka_weight = default_cka_weight

            return CausalLMOutputWithPastAux(
                loss=output.loss,
                logits=output.logits,
                past_key_values=output.past_key_values,
                hidden_states=None,
                attentions=None,
                projector_cka_loss=projector_cka_loss * projector_cka_weight,
                aux_losses=[cka_layers_loss * final_hidden_cka_weight]
            )
        else:
            return output

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)
