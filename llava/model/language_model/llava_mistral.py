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
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         MistralConfig, MistralModel, MistralForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from .llava_llama import CausalLMOutputWithPastAux, LlavaLlamaForCausalLM


_MISTRAL_FORWARD_SUPPORTS_CACHE_POSITION = (
    "cache_position" in inspect.signature(MistralForCausalLM.forward).parameters
)


class LlavaMistralConfig(MistralConfig):
    model_type = "llava_mistral"


class LlavaMistralModel(LlavaMetaModel, MistralModel):
    config_class = LlavaMistralConfig

    def __init__(self, config: MistralConfig):
        super(LlavaMistralModel, self).__init__(config)


class LlavaMistralForCausalLM(MistralForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaMistralConfig
    _compute_masked_linear_cka_loss = LlavaLlamaForCausalLM._compute_masked_linear_cka_loss
    _get_cka_layer_specs = LlavaLlamaForCausalLM._get_cka_layer_specs
    _register_cka_layer_hooks = LlavaLlamaForCausalLM._register_cka_layer_hooks
    _iter_cka_layer_hiddens = LlavaLlamaForCausalLM._iter_cka_layer_hiddens
    _compute_cka_pre_ffn_losses = LlavaLlamaForCausalLM._compute_cka_pre_ffn_losses

    def __init__(self, config):
        super(MistralForCausalLM, self).__init__(config)
        self.model = LlavaMistralModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

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
        cka_enabled = self.get_model().training and getattr(
            self.get_model().config, 'cka_loss', False
        )
        vision_feature_mask = None
        projector_cka_loss = None
        self.last_cka_loss = None
        self.last_cka_projector_loss = None
        self.last_cka_pre_post_loss = None
        self.last_cka_pre_final_loss = None
        self.last_cka_layers_loss = None
        self.last_cka_per_layer_losses = {}
        self.last_cka_subset_vision_feature_mask = None
        self.last_cka_final_hidden = None

        if inputs_embeds is None:
            prepared_inputs = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
            )
            if cka_enabled:
                (
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels,
                    vision_feature_mask,
                    projector_cka_loss,
                    _,
                ) = prepared_inputs
            else:
                (
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels,
                ) = prepared_inputs

        cka_layer_specs = self._get_cka_layer_specs() if cka_enabled else []
        hidden_weight = getattr(self.config, 'cka_loss_final_hidden_weight', None)
        if hidden_weight is None:
            hidden_weight = getattr(self.config, 'cka_loss_weight', 1.0)
        llm_cka_enabled = cka_enabled and bool(cka_layer_specs) and float(hidden_weight) != 0.0

        captured_cka_layer_hiddens = {}
        captured_cka_pre_ffn_hiddens = {}
        cka_layer_hook_handles = []
        if llm_cka_enabled:
            cka_layer_hook_handles = self._register_cka_layer_hooks(
                cka_layer_specs,
                captured_cka_layer_hiddens,
                captured_cka_pre_ffn_hiddens,
            )

        forward_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True if cka_enabled else return_dict,
        )
        if cache_position is not None and _MISTRAL_FORWARD_SUPPORTS_CACHE_POSITION:
            forward_kwargs["cache_position"] = cache_position

        try:
            output = super().forward(**forward_kwargs)
        finally:
            for hook_handle in cka_layer_hook_handles:
                hook_handle.remove()

        if not cka_enabled or output.loss is None:
            return output

        if projector_cka_loss is None:
            projector_cka_loss = output.loss.new_zeros(())
        else:
            projector_cka_loss = projector_cka_loss.to(output.loss.device)

        cka_layers_loss = output.loss.new_zeros(())
        if llm_cka_enabled and vision_feature_mask is not None:
            layer_losses, per_layer_losses = self._compute_cka_pre_ffn_losses(
                cka_layer_specs=cka_layer_specs,
                captured_layer_hiddens=captured_cka_layer_hiddens,
                captured_pre_ffn_hiddens=captured_cka_pre_ffn_hiddens,
                final_hidden=None,
                output_hidden_states=output.hidden_states,
                vision_feature_mask=vision_feature_mask,
                output_device=output.loss.device,
            )
            if layer_losses:
                cka_layers_loss = torch.stack(layer_losses).sum()
            self.last_cka_per_layer_losses = per_layer_losses

        if getattr(self.get_model().config, 'log_gradient_norms', False):
            for spec in cka_layer_specs:
                if spec['kind'] == 'final':
                    self.last_cka_final_hidden = captured_cka_layer_hiddens.get(spec['name'])
                    break

        cka_loss = projector_cka_loss + cka_layers_loss
        self.last_cka_loss = cka_loss.detach()
        self.last_cka_projector_loss = projector_cka_loss.detach()
        self.last_cka_pre_post_loss = projector_cka_loss.detach()
        self.last_cka_pre_final_loss = cka_layers_loss.detach()
        self.last_cka_layers_loss = cka_layers_loss.detach()
        self.last_text_loss = output.loss.detach()
        self._aux_losses = [cka_layers_loss] if llm_cka_enabled else []

        projector_weight = getattr(self.get_model().config, 'cka_loss_projector_weight', None)
        if projector_weight is None:
            projector_weight = getattr(self.get_model().config, 'cka_loss_weight', 1.0)

        return CausalLMOutputWithPastAux(
            loss=output.loss,
            logits=output.logits,
            past_key_values=output.past_key_values,
            hidden_states=None,
            attentions=None,
            projector_cka_loss=projector_cka_loss * projector_weight,
            aux_losses=[cka_layers_loss * hidden_weight] if llm_cka_enabled else [],
        )

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

AutoConfig.register("llava_mistral", LlavaMistralConfig)
AutoModelForCausalLM.register(LlavaMistralConfig, LlavaMistralForCausalLM)
