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

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def _compute_masked_linear_cka_loss(
        self,
        projected_features: torch.FloatTensor,
        layer_hidden_states: torch.FloatTensor,
        vision_feature_mask: torch.BoolTensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        projected_features = projected_features.float()
        layer_hidden_states = layer_hidden_states.float()
        vision_feature_mask = vision_feature_mask.bool()

        cka_losses = []
        for i in range(projected_features.shape[0]):
            cur_mask = vision_feature_mask[i]
            if cur_mask.sum() < 2:
                continue

            x_i = projected_features[i][cur_mask]
            y_i = layer_hidden_states[i][cur_mask]

            x_i = x_i - x_i.mean(dim=0, keepdim=True)
            y_i = y_i - y_i.mean(dim=0, keepdim=True)

            xx = x_i @ x_i.T
            yy = y_i @ y_i.T

            hsic_xy = (xx * yy).sum()
            hsic_xx = xx.square().sum()
            hsic_yy = yy.square().sum()

            denom = torch.sqrt(torch.clamp(hsic_xx * hsic_yy, min=eps))
            cka_i = (hsic_xy / denom).clamp(0.0, 1.0)
            cka_losses.append(1.0 - cka_i)

        if len(cka_losses) == 0:
            return projected_features.new_zeros(())

        return torch.stack(cka_losses).mean()

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
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        cka_enabled = self.get_model().training and self.get_model().config.cka_loss
        vision_feature_mask = None
        pre_post_cka_loss = None

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

        should_output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        if cka_enabled:
            should_output_hidden_states = True

        output =  super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=should_output_hidden_states,
            return_dict=return_dict
        )

        if cka_enabled and output.loss is not None:
            cka_loss = output.loss.new_zeros(())
            cka_component_count = 0
            if pre_post_cka_loss is not None:
                cka_loss = cka_loss + pre_post_cka_loss.to(output.loss.device)
                cka_component_count += 1

            cka_layers_loss = output.loss.new_zeros(())
            cka_loss_layers = getattr(self.get_model().config, 'cka_loss_layers', [1])
            
            # Parse cka_loss_layers: can be "all", list, or None
            if isinstance(cka_loss_layers, str):
                if cka_loss_layers.lower() == "all" and output.hidden_states is not None:
                    target_layers = list(range(len(output.hidden_states)))
                else:
                    target_layers = []
            else:
                target_layers = cka_loss_layers if cka_loss_layers else []
            
            if (
                vision_feature_mask is not None
                and output.hidden_states is not None
                and inputs_embeds is not None
                and target_layers
            ):
                per_layer_losses = {}
                valid_layer_count = 0
                for layer_idx in target_layers:
                    if layer_idx < 0 or layer_idx >= len(output.hidden_states):
                        continue
                    
                    # Compare layer_idx with layer_idx-1 (or inputs_embeds if layer_idx == 0)
                    layer_n_hidden = output.hidden_states[layer_idx]
                    
                    if layer_idx == 0:
                        # Compare layer 0 with projected image features (inputs_embeds)
                        # Detach to prevent gradients from flowing to the reference layer
                        layer_n_minus_1_hidden = inputs_embeds.detach()
                        loss_key = "0"
                    else:
                        # Compare layer_idx with layer_idx-1
                        # Detach to prevent gradients from flowing to the reference layer
                        layer_n_minus_1_hidden = output.hidden_states[layer_idx - 1].detach()
                        loss_key = f"{layer_idx}_{layer_idx-1}"
                    
                    layer_cka = self._compute_masked_linear_cka_loss(
                        projected_features=layer_n_hidden,
                        layer_hidden_states=layer_n_minus_1_hidden,
                        vision_feature_mask=vision_feature_mask,
                    ).to(output.loss.device)
                    per_layer_losses[loss_key] = layer_cka.detach()
                    cka_layers_loss = cka_layers_loss + layer_cka
                    valid_layer_count += 1
                
                # Store per-layer losses for logging
                self.last_cka_per_layer_losses = per_layer_losses
                cka_loss = cka_loss + cka_layers_loss
                cka_component_count += valid_layer_count
            else:
                self.last_cka_per_layer_losses = {}

            if cka_component_count > 0:
                cka_loss = cka_loss / cka_component_count

            # Store losses for logging
            self.last_cka_loss = cka_loss.detach()
            self.last_text_loss = output.loss.detach()
            self.last_cka_pre_post_loss = (
                pre_post_cka_loss.detach() if pre_post_cka_loss is not None else output.loss.new_zeros(()).detach()
            )
            self.last_cka_layers_loss = cka_layers_loss.detach()

            final_hidden_states = output.hidden_states if should_output_hidden_states else None
            
            return CausalLMOutputWithPast(
                loss=output.loss + self.get_model().config.cka_loss_weight * cka_loss,
                logits=output.logits,
                past_key_values=output.past_key_values,
                hidden_states=final_hidden_states,
                attentions=output.attentions,
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

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
