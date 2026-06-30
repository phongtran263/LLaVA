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
from dataclasses import dataclass

import math
import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from llava.constants import IGNORE_INDEX
from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)

@dataclass
class CausalLMOutputWithPastAux(CausalLMOutputWithPast):
    projector_cka_loss: Optional[torch.Tensor] = None
    aux_losses: Optional[List[torch.Tensor]] = None

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
        """
        CKA term between aligned image-token features.

        The caller passes two aligned feature tensors and only positions marked
        by `vision_feature_mask` participate. Feature dimensions may differ; CKA
        compares token-token Gram structure.
        """
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

    def _get_cka_layer_specs(self):
        raw_layers = getattr(self.get_model().config, 'cka_loss_layers', "final")
        if raw_layers == [-1] or raw_layers in (None, "", False):
            return []

        if isinstance(raw_layers, str):
            tokens = [token.strip().lower() for token in raw_layers.split(",") if token.strip()]
        elif isinstance(raw_layers, (list, tuple)):
            tokens = list(raw_layers)
        else:
            tokens = [raw_layers]

        if not tokens:
            return []

        layers = getattr(self.get_model(), "layers", None)
        num_layers = len(layers) if layers is not None else 0
        specs = []
        seen = set()

        def add_final():
            key = ("final", None)
            if key not in seen:
                specs.append({"kind": "final", "name": "final", "layer_idx": None})
                seen.add(key)

        def add_layer(layer_idx):
            if num_layers <= 0 or layer_idx < 1 or layer_idx > num_layers:
                return
            key = ("layer", layer_idx)
            if key not in seen:
                specs.append({"kind": "layer", "name": f"layer_{layer_idx}", "layer_idx": layer_idx})
                seen.add(key)

        for token in tokens:
            token_lower = token.lower() if isinstance(token, str) else token
            if token_lower in ("-1", "none", "off", "false"):
                if len(tokens) == 1:
                    return []
                continue
            if token_lower in ("final", "last"):
                add_final()
                continue
            if token_lower == "all":
                for layer_idx in range(1, num_layers + 1):
                    add_layer(layer_idx)
                continue

            try:
                layer_idx = int(token)
            except (TypeError, ValueError):
                continue
            if layer_idx == -1 and len(tokens) == 1:
                return []
            add_layer(layer_idx)

        if not specs:
            add_final()

        return specs

    def _register_cka_layer_hooks(self, cka_layer_specs, captured_layer_hiddens):
        layers = getattr(self.get_model(), "layers", None)
        if layers is None:
            return []

        handles = []
        for spec in cka_layer_specs:
            if spec["kind"] != "layer":
                continue
            layer_idx = spec["layer_idx"]
            if layer_idx is None or layer_idx < 1 or layer_idx > len(layers):
                continue
            layer_name = spec["name"]

            def capture_layer_hidden(module, module_inputs, module_outputs, layer_name=layer_name):
                hidden_states = module_outputs[0] if isinstance(module_outputs, (tuple, list)) else module_outputs
                if torch.is_tensor(hidden_states):
                    captured_layer_hiddens[layer_name] = hidden_states

            handles.append(layers[layer_idx - 1].register_forward_hook(capture_layer_hidden))

        return handles

    def _iter_cka_layer_hiddens(
        self,
        cka_layer_specs,
        captured_layer_hiddens,
        final_hidden,
        output_hidden_states=None,
    ):
        for spec in cka_layer_specs:
            if spec["kind"] == "final":
                hidden_states = final_hidden
            else:
                hidden_states = captured_layer_hiddens.get(spec["name"])
                if hidden_states is None and output_hidden_states is not None:
                    layer_idx = spec["layer_idx"]
                    if layer_idx is not None and 0 <= layer_idx < len(output_hidden_states):
                        hidden_states = output_hidden_states[layer_idx]

            if torch.is_tensor(hidden_states):
                yield spec["name"], hidden_states

    def _compute_cka_chain_losses(
        self,
        cka_layer_specs,
        captured_layer_hiddens,
        final_hidden,
        output_hidden_states,
        pre_projector_features,
        vision_feature_mask,
        output_device,
    ):
        ordered_hiddens = list(self._iter_cka_layer_hiddens(
            cka_layer_specs,
            captured_layer_hiddens,
            final_hidden,
            output_hidden_states,
        ))
        layer_losses = []
        per_layer_losses = {}
        if not ordered_hiddens:
            return layer_losses, per_layer_losses

        previous_name = "pre_projector"
        previous_hidden = pre_projector_features.detach()
        for layer_name, layer_hidden in ordered_hiddens:
            layer_loss = self._compute_masked_linear_cka_loss(
                projected_features=layer_hidden,
                layer_hidden_states=previous_hidden.detach(),
                vision_feature_mask=vision_feature_mask,
            ).to(output_device)
            layer_losses.append(layer_loss)
            per_layer_losses[f"{previous_name}_to_{layer_name}"] = layer_loss.detach()
            previous_name = layer_name
            previous_hidden = layer_hidden

        return layer_losses, per_layer_losses

    def _get_cka_attention_subset_kwargs(self):
        config = self.get_model().config
        legacy_max_ratio = getattr(config, 'cka_loss_subset_ratio', 1.0)
        max_keep_ratio = getattr(config, 'cka_loss_subset_max_ratio', None)
        if max_keep_ratio is None:
            max_keep_ratio = legacy_max_ratio

        min_keep_ratio = float(getattr(config, 'cka_loss_subset_min_ratio', 0.1) or 0.0)
        max_keep_ratio = float(max_keep_ratio or 0.0)
        fallback_mass = float(getattr(config, 'cka_loss_subset_fallback_mass', 0.75) or 0.0)
        otsu_min_separability = float(getattr(config, 'cka_loss_subset_otsu_min_separability', 0.05) or 0.0)

        min_keep_ratio = max(0.0, min(1.0, min_keep_ratio))
        max_keep_ratio = max(0.0, min(1.0, max_keep_ratio))
        if max_keep_ratio > 0.0:
            max_keep_ratio = max(min_keep_ratio, max_keep_ratio)

        return {
            "min_keep_ratio": min_keep_ratio,
            "max_keep_ratio": max_keep_ratio,
            "fallback_mass": max(0.0, min(1.0, fallback_mass)),
            "otsu_min_separability": max(0.0, min(1.0, otsu_min_separability)),
        }

    def _fallback_keep_count_from_attention_mass(
        self,
        probabilities: torch.Tensor,
        fallback_mass: float,
    ) -> int:
        token_count = int(probabilities.numel())
        if token_count <= 1:
            return token_count

        fallback_mass = float(max(0.0, min(1.0, fallback_mass)))
        if fallback_mass <= 0.0:
            return 1
        if fallback_mass >= 1.0:
            return token_count

        sorted_probabilities = torch.sort(probabilities, descending=True).values
        cumulative_mass = torch.cumsum(sorted_probabilities, dim=0)
        threshold = cumulative_mass.new_tensor(fallback_mass)
        keep_count = int(torch.searchsorted(cumulative_mass, threshold, right=False).item()) + 1
        return max(1, min(token_count, keep_count))

    def _otsu_keep_count_from_log_probs(
        self,
        log_probabilities: torch.Tensor,
        eps: float = 1e-8,
    ) -> Tuple[int, float]:
        token_count = int(log_probabilities.numel())
        if token_count <= 1:
            return token_count, 0.0

        sorted_values = torch.sort(log_probabilities).values
        total_variance = torch.mean((sorted_values - sorted_values.mean()).square())
        if float(total_variance.item()) <= eps:
            return token_count, 0.0

        prefix_sum = torch.cumsum(sorted_values, dim=0)
        split_counts = torch.arange(
            1,
            token_count,
            device=sorted_values.device,
            dtype=sorted_values.dtype,
        )
        left_weight = split_counts / token_count
        right_weight = 1.0 - left_weight
        left_mean = prefix_sum[:-1] / split_counts
        right_mean = (prefix_sum[-1] - prefix_sum[:-1]) / (token_count - split_counts)
        between_class_variance = left_weight * right_weight * (left_mean - right_mean).square()

        best_split = int(torch.argmax(between_class_variance).item())
        separability = float((between_class_variance[best_split] / total_variance.clamp_min(eps)).item())
        # Sorted values are ascending log-probs. Keep the high-prob side as a top-k count.
        keep_count = token_count - (best_split + 1)
        return max(1, min(token_count, keep_count)), separability

    def _select_topk_indices_from_attention_scores(
        self,
        image_scores: torch.Tensor,
        min_keep_tokens: int,
        min_keep_ratio: float,
        max_keep_ratio: float,
        fallback_mass: float,
        otsu_min_separability: float,
        eps: float = 1e-8,
    ) -> torch.LongTensor:
        """
        Convert per-image-token attention scores into a top-k subset.

        The selection is per sample: detach scores, normalize over image tokens,
        run Otsu on log-probabilities, clamp by min/max ratio, and use cumulative
        attention mass when the Otsu split is not separable enough.
        """
        token_count = int(image_scores.numel())
        if token_count == 0:
            return torch.empty(0, dtype=torch.long, device=image_scores.device)
        if token_count <= min_keep_tokens or max_keep_ratio <= 0.0:
            return torch.arange(token_count, dtype=torch.long, device=image_scores.device)

        min_keep_ratio = max(0.0, min(1.0, float(min_keep_ratio)))
        max_keep_ratio = max(0.0, min(1.0, float(max_keep_ratio)))
        min_keep_count = max(min_keep_tokens, int(math.ceil(token_count * min_keep_ratio)))
        min_keep_count = min(token_count, min_keep_count)
        max_keep_count = max(min_keep_count, int(math.floor(token_count * max_keep_ratio)))
        max_keep_count = min(token_count, max_keep_count)

        scores = image_scores.detach().float()
        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        score_sum = scores.sum()
        if float(score_sum.item()) <= eps:
            probabilities = torch.full_like(scores, 1.0 / token_count)
            keep_count = self._fallback_keep_count_from_attention_mass(probabilities, fallback_mass)
        else:
            probabilities = scores / score_sum.clamp_min(eps)
            log_probabilities = torch.log(probabilities.clamp_min(eps))
            keep_count, separability = self._otsu_keep_count_from_log_probs(log_probabilities, eps=eps)
            if separability < otsu_min_separability:
                keep_count = self._fallback_keep_count_from_attention_mass(probabilities, fallback_mass)

        keep_count = max(min_keep_count, min(max_keep_count, keep_count))
        # Stable sorting keeps tie cases deterministic instead of letting topk pick
        # arbitrary equal-score image tokens.
        return torch.argsort(scores, descending=True, stable=True)[:keep_count]

    def _get_cka_attention_query_mask(
        self,
        vision_feature_mask: torch.BoolTensor,
        valid_mask: torch.BoolTensor,
        labels: Optional[torch.LongTensor] = None,
    ) -> torch.BoolTensor:
        text_mask = (~vision_feature_mask) & valid_mask
        query_tokens = str(
            getattr(self.get_model().config, 'cka_loss_subset_query_tokens', 'text') or 'text'
        ).lower().replace('_', '-')

        if query_tokens in ('instruction', 'instructions', 'prompt', 'non-answer', 'nonanswer'):
            if labels is None or labels.shape != vision_feature_mask.shape:
                return text_mask
            labels = labels.to(device=vision_feature_mask.device)
            return text_mask & labels.eq(IGNORE_INDEX)

        return text_mask

    def _select_vision_feature_subset_from_attention(
        self,
        attentions,
        vision_feature_mask: torch.BoolTensor,
        attention_mask: Optional[torch.Tensor],
        select_layer: Optional[int],
        min_keep_ratio: float,
        max_keep_ratio: float,
        fallback_mass: float,
        otsu_min_separability: float,
        labels: Optional[torch.LongTensor] = None,
        min_keep_tokens: int = 16,
    ) -> Optional[torch.BoolTensor]:
        """
        Select a subset of image tokens using text-to-image attention at one LLM layer.

        Per sample, attention mass over image tokens is detached, normalized into
        probabilities, split with Otsu on log-probabilities, then converted into a
        top-k image-token mask. Min/max ratio caps bound the selected count; low
        separability falls back to cumulative attention mass.

        Note: `select_layer` only chooses the attention layer used for subset selection.
        The LLM-hidden CKA term below still compares final hidden states against
        pre-projector vision features.
        """
        if attentions is None or vision_feature_mask is None or select_layer is None:
            return None

        if not isinstance(attentions, (list, tuple)) or len(attentions) == 0:
            return None

        if select_layer < 1:
            return None

        attn_index = select_layer - 1
        if attn_index < 0 or attn_index >= len(attentions):
            return None

        layer_attn = attentions[attn_index]
        if layer_attn is None:
            return None

        if attention_mask is None:
            attention_mask = torch.ones_like(vision_feature_mask, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()

        if max_keep_ratio <= 0.0:
            return vision_feature_mask.clone()

        layer_attn = layer_attn.float()
        selected_masks = []

        for batch_idx in range(layer_attn.shape[0]):
            valid_mask = attention_mask[batch_idx]
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

            # attn_i: [heads, seq, seq]
            attn_i = layer_attn[batch_idx]
            text_to_image = attn_i[:, text_mask][:, :, image_mask]
            if text_to_image.numel() == 0:
                selected_masks.append(image_mask.clone())
                continue

            # Score each image token by average attention received from text queries.
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
            image_positions = torch.where(image_mask)[0]
            selected_mask[image_positions[topk_indices]] = True
            selected_masks.append(selected_mask)

        return torch.stack(selected_masks, dim=0)

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
        """
        Select image tokens from one layer's QK attention without asking the whole model
        to materialize attentions. This keeps flash-attn for the real forward pass.

        This is the fast path for `cka_loss_subset_select_layer`; the fallback path uses
        `output.attentions` when attentions were already requested.
        """
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
                # Training should not use KV cache here. Avoid silently mishandling cached positions.
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

        cos, sin = attention_module.rotary_emb(key_states, seq_len=q_len)
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
                # CKA-enabled multimodal prep also returns the image-token mask and
                # pre-projector vision features aligned to the final input sequence.
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
            # HF LlamaModel stores hidden_states[-1] after final RMSNorm; CKA needs the pre-norm value.
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
            and pre_projector_features is not None
        ):
            layers = getattr(self.get_model(), "layers", None)
            # Config is 1-based for readability; HF module lists are 0-based.
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

        try:
            output = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=should_output_attentions,
                output_hidden_states=should_output_hidden_states,
                return_dict=return_dict
            )
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
            and pre_projector_features is not None
        ):
            subset_vision_feature_mask = captured_subset_vision_feature_mask
            if subset_vision_feature_mask is None and output.attentions is not None:
                # Fallback for runs that materialize attentions instead of using the hook.
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
                # LLM CKA terms form a chain over image tokens:
                # pre_projector -> first requested layer -> ... .
                # The previous endpoint is detached for each edge, so each term
                # updates only the later hidden state in that pair.
                layer_mask = subset_vision_feature_mask if subset_vision_feature_mask is not None else vision_feature_mask
                layer_losses, per_layer_losses = self._compute_cka_chain_losses(
                    cka_layer_specs=cka_layer_specs,
                    captured_layer_hiddens=captured_cka_layer_hiddens,
                    final_hidden=final_hidden,
                    output_hidden_states=output.hidden_states,
                    pre_projector_features=pre_projector_features,
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

            # Keep projector CKA and LLM CKA as separate terms.
            cka_loss = projector_cka_loss + cka_layers_loss

            # Store losses for logging
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

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
