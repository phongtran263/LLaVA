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


from abc import ABC, abstractmethod
import math
import operator
import warnings

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape


def validate_cka_eps(eps):
    """Return a finite, strictly positive numerical-stability constant."""
    try:
        eps = float(eps)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"CKA eps must be a positive finite number, got {eps!r}") from exc

    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"CKA eps must be a positive finite number, got {eps!r}")
    return eps


def validate_cka_channel_keep_ratio(keep_ratio):
    """Return a valid fraction of feature channels retained for CKA."""
    try:
        keep_ratio = float(keep_ratio)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"CKA channel keep ratio must be in (0, 1], got {keep_ratio!r}"
        ) from exc

    if not math.isfinite(keep_ratio) or not 0.0 < keep_ratio <= 1.0:
        raise ValueError(
            f"CKA channel keep ratio must be in (0, 1], got {keep_ratio!r}"
        )
    return keep_ratio


def parse_cka_channel_drop_indices(drop_indices):
    """Parse zero-based channel indices excluded from CKA."""
    if drop_indices is None:
        return ()

    if isinstance(drop_indices, str):
        if not drop_indices.strip():
            return ()
        values = drop_indices.split(",")
        if any(not value.strip() for value in values):
            raise ValueError(
                "CKA channel drop indices must be comma-separated non-negative integers, "
                f"got {drop_indices!r}"
            )
    else:
        try:
            values = list(drop_indices)
        except TypeError:
            values = [drop_indices]

    parsed = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError(
                "CKA channel drop indices must be comma-separated non-negative integers, "
                f"got {drop_indices!r}"
            )
        if isinstance(value, str):
            try:
                index = int(value.strip())
            except ValueError as exc:
                raise ValueError(
                    "CKA channel drop indices must be comma-separated non-negative integers, "
                    f"got {drop_indices!r}"
                ) from exc
        else:
            try:
                index = operator.index(value)
            except TypeError as exc:
                raise ValueError(
                    "CKA channel drop indices must be comma-separated non-negative integers, "
                    f"got {drop_indices!r}"
                ) from exc
        if index < 0:
            raise ValueError(
                "CKA channel drop indices must be comma-separated non-negative integers, "
                f"got {drop_indices!r}"
            )
        parsed.append(index)

    return tuple(sorted(set(parsed)))


def select_cka_feature_channels(
    features,
    keep_ratio=1.0,
    seed=42,
    salt=0,
    drop_indices=None,
):
    """Select the feature channels used by CKA."""
    keep_ratio = validate_cka_channel_keep_ratio(keep_ratio)
    drop_indices = parse_cka_channel_drop_indices(drop_indices)
    if features.ndim == 0:
        raise ValueError("CKA channel selection requires at least one dimension")

    feature_dim = int(features.shape[-1])
    out_of_range_indices = tuple(
        index for index in drop_indices if index >= feature_dim
    )
    if out_of_range_indices:
        raise ValueError(
            f"CKA channel drop indices {out_of_range_indices} are outside "
            f"the feature width {feature_dim}"
        )
    active_drop_indices = drop_indices
    if keep_ratio == 1.0 and not active_drop_indices:
        return features
    if active_drop_indices and len(active_drop_indices) == feature_dim:
        raise ValueError(
            f"CKA channel drop indices remove all {feature_dim} feature channels"
        )

    available_indices = torch.arange(feature_dim, device="cpu")
    if active_drop_indices:
        available_mask = torch.ones(feature_dim, dtype=torch.bool, device="cpu")
        available_mask[list(active_drop_indices)] = False
        available_indices = available_indices[available_mask]

    available_count = int(available_indices.numel())
    keep_count = min(
        available_count,
        max(1, math.ceil(available_count * keep_ratio)),
    )
    if keep_count < available_count:
        try:
            seed = int(seed)
            salt = int(salt)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"CKA channel seed and salt must be integers, got {seed!r}, {salt!r}"
            ) from exc

        modulus = (1 << 63) - 1
        mixed_seed = (
            seed
            + 1_000_003 * (feature_dim + 1)
            + 9_176 * (salt + 1)
        ) % modulus
        generator = torch.Generator(device="cpu")
        generator.manual_seed(mixed_seed)
        selected_positions = torch.randperm(
            available_count,
            generator=generator,
        )[:keep_count]
        available_indices = available_indices[selected_positions]

    if not active_drop_indices and keep_count == feature_dim:
        return features
    indices = available_indices.sort().values.to(device=features.device)
    return features.index_select(-1, indices)


def normalize_centered_cka_features(features):
    """Normalize each centered sample without introducing an absolute scale floor."""
    reduce_dims = tuple(range(1, features.ndim))
    norms = torch.linalg.vector_norm(features, dim=reduce_dims, keepdim=True)
    return features / norms.clamp_min(torch.finfo(features.dtype).tiny)


def cka_similarity_to_loss(cka):
    """Convert CKA similarity to the legacy dissimilarity loss."""
    return 1.0 - cka


class _ProjectorPCGrad(torch.autograd.Function):
    """Run exact one-sided PCGrad over the trainable projector parameters."""

    @staticmethod
    def forward(ctx, projector, parameter_names, image_features, *parameters):
        ctx.set_materialize_grads(False)
        ctx.input_requires_grad = image_features.requires_grad

        # Build a private projector graph on detached aliases. The outer graph
        # sees only this Function, so DDP/ZeRO receives the merged parameter
        # gradient exactly once. Detach shares storage and does not clone the
        # projector weights.
        functional_input = image_features.detach().requires_grad_(
            ctx.input_requires_grad
        )
        functional_parameters = tuple(
            parameter.detach().requires_grad_(True)
            for parameter in parameters
        )
        with torch.enable_grad():
            projected_features = torch.func.functional_call(
                projector,
                dict(zip(parameter_names, functional_parameters)),
                (functional_input,),
                tie_weights=True,
                strict=False,
            )

        ctx.save_for_backward(
            projected_features,
            functional_input,
            *functional_parameters,
        )
        output = projected_features.detach()
        return output.view_as(output), output.view_as(output)

    @staticmethod
    def backward(ctx, task_gradient, cka_gradient):
        projected_features, functional_input, *functional_parameters = (
            ctx.saved_tensors
        )
        targets = (
            ((functional_input,) if ctx.input_requires_grad else ())
            + tuple(functional_parameters)
        )

        def branch_vjp(branch_gradient):
            if branch_gradient is None:
                return (None,) * len(targets)
            with torch.enable_grad():
                return torch.autograd.grad(
                    projected_features,
                    targets,
                    grad_outputs=branch_gradient,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True,
                )

        task_vjp = branch_vjp(task_gradient)
        cka_vjp = branch_vjp(cka_gradient)
        parameter_offset = int(ctx.input_requires_grad)
        task_parameter_grads = task_vjp[parameter_offset:]
        cka_parameter_grads = cka_vjp[parameter_offset:]

        reduction_dtype = (
            torch.float64
            if projected_features.dtype == torch.float64
            else torch.float32
        )
        dot_product = projected_features.new_zeros((), dtype=reduction_dtype)
        task_norm_sq = projected_features.new_zeros((), dtype=reduction_dtype)
        task_for_reduction = None
        cka_for_reduction = None
        for task_grad, cka_grad in zip(
            task_parameter_grads,
            cka_parameter_grads,
        ):
            if task_grad is not None:
                task_for_reduction = task_grad.to(dtype=reduction_dtype).reshape(-1)
                task_norm_sq = task_norm_sq + torch.dot(
                    task_for_reduction,
                    task_for_reduction,
                )
                if cka_grad is not None:
                    cka_for_reduction = cka_grad.to(
                        dtype=reduction_dtype
                    ).reshape(-1)
                    dot_product = dot_product + torch.dot(
                        task_for_reduction,
                        cka_for_reduction,
                    )
        # Do not keep the final parameter-sized FP32 reduction buffers alive
        # while materializing the merged parameter gradients below.
        task_for_reduction = None
        cka_for_reduction = None

        projection_coefficient = torch.where(
            torch.isfinite(dot_product)
            & torch.isfinite(task_norm_sq)
            & (dot_product < 0)
            & (task_norm_sq > 0),
            dot_product
            / task_norm_sq.clamp_min(torch.finfo(reduction_dtype).tiny),
            dot_product.new_zeros(()),
        )

        parameter_grads = []
        for parameter, task_grad, cka_grad in zip(
            functional_parameters,
            task_parameter_grads,
            cka_parameter_grads,
        ):
            if task_grad is None and cka_grad is None:
                merged_grad = torch.zeros_like(parameter)
            elif task_grad is None:
                merged_grad = cka_grad
            elif cka_grad is None:
                merged_grad = task_grad
            else:
                # Merge in one FP32/FP64 buffer. Casting a large coefficient to
                # FP16 first can overflow even when the final gradient is finite.
                task_for_merge = task_grad.to(dtype=reduction_dtype)
                task_for_merge.mul_(1.0 - projection_coefficient)
                task_for_merge.add_(cka_grad)
                merged_grad = task_for_merge.to(dtype=parameter.dtype)
            parameter_grads.append(merged_grad)

        input_grad = None
        if ctx.input_requires_grad:
            task_input_grad = task_vjp[0]
            cka_input_grad = cka_vjp[0]
            if task_input_grad is None and cka_input_grad is None:
                input_grad = torch.zeros_like(functional_input)
            elif task_input_grad is None:
                input_grad = cka_input_grad
            elif cka_input_grad is None:
                input_grad = task_input_grad
            else:
                # PCGrad is scoped to mm_projector parameters. Preserve the
                # ordinary summed gradient if a vision encoder is trainable.
                input_grad = task_input_grad + cka_input_grad

        return None, None, input_grad, *parameter_grads


def project_features_with_pcgrad(projector, image_features):
    """Project once and return task/CKA branches with exact parameter PCGrad.

    The full language model still has one backward pass. Only the small
    projector graph receives two local VJPs, while AMP/loss scaling, gradient
    accumulation, DDP, and ZeRO-2 operate on the final merged gradient normally.
    Projection is local to each projector invocation and training microbatch.
    """
    named_parameters = tuple(
        (name, parameter)
        for name, parameter in projector.named_parameters()
        if parameter.requires_grad
    )
    if not named_parameters:
        projected_features = projector(image_features)
        return projected_features, projected_features

    if any(
        hasattr(parameter, 'ds_id')
        for _, parameter in named_parameters
    ):
        raise RuntimeError(
            "Exact projector PCGrad does not support DeepSpeed ZeRO-3; "
            "use ZeRO-2 or disable --use_pcgrad."
        )

    parameter_names, parameters = zip(*named_parameters)
    return _ProjectorPCGrad.apply(
        projector,
        parameter_names,
        image_features,
        *parameters,
    )


def compute_linear_cka_loss(
    x,
    y,
    eps=1e-8,
    *,
    channel_keep_ratio=1.0,
    channel_seed=42,
    x_channel_drop_indices=None,
    y_channel_drop_indices=None,
    share_channel_indices=True,
):
    """
    Compute per-sample Linear CKA loss.
    For each batch element, treats tokens/patches as samples and features as dimensions.

    Args:
        x, y: Shape (B, L, D) where B=batch, L=sequence length (tokens/patches), D=features
              If 2D, treats as (B, L*D)
        eps: numerical stability constant
        channel_keep_ratio: fixed fraction of rank-3 feature channels retained
        channel_seed: seed used without consuming the global RNG
        x_channel_drop_indices: zero-based x channels excluded before optional sampling
        y_channel_drop_indices: zero-based y channels excluded before optional sampling
        share_channel_indices: reuse channel indices when feature widths match

    Returns:
        Scalar loss (mean of per-sample CKA losses)
    """
    eps = validate_cka_eps(eps)
    channel_keep_ratio = validate_cka_channel_keep_ratio(channel_keep_ratio)
    x_channel_drop_indices = parse_cka_channel_drop_indices(x_channel_drop_indices)
    y_channel_drop_indices = parse_cka_channel_drop_indices(y_channel_drop_indices)

    if x.ndim != y.ndim or x.ndim not in (2, 3):
        raise ValueError(
            f"CKA inputs must both be rank 2 or both be rank 3, got {x.ndim} and {y.ndim}"
        )
    if x.device != y.device:
        raise ValueError(f"CKA inputs must be on the same device, got {x.device} and {y.device}")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Batch size mismatch: {x.shape[0]} vs {y.shape[0]}")
    if x.shape[0] == 0:
        return x.float().sum() * 0.0 + y.float().sum() * 0.0
    if x.ndim == 3 and x.shape[1] != y.shape[1]:
        raise ValueError(f"Token length mismatch: {x.shape[1]} vs {y.shape[1]}")
    if x.ndim == 2 and x.shape[1] != y.shape[1]:
        raise ValueError(f"Feature length mismatch: {x.shape[1]} vs {y.shape[1]}")

    if x.ndim == 3 and (
        channel_keep_ratio < 1.0
        or x_channel_drop_indices
        or y_channel_drop_indices
    ):
        share_indices = (
            share_channel_indices
            and x.shape[-1] == y.shape[-1]
            and x_channel_drop_indices == y_channel_drop_indices
        )
        x = select_cka_feature_channels(
            x,
            keep_ratio=channel_keep_ratio,
            seed=channel_seed,
            salt=0,
            drop_indices=x_channel_drop_indices,
        )
        y = select_cka_feature_channels(
            y,
            keep_ratio=channel_keep_ratio,
            seed=channel_seed,
            salt=0 if share_indices else 1,
            drop_indices=y_channel_drop_indices,
        )

    # Merely casting the inputs to float32 is insufficient under CUDA autocast:
    # bmm/matmul can still be downcast to fp16. CKA's Gram/HSIC reductions are
    # especially sensitive to that, so keep the entire calculation in FP32.
    with torch.autocast(device_type=x.device.type, enabled=False):
        x = x.float()
        y = y.float()

        # Compute per-sample CKA.
        if x.ndim == 3:
            # Already (B, L, D) - tokens as samples, features as dimensions.
            x = x - x.mean(dim=1, keepdim=True)
            y = y - y.mean(dim=1, keepdim=True)
            x = normalize_centered_cka_features(x)
            y = normalize_centered_cka_features(y)

            # Batched Gram matrices: (B, L, D) @ (B, D, L) = (B, L, L).
            xx = torch.bmm(x, x.transpose(1, 2))
            yy = torch.bmm(y, y.transpose(1, 2))

            hsic_xy = (xx * yy).sum(dim=(1, 2))
            hsic_xx = xx.square().sum(dim=(1, 2))
            hsic_yy = yy.square().sum(dim=(1, 2))
        else:
            # 2D input (B, L*D) - flatten approach. This is algebraically
            # equivalent to the old per-sample outer-product Gram computation.
            x = x.flatten(1)
            y = y.flatten(1)
            x = x - x.mean(dim=1, keepdim=True)
            y = y - y.mean(dim=1, keepdim=True)
            x = normalize_centered_cka_features(x)
            y = normalize_centered_cka_features(y)

            xy = (x * y).sum(dim=1)
            xx = x.square().sum(dim=1)
            yy = y.square().sum(dim=1)

            hsic_xy = xy.square()
            hsic_xx = xx.square()
            hsic_yy = yy.square()

        denom = torch.sqrt(torch.clamp(hsic_xx * hsic_yy, min=eps))
        cka = (hsic_xy / denom).clamp(0.0, 1.0)
        return cka_similarity_to_loss(cka).mean()

class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower
        if self.get_vision_tower() is None:
            model_args.hidden_size = self.config.hidden_size
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None and ',' not in pretrain_mm_mlp_adapter:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def warn_if_projector_only_cka(self):
        """Warn once when a backbone cannot honor hidden-layer CKA settings."""
        config = self.get_model().config
        raw_layers = getattr(config, 'cka_loss_layers', 'final')
        disabled_tokens = {"-1", "none", "off", "false"}
        hidden_layers_disabled = (
            raw_layers in (None, "", False)
            or (
                isinstance(raw_layers, str)
                and raw_layers.strip().lower() in disabled_tokens
            )
            or (
                isinstance(raw_layers, (list, tuple))
                and len(raw_layers) == 1
                and str(raw_layers[0]).strip().lower() in disabled_tokens
            )
        )
        if hidden_layers_disabled or getattr(self, '_projector_only_cka_warned', False):
            return

        warnings.warn(
            f"{type(self).__name__} supports projector CKA only; "
            "cka_loss_layers is ignored. Use -1 to silence this warning.",
            UserWarning,
            stacklevel=2,
        )
        self._projector_only_cka_warned = True

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        model = self.get_model()
        config = model.config
        cka_enabled = model.training and getattr(config, 'cka_loss', False)
        projector_weight = getattr(config, 'cka_loss_projector_weight', None)
        if projector_weight is None:
            projector_weight = getattr(config, 'cka_loss_weight', 1.0)

        use_projector_pcgrad = (
            cka_enabled
            and float(projector_weight) != 0.0
            and getattr(config, 'use_pcgrad', False)
        )
        if use_projector_pcgrad:
            (
                projected_image_features,
                projected_features_for_cka,
            ) = project_features_with_pcgrad(
                model.mm_projector,
                image_features,
            )
        else:
            projected_image_features = model.mm_projector(image_features)
            projected_features_for_cka = projected_image_features

        if model.training and getattr(config, 'log_gradient_norms', False):
            self.last_cka_projector_output = projected_image_features
            self.last_cka_projector_cka_output = projected_features_for_cka

        if cka_enabled:

            cka_loss = None
            if float(projector_weight) != 0.0:
                # Projector CKA term: keep the projected image embeddings
                # structurally close to the raw vision-tower patch features.
                cka_loss = compute_linear_cka_loss(
                    image_features,
                    projected_features_for_cka,
                    channel_keep_ratio=getattr(config, 'cka_loss_channel_keep_ratio', 1.0),
                    channel_seed=getattr(config, 'cka_loss_channel_seed', 42),
                    y_channel_drop_indices=getattr(config, 'cka_loss_hidden_channel_drop_indices', None),
                    share_channel_indices=False,
                )
            final_hidden_weight = getattr(config, "cka_loss_final_hidden_weight", None)
            if final_hidden_weight is None:
                final_hidden_weight = getattr(config, "cka_loss_weight", 1.0)
            selected_hidden_specs = (
                self._get_cka_layer_specs()
                if hasattr(self, "_get_cka_layer_specs")
                else []
            )
            needs_vision_reference = (
                float(final_hidden_weight) != 0.0 and bool(selected_hidden_specs)
            )
            # Carry raw vision features only when a selected-hidden CKA term uses
            # them. This avoids extra sequence-sized buffers for projector-only CKA.
            vision_reference = image_features.detach() if needs_vision_reference else None
            return projected_image_features, cka_loss, vision_reference

        return projected_image_features

    def extract_text_features(self, input_ids, attention_mask=None, exit_layer=6):
        with torch.no_grad():
            text_features = []

            for batch_idx, cur_input_ids in enumerate(input_ids):
                cur = cur_input_ids.clone()
                keep = cur != IMAGE_TOKEN_INDEX
                if attention_mask is not None:
                    keep = keep & attention_mask[batch_idx].bool()
                cur = cur[keep].unsqueeze(0)

                if cur.shape[1] == 0:
                    text_features.append(torch.zeros(self.get_model().config.hidden_size, device=cur_input_ids.device, dtype=self.get_model().layers[0].input_layernorm.weight.dtype))
                    continue

                h = self.get_model().embed_tokens(cur)
                h = h.to(self.get_model().layers[0].input_layernorm.weight.dtype)

                for layer in self.get_model().layers[:exit_layer]:
                    h = layer(
                        h,
                        attention_mask=None,
                        position_ids=None,
                        past_key_value=None,
                        output_attentions=False,
                        use_cache=False,
                    )[0]

                text_features.append(h[0, -1].detach())

        return torch.stack(text_features, dim=0)

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if self.get_model().training and getattr(self.get_model().config, 'cka_loss', False):
                return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None, None
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        pre_post_cka_loss = None
        pre_projector_image_features = None

        if type(images) is list or (not isinstance(images, dict) and images.ndim == 5):
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            if self.get_model().training and getattr(self.get_model().config, 'cka_loss', False):
                image_features, pre_post_cka_loss, pre_projector_image_features = self.encode_images(concat_images)
            else:
                image_features = self.encode_images(concat_images)
                if isinstance(image_features, tuple):
                    image_features = image_features[0]
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            if pre_projector_image_features is not None:
                pre_projector_image_features = torch.split(pre_projector_image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
                if pre_projector_image_features is not None:
                    pre_projector_image_features = [x.flatten(0, 1) for x in pre_projector_image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                if pre_projector_image_features is not None:
                    raise ValueError(
                        "Vision-referenced LLM CKA currently requires "
                        "mm_patch_merge_type='flat' because spatial/unpad adds "
                        "projected tokens without aligned raw vision features."
                    )
                # Spatial/unpad can add learned newline tokens after the projector; skip
                # pre-projector-vs-LLM CKA there unless an aligned raw newline exists.
                pre_projector_image_features = None
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            if self.get_model().training and getattr(self.get_model().config, 'cka_loss', False):
                image_features, pre_post_cka_loss, pre_projector_image_features = self.encode_images(images)
            else:
                image_features = self.encode_images(images)
                if isinstance(image_features, tuple):
                    image_features = image_features[0]

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        new_vision_feature_masks = [] if (self.get_model().training and getattr(self.get_model().config, 'cka_loss', False)) else None
        # These raw vision features follow the same insertion, truncation, and padding
        # path as projected image embeddings so later CKA can align token positions.
        new_pre_projector_features = [] if pre_projector_image_features is not None else None
        if pre_projector_image_features is not None:
            sample_pre_projector_feature = pre_projector_image_features[0]
            pre_projector_feature_size = sample_pre_projector_feature.shape[-1]
            pre_projector_feature_dtype = sample_pre_projector_feature.dtype
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_pre_projector_image_features = pre_projector_image_features[cur_image_idx] if new_pre_projector_features is not None else None
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                if new_vision_feature_masks is not None:
                    new_vision_feature_masks.append(
                        torch.zeros(cur_input_embeds.shape[0], dtype=torch.bool, device=cur_input_embeds.device)
                    )
                if new_pre_projector_features is not None:
                    new_pre_projector_features.append(
                        torch.zeros(
                            (cur_input_embeds.shape[0], cur_pre_projector_image_features.shape[-1]),
                            dtype=cur_pre_projector_image_features.dtype,
                            device=cur_input_embeds.device,
                        )
                    )
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_new_vision_feature_mask = [] if new_vision_feature_masks is not None else None
            cur_new_pre_projector_features = [] if new_pre_projector_features is not None else None

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if cur_new_vision_feature_mask is not None:
                    cur_new_vision_feature_mask.append(
                        torch.zeros(cur_input_embeds_no_im[i].shape[0], dtype=torch.bool, device=cur_labels.device)
                    )
                if cur_new_pre_projector_features is not None:
                    cur_new_pre_projector_features.append(
                        torch.zeros(
                            (cur_input_embeds_no_im[i].shape[0], pre_projector_feature_size),
                            dtype=pre_projector_feature_dtype,
                            device=cur_labels.device,
                        )
                    )
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_pre_projector_image_features = pre_projector_image_features[cur_image_idx] if cur_new_pre_projector_features is not None else None
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    if cur_new_vision_feature_mask is not None:
                        cur_new_vision_feature_mask.append(
                            torch.ones(cur_image_features.shape[0], dtype=torch.bool, device=cur_labels.device)
                        )
                    if cur_new_pre_projector_features is not None:
                        # Image-token CKA needs one raw vision feature per projected image token.
                        if cur_pre_projector_image_features.shape[0] != cur_image_features.shape[0]:
                            raise ValueError(
                                "Pre-projector image features must align with projected image features for CKA: "
                                f"{cur_pre_projector_image_features.shape[0]} vs {cur_image_features.shape[0]}"
                            )
                        cur_new_pre_projector_features.append(cur_pre_projector_image_features)

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            if cur_new_pre_projector_features is not None:
                cur_new_pre_projector_features = [x.to(self.device) for x in cur_new_pre_projector_features]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            if new_vision_feature_masks is not None:
                new_vision_feature_masks.append(torch.cat(cur_new_vision_feature_mask))
            if new_pre_projector_features is not None:
                new_pre_projector_features.append(torch.cat(cur_new_pre_projector_features))

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
            if new_vision_feature_masks is not None:
                new_vision_feature_masks = [x[:tokenizer_model_max_length] for x in new_vision_feature_masks]
            if new_pre_projector_features is not None:
                new_pre_projector_features = [x[:tokenizer_model_max_length] for x in new_pre_projector_features]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        vision_feature_mask_padded = None
        if new_vision_feature_masks is not None:
            vision_feature_mask_padded = torch.zeros((batch_size, max_len), dtype=torch.bool, device=attention_mask.device)
        pre_projector_features_padded = None
        if new_pre_projector_features is not None:
            pre_projector_features_padded = torch.zeros(
                (batch_size, max_len, pre_projector_feature_size),
                dtype=new_pre_projector_features[0].dtype,
                device=new_pre_projector_features[0].device,
            )

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                    if vision_feature_mask_padded is not None:
                        vision_feature_mask_padded[i, -cur_len:] = new_vision_feature_masks[i]
                    if pre_projector_features_padded is not None:
                        pre_projector_features_padded[i, -cur_len:] = new_pre_projector_features[i]
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                    if vision_feature_mask_padded is not None:
                        vision_feature_mask_padded[i, :cur_len] = new_vision_feature_masks[i]
                    if pre_projector_features_padded is not None:
                        pre_projector_features_padded[i, :cur_len] = new_pre_projector_features[i]

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        # The attention-based CKA subset selector needs this padded mask even when
        # the original caller did not pass one.
        keep_attention_mask_for_cka = self.get_model().training and getattr(self.get_model().config, 'cka_loss', False) and getattr(self.get_model().config, 'cka_loss_subset_select_layer', None) is not None
        if _attention_mask is None:
            attention_mask = attention_mask if keep_attention_mask_for_cka else None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        if self.get_model().training and getattr(self.get_model().config, 'cka_loss', False):
            return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, vision_feature_mask_padded, pre_post_cka_loss, pre_projector_features_padded
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
