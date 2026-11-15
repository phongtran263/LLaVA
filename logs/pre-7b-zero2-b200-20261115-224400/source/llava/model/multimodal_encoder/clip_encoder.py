import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def validate_cka_anchor_layer(self, layer):
        count = self.config.num_hidden_layers + 1
        if isinstance(layer, bool) or not isinstance(layer, int) or not -count <= layer < count:
            raise ValueError(
                f"cka_loss_vision_anchor_layer must be an integer in [-{count}, {count - 1}], "
                f"got {layer!r}. 0 is the embedding state, 1..L are vision block outputs."
            )

    def feature_select(self, image_forward_outs, layer=None):
        image_features = image_forward_outs.hidden_states[self.select_layer if layer is None else layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images, cka_anchor_layer=None, cka_projector_anchor_layer=None,
                cka_final_anchor_layer=None):
        # CLIP already returns all hidden states. Gather requested features from the
        # same image/crop forward, without changing the normal projector input.
        for layer in (cka_anchor_layer, cka_projector_anchor_layer, cka_final_anchor_layer):
            if layer is not None:
                self.validate_cka_anchor_layer(layer)
        needs_anchor = any(layer is not None for layer in (
            cka_anchor_layer, cka_projector_anchor_layer, cka_final_anchor_layer,
        ))

        def select_features(outputs, dtype):
            # Reuse slices/casts when projector and final share an anchor.
            cache = {}

            def select(layer):
                index = self.select_layer if layer is None else layer
                key = index if index >= 0 else index + self.config.num_hidden_layers + 1
                if key not in cache:
                    cache[key] = self.feature_select(outputs, index).to(dtype)
                return cache[key]

            features = [select(None)]
            if needs_anchor:
                features.append(select(cka_anchor_layer))
            if cka_projector_anchor_layer is not None or cka_final_anchor_layer is not None:
                projector_layer = cka_projector_anchor_layer
                if projector_layer is None:
                    projector_layer = cka_anchor_layer
                features.append(select(projector_layer))
            if cka_final_anchor_layer is not None:
                features.append(select(cka_final_anchor_layer))
            return features

        if type(images) is list:
            count = (1 + int(needs_anchor)
                     + int(cka_projector_anchor_layer is not None or cka_final_anchor_layer is not None)
                     + int(cka_final_anchor_layer is not None))
            features = [[] for _ in range(count)]
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                for values, feature in zip(features, select_features(image_forward_out, image.dtype)):
                    values.append(feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            features = select_features(image_forward_outs, images.dtype)

        return tuple(features) if needs_anchor else features[0]

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



class CLIPVisionTowerS2(CLIPVisionTower):
    def validate_cka_anchor_layer(self, layer):
        raise ValueError("cka_loss_vision_anchor_layer is not supported by the multiscale CLIP S2 tower.")

    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
