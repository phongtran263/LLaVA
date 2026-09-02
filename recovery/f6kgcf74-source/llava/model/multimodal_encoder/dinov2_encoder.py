import torch
import torch.nn as nn
import torch.nn.functional as F 

from transformers import CLIPImageProcessor, AutoImageProcessor, Dinov2Config, AutoModel, Dinov2Model

cfg={
    "crop_size": 256,
    "do_center_crop": True,
    "do_normalize": True,
    "do_resize": True,
    "feature_extractor_type": "CLIPFeatureExtractor",
    "image_mean": [
        0.48145466,
        0.4578275,
        0.40821073
    ],
    "image_std": [
        0.26862954,
        0.26130258,
        0.27577711
    ],
    "resample": 3,
    "size": 256
}

class DINOv2VisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.vision_tower_name = vision_tower
        self.is_loaded = False
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.input_image_size = args.input_image_size

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = Dinov2Config.from_pretrained(self.vision_tower_name)
        
    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor(**cfg)
        if self.input_image_size is not None:
            self.image_processor.size=self.input_image_size
            self.image_processor.crop_size={
                'height':self.input_image_size,
                'width': self.input_image_size
            }
        self.image_mean = torch.tensor(self.image_processor.image_mean).view(1, 3, 1, 1)
        self.image_std = torch.tensor(self.image_processor.image_std).view(1, 3, 1, 1)

        self.dinov2_processor = AutoImageProcessor.from_pretrained(self.vision_tower_name)

        self.vision_tower = Dinov2Model.from_pretrained(self.vision_tower_name, config=self.cfg_only if hasattr(self, 'cfg_only') else None, device_map=device_map)
        self.is_loaded = True

    def feature_select(self, features):
        if self.select_layer < 0:
            selected_features = features[self.select_layer]
        else:
            selected_features = features[self.select_layer]

        if self.select_feature == 'cls':
            return selected_features[:, 0, :]
        elif self.select_feature == 'patch':
            return selected_features[:, 1:, :]
        else:
            raise ValueError(f'Unknown select_feature: {self.select_feature}')

    @torch.no_grad()
    def forward(self, images):
        mean = self.image_mean.to(device=images.device, dtype=images.dtype)
        std = self.image_std.to(device=images.device, dtype=images.dtype)
        images = images * std + mean
        images = self.dinov2_processor(images=images.float().clamp(0, 1), return_tensors="pt")

        image_features = self.vision_tower(**(images.to(device=self.device, dtype=self.dtype)), output_hidden_states=True).last_hidden_state
        b_size, seq_len, feat_dim = image_features.shape
        image_features = image_features[:,1:,:]
        image_features = image_features.transpose(1, 2).reshape(b_size, feat_dim, 16, 16)   
        image_features = F.interpolate(image_features.float(), size=(32, 32), mode='bilinear', align_corners=True).to(dtype=image_features.dtype) 
        image_features = image_features.flatten(2).transpose(1, 2)

        return image_features
    
    @property
    def dummy_feature(self):
        dummy_image = torch.zeros((1, 3, self.cfg_only.image_size, self.cfg_only.image_size))
        dummy_feature = self.forward(dummy_image)
        return dummy_feature
    
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