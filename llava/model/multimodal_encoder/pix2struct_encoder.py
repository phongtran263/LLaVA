import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPImageProcessor, Pix2StructForConditionalGeneration, Pix2StructConfig, AutoProcessor

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

class Pix2StructVisionTower(nn.Module):
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
            self.cfg_only = Pix2StructConfig.from_pretrained(self.vision_tower_name)
        
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

        self.pix2struct_processor = AutoProcessor.from_pretrained("google/pix2struct-large")
        self.pix2struct_processor.image_processor.is_vqa = False
            
        model = Pix2StructForConditionalGeneration.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower = model.encoder
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features
    
    @torch.no_grad()
    def forward(self, images):
        mean = self.image_mean.clone().view(1, 3, 1, 1).to(dtype=images.dtype, device=images.device)
        std = self.image_std.clone().view(1, 3, 1, 1).to(dtype=images.dtype, device=images.device)
        images = (images * std + mean) * 255.0
        images = self.pix2struct_processor(images=images.float(), return_tensors="pt")

        image_features = self.vision_tower(**(images.to(device=self.device, dtype=self.dtype)), output_hidden_states=True).last_hidden_state
        b_size, seq_len, feat_dim = image_features.shape
        image_features = image_features[:,:2025,:]
        image_features = image_features.transpose(1, 2).reshape(b_size, feat_dim, 45, 45)   
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
    


    
        
        