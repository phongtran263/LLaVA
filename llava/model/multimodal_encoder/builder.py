from copy import deepcopy
import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .pix2struct_encoder import Pix2StructVisionTower
from .dinov2_encoder import DINOv2VisionTower
from .owl_encoder import OwlVisionTower
from .multi_encoders import MultiEncoders


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    if ',' in vision_tower:
        return MultiEncoders(vision_tower, args=vision_tower_cfg, **kwargs)
    elif is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    elif 'pix2struct' in vision_tower:
        pix_args = deepcopy(vision_tower_cfg)
        pix_args.input_image_size = 1024
        return Pix2StructVisionTower(vision_tower, args=pix_args, **kwargs)
    
    elif 'dinov2' in vision_tower:
        dino_args = deepcopy(vision_tower_cfg)
        dino_args.input_image_size = 1024
        return DINOv2VisionTower(vision_tower, args=dino_args, **kwargs)

    elif 'owl' in vision_tower:
        codetr_args = deepcopy(vision_tower_cfg)
        codetr_args.input_image_size = 1024
        return OwlVisionTower(vision_tower, args=codetr_args, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
