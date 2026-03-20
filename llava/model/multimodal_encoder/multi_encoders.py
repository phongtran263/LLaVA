import torch
import torch.nn as nn
from copy import deepcopy

from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .pix2struct_encoder import Pix2StructVisionTower
from .dinov2_encoder import DINOv2VisionTower
from .owl_encoder import OwlVisionTower

class MultiEncoders(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.vision_tower_name = vision_tower
        self.is_loaded = False

        self.load_model()

    def load_model(self, device_map=None):
        self.vision_tower = []
        if 'clip' in self.vision_tower_name:
            self.clip_vision_tower = CLIPVisionTower(self.vision_tower_name, args=self.args, delay_load=False)
            self.vision_tower.append(self.clip_vision_tower)
        if 'pix2struct' in self.vision_tower_name:
            pix_args = deepcopy(self.args)
            pix_args.input_image_size = 1024
            self.pix2struct_vision_tower = Pix2StructVisionTower(self.vision_tower_name, args=pix_args, delay_load=False)
            self.vision_tower.append(self.pix2struct_vision_tower)
        if 'dinov2' in self.vision_tower_name:
            dino_args = deepcopy(self.args)
            dino_args.input_image_size = 1024
            self.dinov2_vision_tower = DINOv2VisionTower(self.vision_tower_name, args=dino_args, delay_load=False)
            self.vision_tower.append(self.dinov2_vision_tower)
        if 'owl' in self.vision_tower_name:
            owl_args = deepcopy(self.args)
            owl_args.input_image_size = 1024
            self.owl_vision_tower = OwlVisionTower(self.vision_tower_name, args=owl_args, delay_load=False)
            self.vision_tower.append(self.owl_vision_tower)
    
    @torch.no_grad()
    def forward(self, images):
        vision_outputs = []
        for vision_tower in self.vision_tower:
            vision_out = vision_tower(images)
            vision_outputs.append(vision_out)
        return vision_outputs