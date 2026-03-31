import torch
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F

from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2, CLIPVisionConfig, CLIPImageProcessor
from .pix2struct_encoder import Pix2StructVisionTower, Pix2StructConfig
from .dinov2_encoder import DINOv2VisionTower, Dinov2Config
from .owl_encoder import OwlVisionTower, Owlv2Config
from ..multimodal_projector.builder import build_vision_projector

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

def get_w(weights, keyword):
    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

def load_mm_projector_from_checkpoint(vision_tower, projector_ckpt, args):
    tmp_args = deepcopy(args)
    tmp_args.mm_hidden_size = vision_tower.config.hidden_size
    mm_projector = build_vision_projector(tmp_args)
    mm_projector_weights = torch.load(projector_ckpt, map_location='cpu')
    mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
    return mm_projector

class TextGuidedRouter(nn.Module):
    def __init__(self, text_hidden_size, num_experts, hidden_size=512, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.mlp = nn.Sequential(
            nn.LayerNorm(text_hidden_size),
            nn.Linear(text_hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_experts)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, text_features):
        logits = self.mlp(text_features)
        alphas = torch.softmax(logits, dim=-1)
        return alphas
    
def build_teacher_mixture(teacher_features, router_alphas):
    stacked_teacher_features = torch.stack(teacher_features, dim=1)
    weights = router_alphas.unsqueeze(-1).unsqueeze(-1)
    weighted_teacher_features = stacked_teacher_features * weights
    mixed_teacher_feature = weighted_teacher_features.sum(dim=1)
    return mixed_teacher_feature

def load_balance_loss(router_alphas):
    mean_alphas = router_alphas.mean(dim=0)
    entropy = (mean_alphas * (mean_alphas + 1e-8).log()).sum()
    return entropy

class MultiEncoders(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.vision_tower_names = vision_tower.split(',')
        self.mm_projectors_chkpt = args.pretrain_mm_mlp_adapter.split(',')
        self.is_loaded = False
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.input_image_size = args.input_image_size
        self.delay_load = delay_load
        self.args = args

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = [CLIPVisionConfig.from_pretrained(name) if 'clip' in name else 
                             Pix2StructConfig.from_pretrained(name) if 'pix2struct' in name else 
                             Dinov2Config.from_pretrained(name) if 'dinov2' in name else 
                             Owlv2Config.from_pretrained(name) if 'owl' in name else None
                             for name in self.vision_tower_names]
            
    def load_model(self, device_map=None):
        if self.is_loaded:
            print('MultiEncoders is already loaded, `load_model` called again, skipping.')
            return
        
        self.vision_towers = nn.ModuleList()
        self.mm_projectors = nn.ModuleList()
        for i in range(len(self.vision_tower_names)):
            vision_tower_name = self.vision_tower_names[i]
            projector_ckpt = self.mm_projectors_chkpt[i] 
            if 'clip' in vision_tower_name:
                vision_tower = CLIPVisionTower(vision_tower_name, self.args, delay_load=False)
                vision_tower.input_image_size = vision_tower.config.image_size
                mm_projector = load_mm_projector_from_checkpoint(vision_tower, projector_ckpt, self.args)
            elif 'pix2struct' in vision_tower_name:
                vision_tower = Pix2StructVisionTower(vision_tower_name, self.args, delay_load=False)
                mm_projector = load_mm_projector_from_checkpoint(vision_tower, projector_ckpt, self.args)
            elif 'dinov2' in vision_tower_name:
                vision_tower = DINOv2VisionTower(vision_tower_name, self.args, delay_load=False)
                mm_projector = load_mm_projector_from_checkpoint(vision_tower, projector_ckpt, self.args)
            elif 'owl' in vision_tower_name:
                vision_tower = OwlVisionTower(vision_tower_name, self.args, delay_load=False)
                mm_projector = load_mm_projector_from_checkpoint(vision_tower, projector_ckpt, self.args)
            else:
                raise ValueError(f'Unknown vision tower name: {vision_tower_name}')
            self.vision_towers.append(vision_tower)
            self.mm_projectors.append(mm_projector)

        for vision_tower in self.vision_towers:
            vision_tower.requires_grad_(False)
        for mm_projector in self.mm_projectors:
            mm_projector.requires_grad_(False)

        if self.args.train_mtd:
        #     self.student_vision_tower = CLIPVisionTower(self.args.student_vision_tower, self.args, delay_load=False)
        #     self.student_vision_tower.input_image_size = self.student_vision_tower.config.image_size
        #     self.student_mm_projector = load_mm_projector_from_checkpoint(self.student_vision_tower, self.args.pretrain_student_mm_mlp_adapter, self.args)
        #     self.student_vision_tower.requires_grad_(True)
        #     self.student_mm_projector.requires_grad_(True)
            self.router = TextGuidedRouter(text_hidden_size=self.args.text_hidden_size, num_experts=len(self.vision_tower_names), hidden_size=self.args.router_hidden_size, dropout=self.args.router_dropout)
            self.router.requires_grad_(True)

        self.image_processor = CLIPImageProcessor(**cfg)
        if self.input_image_size is not None:
            self.image_processor.size=self.input_image_size
            self.image_processor.crop_size={
                'height':self.input_image_size,
                'width': self.input_image_size
            }

        self.is_loaded = True

    @torch.no_grad()
    def forward(self, images, text_features=None):
        vision_tower_outputs = []
        for i in range(len(self.vision_towers)):
            if self.vision_towers[i].input_image_size != self.image_processor.size:
                resized_images = F.interpolate(images, size=(self.vision_towers[i].input_image_size, self.vision_towers[i].input_image_size), mode='bilinear', align_corners=False).to(dtype=self.dtype, device=self.device)
            else:
                resized_images = images
            vision_tower_output = self.vision_towers[i](resized_images)
            vision_tower_output = self.mm_projectors[i](vision_tower_output)
            if vision_tower_output.shape[1] == 1024:
                vision_tower_output = vision_tower_output.transpose(1, 2).reshape(vision_tower_output.shape[0], vision_tower_output.shape[2], 32, 32)
                vision_tower_output = F.interpolate(vision_tower_output.float(), size=(24, 24), mode='bilinear', align_corners=True).to(dtype=vision_tower_output.dtype)
                vision_tower_output = vision_tower_output.flatten(2).transpose(1, 2)
            vision_tower_outputs.append(vision_tower_output)
        if self.args.train_mtd and text_features is not None:
            alpha = self.router(text_features)

            if self.training:
                print('Router alphas:', alpha)
                print(len(vision_tower_outputs), 'vision tower outputs')
                stacked = torch.stack(vision_tower_outputs, dim=1)
                weights = alpha.unsqueeze(-1).unsqueeze(-1)
                mixed = (weights * stacked).sum(dim=1)
                return mixed
            else:
                topk = self.args.mtd_topk if self.args.mtd_topk is not None else 2
                topk_vals, topk_idx = torch.topk(alpha, topk, dim=-1) # [B, top_k]
                topk_weights = topk_vals / topk_vals.sum(dim=-1, keepdim=True)

                B, N, D = vision_tower_outputs[0].shape
                mixed = torch.zeros(B, N, D, dtype=self.dtype, device=self.device)
                for k in range(topk):
                    for b in range(B):
                        enc_idx = topk_idx[b, k].item()
                        mixed[b] += topk_weights[b, k] * vision_tower_outputs[enc_idx][b]
                return mixed
        vision_tower_outputs = torch.cat(vision_tower_outputs, dim=-1)
        return vision_tower_outputs
    
    @property
    def dummy_feature(self):
        dummy_features = []
        for vision_tower in self.vision_towers:
            dummy_feature = vision_tower.dummy_feature
            dummy_features.append(dummy_feature)
        return dummy_features
    
    @property
    def dtype(self):
        return self.vision_towers[0].dtype if self.is_loaded else self.cfg_only[0].torch_dtype
    
    @property
    def device(self):
        return next(self.vision_towers[0].parameters()).device if self.is_loaded else torch.device('cpu')
    
    @property
    def config(self):
        if self.is_loaded:
            return [vision_tower.config for vision_tower in self.vision_towers]
        else:
            return self.cfg_only
        
    @property
    def hidden_size(self):
        if self.args.train_mtd:
            return self.args.hidden_size
        return self.args.hidden_size*len(self.vision_towers)
        
    @property
    def num_patches_per_side(self):
        if self.is_loaded:
            return [vision_tower.config.image_size // vision_tower.config.patch_size for vision_tower in self.vision_towers]
        else:
            return [cfg.image_size // cfg.patch_size for cfg in self.cfg_only]
        
    @property
    def num_patches(self):
        if self.is_loaded:
            return [(vision_tower.config.image_size // vision_tower.config.patch_size) ** 2 for vision_tower in self.vision_towers]
        else:
            return [(cfg.image_size // cfg.patch_size) ** 2 for cfg in self.cfg_only]

