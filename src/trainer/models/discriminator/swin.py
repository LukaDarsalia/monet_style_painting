"""
Swin Transformer Discriminator

Uses PyTorch's pretrained Swin Transformer as a discriminator backbone.
No default values - all config must be explicit.
"""

from typing import Any, Dict

import torch
import torch.nn as nn
from torchvision import models


class SwinTransformerDiscriminator(nn.Module):
    """
    Swin Transformer based Discriminator using PyTorch's pretrained model.
    
    All config values are required - no defaults.
    
    Config structure:
        pretrained: bool
        swin_variant: str ('tiny', 'small', 'base')
        freeze_backbone: bool
        use_sigmoid: bool
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # All required - no defaults
        pretrained = config['pretrained']
        swin_variant = config['swin_variant']
        freeze_backbone = config['freeze_backbone']
        use_sigmoid = config['use_sigmoid']
        
        # Select Swin variant and weights
        if swin_variant == 'tiny':
            weights = models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.swin_t(weights=weights)
            embed_dim = 768
        elif swin_variant == 'small':
            weights = models.Swin_S_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.swin_s(weights=weights)
            embed_dim = 768
        elif swin_variant == 'base':
            weights = models.Swin_B_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.swin_b(weights=weights)
            embed_dim = 1024
        else:
            raise ValueError(f"Unknown Swin variant: {swin_variant}. Available: ['tiny', 'small', 'base']")
        
        # Replace classification head with discriminator head
        self.backbone.head = nn.Identity()
        
        # Discriminator head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )
        
        if use_sigmoid:
            self.head.add_module('sigmoid', nn.Sigmoid())
        
        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Tensor of shape (B, 1) with real/fake scores
        """
        # Swin expects 224x224, resize if needed
        if x.shape[2] != 224 or x.shape[3] != 224:
            x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Extract features
        features = self.backbone(x)
        
        # Classification
        out = self.head(features)
        return out
