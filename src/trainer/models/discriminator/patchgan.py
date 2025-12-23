"""
PatchGAN Discriminator Architecture (CycleGAN paper).

No default values - all config must be explicit.
"""

from typing import Any, Dict

import torch
import torch.nn as nn


def build_norm(norm_type: str, channels: int) -> nn.Module:
    """Build normalization layer."""
    if norm_type == 'batch':
        return nn.BatchNorm2d(channels)
    elif norm_type == 'instance':
        return nn.InstanceNorm2d(channels, affine=True)
    elif norm_type == 'layer':
        return nn.GroupNorm(1, channels)
    elif norm_type == 'group':
        num_groups = min(32, channels)
        while channels % num_groups != 0:
            num_groups -= 1
        return nn.GroupNorm(num_groups, channels)
    elif norm_type == 'none':
        return nn.Identity()
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


def build_activation(act_type: str) -> nn.Module:
    """Build activation layer."""
    if act_type == 'relu':
        return nn.ReLU(inplace=True)
    elif act_type == 'leaky_relu':
        return nn.LeakyReLU(0.2, inplace=True)
    elif act_type == 'gelu':
        return nn.GELU()
    elif act_type == 'silu':
        return nn.SiLU(inplace=True)
    elif act_type == 'tanh':
        return nn.Tanh()
    elif act_type == 'sigmoid':
        return nn.Sigmoid()
    elif act_type == 'none':
        return nn.Identity()
    else:
        raise ValueError(f"Unknown activation type: {act_type}")


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator.
    
    All config values are required - no defaults.
    
    Config structure:
        input_channels: int
        base_channels: int
        num_layers: int
        kernel_size: int
        norm: str
        activation: str
        use_sigmoid: bool
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # All required - no defaults
        in_ch = config['input_channels']
        base_ch = config['base_channels']
        num_layers = config['num_layers']
        kernel_size = config['kernel_size']
        norm = config['norm']
        activation = config['activation']
        use_sigmoid = config['use_sigmoid']
        
        padding = 1
        
        # First layer (no normalization)
        layers = [
            nn.Conv2d(in_ch, base_ch, kernel_size, stride=2, padding=padding),
            build_activation(activation),
        ]
        
        # Middle layers
        ch_mult = 1
        ch_mult_prev = 1
        for i in range(1, num_layers):
            ch_mult_prev = ch_mult
            ch_mult = min(2 ** i, 8)
            layers.extend([
                nn.Conv2d(base_ch * ch_mult_prev, base_ch * ch_mult, kernel_size, stride=2, padding=padding, bias=False),
                build_norm(norm, base_ch * ch_mult),
                build_activation(activation),
            ])
        
        # Second to last layer (stride 1)
        ch_mult_prev = ch_mult
        ch_mult = min(2 ** num_layers, 8)
        layers.extend([
            nn.Conv2d(base_ch * ch_mult_prev, base_ch * ch_mult, kernel_size, stride=1, padding=padding, bias=False),
            build_norm(norm, base_ch * ch_mult),
            build_activation(activation),
        ])
        
        # Output layer
        layers.append(nn.Conv2d(base_ch * ch_mult, 1, kernel_size, stride=1, padding=padding))
        
        if use_sigmoid:
            layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

