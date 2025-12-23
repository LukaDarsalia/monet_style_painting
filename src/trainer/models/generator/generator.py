"""
LEGO-style Generator Architecture

Build generators by defining sequential blocks in config.
Each block type maps to a specific nn.Module.
"""

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Block Registry - Maps block names to builder functions
# =============================================================================

BLOCK_REGISTRY: Dict[str, type] = {}


def register_block(name: str):
    """Decorator to register a block type."""
    def decorator(cls):
        BLOCK_REGISTRY[name] = cls
        return cls
    return decorator


# =============================================================================
# Normalization and Activation Factories
# =============================================================================

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


# =============================================================================
# Basic Building Blocks
# =============================================================================

@register_block('conv')
class ConvBlock(nn.Module):
    """
    Convolution block: Conv -> Norm -> Activation
    
    Config:
        in_channels: int
        out_channels: int
        kernel_size: int
        stride: int
        padding: int
        padding_mode: str (optional, defaults to zeros)
        norm: str
        activation: str
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        in_ch = config['in_channels']
        out_ch = config['out_channels']
        kernel = config['kernel_size']
        stride = config['stride']
        padding = config['padding']
        norm = config['norm']
        activation = config['activation']
        padding_mode = config.get('padding_mode', 'zeros')
        
        bias = (norm == 'none')
        
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel, stride=stride, padding=padding, bias=bias,
            padding_mode=padding_mode,
        )
        self.norm = build_norm(norm, out_ch)
        self.act = build_activation(activation)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


@register_block('conv_transpose')
class ConvTransposeBlock(nn.Module):
    """
    Transposed convolution block: ConvTranspose -> Norm -> Activation
    
    Config:
        in_channels: int
        out_channels: int
        kernel_size: int
        stride: int
        padding: int
        output_padding: int
        norm: str
        activation: str
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        in_ch = config['in_channels']
        out_ch = config['out_channels']
        kernel = config['kernel_size']
        stride = config['stride']
        padding = config['padding']
        output_padding = config['output_padding']
        norm = config['norm']
        activation = config['activation']
        
        bias = (norm == 'none')
        
        self.conv = nn.ConvTranspose2d(
            in_ch, out_ch, kernel, stride=stride, 
            padding=padding, output_padding=output_padding, bias=bias
        )
        self.norm = build_norm(norm, out_ch)
        self.act = build_activation(activation)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


@register_block('upsample_conv')
class UpsampleConvBlock(nn.Module):
    """
    Upsample + Conv block: Interpolate -> Conv -> Norm -> Activation
    
    Config:
        in_channels: int
        out_channels: int
        kernel_size: int
        scale_factor: int
        mode: str (bilinear, nearest)
        norm: str
        activation: str
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        in_ch = config['in_channels']
        out_ch = config['out_channels']
        kernel = config['kernel_size']
        self.scale_factor = config['scale_factor']
        self.mode = config['mode']
        norm = config['norm']
        activation = config['activation']
        
        padding = kernel // 2
        bias = (norm == 'none')
        
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=bias)
        self.norm = build_norm(norm, out_ch)
        self.act = build_activation(activation)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        align = self.mode in ['bilinear', 'bicubic']
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode,
                          align_corners=align if align else None)
        return self.act(self.norm(self.conv(x)))


@register_block('residual')
class ResidualBlock(nn.Module):
    """
    Residual block: x + Conv -> Norm -> Act -> Conv -> Norm
    
    Config:
        channels: int
        kernel_size: int
        padding_mode: str (optional, defaults to zeros)
        norm: str
        activation: str
        dropout: float
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        channels = config['channels']
        kernel = config['kernel_size']
        norm = config['norm']
        activation = config['activation']
        dropout = config['dropout']
        padding_mode = config.get('padding_mode', 'zeros')
        
        padding = kernel // 2
        
        layers = [
            nn.Conv2d(
                channels, channels, kernel, padding=padding, bias=False,
                padding_mode=padding_mode,
            ),
            build_norm(norm, channels),
            build_activation(activation),
        ]
        
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        
        layers.extend([
            nn.Conv2d(
                channels, channels, kernel, padding=padding, bias=False,
                padding_mode=padding_mode,
            ),
            build_norm(norm, channels),
        ])
        
        self.block = nn.Sequential(*layers)
        self.act = build_activation(activation)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


@register_block('double_conv')
class DoubleConvBlock(nn.Module):
    """
    Double convolution block: Conv -> Norm -> Act -> Conv -> Norm -> Act
    
    Config:
        in_channels: int
        out_channels: int
        kernel_size: int
        padding_mode: str (optional, defaults to zeros)
        norm: str
        activation: str
        dropout: float
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        in_ch = config['in_channels']
        out_ch = config['out_channels']
        kernel = config['kernel_size']
        norm = config['norm']
        activation = config['activation']
        dropout = config['dropout']
        padding_mode = config.get('padding_mode', 'zeros')
        
        padding = kernel // 2
        
        layers = [
            nn.Conv2d(
                in_ch, out_ch, kernel, padding=padding, bias=False,
                padding_mode=padding_mode,
            ),
            build_norm(norm, out_ch),
            build_activation(activation),
        ]
        
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        
        layers.extend([
            nn.Conv2d(
                out_ch, out_ch, kernel, padding=padding, bias=False,
                padding_mode=padding_mode,
            ),
            build_norm(norm, out_ch),
            build_activation(activation),
        ])
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


@register_block('max_pool')
class MaxPoolBlock(nn.Module):
    """
    Max pooling block.
    
    Config:
        kernel_size: int
        stride: int (optional, defaults to kernel_size)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        kernel = config['kernel_size']
        stride = config.get('stride', kernel)
        
        self.pool = nn.MaxPool2d(kernel, stride=stride)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


@register_block('avg_pool')
class AvgPoolBlock(nn.Module):
    """
    Average pooling block.
    
    Config:
        kernel_size: int
        stride: int (optional, defaults to kernel_size)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        kernel = config['kernel_size']
        stride = config.get('stride', kernel)
        
        self.pool = nn.AvgPool2d(kernel, stride=stride)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


@register_block('dropout')
class DropoutBlock(nn.Module):
    """
    Dropout block.
    
    Config:
        p: float
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.dropout = nn.Dropout2d(config['p'])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x)


# =============================================================================
# Block Sequence Builder
# =============================================================================

def build_block(block_config: Dict[str, Any]) -> nn.Module:
    """Build a single block from config."""
    block_type = block_config['type']
    
    if block_type not in BLOCK_REGISTRY:
        raise ValueError(f"Unknown block type: {block_type}. Available: {list(BLOCK_REGISTRY.keys())}")
    
    return BLOCK_REGISTRY[block_type](block_config)


def build_sequence(blocks_config: List[Dict[str, Any]]) -> nn.ModuleList:
    """Build a sequence of blocks from config list."""
    return nn.ModuleList([build_block(cfg) for cfg in blocks_config])


# =============================================================================
# LEGO Generator
# =============================================================================

class LEGOGenerator(nn.Module):
    """
    LEGO-style Generator where architecture is defined entirely in config.
    
    Supports skip connections for UNet-like architectures.
    
    Config structure:
        encoder:
          - {type: conv, ...}
          - {type: max_pool, ...}
          - ...
        bottleneck:
          - {type: residual, ...}
          - ...
        decoder:
          - {type: upsample_conv, ...}
          - {type: conv, ...}
          - ...
        skip_connections:
          - [encoder_idx, decoder_idx]  # Connect output of encoder[idx] to input of decoder[idx]
          - ...
        output:
          - {type: conv, ...}
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        
        # Build encoder
        self.encoder = build_sequence(config['encoder'])
        
        # Build bottleneck
        self.bottleneck = build_sequence(config['bottleneck'])
        
        # Build decoder
        self.decoder = build_sequence(config['decoder'])
        
        # Build output
        self.output = build_sequence(config['output'])
        
        # Skip connections: list of (encoder_idx, decoder_idx) pairs
        # Store which encoder outputs go to which decoder inputs
        self.skip_connections = config.get('skip_connections', [])
        
        # Validate skip connections
        for enc_idx, dec_idx in self.skip_connections:
            if enc_idx >= len(self.encoder):
                raise ValueError(f"Skip connection encoder index {enc_idx} out of range")
            if dec_idx >= len(self.decoder):
                raise ValueError(f"Skip connection decoder index {dec_idx} out of range")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder - store outputs for skip connections
        encoder_outputs = []
        for i, block in enumerate(self.encoder):
            x = block(x)
            encoder_outputs.append(x)
        
        # Bottleneck
        for block in self.bottleneck:
            x = block(x)
        
        # Decoder with skip connections
        for i, block in enumerate(self.decoder):
            # Check if this decoder block has a skip connection
            for enc_idx, dec_idx in self.skip_connections:
                if dec_idx == i:
                    skip = encoder_outputs[enc_idx]
                    # Handle size mismatch
                    if x.shape[2:] != skip.shape[2:]:
                        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
                    x = torch.cat([x, skip], dim=1)
                    break
            x = block(x)
        
        # Output
        for block in self.output:
            x = block(x)
        
        return x

    @torch.no_grad()
    def trace_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """
        Forward pass that records block-by-block shapes and skip connections.
        
        Returns:
            output tensor, trace list
        """
        trace: List[Dict[str, Any]] = []
        encoder_outputs = []
        
        for i, block in enumerate(self.encoder):
            in_shape = tuple(x.shape)
            x = block(x)
            out_shape = tuple(x.shape)
            encoder_outputs.append(x)
            trace.append({
                'stage': 'encoder',
                'idx': i,
                'block': block.__class__.__name__,
                'in_shape': in_shape,
                'out_shape': out_shape,
            })
        
        for i, block in enumerate(self.bottleneck):
            in_shape = tuple(x.shape)
            x = block(x)
            out_shape = tuple(x.shape)
            trace.append({
                'stage': 'bottleneck',
                'idx': i,
                'block': block.__class__.__name__,
                'in_shape': in_shape,
                'out_shape': out_shape,
            })
        
        for i, block in enumerate(self.decoder):
            skip_info = None
            for enc_idx, dec_idx in self.skip_connections:
                if dec_idx == i:
                    skip = encoder_outputs[enc_idx]
                    resized = False
                    if x.shape[2:] != skip.shape[2:]:
                        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
                        resized = True
                    x = torch.cat([x, skip], dim=1)
                    skip_info = {
                        'from': enc_idx,
                        'skip_shape': tuple(skip.shape),
                        'resized': resized,
                        'concat_shape': tuple(x.shape),
                    }
                    break
            in_shape = tuple(x.shape)
            x = block(x)
            out_shape = tuple(x.shape)
            trace.append({
                'stage': 'decoder',
                'idx': i,
                'block': block.__class__.__name__,
                'in_shape': in_shape,
                'out_shape': out_shape,
                'skip': skip_info,
            })
        
        for i, block in enumerate(self.output):
            in_shape = tuple(x.shape)
            x = block(x)
            out_shape = tuple(x.shape)
            trace.append({
                'stage': 'output',
                'idx': i,
                'block': block.__class__.__name__,
                'in_shape': in_shape,
                'out_shape': out_shape,
            })
        
        return x, trace


# =============================================================================
# Factory Function
# =============================================================================

def build_generator(config: Dict[str, Any]) -> nn.Module:
    """Build generator from config."""
    return LEGOGenerator(config)
