"""
Fundamental building blocks for neural network architectures.

Provides configurable conv, norm, activation, and composite blocks.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Registry Pattern for Components
# =============================================================================

class ComponentRegistry:
    """Registry for dynamically creating components by name."""
    
    _registries: Dict[str, Dict[str, Type]] = {
        'norm': {},
        'activation': {},
        'block': {},
    }
    
    @classmethod
    def register(cls, category: str, name: str):
        """Decorator to register a component."""
        def decorator(component_cls: Type):
            cls._registries[category][name] = component_cls
            return component_cls
        return decorator
    
    @classmethod
    def get(cls, category: str, name: str) -> Type:
        """Get a component class by category and name."""
        if category not in cls._registries:
            raise ValueError(f"Unknown category: {category}")
        if name not in cls._registries[category]:
            raise ValueError(f"Unknown {category}: {name}. Available: {list(cls._registries[category].keys())}")
        return cls._registries[category][name]
    
    @classmethod
    def list_available(cls, category: str) -> List[str]:
        """List available components in a category."""
        return list(cls._registries.get(category, {}).keys())


# =============================================================================
# Normalization Layers
# =============================================================================

@ComponentRegistry.register('norm', 'batch')
class BatchNorm2d(nn.BatchNorm2d):
    """Batch Normalization wrapper."""
    pass


@ComponentRegistry.register('norm', 'instance')
class InstanceNorm2d(nn.InstanceNorm2d):
    """Instance Normalization wrapper."""
    def __init__(self, num_features: int, **kwargs):
        super().__init__(num_features, affine=True, **kwargs)


@ComponentRegistry.register('norm', 'layer')
class LayerNorm2d(nn.Module):
    """Layer Normalization for 2D inputs (B, C, H, W)."""
    def __init__(self, num_features: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


@ComponentRegistry.register('norm', 'group')
class GroupNorm2d(nn.GroupNorm):
    """Group Normalization wrapper."""
    def __init__(self, num_features: int, num_groups: int = 32, **kwargs):
        # Ensure num_groups divides num_features
        num_groups = min(num_groups, num_features)
        while num_features % num_groups != 0:
            num_groups -= 1
        super().__init__(num_groups, num_features, **kwargs)


@ComponentRegistry.register('norm', 'none')
class NoNorm(nn.Identity):
    """No normalization (identity)."""
    def __init__(self, num_features: int = None, **kwargs):
        super().__init__()


# =============================================================================
# Activation Functions
# =============================================================================

@ComponentRegistry.register('activation', 'relu')
class ReLU(nn.ReLU):
    """ReLU activation wrapper."""
    def __init__(self, **kwargs):
        super().__init__(inplace=True)


@ComponentRegistry.register('activation', 'leaky_relu')
class LeakyReLU(nn.LeakyReLU):
    """Leaky ReLU activation wrapper."""
    def __init__(self, negative_slope: float = 0.2, **kwargs):
        super().__init__(negative_slope=negative_slope, inplace=True)


@ComponentRegistry.register('activation', 'gelu')
class GELU(nn.GELU):
    """GELU activation wrapper."""
    pass


@ComponentRegistry.register('activation', 'silu')
class SiLU(nn.SiLU):
    """SiLU/Swish activation wrapper."""
    def __init__(self, **kwargs):
        super().__init__(inplace=True)


@ComponentRegistry.register('activation', 'tanh')
class Tanh(nn.Tanh):
    """Tanh activation wrapper."""
    pass


@ComponentRegistry.register('activation', 'sigmoid')
class Sigmoid(nn.Sigmoid):
    """Sigmoid activation wrapper."""
    pass


@ComponentRegistry.register('activation', 'none')
class NoActivation(nn.Identity):
    """No activation (identity)."""
    pass


# =============================================================================
# Basic Convolution Blocks
# =============================================================================

def get_norm_layer(norm_type: str, num_features: int, **kwargs) -> nn.Module:
    """Factory function for normalization layers."""
    norm_cls = ComponentRegistry.get('norm', norm_type)
    return norm_cls(num_features, **kwargs)


def get_activation(activation_type: str, **kwargs) -> nn.Module:
    """Factory function for activation layers."""
    act_cls = ComponentRegistry.get('activation', activation_type)
    return act_cls(**kwargs)


class ConvBlock(nn.Module):
    """
    Basic convolution block: Conv -> Norm -> Activation
    
    Configurable components and order.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        norm: str = 'batch',
        activation: str = 'relu',
        bias: Optional[bool] = None,
        norm_kwargs: Optional[Dict] = None,
        activation_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        
        if padding is None:
            padding = kernel_size // 2
        
        # Bias is typically False when using normalization
        if bias is None:
            bias = (norm == 'none')
        
        norm_kwargs = norm_kwargs or {}
        activation_kwargs = activation_kwargs or {}
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )
        self.norm = get_norm_layer(norm, out_channels, **norm_kwargs)
        self.act = get_activation(activation, **activation_kwargs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ConvTransposeBlock(nn.Module):
    """
    Transposed convolution block: ConvTranspose -> Norm -> Activation
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        output_padding: int = 0,
        norm: str = 'batch',
        activation: str = 'relu',
        bias: Optional[bool] = None,
        norm_kwargs: Optional[Dict] = None,
        activation_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        
        if bias is None:
            bias = (norm == 'none')
        
        norm_kwargs = norm_kwargs or {}
        activation_kwargs = activation_kwargs or {}
        
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding, bias=bias
        )
        self.norm = get_norm_layer(norm, out_channels, **norm_kwargs)
        self.act = get_activation(activation, **activation_kwargs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class UpsampleConvBlock(nn.Module):
    """
    Upsample + Conv block: Interpolate -> Conv -> Norm -> Activation
    
    Alternative to transposed convolution (avoids checkerboard artifacts).
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        scale_factor: int = 2,
        mode: str = 'bilinear',
        norm: str = 'batch',
        activation: str = 'relu',
        bias: Optional[bool] = None,
        norm_kwargs: Optional[Dict] = None,
        activation_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        
        if bias is None:
            bias = (norm == 'none')
        
        norm_kwargs = norm_kwargs or {}
        activation_kwargs = activation_kwargs or {}
        
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = mode in ['bilinear', 'bicubic']
        
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=padding, bias=bias
        )
        self.norm = get_norm_layer(norm, out_channels, **norm_kwargs)
        self.act = get_activation(activation, **activation_kwargs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x, scale_factor=self.scale_factor, mode=self.mode,
            align_corners=self.align_corners if self.align_corners else None
        )
        return self.act(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    """
    Residual block with configurable convolutions.
    
    x -> Conv -> Norm -> Act -> Conv -> Norm -> (+x) -> Act
    """
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        norm: str = 'batch',
        activation: str = 'relu',
        dropout: float = 0.0,
        norm_kwargs: Optional[Dict] = None,
        activation_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        
        norm_kwargs = norm_kwargs or {}
        activation_kwargs = activation_kwargs or {}
        padding = kernel_size // 2
        
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False),
            get_norm_layer(norm, channels, **norm_kwargs),
            get_activation(activation, **activation_kwargs),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False),
            get_norm_layer(norm, channels, **norm_kwargs),
        )
        self.act = get_activation(activation, **activation_kwargs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class DoubleConvBlock(nn.Module):
    """
    Double convolution block (common in UNet).
    
    Conv -> Norm -> Act -> Conv -> Norm -> Act
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        kernel_size: int = 3,
        norm: str = 'batch',
        activation: str = 'relu',
        dropout: float = 0.0,
        norm_kwargs: Optional[Dict] = None,
        activation_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        
        if mid_channels is None:
            mid_channels = out_channels
        
        norm_kwargs = norm_kwargs or {}
        activation_kwargs = activation_kwargs or {}
        padding = kernel_size // 2
        
        layers = [
            nn.Conv2d(in_channels, mid_channels, kernel_size, padding=padding, bias=False),
            get_norm_layer(norm, mid_channels, **norm_kwargs),
            get_activation(activation, **activation_kwargs),
        ]
        
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        
        layers.extend([
            nn.Conv2d(mid_channels, out_channels, kernel_size, padding=padding, bias=False),
            get_norm_layer(norm, out_channels, **norm_kwargs),
            get_activation(activation, **activation_kwargs),
        ])
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# =============================================================================
# Pooling and Downsampling
# =============================================================================

class Downsample(nn.Module):
    """
    Configurable downsampling layer.
    
    Supports: maxpool, avgpool, strided conv
    """
    
    def __init__(
        self,
        channels: int,
        method: str = 'maxpool',
        factor: int = 2,
        norm: str = 'batch',
        activation: str = 'none',
    ):
        super().__init__()
        
        if method == 'maxpool':
            self.down = nn.MaxPool2d(factor)
        elif method == 'avgpool':
            self.down = nn.AvgPool2d(factor)
        elif method == 'conv':
            self.down = ConvBlock(
                channels, channels, kernel_size=factor, stride=factor, padding=0,
                norm=norm, activation=activation
            )
        else:
            raise ValueError(f"Unknown downsample method: {method}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class Upsample(nn.Module):
    """
    Configurable upsampling layer.
    
    Supports: transpose conv, interpolate + conv
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        method: str = 'transpose',
        factor: int = 2,
        norm: str = 'batch',
        activation: str = 'relu',
    ):
        super().__init__()
        
        if method == 'transpose':
            self.up = ConvTransposeBlock(
                in_channels, out_channels,
                kernel_size=factor, stride=factor, padding=0,
                norm=norm, activation=activation
            )
        elif method == 'bilinear':
            self.up = UpsampleConvBlock(
                in_channels, out_channels,
                scale_factor=factor, mode='bilinear',
                norm=norm, activation=activation
            )
        elif method == 'nearest':
            self.up = UpsampleConvBlock(
                in_channels, out_channels,
                scale_factor=factor, mode='nearest',
                norm=norm, activation=activation
            )
        else:
            raise ValueError(f"Unknown upsample method: {method}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


# =============================================================================
# Attention Blocks
# =============================================================================

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """Spatial attention module."""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.sigmoid(self.conv(y))
        return x * y


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
