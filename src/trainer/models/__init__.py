"""
Model architectures for Monet GAN.

LEGO-style architecture building - define models block by block in config.
"""

from .generator import (
    LEGOGenerator,
    build_generator,
    build_block,
    build_sequence,
    BLOCK_REGISTRY,
)
from .discriminator import (
    build_discriminator,
    PatchGANDiscriminator,
    DISCRIMINATORS,
)

__all__ = [
    # Generator
    "LEGOGenerator",
    "build_generator",
    "build_block",
    "build_sequence",
    "BLOCK_REGISTRY",
    # Discriminator
    "build_discriminator",
    "PatchGANDiscriminator",
    "DISCRIMINATORS",
]
