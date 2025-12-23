"""
Discriminator architectures for Monet GAN.
"""

from typing import Any, Dict

import torch.nn as nn

from .patchgan import PatchGANDiscriminator


DISCRIMINATORS = {
    'patchgan': PatchGANDiscriminator,
}


def build_discriminator(config: Dict[str, Any]) -> nn.Module:
    """
    Build discriminator from config.
    
    Config must have 'type' key specifying discriminator architecture.
    """
    disc_type = config['type']
    
    if disc_type not in DISCRIMINATORS:
        raise ValueError(f"Unknown discriminator type: {disc_type}. Available: {list(DISCRIMINATORS.keys())}")
    
    return DISCRIMINATORS[disc_type](config)


__all__ = [
    "PatchGANDiscriminator",
    "build_discriminator",
    "DISCRIMINATORS",
]
