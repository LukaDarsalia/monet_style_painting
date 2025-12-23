"""
Trainer module for Monet GAN.

Paper-style CycleGAN training with:
- LEGO-style configurable generator architectures
- PatchGAN discriminator builder
- W&B logging and checkpoint management
"""

from .trainer import CycleGANTrainer
from .dataset import (
    ImageDataset,
    UnpairedDataset,
    ImageBuffer,
    create_dataloaders,
)
from .logger import WandbLogger

__all__ = [
    # Main trainer
    "CycleGANTrainer",
    # Dataset
    "ImageDataset",
    "UnpairedDataset",
    "ImageBuffer",
    "create_dataloaders",
    # Logger
    "WandbLogger",
]
