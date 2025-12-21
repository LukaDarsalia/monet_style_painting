"""
Trainer module for Monet GAN.

This module provides a flexible CycleGAN training pipeline with:
- LEGO-style configurable generator architectures (build block-by-block in config)
- Configurable discriminator architectures (PatchGAN, Spectral, MultiScale, Swin)
- Multiple loss functions (LSGAN, WGAN, WGAN-GP, Hinge, Vanilla)
- MiFID evaluation metric (resizes to 256x256 for consistency)
- W&B logging and checkpoint management
"""

from .trainer import CycleGANTrainer
from .losses import (
    GANLoss,
    CycleLoss,
    IdentityLoss,
    PerceptualLoss,
    GradientPenalty,
    LossManager,
    ConstantScheduler,
    CosineScheduler,
    LinearScheduler,
    StepScheduler,
)
from .evaluation import (
    InceptionV3Features,
    MiFIDCalculator,
    Evaluator,
    calculate_frechet_distance,
)
from .dataset import (
    ImageDataset,
    UnpairedDataset,
    ImageBuffer,
    create_dataloaders,
)

__all__ = [
    # Main trainer
    "CycleGANTrainer",
    # Losses
    "GANLoss",
    "CycleLoss",
    "IdentityLoss",
    "PerceptualLoss",
    "GradientPenalty",
    "LossManager",
    # Schedulers
    "ConstantScheduler",
    "CosineScheduler",
    "LinearScheduler",
    "StepScheduler",
    # Evaluation
    "InceptionV3Features",
    "MiFIDCalculator",
    "Evaluator",
    "calculate_frechet_distance",
    # Dataset
    "ImageDataset",
    "UnpairedDataset",
    "ImageBuffer",
    "create_dataloaders",
]
