"""
Loss Functions and Weight Schedulers for GAN Training

Supports CycleGAN, WGAN-GP, LSGAN losses with configurable weight scheduling.
"""

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Weight Schedulers
# =============================================================================

class ConstantScheduler:
    """Constant weight scheduler."""
    
    def __init__(self, value: float):
        self.value = value
    
    def get_value(self, step: int, total_steps: int) -> float:
        return self.value


class CosineScheduler:
    """Cosine annealing weight scheduler."""
    
    def __init__(
        self,
        start_value: float,
        end_value: float,
        warmup_steps: int = 0,
    ):
        self.start_value = start_value
        self.end_value = end_value
        self.warmup_steps = warmup_steps
    
    def get_value(self, step: int, total_steps: int) -> float:
        if step < self.warmup_steps:
            # Linear warmup
            return self.start_value * (step / self.warmup_steps)
        
        # Cosine annealing
        progress = (step - self.warmup_steps) / max(1, total_steps - self.warmup_steps)
        return self.end_value + 0.5 * (self.start_value - self.end_value) * (1 + math.cos(math.pi * progress))


class LinearScheduler:
    """Linear weight scheduler."""
    
    def __init__(
        self,
        start_value: float,
        end_value: float,
        warmup_steps: int = 0,
    ):
        self.start_value = start_value
        self.end_value = end_value
        self.warmup_steps = warmup_steps
    
    def get_value(self, step: int, total_steps: int) -> float:
        if step < self.warmup_steps:
            return self.start_value * (step / self.warmup_steps)
        
        progress = (step - self.warmup_steps) / max(1, total_steps - self.warmup_steps)
        return self.start_value + progress * (self.end_value - self.start_value)


class StepScheduler:
    """Step decay weight scheduler."""
    
    def __init__(
        self,
        initial_value: float,
        decay_factor: float = 0.5,
        decay_steps: int = 1000,
    ):
        self.initial_value = initial_value
        self.decay_factor = decay_factor
        self.decay_steps = decay_steps
    
    def get_value(self, step: int, total_steps: int) -> float:
        num_decays = step // self.decay_steps
        return self.initial_value * (self.decay_factor ** num_decays)


SCHEDULERS = {
    'constant': ConstantScheduler,
    'cosine': CosineScheduler,
    'linear': LinearScheduler,
    'step': StepScheduler,
}


def build_scheduler(config: Union[float, Dict[str, Any]]) -> Any:
    """
    Build a scheduler from config.
    
    If config is a float/int, returns ConstantScheduler.
    If config is a dict, builds the specified scheduler type (type key required).
    """
    if isinstance(config, (int, float)):
        return ConstantScheduler(float(config))
    
    scheduler_type = config['type']
    if scheduler_type not in SCHEDULERS:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}. Available: {list(SCHEDULERS.keys())}")
    
    scheduler_cls = SCHEDULERS[scheduler_type]
    params = {k: v for k, v in config.items() if k != 'type'}
    return scheduler_cls(**params)


# =============================================================================
# GAN Loss Functions
# =============================================================================

class GANLoss(nn.Module):
    """
    Base GAN loss class.
    
    Supports different GAN objectives:
    - vanilla: BCE loss (original GAN)
    - lsgan: MSE loss (Least Squares GAN)
    - wgan-gp: Wasserstein loss with gradient penalty
    - hinge: Hinge loss
    """
    
    def __init__(self, gan_mode: str, target_real_label: float = 1.0, target_fake_label: float = 0.0):
        super().__init__()
        
        self.gan_mode = gan_mode
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'wgan-gp':
            self.loss = None
        elif gan_mode == 'hinge':
            self.loss = None
        else:
            raise ValueError(f"Unknown GAN mode: {gan_mode}")
    
    def get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """Create target tensor with same shape as prediction."""
        target_label = self.real_label if target_is_real else self.fake_label
        return target_label.expand_as(prediction)
    
    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """
        Calculate loss.
        
        Args:
            prediction: Discriminator output
            target_is_real: Whether target should be real (True) or fake (False)
        """
        if self.gan_mode in ['vanilla', 'lsgan']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            return self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgan-gp':
            if target_is_real:
                return -prediction.mean()
            else:
                return prediction.mean()
        elif self.gan_mode == 'hinge':
            if target_is_real:
                return F.relu(1.0 - prediction).mean()
            else:
                return F.relu(1.0 + prediction).mean()


class CycleLoss(nn.Module):
    """Cycle consistency loss for CycleGAN."""
    
    def __init__(self, loss_type: str):
        super().__init__()
        if loss_type == 'l1':
            self.loss = nn.L1Loss()
        elif loss_type == 'l2':
            self.loss = nn.MSELoss()
        else:
            raise ValueError(f"Unknown cycle loss type: {loss_type}")
    
    def forward(self, reconstructed: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        return self.loss(reconstructed, original)


class IdentityLoss(nn.Module):
    """Identity loss for CycleGAN (encourages generator to be identity for target domain)."""
    
    def __init__(self, loss_type: str):
        super().__init__()
        if loss_type == 'l1':
            self.loss = nn.L1Loss()
        elif loss_type == 'l2':
            self.loss = nn.MSELoss()
        else:
            raise ValueError(f"Unknown identity loss type: {loss_type}")
    
    def forward(self, identity_output: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        return self.loss(identity_output, original)


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features.
    
    Compares feature representations rather than pixel values.
    """

    def __init__(
        self,
        layers: List[int] = [3, 8, 15, 22],
        weights: List[float] = None,
        input_mean: Optional[List[float]] = None,
        input_std: Optional[List[float]] = None,
    ):
        super().__init__()
        
        from torchvision import models
        
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.layers = layers
        self.weights = weights or [1.0] * len(layers)
        
        # Extract VGG layers
        self.vgg_layers = nn.ModuleList()
        prev_layer = 0
        for layer in layers:
            self.vgg_layers.append(nn.Sequential(*list(vgg.children())[prev_layer:layer+1]))
            prev_layer = layer + 1
        
        # Freeze VGG
        for param in self.parameters():
            param.requires_grad = False

        # Input normalization (defaults to [-1, 1] -> [0, 1])
        input_mean = input_mean if input_mean is not None else [0.5, 0.5, 0.5]
        input_std = input_std if input_std is not None else [0.5, 0.5, 0.5]
        self.register_buffer('input_mean', torch.tensor(input_mean).view(1, 3, 1, 1))
        self.register_buffer('input_std', torch.tensor(input_std).view(1, 3, 1, 1))
        
        # Normalization for VGG
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize from input space to ImageNet normalization."""
        x = x * self.input_std + self.input_mean
        return (x - self.mean) / self.std
    
    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        generated = self.normalize(generated)
        target = self.normalize(target)
        
        loss = 0.0
        gen_feat = generated
        tgt_feat = target
        
        for layer, weight in zip(self.vgg_layers, self.weights):
            gen_feat = layer(gen_feat)
            tgt_feat = layer(tgt_feat)
            loss += weight * F.l1_loss(gen_feat, tgt_feat)
        
        return loss


class GradientPenalty(nn.Module):
    """Gradient penalty for WGAN-GP (unweighted)."""
    
    def __init__(self, lambda_gp: float = 1.0):
        super().__init__()
        self.lambda_gp = lambda_gp
    
    def forward(
        self,
        discriminator: nn.Module,
        real: torch.Tensor,
        fake: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate gradient penalty and mean gradient norm."""
        batch_size = real.size(0)
        device = real.device
        
        # Random interpolation
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        interpolated = alpha * real + (1 - alpha) * fake
        interpolated.requires_grad_(True)
        
        # Get discriminator output
        d_interpolated = discriminator(interpolated)
        
        if isinstance(d_interpolated, (list, tuple)):
            gp_losses = []
            grad_norms = []
            for d_out in d_interpolated:
                gp_loss, grad_norm = self._gp_from_output(d_out, interpolated)
                gp_losses.append(gp_loss)
                grad_norms.append(grad_norm)
            return torch.stack(gp_losses).mean(), torch.stack(grad_norms).mean()
        
        return self._gp_from_output(d_interpolated, interpolated)
    
    def _gp_from_output(
        self,
        d_out: torch.Tensor,
        interpolated: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gradients = torch.autograd.grad(
            outputs=d_out,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_out),
            create_graph=True,
            retain_graph=True,
        )[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = self.lambda_gp * ((gradient_norm - 1) ** 2).mean()
        return gradient_penalty, gradient_norm.mean()


def _weight_config_enabled(weight_config: Union[float, Dict[str, Any]]) -> bool:
    if isinstance(weight_config, (int, float)):
        return weight_config != 0
    if isinstance(weight_config, dict):
        return True
    return False


# =============================================================================
# Combined Loss Manager
# =============================================================================

class LossManager:
    """
    Manages all loss components with configurable weights.
    
    Config structure:
        gan_mode: 'lsgan'
        cycle_loss_type: 'l1'
        identity_loss_type: 'l1'
        
        weights:
            adversarial: 1.0  # or {type: 'cosine', start_value: 1.0, end_value: 0.5}
            cycle: 10.0
            identity: 5.0
            perceptual: 0.0  # Set > 0 to enable
            gradient_penalty: 10.0  # For WGAN-GP
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device,
        input_mean: Optional[List[float]] = None,
        input_std: Optional[List[float]] = None,
    ):
        self.config = config
        self.device = device
        
        # GAN loss
        gan_mode = config['gan_mode']
        self.gan_loss = GANLoss(gan_mode).to(device)
        
        # Cycle loss
        cycle_type = config['cycle_loss_type']
        self.cycle_loss = CycleLoss(cycle_type).to(device)
        
        # Identity loss
        identity_type = config['identity_loss_type']
        self.identity_loss = IdentityLoss(identity_type).to(device)
        
        # Perceptual loss
        weights_config = config['weights']
        if _weight_config_enabled(weights_config.get('perceptual', 0)):
            self.perceptual_loss = PerceptualLoss(
                input_mean=input_mean,
                input_std=input_std,
            ).to(device)
        else:
            self.perceptual_loss = None
        
        # Gradient penalty (weight applied in compute_discriminator_loss)
        if gan_mode == 'wgan-gp' and _weight_config_enabled(weights_config.get('gradient_penalty', 0)):
            self.gradient_penalty = GradientPenalty().to(device)
        else:
            self.gradient_penalty = None
        
        # Weight schedulers
        self.weight_schedulers = {}
        for loss_name, weight_config in weights_config.items():
            self.weight_schedulers[loss_name] = build_scheduler(weight_config)
    
    def get_weight(self, loss_name: str, step: int, total_steps: int) -> float:
        """Get current weight for a loss component."""
        if loss_name not in self.weight_schedulers:
            return 0.0
        return self.weight_schedulers[loss_name].get_value(step, total_steps)
    
    def _reduce_gan_loss(self, prediction: Any, target_is_real: bool) -> torch.Tensor:
        """Handle single or multiscale discriminator outputs."""
        if isinstance(prediction, (list, tuple)):
            losses = [self.gan_loss(pred, target_is_real) for pred in prediction]
            return torch.stack(losses).mean()
        return self.gan_loss(prediction, target_is_real)
    
    def compute_generator_loss(
        self,
        fake_output: torch.Tensor,
        cycle_reconstructed: torch.Tensor,
        original: torch.Tensor,
        identity_output: Optional[torch.Tensor],
        target: Optional[torch.Tensor],
        step: int,
        total_steps: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total generator loss.
        
        Returns total loss and dict of individual loss values.
        """
        losses = {}
        total_loss = 0.0
        
        # Adversarial loss (generator wants discriminator to think fake is real)
        adv_weight = self.get_weight('adversarial', step, total_steps)
        if adv_weight > 0:
            if self.gan_loss.gan_mode == 'hinge':
                if isinstance(fake_output, (list, tuple)):
                    adv_loss = -torch.stack([pred.mean() for pred in fake_output]).mean()
                else:
                    adv_loss = -fake_output.mean()
            else:
                adv_loss = self._reduce_gan_loss(fake_output, target_is_real=True)
            losses['adversarial'] = adv_loss.item()
            total_loss += adv_weight * adv_loss
        
        # Cycle consistency loss
        cycle_weight = self.get_weight('cycle', step, total_steps)
        if cycle_weight > 0:
            cyc_loss = self.cycle_loss(cycle_reconstructed, original)
            losses['cycle'] = cyc_loss.item()
            total_loss += cycle_weight * cyc_loss
        
        # Identity loss
        identity_weight = self.get_weight('identity', step, total_steps)
        if identity_weight > 0 and identity_output is not None and target is not None:
            idt_loss = self.identity_loss(identity_output, target)
            losses['identity'] = idt_loss.item()
            total_loss += identity_weight * idt_loss
        
        # Perceptual loss
        perceptual_weight = self.get_weight('perceptual', step, total_steps)
        if perceptual_weight > 0 and self.perceptual_loss is not None and target is not None:
            perc_loss = self.perceptual_loss(cycle_reconstructed, original)
            losses['perceptual'] = perc_loss.item()
            total_loss += perceptual_weight * perc_loss
        
        losses['total_generator'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        return total_loss, losses
    
    def compute_discriminator_loss(
        self,
        real_output: torch.Tensor,
        fake_output: torch.Tensor,
        discriminator: Optional[nn.Module] = None,
        real_images: Optional[torch.Tensor] = None,
        fake_images: Optional[torch.Tensor] = None,
        step: int = 0,
        total_steps: int = 1,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute discriminator loss.
        
        Returns total loss and dict of individual loss values.
        """
        losses = {}
        
        # Real loss
        real_loss = self._reduce_gan_loss(real_output, target_is_real=True)
        losses['d_real'] = real_loss.item()
        
        # Fake loss
        fake_loss = self._reduce_gan_loss(fake_output, target_is_real=False)
        losses['d_fake'] = fake_loss.item()
        
        total_loss = (real_loss + fake_loss) * 0.5
        
        # Gradient penalty (for WGAN-GP)
        gp_weight = self.get_weight('gradient_penalty', step, total_steps)
        if gp_weight > 0 and self.gradient_penalty is not None:
            if discriminator is not None and real_images is not None and fake_images is not None:
                gp_loss, gp_grad_norm = self.gradient_penalty(discriminator, real_images, fake_images)
                losses['gradient_penalty'] = gp_loss.item()
                losses['gradient_penalty_grad_norm'] = gp_grad_norm.item()
                total_loss += gp_weight * gp_loss
        
        losses['total_discriminator'] = total_loss.item()
        return total_loss, losses
