"""
CycleGAN Trainer

Main training loop with configurable components, optimizers, and schedulers.
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    LinearLR,
    SequentialLR,
    LambdaLR,
)
from tqdm import tqdm
import wandb

from src.trainer.models.generator import build_generator
from src.trainer.models.discriminator import build_discriminator
from src.trainer.losses import LossManager, build_scheduler
from src.trainer.evaluation import Evaluator
from src.trainer.dataset import ImageBuffer, create_dataloaders


def get_optimizer(params, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """Create optimizer from config."""
    opt_type = config['type'].lower()
    lr = config['lr']
    weight_decay = config.get('weight_decay', 0.0)
    
    if opt_type == 'adam':
        betas = tuple(config['betas'])
        return Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
    elif opt_type == 'adamw':
        betas = tuple(config['betas'])
        return AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)
    elif opt_type == 'sgd':
        momentum = config.get('momentum', 0.9)
        return SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")


def get_lr_scheduler(
    optimizer,
    config: Dict[str, Any],
    total_steps: int,
    steps_per_epoch: Optional[int] = None,
):
    """Create learning rate scheduler from config."""
    scheduler_type = config['type'].lower()
    
    if scheduler_type == 'constant':
        return None
    elif scheduler_type == 'cosine':
        warmup_steps = config.get('warmup_steps', 0)
        eta_min = config.get('eta_min', 0)
        
        if warmup_steps > 0:
            warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=max(1, total_steps - warmup_steps),
                eta_min=eta_min,
            )
            return SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
        else:
            return CosineAnnealingLR(optimizer, T_max=max(1, total_steps), eta_min=eta_min)
    elif scheduler_type == 'step':
        step_size = config['step_size']
        gamma = config['gamma']
        if steps_per_epoch:
            step_size = max(1, int(step_size * steps_per_epoch))
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'linear':
        # Linear decay from initial lr to 0
        start_epoch = config['start_epoch']
        decay_epochs = config['decay_epochs']
        
        if steps_per_epoch:
            start_step = start_epoch * steps_per_epoch
            decay_steps = decay_epochs * steps_per_epoch
            
            def lambda_rule(step):
                if step < start_step:
                    return 1.0
                progress = (step - start_step) / float(decay_steps + 1)
                return max(0.0, 1.0 - progress)
        else:
            def lambda_rule(epoch):
                if epoch < start_epoch:
                    return 1.0
                return max(0.0, 1.0 - max(0, epoch - start_epoch) / float(decay_epochs + 1))
        
        return LambdaLR(optimizer, lr_lambda=lambda_rule)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class CycleGANTrainer:
    """
    CycleGAN Trainer with configurable architecture and training parameters.
    
    Supports:
    - LEGO-style configurable generators (build block-by-block in config)
    - Configurable discriminators (PatchGAN, Spectral, Swin)
    - Multiple loss functions (LSGAN, WGAN-GP, vanilla)
    - Weight scheduling (constant, cosine, linear)
    - n_critic for discriminator
    - MiFID evaluation (resizes to 256x256 for consistency)
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device,
        output_dir: Path,
    ):
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training params
        self.num_epochs = config['training']['num_epochs']
        self.batch_size = config['training']['batch_size']
        self.accumulation_steps = config['training'].get('accumulation_steps', 1)
        
        # Input normalization (defaults to [-1, 1] -> [0, 1])
        normalize_config = config['training'].get('normalize', {})
        self.normalize_mean = normalize_config.get('mean', [0.5, 0.5, 0.5])
        self.normalize_std = normalize_config.get('std', [0.5, 0.5, 0.5])
        
        # n_critic scheduler (how many D updates per G update)
        n_critic_config = config['training']['n_critic']
        self.n_critic_scheduler = build_scheduler(n_critic_config)
        
        # Evaluation config
        self.eval_config = config['evaluation']
        self.evals_per_epoch = self.eval_config['evals_per_epoch']
        self.num_sample_images = self.eval_config['num_sample_images']
        
        # Build models
        self._build_models()
        
        # Build optimizers and schedulers
        self._build_optimizers()
        
        # Build loss manager
        self.loss_manager = LossManager(
            config['losses'],
            device,
            input_mean=self.normalize_mean,
            input_std=self.normalize_std,
        )
        
        # Image buffers for discriminator training
        buffer_size = config['training']['buffer_size']
        self.fake_A_buffer = ImageBuffer(buffer_size)
        self.fake_B_buffer = ImageBuffer(buffer_size)
        
        # Evaluator
        self.evaluator = Evaluator(
            device,
            self.eval_config,
            input_mean=self.normalize_mean,
            input_std=self.normalize_std,
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.optim_step = 0
        self.best_mifid = float('inf')
        self.steps_per_epoch: Optional[int] = None
        self._logged_generator_trace = False
        self._d_accum_steps = 0
        self._g_accum_steps = 0
        self._resumed = False
    
    def _build_models(self):
        """Build generator and discriminator models."""
        gen_config = self.config['generator']
        disc_config = self.config['discriminator']
        
        # Generators: G_A (A->B), G_B (B->A)
        # For Monet: G_A = photo->monet, G_B = monet->photo
        self.G_A = build_generator(gen_config).to(self.device)
        self.G_B = build_generator(gen_config).to(self.device)
        
        # Discriminators: D_A (classifies A), D_B (classifies B)
        # D_A classifies if photo is real/fake, D_B classifies if monet is real/fake
        self.D_A = build_discriminator(disc_config).to(self.device)
        self.D_B = build_discriminator(disc_config).to(self.device)
        
        # Initialize weights
        self._init_weights(self.G_A)
        self._init_weights(self.G_B)
        self._init_weights(self.D_A)
        self._init_weights(self.D_B)
    
    def _init_weights(self, model: nn.Module):
        """Initialize network weights."""
        def init_fn(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)
        
        model.apply(init_fn)
    
    def _build_optimizers(
        self,
        total_steps: Optional[int] = None,
        steps_per_epoch: Optional[int] = None,
        rebuild_optimizers: bool = True,
    ):
        """Build optimizers and LR schedulers."""
        gen_opt_config = self.config['optimizer']['generator']
        disc_opt_config = self.config['optimizer']['discriminator']
        
        if total_steps is not None:
            self.total_steps = total_steps
        elif not hasattr(self, 'total_steps') or self.total_steps is None:
            self.total_steps = self.num_epochs * 1000  # Placeholder
        
        self.steps_per_epoch = steps_per_epoch
        
        # Generator optimizer
        if rebuild_optimizers or not hasattr(self, 'optimizer_G'):
            gen_params = list(self.G_A.parameters()) + list(self.G_B.parameters())
            self.optimizer_G = get_optimizer(gen_params, gen_opt_config)
        
        # Discriminator optimizer
        if rebuild_optimizers or not hasattr(self, 'optimizer_D'):
            disc_params = list(self.D_A.parameters()) + list(self.D_B.parameters())
            self.optimizer_D = get_optimizer(disc_params, disc_opt_config)
        
        # LR schedulers
        gen_scheduler_config = self.config['optimizer']['generator']['scheduler']
        disc_scheduler_config = self.config['optimizer']['discriminator']['scheduler']
        
        self.scheduler_G = get_lr_scheduler(
            self.optimizer_G, gen_scheduler_config, self.total_steps, steps_per_epoch
        )
        self.scheduler_D = get_lr_scheduler(
            self.optimizer_D, disc_scheduler_config, self.total_steps, steps_per_epoch
        )

    def _reset_optimizer_lrs(self):
        """Reset optimizer learning rates to config (for correct scheduler base_lrs)."""
        gen_lr = self.config['optimizer']['generator']['lr']
        for group in self.optimizer_G.param_groups:
            group['lr'] = gen_lr
            if 'initial_lr' in group:
                group['initial_lr'] = gen_lr

        disc_lr = self.config['optimizer']['discriminator']['lr']
        for group in self.optimizer_D.param_groups:
            group['lr'] = disc_lr
            if 'initial_lr' in group:
                group['initial_lr'] = disc_lr

    def _sync_schedulers_to_step(self):
        """Fast-forward schedulers to the current optimizer step."""
        if self.optim_step <= 0:
            return
        if self.scheduler_G is not None:
            self.scheduler_G.step(self.optim_step)
        if self.scheduler_D is not None:
            self.scheduler_D.step(self.optim_step)
    
    def _set_requires_grad(self, models: List[nn.Module], requires_grad: bool):
        """Set requires_grad for all parameters in models."""
        for model in models:
            for param in model.parameters():
                param.requires_grad = requires_grad

    def _denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.normalize_mean, device=tensor.device).view(1, 3, 1, 1)
        std = torch.tensor(self.normalize_std, device=tensor.device).view(1, 3, 1, 1)
        return tensor * std + mean

    @torch.no_grad()
    def _param_norm(self, model: nn.Module, norm_type: float = 2.0) -> float:
        """Compute L2 parameter norm for a model."""
        total = 0.0
        for param in model.parameters():
            if param.requires_grad:
                param_norm = param.detach().norm(norm_type).item()
                total += param_norm ** 2
        return total ** 0.5

    def _format_generator_trace(self, trace: List[Dict[str, Any]]) -> str:
        """Format generator trace into readable lines."""
        lines = []
        for entry in trace:
            line = (
                f"{entry['stage']}[{entry['idx']}] {entry['block']} "
                f"{entry['in_shape']} -> {entry['out_shape']}"
            )
            skip = entry.get('skip')
            if skip is not None:
                line += (
                    f" | skip=encoder[{skip['from']}] {skip['skip_shape']}"
                )
                if skip['resized']:
                    line += " resized"
                line += f" concat={skip['concat_shape']}"
            lines.append(line)
        return "\n".join(lines)

    def _log_generator_trace(self, real_A: torch.Tensor):
        """Log a one-time generator shape trace for debugging."""
        if self._logged_generator_trace:
            return
        
        self._logged_generator_trace = True
        with torch.no_grad():
            _, trace = self.G_A.trace_forward(real_A[:1].to(self.device))
        
        trace_text = self._format_generator_trace(trace)
        print("\n=== Generator Trace (G_A) ===")
        print(trace_text)
        
        if wandb.run is not None:
            wandb.log({'model/generator_trace': trace_text}, step=self.global_step)
    
    def train_step(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            real_A: Real images from domain A (photos)
            real_B: Real images from domain B (monet)
        
        Returns:
            Dict of loss values
        """
        real_A = real_A.to(self.device)
        real_B = real_B.to(self.device)
        
        losses = {}
        
        # Get n_critic for this step
        n_critic = int(self.n_critic_scheduler.get_value(self.optim_step, self.total_steps))
        n_critic = max(1, n_critic)
        
        # =====================
        # Train Discriminators
        # =====================
        if self._d_accum_steps == 0:
            self.optimizer_D.zero_grad()
        for _ in range(n_critic):
            self._set_requires_grad([self.D_A, self.D_B], True)
            
            # Generate fake images
            with torch.no_grad():
                fake_B = self.G_A(real_A)  # photo -> monet
                fake_A = self.G_B(real_B)  # monet -> photo
            
            # Use image buffer
            fake_A_buffered = self.fake_A_buffer.push_and_pop(fake_A)
            fake_B_buffered = self.fake_B_buffer.push_and_pop(fake_B)
            
            # D_A loss (classifies photos)
            pred_real_A = self.D_A(real_A)
            pred_fake_A = self.D_A(fake_A_buffered.detach())
            loss_D_A, losses_D_A = self.loss_manager.compute_discriminator_loss(
                pred_real_A, pred_fake_A,
                discriminator=self.D_A,
                real_images=real_A,
                fake_images=fake_A_buffered.detach(),
                step=self.optim_step,
                total_steps=self.total_steps,
            )
            
            # D_B loss (classifies monet)
            pred_real_B = self.D_B(real_B)
            pred_fake_B = self.D_B(fake_B_buffered.detach())
            loss_D_B, losses_D_B = self.loss_manager.compute_discriminator_loss(
                pred_real_B, pred_fake_B,
                discriminator=self.D_B,
                real_images=real_B,
                fake_images=fake_B_buffered.detach(),
                step=self.optim_step,
                total_steps=self.total_steps,
            )
            
            # Combined discriminator loss
            loss_D = (loss_D_A + loss_D_B) * 0.5
            (loss_D / self.accumulation_steps).backward()
        
        self._d_accum_steps += 1
        if self._d_accum_steps >= self.accumulation_steps:
            self.optimizer_D.step()
            if self.scheduler_D is not None:
                self.scheduler_D.step()
            self._d_accum_steps = 0
        
        losses['D_A'] = losses_D_A['total_discriminator']
        losses['D_B'] = losses_D_B['total_discriminator']
        losses['D_total'] = (losses['D_A'] + losses['D_B']) * 0.5
        
        if self.loss_manager.gan_loss.gan_mode == 'wgan-gp':
            if 'gradient_penalty' in losses_D_A:
                losses['gp_A'] = losses_D_A['gradient_penalty']
            if 'gradient_penalty' in losses_D_B:
                losses['gp_B'] = losses_D_B['gradient_penalty']
            if 'gradient_penalty_grad_norm' in losses_D_A:
                losses['gp_grad_norm_A'] = losses_D_A['gradient_penalty_grad_norm']
            if 'gradient_penalty_grad_norm' in losses_D_B:
                losses['gp_grad_norm_B'] = losses_D_B['gradient_penalty_grad_norm']
            losses['D_A_param_norm'] = self._param_norm(self.D_A)
            losses['D_B_param_norm'] = self._param_norm(self.D_B)
        
        # ================
        # Train Generators
        # ================
        self._set_requires_grad([self.D_A, self.D_B], False)
        if self._g_accum_steps == 0:
            self.optimizer_G.zero_grad()
        
        # Forward pass
        fake_B = self.G_A(real_A)  # photo -> monet
        fake_A = self.G_B(real_B)  # monet -> photo
        
        # Cycle consistency
        rec_A = self.G_B(fake_B)  # photo -> monet -> photo
        rec_B = self.G_A(fake_A)  # monet -> photo -> monet
        
        # Identity (optional)
        use_identity = self.config['losses']['weights'].get('identity', 0) > 0
        if use_identity:
            idt_A = self.G_B(real_A)  # G_B should be identity for photos
            idt_B = self.G_A(real_B)  # G_A should be identity for monet
        else:
            idt_A = None
            idt_B = None
        
        # Generator adversarial loss
        pred_fake_B = self.D_B(fake_B)
        loss_G_A, losses_G_A = self.loss_manager.compute_generator_loss(
            pred_fake_B, rec_A, real_A,
            idt_B, real_B,
            step=self.optim_step,
            total_steps=self.total_steps,
        )
        
        pred_fake_A = self.D_A(fake_A)
        loss_G_B, losses_G_B = self.loss_manager.compute_generator_loss(
            pred_fake_A, rec_B, real_B,
            idt_A, real_A,
            step=self.optim_step,
            total_steps=self.total_steps,
        )
        
        # Combined generator loss
        loss_G = loss_G_A + loss_G_B
        (loss_G / self.accumulation_steps).backward()
        
        self._g_accum_steps += 1
        if self._g_accum_steps >= self.accumulation_steps:
            self.optimizer_G.step()
            if self.scheduler_G is not None:
                self.scheduler_G.step()
            self._g_accum_steps = 0
            self.optim_step += 1
        
        losses['G_A'] = losses_G_A['total_generator']
        losses['G_B'] = losses_G_B['total_generator']
        losses['G_total'] = (losses['G_A'] + losses['G_B'])
        losses['cycle_A'] = losses_G_A.get('cycle', 0)
        losses['cycle_B'] = losses_G_B.get('cycle', 0)
        losses['adversarial_A'] = losses_G_A.get('adversarial', 0)
        losses['adversarial_B'] = losses_G_B.get('adversarial', 0)
        
        if use_identity:
            losses['identity_A'] = losses_G_A.get('identity', 0)
            losses['identity_B'] = losses_G_B.get('identity', 0)
        
        # Learning rates
        losses['lr_G'] = self.optimizer_G.param_groups[0]['lr']
        losses['lr_D'] = self.optimizer_D.param_groups[0]['lr']
        losses['n_critic'] = n_critic
        losses['accumulation_steps'] = self.accumulation_steps
        
        self.global_step += 1
        
        return losses
    
    @torch.no_grad()
    def generate_samples(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Generate sample images for visualization."""
        self.G_A.eval()
        self.G_B.eval()
        
        real_A = real_A[:self.num_sample_images].to(self.device)
        real_B = real_B[:self.num_sample_images].to(self.device)
        
        fake_B = self.G_A(real_A)
        fake_A = self.G_B(real_B)
        rec_A = self.G_B(fake_B)
        rec_B = self.G_A(fake_A)
        
        self.G_A.train()
        self.G_B.train()
        
        return {
            'real_A': real_A,
            'fake_B': fake_B,
            'rec_A': rec_A,
            'real_B': real_B,
            'fake_A': fake_A,
            'rec_B': rec_B,
        }
    
    def log_samples_to_wandb(
        self,
        samples: Dict[str, torch.Tensor],
        prefix: str = '',
    ):
        """Log sample images to W&B."""
        def tensor_to_images(tensor: torch.Tensor) -> List:
            # Normalize back to [0, 1] for logging
            tensor = self._denormalize(tensor)
            tensor = tensor.clamp(0, 1)
            images = []
            for img in tensor:
                img_np = img.permute(1, 2, 0).cpu().numpy()
                images.append(wandb.Image(img_np))
            return images
        
        log_dict = {}
        for name, tensor in samples.items():
            key = f"{prefix}samples/{name}" if prefix else f"samples/{name}"
            log_dict[key] = tensor_to_images(tensor)
        
        wandb.log(log_dict, step=self.global_step)
    
    def save_checkpoint(self, path: Optional[Path] = None, is_best: bool = False):
        """Save training checkpoint."""
        if path is None:
            path = self.output_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'optim_step': self.optim_step,
            'best_mifid': self.best_mifid,
            'G_A': self.G_A.state_dict(),
            'G_B': self.G_B.state_dict(),
            'D_A': self.D_A.state_dict(),
            'D_B': self.D_B.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict(),
            'config': self.config,
        }
        
        if self.scheduler_G is not None:
            checkpoint['scheduler_G'] = self.scheduler_G.state_dict()
        if self.scheduler_D is not None:
            checkpoint['scheduler_D'] = self.scheduler_D.state_dict()
        
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path: Path):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.optim_step = checkpoint.get('optim_step', self.global_step)
        self.best_mifid = checkpoint.get('best_mifid', float('inf'))
        
        self.G_A.load_state_dict(checkpoint['G_A'])
        self.G_B.load_state_dict(checkpoint['G_B'])
        self.D_A.load_state_dict(checkpoint['D_A'])
        self.D_B.load_state_dict(checkpoint['D_B'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        
        if 'scheduler_G' in checkpoint and self.scheduler_G is not None:
            self.scheduler_G.load_state_dict(checkpoint['scheduler_G'])
        if 'scheduler_D' in checkpoint and self.scheduler_D is not None:
            self.scheduler_D.load_state_dict(checkpoint['scheduler_D'])
        self._resumed = True
    
    def train(self, dataloaders: Dict):
        """
        Main training loop.
        
        Args:
            dataloaders: Dict with 'train', 'test', 'monet', 'photo_train' loaders
        """
        train_loader = dataloaders['train']
        
        # Update total steps (optimizer steps with accumulation)
        steps_per_epoch = len(train_loader)
        optim_steps_per_epoch = math.ceil(steps_per_epoch / self.accumulation_steps)
        self.total_steps = self.num_epochs * optim_steps_per_epoch
        
        # Rebuild schedulers with correct total_steps
        if self._resumed:
            self._reset_optimizer_lrs()
        self._build_optimizers(
            total_steps=self.total_steps,
            steps_per_epoch=optim_steps_per_epoch,
            rebuild_optimizers=False,
        )
        if self._resumed:
            self._sync_schedulers_to_step()
        
        # Compute training features for MiFID (once)
        print("\n=== Computing training features for MiFID ===")
        self.evaluator.compute_training_features(
            dataloaders['monet'],
            max_samples=self.eval_config['num_evaluation_samples']
        )
        
        # Evaluation interval
        eval_interval = max(1, steps_per_epoch // self.evals_per_epoch)
        
        print(f"\n=== Starting Training ===")
        print(f"  Epochs: {self.num_epochs}")
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Optimizer steps per epoch: {optim_steps_per_epoch}")
        print(f"  Accumulation steps: {self.accumulation_steps}")
        print(f"  Total steps: {self.total_steps}")
        print(f"  Evaluations per epoch: {self.evals_per_epoch}")
        
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            self.G_A.train()
            self.G_B.train()
            self.D_A.train()
            self.D_B.train()
            
            epoch_losses = {}
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            for batch_idx, batch in enumerate(pbar):
                real_A = batch['A']
                real_B = batch['B']
                
                self._log_generator_trace(real_A)
                
                losses = self.train_step(real_A, real_B)
                
                # Accumulate losses
                for k, v in losses.items():
                    if k not in epoch_losses:
                        epoch_losses[k] = []
                    epoch_losses[k].append(v)
                
                # Update progress bar
                pbar.set_postfix({
                    'G': f"{losses['G_total']:.3f}",
                    'D': f"{losses['D_total']:.3f}",
                })
                
                # Log to W&B
                wandb.log(losses, step=self.global_step)
                
                # Evaluation
                if (batch_idx + 1) % eval_interval == 0:
                    # Generate samples
                    samples = self.generate_samples(real_A, real_B)
                    self.log_samples_to_wandb(samples, prefix='train_')
                    
                    # Test samples
                    test_batch = next(iter(dataloaders['test']))
                    test_A = test_batch if not isinstance(test_batch, dict) else test_batch.get('A', test_batch)
                    test_samples = self.generate_samples(test_A, real_B)
                    self.log_samples_to_wandb(test_samples, prefix='test_')
            
            # Flush remaining accumulated grads at end of epoch
            if self._d_accum_steps > 0:
                self.optimizer_D.step()
                if self.scheduler_D is not None:
                    self.scheduler_D.step()
                self._d_accum_steps = 0
            
            if self._g_accum_steps > 0:
                self.optimizer_G.step()
                if self.scheduler_G is not None:
                    self.scheduler_G.step()
                self._g_accum_steps = 0
                self.optim_step += 1
            
            # End of epoch
            avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
            print(f"\nEpoch {epoch+1} - G: {avg_losses['G_total']:.4f}, D: {avg_losses['D_total']:.4f}")
            
            # Full evaluation at end of epoch
            print("\n=== Running Evaluation ===")
            metrics = self.evaluator.evaluate(
                self.G_A,
                dataloaders['photo_train'],
                dataloaders['monet'],
                max_samples=self.eval_config['num_evaluation_samples'],
            )
            
            print(f"  FID: {metrics['fid']:.2f}")
            print(f"  MiFID: {metrics['mifid']:.2f}")
            print(f"  Memorization Distance: {metrics['memorization_distance']:.4f}")
            
            wandb.log({
                'eval/fid': metrics['fid'],
                'eval/mifid': metrics['mifid'],
                'eval/memorization_distance': metrics['memorization_distance'],
                'epoch': epoch + 1,
            }, step=self.global_step)
            
            # Save best checkpoint (best_model.pt only)
            is_best = metrics['mifid'] < self.best_mifid
            if is_best:
                self.best_mifid = metrics['mifid']
                print(f"  New best MiFID: {self.best_mifid:.2f}")
                best_path = self.output_dir / "best_model.pt"
                self.save_checkpoint(path=best_path, is_best=False)
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config['training']['save_interval'] == 0:
                self.save_checkpoint()
        
        print("\n=== Training Complete ===")
        print(f"  Best MiFID: {self.best_mifid:.2f}")
