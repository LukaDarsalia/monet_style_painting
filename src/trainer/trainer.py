"""
CycleGAN Trainer (paper-style).

Implements the original CycleGAN objective with:
- LSGAN adversarial loss
- L1 cycle-consistency loss
- Optional identity loss
- Image buffer for discriminator training
- Linear LR decay with a wait period
"""

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import wandb

from src.trainer.models.generator import build_generator
from src.trainer.models.discriminator import build_discriminator
from src.trainer.dataset import ImageBuffer
from src.trainer.logger import WandbLogger


class CycleGANTrainer:
    """Minimal CycleGAN trainer aligned with the original paper."""

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

        training = config['training']
        self.num_epochs = training['num_epochs']
        self.batch_size = training['batch_size']
        self.buffer_size = training.get('buffer_size', 50)
        self.save_interval = training.get('save_interval', 1)
        self.log_interval = training.get('log_interval', 1)
        self.image_log_interval = training.get('image_log_interval', 0)
        self.num_sample_images = training.get('num_sample_images', 4)

        normalize = training.get('normalize', {})
        self.normalize_mean = normalize.get('mean', [0.5, 0.5, 0.5])
        self.normalize_std = normalize.get('std', [0.5, 0.5, 0.5])

        losses = config['losses']
        self.lambda_cycle = losses['lambda_cycle']
        self.lambda_identity = losses.get('lambda_identity', 0.0)

        optim = config['optimizer']
        self.base_lr = optim['lr']
        self.betas = tuple(optim['betas'])

        sched = config.get('scheduler', {})
        self.wait_epochs = sched.get('wait_epochs', 0)
        self.decay_epochs = sched.get('decay_epochs', 0)
        self.min_lr = sched.get('min_lr', 0.0)

        # Build models
        self._build_models()

        # Optimizers
        gen_params = list(self.G_A.parameters()) + list(self.G_B.parameters())
        disc_params = list(self.D_A.parameters()) + list(self.D_B.parameters())
        self.optimizer_G = Adam(gen_params, lr=self.base_lr, betas=self.betas)
        self.optimizer_D = Adam(disc_params, lr=self.base_lr, betas=self.betas)

        # Losses
        self.criterion_gan = nn.MSELoss().to(self.device)
        self.criterion_cycle = nn.L1Loss().to(self.device)
        self.criterion_identity = nn.L1Loss().to(self.device)

        # Image buffers
        self.fake_A_buffer = ImageBuffer(self.buffer_size)
        self.fake_B_buffer = ImageBuffer(self.buffer_size)

        # Logger
        self.logger = WandbLogger(wandb.run, log_interval=self.log_interval)

        # Training state
        self.current_epoch = 0
        self.global_step = 0

    def _build_models(self) -> None:
        gen_config = self.config['generator']
        disc_config = self.config['discriminator']

        self.G_A = build_generator(gen_config).to(self.device)
        self.G_B = build_generator(gen_config).to(self.device)
        self.D_A = build_discriminator(disc_config).to(self.device)
        self.D_B = build_discriminator(disc_config).to(self.device)

        self._init_weights(self.G_A)
        self._init_weights(self.G_B)
        self._init_weights(self.D_A)
        self._init_weights(self.D_B)

    def _init_weights(self, model: nn.Module) -> None:
        def init_fn(m: nn.Module) -> None:
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

        model.apply(init_fn)

    def _set_requires_grad(self, models, requires_grad: bool) -> None:
        for model in models:
            for param in model.parameters():
                param.requires_grad = requires_grad

    def _denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.normalize_mean, device=tensor.device).view(1, 3, 1, 1)
        std = torch.tensor(self.normalize_std, device=tensor.device).view(1, 3, 1, 1)
        return tensor * std + mean

    def _compute_lr(self, epoch: int) -> float:
        if self.decay_epochs <= 0:
            return self.base_lr
        if epoch < self.wait_epochs:
            return self.base_lr
        progress = min(epoch - self.wait_epochs, self.decay_epochs) / float(self.decay_epochs)
        return self.base_lr - (self.base_lr - self.min_lr) * progress

    def _update_lr(self, epoch: int) -> None:
        lr = self._compute_lr(epoch)
        for group in self.optimizer_G.param_groups:
            group['lr'] = lr
        for group in self.optimizer_D.param_groups:
            group['lr'] = lr

    def _gan_loss(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target_value = 1.0 if target_is_real else 0.0
        target = torch.full_like(prediction, target_value)
        return self.criterion_gan(prediction, target)

    def _log_images(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
        fake_A: torch.Tensor,
        fake_B: torch.Tensor,
        rec_A: torch.Tensor,
        rec_B: torch.Tensor,
        step: int,
    ) -> None:
        if self.image_log_interval <= 0:
            return
        if step % self.image_log_interval != 0:
            return

        def to_images(tensor: torch.Tensor):
            tensor = self._denormalize(tensor)
            tensor = tensor.clamp(0, 1)
            images = []
            for img in tensor[: self.num_sample_images]:
                img_np = img.detach().permute(1, 2, 0).cpu().numpy()
                images.append(wandb.Image(img_np))
            return images

        self.logger.log_images({
            'samples/real_A': to_images(real_A),
            'samples/real_B': to_images(real_B),
            'samples/fake_B': to_images(fake_B),
            'samples/fake_A': to_images(fake_A),
            'samples/rec_A': to_images(rec_A),
            'samples/rec_B': to_images(rec_B),
        }, step=step)

    def save_checkpoint(self, path: Optional[Path] = None) -> None:
        if path is None:
            path = self.output_dir / f"checkpoint_epoch_{self.current_epoch}.pt"

        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'G_A': self.G_A.state_dict(),
            'G_B': self.G_B.state_dict(),
            'D_A': self.D_A.state_dict(),
            'D_B': self.D_B.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict(),
            'config': self.config,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint.get('global_step', 0)

        self.G_A.load_state_dict(checkpoint['G_A'])
        self.G_B.load_state_dict(checkpoint['G_B'])
        self.D_A.load_state_dict(checkpoint['D_A'])
        self.D_B.load_state_dict(checkpoint['D_B'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])

    def train(self, dataloaders: Dict[str, torch.utils.data.DataLoader]) -> None:
        train_loader = dataloaders['train']

        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            self._update_lr(epoch)

            self.G_A.train()
            self.G_B.train()
            self.D_A.train()
            self.D_B.train()

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            for batch in pbar:
                real_A = batch['A'].to(self.device)
                real_B = batch['B'].to(self.device)

                # =====================
                # Train Discriminators
                # =====================
                self._set_requires_grad([self.D_A, self.D_B], True)
                self.optimizer_D.zero_grad()

                with torch.no_grad():
                    fake_B = self.G_A(real_A)
                    fake_A = self.G_B(real_B)

                fake_A_buffered = self.fake_A_buffer.push_and_pop(fake_A)
                fake_B_buffered = self.fake_B_buffer.push_and_pop(fake_B)

                pred_real_A = self.D_A(real_A)
                pred_fake_A = self.D_A(fake_A_buffered.detach())
                loss_D_A = 0.5 * (
                    self._gan_loss(pred_real_A, True) +
                    self._gan_loss(pred_fake_A, False)
                )

                pred_real_B = self.D_B(real_B)
                pred_fake_B = self.D_B(fake_B_buffered.detach())
                loss_D_B = 0.5 * (
                    self._gan_loss(pred_real_B, True) +
                    self._gan_loss(pred_fake_B, False)
                )

                loss_D = 0.5 * (loss_D_A + loss_D_B)
                loss_D.backward()
                self.optimizer_D.step()

                # =================
                # Train Generators
                # =================
                self._set_requires_grad([self.D_A, self.D_B], False)
                self.optimizer_G.zero_grad()

                fake_B = self.G_A(real_A)
                fake_A = self.G_B(real_B)

                pred_fake_B = self.D_B(fake_B)
                pred_fake_A = self.D_A(fake_A)

                loss_G_A = self._gan_loss(pred_fake_B, True)
                loss_G_B = self._gan_loss(pred_fake_A, True)

                rec_A = self.G_B(fake_B)
                rec_B = self.G_A(fake_A)
                loss_cycle_A = self.criterion_cycle(rec_A, real_A)
                loss_cycle_B = self.criterion_cycle(rec_B, real_B)
                loss_cycle = loss_cycle_A + loss_cycle_B

                loss_identity = torch.tensor(0.0, device=self.device)
                if self.lambda_identity > 0:
                    idt_A = self.G_B(real_A)
                    idt_B = self.G_A(real_B)
                    loss_idt_A = self.criterion_identity(idt_A, real_A)
                    loss_idt_B = self.criterion_identity(idt_B, real_B)
                    loss_identity = loss_idt_A + loss_idt_B

                loss_G = loss_G_A + loss_G_B + self.lambda_cycle * loss_cycle + self.lambda_identity * loss_identity
                loss_G.backward()
                self.optimizer_G.step()

                # Logging
                self.global_step += 1
                metrics = {
                    'loss/G_total': loss_G.item(),
                    'loss/G_adv_A': loss_G_A.item(),
                    'loss/G_adv_B': loss_G_B.item(),
                    'loss/cycle_A': loss_cycle_A.item(),
                    'loss/cycle_B': loss_cycle_B.item(),
                    'loss/identity': loss_identity.item(),
                    'loss/D_A': loss_D_A.item(),
                    'loss/D_B': loss_D_B.item(),
                    'loss/D_total': loss_D.item(),
                    'lr_G': self.optimizer_G.param_groups[0]['lr'],
                    'lr_D': self.optimizer_D.param_groups[0]['lr'],
                    'epoch': epoch + 1,
                }
                self.logger.log_metrics(metrics, step=self.global_step)

                pbar.set_postfix({
                    'G': f"{loss_G.item():.3f}",
                    'D': f"{loss_D.item():.3f}",
                })

                self._log_images(real_A, real_B, fake_A, fake_B, rec_A, rec_B, step=self.global_step)

            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint()

        print("\n=== Training Complete ===")
