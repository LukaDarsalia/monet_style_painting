"""
W&B logging helper for CycleGAN training.
"""

from typing import Any, Dict, Optional

import wandb


class WandbLogger:
    """Simple W&B logger with step-aware logging."""

    def __init__(self, run: Optional["wandb.sdk.wandb_run.Run"], log_interval: int = 1):
        self.run = run
        self.log_interval = max(1, log_interval)

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        if self.run is None:
            return
        if step % self.log_interval != 0:
            return
        wandb.log(metrics, step=step)

    def log_images(self, images: Dict[str, Any], step: int) -> None:
        if self.run is None:
            return
        wandb.log(images, step=step)
