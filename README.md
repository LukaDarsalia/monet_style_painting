# Monet GAN Pipeline

Three-stage pipeline for Monet-style image translation using a CycleGAN paper-style trainer.

## Overview
- Loader: downloads Kaggle dataset, resizes images, splits photos into train/test, logs samples to W&B.
- Augmenter: copies or augments train images, logs augmented samples to W&B.
- Trainer: CycleGAN paper objective (LSGAN + cycle L1 + optional identity), PatchGAN discriminator.

Artifacts flow through W&B and are stored as tarballs in S3 between stages.

## Quick Start
1) Install deps:
```
pip install -r requirements.txt
```

2) Credentials:
- Kaggle CLI: `~/.kaggle/kaggle.json`
- AWS + W&B: `.env` with
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`
  - `WANDB_API_KEY`

3) Run the pipeline:
```
python -m src.loader.runner --config config/loader/experiment_256x256.yaml
python -m src.augmenter.runner --config config/augmenter/experiment_light_style.yaml
python -m src.trainer.runner --config config/trainer/no_experiment.yaml
```

Useful flags:
- Loader: `--force-download`
- Augmenter/Trainer: `--artifact-version`
- Trainer: `--resume path/to/checkpoint.pt`
- Any stage: `--test` (adds TEST tag to W&B run)

## Data Layout
The loader/augmenter produce:
```
data_dir/
  train/
    monet/
    photo/
  test/
    photo/
```
Trainer uses unpaired sampling between `train/monet` and `train/photo`.

## Configuration
Configs support `_base_` inheritance (see `src/utils/config.py`).
Base configs:
- `config/loader/base.yaml`
- `config/augmenter/base.yaml`
- `config/trainer/base.yaml` (CycleGAN paper architecture)

Each experiment config sets `run_name` and `description` and overrides:
- `training`: epochs, batch size, logging intervals, normalization
- `optimizer`: `lr`, `betas`
- `scheduler`: linear decay with `wait_epochs`, `decay_epochs`, `min_lr`
- `losses`: `lambda_cycle`, `lambda_identity`

## Trainer Details
- Losses: LSGAN (MSE), cycle L1, optional identity L1.
- Discriminator: 70x70 PatchGAN.
- Generator: LEGO-style blocks defined in config.
- LR schedule: constant for `wait_epochs`, then linear decay to `min_lr`.
- Logging: W&B metrics + training-batch image samples.

Generator blocks (`src/trainer/models/generator/generator.py`):
`conv`, `conv_transpose`, `upsample_conv`, `residual`, `double_conv`, `max_pool`,
`avg_pool`, `dropout` plus optional `skip_connections`.
