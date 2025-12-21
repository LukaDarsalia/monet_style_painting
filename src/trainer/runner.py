"""
Trainer Pipeline Runner

Usage:
    python -m src.trainer.runner --config config/trainer/experiment_cyclegan.yaml
    python -m src.trainer.runner --config config/trainer/experiment_cyclegan.yaml --resume checkpoints/checkpoint.pt
    python -m src.trainer.runner --config config/trainer/experiment_cyclegan.yaml --test
"""

from pathlib import Path
from typing import List

import click
import torch
import wandb

from src.trainer.trainer import CycleGANTrainer
from src.trainer.dataset import create_dataloaders
from src.utils.config import load_config
from src.utils.s3 import get_s3_loader, generate_folder_name


@click.command()
@click.option("--config", "-c", required=True, type=str, help="Path to configuration YAML file")
@click.option("--artifact-version", "-v", default=None, type=str, help="Override input artifact version")
@click.option("--resume", "-r", default=None, type=str, help="Path to checkpoint to resume from")
@click.option("--test", is_flag=True, default=False, help="Add TEST tag to W&B run")
def main(config: str, artifact_version: str, resume: str, test: bool) -> None:
    """Run the Monet GAN training pipeline."""
    
    print("\n=== Pipeline Configuration ===")
    print(f"  config: {config}")
    print(f"  artifact_version: {artifact_version}")
    print(f"  resume: {resume}")
    print(f"  test: {test}")

    # Load and validate config
    cfg = load_config(config)
    
    project = cfg['wandb']['project_name']
    run_name = cfg['run_name']
    description = cfg['description']
    bucket = cfg['s3']['bucket_name']
    
    # Input artifact settings
    input_artifact_name = cfg['input_artifact']['name']
    input_artifact_version = artifact_version or cfg['input_artifact'].get('version', 'latest')
    
    # Build tags
    tags: List[str] = cfg.get('wandb', {}).get('tags', ['pipeline', 'trainer'])
    if test:
        tags.append('TEST')

    print(f"\n=== Starting W&B Run ===")
    print(f"  Project: {project}")
    print(f"  Run Name: {run_name}")
    print(f"  Tags: {tags}")
    print(f"  Input Artifact: {input_artifact_name}:{input_artifact_version}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    with wandb.init(
        project=project,
        name=run_name,
        job_type="train",
        tags=tags,
        notes=description,
        config=cfg,
        save_code=True,
    ) as run:
        # Download input artifact
        print("\n=== Step 1: Downloading Input Artifact ===")
        input_artifact = run.use_artifact(f"{input_artifact_name}:{input_artifact_version}")
        
        # Get S3 reference from artifact
        s3_loader = get_s3_loader(bucket)
        
        # Find S3 reference in artifact
        s3_ref = None
        for ref in input_artifact.manifest.entries.values():
            if hasattr(ref, 'ref') and ref.ref and ref.ref.startswith('s3://'):
                s3_ref = ref.ref
                break
        
        if not s3_ref:
            raise RuntimeError(f"No S3 reference found in artifact {input_artifact_name}:{input_artifact_version}")
        
        # Parse S3 reference
        s3_path = s3_ref.replace(f"s3://{bucket}/", "")
        print(f"  S3 Path: {s3_path}")
        
        # Setup input directory
        input_folder_dir = Path('artifacts') / 'trainer' / 'input' / generate_folder_name()
        input_folder_dir.mkdir(parents=True, exist_ok=True)
        
        # Download and extract from S3
        print(f"  Downloading and extracting to {input_folder_dir}...")
        s3_loader.download_and_extract_tarball(s3_path, input_folder_dir)
        
        # Find the actual data directory
        data_dir = _find_data_dir(input_folder_dir)
        if data_dir is None:
            raise RuntimeError(f"Could not find data directory in {input_folder_dir}")
        
        print(f"  Data directory: {data_dir}")
        
        # Create DataLoaders
        print("\n=== Step 2: Creating DataLoaders ===")
        dataloaders = create_dataloaders(cfg['training'], data_dir)
        print(f"  Train batches: {len(dataloaders['train'])}")
        print(f"  Test batches: {len(dataloaders['test'])}")
        print(f"  Monet images: {len(dataloaders['monet'].dataset)}")
        print(f"  Photo train images: {len(dataloaders['photo_train'].dataset)}")
        
        # Setup output directory
        output_folder_dir = Path('artifacts') / 'trainer' / 'output' / generate_folder_name()
        output_folder_dir.mkdir(parents=True, exist_ok=True)
        
        # Create trainer
        print("\n=== Step 3: Initializing Trainer ===")
        trainer = CycleGANTrainer(cfg, device, output_folder_dir)
        
        # Print model info
        print(f"\n  Generator G_A parameters: {_count_parameters(trainer.G_A):,}")
        print(f"  Generator G_B parameters: {_count_parameters(trainer.G_B):,}")
        print(f"  Discriminator D_A parameters: {_count_parameters(trainer.D_A):,}")
        print(f"  Discriminator D_B parameters: {_count_parameters(trainer.D_B):,}")
        
        # Resume from checkpoint if specified
        if resume:
            print(f"\n  Resuming from checkpoint: {resume}")
            trainer.load_checkpoint(Path(resume))
        
        # Train
        print("\n=== Step 4: Training ===")
        trainer.train(dataloaders)
        
        # Create output artifact
        print("\n=== Step 5: Saving Model Artifact ===")
        output_artifact = wandb.Artifact(
            name="monet-cyclegan-model",
            type="model",
            description=description,
            metadata={
                'input_artifact': f"{input_artifact_name}:{input_artifact_version}",
                'best_mifid': trainer.best_mifid,
                'num_epochs': trainer.num_epochs,
                'config': cfg,
            }
        )
        
        # Add best model checkpoint
        best_model_path = output_folder_dir / "best_model.pt"
        if best_model_path.exists():
            output_artifact.add_file(str(best_model_path), name="best_model.pt")
        
        # Upload to S3
        print("\n=== Step 6: Uploading to S3 ===")
        s3_key = s3_loader.upload_as_tarball(output_folder_dir)
        
        output_artifact.add_reference(f"s3://{bucket}/{s3_key}")
        run.log_artifact(output_artifact)
        
        print("\n=== Pipeline Completed Successfully ===")
        print(f"  Best MiFID: {trainer.best_mifid:.2f}")


def _find_data_dir(root_dir: Path) -> Path:
    """Find the data directory containing train/monet or train/photo structure."""
    if (root_dir / 'train' / 'monet').exists() or (root_dir / 'train' / 'photo').exists():
        return root_dir
    
    for subdir in root_dir.iterdir():
        if subdir.is_dir():
            if (subdir / 'train' / 'monet').exists() or (subdir / 'train' / 'photo').exists():
                return subdir
            
            for subsubdir in subdir.iterdir():
                if subsubdir.is_dir():
                    if (subsubdir / 'train' / 'monet').exists() or (subsubdir / 'train' / 'photo').exists():
                        return subsubdir
    
    return None


def _count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    main()
