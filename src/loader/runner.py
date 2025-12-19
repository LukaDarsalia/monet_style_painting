"""
Loader Pipeline Runner

Usage:
    python -m src.loader.runner --config config/loader/experiment_256x256.yaml
    python -m src.loader.runner --config config/loader/experiment_64x64.yaml --force-download
    python -m src.loader.runner --config config/loader/experiment_256x256.yaml --test
"""

from pathlib import Path
from typing import List

import click
import wandb

from src.loader.loader import Loader
from src.utils.config import load_config
from src.utils.s3 import get_s3_loader, generate_folder_name


@click.command()
@click.option("--config", "-c", required=True, type=str, help="Path to configuration YAML file")
@click.option("--force-download", is_flag=True, default=False, help="Force re-download dataset")
@click.option("--test", is_flag=True, default=False, help="Add TEST tag to W&B run")
def main(config: str, force_download: bool, test: bool) -> None:
    """Run the Monet GAN data loading pipeline."""
    
    print("\n=== Pipeline Configuration ===")
    print(f"  config: {config}")
    print(f"  force_download: {force_download}")
    print(f"  test: {test}")

    # Load and validate config
    cfg = load_config(config)
    
    project = cfg['wandb']['project_name']
    run_name = cfg['run_name']
    description = cfg['description']
    bucket = cfg['s3']['bucket_name']
    
    # Build tags
    tags: List[str] = cfg.get('wandb', {}).get('tags', ['pipeline', 'loader'])
    if test:
        tags.append('TEST')

    print(f"\n=== Starting W&B Run ===")
    print(f"  Project: {project}")
    print(f"  Run Name: {run_name}")
    print(f"  Tags: {tags}")

    with wandb.init(
        project=project,
        name=run_name,
        job_type="load-data",
        tags=tags,
        notes=description,
        config=cfg,
        save_code=True,
    ) as run:
        # Create artifact
        artifact = wandb.Artifact(
            name="monet-dataset",
            type="dataset",
            description=description
        )

        # Setup output folder
        output_folder_dir = Path('artifacts') / 'loader' / generate_folder_name()
        output_folder_dir.mkdir(parents=True, exist_ok=True)
        artifact.metadata.update({'output_dir': str(output_folder_dir)})

        # Run loader
        loader = Loader(
            artifact=artifact,
            output_dir=str(output_folder_dir),
            config_path=config,
            force_download=force_download
        )

        print("\n=== Loader Configuration ===")
        loader.print_config()

        loader.run_loading()

        # Upload to S3
        print("\n=== Uploading to S3 ===")
        s3_loader = get_s3_loader(bucket)
        s3_key = s3_loader.upload_as_tarball(output_folder_dir)
        
        artifact.add_reference(f"s3://{bucket}/{s3_key}")
        run.log_artifact(artifact)
        
        print("\n=== Pipeline Completed Successfully ===")


if __name__ == '__main__':
    main()
