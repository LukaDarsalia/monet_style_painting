"""
Augmenter Pipeline Runner

Usage:
    python -m src.augmenter.runner --config config/augmenter/experiment_standard.yaml
    python -m src.augmenter.runner --config config/augmenter/experiment_standard.yaml --artifact-version v1
    python -m src.augmenter.runner --config config/augmenter/experiment_standard.yaml --test
"""

from pathlib import Path
from typing import List

import click
import wandb

from src.augmenter.augmenter import Augmenter
from src.utils.config import load_config
from src.utils.s3 import get_s3_loader, generate_folder_name


@click.command()
@click.option("--config", "-c", required=True, type=str, help="Path to configuration YAML file")
@click.option("--artifact-version", "-v", default=None, type=str, help="Override input artifact version")
@click.option("--test", is_flag=True, default=False, help="Add TEST tag to W&B run")
def main(config: str, artifact_version: str, test: bool) -> None:
    """Run the Monet GAN data augmentation pipeline."""
    
    print("\n=== Pipeline Configuration ===")
    print(f"  config: {config}")
    print(f"  artifact_version: {artifact_version}")
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
    tags: List[str] = cfg.get('wandb', {}).get('tags', ['pipeline', 'augmenter'])
    if test:
        tags.append('TEST')

    print(f"\n=== Starting W&B Run ===")
    print(f"  Project: {project}")
    print(f"  Run Name: {run_name}")
    print(f"  Tags: {tags}")
    print(f"  Input Artifact: {input_artifact_name}:{input_artifact_version}")

    with wandb.init(
        project=project,
        name=run_name,
        job_type="augment-data",
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
        s3_refs = [ref for ref in input_artifact.manifest.entries.keys() if ref.startswith('s3://')]
        
        if not s3_refs:
            # Try to get from artifact references
            refs = list(input_artifact.manifest.entries.values())
            s3_ref = None
            for ref in refs:
                if hasattr(ref, 'ref') and ref.ref and ref.ref.startswith('s3://'):
                    s3_ref = ref.ref
                    break
            
            if not s3_ref:
                raise RuntimeError(f"No S3 reference found in artifact {input_artifact_name}:{input_artifact_version}")
        else:
            s3_ref = s3_refs[0]
        
        # Parse S3 reference
        s3_path = s3_ref.replace(f"s3://{bucket}/", "")
        print(f"  S3 Path: {s3_path}")
        
        # Setup input directory
        input_folder_dir = Path('artifacts') / 'augmenter' / 'input' / generate_folder_name()
        input_folder_dir.mkdir(parents=True, exist_ok=True)
        
        # Download and extract from S3
        print(f"  Downloading and extracting to {input_folder_dir}...")
        s3_loader.download_and_extract_tarball(s3_path, input_folder_dir)
        
        # Find the actual data directory (may be nested)
        # Look for train/monet or train/photo structure
        data_dir = _find_data_dir(input_folder_dir)
        if data_dir is None:
            raise RuntimeError(f"Could not find data directory with train/monet or train/photo structure in {input_folder_dir}")
        
        print(f"  Data directory: {data_dir}")
        
        # Create output artifact
        output_artifact = wandb.Artifact(
            name="monet-dataset-augmented",
            type="dataset",
            description=description
        )

        # Setup output folder
        output_folder_dir = Path('artifacts') / 'augmenter' / 'output' / generate_folder_name()
        output_folder_dir.mkdir(parents=True, exist_ok=True)
        output_artifact.metadata.update({
            'output_dir': str(output_folder_dir),
            'input_artifact': f"{input_artifact_name}:{input_artifact_version}"
        })

        # Run augmenter
        augmenter = Augmenter(
            artifact=output_artifact,
            input_dir=str(data_dir),
            output_dir=str(output_folder_dir),
            config_path=config,
        )

        print("\n=== Augmenter Configuration ===")
        augmenter.print_config()

        print("\n=== Step 2: Running Augmentation ===")
        results = augmenter.run_augmentation()
        
        augmenter.print_summary(results)

        # Upload to S3
        print("\n=== Step 4: Uploading to S3 ===")
        s3_key = s3_loader.upload_as_tarball(output_folder_dir)
        
        output_artifact.add_reference(f"s3://{bucket}/{s3_key}")
        run.log_artifact(output_artifact)
        
        print("\n=== Pipeline Completed Successfully ===")


def _find_data_dir(root_dir: Path) -> Path:
    """
    Find the data directory containing train/monet or train/photo structure.
    
    The extracted tarball may have nested directories, so we need to search.
    """
    # Check if root_dir itself has the structure
    if (root_dir / 'train' / 'monet').exists() or (root_dir / 'train' / 'photo').exists():
        return root_dir
    
    # Search one level deep
    for subdir in root_dir.iterdir():
        if subdir.is_dir():
            if (subdir / 'train' / 'monet').exists() or (subdir / 'train' / 'photo').exists():
                return subdir
            
            # Search two levels deep
            for subsubdir in subdir.iterdir():
                if subsubdir.is_dir():
                    if (subsubdir / 'train' / 'monet').exists() or (subsubdir / 'train' / 'photo').exists():
                        return subsubdir
    
    return None


if __name__ == '__main__':
    main()
