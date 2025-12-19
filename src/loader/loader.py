"""
Monet GAN Data Loader

Handles downloading from Kaggle, resizing images, splitting photos into
train/test sets, and logging samples to W&B.
"""

import random
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import wandb
import yaml
from PIL import Image
from tqdm import tqdm


class Loader:
    """Loads and processes the Monet GAN dataset."""

    def __init__(
        self,
        artifact: wandb.Artifact,
        output_dir: str,
        config_path: str,
        force_download: bool
    ):
        self.artifact = artifact
        self.output_dir = Path(output_dir)
        self.config_path = Path(config_path)
        self.force_download = force_download
        self.config = self._load_config()
        
        # Extract config values
        self.competition_name = self.config['dataset']['competition_name']
        self.raw_data_dir = Path(self.config['dataset']['raw_data_dir'])
        self.target_size = self.config['dataset']['image']['target_size']
        self.train_ratio = self.config['split']['train_ratio']
        self.random_seed = self.config['split']['random_seed']
        
        random.seed(self.random_seed)
        self._log_files_to_artifact()

    def _load_config(self) -> Dict[str, Any]:
        """Load config with inheritance support."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f) or {}

        if '_base_' in config:
            base_path = self.config_path.parent / config['_base_']
            with open(base_path, 'r') as f:
                base_config = yaml.safe_load(f) or {}
            config = self._deep_merge(base_config, {k: v for k, v in config.items() if k != '_base_'})

        return config

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dicts."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _log_files_to_artifact(self) -> None:
        """Log merged config and source files to artifact."""
        # Save merged config
        merged_config_path = Path(tempfile.gettempdir()) / "merged_config.yaml"
        with open(merged_config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        self.artifact.add_file(str(merged_config_path), name="config.yaml")

        # Log source files
        loader_dir = Path(__file__).parent
        for file_path in loader_dir.glob("*.py"):
            self.artifact.add_file(str(file_path), name=f"loader_code/{file_path.name}")

    def print_config(self) -> None:
        """Print configuration summary."""
        print(f"  Competition: {self.competition_name}")
        print(f"  Target Size: {self.target_size}x{self.target_size}")
        print(f"  Train Ratio: {self.train_ratio}")
        print(f"  Test Ratio: {1 - self.train_ratio}")
        print(f"  Random Seed: {self.random_seed}")
        print(f"  Output Dir: {self.output_dir}")

    def _download_dataset(self) -> Path:
        """Download dataset from Kaggle and extract it."""
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        zip_file = self.raw_data_dir / f"{self.competition_name}.zip"
        extract_dir = self.raw_data_dir / "extracted"

        # Check if already exists
        if not self.force_download and extract_dir.exists():
            if (extract_dir / 'monet_jpg').exists() and (extract_dir / 'photo_jpg').exists():
                print(f"  Dataset already exists at {extract_dir}")
                return extract_dir

        # Check credentials
        kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
        if not kaggle_json.exists():
            raise RuntimeError("Kaggle credentials not found at ~/.kaggle/kaggle.json")

        # Download
        if not zip_file.exists() or self.force_download:
            print(f"  Downloading from Kaggle: {self.competition_name}")
            try:
                subprocess.run(
                    ["kaggle", "competitions", "download", "-c", self.competition_name, "-p", str(self.raw_data_dir)],
                    capture_output=True, text=True, check=True
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to download: {e.stderr}")
            except FileNotFoundError:
                raise RuntimeError("Kaggle CLI not found. Install with: pip install kaggle")

        # Extract
        print(f"  Extracting {zip_file}...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_file, 'r') as zf:
            for member in tqdm(zf.namelist(), desc="  Extracting"):
                zf.extract(member, extract_dir)

        return extract_dir

    def _get_image_files(self, directory: Path) -> List[Path]:
        """Get sorted list of image files in directory."""
        extensions = {'.jpg', '.jpeg', '.png'}
        return sorted([f for f in directory.iterdir() if f.suffix.lower() in extensions])

    def _resize_image(self, image_path: Path, output_path: Path) -> None:
        """Resize image to target size and save."""
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            if img.size != (self.target_size, self.target_size):
                img = img.resize((self.target_size, self.target_size), Image.Resampling.LANCZOS)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path, 'JPEG', quality=95)

    def _process_images(self, raw_dir: Path) -> Dict[str, Dict[str, int]]:
        """Process images: resize monet (all to train), split photos (train/test)."""
        results = {'monet': {'train': 0}, 'photo': {'train': 0, 'test': 0}}

        # Monet - all to train
        print("\n  Processing Monet paintings (all to train)...")
        monet_files = self._get_image_files(raw_dir / 'monet_jpg')
        monet_train_dir = self.output_dir / 'train' / 'monet'
        for f in tqdm(monet_files, desc="  Monet"):
            self._resize_image(f, monet_train_dir / f.name)
        results['monet']['train'] = len(monet_files)

        # Photos - split train/test
        print(f"\n  Processing photos ({int(self.train_ratio*100)}/{int((1-self.train_ratio)*100)} train/test)...")
        photo_files = self._get_image_files(raw_dir / 'photo_jpg')
        random.shuffle(photo_files)
        
        split_idx = int(len(photo_files) * self.train_ratio)
        train_photos, test_photos = photo_files[:split_idx], photo_files[split_idx:]

        photo_train_dir = self.output_dir / 'train' / 'photo'
        for f in tqdm(train_photos, desc="  Photo Train"):
            self._resize_image(f, photo_train_dir / f.name)
        results['photo']['train'] = len(train_photos)

        photo_test_dir = self.output_dir / 'test' / 'photo'
        for f in tqdm(test_photos, desc="  Photo Test"):
            self._resize_image(f, photo_test_dir / f.name)
        results['photo']['test'] = len(test_photos)

        return results

    def _log_sample_images(self) -> None:
        """Log 8 sample images per category to W&B."""
        print("\n  Logging sample images to W&B...")
        
        samples = {
            "samples/monet_train": self.output_dir / 'train' / 'monet',
            "samples/photo_train": self.output_dir / 'train' / 'photo',
            "samples/photo_test": self.output_dir / 'test' / 'photo',
        }
        
        for key, directory in samples.items():
            files = self._get_image_files(directory)
            if files:
                sample_files = random.sample(files, min(8, len(files)))
                images = [wandb.Image(np.array(Image.open(f)), caption=f.name) for f in sample_files]
                wandb.log({key: images})

    def run_loading(self) -> None:
        """Run the complete loading pipeline."""
        print("\n=== Step 1: Downloading Dataset ===")
        raw_dir = self._download_dataset()

        print("\n=== Step 2: Processing Images ===")
        results = self._process_images(raw_dir)

        print("\n=== Processing Summary ===")
        print(f"  Monet (train): {results['monet']['train']}")
        print(f"  Photo (train): {results['photo']['train']}")
        print(f"  Photo (test):  {results['photo']['test']}")

        # Update artifact metadata
        self.artifact.metadata.update({
            'target_size': self.target_size,
            'train_ratio': self.train_ratio,
            'random_seed': self.random_seed,
            **{f"{cat}_{split}": count for cat, splits in results.items() for split, count in splits.items()}
        })

        # Log metrics
        wandb.log({
            'monet_train': results['monet']['train'],
            'photo_train': results['photo']['train'],
            'photo_test': results['photo']['test'],
        })

        print("\n=== Step 3: Logging Samples to W&B ===")
        self._log_sample_images()

        print("\n  Data loading completed!")
