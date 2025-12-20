"""
Monet GAN Data Augmenter

Handles downloading artifacts from W&B, applying configurable augmentations
to train images, and logging augmented samples to W&B.
"""

import random
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import wandb
import yaml
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm


class TransformRegistry:
    """Registry for image transformation functions."""
    
    _transforms: Dict[str, Callable] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a transform function."""
        def decorator(func: Callable):
            cls._transforms[name] = func
            return func
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Callable:
        """Get a transform function by name."""
        if name not in cls._transforms:
            raise ValueError(f"Unknown transform: {name}. Available: {list(cls._transforms.keys())}")
        return cls._transforms[name]
    
    @classmethod
    def list_transforms(cls) -> List[str]:
        """List all registered transforms."""
        return list(cls._transforms.keys())


# Register all transforms
@TransformRegistry.register("random_rotation")
def random_rotation(img: Image.Image, degrees: float = 15) -> Image.Image:
    """Rotate image by a random angle within [-degrees, degrees]."""
    angle = random.uniform(-degrees, degrees)
    return img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))


@TransformRegistry.register("random_horizontal_flip")
def random_horizontal_flip(img: Image.Image, probability: float = 0.5) -> Image.Image:
    """Flip image horizontally with given probability."""
    if random.random() < probability:
        return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return img


@TransformRegistry.register("random_vertical_flip")
def random_vertical_flip(img: Image.Image, probability: float = 0.5) -> Image.Image:
    """Flip image vertically with given probability."""
    if random.random() < probability:
        return img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    return img


@TransformRegistry.register("color_jitter")
def color_jitter(
    img: Image.Image,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = 0.1
) -> Image.Image:
    """Randomly adjust brightness, contrast, saturation, and hue."""
    # Brightness
    if brightness > 0:
        factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(factor)
    
    # Contrast
    if contrast > 0:
        factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(factor)
    
    # Saturation
    if saturation > 0:
        factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(factor)
    
    # Hue - convert to HSV, shift hue, convert back
    if hue > 0:
        hue_shift = random.uniform(-hue, hue)
        if abs(hue_shift) > 0.01:
            hsv_img = img.convert('HSV')
            h, s, v = hsv_img.split()
            h_array = np.array(h, dtype=np.int16)
            h_array = (h_array + int(hue_shift * 255)) % 256
            h = Image.fromarray(h_array.astype(np.uint8), mode='L')
            img = Image.merge('HSV', (h, s, v)).convert('RGB')
    
    return img


@TransformRegistry.register("random_crop")
def random_crop(
    img: Image.Image,
    scale_min: float = 0.8,
    scale_max: float = 1.0
) -> Image.Image:
    """Random crop and resize back to original size."""
    width, height = img.size
    scale = random.uniform(scale_min, scale_max)
    
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    left = random.randint(0, width - new_width)
    top = random.randint(0, height - new_height)
    
    img = img.crop((left, top, left + new_width, top + new_height))
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    
    return img


@TransformRegistry.register("gaussian_blur")
def gaussian_blur(
    img: Image.Image,
    sigma_min: float = 0.1,
    sigma_max: float = 2.0
) -> Image.Image:
    """Apply Gaussian blur with random sigma."""
    sigma = random.uniform(sigma_min, sigma_max)
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


@TransformRegistry.register("random_affine")
def random_affine(
    img: Image.Image,
    degrees: float = 10,
    translate: float = 0.1,
    scale: Tuple[float, float] = (0.9, 1.1),
    shear: float = 10
) -> Image.Image:
    """Apply random affine transformation."""
    width, height = img.size
    
    # Rotation
    angle = random.uniform(-degrees, degrees)
    
    # Translation
    max_tx = translate * width
    max_ty = translate * height
    tx = random.uniform(-max_tx, max_tx)
    ty = random.uniform(-max_ty, max_ty)
    
    # Scale
    s = random.uniform(scale[0], scale[1])
    
    # Shear
    shear_x = random.uniform(-shear, shear)
    
    # Apply transforms
    img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
    img = img.transform(
        (width, height),
        Image.Transform.AFFINE,
        (1, shear_x / 100, tx, 0, 1, ty),
        resample=Image.Resampling.BILINEAR
    )
    
    # Scale
    new_size = (int(width * s), int(height * s))
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    # Center crop or pad back to original size
    if s > 1:
        left = (new_size[0] - width) // 2
        top = (new_size[1] - height) // 2
        img = img.crop((left, top, left + width, top + height))
    else:
        # Pad with black
        new_img = Image.new('RGB', (width, height), (0, 0, 0))
        left = (width - new_size[0]) // 2
        top = (height - new_size[1]) // 2
        new_img.paste(img, (left, top))
        img = new_img
    
    return img


@TransformRegistry.register("random_perspective")
def random_perspective(
    img: Image.Image,
    distortion_scale: float = 0.2,
    probability: float = 0.5
) -> Image.Image:
    """Apply random perspective transformation."""
    if random.random() > probability:
        return img
    
    width, height = img.size
    
    # Calculate perspective coefficients
    half_w = width / 2
    half_h = height / 2
    
    # Random distortion for each corner
    tl = (random.uniform(0, distortion_scale * half_w), 
          random.uniform(0, distortion_scale * half_h))
    tr = (width - random.uniform(0, distortion_scale * half_w),
          random.uniform(0, distortion_scale * half_h))
    br = (width - random.uniform(0, distortion_scale * half_w),
          height - random.uniform(0, distortion_scale * half_h))
    bl = (random.uniform(0, distortion_scale * half_w),
          height - random.uniform(0, distortion_scale * half_h))
    
    # Find perspective transform coefficients
    coeffs = _find_perspective_coeffs(
        [(0, 0), (width, 0), (width, height), (0, height)],
        [tl, tr, br, bl]
    )
    
    return img.transform((width, height), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BILINEAR)


def _find_perspective_coeffs(source_coords, target_coords):
    """Find coefficients for perspective transform."""
    matrix = []
    for s, t in zip(source_coords, target_coords):
        matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0]*t[0], -s[0]*t[1]])
        matrix.append([0, 0, 0, t[0], t[1], 1, -s[1]*t[0], -s[1]*t[1]])
    
    A = np.array(matrix, dtype=np.float64)
    B = np.array([s[0] for s in source_coords] + [s[1] for s in source_coords], dtype=np.float64)
    
    res = np.linalg.lstsq(A, B, rcond=None)[0]
    return tuple(res)


@TransformRegistry.register("random_erasing")
def random_erasing(
    img: Image.Image,
    probability: float = 0.5,
    scale_min: float = 0.02,
    scale_max: float = 0.33,
    ratio_min: float = 0.3,
    ratio_max: float = 3.3
) -> Image.Image:
    """Randomly erase a rectangular region."""
    if random.random() > probability:
        return img
    
    img_array = np.array(img)
    height, width, _ = img_array.shape
    area = height * width
    
    for _ in range(10):  # Try up to 10 times to find valid region
        target_area = random.uniform(scale_min, scale_max) * area
        aspect_ratio = random.uniform(ratio_min, ratio_max)
        
        h = int(round(np.sqrt(target_area * aspect_ratio)))
        w = int(round(np.sqrt(target_area / aspect_ratio)))
        
        if h < height and w < width:
            top = random.randint(0, height - h)
            left = random.randint(0, width - w)
            
            # Fill with random values or mean color
            img_array[top:top+h, left:left+w] = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            return Image.fromarray(img_array)
    
    return img


class AugmentationPipeline:
    """Pipeline for applying a sequence of transforms."""
    
    def __init__(self, transforms: List[Dict[str, Any]]):
        """
        Initialize pipeline with list of transforms.
        
        Args:
            transforms: List of dicts with 'name' and 'params' keys
        """
        self.transforms = []
        for t in transforms:
            name = t['name']
            params = t.get('params', {})
            func = TransformRegistry.get(name)
            self.transforms.append((name, func, params))
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply all transforms sequentially."""
        for _, func, params in self.transforms:
            img = func(img, **params)
        return img
    
    def get_transform_names(self) -> List[str]:
        """Get list of transform names in pipeline."""
        return [name for name, _, _ in self.transforms]


class Augmenter:
    """Augments the Monet GAN dataset based on configuration."""

    def __init__(
        self,
        artifact: wandb.Artifact,
        input_dir: str,
        output_dir: str,
        config_path: str,
    ):
        self.artifact = artifact
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Extract config values
        self.random_seed = self.config['output']['random_seed']
        self.augmentations = self.config.get('augmentations', {})
        
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
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
        merged_config_path = Path(tempfile.gettempdir()) / "merged_augmenter_config.yaml"
        with open(merged_config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        self.artifact.add_file(str(merged_config_path), name="config.yaml")

        # Log source files
        augmenter_dir = Path(__file__).parent
        for file_path in augmenter_dir.glob("*.py"):
            self.artifact.add_file(str(file_path), name=f"augmenter_code/{file_path.name}")

    def print_config(self) -> None:
        """Print configuration summary."""
        print(f"  Random Seed: {self.random_seed}")
        print(f"  Input Dir: {self.input_dir}")
        print(f"  Output Dir: {self.output_dir}")
        print(f"\n  Augmentation Groups:")
        
        for img_type, aug_groups in self.augmentations.items():
            if aug_groups:
                print(f"\n    {img_type}:")
                for aug_name, aug_config in aug_groups.items():
                    transforms = [t['name'] for t in aug_config.get('transforms', [])]
                    fraction = aug_config.get('fraction', 0)
                    print(f"      {aug_name}: {transforms} (fraction={fraction})")

    def _get_image_files(self, directory: Path) -> List[Path]:
        """Get sorted list of image files in directory."""
        if not directory.exists():
            return []
        extensions = {'.jpg', '.jpeg', '.png'}
        return sorted([f for f in directory.iterdir() if f.suffix.lower() in extensions])

    def _copy_original_images(self, src_dir: Path, dst_dir: Path) -> int:
        """Copy all original images to output directory."""
        dst_dir.mkdir(parents=True, exist_ok=True)
        files = self._get_image_files(src_dir)
        
        for f in tqdm(files, desc=f"  Copying {src_dir.name}"):
            with Image.open(f) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(dst_dir / f.name, 'JPEG', quality=95)
        
        return len(files)

    def _apply_augmentations(
        self,
        src_dir: Path,
        dst_dir: Path,
        aug_groups: Dict[str, Any]
    ) -> Dict[str, Tuple[int, List[Path]]]:
        """
        Apply augmentations to images in source directory.
        
        Returns dict mapping aug_name to (count, list of augmented file paths)
        """
        if not aug_groups:
            return {}
        
        results: Dict[str, Tuple[int, List[Path]]] = {}
        all_files = self._get_image_files(src_dir)
        
        if not all_files:
            return results
        
        # Calculate how many images each augmentation group gets
        # Images are sampled WITHOUT replacement across groups
        total_files = len(all_files)
        remaining_files = all_files.copy()
        random.shuffle(remaining_files)
        
        # Calculate fractions and validate they don't exceed 1.0
        total_fraction = sum(cfg.get('fraction', 0) for cfg in aug_groups.values())
        if total_fraction > 1.0:
            print(f"  Warning: Total augmentation fraction ({total_fraction}) > 1.0, will be normalized")
        
        for aug_name, aug_config in aug_groups.items():
            fraction = aug_config.get('fraction', 0)
            transforms = aug_config.get('transforms', [])
            
            if fraction <= 0 or not transforms:
                results[aug_name] = (0, [])
                continue
            
            # Calculate number of images for this augmentation
            num_images = int(total_files * fraction)
            num_images = min(num_images, len(remaining_files))
            
            if num_images == 0:
                results[aug_name] = (0, [])
                continue
            
            # Select images for this augmentation (without replacement)
            selected_files = remaining_files[:num_images]
            remaining_files = remaining_files[num_images:]
            
            # Create augmentation pipeline
            pipeline = AugmentationPipeline(transforms)
            
            # Apply augmentations
            augmented_paths: List[Path] = []
            for f in tqdm(selected_files, desc=f"  Augmenting {aug_name}"):
                with Image.open(f) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    augmented_img = pipeline(img)
                    
                    # Save with aug prefix
                    output_name = f"{aug_name}_{f.name}"
                    output_path = dst_dir / output_name
                    augmented_img.save(output_path, 'JPEG', quality=95)
                    augmented_paths.append(output_path)
            
            results[aug_name] = (len(selected_files), augmented_paths)
        
        return results

    def _log_sample_images(
        self,
        aug_results: Dict[str, Dict[str, Tuple[int, List[Path]]]],
        num_samples: int = 8
    ) -> None:
        """Log sample augmented images to W&B."""
        print("\n  Logging sample images to W&B...")
        
        for img_type, type_results in aug_results.items():
            for aug_name, (count, paths) in type_results.items():
                if count == 0 or not paths:
                    continue
                
                sample_paths = random.sample(paths, min(num_samples, len(paths)))
                images = [
                    wandb.Image(np.array(Image.open(p)), caption=p.name)
                    for p in sample_paths
                ]
                wandb.log({f"samples/{img_type}_{aug_name}": images})

    def run_augmentation(self) -> Dict[str, Any]:
        """Run the complete augmentation pipeline."""
        results = {
            'original': {'photo': 0, 'monet': 0},
            'augmented': {}
        }
        
        # Process both photo and monet training images
        for img_type in ['photo', 'monet']:
            src_train_dir = self.input_dir / 'train' / img_type
            dst_train_dir = self.output_dir / 'train' / img_type
            
            if not src_train_dir.exists():
                print(f"  Warning: {src_train_dir} does not exist, skipping {img_type}")
                continue
            
            # Step 1: Copy original images
            print(f"\n  Copying original {img_type} images...")
            dst_train_dir.mkdir(parents=True, exist_ok=True)
            original_count = self._copy_original_images(src_train_dir, dst_train_dir)
            results['original'][img_type] = original_count
            
            # Step 2: Apply augmentations
            aug_groups = self.augmentations.get(img_type, {})
            if aug_groups:
                print(f"\n  Applying augmentations to {img_type}...")
                aug_results = self._apply_augmentations(src_train_dir, dst_train_dir, aug_groups)
                results['augmented'][img_type] = aug_results
            else:
                results['augmented'][img_type] = {}
        
        # Copy test images (photos only, no augmentation)
        src_test_dir = self.input_dir / 'test' / 'photo'
        dst_test_dir = self.output_dir / 'test' / 'photo'
        if src_test_dir.exists():
            print("\n  Copying test photos (no augmentation)...")
            dst_test_dir.mkdir(parents=True, exist_ok=True)
            test_count = self._copy_original_images(src_test_dir, dst_test_dir)
            results['original']['photo_test'] = test_count
        
        # Log samples to W&B
        print("\n=== Step 3: Logging Samples to W&B ===")
        self._log_sample_images(results['augmented'])
        
        # Update artifact metadata
        metadata = {
            'random_seed': self.random_seed,
            'original_photo_train': results['original']['photo'],
            'original_monet_train': results['original']['monet'],
            'photo_test': results['original'].get('photo_test', 0),
        }
        
        # Add augmentation counts
        for img_type, type_results in results['augmented'].items():
            for aug_name, (count, _) in type_results.items():
                metadata[f'{img_type}_{aug_name}'] = count
        
        self.artifact.metadata.update(metadata)
        
        # Log metrics to W&B
        total_photo_train = results['original']['photo'] + sum(
            count for count, _ in results['augmented'].get('photo', {}).values()
        )
        total_monet_train = results['original']['monet'] + sum(
            count for count, _ in results['augmented'].get('monet', {}).values()
        )
        
        wandb.log({
            'original_photo_train': results['original']['photo'],
            'original_monet_train': results['original']['monet'],
            'photo_test': results['original'].get('photo_test', 0),
            'total_photo_train': total_photo_train,
            'total_monet_train': total_monet_train,
            **{f'{img_type}_{aug_name}': count 
               for img_type, type_results in results['augmented'].items() 
               for aug_name, (count, _) in type_results.items()}
        })
        
        return results

    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print augmentation summary."""
        print("\n=== Augmentation Summary ===")
        print(f"  Original photo (train): {results['original']['photo']}")
        print(f"  Original monet (train): {results['original']['monet']}")
        print(f"  Photo (test): {results['original'].get('photo_test', 0)}")
        
        for img_type, type_results in results['augmented'].items():
            if type_results:
                print(f"\n  {img_type.capitalize()} Augmentations:")
                for aug_name, (count, _) in type_results.items():
                    print(f"    {aug_name}: {count} images")
        
        # Calculate totals
        total_photo = results['original']['photo']
        total_monet = results['original']['monet']
        
        for aug_name, (count, _) in results['augmented'].get('photo', {}).items():
            total_photo += count
        for aug_name, (count, _) in results['augmented'].get('monet', {}).items():
            total_monet += count
        
        print(f"\n  Total photo (train): {total_photo}")
        print(f"  Total monet (train): {total_monet}")
