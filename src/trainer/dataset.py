"""
Dataset and DataLoader utilities for Monet GAN training.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    """
    Simple image dataset that loads images from a directory.
    
    Supports paired (A and B in same order) or unpaired modes.
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png'),
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.extensions = extensions
        
        # Collect image paths
        self.image_paths = sorted([
            p for p in self.root_dir.iterdir()
            if p.suffix.lower() in extensions
        ])
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {root_dir}")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image


class UnpairedDataset(Dataset):
    """
    Dataset for unpaired image-to-image translation (e.g., CycleGAN).
    
    Returns images from domain A and domain B (not necessarily corresponding).
    """
    
    def __init__(
        self,
        root_a: Union[str, Path],
        root_b: Union[str, Path],
        transform: Optional[Callable] = None,
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png'),
    ):
        self.dataset_a = ImageDataset(root_a, transform, extensions)
        self.dataset_b = ImageDataset(root_b, transform, extensions)
        
        self.len_a = len(self.dataset_a)
        self.len_b = len(self.dataset_b)
    
    def __len__(self) -> int:
        return max(self.len_a, self.len_b)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Wrap around if one dataset is smaller
        idx_a = idx % self.len_a
        idx_b = idx % self.len_b
        
        return {
            'A': self.dataset_a[idx_a],
            'B': self.dataset_b[idx_b],
        }


DEFAULT_MEAN = [0.5, 0.5, 0.5]
DEFAULT_STD = [0.5, 0.5, 0.5]


def get_default_transform(
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
) -> Callable:
    """Get default image transform (images are already resized by augmenter stage)."""
    import torchvision.transforms as T
    
    mean = mean if mean is not None else DEFAULT_MEAN
    std = std if std is not None else DEFAULT_STD
    
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def get_train_transform(
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
) -> Callable:
    """Get training transform (images are already resized)."""
    import torchvision.transforms as T
    
    mean = mean if mean is not None else DEFAULT_MEAN
    std = std if std is not None else DEFAULT_STD
    
    transforms = [
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ]
    
    return T.Compose(transforms)


def create_dataloaders(
    config: Dict[str, Any],
    data_dir: Union[str, Path],
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for training.
    
    Expected directory structure:
        data_dir/
            train/
                monet/
                photo/
            test/
                photo/
    
    Config:
        batch_size: 4
        num_workers: 4
        normalize:
            mean: [0.5, 0.5, 0.5]
            std: [0.5, 0.5, 0.5]
    """
    data_dir = Path(data_dir)
    
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    normalize_config = config.get('normalize', {})
    mean = normalize_config.get('mean', DEFAULT_MEAN)
    std = normalize_config.get('std', DEFAULT_STD)
    
    train_transform = get_train_transform(mean=mean, std=std)
    test_transform = get_default_transform(mean=mean, std=std)
    
    # Training dataset (unpaired: photo <-> monet)
    train_dataset = UnpairedDataset(
        root_a=data_dir / 'train' / 'photo',
        root_b=data_dir / 'train' / 'monet',
        transform=train_transform,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    # Test dataset (photo only for generating monet-style images)
    test_photo_dataset = ImageDataset(
        root_dir=data_dir / 'test' / 'photo',
        transform=test_transform,
    )
    
    test_loader = DataLoader(
        test_photo_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    # Monet dataset for evaluation (real target domain)
    monet_dataset = ImageDataset(
        root_dir=data_dir / 'train' / 'monet',
        transform=test_transform,
    )
    
    monet_loader = DataLoader(
        monet_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    # Photo training dataset for evaluation
    photo_train_dataset = ImageDataset(
        root_dir=data_dir / 'train' / 'photo',
        transform=test_transform,
    )
    
    photo_train_loader = DataLoader(
        photo_train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return {
        'train': train_loader,
        'test': test_loader,
        'monet': monet_loader,
        'photo_train': photo_train_loader,
    }


class ImageBuffer:
    """
    Image buffer for discriminator training.
    
    Stores previously generated images to stabilize training.
    """
    
    def __init__(self, buffer_size: int = 50):
        self.buffer_size = buffer_size
        self.buffer: List[torch.Tensor] = []
    
    def push_and_pop(self, images: torch.Tensor) -> torch.Tensor:
        """
        Add images to buffer and return a mix of new and buffered images.
        """
        if self.buffer_size == 0:
            return images
        
        return_images = []
        for image in images:
            image = image.unsqueeze(0)
            
            if len(self.buffer) < self.buffer_size:
                self.buffer.append(image)
                return_images.append(image)
            else:
                if np.random.random() > 0.5:
                    # Return random buffered image and add new one
                    idx = np.random.randint(0, self.buffer_size)
                    return_images.append(self.buffer[idx].clone())
                    self.buffer[idx] = image
                else:
                    # Return new image
                    return_images.append(image)
        
        return torch.cat(return_images, dim=0)
