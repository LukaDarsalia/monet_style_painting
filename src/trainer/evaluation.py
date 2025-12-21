"""
MiFID (Memorization-informed Fréchet Inception Distance) Evaluation

Simplified implementation using pretrained Inception V3 from torchvision.
Based on Kaggle's MiFID metric implementation.
"""

from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg
from tqdm import tqdm
from torchvision import models


class InceptionV3Features(nn.Module):
    """
    Inception V3 feature extractor for FID calculation.
    
    Uses pretrained Inception V3 and extracts 2048-dimensional features
    from the final pooling layer.
    """
    
    def __init__(
        self,
        input_mean: Optional[List[float]] = None,
        input_std: Optional[List[float]] = None,
    ):
        super().__init__()
        
        # Load pretrained Inception V3
        inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        inception.eval()
        
        # Build feature extractor (up to final pooling)
        self.blocks = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )
        
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
        
        # Input normalization (defaults to [-1, 1] -> [0, 1])
        input_mean = input_mean if input_mean is not None else [0.5, 0.5, 0.5]
        input_std = input_std if input_std is not None else [0.5, 0.5, 0.5]
        self.register_buffer('input_mean', torch.tensor(input_mean).view(1, 3, 1, 1))
        self.register_buffer('input_std', torch.tensor(input_std).view(1, 3, 1, 1))
        
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess images for Inception.
        
        Expects input normalized with input_mean/input_std.
        Resizes to 299x299 and normalizes with ImageNet stats.
        """
        # Resize to 299x299 (Inception input size)
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Input normalization -> [0, 1] -> ImageNet normalized
        x = x * self.input_std + self.input_mean
        x = (x - self.mean) / self.std
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract 2048-dimensional features.
        
        Args:
            x: Images in [-1, 1] range, shape (B, 3, H, W)
            
        Returns:
            Features of shape (B, 2048)
        """
        x = self.preprocess(x)
        x = self.blocks(x)
        return x.view(x.size(0), -1)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    """Normalize each row to have unit length."""
    norms = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    return np.nan_to_num(x / norms)


def calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6
) -> float:
    """
    Calculate Fréchet distance between two multivariate Gaussians.
    
    FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
    
    Stable implementation by Dougal J. Sutherland.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    assert mu1.shape == mu2.shape, "Mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Covariance matrices have different dimensions"
    
    diff = mu1 - mu2
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    if not np.isfinite(covmean).all():
        msg = f"FID calculation produces singular product; adding {eps} to diagonal of cov estimates"
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_activation_statistics(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate mean and covariance of features."""
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def cosine_distance(features1: np.ndarray, features2: np.ndarray) -> float:
    """
    Calculate mean minimum cosine distance from features1 to features2.
    
    For each vector in features1, find the minimum cosine distance to any vector in features2.
    Returns the mean of these minimum distances.
    """
    # Remove zero vectors
    features1_nozero = features1[np.sum(features1, axis=1) != 0]
    features2_nozero = features2[np.sum(features2, axis=1) != 0]
    
    # Normalize
    norm_f1 = normalize_rows(features1_nozero)
    norm_f2 = normalize_rows(features2_nozero)
    
    # Cosine distance = 1 - |cosine_similarity|
    # Process in batches for memory efficiency
    min_distances = []
    batch_size = 100
    
    for i in range(0, len(norm_f1), batch_size):
        batch = norm_f1[i:i+batch_size]
        # Cosine similarity
        similarity = np.abs(np.matmul(batch, norm_f2.T))
        # Cosine distance
        distance = 1.0 - similarity
        # Minimum distance for each sample
        min_dist = np.min(distance, axis=1)
        min_distances.append(min_dist)
    
    min_distances = np.concatenate(min_distances)
    return float(np.mean(min_distances))


def distance_thresholding(d: float, eps: float) -> float:
    """Apply threshold to memorization distance."""
    return d if d < eps else 1.0


class MiFIDCalculator:
    """
    Calculator for MiFID (Memorization-informed FID).
    
    MiFID = FID / d_thr
    
    where d_thr is the thresholded memorization distance (cosine distance).
    Based on Kaggle's generative dog images competition metric.
    """
    
    def __init__(
        self,
        device: torch.device,
        cosine_distance_eps: float = 0.1,
        input_mean: Optional[List[float]] = None,
        input_std: Optional[List[float]] = None,
    ):
        """
        Args:
            device: Device to run computations on
            cosine_distance_eps: Threshold for memorization distance (default 0.1)
        """
        self.device = device
        self.cosine_distance_eps = cosine_distance_eps
        
        # Feature extractor
        self.inception = InceptionV3Features(
            input_mean=input_mean,
            input_std=input_std,
        ).to(device).eval()
    
    @torch.no_grad()
    def extract_features(
        self,
        images: torch.Tensor,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Extract Inception features from images tensor."""
        self.inception.eval()
        
        features = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(self.device)
            # Resize to 256x256 for consistent evaluation
            if batch.shape[2] != 256 or batch.shape[3] != 256:
                batch = F.interpolate(batch, size=(256, 256), mode='bilinear', align_corners=False)
            feat = self.inception(batch)
            features.append(feat.cpu().numpy())
        
        return np.concatenate(features, axis=0)
    
    @torch.no_grad()
    def extract_features_from_dataloader(
        self,
        dataloader,
        max_samples: Optional[int] = None,
        desc: str = "Extracting features",
    ) -> np.ndarray:
        """Extract features from a DataLoader."""
        self.inception.eval()
        
        features = []
        num_samples = 0
        
        for batch in tqdm(dataloader, desc=desc):
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
            
            images = images.to(self.device)
            # Resize to 256x256 for consistent evaluation
            if images.shape[2] != 256 or images.shape[3] != 256:
                images = F.interpolate(images, size=(256, 256), mode='bilinear', align_corners=False)
            feat = self.inception(images)
            features.append(feat.cpu().numpy())
            
            num_samples += len(images)
            if max_samples is not None and num_samples >= max_samples:
                break
        
        features = np.concatenate(features, axis=0)
        if max_samples is not None:
            features = features[:max_samples]
        
        return features
    
    @torch.no_grad()
    def generate_features(
        self,
        generator: nn.Module,
        source_images: torch.Tensor,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Generate images and extract their features."""
        generator.eval()
        self.inception.eval()
        
        features = []
        for i in range(0, len(source_images), batch_size):
            batch = source_images[i:i+batch_size].to(self.device)
            generated = generator(batch)
            # Resize to 256x256 for consistent evaluation
            if generated.shape[2] != 256 or generated.shape[3] != 256:
                generated = F.interpolate(generated, size=(256, 256), mode='bilinear', align_corners=False)
            feat = self.inception(generated)
            features.append(feat.cpu().numpy())
        
        return np.concatenate(features, axis=0)
    
    def calculate_fid(
        self,
        real_features: np.ndarray,
        generated_features: np.ndarray,
    ) -> float:
        """Calculate FID between real and generated features."""
        mu_real, sigma_real = calculate_activation_statistics(real_features)
        mu_gen, sigma_gen = calculate_activation_statistics(generated_features)
        
        fid = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
        return float(fid)
    
    def calculate_memorization_distance(
        self,
        generated_features: np.ndarray,
        training_features: np.ndarray,
    ) -> float:
        """Calculate memorization distance (cosine distance)."""
        return cosine_distance(generated_features, training_features)
    
    def calculate_mifid(
        self,
        real_features: np.ndarray,
        generated_features: np.ndarray,
        training_features: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Calculate MiFID.
        
        Args:
            real_features: Features from real target images
            generated_features: Features from generated images  
            training_features: Features from training images (for memorization check)
        
        Returns:
            - MiFID score
            - FID score
            - Memorization distance (before thresholding)
        """
        # Calculate FID
        fid = self.calculate_fid(real_features, generated_features)
        
        # Calculate memorization distance
        mem_dist = self.calculate_memorization_distance(generated_features, training_features)
        
        # Apply threshold
        d_thr = distance_thresholding(mem_dist, self.cosine_distance_eps)
        
        # MiFID (with small epsilon to avoid division issues)
        fid_epsilon = 1e-15
        mifid = fid / (d_thr + fid_epsilon)
        
        return mifid, fid, mem_dist


class Evaluator:
    """
    Complete evaluator for GAN training.
    
    Calculates FID, MiFID, and can generate sample images for visualization.
    """
    
    def __init__(
        self,
        device: torch.device,
        config: Dict[str, Any],
        input_mean: Optional[List[float]] = None,
        input_std: Optional[List[float]] = None,
    ):
        self.device = device
        self.config = config
        
        self.cosine_distance_eps = config['cosine_distance_eps']
        self.num_samples = config['num_evaluation_samples']
        self.batch_size = config['batch_size']
        
        self.mifid_calculator = MiFIDCalculator(
            device,
            self.cosine_distance_eps,
            input_mean=input_mean,
            input_std=input_std,
        )
        
        # Cache for training features (computed once)
        self._training_features_cache = None
    
    def set_training_features(self, features: np.ndarray):
        """Cache training features for memorization calculation."""
        self._training_features_cache = features
    
    @torch.no_grad()
    def compute_training_features(
        self,
        dataloader,
        max_samples: Optional[int] = None,
    ) -> np.ndarray:
        """Compute and cache training features."""
        features = self.mifid_calculator.extract_features_from_dataloader(
            dataloader,
            max_samples=max_samples,
            desc="Computing training features"
        )
        self._training_features_cache = features
        return features
    
    @torch.no_grad()
    def evaluate(
        self,
        generator: nn.Module,
        source_dataloader,
        target_dataloader,
        max_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Run full evaluation.
        
        Args:
            generator: Generator model (source -> target)
            source_dataloader: DataLoader for source domain images
            target_dataloader: DataLoader for target domain images (real targets)
            max_samples: Maximum number of samples to use
        
        Returns:
            Dict with FID, MiFID, memorization distance
        """
        max_samples = max_samples or self.num_samples
        
        generator.eval()
        
        # Collect source images
        source_images = []
        for batch in source_dataloader:
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
            source_images.append(images)
            if sum(len(x) for x in source_images) >= max_samples:
                break
        source_images = torch.cat(source_images, dim=0)[:max_samples]
        
        # Generate images and extract features
        generated_features = self.mifid_calculator.generate_features(
            generator, source_images, self.batch_size
        )
        
        # Extract target (real) features
        target_features = self.mifid_calculator.extract_features_from_dataloader(
            target_dataloader,
            max_samples=max_samples,
            desc="Extracting target features"
        )
        
        # Get training features (use target features if not cached)
        if self._training_features_cache is None:
            training_features = target_features
        else:
            training_features = self._training_features_cache
        
        # Calculate metrics
        mifid, fid, mem_dist = self.mifid_calculator.calculate_mifid(
            target_features, generated_features, training_features
        )
        
        return {
            'mifid': mifid,
            'fid': fid,
            'memorization_distance': mem_dist,
        }
    
    @torch.no_grad()
    def generate_samples(
        self,
        generator: nn.Module,
        source_images: torch.Tensor,
        num_samples: int = 8,
    ) -> torch.Tensor:
        """Generate sample images for visualization."""
        generator.eval()
        source_batch = source_images[:num_samples].to(self.device)
        generated = generator(source_batch)
        return generated
