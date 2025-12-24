"""Augmentation configs and transforms for Grain pipelines.

Implementation note: Augmentations should match standard libraries (torchvision.transforms.v2, torchaudio) or cite papers. This ensures reproducibility and makes it easier to compare with published results. Always cite the source when implementing an augmentation.

Each augmentation is a frozen dataclass that subclasses grain.transforms.Map, with a `map` method that transforms the sample dict in-place.
"""

import dataclasses

import beartype
import grain.python as grain
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float


# -----------------------------------------------------------------------------
# Augmentation Transforms (Grain-compatible)
# -----------------------------------------------------------------------------


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class RandomCrop(grain.MapTransform):
    """Pad then random crop back to original size.

    Matches torchvision.transforms.v2.RandomCrop with padding_mode='constant', fill=0.
    """

    padding: int = 4

    def map(self, sample: dict) -> dict:
        x = sample["data"]
        h, w = x.shape
        padded = jnp.pad(x, self.padding)
        top = np.random.randint(0, 2 * self.padding + 1)
        left = np.random.randint(0, 2 * self.padding + 1)
        sample["data"] = jax.lax.dynamic_slice(padded, (top, left), (h, w))
        return sample


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class HorizontalFlip(grain.MapTransform):
    """Random horizontal flip.

    Matches torchvision.transforms.v2.RandomHorizontalFlip.
    """

    p: float = 0.5

    def map(self, sample: dict) -> dict:
        if np.random.random() < self.p:
            sample["data"] = jnp.flip(sample["data"], axis=-1)
        return sample


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class GaussianNoise(grain.MapTransform):
    """Add Gaussian noise.

    Standard augmentation for audio/spectrograms. See torchaudio or audiomentations.
    """

    std: float = 0.1

    def map(self, sample: dict) -> dict:
        x = sample["data"]
        noise = np.random.randn(*x.shape).astype(np.float32) * self.std
        sample["data"] = x + noise
        return sample


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class FreqMask(grain.MapTransform):
    """SpecAugment frequency masking (zeros out horizontal bands).

    From "SpecAugment" (Park et al., 2019). https://arxiv.org/abs/1904.08779
    """

    max_width: int = 10
    n_masks: int = 2

    def map(self, sample: dict) -> dict:
        x = sample["data"]
        h, w = x.shape
        x_np = np.asarray(x).copy()
        for _ in range(self.n_masks):
            width = np.random.randint(0, self.max_width + 1)
            start = np.random.randint(0, max(1, h - width + 1))
            x_np[start : start + width, :] = 0
        sample["data"] = jnp.asarray(x_np)
        return sample


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class TimeMask(grain.MapTransform):
    """SpecAugment time masking (zeros out vertical bands).

    From "SpecAugment" (Park et al., 2019). https://arxiv.org/abs/1904.08779
    """

    max_width: int = 20
    n_masks: int = 2

    def map(self, sample: dict) -> dict:
        x = sample["data"]
        h, w = x.shape
        x_np = np.asarray(x).copy()
        for _ in range(self.n_masks):
            width = np.random.randint(0, self.max_width + 1)
            start = np.random.randint(0, max(1, w - width + 1))
            x_np[:, start : start + width] = 0
        sample["data"] = jnp.asarray(x_np)
        return sample


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class RandomResizedCrop(grain.MapTransform):
    """Random crop then resize to target size (like torchvision).

    Matches torchvision.transforms.v2.RandomResizedCrop.
    """

    scale_min: float = 0.08
    scale_max: float = 1.0
    ratio_min: float = 0.75
    ratio_max: float = 1.333

    def map(self, sample: dict) -> dict:
        x = sample["data"]
        h, w = x.shape
        area = h * w

        for _ in range(10):
            target_area = area * (
                self.scale_min + np.random.random() * (self.scale_max - self.scale_min)
            )
            aspect_ratio = self.ratio_min + np.random.random() * (
                self.ratio_max - self.ratio_min
            )

            crop_w = int(round((target_area * aspect_ratio) ** 0.5))
            crop_h = int(round((target_area / aspect_ratio) ** 0.5))

            if 0 < crop_w <= w and 0 < crop_h <= h:
                top = np.random.randint(0, h - crop_h + 1)
                left = np.random.randint(0, w - crop_w + 1)
                cropped = x[top : top + crop_h, left : left + crop_w]
                sample["data"] = jax.image.resize(cropped, (h, w), method="bilinear")
                return sample

        return sample


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class ColorJitter(grain.MapTransform):
    """Random brightness/contrast jitter for grayscale images.

    Simplified version of torchvision.transforms.v2.ColorJitter (brightness/contrast only).
    """

    brightness: float = 0.8
    contrast: float = 0.8

    def map(self, sample: dict) -> dict:
        x = sample["data"]

        if self.brightness > 0:
            factor = 1.0 + (np.random.random() * 2 - 1) * self.brightness
            x = x * factor

        if self.contrast > 0:
            factor = 1.0 + (np.random.random() * 2 - 1) * self.contrast
            mean = jnp.mean(x)
            x = (x - mean) * factor + mean

        sample["data"] = x
        return sample


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class GaussianBlur(grain.MapTransform):
    """Apply Gaussian blur with random sigma.

    Matches torchvision.transforms.v2.GaussianBlur.
    """

    kernel_size: int = 7
    sigma_min: float = 0.1
    sigma_max: float = 2.0

    def map(self, sample: dict) -> dict:
        from jax.scipy.signal import convolve2d

        x = sample["data"]
        sigma = self.sigma_min + np.random.random() * (self.sigma_max - self.sigma_min)

        k = self.kernel_size
        half = k // 2
        x_coords = jnp.arange(k, dtype=x.dtype) - half
        kernel_1d = jnp.exp(-(x_coords**2) / (2 * sigma**2))
        kernel_1d = kernel_1d / jnp.sum(kernel_1d)
        kernel_2d = jnp.outer(kernel_1d, kernel_1d)

        sample["data"] = convolve2d(x, kernel_2d, mode="same")
        return sample


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Solarize(grain.MapTransform):
    """Invert pixels above threshold.

    Matches torchvision.transforms.v2.RandomSolarize. Threshold is in [0, 1] for normalized inputs.
    """

    threshold: float = 0.5
    p: float = 0.2

    def map(self, sample: dict) -> dict:
        if np.random.random() < self.p:
            x = sample["data"]
            sample["data"] = jnp.where(x > self.threshold, 1.0 - x, x)
        return sample


# Type alias for augmentation configs
Config = (
    RandomCrop
    | HorizontalFlip
    | GaussianNoise
    | FreqMask
    | TimeMask
    | RandomResizedCrop
    | ColorJitter
    | GaussianBlur
    | Solarize
)


# -----------------------------------------------------------------------------
# Helper to apply augmentations directly (for non-Grain usage)
# -----------------------------------------------------------------------------


@beartype.beartype
def apply(x_hw: Float[Array, "h w"], augmentations: list[Config]) -> Array:
    """Apply a list of augmentations to a 2D array.

    This is a convenience function for applying augmentations outside of a Grain pipeline.
    For Grain pipelines, use the transforms directly with dataset.map().
    """
    sample = {"data": x_hw}
    for aug in augmentations:
        sample = aug.map(sample)
    return sample["data"]
