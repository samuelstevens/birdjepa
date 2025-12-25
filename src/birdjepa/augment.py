"""Augmentation configs and transforms for Grain pipelines.

Implementation note: Augmentations run in dataloader worker processes and must use numpy/scipy only (no JAX). This avoids GPU memory contention with the main training process.

Each augmentation is a frozen dataclass that subclasses grain.transforms.Map, with a `map` method that transforms the sample dict in-place.
"""

import dataclasses

import beartype
import grain.python as grain
import jax
import numpy as np


def _assert_numpy(x, name: str = "x"):
    """Assert that x is a numpy array, not a JAX array."""
    assert isinstance(x, np.ndarray), f"{name} must be numpy array, got {type(x)}"
    assert not isinstance(x, jax.Array), f"{name} must not be JAX array"


# -----------------------------------------------------------------------------
# Augmentation Transforms (Grain-compatible, numpy-only)
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
        _assert_numpy(x, "data")
        h, w = x.shape
        padded = np.pad(x, self.padding)
        top = np.random.randint(0, 2 * self.padding + 1)
        left = np.random.randint(0, 2 * self.padding + 1)
        sample["data"] = padded[top : top + h, left : left + w]
        return sample


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class HorizontalFlip(grain.MapTransform):
    """Random horizontal flip.

    Matches torchvision.transforms.v2.RandomHorizontalFlip.
    """

    p: float = 0.5

    def map(self, sample: dict) -> dict:
        x = sample["data"]
        _assert_numpy(x, "data")
        if np.random.random() < self.p:
            sample["data"] = np.flip(x, axis=-1).copy()
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
        _assert_numpy(x, "data")
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
        _assert_numpy(x, "data")
        h, w = x.shape
        x = x.copy()
        for _ in range(self.n_masks):
            width = np.random.randint(0, self.max_width + 1)
            start = np.random.randint(0, max(1, h - width + 1))
            x[start : start + width, :] = 0
        sample["data"] = x
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
        _assert_numpy(x, "data")
        h, w = x.shape
        x = x.copy()
        for _ in range(self.n_masks):
            width = np.random.randint(0, self.max_width + 1)
            start = np.random.randint(0, max(1, w - width + 1))
            x[:, start : start + width] = 0
        sample["data"] = x
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
        from scipy.ndimage import zoom

        x = sample["data"]
        _assert_numpy(x, "data")
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
                zoom_h, zoom_w = h / crop_h, w / crop_w
                sample["data"] = zoom(cropped, (zoom_h, zoom_w), order=1)
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
        _assert_numpy(x, "data")

        if self.brightness > 0:
            factor = 1.0 + (np.random.random() * 2 - 1) * self.brightness
            x = x * factor

        if self.contrast > 0:
            factor = 1.0 + (np.random.random() * 2 - 1) * self.contrast
            mean = np.mean(x)
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
        from scipy.ndimage import gaussian_filter

        x = sample["data"]
        _assert_numpy(x, "data")
        sigma = self.sigma_min + np.random.random() * (self.sigma_max - self.sigma_min)
        sample["data"] = gaussian_filter(x, sigma=sigma)
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
        x = sample["data"]
        _assert_numpy(x, "data")
        if np.random.random() < self.p:
            sample["data"] = np.where(x > self.threshold, 1.0 - x, x)
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
def apply(x_hw: np.ndarray, augmentations: list[Config]) -> np.ndarray:
    """Apply a list of augmentations to a 2D numpy array.

    This is a convenience function for applying augmentations outside of a Grain pipeline.
    For Grain pipelines, use the transforms directly with dataset.map().
    """
    _assert_numpy(x_hw, "x_hw")
    sample = {"data": x_hw}
    for aug in augmentations:
        sample = aug.map(sample)
    return sample["data"]
