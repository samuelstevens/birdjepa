"""Augmentation configs and transforms.

Implementation note: Augmentations should match standard libraries (torchvision.transforms.v2, torchaudio) or cite papers. This ensures reproducibility and makes it easier to compare with published results. Always cite the source when implementing an augmentation.
"""

import dataclasses

import beartype
import torch
import torch.nn.functional as F
from jaxtyping import Float, jaxtyped
from torch import Tensor


# -----------------------------------------------------------------------------
# Augmentation Configs
# -----------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RandomCrop:
    """Pad then random crop back to original size."""

    padding: int = 4


@dataclasses.dataclass(frozen=True)
class HorizontalFlip:
    """Random horizontal flip."""

    p: float = 0.5


@dataclasses.dataclass(frozen=True)
class GaussianNoise:
    """Add Gaussian noise."""

    std: float = 0.1


@dataclasses.dataclass(frozen=True)
class FreqMask:
    """SpecAugment frequency masking (zeros out horizontal bands)."""

    max_width: int = 10
    n_masks: int = 2


@dataclasses.dataclass(frozen=True)
class TimeMask:
    """SpecAugment time masking (zeros out vertical bands)."""

    max_width: int = 20
    n_masks: int = 2


@dataclasses.dataclass(frozen=True)
class RandomResizedCrop:
    """Random crop then resize to target size (like torchvision)."""

    scale_min: float = 0.08
    scale_max: float = 1.0
    ratio_min: float = 0.75
    ratio_max: float = 1.333


@dataclasses.dataclass(frozen=True)
class ColorJitter:
    """Random brightness/contrast jitter for grayscale images."""

    brightness: float = 0.8
    contrast: float = 0.8


@dataclasses.dataclass(frozen=True)
class GaussianBlur:
    """Apply Gaussian blur with random sigma."""

    kernel_size: int = 7
    sigma_min: float = 0.1
    sigma_max: float = 2.0


@dataclasses.dataclass(frozen=True)
class Solarize:
    """Invert pixels above threshold."""

    threshold: float = 0.5
    p: float = 0.2


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
# Augmentation Implementations
# -----------------------------------------------------------------------------


@jaxtyped(typechecker=beartype.beartype)
def apply_random_crop(
    x_hw: Float[Tensor, "h w"], cfg: RandomCrop
) -> Float[Tensor, "h w"]:
    """Pad with zeros then random crop back to original size.

    Matches torchvision.transforms.v2.RandomCrop with padding_mode='constant', fill=0.
    """
    h, w = x_hw.shape
    padded = F.pad(x_hw, [cfg.padding] * 4)
    top = torch.randint(0, 2 * cfg.padding + 1, ()).item()
    left = torch.randint(0, 2 * cfg.padding + 1, ()).item()
    return padded[top : top + h, left : left + w]


@jaxtyped(typechecker=beartype.beartype)
def apply_horizontal_flip(
    x_hw: Float[Tensor, "h w"], cfg: HorizontalFlip
) -> Float[Tensor, "h w"]:
    """Random horizontal flip.

    Matches torchvision.transforms.v2.RandomHorizontalFlip.
    """
    if torch.rand(()).item() < cfg.p:
        return x_hw.flip(-1)
    return x_hw


@jaxtyped(typechecker=beartype.beartype)
def apply_gaussian_noise(
    x_hw: Float[Tensor, "h w"], cfg: GaussianNoise
) -> Float[Tensor, "h w"]:
    """Add Gaussian noise.

    Standard augmentation for audio/spectrograms. See torchaudio or audiomentations.
    """
    return x_hw + torch.randn_like(x_hw) * cfg.std


@jaxtyped(typechecker=beartype.beartype)
def apply_freq_mask(x_hw: Float[Tensor, "h w"], cfg: FreqMask) -> Float[Tensor, "h w"]:
    """SpecAugment frequency masking (zeros out horizontal bands).

    From "SpecAugment" (Park et al., 2019). https://arxiv.org/abs/1904.08779
    """
    h, w = x_hw.shape
    x = x_hw.clone()
    for _ in range(cfg.n_masks):
        width = torch.randint(0, cfg.max_width + 1, ()).item()
        start = torch.randint(0, max(1, h - width + 1), ()).item()
        x[start : start + width, :] = 0
    return x


@jaxtyped(typechecker=beartype.beartype)
def apply_time_mask(x_hw: Float[Tensor, "h w"], cfg: TimeMask) -> Float[Tensor, "h w"]:
    """SpecAugment time masking (zeros out vertical bands).

    From "SpecAugment" (Park et al., 2019). https://arxiv.org/abs/1904.08779
    """
    h, w = x_hw.shape
    x = x_hw.clone()
    for _ in range(cfg.n_masks):
        width = torch.randint(0, cfg.max_width + 1, ()).item()
        start = torch.randint(0, max(1, w - width + 1), ()).item()
        x[:, start : start + width] = 0
    return x


@jaxtyped(typechecker=beartype.beartype)
def apply_random_resized_crop(
    x_hw: Float[Tensor, "h w"], cfg: RandomResizedCrop
) -> Float[Tensor, "h w"]:
    """Random crop then resize back to original size.

    Matches torchvision.transforms.v2.RandomResizedCrop.
    """
    h, w = x_hw.shape
    area = h * w

    # Try to find valid crop parameters
    for _ in range(10):
        target_area = area * (
            cfg.scale_min + torch.rand(()).item() * (cfg.scale_max - cfg.scale_min)
        )
        aspect_ratio = cfg.ratio_min + torch.rand(()).item() * (
            cfg.ratio_max - cfg.ratio_min
        )

        crop_w = int(round((target_area * aspect_ratio) ** 0.5))
        crop_h = int(round((target_area / aspect_ratio) ** 0.5))

        if 0 < crop_w <= w and 0 < crop_h <= h:
            top = torch.randint(0, h - crop_h + 1, ()).item()
            left = torch.randint(0, w - crop_w + 1, ()).item()
            cropped = x_hw[top : top + crop_h, left : left + crop_w]
            # Resize back to original size using bilinear interpolation
            resized = F.interpolate(
                cropped.unsqueeze(0).unsqueeze(0),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            return resized.squeeze(0).squeeze(0)

    # Fallback: center crop
    return x_hw


@jaxtyped(typechecker=beartype.beartype)
def apply_color_jitter(
    x_hw: Float[Tensor, "h w"], cfg: ColorJitter
) -> Float[Tensor, "h w"]:
    """Random brightness/contrast jitter for grayscale images.

    Simplified version of torchvision.transforms.v2.ColorJitter (brightness/contrast only).
    """
    # Brightness: add random offset
    if cfg.brightness > 0:
        brightness_factor = 1.0 + (torch.rand(()).item() * 2 - 1) * cfg.brightness
        x_hw = x_hw * brightness_factor

    # Contrast: adjust around mean
    if cfg.contrast > 0:
        contrast_factor = 1.0 + (torch.rand(()).item() * 2 - 1) * cfg.contrast
        mean = x_hw.mean()
        x_hw = (x_hw - mean) * contrast_factor + mean

    return x_hw


@jaxtyped(typechecker=beartype.beartype)
def apply_gaussian_blur(
    x_hw: Float[Tensor, "h w"], cfg: GaussianBlur
) -> Float[Tensor, "h w"]:
    """Apply Gaussian blur with random sigma.

    Matches torchvision.transforms.v2.GaussianBlur.
    """
    sigma = cfg.sigma_min + torch.rand(()).item() * (cfg.sigma_max - cfg.sigma_min)

    # Create 1D Gaussian kernel
    k = cfg.kernel_size
    half = k // 2
    x_coords = torch.arange(k, dtype=x_hw.dtype, device=x_hw.device) - half
    kernel_1d = torch.exp(-(x_coords**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()

    # Create 2D kernel via outer product
    kernel_2d = kernel_1d.unsqueeze(1) @ kernel_1d.unsqueeze(0)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)

    # Apply convolution with padding
    x_4d = x_hw.unsqueeze(0).unsqueeze(0)
    blurred = F.conv2d(x_4d, kernel_2d, padding=half)
    return blurred.squeeze(0).squeeze(0)


@jaxtyped(typechecker=beartype.beartype)
def apply_solarize(x_hw: Float[Tensor, "h w"], cfg: Solarize) -> Float[Tensor, "h w"]:
    """Invert pixels above threshold with probability p.

    Matches torchvision.transforms.v2.RandomSolarize. Threshold is in [0, 1] for normalized inputs.
    """
    if torch.rand(()).item() < cfg.p:
        # Assuming input is normalized to [0, 1]
        mask = x_hw > cfg.threshold
        x_hw = x_hw.clone()
        x_hw[mask] = 1.0 - x_hw[mask]
    return x_hw


@beartype.beartype
def apply(x_hw: Float[Tensor, "h w"], augmentations: list[Config]) -> Tensor:
    """Apply a list of augmentations to a 2D tensor."""
    for aug in augmentations:
        if isinstance(aug, RandomCrop):
            x_hw = apply_random_crop(x_hw, aug)
        elif isinstance(aug, HorizontalFlip):
            x_hw = apply_horizontal_flip(x_hw, aug)
        elif isinstance(aug, GaussianNoise):
            x_hw = apply_gaussian_noise(x_hw, aug)
        elif isinstance(aug, FreqMask):
            x_hw = apply_freq_mask(x_hw, aug)
        elif isinstance(aug, TimeMask):
            x_hw = apply_time_mask(x_hw, aug)
        elif isinstance(aug, RandomResizedCrop):
            x_hw = apply_random_resized_crop(x_hw, aug)
        elif isinstance(aug, ColorJitter):
            x_hw = apply_color_jitter(x_hw, aug)
        elif isinstance(aug, GaussianBlur):
            x_hw = apply_gaussian_blur(x_hw, aug)
        elif isinstance(aug, Solarize):
            x_hw = apply_solarize(x_hw, aug)
        else:
            raise ValueError(f"Unknown augmentation: {type(aug)}")
    return x_hw
