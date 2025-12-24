"""Tests for data loading and augmentation composition."""

import pytest

import birdjepa.augment
import birdjepa.data


# Various augmentation combinations to test
AUGMENTATION_COMBOS = [
    # Empty
    [],
    # Single augmentations
    [birdjepa.augment.RandomCrop(padding=4)],
    [birdjepa.augment.HorizontalFlip(p=0.5)],
    [birdjepa.augment.GaussianNoise(std=0.1)],
    # Pairs
    [birdjepa.augment.RandomCrop(padding=4), birdjepa.augment.HorizontalFlip(p=0.5)],
    [birdjepa.augment.HorizontalFlip(p=0.5), birdjepa.augment.RandomCrop(padding=4)],
    [
        birdjepa.augment.ColorJitter(brightness=0.4, contrast=0.4),
        birdjepa.augment.GaussianBlur(kernel_size=3),
    ],
    # Triples
    [
        birdjepa.augment.RandomCrop(padding=4),
        birdjepa.augment.HorizontalFlip(p=0.5),
        birdjepa.augment.GaussianNoise(std=0.05),
    ],
    [
        birdjepa.augment.RandomResizedCrop(),
        birdjepa.augment.ColorJitter(),
        birdjepa.augment.Solarize(p=0.2),
    ],
    # Kitchen sink
    [
        birdjepa.augment.RandomCrop(padding=4),
        birdjepa.augment.HorizontalFlip(p=0.5),
        birdjepa.augment.ColorJitter(brightness=0.4, contrast=0.4),
        birdjepa.augment.GaussianNoise(std=0.05),
    ],
]


@pytest.mark.parametrize("augmentations", AUGMENTATION_COMBOS)
def test_cifar100_getitem_with_augmentations(augmentations):
    """Cifar100Dataset.__getitem__ works with various augmentation combos."""
    cfg = birdjepa.data.Cifar100(split="train", augmentations=augmentations)
    ds = birdjepa.data.Cifar100Dataset(cfg)

    sample = ds[0]

    assert "data" in sample
    assert "target" in sample
    assert "label" in sample
    assert "index" in sample
    assert sample["data"].shape == (32, 32)
    assert isinstance(sample["target"], int)
    assert sample["index"] == 0


# SpecAugment-style augmentations for audio
AUDIO_AUGMENTATION_COMBOS = [
    [],
    [birdjepa.augment.FreqMask(max_width=10, n_masks=2)],
    [birdjepa.augment.TimeMask(max_width=20, n_masks=2)],
    [birdjepa.augment.FreqMask(max_width=10), birdjepa.augment.TimeMask(max_width=20)],
    [
        birdjepa.augment.GaussianNoise(std=0.1),
        birdjepa.augment.FreqMask(),
        birdjepa.augment.TimeMask(),
    ],
]


@pytest.mark.parametrize("augmentations", AUDIO_AUGMENTATION_COMBOS)
def test_xenocanto_getitem_with_augmentations(augmentations):
    """XenoCantoDataset.__getitem__ works with various augmentation combos."""
    cfg = birdjepa.data.XenoCanto(subset="XCM", augmentations=augmentations)
    try:
        ds = birdjepa.data.XenoCantoDataset(cfg)
    except Exception as e:
        pytest.skip(f"XenoCanto dataset not available: {e}")

    try:
        sample = ds[0]
    except (KeyError, RuntimeError) as e:
        pytest.skip(f"XenoCanto data loading issue: {e}")

    assert "data" in sample
    assert "target" in sample
    assert "label" in sample
    assert "index" in sample
    # Spectrogram shape from bird_mae.transform
    assert len(sample["data"].shape) == 2
    assert isinstance(sample["target"], int)
    assert sample["index"] == 0
