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
def test_indexed_xc_getitem_with_augmentations(augmentations):
    """IndexedXenoCantoDataset.__getitem__ works with various augmentation combos."""
    cfg = birdjepa.data.XenoCanto(subset="XCM", augmentations=augmentations)
    ds = birdjepa.data.IndexedXenoCantoDataset(cfg)

    sample = ds[0]

    assert "data" in sample
    assert "target" in sample
    assert "label" in sample
    assert "index" in sample
    # Spectrogram shape from bird_mae.transform
    assert len(sample["data"].shape) == 2
    assert isinstance(sample["target"], int)
    assert sample["index"] == 0


def test_rust_xc_loader_respects_n_samples():
    """RustXenoCantoLoader should respect XenoCanto.n_samples."""
    cfg = birdjepa.data.XenoCanto(subset="XCM", split="train", n_samples=1)
    loader = birdjepa.data.RustXenoCantoLoader(
        cfg,
        seed=0,
        batch_size=2,
        n_workers=1,
        shuffle_buffer_size=32,
        shuffle_min_size=0,
        infinite=False,
    )

    assert len(loader) == cfg.n_samples


def test_rust_xc_loader_counts_n_samples():
    """RustXenoCantoLoader should yield exactly n_samples items."""
    cfg = birdjepa.data.XenoCanto(subset="XCM", split="train", n_samples=5)
    loader = birdjepa.data.RustXenoCantoLoader(
        cfg,
        seed=0,
        batch_size=2,
        n_workers=1,
        shuffle_buffer_size=32,
        shuffle_min_size=0,
        infinite=False,
    )

    n_seen = 0
    for batch in loader:
        n_seen += batch["index"].shape[0]

    assert n_seen == cfg.n_samples


def test_rust_xc_loader_counts_small_n_samples():
    """RustXenoCantoLoader should handle n_samples < batch_size."""
    cfg = birdjepa.data.XenoCanto(subset="XCM", split="train", n_samples=1)
    loader = birdjepa.data.RustXenoCantoLoader(
        cfg,
        seed=0,
        batch_size=8,
        n_workers=1,
        shuffle_buffer_size=32,
        shuffle_min_size=0,
        infinite=False,
    )

    n_seen = 0
    for batch in loader:
        n_seen += batch["index"].shape[0]

    assert n_seen == cfg.n_samples


def test_rust_xc_loader_requires_n_samples_with_infinite_false():
    """RustXenoCantoLoader rejects infinite=False when n_samples is None."""
    cfg = birdjepa.data.XenoCanto(subset="XCM", split="train", n_samples=None)
    with pytest.raises(AssertionError, match="n_samples=None requires infinite=True"):
        birdjepa.data.RustXenoCantoLoader(
            cfg,
            seed=0,
            batch_size=2,
            n_workers=1,
            shuffle_buffer_size=32,
            shuffle_min_size=0,
            infinite=False,
        )


def test_rust_xc_loader_rejects_n_samples_with_infinite_true():
    """RustXenoCantoLoader rejects n_samples when infinite=True."""
    cfg = birdjepa.data.XenoCanto(subset="XCM", split="train", n_samples=1)
    with pytest.raises(AssertionError, match="n_samples requires infinite=False"):
        birdjepa.data.RustXenoCantoLoader(
            cfg,
            seed=0,
            batch_size=2,
            n_workers=1,
            shuffle_buffer_size=32,
            shuffle_min_size=0,
            infinite=True,
        )


def test_rust_xc_loader_rejects_zero_n_samples():
    """RustXenoCantoLoader rejects n_samples=0."""
    cfg = birdjepa.data.XenoCanto(subset="XCM", split="train", n_samples=0)
    with pytest.raises(AssertionError, match="n_samples must be > 0"):
        birdjepa.data.RustXenoCantoLoader(
            cfg,
            seed=0,
            batch_size=2,
            n_workers=1,
            shuffle_buffer_size=32,
            shuffle_min_size=0,
            infinite=False,
        )
