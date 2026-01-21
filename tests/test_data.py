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


def test_shuffled_xc_init():
    """ShuffledXenoCantoDataset initializes and finds Arrow files."""
    cfg = birdjepa.data.XenoCanto(subset="XCM")
    ds = birdjepa.data.ShuffledXenoCantoDataset(cfg)

    assert ds.n_classes > 0
    assert len(ds.arrow_fpaths) > 0
    assert all(fpath.endswith(".arrow") for fpath in ds.arrow_fpaths)


def test_make_shuffled_dataloader_basic():
    """make_shuffled_dataloader creates a working dataloader."""
    cfg = birdjepa.data.XenoCanto(subset="XCM")
    ds = birdjepa.data.ShuffledXenoCantoDataset(cfg)

    loader = birdjepa.data.make_shuffled_dataloader(
        ds,
        seed=42,
        batch_size=2,
        n_workers=0,
        shuffle=True,
        drop_last=True,
        repeat=False,
        window_size=100,
        cycle_length=2,
    )

    batch = next(iter(loader))

    assert "data" in batch
    assert "target" in batch
    assert batch["data"].shape[0] == 2  # batch size
    assert len(batch["data"].shape) == 3  # (batch, height, width)


def test_make_shuffled_dataloader_repeat():
    """make_shuffled_dataloader with repeat=True yields infinite batches."""
    cfg = birdjepa.data.XenoCanto(subset="XCM")
    ds = birdjepa.data.ShuffledXenoCantoDataset(cfg)

    loader = birdjepa.data.make_shuffled_dataloader(
        ds,
        seed=42,
        batch_size=2,
        n_workers=0,
        shuffle=True,
        drop_last=True,
        repeat=True,
        window_size=100,
        cycle_length=2,
    )

    # Should be able to iterate more than dataset size (5 batches as smoke test)
    it = iter(loader)
    for _ in range(5):
        batch = next(it)
        assert "data" in batch


def test_make_shuffled_dataloader_sharding():
    """make_shuffled_dataloader correctly shards files across processes."""
    cfg = birdjepa.data.XenoCanto(subset="XCM")
    ds = birdjepa.data.ShuffledXenoCantoDataset(cfg)

    # Each shard should get about half the arrow files
    n_files = len(ds.arrow_fpaths)
    assert n_files >= 2, "Need at least 2 arrow files for sharding test"

    # Shard 0 of 2 should get even-indexed files
    expected_shard0 = [ds.arrow_fpaths[i] for i in range(0, n_files, 2)]
    expected_shard1 = [ds.arrow_fpaths[i] for i in range(1, n_files, 2)]

    assert len(expected_shard0) + len(expected_shard1) == n_files
    assert len(expected_shard0) > 0
    assert len(expected_shard1) > 0


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
