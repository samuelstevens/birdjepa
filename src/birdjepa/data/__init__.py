"""Data loading for BirdJEPA pretraining."""

import dataclasses
import typing as tp

import beartype
import grain.python as grain
import numpy as np

import birdjepa.augment
from birdjepa.nn import bird_mae

SR_HZ = bird_mae.BIRDMAE_SR_HZ


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class XenoCanto:
    """Configuration for XenoCanto dataset."""

    subset: tp.Literal["XCM", "XCL"] = "XCM"
    """XCM (90k samples, 409 species) or XCL (530k samples, 10k species)."""
    split: str = "train"
    """Dataset split (XCM/XCL only have 'train')."""
    clip_sec: float = 5.0
    """Clip length in seconds."""
    truncate: tp.Literal["random", "start", "end"] = "random"
    """How to truncate audio longer than clip_sec."""
    seed: int = 42
    """Random seed for reproducible crops."""
    n_samples: int | None = None
    """Number of samples to use (None = all). For eval subsets."""
    augmentations: list[birdjepa.augment.Config] = dataclasses.field(
        default_factory=list
    )
    """List of augmentations to apply."""


@dataclasses.dataclass(frozen=True)
class Cifar100:
    """Configuration for CIFAR-100 dataset."""

    split: tp.Literal["train", "test"] = "train"
    """Dataset split."""
    augmentations: list[birdjepa.augment.Config] = dataclasses.field(
        default_factory=list
    )
    """List of augmentations to apply."""


Config = XenoCanto | Cifar100


@tp.runtime_checkable
class Dataset(tp.Protocol):
    """Protocol for datasets."""

    n_classes: int

    def __len__(self) -> int: ...


class XenoCantoDataset(grain.RandomAccessDataSource):
    """XenoCanto dataset for pretraining.

    Loads audio from HuggingFace BirdSet, takes crops, computes spectrograms on-the-fly.
    """

    def __init__(self, cfg: XenoCanto):
        import datasets

        self.cfg = cfg
        self.ds = datasets.load_dataset(
            "samuelstevens/BirdSet", cfg.subset, split=cfg.split
        )
        # Decode audio bytes to arrays
        self.ds = self.ds.cast_column("audio", datasets.Audio(sampling_rate=SR_HZ))
        self.rng = np.random.default_rng(cfg.seed)
        self.max_samples = int(SR_HZ * cfg.clip_sec)
        # Get class labels
        self._class_labels = self.ds.features["ebird_code"]
        self.n_classes = self._class_labels.num_classes
        # Each row is a unique recording - use index as source ID
        self.n_sources = len(self.ds)

        # Fixed random subset if n_samples specified
        if cfg.n_samples is not None:
            subset_rng = np.random.default_rng(cfg.seed)
            n = min(cfg.n_samples, len(self.ds))
            self._indices = subset_rng.choice(len(self.ds), size=n, replace=False)
        else:
            self._indices = None

    def __len__(self) -> int:
        if self._indices is not None:
            return len(self._indices)
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        if self._indices is not None:
            idx = self._indices[idx]
        item = self.ds[idx]
        audio = item["audio"]
        waveform = np.array(audio["array"], dtype=np.float32)

        # Truncate or pad to clip_sec (audio already resampled by cast_column)
        n_samples = len(waveform)
        if n_samples > self.max_samples:
            if self.cfg.truncate == "random":
                start = self.rng.integers(0, n_samples - self.max_samples)
            elif self.cfg.truncate == "start":
                start = 0
            else:  # end
                start = n_samples - self.max_samples
            waveform = waveform[start : start + self.max_samples]
        elif n_samples < self.max_samples:
            waveform = np.pad(waveform, (0, self.max_samples - n_samples))

        spec = bird_mae.transform(waveform)
        spec = birdjepa.augment.apply(spec, self.cfg.augmentations)
        target = item["ebird_code"]
        label = self._class_labels.int2str(target)
        return {"data": spec, "label": label, "target": target, "index": idx}


# CIFAR-100 normalization (grayscale approximation)
CIFAR_MEAN = 0.4914
CIFAR_STD = 0.2470


# CIFAR-100 class names (fine labels)
CIFAR100_CLASSES = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm",
]


class Cifar100Dataset(grain.RandomAccessDataSource):
    """CIFAR-100 dataset for pretraining.

    Loads CIFAR-100 from torchvision, converts to grayscale.
    Returns images of shape (32, 32).
    """

    def __init__(self, cfg: Cifar100):
        import torchvision.datasets

        self.cfg = cfg
        train = cfg.split == "train"
        self.ds = torchvision.datasets.CIFAR100(
            root="./data", train=train, download=True
        )
        self.n_classes = 100

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        img, target = self.ds[idx]  # PIL Image, int
        # Convert to grayscale array: (H, W)
        img = np.array(img, dtype=np.float32)  # (32, 32, 3)
        # RGB to grayscale: 0.299*R + 0.587*G + 0.114*B
        img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        # Normalize to [0, 1] then standardize
        img = img / 255.0
        img = (img - CIFAR_MEAN) / CIFAR_STD
        img = birdjepa.augment.apply(img, self.cfg.augmentations)
        label = CIFAR100_CLASSES[target]
        return {"data": img, "label": label, "target": target, "index": idx}


@beartype.beartype
def make_dataloader(
    source: grain.RandomAccessDataSource,
    *,
    seed: int,
    batch_size: int,
    n_workers: int,
    shuffle: bool,
    drop_last: bool,
):
    """Create a Grain dataloader from a dataset source.

    Args:
        source: Dataset implementing RandomAccessDataSource.
        seed: Random seed for shuffling.
        batch_size: Batch size.
        n_workers: Number of prefetch workers.
        shuffle: Whether to shuffle.
        drop_last: Whether to drop the last incomplete batch.

    Returns:
        Grain IterDataset for iteration.
    """
    ds = grain.MapDataset.source(source).seed(seed)
    if shuffle:
        ds = ds.shuffle()
    ds = ds.batch(batch_size=batch_size, drop_remainder=drop_last)
    iter_ds = ds.to_iter_dataset(
        read_options=grain.ReadOptions(num_threads=2, prefetch_buffer_size=64)
    )
    if n_workers > 0:
        iter_ds = iter_ds.mp_prefetch(
            grain.MultiprocessingOptions(
                num_workers=n_workers, per_worker_buffer_size=2
            )
        )
    return iter_ds
