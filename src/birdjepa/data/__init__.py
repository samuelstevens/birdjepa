"""Data loading for BirdJEPA pretraining."""

import dataclasses
import logging
import typing as tp

import beartype
import grain.python as grain
import numpy as np
import tyro.conf

import birdjepa.augment
from birdjepa.nn import bird_mae

SR_HZ = bird_mae.BIRDMAE_SR_HZ

logger = logging.getLogger(__name__)

BAD_XCL_INDICES = {
    495640,
    495763,
    497541,
    498645,
    501254,
}


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class XenoCanto:
    """Configuration for XenoCanto dataset."""

    key: tyro.conf.Suppress[tp.Literal["xeno-canto"]] = "xeno-canto"
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

    key: tyro.conf.Suppress[tp.Literal["cifar100"]] = "cifar100"
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

    def __getitem__(self, idx: int) -> dict: ...


class IndexedXenoCantoDataset(grain.RandomAccessDataSource):
    """XenoCanto dataset using random access (slow on NFS).

    Loads audio from HuggingFace BirdSet, takes crops, computes spectrograms on-the-fly.
    Use ShuffledXenoCantoDataset for NFS-friendly sequential reading.
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

        exclude_indices: set[int] = set()
        if cfg.subset == "XCL" and cfg.split == "train":
            exclude_indices.update(BAD_XCL_INDICES)

        if exclude_indices:
            n_total = len(self.ds)
            msg = f"exclude_indices has out-of-range indices (0..{n_total - 1})"
            assert min(exclude_indices) >= 0 and max(exclude_indices) < n_total, msg
            allowed_indices = np.array(
                [i for i in range(n_total) if i not in exclude_indices],
                dtype=np.int64,
            )
            n_excluded = len(exclude_indices)
            if n_excluded < 10:
                logger.info(
                    "Excluding %d/%d samples from %s %s: %s",
                    n_excluded,
                    n_total,
                    cfg.subset,
                    cfg.split,
                    sorted(exclude_indices),
                )
            else:
                logger.info(
                    "Excluding %d/%d samples from %s %s",
                    n_excluded,
                    n_total,
                    cfg.subset,
                    cfg.split,
                )
        else:
            allowed_indices = None

        # Fixed random subset if n_samples specified
        if cfg.n_samples is not None:
            subset_rng = np.random.default_rng(cfg.seed)
            if allowed_indices is None:
                allowed_indices = np.arange(len(self.ds), dtype=np.int64)
            n = min(cfg.n_samples, len(allowed_indices))
            self._indices = subset_rng.choice(allowed_indices, size=n, replace=False)
        else:
            self._indices = allowed_indices if allowed_indices is not None else None

    def __len__(self) -> int:
        if self._indices is not None:
            return len(self._indices)
        return len(self.ds)

    _timing_count: int = 0
    _timing_hf: float = 0.0
    _timing_spec: float = 0.0

    def __getitem__(self, idx: int) -> dict:
        import time

        if self._indices is not None:
            idx = int(self._indices[idx])

        t0 = time.perf_counter()
        item = self.ds[idx]
        audio = item["audio"]
        waveform = np.array(audio["array"], dtype=np.float32)
        t1 = time.perf_counter()

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
        t2 = time.perf_counter()

        # Accumulate timing stats
        IndexedXenoCantoDataset._timing_count += 1
        IndexedXenoCantoDataset._timing_hf += t1 - t0
        IndexedXenoCantoDataset._timing_spec += t2 - t1
        if IndexedXenoCantoDataset._timing_count % 100 == 0:
            n = IndexedXenoCantoDataset._timing_count
            hf_avg = IndexedXenoCantoDataset._timing_hf / n * 1000
            spec_avg = IndexedXenoCantoDataset._timing_spec / n * 1000
            print(
                f"TIMING: n={n} hf_load={hf_avg:.1f}ms spec={spec_avg:.1f}ms",
                flush=True,
            )

        target = item["ebird_code"]
        label = self._class_labels.int2str(target)
        return {"data": spec, "label": label, "target": target, "index": idx}


class ArrowTableSource:
    """Lazy accessor for Arrow table rows - avoids copying all data upfront."""

    def __init__(self, table):
        self._table = table
        self._cols = table.column_names

    def __len__(self) -> int:
        return len(self._table)

    def __getitem__(self, idx: int) -> dict:
        return {col: self._table[col][idx].as_py() for col in self._cols}


class ShuffledXenoCantoDataset:
    """XenoCanto dataset using sequential Arrow file reading (NFS-friendly).

    Uses grain's hierarchical shuffle pattern:
    1. Shuffle Arrow file paths (cheap random access on ~200 paths)
    2. Read each Arrow file sequentially
    3. Interleave multiple shards for mixing
    4. Window shuffle buffer for local randomness
    """

    def __init__(self, cfg: XenoCanto):
        import datasets

        self.cfg = cfg

        # Load dataset to get Arrow file paths (HF handles download/validation)
        ds = datasets.load_dataset("samuelstevens/BirdSet", cfg.subset, split=cfg.split)
        self.arrow_fpaths = [f["filename"] for f in ds.cache_files]
        assert self.arrow_fpaths, f"No Arrow files found for {cfg.subset}/{cfg.split}"

        # Get class labels
        self._class_labels = ds.features["ebird_code"]
        self.n_classes = self._class_labels.num_classes

        # Audio feature for decoding
        self._audio_feature = datasets.Audio(sampling_rate=SR_HZ)
        self.max_samples = int(SR_HZ * cfg.clip_sec)

        # Bad indices to filter (per shard filtering happens in decode)
        self._bad_indices = (
            BAD_XCL_INDICES if cfg.subset == "XCL" and cfg.split == "train" else set()
        )

        # RNG for truncation
        self._rng_seed = cfg.seed

        # Store dataset size for __len__ (needed for logging)
        self._n_samples = len(ds)

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> dict:
        raise NotImplementedError(
            "ShuffledXenoCantoDataset doesn't support random access. "
            "Use make_shuffled_dataloader() for iteration."
        )

    def make_shard_iter(self, arrow_fpath: str) -> grain.IterDataset:
        """Load one Arrow shard as IterDataset (sequential read)."""
        import pyarrow as pa

        # Read Arrow file sequentially (NFS-friendly)
        # HuggingFace uses IPC stream format, not file format
        mmap = pa.memory_map(arrow_fpath, "r")
        reader = pa.ipc.open_stream(mmap)
        table = reader.read_all()

        # Wrap table in a lazy accessor that converts rows on demand
        # (to_pylist() is slow because it copies all audio bytes upfront)
        source = ArrowTableSource(table)

        # Wrap as MapDataset then convert to lazy IterDataset
        return grain.MapDataset.source(source).to_iter_dataset()

    def decode_and_transform(self, row: dict) -> dict | None:
        """Decode audio bytes and compute spectrogram."""
        # Filter bad indices
        idx = row.get("__index_level_0__", -1)
        if idx in self._bad_indices:
            return None

        # Decode audio using HF Audio feature
        decoded = self._audio_feature.decode_example(row["audio"])
        waveform = np.array(decoded["array"], dtype=np.float32)

        # Truncate or pad to clip_sec
        n_samples = len(waveform)
        rng = np.random.default_rng(self._rng_seed + idx)
        if n_samples > self.max_samples:
            if self.cfg.truncate == "random":
                start = rng.integers(0, n_samples - self.max_samples)
            elif self.cfg.truncate == "start":
                start = 0
            else:  # end
                start = n_samples - self.max_samples
            waveform = waveform[start : start + self.max_samples]
        elif n_samples < self.max_samples:
            waveform = np.pad(waveform, (0, self.max_samples - n_samples))

        spec = bird_mae.transform(waveform)
        spec = birdjepa.augment.apply(spec, self.cfg.augmentations)

        target = row["ebird_code"]
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
def make_shuffled_dataloader(
    source: ShuffledXenoCantoDataset,
    *,
    seed: int,
    batch_size: int,
    n_workers: int,
    shuffle: bool,
    drop_last: bool,
    repeat: bool = False,
    shard_index: int = 0,
    shard_count: int = 1,
    window_size: int = 8000,
    cycle_length: int = 4,
):
    """Create dataloader with sequential Arrow file reading.

    Uses grain's hierarchical shuffle pattern for NFS-friendly data loading:
    1. Shuffle Arrow file paths (cheap random access on ~200 paths)
    2. Read each file sequentially via InterleaveIterDataset
    3. Window shuffle buffer for local randomness

    Args:
        source: ShuffledXenoCantoDataset with Arrow file paths.
        seed: Random seed for shuffling.
        batch_size: Batch size.
        n_workers: Number of prefetch workers.
        shuffle: Whether to shuffle.
        drop_last: Whether to drop the last incomplete batch.
        repeat: Whether to repeat indefinitely.
        shard_index: Index of this shard (0-indexed). Use jax.process_index().
        shard_count: Total number of shards. Use jax.process_count().
        window_size: Shuffle buffer size for local randomness (default 8000).
        cycle_length: Number of shards to interleave concurrently (default 4).

    Returns:
        Grain IterDataset for iteration.
    """
    # Shard the file list across processes
    arrow_fpaths = source.arrow_fpaths
    if shard_count > 1:
        arrow_fpaths = arrow_fpaths[shard_index::shard_count]

    # Build grain pipeline: shuffle shard order
    ds = grain.MapDataset.source(arrow_fpaths).seed(seed)
    if shuffle:
        ds = ds.shuffle()
    if repeat:
        ds = ds.repeat()

    # Map to IterDatasets (lazy loading per shard)
    ds = ds.map(source.make_shard_iter)

    # Interleave shards for mixing (read from multiple shards concurrently)
    ds = grain.experimental.InterleaveIterDataset(
        ds,
        cycle_length=cycle_length,
        iter_buffer_size=2,
    )

    # Window shuffle for local randomness (after interleaving)
    if shuffle and window_size > 0:
        ds = grain.experimental.WindowShuffleIterDataset(
            ds,
            window_size=window_size,
            seed=seed,
        )

    # Decode audio and transform to spectrogram, filtering bad samples
    ds = ds.map(source.decode_and_transform)
    ds = ds.filter(lambda x: x is not None)

    # Batch
    ds = ds.batch(batch_size=batch_size, drop_remainder=drop_last)

    # Multiprocess prefetch
    if n_workers > 0:
        ds = ds.mp_prefetch(
            grain.MultiprocessingOptions(
                num_workers=n_workers, per_worker_buffer_size=2
            )
        )

    return ds
