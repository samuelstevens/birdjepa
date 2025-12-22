"""Data loading for BirdJEPA pretraining."""

import dataclasses
import typing

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import Tensor

import birdjepa.augment

# Spectrogram config (matches Bird-MAE)
SR_HZ = 32_000
TARGET_T = 512
N_MELS = 128
MEAN = -7.2
STD = 4.43


@dataclasses.dataclass(frozen=True)
class XenoCanto:
    """Configuration for XenoCanto dataset."""

    subset: typing.Literal["XCM", "XCL"] = "XCM"
    """XCM (90k samples, 409 species) or XCL (530k samples, 10k species)."""
    split: str = "train"
    """Dataset split (XCM/XCL only have 'train')."""
    clip_sec: float = 5.0
    """Clip length in seconds."""
    truncate: typing.Literal["random", "start", "end"] = "random"
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

    split: typing.Literal["train", "test"] = "train"
    """Dataset split."""
    augmentations: list[birdjepa.augment.Config] = dataclasses.field(
        default_factory=list
    )
    """List of augmentations to apply."""


Config = XenoCanto | Cifar100


def compute_spectrogram(waveform: np.ndarray, *, clip_sec: float = 5.0) -> Tensor:
    """Compute log-mel spectrogram from waveform.

    Args:
        waveform: 1D array of audio samples at 32kHz.
        clip_sec: Expected clip length in seconds.

    Returns:
        Tensor of shape (T, 128) where T depends on clip_sec.
    """
    import torchaudio.compliance.kaldi

    waveform = torch.from_numpy(waveform).to(torch.float32)
    assert waveform.ndim == 1
    n_samples = waveform.shape[0]
    max_len = int(SR_HZ * clip_sec)

    # Pad/truncate to clip_sec
    if n_samples < max_len:
        waveform = F.pad(waveform, (0, max_len - n_samples))
    else:
        waveform = waveform[:max_len]

    # Mean-center
    waveform = waveform - waveform.mean()

    # Kaldi fbank
    fb = torchaudio.compliance.kaldi.fbank(
        waveform[None, :],
        htk_compat=True,
        sample_frequency=SR_HZ,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=N_MELS,
        dither=0.0,
        frame_shift=10.0,
    )  # [T, 128]

    # Target frames: 512 for 5s at 10ms frame shift
    target_t = int(clip_sec * 100)
    t = fb.shape[0]
    if t < target_t:
        min_val = fb.min()
        fb = F.pad(fb, (0, 0, 0, target_t - t), value=min_val.item())
    elif t > target_t:
        fb = fb[:target_t]

    # Normalize
    fb = (fb - MEAN) / (STD * 2.0)

    assert fb.shape == (target_t, N_MELS), fb.shape
    return fb


class XenoCantoDataset(torch.utils.data.Dataset):
    """XenoCanto dataset for pretraining.

    Loads audio from HuggingFace BirdSet, takes crops, computes spectrograms on-the-fly.
    """

    def __init__(self, cfg: XenoCanto):
        import datasets

        self.cfg = cfg
        self.ds = datasets.load_dataset(
            "samuelstevens/BirdSet", cfg.subset, split=cfg.split
        )
        self.rng = np.random.default_rng(cfg.seed)
        self.max_samples = int(SR_HZ * cfg.clip_sec)
        # Get class labels
        self._class_labels = self.ds.features["ebird_code"]
        self.n_classes = self._class_labels.num_classes

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
        sr = audio["sampling_rate"]

        # Resample if needed
        if sr != SR_HZ:
            import torchaudio.functional

            waveform = torch.from_numpy(waveform)
            waveform = torchaudio.functional.resample(waveform, sr, SR_HZ)
            waveform = waveform.numpy()

        # Truncate or pad to clip_sec
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

        spec = compute_spectrogram(waveform, clip_sec=self.cfg.clip_sec)
        spec = birdjepa.augment.apply(spec, self.cfg.augmentations)
        target = item["ebird_code"]
        label = self._class_labels.int2str(target)
        return {"data": spec, "label": label, "target": target}


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


class Cifar100Dataset(torch.utils.data.Dataset):
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
        # Convert to grayscale tensor: (H, W)
        img = torch.from_numpy(np.array(img)).float()  # (32, 32, 3)
        # RGB to grayscale: 0.299*R + 0.587*G + 0.114*B
        img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        # Normalize to [0, 1] then standardize
        img = img / 255.0
        img = (img - CIFAR_MEAN) / CIFAR_STD
        img = birdjepa.augment.apply(img, self.cfg.augmentations)
        label = CIFAR100_CLASSES[target]
        return {"data": img, "label": label, "target": target}
