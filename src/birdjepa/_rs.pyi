from __future__ import annotations

import typing as tp

import numpy as np

class Loader:
    def __init__(
        self,
        arrow_files: list[str],
        seed: int = 0,
        batch_size: int = 64,
        shuffle_buffer_size: int = 10000,
        shuffle_min_size: int = 8000,
        sample_rate: int = 32000,
        clip_seconds: float = 5.0,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 320,
        win_length: int = 800,
        f_min: float = 20.0,
        preemphasis: float = 0.97,
        n_workers: int = 8,
        raw_channel_size: int = 256,
        infinite: bool = True,
        augment: bool = True,
        n_samples: int | None = None,
    ) -> None: ...
    def __iter__(self) -> Loader: ...
    def __next__(self) -> dict[str, tp.Any]: ...  # includes 'shard_ids'
    def diagnostics(self) -> dict[str, int]: ...

class SpectrogramTransform:
    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        f_min: float,
        f_max: float,
        win_length: int,
        preemphasis: float,
    ) -> None: ...

def decode_audio(bytes: bytes) -> np.ndarray: ...
def build_profile() -> str: ...
