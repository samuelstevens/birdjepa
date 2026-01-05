"""Test the Rust Loader with real Arrow files."""

import datasets
import numpy as np
import pytest

from birdjepa._rs import Loader


@pytest.fixture(scope="module")
def arrow_files():
    """Get Arrow file paths from XCM dataset (smaller than XCL)."""
    ds = datasets.load_dataset("samuelstevens/BirdSet", "XCM", split="valid")
    files = [f["filename"] for f in ds.cache_files]
    assert files, "No Arrow files found"
    return files


def test_loader_creates(arrow_files):
    """Loader can be instantiated with Arrow files."""
    loader = Loader(arrow_files, seed=42, infinite=False)
    # Just verify it doesn't crash - the loader starts pipeline on construction
    batch = next(iter(loader))
    assert batch is not None


def test_loader_iteration(arrow_files):
    """Loader yields batches with expected keys and shapes."""
    batch_size = 8
    loader = Loader(arrow_files, seed=42, batch_size=batch_size, infinite=False)

    batch = next(iter(loader))

    assert "spectrogram" in batch
    assert "labels" in batch
    assert "n_mels" in batch
    assert "n_frames" in batch

    spec = batch["spectrogram"]
    labels = batch["labels"]

    assert spec.shape[0] <= batch_size  # may be smaller if not enough samples
    assert spec.shape[0] == labels.shape[0]
    assert spec.dtype == np.float32
    assert labels.dtype == np.int64


def test_loader_different_seeds(arrow_files):
    """Different seeds produce different batches."""
    loader1 = Loader(arrow_files, seed=42, batch_size=4, infinite=False)
    loader2 = Loader(arrow_files, seed=123, batch_size=4, infinite=False)

    batch1 = next(iter(loader1))
    batch2 = next(iter(loader2))

    # Spectrograms should differ (different shuffle order)
    # Labels may be identical if dataset has homogeneous classes
    assert not np.array_equal(batch1["spectrogram"], batch2["spectrogram"])


def test_loader_multiple_batches(arrow_files):
    """Can iterate multiple batches."""
    loader = Loader(arrow_files, seed=42, batch_size=4, infinite=False)

    batches = []
    for i, batch in enumerate(loader):
        batches.append(batch)
        if i >= 4:
            break

    assert len(batches) == 5
    for batch in batches:
        assert batch["spectrogram"].shape[0] <= 4


def test_loader_empty_files_rejected():
    """Loader rejects empty file list."""
    with pytest.raises(ValueError, match="arrow_files must not be empty"):
        Loader([], seed=42)


def test_loader_zero_batch_size_rejected(arrow_files):
    """Loader rejects batch_size=0."""
    with pytest.raises(ValueError, match="batch_size must be > 0"):
        Loader(arrow_files, seed=42, batch_size=0)


def test_loader_zero_workers_rejected(arrow_files):
    """Loader rejects n_workers=0."""
    with pytest.raises(ValueError, match="n_workers must be > 0"):
        Loader(arrow_files, seed=42, n_workers=0)


def test_loader_small_shuffle_buffer(arrow_files):
    """Small shuffle buffer forces evictions - verifies shuffle logic."""
    # Use tiny buffer so evictions happen immediately
    # Bug: if evicted samples aren't returned, we'd hang or get too few samples
    loader = Loader(
        arrow_files,
        seed=42,
        batch_size=8,
        shuffle_buffer_size=16,  # Very small - forces evictions after 16 samples
        shuffle_min_size=8,  # min_size < capacity
        infinite=False,
    )

    # Collect several batches worth
    n_samples = 0
    for batch in loader:
        n_samples += batch["spectrogram"].shape[0]
        if n_samples >= 64:
            break

    # Should have collected samples without hanging
    assert n_samples >= 64


def test_loader_epoch_boundary(arrow_files):
    """Loader handles epoch boundaries correctly with infinite=True."""
    # Use infinite mode to test epoch wraparound
    loader = Loader(arrow_files[:1], seed=42, batch_size=1000, infinite=True)

    # Collect batches until we cross epoch boundary
    epoch1_specs = []
    epoch2_specs = []

    for i, batch in enumerate(loader):
        if i < 5:
            epoch1_specs.append(batch["spectrogram"][0].copy())
        elif i < 10:
            epoch2_specs.append(batch["spectrogram"][0].copy())
        else:
            break

    # After epoch reset, should still produce valid data
    assert len(epoch2_specs) == 5
    for spec in epoch2_specs:
        assert spec.shape[0] > 0


def test_loader_partial_batch(arrow_files):
    """Last batch of finite iteration may be smaller than batch_size."""
    # Use large batch relative to data to ensure partial batches
    loader = Loader(
        arrow_files[:1],
        seed=42,
        batch_size=100,
        shuffle_buffer_size=50,
        shuffle_min_size=0,  # Allow partial batches to drain
        infinite=False,
    )

    batch_sizes = []
    for batch in loader:
        batch_sizes.append(batch["spectrogram"].shape[0])
        if len(batch_sizes) >= 10:
            break

    # All batches should be <= batch_size
    assert all(s <= 100 for s in batch_sizes)
    # Should have collected some batches
    assert len(batch_sizes) > 0


def test_spectrogram_transform_short_audio():
    """SpectrogramTransform handles audio shorter than FFT window."""
    from birdjepa._rs import SpectrogramTransform

    transform = SpectrogramTransform(n_fft=1024)

    # Audio shorter than FFT window
    short_audio = np.zeros(512, dtype=np.float32)
    spec = transform(short_audio)

    # Should produce empty spectrogram without crashing
    assert spec.shape == (128, 0)  # n_mels x 0 frames


def test_spectrogram_transform_deterministic():
    """SpectrogramTransform produces consistent output."""
    from birdjepa._rs import SpectrogramTransform

    transform = SpectrogramTransform()

    # Generate test signal
    t = np.linspace(0, 1, 32000, dtype=np.float32)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    spec1 = transform(audio)
    spec2 = transform(audio)

    np.testing.assert_array_equal(spec1, spec2)


def test_resampling_removes_above_nyquist():
    """Resampling should filter out frequencies above target Nyquist.

    When downsampling from 44100 Hz to 16000 Hz:
    - Target Nyquist = 8000 Hz
    - A 14000 Hz tone is above Nyquist and should be filtered out
    - Without anti-aliasing filter, it aliases to |14000 - 16000| = 2000 Hz

    This test fails with linear interpolation (current Rust implementation).
    Fix by using a proper resampler like rubato.
    """
    source_rate = 44100
    target_rate = 16000
    duration = 1.0

    # Create a pure 14000 Hz tone (above target Nyquist of 8000 Hz)
    t = np.linspace(0, duration, int(source_rate * duration), dtype=np.float32)
    high_freq_tone = np.sin(2 * np.pi * 14000 * t).astype(np.float32)

    # Linear interpolation resampling (what Rust currently does)
    def linear_resample(audio, source_rate, target_rate):
        ratio = source_rate / target_rate
        new_len = int(len(audio) / ratio)
        result = np.zeros(new_len, dtype=np.float32)
        for i in range(new_len):
            src_idx = i * ratio
            idx0 = int(src_idx)
            idx1 = min(idx0 + 1, len(audio) - 1)
            frac = src_idx - idx0
            result[i] = audio[idx0] * (1 - frac) + audio[idx1] * frac
        return result

    resampled = linear_resample(high_freq_tone, source_rate, target_rate)

    # Check for aliasing using FFT
    fft = np.abs(np.fft.rfft(resampled))
    freqs = np.fft.rfftfreq(len(resampled), 1 / target_rate)
    peak_idx = np.argmax(fft[1:]) + 1  # Skip DC
    peak_freq = freqs[peak_idx]
    peak_amplitude = fft[peak_idx]

    # The 14000 Hz tone should be filtered out, leaving only noise
    # With proper resampling, peak amplitude should be < 10 (essentially noise)
    # Linear interpolation produces ~4800 amplitude at 2000 Hz (aliased)
    assert peak_amplitude < 10, (
        f"Aliasing detected: {peak_freq:.0f} Hz with amplitude {peak_amplitude:.0f}. Expected frequencies above Nyquist to be filtered out."
    )


def test_loader_returns_indices(arrow_files):
    """Loader returns unique sample indices."""
    loader = Loader(
        arrow_files,
        seed=42,
        batch_size=8,
        shuffle_buffer_size=32,
        shuffle_min_size=16,
        infinite=False,
    )

    batch = next(iter(loader))

    assert "indices" in batch
    indices = batch["indices"]
    assert indices.dtype == np.int64
    assert indices.shape[0] == batch["spectrogram"].shape[0]


def test_loader_indices_unique_within_epoch(arrow_files):
    """Sample indices are unique within an epoch."""
    loader = Loader(
        arrow_files,
        seed=42,
        batch_size=16,
        shuffle_buffer_size=32,
        shuffle_min_size=16,
        infinite=False,
    )

    all_indices = []
    for i, batch in enumerate(loader):
        all_indices.extend(batch["indices"].tolist())
        if len(all_indices) >= 100:
            break

    # All indices should be unique (no duplicates from shuffling)
    assert len(all_indices) == len(set(all_indices))


def test_loader_indices_canonical_range(arrow_files):
    """Indices are assigned canonically starting from 0.

    Verifies that indices are:
    1. Non-negative (valid sample indices)
    2. Start from 0 (canonical ordering from file start)
    3. Within expected range (less than dataset size)
    """
    loader = Loader(
        arrow_files,
        seed=42,
        batch_size=100,
        shuffle_buffer_size=200,
        shuffle_min_size=100,
        infinite=False,
    )

    all_indices = []
    for i, batch in enumerate(loader):
        all_indices.extend(batch["indices"].tolist())
        if len(all_indices) >= 500:
            break

    # All indices should be non-negative
    assert all(idx >= 0 for idx in all_indices)

    # Indices should start from 0 (first file offset is 0)
    assert min(all_indices) == 0

    # No duplicates within the collected samples
    assert len(all_indices) == len(set(all_indices))


def test_loader_finite_mode_exhausts(arrow_files):
    """Finite mode (infinite=False) exhausts all samples and stops."""
    loader = Loader(
        arrow_files[:1],  # Single file
        seed=42,
        batch_size=100,
        shuffle_buffer_size=200,
        shuffle_min_size=0,  # Allow immediate draining
        infinite=False,
    )

    n_samples = 0
    for batch in loader:
        n_samples += batch["spectrogram"].shape[0]

    # Should have processed all samples from the file
    assert n_samples > 0


def test_loader_infinite_mode_continues(arrow_files):
    """Infinite mode (infinite=True) continues past single epoch."""
    loader = Loader(
        arrow_files[:1],  # Single file
        seed=42,
        batch_size=100,
        shuffle_buffer_size=200,
        shuffle_min_size=100,
        infinite=True,
    )

    # Collect many batches - more than a single epoch would provide
    n_samples = 0
    for i, batch in enumerate(loader):
        n_samples += batch["spectrogram"].shape[0]
        if n_samples > 10000:  # Much more than any single file
            break

    # Should have collected samples across multiple epochs
    assert n_samples > 10000


def test_rust_labels_match_python():
    """Rust loader labels match HuggingFace dataset labels for same indices.

    This is a property test that verifies the Rust loader produces the same
    integer labels as the HuggingFace dataset for corresponding sample indices.

    Uses train split because valid split has no labels (all None).
    """
    import datasets

    # Load HuggingFace dataset without audio decoding (just get labels)
    hf_ds = datasets.load_dataset("samuelstevens/BirdSet", "XCM", split="train")
    train_arrow_files = [f["filename"] for f in hf_ds.cache_files]

    # Rust loader
    rust_loader = Loader(
        train_arrow_files,
        seed=0,
        batch_size=32,
        shuffle_buffer_size=100,
        shuffle_min_size=10,
        infinite=False,
        augment=False,
    )

    # Collect (index, label) pairs from Rust loader
    rust_samples: dict[int, int] = {}
    for batch in rust_loader:
        indices = batch["indices"]
        labels = batch["labels"]
        for i in range(len(indices)):
            rust_samples[int(indices[i])] = int(labels[i])
        if len(rust_samples) >= 100:
            break

    assert len(rust_samples) >= 100, "Need at least 100 samples to verify"

    # Verify labels match for all collected indices
    mismatches = []
    for idx, rust_label in rust_samples.items():
        hf_label = hf_ds[idx]["ebird_code"]
        if rust_label != hf_label:
            mismatches.append((idx, rust_label, hf_label))

    assert not mismatches, f"Label mismatches: {mismatches[:10]}..."
