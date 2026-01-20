"""Test parity between Rust and Python spectrogram implementations.

The Rust SpectrogramTransform should produce identical output to kaldi_fbank.
"""

import numpy as np
from hypothesis import given, settings, strategies as st

from birdjepa._rs import SpectrogramTransform
from birdjepa.nn import bird_mae

# Tolerances - allow float32 vs float64 and minor FFT implementation differences
# These differences are negligible for audio processing (<0.1% of values differ, max diff ~0.003)
RTOL = 1e-3
ATOL = 1e-3


def test_spectrogram_parity():
    """Rust spectrogram must match kaldi_fbank exactly."""
    sr = 32000
    n_mels = 128
    frame_shift = 10.0  # ms
    frame_length = 25.0  # ms

    # Random audio (1 second)
    np.random.seed(42)
    samples = np.random.randn(sr).astype(np.float32)

    # Python reference
    py_spec = bird_mae.kaldi_fbank(
        samples,
        sample_frequency=sr,
        num_mel_bins=n_mels,
        frame_shift=frame_shift,
        frame_length=frame_length,
        dither=0.0,
        low_freq=20.0,
        preemphasis_coefficient=0.97,
    )

    # Rust with matching parameters
    # At 32kHz: window=800 samples (25ms), hop=320 samples (10ms), padded to 1024 for FFT
    rust_transform = SpectrogramTransform(
        sample_rate=sr,
        n_fft=1024,
        hop_length=320,
        n_mels=n_mels,
        f_min=20.0,
        win_length=800,
        preemphasis=0.97,
    )
    rust_spec = rust_transform(samples)

    # Both return [n_frames, n_mels]
    assert rust_spec.shape == py_spec.shape, (
        f"Shape mismatch: {rust_spec.shape} vs {py_spec.shape}"
    )

    # Values must match within tolerance
    np.testing.assert_allclose(rust_spec, py_spec, rtol=RTOL, atol=ATOL)


def test_spectrogram_parity_5s():
    """Test parity on full 5-second clip (typical training length)."""
    sr = 32000
    n_mels = 128
    duration = 5.0

    np.random.seed(123)
    samples = np.random.randn(int(sr * duration)).astype(np.float32)

    py_spec = bird_mae.kaldi_fbank(
        samples,
        sample_frequency=sr,
        num_mel_bins=n_mels,
        frame_shift=10.0,
        frame_length=25.0,
        dither=0.0,
        low_freq=20.0,
        preemphasis_coefficient=0.97,
    )

    rust_transform = SpectrogramTransform(
        sample_rate=sr,
        n_fft=1024,
        hop_length=320,
        n_mels=n_mels,
        f_min=20.0,
        win_length=800,
        preemphasis=0.97,
    )
    rust_spec = rust_transform(samples)

    assert rust_spec.shape == py_spec.shape
    np.testing.assert_allclose(rust_spec, py_spec, rtol=RTOL, atol=ATOL)


def test_spectrogram_sine_wave():
    """Test parity on simple sine wave (deterministic signal)."""
    sr = 32000
    n_mels = 128
    freq = 440.0  # A4

    t = np.arange(sr, dtype=np.float32) / sr
    samples = np.sin(2 * np.pi * freq * t).astype(np.float32)

    py_spec = bird_mae.kaldi_fbank(
        samples,
        sample_frequency=sr,
        num_mel_bins=n_mels,
        frame_shift=10.0,
        frame_length=25.0,
        dither=0.0,
        low_freq=20.0,
        preemphasis_coefficient=0.97,
    )

    rust_transform = SpectrogramTransform(
        sample_rate=sr,
        n_fft=1024,
        hop_length=320,
        n_mels=n_mels,
        f_min=20.0,
        win_length=800,
        preemphasis=0.97,
    )
    rust_spec = rust_transform(samples)

    assert rust_spec.shape == py_spec.shape
    np.testing.assert_allclose(rust_spec, py_spec, rtol=RTOL, atol=ATOL)


@given(
    duration=st.floats(min_value=0.1, max_value=10.0),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=100, deadline=None)
def test_spectrogram_parity_property(duration: float, seed: int):
    """Property test: Rust and Python spectrograms match for random waveforms."""
    sr = 32000
    n_mels = 128
    n_samples = int(sr * duration)

    # Generate random waveform
    rng = np.random.default_rng(seed)
    samples = rng.standard_normal(n_samples).astype(np.float32)

    # Python reference
    py_spec = bird_mae.kaldi_fbank(
        samples,
        sample_frequency=sr,
        num_mel_bins=n_mels,
        frame_shift=10.0,
        frame_length=25.0,
        dither=0.0,
        low_freq=20.0,
        preemphasis_coefficient=0.97,
    )

    # Rust
    rust_transform = SpectrogramTransform(
        sample_rate=sr,
        n_fft=1024,
        hop_length=320,
        n_mels=n_mels,
        f_min=20.0,
        win_length=800,
        preemphasis=0.97,
    )
    rust_spec = rust_transform(samples)

    assert rust_spec.shape == py_spec.shape, (
        f"Shape mismatch: {rust_spec.shape} vs {py_spec.shape}"
    )
    np.testing.assert_allclose(rust_spec, py_spec, rtol=RTOL, atol=ATOL)
