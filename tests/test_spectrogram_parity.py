"""Test parity between Rust and Python spectrogram implementations.

The Rust SpectrogramTransform should produce identical output to kaldi_fbank.
"""

import numpy as np
import pytest

from birdjepa._rs import SpectrogramTransform
from birdjepa.nn import bird_mae

# Tolerances matching test_bird_mae.py
RTOL = 1e-4
ATOL = 1e-5


@pytest.mark.xfail(
    reason="Rust uses standard mel filterbank; kaldi_fbank uses Kaldi-specific parameters (preemphasis, dithering, window function). Fix by matching Kaldi exactly."
)
def test_spectrogram_parity():
    """Rust spectrogram must match kaldi_fbank exactly."""
    sr = 32000
    n_mels = 128
    frame_shift = 10.0  # ms

    # Random audio
    samples = np.random.randn(sr).astype(np.float32)

    # Python reference
    py_spec = bird_mae.kaldi_fbank(
        samples,
        sample_frequency=sr,
        num_mel_bins=n_mels,
        frame_shift=frame_shift,
        dither=0.0,
    )

    # Rust (should match kaldi_fbank parameters)
    # kaldi_fbank uses: frame_length=25ms, frame_shift=10ms, preemphasis=0.97
    # At 32kHz: window=800 samples (padded to 1024), hop=320 samples
    rust_transform = SpectrogramTransform(
        sample_rate=sr,
        n_fft=1024,
        hop_length=320,
        n_mels=n_mels,
    )
    rust_spec = rust_transform(samples)

    # kaldi_fbank returns [time, mels], rust returns [mels, time]
    rust_spec_t = rust_spec.T

    # Shapes must match exactly
    assert rust_spec_t.shape == py_spec.shape, (
        f"Shape mismatch: {rust_spec_t.shape} vs {py_spec.shape}"
    )

    # Values must match within tolerance
    np.testing.assert_allclose(rust_spec_t, py_spec, rtol=RTOL, atol=ATOL)
