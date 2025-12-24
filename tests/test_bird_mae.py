"""
Parity tests comparing JAX bird_mae implementation to PyTorch reference.

All tests use float32 precision to match Bird-MAE's internal processing.
Tolerances (rtol=1e-4, atol=2e-3) account for:
- FFT implementation differences between JAX and torch
- Log amplification of small power spectrum differences (~1e-7 -> ~1e-3)
- Float32 accumulation errors in dot products
"""

import numpy as np
import torch
from hypothesis import given, settings, strategies as st

from birdjepa.nn import bird_mae
from birdjepa.nn import bird_mae_pt


# Unified tolerances for float32 parity (JAX FFT vs torch FFT)
# - FFT implementation differences between XLA and torch: ~1e-5 in spectrum
# - Log amplifies small differences: ~1e-5 -> ~1e-2 in low-energy bins
# - Max absolute diff observed: ~8e-3 (log-mel values range from -16 to 0)
# - Max relative diff observed: ~6e-4
RTOL = 1e-3
ATOL = 1e-2


def _to_numpy(x):
    """Convert JAX array to numpy for comparison."""
    return np.asarray(x)


# --- Hypothesis strategies ---


@st.composite
def waveform_strategy(draw, min_samples=1000, max_samples=200_000):
    """Generate random waveforms in float32."""
    n_samples = draw(st.integers(min_value=min_samples, max_value=max_samples))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)
    waveform = rng.standard_normal(n_samples).astype(np.float32)
    return waveform


@st.composite
def waveform_5s_strategy(draw):
    """Generate 5-second waveforms at 32kHz in float32."""
    n_samples = 32_000 * 5  # 160,000 samples
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)
    waveform = rng.standard_normal(n_samples).astype(np.float32)
    return waveform


# --- Transform parity tests ---


@given(waveform=waveform_5s_strategy())
@settings(max_examples=50, deadline=None)
def test_transform_parity_random_5s(waveform):
    """Test that JAX transform matches PT transform for random 5s waveforms."""
    jax_out = bird_mae.transform(waveform)
    pt_out = bird_mae_pt.transform(waveform)

    assert jax_out.shape == pt_out.shape, f"{jax_out.shape} != {pt_out.shape}"
    np.testing.assert_allclose(_to_numpy(jax_out), pt_out.numpy(), rtol=RTOL, atol=ATOL)


@given(waveform=waveform_strategy(min_samples=1000, max_samples=300_000))
@settings(max_examples=50, deadline=None)
def test_transform_parity_variable_length(waveform):
    """Test that JAX transform matches PT transform for variable length waveforms."""
    jax_out = bird_mae.transform(waveform)
    pt_out = bird_mae_pt.transform(waveform)

    assert jax_out.shape == pt_out.shape, f"{jax_out.shape} != {pt_out.shape}"
    np.testing.assert_allclose(_to_numpy(jax_out), pt_out.numpy(), rtol=RTOL, atol=ATOL)


def test_transform_parity_short_audio():
    """Test that JAX transform matches PT transform for very short audio."""
    waveform = np.random.randn(1000).astype(np.float32)

    jax_out = bird_mae.transform(waveform)
    pt_out = bird_mae_pt.transform(waveform)

    assert jax_out.shape == pt_out.shape
    np.testing.assert_allclose(_to_numpy(jax_out), pt_out.numpy(), rtol=RTOL, atol=ATOL)


def test_transform_parity_long_audio():
    """Test that JAX transform matches PT transform for audio longer than 5s."""
    waveform = np.random.randn(32_000 * 10).astype(np.float32)

    jax_out = bird_mae.transform(waveform)
    pt_out = bird_mae_pt.transform(waveform)

    assert jax_out.shape == pt_out.shape
    np.testing.assert_allclose(_to_numpy(jax_out), pt_out.numpy(), rtol=RTOL, atol=ATOL)


def test_transform_parity_zeros():
    """Test that JAX transform matches PT transform for silent audio."""
    waveform = np.zeros(32_000 * 5, dtype=np.float32)

    jax_out = bird_mae.transform(waveform)
    pt_out = bird_mae_pt.transform(waveform)

    assert jax_out.shape == pt_out.shape
    np.testing.assert_allclose(_to_numpy(jax_out), pt_out.numpy(), rtol=RTOL, atol=ATOL)


def test_transform_parity_sine_wave():
    """Test that JAX transform matches PT transform for a pure sine wave."""
    sr = 32_000
    duration = 5
    freq = 440  # A4
    t = np.linspace(0, duration, sr * duration, dtype=np.float32)
    waveform = np.sin(2 * np.pi * freq * t).astype(np.float32)

    jax_out = bird_mae.transform(waveform)
    pt_out = bird_mae_pt.transform(waveform)

    assert jax_out.shape == pt_out.shape
    np.testing.assert_allclose(_to_numpy(jax_out), pt_out.numpy(), rtol=RTOL, atol=ATOL)


# --- kaldi_fbank parity tests ---


@given(waveform=waveform_strategy(min_samples=8000, max_samples=160_000))
@settings(max_examples=50, deadline=None)
def test_kaldi_fbank_parity(waveform):
    """Test that our kaldi_fbank matches torchaudio's implementation."""
    import torchaudio.compliance.kaldi

    jax_out = bird_mae.kaldi_fbank(
        waveform,
        htk_compat=True,
        sample_frequency=32_000,
        use_energy=False,
        num_mel_bins=128,
        dither=0.0,
        frame_shift=10.0,
    )

    waveform_pt = torch.from_numpy(waveform).unsqueeze(0)
    pt_out = torchaudio.compliance.kaldi.fbank(
        waveform_pt,
        htk_compat=True,
        sample_frequency=32_000,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=128,
        dither=0.0,
        frame_shift=10.0,
    )

    assert jax_out.shape == tuple(pt_out.shape), f"{jax_out.shape} != {pt_out.shape}"
    np.testing.assert_allclose(_to_numpy(jax_out), pt_out.numpy(), rtol=RTOL, atol=ATOL)


def test_kaldi_fbank_parity_default_params():
    """Test kaldi_fbank with default parameters."""
    import torchaudio.compliance.kaldi

    waveform = np.random.randn(16000).astype(np.float32)

    jax_out = bird_mae.kaldi_fbank(waveform, dither=0.0)

    waveform_pt = torch.from_numpy(waveform).unsqueeze(0)
    pt_out = torchaudio.compliance.kaldi.fbank(
        waveform_pt,
        window_type="hanning",
        dither=0.0,
    )

    assert jax_out.shape == tuple(pt_out.shape)
    np.testing.assert_allclose(_to_numpy(jax_out), pt_out.numpy(), rtol=RTOL, atol=ATOL)


@given(
    num_mel_bins=st.sampled_from([23, 40, 64, 80, 128]),
    frame_shift=st.sampled_from([10.0, 20.0]),
    frame_length=st.sampled_from([25.0, 30.0]),
)
@settings(max_examples=20, deadline=None)
def test_kaldi_fbank_parity_various_params(num_mel_bins, frame_shift, frame_length):
    """Test kaldi_fbank with various parameter combinations."""
    import torchaudio.compliance.kaldi

    waveform = np.random.randn(32000).astype(np.float32)

    jax_out = bird_mae.kaldi_fbank(
        waveform,
        sample_frequency=32_000,
        num_mel_bins=num_mel_bins,
        frame_shift=frame_shift,
        frame_length=frame_length,
        dither=0.0,
    )

    waveform_pt = torch.from_numpy(waveform).unsqueeze(0)
    pt_out = torchaudio.compliance.kaldi.fbank(
        waveform_pt,
        sample_frequency=32_000,
        window_type="hanning",
        num_mel_bins=num_mel_bins,
        frame_shift=frame_shift,
        frame_length=frame_length,
        dither=0.0,
    )

    assert jax_out.shape == tuple(pt_out.shape), f"{jax_out.shape} != {pt_out.shape}"
    np.testing.assert_allclose(_to_numpy(jax_out), pt_out.numpy(), rtol=RTOL, atol=ATOL)


def test_kaldi_fbank_parity_sine_wave():
    """Test kaldi_fbank with a pure sine wave for deterministic comparison."""
    import torchaudio.compliance.kaldi

    sr = 32_000
    duration = 1
    freq = 1000
    t = np.linspace(0, duration, sr * duration, dtype=np.float32)
    waveform = np.sin(2 * np.pi * freq * t).astype(np.float32)

    jax_out = bird_mae.kaldi_fbank(
        waveform,
        sample_frequency=sr,
        num_mel_bins=128,
        dither=0.0,
    )

    waveform_pt = torch.from_numpy(waveform).unsqueeze(0)
    pt_out = torchaudio.compliance.kaldi.fbank(
        waveform_pt,
        sample_frequency=sr,
        window_type="hanning",
        num_mel_bins=128,
        dither=0.0,
    )

    assert jax_out.shape == tuple(pt_out.shape)
    np.testing.assert_allclose(_to_numpy(jax_out), pt_out.numpy(), rtol=RTOL, atol=ATOL)


@given(waveform=waveform_strategy(min_samples=8000, max_samples=80_000))
@settings(max_examples=20, deadline=None)
def test_kaldi_fbank_parity_use_energy(waveform):
    """Test kaldi_fbank with use_energy=True."""
    import torchaudio.compliance.kaldi

    jax_out = bird_mae.kaldi_fbank(
        waveform,
        sample_frequency=32_000,
        num_mel_bins=128,
        use_energy=True,
        htk_compat=True,
        dither=0.0,
    )

    waveform_pt = torch.from_numpy(waveform).unsqueeze(0)
    pt_out = torchaudio.compliance.kaldi.fbank(
        waveform_pt,
        sample_frequency=32_000,
        window_type="hanning",
        num_mel_bins=128,
        use_energy=True,
        htk_compat=True,
        dither=0.0,
    )

    assert jax_out.shape == tuple(pt_out.shape), f"{jax_out.shape} != {pt_out.shape}"
    np.testing.assert_allclose(_to_numpy(jax_out), pt_out.numpy(), rtol=RTOL, atol=ATOL)


def test_kaldi_fbank_parity_use_energy_htk_compat_false():
    """Test kaldi_fbank with use_energy=True and htk_compat=False (energy first)."""
    import torchaudio.compliance.kaldi

    waveform = np.random.randn(32000).astype(np.float32)

    jax_out = bird_mae.kaldi_fbank(
        waveform,
        sample_frequency=32_000,
        num_mel_bins=128,
        use_energy=True,
        htk_compat=False,
        dither=0.0,
    )

    waveform_pt = torch.from_numpy(waveform).unsqueeze(0)
    pt_out = torchaudio.compliance.kaldi.fbank(
        waveform_pt,
        sample_frequency=32_000,
        window_type="hanning",
        num_mel_bins=128,
        use_energy=True,
        htk_compat=False,
        dither=0.0,
    )

    assert jax_out.shape == tuple(pt_out.shape)
    assert jax_out.shape[1] == 129  # 128 mel bins + 1 energy
    np.testing.assert_allclose(_to_numpy(jax_out), pt_out.numpy(), rtol=RTOL, atol=ATOL)
