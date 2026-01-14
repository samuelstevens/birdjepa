"""Tests for Perch audio embedding model.

Tests the PerchBackbone inference and transform function.
"""

import os
import numpy as np
import pytest
from birdjepa.nn import perch
from birdjepa.benchmark.registry import load
import jax.numpy as jnp
import jax.random as jr

# =============== #
# Transform Tests #
# =============== #


def test_perch_transform_shape():
    """Perch transform produces correct spectrogram shape."""

    # 5 seconds at 32kHz
    waveform = np.random.randn(perch.PERCH_TARGET_SAMPLES).astype(np.float32)
    spec = perch.transform(waveform)

    # Should be [~500, 128]
    assert spec.shape == (perch.PERCH_TARGET_T, perch.PERCH_N_MELS)
    assert spec.dtype == np.float32


def test_perch_transform_short_audio():
    """Transform pads short audio to target length."""

    # 1 second (shorter than 5s target)
    waveform = np.random.randn(perch.PERCH_SR_HZ).astype(np.float32)
    spec = perch.transform(waveform)

    # Should still produce correct shape
    assert spec.shape == (perch.PERCH_TARGET_T, perch.PERCH_N_MELS)


def test_perch_transform_long_audio():
    """Transform truncates long audio to target length."""

    # 10 seconds (longer than 5s target)
    waveform = np.random.randn(perch.PERCH_SR_HZ * 10).astype(np.float32)
    spec = perch.transform(waveform)

    # Should still produce correct shape
    assert spec.shape == (perch.PERCH_TARGET_T, perch.PERCH_N_MELS)


def test_perch_transform_deterministic():
    """Transform is deterministic for same input."""

    np.random.seed(42)
    waveform = np.random.randn(perch.PERCH_TARGET_SAMPLES).astype(np.float32)

    spec1 = perch.transform(waveform)
    spec2 = perch.transform(waveform)

    np.testing.assert_array_equal(spec1, spec2)


def test_perch_transform_sine_wave():
    """Transform produces expected pattern for sine wave."""

    # 440 Hz sine wave (A4)
    t = np.arange(perch.PERCH_TARGET_SAMPLES) / perch.PERCH_SR_HZ
    waveform = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    spec = perch.transform(waveform)

    assert spec.shape == (perch.PERCH_TARGET_T, perch.PERCH_N_MELS)
    # Sine wave should have energy concentrated in specific mel bands
    assert spec.max() > spec.min()  # Not constant


# =============================================================================
# EfficientNet Architecture Tests
# =============================================================================


def test_efficientnet_architecture():
    """EfficientNet-B3 has correct output dimension."""

    model = perch.EfficientNet(key=jr.key(0))
    assert model.output_dim == perch.EFFICIENTNET_B3_OUTPUT_DIM  # 1536


def test_efficientnet_forward_pass():
    """EfficientNet forward pass produces correct shape."""

    model = perch.EfficientNet(key=jr.key(0))

    # Batch of spectrograms [B, 1, T, M]
    batch_size = 2
    x = jnp.ones((batch_size, 1, perch.PERCH_TARGET_T, perch.PERCH_N_MELS))
    out = model(x)

    assert out.shape == (batch_size, perch.EFFICIENTNET_B3_OUTPUT_DIM)


# =============================================================================
# PerchBackbone Integration Tests
# =============================================================================


@pytest.mark.slow
def test_perch_backbone_inference():
    """PerchBackbone produces embeddings with correct shape."""

    backbone = load("perch", "perch_v2")

    # Create batch of waveforms
    batch_size = 2
    batch = np.random.randn(batch_size, 160_000).astype(np.float32)

    embeddings = backbone.encode(batch)

    assert embeddings.shape == (batch_size, 1536)


@pytest.mark.slow
def test_perch_backbone_deterministic():
    """PerchBackbone produces deterministic outputs."""

    backbone = load("perch", "perch_v2")

    # Same input
    np.random.seed(42)
    batch = np.random.randn(1, 160_000).astype(np.float32)

    emb1 = backbone.encode(batch)
    emb2 = backbone.encode(batch)

    np.testing.assert_allclose(emb1, emb2, rtol=1e-5, atol=1e-5)


@pytest.mark.slow
def test_perch_backbone_transform():
    """PerchBackbone transform pads/truncates correctly."""

    backbone = load("perch", "perch_v2")
    transform = backbone.make_audio_transform()

    # Short audio (1 second)
    short = np.random.randn(32_000).astype(np.float32)
    assert len(transform(short)) == 160_000

    # Long audio (10 seconds)
    long = np.random.randn(320_000).astype(np.float32)
    assert len(transform(long)) == 160_000

    # Exact length
    exact = np.random.randn(160_000).astype(np.float32)
    assert len(transform(exact)) == 160_000


# =============================================================================
# Frontend Parity Tests (TF spectrogram vs numpy implementation)
# =============================================================================


@pytest.mark.slow
def test_perch_frontend_parity():
    """Test that numpy spectrogram matches TensorFlow model output.

    This test verifies that our pure numpy/scipy implementation of the
    Perch frontend produces spectrograms that closely match the TensorFlow
    model's spectrogram output. This is important for:
    1. Validating our understanding of Perch's preprocessing
    2. Enabling future pure-JAX inference without TensorFlow dependency

    Expected correlation: >0.85 for typical audio signals
    """
    # Skip if TensorFlow not available
    pytest.importorskip("tensorflow")
    pytest.importorskip("kagglehub")

    import tensorflow as tf
    import kagglehub

    # Load TF model (hide GPUs to use CPU model)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    path = kagglehub.model_download(
        "google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu"
    )
    model = tf.saved_model.load(path)

    # Test signal: mixed tones + noise
    np.random.seed(42)
    sr = 32000
    n_samples = sr * 5
    t = np.linspace(0, 5, n_samples, dtype=np.float32)
    audio = (
        np.sin(2 * np.pi * 1000 * t) * 0.3
        + np.sin(2 * np.pi * 3000 * t) * 0.2
        + np.random.randn(n_samples).astype(np.float32) * 0.1
    )

    # Get TF spectrogram
    result = model.signatures["serving_default"](inputs=tf.constant(audio[None, :]))
    tf_spec = result["spectrogram"].numpy()[0]

    # Get our spectrogram
    our_spec = perch.transform(audio)

    # Check shape
    assert our_spec.shape == tf_spec.shape, (
        f"Shape mismatch: {our_spec.shape} vs {tf_spec.shape}"
    )

    # Check correlation
    corr = np.corrcoef(our_spec.flatten(), tf_spec.flatten())[0, 1]
    assert corr > 0.85, f"Correlation too low: {corr:.4f} (expected >0.85)"

    # Check MAE is reasonable
    mae = np.mean(np.abs(our_spec - tf_spec))
    assert mae < 0.1, f"MAE too high: {mae:.4f} (expected <0.1)"


@pytest.mark.slow
def test_perch_frontend_parity_white_noise():
    """Test frontend parity with white noise input."""
    pytest.importorskip("tensorflow")
    pytest.importorskip("kagglehub")

    import tensorflow as tf
    import kagglehub

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    path = kagglehub.model_download(
        "google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu"
    )
    model = tf.saved_model.load(path)

    # White noise
    np.random.seed(123)
    audio = np.random.randn(160_000).astype(np.float32) * 0.3

    tf_spec = model.signatures["serving_default"](
        inputs=tf.constant(audio[None, :])
    )["spectrogram"].numpy()[0]
    our_spec = perch.transform(audio)

    corr = np.corrcoef(our_spec.flatten(), tf_spec.flatten())[0, 1]
    assert corr > 0.85, f"White noise correlation too low: {corr:.4f}"
