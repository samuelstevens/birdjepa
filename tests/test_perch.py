"""Tests for Perch audio embedding model.

Tests the PerchBackbone inference and transform function.
"""

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
