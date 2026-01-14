"""
Perch 2.0 audio embedding model in JAX/Equinox.

Perch uses EfficientNet-B3 (~12M params) to produce 1536-dim embeddings from
5-second audio clips. This module provides:
- `transform`: waveform -> log-mel spectrogram (matches TF model with >0.85 correlation)
- `EfficientNet`: the model architecture in Equinox
- `load_tf`: loads TF model for CPU inference (recommended)
- `load`: JAX model with random weights (weight conversion not implemented)

Frontend notes:
- Uses log-mel spectrogram with O'Shaughnessy mel scale (TensorFlow-compatible)
- STFT: 1024 FFT, 640 window (20ms), 320 hop (10ms)
- Log scaling: 0.1 * log(max(mel, 1e-3))
- Correlation with TF model spectrogram output: >0.85 for typical audio

Reference: https://arxiv.org/abs/2508.04665
Source: https://github.com/google-research/chirp/blob/main/chirp/models/efficientnet.py
"""

import dataclasses
import logging
import math
import os.path

import beartype
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped

logger = logging.getLogger(__name__)

# =============================================================================
# Perch audio preprocessing constants
# =============================================================================

PERCH_SR_HZ = 32_000
PERCH_CLIP_SEC = 5
PERCH_N_MELS = 128
PERCH_HOP_MS = 10
PERCH_WIN_MS = 20
PERCH_FMIN = 60
PERCH_FMAX = 16_000

# Derived constants
PERCH_HOP_SAMPLES = int(PERCH_SR_HZ * PERCH_HOP_MS / 1000)  # 320
PERCH_WIN_SAMPLES = int(PERCH_SR_HZ * PERCH_WIN_MS / 1000)  # 640
PERCH_TARGET_SAMPLES = PERCH_SR_HZ * PERCH_CLIP_SEC  # 160,000
PERCH_TARGET_T = 500  # 160000 samples / 320 hop = 500 frames

# EfficientNet-B3 config
EFFICIENTNET_B3_WIDTH = 1.2
EFFICIENTNET_B3_DEPTH = 1.4
EFFICIENTNET_B3_DROPOUT = 0.3
EFFICIENTNET_B3_OUTPUT_DIM = 1536


# =============================================================================
# Transform: waveform -> spectrogram
# =============================================================================

# Perch frontend parameters (from chirp/models/perch_2.py)
# - n_fft: 1024 (next power of 2 >= kernel_size)
# - kernel_size (window): 640 (20ms at 32kHz)
# - stride (hop): 320 (10ms at 32kHz)
# - power: 1.0 (magnitude, not squared)
# - mel bins: 128
# - freq range: 60-16000 Hz
# - log scaling: 0.1 * log(max(mel, floor)) where floor=1e-3 empirically works best
#
# Note: Uses O'Shaughnessy mel scale (like TensorFlow/chirp), not HTK.

PERCH_N_FFT = 1024
PERCH_LOG_FLOOR = 1e-5  # Determined empirically from silent audio test
PERCH_LOG_SCALAR = 0.1


def _hertz_to_mel(f: np.ndarray | float) -> np.ndarray | float:
    """Convert Hz to mel using O'Shaughnessy formula (TensorFlow-compatible)."""
    return 1127.0 * np.log1p(np.asarray(f) / 700.0)


def _make_mel_filterbank(
    n_fft: int,
    n_mels: int,
    sample_rate: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """Create mel filterbank matrix using O'Shaughnessy scale.

    This matches TensorFlow's tf.signal.linear_to_mel_weight_matrix.

    Returns:
        Filterbank matrix of shape [n_fft//2+1, n_mels]
    """
    num_spectrogram_bins = n_fft // 2 + 1
    nyquist = sample_rate / 2.0

    # Linear frequencies for each FFT bin (exclude DC bin like TF does)
    linear_frequencies = np.linspace(0.0, nyquist, num_spectrogram_bins)[1:]
    spectrogram_bins_mel = _hertz_to_mel(linear_frequencies)[:, np.newaxis]

    # Mel band edges
    band_edges_mel = np.linspace(
        _hertz_to_mel(fmin), _hertz_to_mel(fmax), n_mels + 2
    )

    lower_edge_mel = band_edges_mel[np.newaxis, :-2]
    center_mel = band_edges_mel[np.newaxis, 1:-1]
    upper_edge_mel = band_edges_mel[np.newaxis, 2:]

    # Triangle filter slopes
    lower_slopes = (spectrogram_bins_mel - lower_edge_mel) / (
        center_mel - lower_edge_mel
    )
    upper_slopes = (upper_edge_mel - spectrogram_bins_mel) / (
        upper_edge_mel - center_mel
    )

    mel_weights = np.maximum(0.0, np.minimum(lower_slopes, upper_slopes))

    # Re-add zeroed DC bin (index 0)
    return np.pad(mel_weights, ((1, 0), (0, 0)))


@jaxtyped(typechecker=beartype.beartype)
def transform(
    waveform: Float[np.ndarray, " samples"],
) -> Float[np.ndarray, "time mels"]:
    """
    Transform waveform to log-mel spectrogram for Perch.

    Uses scipy.signal.stft which matches TensorFlow's STFT behavior
    (both divide by window sum for normalization).

    Args:
        waveform: 1D numpy array of audio samples at 32kHz

    Returns:
        Numpy array of shape [500, 128] log-mel spectrogram
    """
    from scipy import signal as scipy_signal

    waveform = waveform.astype(np.float32)

    # 1) Pad/truncate to exactly 5 seconds
    n_samples = len(waveform)
    if n_samples < PERCH_TARGET_SAMPLES:
        waveform = np.pad(
            waveform, (0, PERCH_TARGET_SAMPLES - n_samples), mode="constant"
        )
    else:
        waveform = waveform[:PERCH_TARGET_SAMPLES]

    # 2) STFT using scipy (matches chirp's jsp.signal.stft)
    _, _, stfts = scipy_signal.stft(
        waveform,
        fs=PERCH_SR_HZ,
        nperseg=PERCH_WIN_SAMPLES,
        noverlap=PERCH_WIN_SAMPLES - PERCH_HOP_SAMPLES,
        nfft=PERCH_N_FFT,
        padded=False,  # Match chirp's padded=False
        boundary="zeros",
    )

    # Remove last frame if input divisible by stride (chirp behavior)
    if len(waveform) % PERCH_HOP_SAMPLES == 0:
        stfts = stfts[:, :-1]

    # Transpose to [time, freq] and take magnitude (power=1.0)
    stfts = stfts.T[:PERCH_TARGET_T]
    magnitude = np.abs(stfts)

    # 3) Mel filterbank (O'Shaughnessy scale)
    mel_filterbank = _make_mel_filterbank(
        PERCH_N_FFT, PERCH_N_MELS, PERCH_SR_HZ, PERCH_FMIN, PERCH_FMAX
    )
    mel_spec = magnitude @ mel_filterbank

    # 4) Log scaling: scalar * log(max(mel, floor))
    mel_spec = PERCH_LOG_SCALAR * np.log(np.maximum(mel_spec, PERCH_LOG_FLOOR))

    # 5) Ensure correct shape (pad if needed)
    if mel_spec.shape[0] < PERCH_TARGET_T:
        pad_frames = PERCH_TARGET_T - mel_spec.shape[0]
        floor_value = PERCH_LOG_SCALAR * np.log(PERCH_LOG_FLOOR)
        mel_spec = np.pad(
            mel_spec, ((0, pad_frames), (0, 0)), constant_values=floor_value
        )

    return mel_spec.astype(np.float32)


# =============================================================================
# EfficientNet-B3 Architecture
# =============================================================================


def _round_filters(filters: int, width_coefficient: float, divisor: int = 8) -> int:
    """Round number of filters based on width coefficient."""
    filters = filters * width_coefficient
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    # Make sure round down doesn't go down by more than 10%
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def _round_repeats(repeats: int, depth_coefficient: float) -> int:
    """Round number of repeats based on depth coefficient."""
    return int(math.ceil(depth_coefficient * repeats))


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class BlockArgs:
    """Arguments for a single MBConv block."""

    kernel_size: int
    num_repeat: int
    input_filters: int
    output_filters: int
    expand_ratio: int
    se_ratio: float
    strides: int


# EfficientNet-B0 base block configuration
_DEFAULT_BLOCKS_ARGS = [
    BlockArgs(3, 1, 32, 16, 1, 0.25, 1),
    BlockArgs(3, 2, 16, 24, 6, 0.25, 2),
    BlockArgs(5, 2, 24, 40, 6, 0.25, 2),
    BlockArgs(3, 3, 40, 80, 6, 0.25, 2),
    BlockArgs(5, 3, 80, 112, 6, 0.25, 1),
    BlockArgs(5, 4, 112, 192, 6, 0.25, 2),
    BlockArgs(3, 1, 192, 320, 6, 0.25, 1),
]


class DepthwiseConv2d(eqx.Module):
    """Depthwise 2D convolution."""

    weight: Float[Array, "channels 1 kh kw"]
    kernel_size: int = eqx.field(static=True)
    stride: int = eqx.field(static=True)
    padding: int = eqx.field(static=True)

    def __init__(
        self, channels: int, kernel_size: int, stride: int, *, key: PRNGKeyArray
    ):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2
        # Initialize with truncated normal
        self.weight = (
            jr.truncated_normal(key, -2, 2, (channels, 1, kernel_size, kernel_size))
            * 0.02
        )

    def __call__(
        self, x: Float[Array, "batch channels height width"]
    ) -> Float[Array, "batch channels h2 w2"]:
        # Depthwise conv: apply each filter to one channel
        return jax.lax.conv_general_dilated(
            x,
            self.weight,
            window_strides=(self.stride, self.stride),
            padding=[(self.padding, self.padding), (self.padding, self.padding)],
            feature_group_count=x.shape[1],  # Depthwise
        )


class Conv2d(eqx.Module):
    """Standard 2D convolution."""

    weight: Float[Array, "out_c in_c kh kw"]
    kernel_size: int = eqx.field(static=True)
    stride: int = eqx.field(static=True)
    padding: int = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        *,
        key: PRNGKeyArray,
    ):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2
        self.weight = (
            jr.truncated_normal(
                key, -2, 2, (out_channels, in_channels, kernel_size, kernel_size)
            )
            * 0.02
        )

    def __call__(
        self, x: Float[Array, "batch in_c height width"]
    ) -> Float[Array, "batch out_c h2 w2"]:
        return jax.lax.conv_general_dilated(
            x,
            self.weight,
            window_strides=(self.stride, self.stride),
            padding=[(self.padding, self.padding), (self.padding, self.padding)],
        )


class BatchNorm(eqx.Module):
    """Batch normalization (inference mode - uses running stats)."""

    gamma: Float[Array, " channels"]
    beta: Float[Array, " channels"]
    running_mean: Float[Array, " channels"]
    running_var: Float[Array, " channels"]
    eps: float = eqx.field(static=True)

    def __init__(self, channels: int):
        self.gamma = jnp.ones(channels)
        self.beta = jnp.zeros(channels)
        self.running_mean = jnp.zeros(channels)
        self.running_var = jnp.ones(channels)
        self.eps = 1e-3

    def __call__(
        self, x: Float[Array, "batch channels height width"]
    ) -> Float[Array, "batch channels height width"]:
        # Inference mode: use running stats
        mean = self.running_mean[None, :, None, None]
        var = self.running_var[None, :, None, None]
        gamma = self.gamma[None, :, None, None]
        beta = self.beta[None, :, None, None]
        return gamma * (x - mean) / jnp.sqrt(var + self.eps) + beta


class SqueezeExcitation(eqx.Module):
    """Squeeze-and-Excitation block."""

    fc1_weight: Float[Array, "reduced channels"]
    fc1_bias: Float[Array, " reduced"]
    fc2_weight: Float[Array, "channels reduced"]
    fc2_bias: Float[Array, " channels"]

    def __init__(self, channels: int, reduced: int, *, key: PRNGKeyArray):
        k1, k2 = jr.split(key)
        self.fc1_weight = jr.truncated_normal(k1, -2, 2, (reduced, channels)) * 0.02
        self.fc1_bias = jnp.zeros(reduced)
        self.fc2_weight = jr.truncated_normal(k2, -2, 2, (channels, reduced)) * 0.02
        self.fc2_bias = jnp.zeros(channels)

    def __call__(
        self, x: Float[Array, "batch channels height width"]
    ) -> Float[Array, "batch channels height width"]:
        # Global average pooling
        se = x.mean(axis=(2, 3))  # [B, C]
        # FC layers
        se = se @ self.fc1_weight.T + self.fc1_bias
        se = jax.nn.swish(se)
        se = se @ self.fc2_weight.T + self.fc2_bias
        se = jax.nn.sigmoid(se)
        # Scale
        return x * se[:, :, None, None]


class MBConvBlock(eqx.Module):
    """Mobile Inverted Bottleneck Convolution block."""

    # Expansion phase
    expand_conv: Conv2d | None
    expand_bn: BatchNorm | None
    # Depthwise phase
    depthwise_conv: DepthwiseConv2d
    depthwise_bn: BatchNorm
    # Squeeze-excitation
    se: SqueezeExcitation
    # Output phase
    project_conv: Conv2d
    project_bn: BatchNorm
    # Config
    use_residual: bool = eqx.field(static=True)
    expand_ratio: int = eqx.field(static=True)

    def __init__(
        self,
        input_filters: int,
        output_filters: int,
        kernel_size: int,
        stride: int,
        expand_ratio: int,
        se_ratio: float,
        *,
        key: PRNGKeyArray,
    ):
        keys = jr.split(key, 5)
        self.expand_ratio = expand_ratio
        self.use_residual = (stride == 1) and (input_filters == output_filters)

        # Expansion phase (if expand_ratio > 1)
        expanded_filters = input_filters * expand_ratio
        if expand_ratio != 1:
            self.expand_conv = Conv2d(input_filters, expanded_filters, 1, key=keys[0])
            self.expand_bn = BatchNorm(expanded_filters)
        else:
            self.expand_conv = None
            self.expand_bn = None

        # Depthwise convolution
        self.depthwise_conv = DepthwiseConv2d(
            expanded_filters, kernel_size, stride, key=keys[1]
        )
        self.depthwise_bn = BatchNorm(expanded_filters)

        # Squeeze-excitation
        se_filters = max(1, int(input_filters * se_ratio))
        self.se = SqueezeExcitation(expanded_filters, se_filters, key=keys[2])

        # Output projection
        self.project_conv = Conv2d(expanded_filters, output_filters, 1, key=keys[3])
        self.project_bn = BatchNorm(output_filters)

    def __call__(
        self, x: Float[Array, "batch c h w"]
    ) -> Float[Array, "batch c2 h2 w2"]:
        residual = x

        # Expansion
        if self.expand_conv is not None:
            x = self.expand_conv(x)
            x = self.expand_bn(x)
            x = jax.nn.swish(x)

        # Depthwise
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = jax.nn.swish(x)

        # Squeeze-excitation
        x = self.se(x)

        # Project
        x = self.project_conv(x)
        x = self.project_bn(x)

        # Residual
        if self.use_residual:
            x = x + residual

        return x


class EfficientNet(eqx.Module):
    """EfficientNet-B3 model for Perch embeddings."""

    # Stem
    stem_conv: Conv2d
    stem_bn: BatchNorm
    # Blocks
    blocks: list[MBConvBlock]
    # Head
    head_conv: Conv2d
    head_bn: BatchNorm
    # Config
    output_dim: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        width_coefficient: float = EFFICIENTNET_B3_WIDTH,
        depth_coefficient: float = EFFICIENTNET_B3_DEPTH,
        input_channels: int = 1,  # Mono spectrogram
        key: PRNGKeyArray,
    ):
        keys = jr.split(key, 100)
        key_idx = 0

        # Stem: 3x3 conv, stride 2
        stem_filters = _round_filters(32, width_coefficient)
        self.stem_conv = Conv2d(
            input_channels, stem_filters, 3, stride=2, key=keys[key_idx]
        )
        key_idx += 1
        self.stem_bn = BatchNorm(stem_filters)

        # Build blocks
        self.blocks = []
        prev_filters = stem_filters

        for block_args in _DEFAULT_BLOCKS_ARGS:
            # Scale filters and repeats
            input_f = _round_filters(block_args.input_filters, width_coefficient)
            output_f = _round_filters(block_args.output_filters, width_coefficient)
            num_repeat = _round_repeats(block_args.num_repeat, depth_coefficient)

            # First block may have stride > 1
            for i in range(num_repeat):
                stride = block_args.strides if i == 0 else 1
                in_f = prev_filters if i == 0 else output_f

                block = MBConvBlock(
                    input_filters=in_f,
                    output_filters=output_f,
                    kernel_size=block_args.kernel_size,
                    stride=stride,
                    expand_ratio=block_args.expand_ratio,
                    se_ratio=block_args.se_ratio,
                    key=keys[key_idx],
                )
                key_idx += 1
                self.blocks.append(block)
                prev_filters = output_f

        # Head: 1x1 conv to expand to output_dim
        self.output_dim = _round_filters(1280, width_coefficient)  # B3: 1536
        self.head_conv = Conv2d(prev_filters, self.output_dim, 1, key=keys[key_idx])
        key_idx += 1
        self.head_bn = BatchNorm(self.output_dim)

    def __call__(
        self, x: Float[Array, "batch 1 time mels"]
    ) -> Float[Array, "batch dim"]:
        """Forward pass returning pooled embeddings."""
        # Stem
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = jax.nn.swish(x)

        # Blocks
        for block in self.blocks:
            x = block(x)

        # Head
        x = self.head_conv(x)
        x = self.head_bn(x)
        x = jax.nn.swish(x)

        # Global average pooling
        x = x.mean(axis=(2, 3))

        return x


# =============================================================================
# TensorFlow Model Wrapper (for inference)
# =============================================================================


class PerchTFModel:
    """Wrapper around TensorFlow Perch model for CPU-only inference.

    This class loads the TensorFlow SavedModel and runs inference on CPU,
    avoiding GPU memory conflicts with JAX.
    """

    def __init__(self, model_path: str):
        import os

        # Suppress TF logging
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

        import tensorflow as tf

        # Force TensorFlow to CPU only (don't modify CUDA_VISIBLE_DEVICES
        # since that affects JAX too)
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError:
            # Already initialized - that's OK
            pass

        self.model = tf.saved_model.load(model_path)
        self.serve = self.model.signatures["serving_default"]
        self._tf = tf

    def embed(self, waveform: np.ndarray) -> np.ndarray:
        """Embed audio waveforms.

        Args:
            waveform: Audio array of shape [batch, 160000] at 32kHz

        Returns:
            Embeddings of shape [batch, 1536]
        """
        # Ensure correct shape
        if waveform.ndim == 1:
            waveform = waveform[None, :]

        # Run TF inference
        inputs = self._tf.constant(waveform, dtype=self._tf.float32)
        outputs = self.serve(inputs=inputs)

        # Return embedding as numpy
        return outputs["embedding"].numpy()


@beartype.beartype
def load_tf(ckpt: str = "perch_v2") -> PerchTFModel:
    """Load Perch TensorFlow model for CPU inference.

    This downloads the model from Kaggle and wraps it for inference.
    TensorFlow runs on CPU only to avoid GPU conflicts with JAX.

    Args:
        ckpt: Model variant (default: "perch_v2")

    Returns:
        PerchTFModel wrapper for inference
    """
    import kagglehub

    # Map checkpoint names to Kaggle slugs
    slug_map = {
        "perch_v2": "google/bird-vocalization-classifier/tensorFlow2/perch_v2",
        "perch_v2_cpu": "google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu",
    }

    if ckpt not in slug_map:
        raise ValueError(
            f"Unknown checkpoint '{ckpt}'. Available: {list(slug_map.keys())}"
        )

    # Download model
    model_path = kagglehub.model_download(slug_map[ckpt])
    logger.info("Loaded Perch TF model from: %s", model_path)

    return PerchTFModel(model_path)


# =============================================================================
# JAX Model Loading (placeholder for future pure-JAX implementation)
# =============================================================================

# Weight conversion notes:
# The TF checkpoint was created via jax2tf from Google's chirp Flax model.
# The variable names are anonymous (_tf_var_leaves/0, /1, etc.) and the
# pytree structure differs from our Equinox implementation because chirp
# creates MBConv blocks dynamically in __call__ rather than as attributes.
#
# Options for proper weight loading:
# 1. Request JAX/Flax checkpoint from Google (see chirp repo issues)
# 2. Port chirp's exact Flax model structure to match their pytree
# 3. Trace through both models to build explicit variable mapping
#
# For now, use load_tf() which wraps the TF model for CPU inference.
# Reference: https://github.com/google-research/chirp/blob/main/chirp/models/efficientnet.py
#
# (sam) https://github.com/google-research/perch/issues/652 and https://github.com/google-research/perch/issues/661 and https://github.com/google-research/perch-hoplite/issues/56 for some more context in the weight loading. given the complexity, I would like to make this a one-time cost and write a script to download the anonymously named tf model, do a forwrd pass, and move it over to equinox. https://github.com/samuelstevens/dinov3/blob/main/test_jax.py is an example of writing a bunch of tests for converting dinov3 to jax/equinox and might be useful.


@beartype.beartype
def load(ckpt: str = "perch_v2", *, key: PRNGKeyArray | None = None) -> EfficientNet:
    """Load Perch model with pretrained weights (JAX/Equinox).

    NOTE: Weight conversion from TensorFlow is not yet implemented.
    The TF checkpoint uses anonymous variable names from jax2tf export,
    and the chirp Flax model has a different pytree structure than our
    Equinox implementation. For actual inference, use load_tf() instead.

    Args:
        ckpt: Checkpoint name (default: "perch_v2")
        key: PRNG key for model initialization

    Returns:
        EfficientNet model (random weights - conversion not implemented)
    """
    if key is None:
        key = jr.key(0)

    model = EfficientNet(key=key)

    logger.warning(
        "JAX weight conversion not yet implemented. Use load_tf() for actual inference."
    )
    return model
