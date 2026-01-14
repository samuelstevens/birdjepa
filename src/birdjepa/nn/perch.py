"""
Perch 2.0 audio embedding model in JAX/Equinox.

Perch uses EfficientNet-B3 (~12M params) to produce 1536-dim embeddings from
5-second audio clips. This module provides:
- `transform`: waveform -> spectrogram preprocessing
- `EfficientNet`: the model architecture in Equinox
- `load`: downloads TF weights and converts to JAX on-the-fly

Reference: https://arxiv.org/abs/2508.04665
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
PERCH_TARGET_T = int((PERCH_TARGET_SAMPLES - PERCH_WIN_SAMPLES) / PERCH_HOP_SAMPLES) + 1  # ~500

# EfficientNet-B3 config
EFFICIENTNET_B3_WIDTH = 1.2
EFFICIENTNET_B3_DEPTH = 1.4
EFFICIENTNET_B3_DROPOUT = 0.3
EFFICIENTNET_B3_OUTPUT_DIM = 1536


# =============================================================================
# Transform: waveform -> spectrogram
# =============================================================================


def _make_mel_filterbank(
    n_fft: int,
    n_mels: int,
    sample_rate: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """Create mel filterbank matrix."""
    # Mel scale conversion
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    # Create mel points
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    # Convert to FFT bin indices
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    # Create filterbank
    filterbank = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]

        # Rising slope
        for j in range(left, center):
            if center != left:
                filterbank[i, j] = (j - left) / (center - left)

        # Falling slope
        for j in range(center, right):
            if right != center:
                filterbank[i, j] = (right - j) / (right - center)

    return filterbank


@jaxtyped(typechecker=beartype.beartype)
def transform(
    waveform: Float[np.ndarray, " samples"],
) -> Float[np.ndarray, "time mels"]:
    """
    Transform waveform to log-mel spectrogram for Perch.

    Args:
        waveform: 1D numpy array of audio samples at 32kHz

    Returns:
        Numpy array of shape [~500, 128] log-mel spectrogram
    """
    waveform = waveform.astype(np.float32)

    # 1) Pad/truncate to exactly 5 seconds
    n_samples = len(waveform)
    if n_samples < PERCH_TARGET_SAMPLES:
        waveform = np.pad(
            waveform, (0, PERCH_TARGET_SAMPLES - n_samples), mode="constant"
        )
    else:
        waveform = waveform[:PERCH_TARGET_SAMPLES]

    # 2) STFT
    n_fft = PERCH_WIN_SAMPLES * 2  # Use 2x window for better freq resolution
    hop_length = PERCH_HOP_SAMPLES
    win_length = PERCH_WIN_SAMPLES

    # Hann window
    window = np.hanning(win_length)

    # Pad for centering
    pad_amount = n_fft // 2
    waveform_padded = np.pad(waveform, (pad_amount, pad_amount), mode="reflect")

    # Frame the signal
    n_frames = 1 + (len(waveform_padded) - n_fft) // hop_length
    frames = np.zeros((n_frames, n_fft))
    for i in range(n_frames):
        start = i * hop_length
        frame = waveform_padded[start : start + n_fft]
        # Apply window (center of frame)
        frame_windowed = np.zeros(n_fft)
        win_start = (n_fft - win_length) // 2
        frame_windowed[win_start : win_start + win_length] = (
            frame[win_start : win_start + win_length] * window
        )
        frames[i] = frame_windowed

    # FFT
    spectrum = np.fft.rfft(frames, n=n_fft)
    power_spectrum = np.abs(spectrum) ** 2

    # 3) Mel filterbank
    mel_filterbank = _make_mel_filterbank(
        n_fft, PERCH_N_MELS, PERCH_SR_HZ, PERCH_FMIN, PERCH_FMAX
    )
    mel_spec = np.dot(power_spectrum, mel_filterbank.T)

    # 4) Log compression
    mel_spec = np.log(mel_spec + 1e-6)

    # 5) Ensure correct shape (truncate/pad to target frames)
    if mel_spec.shape[0] > PERCH_TARGET_T:
        mel_spec = mel_spec[:PERCH_TARGET_T]
    elif mel_spec.shape[0] < PERCH_TARGET_T:
        pad_frames = PERCH_TARGET_T - mel_spec.shape[0]
        mel_spec = np.pad(mel_spec, ((0, pad_frames), (0, 0)), mode="constant")

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

    def __init__(self, channels: int, kernel_size: int, stride: int, *, key: PRNGKeyArray):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2
        # Initialize with truncated normal
        self.weight = jr.truncated_normal(key, -2, 2, (channels, 1, kernel_size, kernel_size)) * 0.02

    def __call__(self, x: Float[Array, "batch channels height width"]) -> Float[Array, "batch channels h2 w2"]:
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
        self.weight = jr.truncated_normal(
            key, -2, 2, (out_channels, in_channels, kernel_size, kernel_size)
        ) * 0.02

    def __call__(self, x: Float[Array, "batch in_c height width"]) -> Float[Array, "batch out_c h2 w2"]:
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

    def __call__(self, x: Float[Array, "batch channels height width"]) -> Float[Array, "batch channels height width"]:
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

    def __call__(self, x: Float[Array, "batch channels height width"]) -> Float[Array, "batch channels height width"]:
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
        self.depthwise_conv = DepthwiseConv2d(expanded_filters, kernel_size, stride, key=keys[1])
        self.depthwise_bn = BatchNorm(expanded_filters)

        # Squeeze-excitation
        se_filters = max(1, int(input_filters * se_ratio))
        self.se = SqueezeExcitation(expanded_filters, se_filters, key=keys[2])

        # Output projection
        self.project_conv = Conv2d(expanded_filters, output_filters, 1, key=keys[3])
        self.project_bn = BatchNorm(output_filters)

    def __call__(self, x: Float[Array, "batch c h w"]) -> Float[Array, "batch c2 h2 w2"]:
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
        self.stem_conv = Conv2d(input_channels, stem_filters, 3, stride=2, key=keys[key_idx])
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
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        import tensorflow as tf

        # Force TensorFlow to CPU only (don't modify CUDA_VISIBLE_DEVICES
        # since that affects JAX too)
        try:
            tf.config.set_visible_devices([], 'GPU')
        except RuntimeError:
            # Already initialized - that's OK
            pass

        self.model = tf.saved_model.load(model_path)
        self.serve = self.model.signatures['serving_default']
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
        return outputs['embedding'].numpy()


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
        raise ValueError(f"Unknown checkpoint '{ckpt}'. Available: {list(slug_map.keys())}")

    # Download model
    model_path = kagglehub.model_download(slug_map[ckpt])
    logger.info("Loaded Perch TF model from: %s", model_path)

    return PerchTFModel(model_path)


# =============================================================================
# JAX Model Loading (placeholder for future pure-JAX implementation)
# =============================================================================


@beartype.beartype
def load(ckpt: str = "perch_v2", *, key: PRNGKeyArray | None = None) -> EfficientNet:
    """Load Perch model with pretrained weights (JAX/Equinox).

    NOTE: Weight conversion from TensorFlow is not yet implemented.
    For now, use load_tf() for actual inference, or this function
    for architecture testing with random weights.

    Args:
        ckpt: Checkpoint name (default: "perch_v2")
        key: PRNG key for model initialization

    Returns:
        EfficientNet model (random weights - conversion TODO)
    """
    if key is None:
        key = jr.key(0)

    model = EfficientNet(key=key)

    logger.warning(
        "JAX weight conversion not yet implemented. "
        "Use load_tf() for actual inference."
    )
    return model
