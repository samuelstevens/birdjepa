"""
Registry for audio backbones used in benchmarking.
"""

import logging

import beartype
from jaxtyping import Array, Float, jaxtyped

logger = logging.getLogger(__name__)


class AudioBackbone:
    """A frozen audio model that embeds batches of spectrograms into feature vectors."""

    def encode(
        self, batch: Float[Array, "batch time mel"]
    ) -> Float[Array, "batch dim"]:
        """Encode a batch of spectrograms, returning [batch, dim] features.

        Args:
            batch: Preprocessed audio batch. Shape depends on model:
                - Bird-MAE: [batch, 512, 128]
                - Perch: [batch, 500, 128]

        Returns:
            Feature vectors of shape [batch, dim]
        """
        raise NotImplementedError

    def make_audio_transform(self):
        """Return the preprocessing function (waveform -> spectrogram).

        Returns a callable that transforms raw waveform arrays into the format
        expected by this backbone's encode() method.
        """
        raise NotImplementedError


_registry: dict[str, type[AudioBackbone]] = {}


@beartype.beartype
def register(name: str, cls: type[AudioBackbone]):
    """Register an audio backbone class."""
    if name in _registry:
        logger.warning("Overwriting '%s' in registry.", name)
    _registry[name] = cls


@beartype.beartype
def load(org: str, ckpt: str) -> AudioBackbone:
    """Load a registered audio backbone."""
    if org not in _registry:
        raise ValueError(f"'{org}' not in registry. Available: {list(_registry)}")
    return _registry[org](ckpt)


def list_backbones() -> list[str]:
    """List registered backbone names."""
    return list(_registry)


# Bird-MAE JAX wrapper
class BirdMAEBackbone(AudioBackbone):
    """Wrapper around Bird-MAE (JAX) for the benchmark registry."""

    def __init__(self, ckpt: str):
        from birdjepa.nn import bird_mae, bird_mae_jax

        self.model = bird_mae_jax.load(ckpt)
        self._transform = bird_mae.transform
        self._ckpt = ckpt

    @jaxtyped(typechecker=beartype.beartype)
    def encode(
        self, batch: Float[Array, "batch time mel"]
    ) -> Float[Array, "batch dim"]:
        # batch: [B, 512, 128] -> add channel dim -> [B, 1, 512, 128]
        x = batch[:, None, :, :]
        out = self.model(x)
        return out["pooled"]

    def make_audio_transform(self):
        return self._transform


register("bird-mae", BirdMAEBackbone)


# BirdJEPA local checkpoint wrapper
class BirdJEPABackbone(AudioBackbone):
    """Wrapper around BirdJEPA checkpoints for the benchmark registry.

    model_ckpt should be the path to the checkpoint directory (e.g., /path/to/wandb_run_id).
    """

    def __init__(self, ckpt_path: str):
        import pathlib

        import birdjepa.checkpoints
        import birdjepa.nn.transformer as transformer
        from birdjepa.nn import bird_mae

        ckpt_path = pathlib.Path(ckpt_path)
        self.model, self.cfg = birdjepa.checkpoints.load_eval(
            ckpt_path, transformer.TransformerModel, transformer.Transformer
        )
        self._transform = bird_mae.transform
        self._ckpt_path = str(ckpt_path)

    @jaxtyped(typechecker=beartype.beartype)
    def encode(
        self, batch: Float[Array, "batch time mel"]
    ) -> Float[Array, "batch dim"]:
        import jax.random as jr

        import birdjepa.nn.transformer as transformer

        # batch: [B, 512, 128] -> patchify -> [B, N, patch_dim]
        x_bnk, grid = transformer.patchify(batch, self.cfg)

        # Run encoder (pass dummy key for dropout - not used in inference)
        out = self.model(x_bnk, grid=grid, key=jr.key(0))

        # Mean pool over CLS tokens (or patches if no CLS)
        if "cls" in out and out["cls"].shape[1] > 0:
            pooled = out["cls"].mean(axis=1)
        else:
            pooled = out["patches"].mean(axis=1)

        return pooled

    def make_audio_transform(self):
        return self._transform


register("birdjepa", BirdJEPABackbone)


# Perch TF wrapper (uses TensorFlow on CPU for inference)
class PerchBackbone(AudioBackbone):
    """Wrapper around Perch 2.0 TensorFlow model.

    Uses TensorFlow on CPU only to avoid GPU conflicts with JAX.
    The model takes raw waveforms directly and produces embeddings.

    Note: Uses perch_v2_cpu by default since the CUDA variant cannot run on CPU.
    """

    SAMPLE_RATE = 32_000
    TARGET_SAMPLES = 160_000  # 5 seconds at 32kHz

    def __init__(self, ckpt: str = "perch_v2_cpu"):
        from birdjepa.nn import perch

        # Map common names to CPU variant
        if ckpt in ("perch_v2", "perch"):
            ckpt = "perch_v2_cpu"

        self.model = perch.load_tf(ckpt)
        self._ckpt = ckpt

    def encode(self, batch) -> Float[Array, "batch dim"]:
        """Encode audio waveforms to embeddings.

        Args:
            batch: Audio waveforms of shape [batch, 160000] (5s at 32kHz).
                   Can be numpy or JAX arrays.

        Returns:
            Embeddings of shape [batch, 1536]
        """
        import numpy as np

        import jax.numpy as jnp

        # Convert to numpy for TF
        batch_np = np.asarray(batch, dtype=np.float32)

        # Run TF inference (on CPU)
        embeddings = self.model.embed(batch_np)

        # Convert back to JAX array
        return jnp.array(embeddings)

    def make_audio_transform(self):
        """Return transform that pads/truncates waveform to 160k samples."""
        return _perch_waveform_transform


def _perch_waveform_transform(waveform):
    """Pad or truncate waveform to 160,000 samples (5s at 32kHz)."""
    import numpy as np

    target = PerchBackbone.TARGET_SAMPLES
    waveform = np.asarray(waveform, dtype=np.float32)

    if len(waveform) < target:
        waveform = np.pad(waveform, (0, target - len(waveform)), mode="constant")
    else:
        waveform = waveform[:target]

    return waveform


register("perch", PerchBackbone)
