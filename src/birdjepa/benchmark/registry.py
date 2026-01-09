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
        self, batch: Float[Array, "batch time mels"]
    ) -> Float[Array, "batch dim"]:
        """Encode a batch of spectrograms, returning [batch, dim] features."""
        raise NotImplementedError

    def make_audio_transform(self):
        """Return the preprocessing function (waveform -> spectrogram)."""
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
        self, batch: Float[Array, "batch time mels"]
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
        import orbax.checkpoint as ocp

        import birdjepa.nn.transformer as transformer
        from birdjepa.nn import bird_mae

        # Default config - ViT-S/16 for audio (512x128 -> 32x8 patches)
        # TODO: Could infer from checkpoint metadata if we save it
        model_cfg = transformer.Transformer(
            input_h=512,
            input_w=128,
            patch_h=16,
            patch_w=16,
            embed_dim=384,
            depth=12,
            n_heads=6,
            n_cls_tokens=1,
        )

        # Create abstract model for restore
        import jax.random as jr

        abstract_encoder = transformer.TransformerModel(model_cfg, key=jr.key(0))

        # Load checkpoint
        mngr = ocp.CheckpointManager(ckpt_path)
        step = mngr.latest_step()
        assert step is not None, f"No checkpoint found at {ckpt_path}"

        restored = mngr.restore(
            step,
            args=ocp.args.Composite(
                state=ocp.args.PyTreeRestore(
                    {"encoder": abstract_encoder}, partial_restore=True
                ),
                metadata=ocp.args.JsonRestore(),
            ),
        )
        self.model = restored["state"]["encoder"]
        self.cfg = model_cfg
        self._transform = bird_mae.transform
        self._ckpt_path = ckpt_path

        logger.info("Loaded BirdJEPA checkpoint: %s (step %d)", ckpt_path, step)

    @jaxtyped(typechecker=beartype.beartype)
    def encode(
        self, batch: Float[Array, "batch time mels"]
    ) -> Float[Array, "batch dim"]:
        import birdjepa.nn.transformer as transformer

        # batch: [B, 512, 128] -> patchify -> [B, N, patch_dim]
        x_bnk, grid = transformer.patchify(batch, self.cfg)

        # Run encoder
        out = self.model(x_bnk, grid=grid, key=None)

        # Mean pool over CLS tokens (or patches if no CLS)
        if "cls" in out and out["cls"].shape[1] > 0:
            pooled = out["cls"].mean(axis=1)
        else:
            pooled = out["patches"].mean(axis=1)

        return pooled

    def make_audio_transform(self):
        return self._transform


register("birdjepa", BirdJEPABackbone)
