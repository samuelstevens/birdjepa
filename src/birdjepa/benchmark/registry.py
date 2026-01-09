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


def _load_legacy_checkpoint(ckpt_path, step):
    """Load encoder from legacy checkpoint format (no separate encoder item).

    Infers config from array shapes and uses tensorstore to read arrays directly.
    """
    import json

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import tensorstore as ts

    import birdjepa.nn.transformer as transformer

    # Read checkpoint metadata to get shapes
    meta_path = ckpt_path / str(step) / "state" / "_METADATA"
    with open(meta_path) as fd:
        meta = json.load(fd)

    # Parse shapes from metadata
    shapes = {}
    for key_str, val in meta.get("tree_metadata", {}).items():
        key_tuple = eval(key_str)
        shape = val.get("value_metadata", {}).get("write_shape")
        if shape:
            shapes[key_tuple] = tuple(shape)

    # Infer config from checkpoint shapes
    depth = shapes.get(("encoder", "blocks", "norm1", "weight"), (12, 384))[0]
    embed_dim = shapes.get(("encoder", "blocks", "norm1", "weight"), (12, 384))[1]
    pos_shape = shapes.get(("encoder", "pos_embed_hw"), (1, 32, 8, 384))
    h_patches, w_patches = pos_shape[1], pos_shape[2]
    qkv_shape = shapes.get(("encoder", "blocks", "attn", "qkv", "weight"))
    n_heads = qkv_shape[1] // 3 // 64 if qkv_shape else 6

    model_cfg = transformer.Transformer(
        input_h=h_patches * 16,
        input_w=w_patches * 16,
        patch_h=16,
        patch_w=16,
        embed_dim=embed_dim,
        depth=depth,
        n_heads=n_heads,
        n_cls_tokens=1,
        use_scan=True,
    )

    abstract_encoder = transformer.TransformerModel(model_cfg, key=jr.key(0))
    state_path = ckpt_path / str(step) / "state"

    def read_array(key_path: tuple) -> jax.Array | None:
        key_str = str(key_path)
        key_info = meta["tree_metadata"].get(key_str, {})
        val_meta = key_info.get("value_metadata", {})
        if val_meta.get("value_type") == "None" or val_meta.get("skip_deserialize"):
            return None
        if val_meta.get("write_shape") is None:
            return None

        ts_key = ".".join(str(k) for k in key_path)
        spec = {
            "driver": "zarr",
            "kvstore": {
                "driver": "ocdbt",
                "base": f"file://{state_path}",
                "path": ts_key,
            },
        }
        try:
            return jnp.array(ts.open(spec, read=True).result().read().result())
        except Exception:
            return None

    leaves, treedef = jax.tree_util.tree_flatten_with_path(abstract_encoder)
    new_leaves = []
    for keypath, leaf in leaves:
        full_key = ("encoder",) + tuple(
            k.key if hasattr(k, "key") else str(k) for k in keypath
        )
        if eqx.is_array(leaf):
            restored = read_array(full_key)
            new_leaves.append(
                restored
                if restored is not None and restored.shape == leaf.shape
                else leaf
            )
        else:
            new_leaves.append(leaf)

    encoder = jax.tree_util.tree_unflatten(treedef, new_leaves)
    logger.info(
        "Inferred config: depth=%d, embed_dim=%d, n_heads=%d", depth, embed_dim, n_heads
    )
    return encoder, model_cfg


# BirdJEPA local checkpoint wrapper
class BirdJEPABackbone(AudioBackbone):
    """Wrapper around BirdJEPA checkpoints for the benchmark registry.

    model_ckpt should be the path to the checkpoint directory (e.g., /path/to/wandb_run_id).
    """

    def __init__(self, ckpt_path: str):
        import json
        import pathlib

        import jax.random as jr
        import orbax.checkpoint as ocp

        import birdjepa.nn.transformer as transformer
        from birdjepa.nn import bird_mae

        ckpt_path = pathlib.Path(ckpt_path)

        # Get latest checkpoint step
        mngr = ocp.CheckpointManager(ckpt_path)
        step = mngr.latest_step()
        assert step is not None, f"No checkpoint found at {ckpt_path}"

        ckpt_dir = ckpt_path / str(step)
        if not (ckpt_dir / "encoder").exists():
            # Legacy format: infer config from array shapes, use tensorstore
            self.model, model_cfg = _load_legacy_checkpoint(ckpt_path, step)
            logger.warning(
                "Using legacy checkpoint format. Re-save with newer pretrain.py for cleaner loading."
            )
        else:
            # New format: load config from metadata, restore encoder directly
            with open(ckpt_dir / "metadata" / "metadata") as fd:
                metadata = json.load(fd)
            model_cfg = transformer.Transformer(**metadata["encoder_config"])

            empty = transformer.TransformerModel(model_cfg, key=jr.key(0))
            mngr = ocp.CheckpointManager(
                ckpt_path,
                options=ocp.CheckpointManagerOptions(
                    single_host_load_and_broadcast=True
                ),
            )
            restored = mngr.restore(
                step, args=ocp.args.Composite(encoder=ocp.args.StandardRestore(empty))
            )
            self.model = restored["encoder"]
            logger.info("Loaded BirdJEPA checkpoint: %s (step %d)", ckpt_path, step)

        self.cfg = model_cfg
        self._transform = bird_mae.transform
        self._ckpt_path = str(ckpt_path)

    @jaxtyped(typechecker=beartype.beartype)
    def encode(
        self, batch: Float[Array, "batch time mels"]
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
