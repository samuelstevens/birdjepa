"""Checkpoint save/load utilities for BirdJEPA models.

Wraps Orbax checkpoint functionality for:
- Training: save/restore full state (encoder, objective, probe, opt_state)
- Inference: restore just the encoder with config
"""

import dataclasses
import logging
import pathlib

import beartype
import equinox as eqx
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

import birdjepa.helpers

logger = logging.getLogger(__name__)


def _to_numpy(pytree):
    """Convert JAX arrays to numpy arrays for checkpoint save.

    This avoids Orbax's sharding-related bugs where replicated arrays get
    scaled by sqrt(n_devices) during restore. By saving numpy arrays,
    we bypass the problematic sharding handling entirely.

    Must be called AFTER arrays are materialized (e.g., after training step).
    """

    def convert(x):
        if hasattr(x, "block_until_ready"):
            x.block_until_ready()  # Ensure computation is complete
        if hasattr(x, "device"):
            return jax.device_get(x)  # Convert to numpy
        return x

    return jax.tree.map(convert, pytree)


def _none_to_nan(value: float | None) -> float:
    return float("nan") if value is None else value


def _opt_state_stats(opt_state) -> dict:
    float_tree = eqx.filter(opt_state, eqx.is_inexact_array)
    float_leaves = [leaf for leaf in jax.tree.leaves(float_tree) if leaf is not None]

    array_tree = eqx.filter(opt_state, eqx.is_array)
    count_values = [
        int(leaf)
        for leaf in jax.tree.leaves(array_tree)
        if leaf is not None
        and jnp.issubdtype(leaf.dtype, jnp.integer)
        and leaf.shape == ()
    ]

    if not float_leaves:
        return {
            "l2": None,
            "abs_mean": None,
            "abs_max": None,
            "count_values": count_values,
        }

    n_elems = sum(leaf.size for leaf in float_leaves)
    l2 = float(birdjepa.helpers.tree_l2_norm(float_leaves))
    abs_sum = jax.tree_util.tree_reduce(
        jax.lax.add, [jnp.sum(jnp.abs(leaf)) for leaf in float_leaves]
    )
    abs_max = jax.tree_util.tree_reduce(
        jax.lax.max, [jnp.max(jnp.abs(leaf)) for leaf in float_leaves]
    )
    abs_mean = float(abs_sum) / max(1, n_elems)
    abs_max = float(abs_max)
    return {
        "l2": l2,
        "abs_mean": abs_mean,
        "abs_max": abs_max,
        "count_values": count_values,
    }


@beartype.beartype
def get_opt_state_stats(opt_state) -> dict:
    return _opt_state_stats(opt_state)


class CheckpointManager:
    """Wrapper around Orbax CheckpointManager for BirdJEPA training."""

    @beartype.beartype
    def __init__(
        self,
        ckpt_dir: pathlib.Path,
        *,
        save_interval_steps: int = 500,
        max_to_keep: int = 5,
        async_checkpointing: bool = True,
    ):
        """Create a checkpoint manager for training.

        Saves three items:
        - state: training state (objective, probe, opt_state)
        - encoder: encoder model (separate for clean inference loading)
        - metadata: JSON with step and encoder_config
        """
        self._mngr = ocp.CheckpointManager(
            ckpt_dir,
            options=ocp.CheckpointManagerOptions(
                save_interval_steps=save_interval_steps,
                max_to_keep=max_to_keep,
                enable_async_checkpointing=async_checkpointing,
                single_host_load_and_broadcast=True,
            ),
            item_names=("state", "encoder", "metadata"),
        )

    @beartype.beartype
    def save(
        self,
        step: int,
        *,
        encoder,
        objective,
        probe,
        opt_state,
        encoder_config,
        prng_key=None,
        param_norm: float | None = None,
        force: bool = False,
    ):
        """Save a training checkpoint.

        Args:
            step: Training step number.
            encoder: Encoder model (TransformerModel).
            objective: Objective model.
            probe: Linear probe model.
            opt_state: Optimizer state.
            encoder_config: Transformer config dataclass for the encoder.
            prng_key: Optional PRNG key to save for deterministic resume.
            param_norm: Optional param_norm for the saved state (post-update).
            force: If True, save even if step doesn't match save_interval.
        """
        if not self.should_save(step, force=force):
            return

        # Compute encoder-only norm BEFORE conversion (on original JAX arrays)
        encoder_params = eqx.filter(encoder, eqx.is_inexact_array)
        encoder_norm = float(birdjepa.helpers.tree_l2_norm(encoder_params))

        # Convert to numpy to avoid Orbax sharding bugs during restore
        encoder_np = _to_numpy(encoder)

        # Verify conversion preserved values
        encoder_np_params = eqx.filter(encoder_np, eqx.is_inexact_array)
        encoder_norm_after = float(birdjepa.helpers.tree_l2_norm(encoder_np_params))
        if abs(encoder_norm - encoder_norm_after) > 0.01:
            logger.error(
                "CKPT_BUG: encoder_norm changed during numpy conversion! before=%.6f after=%.6f",
                encoder_norm,
                encoder_norm_after,
            )
        logger.info("CKPT_SAVE step=%d encoder_norm=%.6f", step, encoder_norm)

        opt_stats = _opt_state_stats(opt_state)
        logger.info(
            "CKPT_SAVE step=%d opt_state_l2=%.6f abs_mean=%.6f abs_max=%.6f count_values=%s",
            step,
            _none_to_nan(opt_stats["l2"]),
            _none_to_nan(opt_stats["abs_mean"]),
            _none_to_nan(opt_stats["abs_max"]),
            opt_stats["count_values"],
        )

        metadata = {
            "step": step,
            "encoder_config": dataclasses.asdict(encoder_config),
            "encoder_norm": encoder_norm,  # For verification on restore
            "opt_state_l2": opt_stats["l2"],
            "opt_state_abs_mean": opt_stats["abs_mean"],
            "opt_state_abs_max": opt_stats["abs_max"],
            "opt_state_count_values": opt_stats["count_values"],
        }
        if param_norm is not None:
            metadata["param_norm"] = float(param_norm)
        if prng_key is not None:
            key_data = jax.random.key_data(prng_key)
            metadata["prng_key"] = jax.device_get(key_data).tolist()

        self._mngr.save(
            step,
            args=ocp.args.Composite(
                state=ocp.args.StandardSave({
                    "objective": _to_numpy(objective),
                    "probe": _to_numpy(probe),
                    "opt_state": _to_numpy(opt_state),
                }),
                encoder=ocp.args.StandardSave(encoder_np),
                metadata=ocp.args.JsonSave(metadata),
            ),
            force=force,
        )

    @beartype.beartype
    def should_save(self, step: int, *, force: bool = False) -> bool:
        if step in self._mngr.all_steps():
            return False
        if force:
            return True
        return bool(self._mngr.should_save(step))

    @beartype.beartype
    def load_training(self, encoder, objective, probe, opt_state, *, encoder_config):
        """Load latest checkpoint for training resume.

        Returns (encoder, objective, probe, opt_state, start_step, prng_key) or None if no checkpoint.

        Args:
            encoder: Abstract encoder structure for restore.
            objective: Abstract objective structure for restore.
            probe: Abstract probe structure for restore.
            opt_state: Abstract optimizer state structure for restore.
            encoder_config: Current encoder config dataclass. Must match saved config.

        Note: PRNG key is only restored if saved in metadata. Caller should handle None.
        """
        step = self._mngr.latest_step()
        if step is None:
            return None

        abstract_state = {
            "objective": objective,
            "probe": probe,
            "opt_state": opt_state,
        }
        restored = self._mngr.restore(
            step,
            args=ocp.args.Composite(
                state=ocp.args.StandardRestore(abstract_state),
                encoder=ocp.args.StandardRestore(encoder),
                metadata=ocp.args.JsonRestore(),
            ),
        )

        # Validate config matches to catch silent mismatches in static fields
        saved_config = restored["metadata"]["encoder_config"]
        current_config = dataclasses.asdict(encoder_config)
        assert saved_config == current_config, (
            f"Encoder config mismatch: checkpoint has {saved_config}, current is {current_config}"
        )

        # Verify encoder norm matches what was saved
        restored_encoder = restored["encoder"]
        restored_params = eqx.filter(restored_encoder, eqx.is_inexact_array)
        restored_norm = float(birdjepa.helpers.tree_l2_norm(restored_params))
        saved_norm = restored["metadata"].get("encoder_norm")

        if saved_norm is not None:
            ratio = saved_norm / restored_norm if restored_norm > 0 else float("inf")
            delta = abs(saved_norm - restored_norm)
            logger.info(
                "CKPT_RESTORE step=%d saved_encoder_norm=%.6f restored_encoder_norm=%.6f ratio=%.6f delta=%.6f",
                restored["metadata"]["step"],
                saved_norm,
                restored_norm,
                ratio,
                delta,
            )
            if delta > 1.0:
                logger.warning(
                    "CKPT_MISMATCH: encoder_norm changed by %.2f (ratio=%.4f) during checkpoint restore!",
                    delta,
                    ratio,
                )
        else:
            logger.info(
                "Loaded checkpoint step=%d (no encoder_norm in metadata)",
                restored["metadata"]["step"],
            )

        opt_state = restored["state"]["opt_state"]
        opt_stats = _opt_state_stats(opt_state)
        saved_opt_l2 = restored["metadata"].get("opt_state_l2")
        saved_opt_abs_mean = restored["metadata"].get("opt_state_abs_mean")
        saved_opt_abs_max = restored["metadata"].get("opt_state_abs_max")
        saved_counts = restored["metadata"].get("opt_state_count_values")
        logger.info(
            "CKPT_RESTORE step=%d opt_state_l2=%.6f abs_mean=%.6f abs_max=%.6f count_values=%s",
            restored["metadata"]["step"],
            _none_to_nan(opt_stats["l2"]),
            _none_to_nan(opt_stats["abs_mean"]),
            _none_to_nan(opt_stats["abs_max"]),
            opt_stats["count_values"],
        )
        if saved_opt_l2 is not None and opt_stats["l2"] is not None:
            delta = abs(saved_opt_l2 - opt_stats["l2"])
            ratio = (
                saved_opt_l2 / opt_stats["l2"] if opt_stats["l2"] > 0 else float("inf")
            )
            logger.info(
                "CKPT_OPT_STATE_DELTA step=%d saved_l2=%.6f restored_l2=%.6f ratio=%.6f delta=%.6f",
                restored["metadata"]["step"],
                saved_opt_l2,
                opt_stats["l2"],
                ratio,
                delta,
            )
        if saved_opt_abs_mean is not None and opt_stats["abs_mean"] is not None:
            delta = abs(saved_opt_abs_mean - opt_stats["abs_mean"])
            logger.info(
                "CKPT_OPT_STATE_ABS_MEAN_DELTA step=%d saved=%.6f restored=%.6f delta=%.6f",
                restored["metadata"]["step"],
                saved_opt_abs_mean,
                opt_stats["abs_mean"],
                delta,
            )
        if saved_opt_abs_max is not None and opt_stats["abs_max"] is not None:
            delta = abs(saved_opt_abs_max - opt_stats["abs_max"])
            logger.info(
                "CKPT_OPT_STATE_ABS_MAX_DELTA step=%d saved=%.6f restored=%.6f delta=%.6f",
                restored["metadata"]["step"],
                saved_opt_abs_max,
                opt_stats["abs_max"],
                delta,
            )
        if saved_counts is not None and opt_stats["count_values"]:
            if saved_counts != opt_stats["count_values"]:
                logger.warning(
                    "CKPT_COUNT_MISMATCH step=%d saved_counts=%s restored_counts=%s",
                    restored["metadata"]["step"],
                    saved_counts,
                    opt_stats["count_values"],
                )
        if opt_stats["count_values"]:
            step_val = int(restored["metadata"]["step"])
            if not all(count == step_val for count in opt_stats["count_values"]):
                logger.warning(
                    "CKPT_COUNT_STEP_MISMATCH step=%d opt_state_counts=%s",
                    step_val,
                    opt_stats["count_values"],
                )

        prng_key = None
        saved_prng_key = restored["metadata"].get("prng_key")
        if saved_prng_key is not None:
            msg = "jax.random.wrap_key_data is required to restore PRNG key"
            assert hasattr(jax.random, "wrap_key_data"), msg
            key_data = jnp.array(saved_prng_key, dtype=jnp.uint32)
            prng_key = jax.random.wrap_key_data(key_data)

        return (
            restored_encoder,
            restored["state"]["objective"],
            restored["state"]["probe"],
            restored["state"]["opt_state"],
            restored["metadata"]["step"],
            prng_key,
        )

    def wait_until_finished(self):
        """Wait for async checkpoint saves to complete."""
        self._mngr.wait_until_finished()


@beartype.beartype
def load_eval(ckpt_path: pathlib.Path, encoder_cls, transformer_config_cls):
    """Load just the encoder from a checkpoint for evaluation/inference.

    Args:
        ckpt_path: Path to checkpoint directory (e.g., /path/to/wandb_run_id).
        encoder_cls: TransformerModel class to instantiate.
        transformer_config_cls: Transformer config dataclass.

    Returns:
        (encoder, config): The loaded encoder model and its config.
    """
    import jax.random as jr

    mngr = ocp.CheckpointManager(
        ckpt_path,
        options=ocp.CheckpointManagerOptions(single_host_load_and_broadcast=True),
        item_names=("state", "encoder", "metadata"),
    )
    step = mngr.latest_step()
    assert step is not None, f"No checkpoint found at {ckpt_path}"

    # Load metadata first to get encoder config
    restored = mngr.restore(
        step, args=ocp.args.Composite(metadata=ocp.args.JsonRestore())
    )
    config = transformer_config_cls(**restored["metadata"]["encoder_config"])

    # Load encoder with abstract structure
    abstract_encoder = encoder_cls(config, key=jr.key(0))
    restored = mngr.restore(
        step,
        args=ocp.args.Composite(encoder=ocp.args.StandardRestore(abstract_encoder)),
    )
    logger.info("Loaded encoder from %s (step %d)", ckpt_path, step)
    return restored["encoder"], config
