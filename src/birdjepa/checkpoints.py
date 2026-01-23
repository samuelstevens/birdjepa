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
            param_norm: Optional param_norm for the saved state (post-update).
            force: If True, save even if step doesn't match save_interval.
        """
        # Skip if checkpoint already exists at this step (force only bypasses
        # save_interval check, not the "already exists" check)
        if step in self._mngr.all_steps():
            logger.debug("Checkpoint for step %d already exists, skipping", step)
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

        metadata = {
            "step": step,
            "encoder_config": dataclasses.asdict(encoder_config),
            "encoder_norm": encoder_norm,  # For verification on restore
        }
        if param_norm is not None:
            metadata["param_norm"] = float(param_norm)

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

        Returns (encoder, objective, probe, opt_state, start_step) or None if no checkpoint.

        Args:
            encoder: Abstract encoder structure for restore.
            objective: Abstract objective structure for restore.
            probe: Abstract probe structure for restore.
            opt_state: Abstract optimizer state structure for restore.
            encoder_config: Current encoder config dataclass. Must match saved config.

        Note: JAX PRNG key is not checkpointed (host-local, can't serialize in distributed mode).
        Caller should continue using existing key after resume.
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

        return (
            restored_encoder,
            restored["state"]["objective"],
            restored["state"]["probe"],
            restored["state"]["opt_state"],
            restored["metadata"]["step"],
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
