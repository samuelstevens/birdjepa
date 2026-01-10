"""Checkpoint save/load utilities for BirdJEPA models.

Wraps Orbax checkpoint functionality for:
- Training: save/restore full state (encoder, objective, probe, opt_state)
- Inference: restore just the encoder with config
"""

import dataclasses
import logging
import pathlib

import beartype
import orbax.checkpoint as ocp

logger = logging.getLogger(__name__)


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
            force: If True, save even if step doesn't match save_interval.
        """
        self._mngr.save(
            step,
            args=ocp.args.Composite(
                state=ocp.args.StandardSave({
                    "objective": objective,
                    "probe": probe,
                    "opt_state": opt_state,
                }),
                encoder=ocp.args.StandardSave(encoder),
                metadata=ocp.args.JsonSave({
                    "step": step,
                    "encoder_config": dataclasses.asdict(encoder_config),
                }),
            ),
            force=force,
        )

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

        logger.info("Loaded checkpoint step=%d", restored["metadata"]["step"])
        return (
            restored["encoder"],
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
