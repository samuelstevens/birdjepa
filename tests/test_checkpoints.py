"""Tests for checkpoint save/load functionality."""

import pathlib
import tempfile

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import birdjepa.checkpoints as ckpt
import birdjepa.nn.transformer as transformer


# Small configs for fast tests
SMALL_CONFIGS = [
    transformer.Transformer(
        input_h=64,
        input_w=32,
        patch_h=16,
        patch_w=16,
        embed_dim=32,
        depth=2,
        n_heads=2,
    ),
    transformer.Transformer(
        input_h=128,
        input_w=64,
        patch_h=16,
        patch_w=16,
        embed_dim=64,
        depth=3,
        n_heads=4,
    ),
]

SEEDS = [0, 42, 123]


def arrays_equal(tree1, tree2) -> bool:
    """Check if two pytrees have identical array values."""
    leaves1 = jax.tree.leaves(tree1)
    leaves2 = jax.tree.leaves(tree2)
    if len(leaves1) != len(leaves2):
        return False
    for l1, l2 in zip(leaves1, leaves2):
        if eqx.is_array(l1) and eqx.is_array(l2):
            if not jnp.allclose(l1, l2):
                return False
        elif l1 != l2:
            return False
    return True


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("config", SMALL_CONFIGS)
def test_encoder_roundtrip(seed: int, config: transformer.Transformer):
    """Test that encoder survives save/load roundtrip with identical params."""
    key = jr.key(seed)

    # Create encoder
    encoder = transformer.TransformerModel(config, key=key)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = pathlib.Path(tmpdir)

        # Create minimal objective/probe for save (not testing these)
        key, subkey = jr.split(key)
        dummy_linear = eqx.nn.Linear(config.embed_dim, 10, key=subkey)

        # Create checkpoint manager and save
        mngr = ckpt.CheckpointManager(
            ckpt_path, save_interval_steps=1, max_to_keep=1, async_checkpointing=False
        )
        mngr.save(
            step=1,
            encoder=encoder,
            objective=dummy_linear,
            probe=dummy_linear,
            opt_state={},
            encoder_config=config,
            force=True,
        )
        mngr.wait_until_finished()

        # Load encoder for inference
        loaded_encoder, loaded_config = ckpt.load_eval(
            ckpt_path, transformer.TransformerModel, transformer.Transformer
        )

        # Verify config matches
        assert loaded_config == config

        # Verify encoder params match
        assert arrays_equal(encoder, loaded_encoder), "Encoder params differ after load"


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("config", SMALL_CONFIGS)
def test_training_state_roundtrip(seed: int, config: transformer.Transformer):
    """Test that full training state survives save/load roundtrip."""
    key = jr.key(seed)

    # Create models
    key, k1, k2, k3 = jr.split(key, 4)
    encoder = transformer.TransformerModel(config, key=k1)
    objective = eqx.nn.Linear(config.embed_dim, config.embed_dim, key=k2)
    probe = eqx.nn.Linear(config.embed_dim, 10, key=k3)
    opt_state = {"dummy": jnp.array([1.0, 2.0, 3.0])}

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = pathlib.Path(tmpdir)

        # Save
        mngr = ckpt.CheckpointManager(
            ckpt_path, save_interval_steps=1, max_to_keep=1, async_checkpointing=False
        )
        mngr.save(
            step=42,
            encoder=encoder,
            objective=objective,
            probe=probe,
            opt_state=opt_state,
            encoder_config=config,
            force=True,
        )
        mngr.wait_until_finished()

        # Create new manager for loading (simulates new process)
        mngr2 = ckpt.CheckpointManager(
            ckpt_path, save_interval_steps=1, max_to_keep=1, async_checkpointing=False
        )

        # Create abstract state for restore
        key, k1, k2, k3 = jr.split(key, 4)
        abstract_encoder = transformer.TransformerModel(config, key=k1)
        abstract_objective = eqx.nn.Linear(config.embed_dim, config.embed_dim, key=k2)
        abstract_probe = eqx.nn.Linear(config.embed_dim, 10, key=k3)
        abstract_opt_state = {"dummy": jnp.zeros(3)}

        # Load
        result = mngr2.load_training(
            abstract_encoder,
            abstract_objective,
            abstract_probe,
            abstract_opt_state,
            encoder_config=config,
        )
        assert result is not None
        (
            loaded_encoder,
            loaded_objective,
            loaded_probe,
            loaded_opt_state,
            loaded_step,
        ) = result

        # Verify step
        assert loaded_step == 42

        # Verify all params match
        assert arrays_equal(encoder, loaded_encoder), "Encoder params differ"
        assert arrays_equal(objective, loaded_objective), "Objective params differ"
        assert arrays_equal(probe, loaded_probe), "Probe params differ"
        assert jnp.allclose(opt_state["dummy"], loaded_opt_state["dummy"]), (
            "Opt state differs"
        )


def test_load_training_empty_returns_none():
    """Test that load_training from empty directory returns None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = pathlib.Path(tmpdir)
        mngr = ckpt.CheckpointManager(ckpt_path, async_checkpointing=False)

        # Should return None when no checkpoint exists
        key = jr.key(0)
        config = SMALL_CONFIGS[0]
        encoder = transformer.TransformerModel(config, key=key)
        dummy = eqx.nn.Linear(config.embed_dim, 10, key=key)

        result = mngr.load_training(encoder, dummy, dummy, {}, encoder_config=config)
        assert result is None


def test_load_eval_empty_raises():
    """Test that load_eval from empty directory raises AssertionError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = pathlib.Path(tmpdir)

        with pytest.raises(AssertionError, match="No checkpoint found"):
            ckpt.load_eval(
                ckpt_path, transformer.TransformerModel, transformer.Transformer
            )


def test_load_training_config_mismatch_raises():
    """Test that load_training raises when config doesn't match saved checkpoint."""
    key = jr.key(0)
    save_config = SMALL_CONFIGS[0]

    # Create a different config with same array shapes but different static field
    load_config = transformer.Transformer(
        input_h=save_config.input_h,
        input_w=save_config.input_w,
        patch_h=save_config.patch_h,
        patch_w=save_config.patch_w,
        embed_dim=save_config.embed_dim,
        depth=save_config.depth,
        n_heads=save_config.n_heads,
        dropout=0.5,  # Different static field (save_config has dropout=0.0)
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = pathlib.Path(tmpdir)

        # Save with original config
        key, k1 = jr.split(key)
        encoder = transformer.TransformerModel(save_config, key=k1)
        dummy = eqx.nn.Linear(save_config.embed_dim, 10, key=key)

        mngr = ckpt.CheckpointManager(
            ckpt_path, save_interval_steps=1, max_to_keep=1, async_checkpointing=False
        )
        mngr.save(
            step=1,
            encoder=encoder,
            objective=dummy,
            probe=dummy,
            opt_state={},
            encoder_config=save_config,
            force=True,
        )
        mngr.wait_until_finished()

        # Try to load with different config
        mngr2 = ckpt.CheckpointManager(
            ckpt_path, save_interval_steps=1, max_to_keep=1, async_checkpointing=False
        )
        key, k1 = jr.split(key)
        abstract_encoder = transformer.TransformerModel(load_config, key=k1)

        with pytest.raises(AssertionError, match="Encoder config mismatch"):
            mngr2.load_training(
                abstract_encoder, dummy, dummy, {}, encoder_config=load_config
            )
