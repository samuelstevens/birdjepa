"""Tests for optimizer functionality (Muon, AdamW)."""

import jax
import jax.numpy as jnp
import optax

import birdjepa.pretrain as pretrain


# -----------------------------------------------------------------------------
# Muon optimizer tests
# -----------------------------------------------------------------------------


def test_muon_config_parsing():
    """optimizer='muon' is a valid config option."""
    cfg = pretrain.Config(optimizer="muon")
    assert cfg.optimizer == "muon"


def test_adamw_config_default():
    """Pretrain config has optimizer field with 'adamw' as default."""
    cfg = pretrain.Config()
    assert cfg.optimizer == "adamw"


def test_muon_optimizer_initialization():
    """optax.contrib.muon initializes correctly."""
    schedule = optax.constant_schedule(1e-3)
    optim = optax.contrib.muon(learning_rate=schedule)

    # Test with mixed param shapes (2D -> muon, 1D -> adamw fallback)
    params = {
        "weight_2d": jnp.ones((64, 64)),
        "bias_1d": jnp.ones((64,)),
    }

    state = optim.init(params)
    assert state is not None


def test_muon_optimizer_step():
    """Muon optimizer produces valid gradient updates."""
    schedule = optax.constant_schedule(1e-3)
    optim = optax.contrib.muon(learning_rate=schedule)

    params = {"weight": jnp.ones((32, 32)), "bias": jnp.ones((32,))}
    grads = {"weight": jnp.ones((32, 32)) * 0.1, "bias": jnp.ones((32,)) * 0.1}

    state = optim.init(params)
    updates, new_state = optim.update(grads, state, params)

    # Updates should be finite
    for v in jax.tree_util.tree_leaves(updates):
        assert not jnp.any(jnp.isnan(v))
        assert not jnp.any(jnp.isinf(v))
