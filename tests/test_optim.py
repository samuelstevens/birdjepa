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


def test_wsd_schedule_guards_duplicate_boundaries(monkeypatch):
    """wsd_schedule should avoid duplicate join_schedules boundaries."""
    original_join = pretrain.optax.join_schedules
    boundaries_record: dict[str, list[int]] = {}

    def join_schedules(schedules, boundaries):
        boundaries_record["boundaries"] = list(boundaries)
        return original_join(schedules, boundaries)

    monkeypatch.setattr(pretrain.optax, "join_schedules", join_schedules)

    try:
        pretrain.wsd_schedule(
            peak_value=1.0,
            total_steps=10,
            warmup_steps=3,
            decay_steps=7,
            end_value=0.0,
        )
    except AssertionError:
        return

    boundaries = boundaries_record.get("boundaries")
    assert boundaries is not None, "Expected join_schedules to be called."
    assert len(boundaries) == len(set(boundaries)), (
        f"Duplicate boundaries: {boundaries}"
    )
