"""JAX tests for training objectives."""

import jax
import jax.numpy as jnp

import birdjepa.nn.objectives as objectives
import birdjepa.nn.transformer as transformer


def test_block_masking():
    """Block masking creates contiguous masked regions."""
    # 8x8 patch grid, 2x2 blocks, 75% mask ratio
    n_h, n_w = 8, 8
    block_size = 2
    mask_ratio = 0.75
    batch_size = 4

    key = jax.random.key(42)
    mask_bn = objectives.make_block_mask(
        batch_size, n_h, n_w, block_size, mask_ratio, key=key
    )

    # Check shape
    assert mask_bn.shape == (batch_size, n_h * n_w)

    # Check mask ratio is approximately correct (exact due to adjustment)
    n_masked = mask_bn.sum(axis=1)
    expected_masked = int(n_h * n_w * mask_ratio)
    assert bool(jnp.all(n_masked == expected_masked))


def test_block_masking_reproducible():
    """Block masking is reproducible with same seed."""
    n_h, n_w = 8, 8
    block_size = 2
    mask_ratio = 0.75
    batch_size = 4

    key1 = jax.random.key(123)
    mask1 = objectives.make_block_mask(
        batch_size, n_h, n_w, block_size, mask_ratio, key=key1
    )

    key2 = jax.random.key(123)
    mask2 = objectives.make_block_mask(
        batch_size, n_h, n_w, block_size, mask_ratio, key=key2
    )

    assert bool(jnp.array_equal(mask1, mask2))


def test_pixio_forward_smoke():
    """Pixio forward pass runs without errors."""
    model_cfg = transformer.Config(
        input_h=16, input_w=16, patch_h=4, patch_w=4, embed_dim=32, depth=1, n_heads=2
    )
    pixio_cfg = objectives.PixioConfig(
        decoder_depth=1, decoder_dim=16, decoder_heads=2, mask_ratio=0.75, block_size=2
    )

    key = jax.random.key(0)
    enc_key, obj_key, data_key, fwd_key = jax.random.split(key, 4)
    encoder = transformer.Transformer(model_cfg, key=enc_key)
    pixio = objectives.Pixio(model_cfg, pixio_cfg, key=obj_key)

    batch = {
        "data": jax.random.normal(data_key, (2, 16, 16)),
        "target": jnp.array([0, 1]),
    }

    losses, emb, targets = pixio(batch, encoder, key=fwd_key)

    # Check loss exists and is scalar
    assert "mse" in losses
    assert losses["mse"].shape == ()

    # Check embeddings shape
    assert emb.shape[0] == 2
    assert emb.shape[1] == 32

    # Check targets preserved
    assert targets.shape == (2,)


def test_pixio_loss_nonzero():
    """Pixio loss should be non-zero (actually reconstructing)."""
    model_cfg = transformer.Config(
        input_h=16, input_w=16, patch_h=4, patch_w=4, embed_dim=32, depth=1, n_heads=2
    )
    pixio_cfg = objectives.PixioConfig(
        decoder_depth=1, decoder_dim=16, decoder_heads=2, mask_ratio=0.75, block_size=2
    )

    key = jax.random.key(0)
    enc_key, obj_key, data_key, fwd_key = jax.random.split(key, 4)
    encoder = transformer.Transformer(model_cfg, key=enc_key)
    pixio = objectives.Pixio(model_cfg, pixio_cfg, key=obj_key)

    batch = {
        "data": jax.random.normal(data_key, (2, 16, 16)),
        "target": jnp.array([0, 1]),
    }

    losses, _, _ = pixio(batch, encoder, key=fwd_key)

    assert bool(losses["mse"] > 0)


def test_supervised_forward_smoke():
    """Supervised objective forward pass with new API."""
    model_cfg = transformer.Config(
        input_h=32, input_w=32, patch_h=4, patch_w=4, embed_dim=64, depth=2, n_heads=4
    )

    key = jax.random.key(0)
    enc_key, obj_key, data_key, fwd_key = jax.random.split(key, 4)
    encoder = transformer.Transformer(model_cfg, key=enc_key)
    supervised = objectives.Supervised(model_cfg, n_classes=10, key=obj_key)

    batch = {
        "data": jax.random.normal(data_key, (2, 32, 32)),
        "target": jnp.array([0, 1]),
    }

    losses, emb, targets = supervised(batch, encoder, key=fwd_key)

    assert "ce" in losses
    assert losses["ce"].shape == ()
    assert emb.shape == (2, 64)
    assert targets.shape == (2,)


def test_lejepa_forward_smoke():
    """LeJEPA objective forward pass with new API."""
    model_cfg = transformer.Config(
        input_h=32, input_w=32, patch_h=4, patch_w=4, embed_dim=64, depth=2, n_heads=4
    )
    lejepa_cfg = objectives.LeJEPAConfig(n_views=2, proj_dim=16)

    key = jax.random.key(0)
    enc_key, obj_key, data_key, fwd_key = jax.random.split(key, 4)
    encoder = transformer.Transformer(model_cfg, key=enc_key)
    lejepa = objectives.LeJEPA(model_cfg, lejepa_cfg, key=obj_key)

    batch = {
        "views": jax.random.normal(data_key, (2, 2, 32, 32)),
        "target": jnp.array([0, 1]),
    }

    losses, emb, targets = lejepa(batch, encoder, key=fwd_key)

    assert "inv" in losses
    assert "sigreg" in losses
    assert emb.shape == (4, 64)
    assert targets.shape == (4,)
