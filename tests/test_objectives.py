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


def test_pixio_fwd_smoke():
    """Pixio forward pass runs without errors."""
    model_cfg = transformer.Transformer(
        input_h=16, input_w=16, patch_h=4, patch_w=4, embed_dim=32, depth=1, n_heads=2
    )
    pixio_cfg = objectives.PixioConfig(
        decoder_depth=1, decoder_dim=16, decoder_heads=2, mask_ratio=0.75, block_size=2
    )

    key = jax.random.key(0)
    enc_key, obj_key, data_key, fwd_key = jax.random.split(key, 4)
    encoder = transformer.TransformerModel(model_cfg, key=enc_key)
    pixio = objectives.Pixio(model_cfg, pixio_cfg, key=obj_key)

    batch = {
        "data": jax.random.normal(data_key, (2, 16, 16)),
        "target": jnp.array([0, 1]),
    }

    losses, out, targets = pixio(batch, encoder, key=fwd_key, mode="train")

    # Check loss exists and is scalar
    assert "mse" in losses
    assert losses["mse"].shape == ()

    # Check embeddings shape
    assert out.cls.shape == (2, 1, 32)
    assert out.patches.shape[0] == 2
    assert out.patches.shape[2] == 32

    # Check targets preserved
    assert targets.shape == (2,)


def test_pixio_loss_nonzero():
    """Pixio loss should be non-zero (actually reconstructing)."""
    model_cfg = transformer.Transformer(
        input_h=16, input_w=16, patch_h=4, patch_w=4, embed_dim=32, depth=1, n_heads=2
    )
    pixio_cfg = objectives.PixioConfig(
        decoder_depth=1, decoder_dim=16, decoder_heads=2, mask_ratio=0.75, block_size=2
    )

    key = jax.random.key(0)
    enc_key, obj_key, data_key, fwd_key = jax.random.split(key, 4)
    encoder = transformer.TransformerModel(model_cfg, key=enc_key)
    pixio = objectives.Pixio(model_cfg, pixio_cfg, key=obj_key)

    batch = {
        "data": jax.random.normal(data_key, (2, 16, 16)),
        "target": jnp.array([0, 1]),
    }

    losses, _, _ = pixio(batch, encoder, key=fwd_key, mode="train")

    assert bool(losses["mse"] > 0)


def test_pixio_visible_count_matches_mask_count():
    """Pixio should keep all unmasked patches when mask count is rounded."""
    model_cfg = transformer.Transformer(
        input_h=12, input_w=12, patch_h=4, patch_w=4, embed_dim=16, depth=1, n_heads=2
    )
    pixio_cfg = objectives.PixioConfig(
        decoder_depth=1, decoder_dim=8, decoder_heads=2, mask_ratio=0.5, block_size=1
    )

    key = jax.random.key(0)
    enc_key, obj_key, data_key, fwd_key = jax.random.split(key, 4)
    encoder = transformer.TransformerModel(model_cfg, key=enc_key)
    pixio = objectives.Pixio(model_cfg, pixio_cfg, key=obj_key)

    batch = {
        "data": jax.random.normal(data_key, (2, 12, 12)),
        "target": jnp.array([0, 1]),
    }

    n_h = model_cfg.input_h // model_cfg.patch_h
    n_w = model_cfg.input_w // model_cfg.patch_w
    n_patches = n_h * n_w
    n_masked_target = int(n_patches * pixio_cfg.mask_ratio)
    assert n_patches * pixio_cfg.mask_ratio != n_masked_target

    mask_key, _, _ = jax.random.split(fwd_key, 3)
    mask_bn = objectives.make_block_mask(
        batch["data"].shape[0],
        n_h,
        n_w,
        pixio_cfg.block_size,
        pixio_cfg.mask_ratio,
        key=mask_key,
    )
    n_masked = int(mask_bn.sum(axis=1)[0].item())
    assert bool(jnp.all(mask_bn.sum(axis=1) == n_masked))

    losses, out, _ = pixio(batch, encoder, key=fwd_key, mode="train")
    assert "mse" in losses
    assert out.patches.shape[1] == n_patches - n_masked


def test_supervised_fwd_smoke():
    """Supervised objective forward pass with new API."""
    model_cfg = transformer.Transformer(
        input_h=32, input_w=32, patch_h=4, patch_w=4, embed_dim=64, depth=2, n_heads=4
    )

    key = jax.random.key(0)
    enc_key, obj_key, data_key, fwd_key = jax.random.split(key, 4)
    encoder = transformer.TransformerModel(model_cfg, key=enc_key)
    supervised = objectives.Supervised(model_cfg, n_classes=10, key=obj_key)

    batch = {
        "data": jax.random.normal(data_key, (2, 32, 32)),
        "target": jnp.array([0, 1]),
    }

    losses, out, targets = supervised(batch, encoder, key=fwd_key, mode="train")

    assert "ce" in losses
    assert losses["ce"].shape == ()
    assert out.cls.shape == (2, 1, 64)
    assert targets.shape == (2,)


def test_eval_disables_encoder_dropout():
    """Eval should be deterministic even with different keys."""
    model_cfg = transformer.Transformer(
        input_h=16,
        input_w=16,
        patch_h=4,
        patch_w=4,
        embed_dim=32,
        depth=1,
        n_heads=2,
        dropout=0.5,
    )

    key = jax.random.key(0)
    enc_key, obj_key, data_key, eval_key1, eval_key2 = jax.random.split(key, 5)
    encoder = transformer.TransformerModel(model_cfg, key=enc_key)
    supervised = objectives.Supervised(model_cfg, n_classes=3, key=obj_key)

    batch = {
        "data": jax.random.normal(data_key, (2, 16, 16)),
        "target": jnp.array([0, 1]),
    }

    _, out1, _ = supervised(batch, encoder, key=eval_key1, mode="eval")
    _, out2, _ = supervised(batch, encoder, key=eval_key2, mode="eval")

    assert bool(jnp.array_equal(out1.cls, out2.cls))


def test_lejepa_fwd_smoke():
    """LeJEPA objective forward pass with new API."""
    model_cfg = transformer.Transformer(
        input_h=16, input_w=16, patch_h=4, patch_w=4, embed_dim=32, depth=1, n_heads=2
    )
    lejepa_cfg = objectives.LeJEPAConfig(n_views=2, proj_dim=8)

    key = jax.random.key(0)
    enc_key, obj_key, data_key, fwd_key = jax.random.split(key, 4)
    encoder = transformer.TransformerModel(model_cfg, key=enc_key)
    lejepa = objectives.LeJEPA(model_cfg, lejepa_cfg, key=obj_key)

    batch = {
        "views": jax.random.normal(data_key, (2, 2, 16, 16)),
        "target": jnp.array([0, 1]),
    }

    losses, out, targets = lejepa(batch, encoder, key=fwd_key, mode="train")

    assert "inv" in losses
    assert "sigreg" in losses
    assert out.cls.shape == (2, 2, 1, 32)
    assert targets.shape == (2, 2)
