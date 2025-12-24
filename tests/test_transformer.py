"""JAX tests for transformer utilities."""

from hypothesis import given, settings
from hypothesis import strategies as st

import jax
import jax.numpy as jnp

from birdjepa.nn import transformer


def test_transformer_forward_smoke():
    """Transformer forward with (B, T, kernel) + grid API."""
    cfg = transformer.Config(
        input_h=32, input_w=32, patch_h=4, patch_w=4, embed_dim=64, depth=2, n_heads=4
    )
    key = jax.random.key(0)
    model_key, data_key, fwd_key = jax.random.split(key, 3)
    model = transformer.Transformer(cfg, key=model_key)

    x_bhw = jax.random.normal(data_key, (2, 32, 32))
    x_bnk, grid_bn2 = transformer.patchify(x_bhw, cfg)

    out = model(x_bnk, grid=grid_bn2, key=fwd_key)

    assert out["cls"].shape == (2, 1, 64)
    assert out["patches"].shape == (2, 64, 64)
    assert out["reg"].shape == (2, 0, 64)


def test_transformer_visible_patches_only():
    """Transformer should work with subset of patches (for MAE encoder)."""
    cfg = transformer.Config(
        input_h=32, input_w=32, patch_h=4, patch_w=4, embed_dim=64, depth=2, n_heads=4
    )
    key = jax.random.key(0)
    model_key, data_key, fwd_key = jax.random.split(key, 3)
    model = transformer.Transformer(cfg, key=model_key)

    x_bhw = jax.random.normal(data_key, (2, 32, 32))
    x_bnk, grid_bn2 = transformer.patchify(x_bhw, cfg)
    x_visible = x_bnk[:, :16]
    grid_visible = grid_bn2[:, :16]

    out = model(x_visible, grid=grid_visible, key=fwd_key)

    assert out["cls"].shape == (2, 1, 64)
    assert out["patches"].shape == (2, 16, 64)


@st.composite
def patch_config(draw):
    """Generate valid (h, w, patch_h, patch_w) where dimensions are divisible."""
    patch_h = draw(st.sampled_from([2, 4, 8, 16]))
    patch_w = draw(st.sampled_from([2, 4, 8, 16]))
    n_patches_h = draw(st.integers(min_value=2, max_value=16))
    n_patches_w = draw(st.integers(min_value=2, max_value=16))
    h = patch_h * n_patches_h
    w = patch_w * n_patches_w
    return h, w, patch_h, patch_w


@given(config=patch_config(), batch_size=st.integers(min_value=1, max_value=4))
@settings(max_examples=100, deadline=None)
def test_patchify_roundtrip(config, batch_size):
    """Patchify then unpatchify should recover original."""
    h, w, patch_h, patch_w = config
    cfg = transformer.Config(input_h=h, input_w=w, patch_h=patch_h, patch_w=patch_w)
    key = jax.random.key(0)
    x_bhw = jax.random.normal(key, (batch_size, h, w))

    x_bnk, grid_bn2 = transformer.patchify(x_bhw, cfg)

    # Check shapes
    n_patches_h = h // patch_h
    n_patches_w = w // patch_w
    n_patches = n_patches_h * n_patches_w
    kernel = patch_h * patch_w
    assert x_bnk.shape == (batch_size, n_patches, kernel)
    assert grid_bn2.shape == (batch_size, n_patches, 2)

    # Check grid coordinates cover full grid
    assert grid_bn2[0, :, 0].max() == n_patches_h - 1
    assert grid_bn2[0, :, 1].max() == n_patches_w - 1

    # Unpatchify and check roundtrip
    x_recovered = transformer.unpatchify(x_bnk, grid_bn2, cfg)
    assert x_recovered.shape == x_bhw.shape
    assert bool(jnp.allclose(x_recovered, x_bhw))
