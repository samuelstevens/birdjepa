"""PyTorch tests for transformer utilities."""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import birdjepa.nn.transformer_pt as transformer


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
    x_bhw = torch.randn(batch_size, h, w)

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
    assert torch.allclose(x_recovered, x_bhw)
