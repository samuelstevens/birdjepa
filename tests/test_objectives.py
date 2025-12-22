"""Tests for training objectives and transformer API."""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import birdjepa.nn.transformer as transformer
import birdjepa.nn.objectives as objectives


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


def test_transformer_new_api_smoke():
    """Transformer forward with (B, T, kernel) + grid API."""
    cfg = transformer.Config(
        input_h=32, input_w=32, patch_h=4, patch_w=4, embed_dim=64, depth=2, n_heads=4
    )
    model = transformer.Transformer(cfg)

    # Patchify input
    x_bhw = torch.randn(2, 32, 32)
    x_bnk, grid_bn2 = transformer.patchify(x_bhw, cfg)

    # Forward with new API (returns dict)
    out = model(x_bnk, grid=grid_bn2)

    # Check shapes
    assert out["cls"].shape == (2, 1, 64)  # (B, n_cls_tokens, D)
    assert out["patches"].shape == (2, 64, 64)  # (B, n_patches, D)
    assert out["reg"].shape == (2, 0, 64)  # (B, n_reg_tokens, D)


def test_transformer_visible_patches_only():
    """Transformer should work with subset of patches (for MAE encoder)."""
    cfg = transformer.Config(
        input_h=32, input_w=32, patch_h=4, patch_w=4, embed_dim=64, depth=2, n_heads=4
    )
    model = transformer.Transformer(cfg)

    # Patchify and take only first 16 patches (25% visible)
    x_bhw = torch.randn(2, 32, 32)
    x_bnk, grid_bn2 = transformer.patchify(x_bhw, cfg)
    x_visible = x_bnk[:, :16]
    grid_visible = grid_bn2[:, :16]

    # Forward with visible patches only (returns dict)
    out = model(x_visible, grid=grid_visible)

    # Check shapes - should match number of visible patches
    assert out["cls"].shape == (2, 1, 64)  # (B, n_cls_tokens, D)
    assert out["patches"].shape == (2, 16, 64)  # (B, n_visible, D)


def test_block_masking():
    """Block masking creates contiguous masked regions."""
    # 8x8 patch grid, 2x2 blocks, 75% mask ratio
    n_h, n_w = 8, 8
    block_size = 2
    mask_ratio = 0.75
    batch_size = 4

    generator = torch.Generator().manual_seed(42)
    mask_bn = objectives.make_block_mask(
        batch_size, n_h, n_w, block_size, mask_ratio, generator=generator
    )

    # Check shape
    assert mask_bn.shape == (batch_size, n_h * n_w)

    # Check mask ratio is approximately correct (exact due to adjustment)
    n_masked = mask_bn.sum(dim=1).float()
    expected_masked = int(n_h * n_w * mask_ratio)
    assert (n_masked == expected_masked).all(), (
        f"Expected {expected_masked}, got {n_masked}"
    )


def test_block_masking_reproducible():
    """Block masking is reproducible with same seed."""
    n_h, n_w = 8, 8
    block_size = 2
    mask_ratio = 0.75
    batch_size = 4

    gen1 = torch.Generator().manual_seed(123)
    mask1 = objectives.make_block_mask(
        batch_size, n_h, n_w, block_size, mask_ratio, generator=gen1
    )

    gen2 = torch.Generator().manual_seed(123)
    mask2 = objectives.make_block_mask(
        batch_size, n_h, n_w, block_size, mask_ratio, generator=gen2
    )

    assert torch.equal(mask1, mask2)


def test_pixio_forward_smoke():
    """Pixio forward pass runs without errors."""
    model_cfg = transformer.Config(
        input_h=32, input_w=32, patch_h=4, patch_w=4, embed_dim=64, depth=2, n_heads=4
    )
    pixio_cfg = objectives.PixioConfig(
        decoder_depth=2, decoder_dim=32, mask_ratio=0.75, block_size=2
    )

    encoder = transformer.Transformer(model_cfg)
    pixio = objectives.Pixio(model_cfg, pixio_cfg)

    batch = {"data": torch.randn(2, 32, 32), "target": torch.tensor([0, 1])}

    losses, emb, targets = pixio(batch, encoder)

    # Check loss exists and is scalar
    assert "mse" in losses
    assert losses["mse"].shape == ()

    # Check embeddings shape
    assert emb.shape[0] == 2
    assert emb.shape[1] == 64  # embed_dim

    # Check targets preserved
    assert targets.shape == (2,)


def test_pixio_loss_nonzero():
    """Pixio loss should be non-zero (actually reconstructing)."""
    model_cfg = transformer.Config(
        input_h=32, input_w=32, patch_h=4, patch_w=4, embed_dim=64, depth=2, n_heads=4
    )
    pixio_cfg = objectives.PixioConfig(
        decoder_depth=2, decoder_dim=32, mask_ratio=0.75, block_size=2
    )

    encoder = transformer.Transformer(model_cfg)
    pixio = objectives.Pixio(model_cfg, pixio_cfg)

    batch = {"data": torch.randn(2, 32, 32), "target": torch.tensor([0, 1])}

    losses, _, _ = pixio(batch, encoder)

    # Loss should be positive (MSE of random predictions vs targets)
    assert losses["mse"] > 0


def test_supervised_forward_smoke():
    """Supervised objective forward pass with new API."""
    model_cfg = transformer.Config(
        input_h=32, input_w=32, patch_h=4, patch_w=4, embed_dim=64, depth=2, n_heads=4
    )

    encoder = transformer.Transformer(model_cfg)
    supervised = objectives.Supervised(model_cfg, n_classes=10)

    batch = {"data": torch.randn(2, 32, 32), "target": torch.tensor([0, 1])}

    losses, emb, targets = supervised(batch, encoder)

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

    encoder = transformer.Transformer(model_cfg)
    lejepa = objectives.LeJEPA(model_cfg, lejepa_cfg)

    # LeJEPA expects multi-view input
    batch = {"views": torch.randn(2, 2, 32, 32), "target": torch.tensor([0, 1])}

    losses, emb, targets = lejepa(batch, encoder)

    assert "inv" in losses
    assert "sigreg" in losses
    assert emb.shape == (4, 64)  # 2 samples x 2 views
    assert targets.shape == (4,)
