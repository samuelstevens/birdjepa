"""PyTorch tests for training objectives and transformer API."""

import numpy as np
import torch
import typing as tp
from hypothesis import given, settings
from hypothesis import strategies as st

import birdjepa.nn.objectives as objectives_jax
import birdjepa.nn.objectives_pt as objectives
import birdjepa.nn.transformer_pt as transformer

import equinox as eqx
import jax
import jax.numpy as jnp


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


def _to_torch(x: object) -> torch.Tensor:
    return torch.from_numpy(np.array(x))


def _set_linear_pt(linear: torch.nn.Linear, weight: np.ndarray, bias: np.ndarray):
    linear.weight.data = torch.tensor(weight, dtype=linear.weight.dtype)
    linear.bias.data = torch.tensor(bias, dtype=linear.bias.dtype)


def _set_linear_jax(linear, weight: np.ndarray, bias: np.ndarray):
    linear = eqx.tree_at(lambda m: m.weight, linear, jnp.array(weight))
    linear = eqx.tree_at(lambda m: m.bias, linear, jnp.array(bias))
    return linear


def _make_dummy_encoder_pt(embed_dim: int, n_cls_tokens: int):
    def encoder(x_bnk: torch.Tensor, grid: torch.Tensor):
        b, n, _ = x_bnk.shape
        base = x_bnk.mean(dim=-1)
        cls_base = base.mean(dim=1, keepdim=True)
        cls = cls_base.unsqueeze(-1).repeat(1, n_cls_tokens, embed_dim)
        patches = base.unsqueeze(-1).repeat(1, n, embed_dim)
        reg = torch.zeros(b, 0, embed_dim, device=x_bnk.device, dtype=x_bnk.dtype)
        return {"cls": cls, "patches": patches, "reg": reg}

    return encoder


def _dummy_encoder_jax(x_bnk, grid, *, embed_dim: int, n_cls_tokens: int):
    b, n, _ = x_bnk.shape
    base = x_bnk.mean(axis=-1)
    cls_base = base.mean(axis=1, keepdims=True)
    cls = jnp.repeat(cls_base[:, None, :], n_cls_tokens, axis=1)
    cls = jnp.repeat(cls, embed_dim, axis=-1)
    patches = jnp.repeat(base[:, :, None], embed_dim, axis=-1)
    reg = jnp.zeros((b, 0, embed_dim), dtype=x_bnk.dtype)
    return {"cls": cls, "patches": patches, "reg": reg}


@st.composite
def supervised_parity_case(draw):
    combos = [
        (4, 1),
        (4, 2),
        (8, 1),
        (8, 2),
        (8, 4),
        (16, 1),
        (16, 2),
        (16, 4),
    ]
    embed_dim, n_heads = draw(st.sampled_from(combos))
    patch_h = draw(st.sampled_from([2, 4]))
    patch_w = draw(st.sampled_from([2, 4]))
    n_patches_h = draw(st.integers(min_value=1, max_value=4))
    n_patches_w = draw(st.integers(min_value=1, max_value=4))
    h = patch_h * n_patches_h
    w = patch_w * n_patches_w
    n_classes = draw(st.integers(min_value=2, max_value=8))
    batch_size = draw(st.integers(min_value=1, max_value=4))
    seed = draw(st.integers(min_value=0, max_value=2**16 - 1))
    return h, w, patch_h, patch_w, embed_dim, n_heads, n_classes, batch_size, seed


@given(case=supervised_parity_case())
@settings(max_examples=200, deadline=None)
def test_supervised_parity(case):
    (
        h,
        w,
        patch_h,
        patch_w,
        embed_dim,
        n_heads,
        n_classes,
        batch_size,
        seed,
    ) = case
    cfg = transformer.Config(
        input_h=h,
        input_w=w,
        patch_h=patch_h,
        patch_w=patch_w,
        embed_dim=embed_dim,
        depth=2,
        n_heads=n_heads,
        n_cls_tokens=1,
    )

    rng = np.random.default_rng(seed)
    data = rng.standard_normal((batch_size, h, w), dtype=np.float32)
    targets = rng.integers(0, n_classes, size=(batch_size,), dtype=np.int64)
    weight = rng.standard_normal((n_classes, embed_dim), dtype=np.float32)
    bias = rng.standard_normal((n_classes,), dtype=np.float32)

    encoder_pt = _make_dummy_encoder_pt(embed_dim, cfg.n_cls_tokens)
    obj_pt = objectives.Supervised(cfg, n_classes=n_classes)
    _set_linear_pt(obj_pt.head, weight, bias)
    batch_pt = {"data": torch.tensor(data), "target": torch.tensor(targets)}
    losses_pt, emb_pt, targets_pt = obj_pt(batch_pt, encoder_pt)

    key = jax.random.PRNGKey(seed)
    obj_jax = objectives_jax.Supervised(cfg, n_classes=n_classes, key=key)
    obj_jax = eqx.tree_at(
        lambda m: tp.cast(tp.Any, m).head,
        obj_jax,
        _set_linear_jax(tp.cast(tp.Any, obj_jax).head, weight, bias),
    )
    batch_jax = {"data": jnp.array(data), "target": jnp.array(targets)}
    losses_jax, emb_jax, targets_jax = obj_jax(
        batch_jax,
        lambda x_bnk, grid: _dummy_encoder_jax(
            x_bnk, grid, embed_dim=embed_dim, n_cls_tokens=cfg.n_cls_tokens
        ),
    )

    torch.testing.assert_close(
        _to_torch(losses_jax["ce"]),
        losses_pt["ce"],
        rtol=1e-7,
        atol=1e-7,
    )
    torch.testing.assert_close(_to_torch(emb_jax), emb_pt, rtol=1e-7, atol=1e-7)
    torch.testing.assert_close(_to_torch(targets_jax), targets_pt, rtol=1e-7, atol=1e-7)


@st.composite
def block_mask_case(draw):
    n_h = draw(st.integers(min_value=2, max_value=12))
    n_w = draw(st.integers(min_value=2, max_value=12))
    block_size = draw(st.integers(min_value=1, max_value=4))
    mask_ratio = draw(
        st.floats(min_value=0.0, max_value=0.95, allow_nan=False, allow_infinity=False)
    )
    batch_size = draw(st.integers(min_value=1, max_value=4))
    seed = draw(st.integers(min_value=0, max_value=2**16 - 1))
    return n_h, n_w, block_size, mask_ratio, batch_size, seed


@given(case=block_mask_case())
@settings(max_examples=200, deadline=None)
def test_make_block_mask_parity(case):
    n_h, n_w, block_size, mask_ratio, batch_size, seed = case
    gen_pt = torch.Generator().manual_seed(seed)
    mask_pt = objectives.make_block_mask(
        batch_size, n_h, n_w, block_size, mask_ratio, generator=gen_pt
    )
    key = jax.random.PRNGKey(seed)
    mask_jax = objectives_jax.make_block_mask(
        batch_size, n_h, n_w, block_size, mask_ratio, generator=key
    )
    assert np.array_equal(mask_pt.numpy(), np.array(mask_jax))
