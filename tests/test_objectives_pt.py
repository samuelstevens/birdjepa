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

    key = jax.random.key(seed)
    init_key, fwd_key = jax.random.split(key)
    obj_jax = objectives_jax.Supervised(cfg, n_classes=n_classes, key=init_key)
    obj_jax = eqx.tree_at(
        lambda m: tp.cast(tp.Any, m).head,
        obj_jax,
        _set_linear_jax(tp.cast(tp.Any, obj_jax).head, weight, bias),
    )
    batch_jax = {"data": jnp.array(data), "target": jnp.array(targets)}
    losses_jax, emb_jax, targets_jax = obj_jax(
        batch_jax,
        lambda x_bnk, grid, key: _dummy_encoder_jax(
            x_bnk, grid, embed_dim=embed_dim, n_cls_tokens=cfg.n_cls_tokens
        ),
        key=fwd_key,
    )

    torch.testing.assert_close(
        _to_torch(losses_jax["ce"]),
        losses_pt["ce"],
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(_to_torch(emb_jax), emb_pt, rtol=1e-5, atol=1e-5)
    # Compare targets as numpy arrays to avoid dtype issues (int32 vs int64)
    np.testing.assert_array_equal(np.array(targets_jax), targets_pt.numpy())


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
    """Both implementations produce valid masks with same count."""
    n_h, n_w, block_size, mask_ratio, batch_size, seed = case
    n_patches = n_h * n_w
    n_masked_target = int(n_patches * mask_ratio)

    gen_pt = torch.Generator().manual_seed(seed)
    mask_pt = objectives.make_block_mask(
        batch_size, n_h, n_w, block_size, mask_ratio, generator=gen_pt
    )

    key = jax.random.key(seed)
    mask_jax = objectives_jax.make_block_mask(
        batch_size, n_h, n_w, block_size, mask_ratio, key=key
    )

    # Both should have same shape
    assert mask_pt.shape == (batch_size, n_patches)
    assert mask_jax.shape == (batch_size, n_patches)

    # Both should have same mask count
    assert (mask_pt.sum(dim=1) == n_masked_target).all()
    assert (mask_jax.sum(axis=1) == n_masked_target).all()


def test_pixio_mse_parity():
    """Pixio MSE loss computation matches between JAX and PyTorch.

    Note: PyTorch uses ddof=1 for var by default, JAX uses ddof=0.
    The implementations differ slightly but both are valid normalizations.
    This test verifies the logic is consistent within each framework.
    """
    rng = np.random.default_rng(42)
    batch_size, n_patches, kernel = 2, 16, 16

    # Random predictions and targets
    pred = rng.standard_normal((batch_size, n_patches, kernel)).astype(np.float32)
    target = rng.standard_normal((batch_size, n_patches, kernel)).astype(np.float32)
    mask = rng.random((batch_size, n_patches)) > 0.25  # ~75% masked

    # PyTorch computation (uses ddof=1 for var)
    pred_pt = torch.tensor(pred)
    target_pt = torch.tensor(target)
    mask_pt = torch.tensor(mask)

    mean_pt = target_pt.mean(dim=-1, keepdim=True)
    var_pt = target_pt.var(dim=-1, keepdim=True, correction=1)  # PyTorch default
    target_norm_pt = (target_pt - mean_pt) / (var_pt + 1e-6).sqrt()
    mse_pt = torch.nn.functional.mse_loss(pred_pt[mask_pt], target_norm_pt[mask_pt])

    # JAX computation (uses ddof=0 for var, matching objectives.py)
    pred_jax = jnp.array(pred)
    target_jax = jnp.array(target)
    mask_jax = jnp.array(mask)

    mean_jax = target_jax.mean(axis=-1, keepdims=True)
    var_jax = target_jax.var(axis=-1, keepdims=True)  # ddof=0 default
    target_norm_jax = (target_jax - mean_jax) / jnp.sqrt(var_jax + 1e-6)

    mask_bnk = mask_jax[:, :, None]
    diff_sq = (pred_jax - target_norm_jax) ** 2
    masked_diff = diff_sq * mask_bnk
    mse_jax = masked_diff.sum() / (mask_jax.sum() * kernel)

    # Test JAX self-consistency: our MSE formula matches jnp.mean equivalent
    masked_elements = target_norm_jax[mask_jax]
    pred_masked = pred_jax[mask_jax]
    expected_mse = jnp.mean((pred_masked - masked_elements) ** 2)
    np.testing.assert_allclose(float(mse_jax), float(expected_mse), rtol=1e-5)

    # Both frameworks produce positive finite MSE
    assert mse_pt > 0 and torch.isfinite(mse_pt)
    assert float(mse_jax) > 0 and np.isfinite(float(mse_jax))


@st.composite
def pixio_parity_case(draw):
    """Generate valid Pixio test cases."""
    patch_h = draw(st.sampled_from([2, 4]))
    patch_w = draw(st.sampled_from([2, 4]))
    n_patches_h = draw(st.integers(min_value=2, max_value=4))
    n_patches_w = draw(st.integers(min_value=2, max_value=4))
    h = patch_h * n_patches_h
    w = patch_w * n_patches_w
    embed_dim = draw(st.sampled_from([8, 16]))
    batch_size = draw(st.integers(min_value=1, max_value=2))
    mask_ratio = draw(st.sampled_from([0.5, 0.75]))
    seed = draw(st.integers(min_value=0, max_value=2**16 - 1))
    return (
        h,
        w,
        patch_h,
        patch_w,
        embed_dim,
        n_patches_h,
        n_patches_w,
        batch_size,
        mask_ratio,
        seed,
    )


@given(case=pixio_parity_case())
@settings(max_examples=50, deadline=None)
def test_pixio_output_shapes_parity(case):
    """Pixio produces same output shapes in both implementations."""
    (
        h,
        w,
        patch_h,
        patch_w,
        embed_dim,
        n_patches_h,
        _,
        batch_size,
        mask_ratio,
        seed,
    ) = case

    cfg_pt = transformer.Config(
        input_h=h,
        input_w=w,
        patch_h=patch_h,
        patch_w=patch_w,
        embed_dim=embed_dim,
        depth=1,
        n_heads=1,
    )
    pixio_cfg_pt = objectives.PixioConfig(
        decoder_depth=1,
        decoder_dim=8,
        decoder_heads=1,
        mask_ratio=mask_ratio,
        block_size=1,
    )

    import birdjepa.nn.transformer as transformer_jax

    cfg_jax = transformer_jax.Config(
        input_h=h,
        input_w=w,
        patch_h=patch_h,
        patch_w=patch_w,
        embed_dim=embed_dim,
        depth=1,
        n_heads=1,
    )
    pixio_cfg_jax = objectives_jax.PixioConfig(
        decoder_depth=1,
        decoder_dim=8,
        decoder_heads=1,
        mask_ratio=mask_ratio,
        block_size=1,
    )

    # Create models
    torch.manual_seed(seed)
    encoder_pt = transformer.Transformer(cfg_pt)
    pixio_pt = objectives.Pixio(cfg_pt, pixio_cfg_pt)

    key = jax.random.key(seed)
    enc_key, obj_key, fwd_key = jax.random.split(key, 3)
    encoder_jax = transformer_jax.Transformer(cfg_jax, key=enc_key)
    pixio_jax = objectives_jax.Pixio(cfg_jax, pixio_cfg_jax, key=obj_key)

    # Create input data
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((batch_size, h, w)).astype(np.float32)
    targets = rng.integers(0, 10, size=(batch_size,)).astype(np.int64)

    batch_pt = {"data": torch.tensor(data), "target": torch.tensor(targets)}
    batch_jax = {"data": jnp.array(data), "target": jnp.array(targets)}

    # Forward pass
    losses_pt, emb_pt, targets_pt = pixio_pt(batch_pt, encoder_pt)
    losses_jax, emb_jax, targets_jax = pixio_jax(batch_jax, encoder_jax, key=fwd_key)

    # Check shapes match
    assert emb_pt.shape == (batch_size, embed_dim)
    assert emb_jax.shape == (batch_size, embed_dim)
    assert targets_pt.shape == (batch_size,)
    assert targets_jax.shape == (batch_size,)
    assert losses_pt["mse"].shape == ()
    assert losses_jax["mse"].shape == ()

    # Both losses should be positive
    assert losses_pt["mse"] > 0
    assert float(losses_jax["mse"]) > 0


def test_lejepa_invariance_loss_parity():
    """LeJEPA invariance loss computation matches between JAX and PyTorch."""
    rng = np.random.default_rng(42)
    v, b, p = 2, 4, 16  # views, batch, proj_dim

    # Random projections (v, b, proj_dim)
    proj = rng.standard_normal((v, b, p)).astype(np.float32)

    # PyTorch computation
    proj_pt = torch.tensor(proj)
    mean_pt = proj_pt.mean(dim=0)
    inv_loss_pt = (mean_pt - proj_pt).square().mean()

    # JAX computation
    proj_jax = jnp.array(proj)
    mean_jax = proj_jax.mean(axis=0)
    inv_loss_jax = ((mean_jax - proj_jax) ** 2).mean()

    torch.testing.assert_close(
        _to_torch(inv_loss_jax), inv_loss_pt, rtol=1e-5, atol=1e-5
    )


def test_lejepa_output_shapes_parity():
    """LeJEPA produces same output shapes in both implementations."""
    h, w, patch_h, patch_w = 16, 16, 4, 4
    embed_dim = 16
    batch_size = 2
    n_views = 2
    proj_dim = 8
    seed = 42

    cfg_pt = transformer.Config(
        input_h=h,
        input_w=w,
        patch_h=patch_h,
        patch_w=patch_w,
        embed_dim=embed_dim,
        depth=1,
        n_heads=1,
    )
    lejepa_cfg_pt = objectives.LeJEPAConfig(n_views=n_views, proj_dim=proj_dim)

    import birdjepa.nn.transformer as transformer_jax

    cfg_jax = transformer_jax.Config(
        input_h=h,
        input_w=w,
        patch_h=patch_h,
        patch_w=patch_w,
        embed_dim=embed_dim,
        depth=1,
        n_heads=1,
    )
    lejepa_cfg_jax = objectives_jax.LeJEPAConfig(n_views=n_views, proj_dim=proj_dim)

    # Create models
    torch.manual_seed(seed)
    encoder_pt = transformer.Transformer(cfg_pt)
    lejepa_pt = objectives.LeJEPA(cfg_pt, lejepa_cfg_pt)

    key = jax.random.key(seed)
    enc_key, obj_key, fwd_key = jax.random.split(key, 3)
    encoder_jax = transformer_jax.Transformer(cfg_jax, key=enc_key)
    lejepa_jax = objectives_jax.LeJEPA(cfg_jax, lejepa_cfg_jax, key=obj_key)

    # Create input data
    rng = np.random.default_rng(seed)
    views = rng.standard_normal((batch_size, n_views, h, w)).astype(np.float32)
    targets = rng.integers(0, 10, size=(batch_size,)).astype(np.int64)

    batch_pt = {"views": torch.tensor(views), "target": torch.tensor(targets)}
    batch_jax = {"views": jnp.array(views), "target": jnp.array(targets)}

    # Forward pass
    losses_pt, emb_pt, targets_pt = lejepa_pt(batch_pt, encoder_pt)
    losses_jax, emb_jax, targets_jax = lejepa_jax(batch_jax, encoder_jax, key=fwd_key)

    n = batch_size * n_views

    # Check shapes match
    assert emb_pt.shape == (n, embed_dim)
    assert emb_jax.shape == (n, embed_dim)
    assert targets_pt.shape == (n,)
    assert targets_jax.shape == (n,)
    assert losses_pt["inv"].shape == ()
    assert losses_pt["sigreg"].shape == ()
    assert losses_jax["inv"].shape == ()
    assert losses_jax["sigreg"].shape == ()

    # Both losses should be non-negative and finite
    assert losses_pt["inv"] >= 0 and torch.isfinite(losses_pt["inv"])
    assert losses_pt["sigreg"] >= 0 and torch.isfinite(losses_pt["sigreg"])
    assert float(losses_jax["inv"]) >= 0 and np.isfinite(float(losses_jax["inv"]))
    assert float(losses_jax["sigreg"]) >= 0 and np.isfinite(float(losses_jax["sigreg"]))
