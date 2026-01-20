"""JAX tests for transformer utilities."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from hypothesis import given, settings
from hypothesis import strategies as st

from birdjepa.nn import transformer


def test_transformer_forward_smoke():
    """Transformer forward with (B, T, kernel) + grid API."""
    cfg = transformer.Transformer(
        input_h=32, input_w=32, patch_h=4, patch_w=4, embed_dim=64, depth=2, n_heads=4
    )
    key = jax.random.key(0)
    model_key, data_key, fwd_key = jax.random.split(key, 3)
    model = transformer.TransformerModel(cfg, key=model_key)

    x_bhw = jax.random.normal(data_key, (2, 32, 32))
    x_bnk, grid_bn2 = transformer.patchify(x_bhw, cfg)

    out = model(x_bnk, grid=grid_bn2, key=fwd_key)

    assert out.cls.shape == (2, 1, 64)
    assert out.patches.shape == (2, 64, 64)
    assert out.reg.shape == (2, 0, 64)


def test_transformer_visible_patches_only():
    """Transformer should work with subset of patches (for MAE encoder)."""
    cfg = transformer.Transformer(
        input_h=32, input_w=32, patch_h=4, patch_w=4, embed_dim=64, depth=2, n_heads=4
    )
    key = jax.random.key(0)
    model_key, data_key, fwd_key = jax.random.split(key, 3)
    model = transformer.TransformerModel(cfg, key=model_key)

    x_bhw = jax.random.normal(data_key, (2, 32, 32))
    x_bnk, grid_bn2 = transformer.patchify(x_bhw, cfg)
    x_visible = x_bnk[:, :16]
    grid_visible = grid_bn2[:, :16]

    out = model(x_visible, grid=grid_visible, key=fwd_key)

    assert out.cls.shape == (2, 1, 64)
    assert out.patches.shape == (2, 16, 64)


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
    cfg = transformer.Transformer(
        input_h=h, input_w=w, patch_h=patch_h, patch_w=patch_w
    )
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


# -----------------------------------------------------------------------------
# RopePositionEmbedding tests
# -----------------------------------------------------------------------------


def test_rope_embedding_shape():
    """RopePositionEmbedding returns correct shapes."""
    embed_dim, n_heads = 64, 4
    d_head = embed_dim // n_heads
    rope = transformer.RopePositionEmbedding(embed_dim, n_heads, base=100.0)

    b, t = 2, 16
    grid = jnp.zeros((b, t, 2), dtype=jnp.int32)

    sin, cos = rope(grid)

    assert sin.shape == (b, t, d_head)
    assert cos.shape == (b, t, d_head)


def test_rope_embedding_different_positions():
    """Different grid positions produce different embeddings."""
    embed_dim, n_heads = 64, 4
    rope = transformer.RopePositionEmbedding(embed_dim, n_heads, base=100.0)

    grid1 = jnp.array([[[0, 0], [0, 1], [1, 0], [1, 1]]])
    grid2 = jnp.array([[[2, 2], [2, 3], [3, 2], [3, 3]]])

    sin1, cos1 = rope(grid1)
    sin2, cos2 = rope(grid2)

    assert not jnp.allclose(sin1, sin2)
    assert not jnp.allclose(cos1, cos2)


def test_rope_embedding_deterministic():
    """Same grid produces same sin/cos every time."""
    embed_dim, n_heads = 64, 4
    rope = transformer.RopePositionEmbedding(embed_dim, n_heads, base=100.0)

    grid = jnp.array([[[0, 0], [1, 1], [2, 2]]])

    sin1, cos1 = rope(grid)
    sin2, cos2 = rope(grid)

    assert jnp.allclose(sin1, sin2)
    assert jnp.allclose(cos1, cos2)


def test_rope_embedding_bounds():
    """Sin/cos values are in [-1, 1]."""
    embed_dim, n_heads = 64, 4
    rope = transformer.RopePositionEmbedding(embed_dim, n_heads, base=100.0)

    h_idx = jnp.arange(32)
    w_idx = jnp.arange(32)
    grid_hw = jnp.stack(jnp.meshgrid(h_idx, w_idx, indexing="ij")).reshape(2, -1).T
    grid = grid_hw[None]

    sin, cos = rope(grid)

    assert jnp.all(sin >= -1.0) and jnp.all(sin <= 1.0)
    assert jnp.all(cos >= -1.0) and jnp.all(cos <= 1.0)


# -----------------------------------------------------------------------------
# _rotate_half tests
# -----------------------------------------------------------------------------


def test_rotate_half_shape():
    """_rotate_half preserves shape."""
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    y = transformer._rotate_half(x)
    assert y.shape == x.shape


def test_rotate_half_correctness():
    """_rotate_half produces [-x2, x1] from [x1, x2]."""
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    y = transformer._rotate_half(x)
    expected = jnp.array([-3.0, -4.0, 1.0, 2.0])
    assert jnp.allclose(y, expected)


def test_rotate_half_batched():
    """_rotate_half works on batched input."""
    x = jnp.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    y = transformer._rotate_half(x)
    expected = jnp.array([[-3.0, -4.0, 1.0, 2.0], [-7.0, -8.0, 5.0, 6.0]])
    assert jnp.allclose(y, expected)


def test_rotate_half_double_negates():
    """Applying _rotate_half twice negates the input."""
    x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    y = transformer._rotate_half(transformer._rotate_half(x))
    assert jnp.allclose(y, -x)


# -----------------------------------------------------------------------------
# _apply_rope tests
# -----------------------------------------------------------------------------


def test_apply_rope_prefix_unchanged():
    """Prefix tokens (CLS, etc) should not be modified by RoPE."""
    b, n, h, d = 2, 10, 4, 16
    n_prefix = 2
    t = n - n_prefix

    key = jr.key(0)
    q = jr.normal(key, (b, n, h, d))
    k = jr.normal(jr.key(1), (b, n, h, d))

    sin = jnp.ones((b, t, d))
    cos = jnp.ones((b, t, d))

    q_out, k_out = transformer._apply_rope(q, k, sin, cos, n_prefix)

    assert jnp.allclose(q_out[:, :n_prefix], q[:, :n_prefix])
    assert jnp.allclose(k_out[:, :n_prefix], k[:, :n_prefix])


def test_apply_rope_position_tokens_modified():
    """Position tokens should be modified by RoPE."""
    b, n, h, d = 2, 10, 4, 16
    n_prefix = 2
    t = n - n_prefix

    key = jr.key(0)
    q = jr.normal(key, (b, n, h, d))
    k = jr.normal(jr.key(1), (b, n, h, d))

    sin = jnp.ones((b, t, d)) * 0.5
    cos = jnp.ones((b, t, d)) * 0.5

    q_out, k_out = transformer._apply_rope(q, k, sin, cos, n_prefix)

    assert not jnp.allclose(q_out[:, n_prefix:], q[:, n_prefix:])
    assert not jnp.allclose(k_out[:, n_prefix:], k[:, n_prefix:])


def test_apply_rope_shape_preservation():
    """_apply_rope preserves shapes."""
    b, n, h, d = 2, 10, 4, 16
    n_prefix = 2
    t = n - n_prefix

    q = jnp.ones((b, n, h, d))
    k = jnp.ones((b, n, h, d))
    sin = jnp.ones((b, t, d))
    cos = jnp.ones((b, t, d))

    q_out, k_out = transformer._apply_rope(q, k, sin, cos, n_prefix)

    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


# -----------------------------------------------------------------------------
# Attention with RoPE tests
# -----------------------------------------------------------------------------


def test_attention_without_rope():
    """Attention works without RoPE (backward compatibility)."""
    cfg = transformer.Transformer(embed_dim=64, n_heads=4, use_rope=False)
    key = jr.key(0)
    attn = transformer.Attention(cfg, key=key)

    x = jnp.ones((2, 16, 64))
    out = attn(x)

    assert out.shape == x.shape


def test_attention_with_rope():
    """Attention with RoPE produces different outputs for different positions."""
    cfg = transformer.Transformer(embed_dim=64, n_heads=4, use_rope=True)
    key = jr.key(0)
    attn = transformer.Attention(cfg, key=key)

    x = jnp.ones((2, 10, 64))
    n_prefix = 2

    rope_embed = transformer.RopePositionEmbedding(64, 4, base=100.0)

    grid1 = jnp.array([
        [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1], [3, 0], [3, 1]]
    ])
    grid1 = jnp.broadcast_to(grid1, (2, 8, 2))
    grid2 = jnp.array([
        [[4, 4], [4, 5], [5, 4], [5, 5], [6, 4], [6, 5], [7, 4], [7, 5]]
    ])
    grid2 = jnp.broadcast_to(grid2, (2, 8, 2))

    rope1 = rope_embed(grid1)
    rope2 = rope_embed(grid2)

    out1 = attn(x, rope=rope1, n_prefix=n_prefix)
    out2 = attn(x, rope=rope2, n_prefix=n_prefix)

    assert not jnp.allclose(out1, out2)


def test_attention_qk_norm_and_rope():
    """QK-norm and RoPE can be enabled together."""
    cfg = transformer.Transformer(
        embed_dim=64, n_heads=4, use_rope=True, use_qk_norm=True
    )
    key = jr.key(0)
    attn = transformer.Attention(cfg, key=key)

    x = jnp.ones((2, 10, 64))
    rope_embed = transformer.RopePositionEmbedding(64, 4, base=100.0)
    grid = jnp.zeros((2, 8, 2), dtype=jnp.int32)
    rope = rope_embed(grid)

    out = attn(x, rope=rope, n_prefix=2)

    assert out.shape == x.shape
    assert not jnp.any(jnp.isnan(out))


# -----------------------------------------------------------------------------
# TransformerModel with RoPE tests
# -----------------------------------------------------------------------------


def test_transformer_rope_mode():
    """With use_rope=True, rope_embed is set and pos_embed_hw is None."""
    cfg = transformer.Transformer(
        input_h=32,
        input_w=32,
        patch_h=4,
        patch_w=4,
        embed_dim=64,
        depth=2,
        n_heads=4,
        use_rope=True,
    )
    model = transformer.TransformerModel(cfg, key=jr.key(0))

    assert model.rope_embed is not None
    assert model.pos_embed_hw is None


def test_transformer_absolute_mode():
    """With use_rope=False, pos_embed_hw is set and rope_embed is None."""
    cfg = transformer.Transformer(
        input_h=32,
        input_w=32,
        patch_h=4,
        patch_w=4,
        embed_dim=64,
        depth=2,
        n_heads=4,
        use_rope=False,
    )
    model = transformer.TransformerModel(cfg, key=jr.key(0))

    assert model.rope_embed is None
    assert model.pos_embed_hw is not None


def test_transformer_output_shapes_both_modes():
    """Output shapes are the same regardless of position encoding mode."""
    base_cfg = dict(
        input_h=32, input_w=32, patch_h=4, patch_w=4, embed_dim=64, depth=2, n_heads=4
    )

    cfg_rope = transformer.Transformer(**base_cfg, use_rope=True)
    cfg_abs = transformer.Transformer(**base_cfg, use_rope=False)

    model_rope = transformer.TransformerModel(cfg_rope, key=jr.key(0))
    model_abs = transformer.TransformerModel(cfg_abs, key=jr.key(1))

    x = jnp.ones((2, 32, 32))
    x_bnk, grid = transformer.patchify(x, cfg_rope)

    out_rope = model_rope(x_bnk, grid=grid, key=jr.key(2))
    out_abs = model_abs(x_bnk, grid=grid, key=jr.key(3))

    assert out_rope.cls.shape == out_abs.cls.shape
    assert out_rope.patches.shape == out_abs.patches.shape


def test_transformer_gradient_flow_with_rope():
    """Gradients propagate through RoPE correctly."""
    cfg = transformer.Transformer(
        input_h=32,
        input_w=32,
        patch_h=4,
        patch_w=4,
        embed_dim=64,
        depth=2,
        n_heads=4,
        use_rope=True,
        grad_ckpt=False,
    )
    model = transformer.TransformerModel(cfg, key=jr.key(0))

    x = jnp.ones((2, 32, 32))
    x_bnk, grid = transformer.patchify(x, cfg)

    def loss_fn(m):
        out = m(x_bnk, grid=grid, key=jr.key(1))
        return out.cls.mean() + out.patches.mean()

    grads = eqx.filter_grad(loss_fn)(model)

    grad_leaves = [x for x in jax.tree_util.tree_leaves(grads) if x is not None]
    assert len(grad_leaves) > 0
    for g in grad_leaves:
        assert not jnp.any(jnp.isnan(g))
        assert not jnp.any(jnp.isinf(g))


def test_transformer_scan_vs_loop_with_rope():
    """Scan and loop modes produce same outputs with RoPE."""
    base_cfg = dict(
        input_h=32,
        input_w=32,
        patch_h=4,
        patch_w=4,
        embed_dim=64,
        depth=2,
        n_heads=4,
        use_rope=True,
        grad_ckpt=False,
    )

    cfg_scan = transformer.Transformer(**base_cfg, use_scan=True)
    cfg_loop = transformer.Transformer(**base_cfg, use_scan=False)

    model_scan = transformer.TransformerModel(cfg_scan, key=jr.key(42))
    model_loop = transformer.TransformerModel(cfg_loop, key=jr.key(42))

    x = jnp.ones((2, 32, 32))
    x_bnk, grid = transformer.patchify(x, cfg_scan)

    out_scan = model_scan(x_bnk, grid=grid, key=jr.key(0))
    out_loop = model_loop(x_bnk, grid=grid, key=jr.key(0))

    assert jnp.allclose(out_scan.cls, out_loop.cls, atol=1e-5)
    assert jnp.allclose(out_scan.patches, out_loop.patches, atol=1e-5)


# -----------------------------------------------------------------------------
# Transformer config tests
# -----------------------------------------------------------------------------


def test_transformer_config_free_wins_defaults():
    """New 'free wins' config fields have correct defaults."""
    cfg = transformer.Transformer()

    assert cfg.use_rope is False
    assert cfg.rope_base == 100.0
    assert cfg.use_qk_norm is False
    assert cfg.use_swiglu is False
    assert cfg.use_layerscale is False


# -----------------------------------------------------------------------------
# Integration tests: all free wins
# -----------------------------------------------------------------------------


def test_all_free_wins_forward():
    """Model with all free wins enabled runs forward pass."""
    cfg = transformer.Transformer(
        input_h=32,
        input_w=32,
        patch_h=4,
        patch_w=4,
        embed_dim=64,
        depth=2,
        n_heads=4,
        use_rope=True,
        use_qk_norm=True,
        use_swiglu=True,
        use_layerscale=True,
    )
    model = transformer.TransformerModel(cfg, key=jr.key(0))

    x = jnp.ones((2, 32, 32))
    x_bnk, grid = transformer.patchify(x, cfg)

    out = model(x_bnk, grid=grid, key=jr.key(1))

    assert out.cls.shape == (2, 1, 64)
    assert out.patches.shape == (2, 64, 64)
    assert not jnp.any(jnp.isnan(out.cls))
    assert not jnp.any(jnp.isnan(out.patches))


def test_all_free_wins_backward():
    """Model with all free wins computes gradients without NaN/Inf."""
    cfg = transformer.Transformer(
        input_h=32,
        input_w=32,
        patch_h=4,
        patch_w=4,
        embed_dim=64,
        depth=2,
        n_heads=4,
        use_rope=True,
        use_qk_norm=True,
        use_swiglu=True,
        use_layerscale=True,
        grad_ckpt=False,
    )
    model = transformer.TransformerModel(cfg, key=jr.key(0))

    x = jnp.ones((2, 32, 32))
    x_bnk, grid = transformer.patchify(x, cfg)

    def loss_fn(m):
        out = m(x_bnk, grid=grid, key=jr.key(1))
        return out.cls.mean() + out.patches.mean()

    grads = eqx.filter_grad(loss_fn)(model)

    grad_leaves = [x for x in jax.tree_util.tree_leaves(grads) if x is not None]
    assert len(grad_leaves) > 0
    for g in grad_leaves:
        assert not jnp.any(jnp.isnan(g)), "NaN in gradients"
        assert not jnp.any(jnp.isinf(g)), "Inf in gradients"


@given(
    use_rope=st.booleans(),
    use_qk_norm=st.booleans(),
    use_swiglu=st.booleans(),
    use_layerscale=st.booleans(),
)
@settings(max_examples=16, deadline=None)
def test_free_wins_combinations(use_rope, use_qk_norm, use_swiglu, use_layerscale):
    """All combinations of free wins produce valid outputs."""
    cfg = transformer.Transformer(
        input_h=16,
        input_w=16,
        patch_h=4,
        patch_w=4,
        embed_dim=32,
        depth=1,
        n_heads=4,
        use_rope=use_rope,
        use_qk_norm=use_qk_norm,
        use_swiglu=use_swiglu,
        use_layerscale=use_layerscale,
    )
    model = transformer.TransformerModel(cfg, key=jr.key(0))

    x = jnp.ones((1, 16, 16))
    x_bnk, grid = transformer.patchify(x, cfg)

    out = model(x_bnk, grid=grid, key=jr.key(1))

    assert out.cls.shape == (1, 1, 32)
    assert not jnp.any(jnp.isnan(out.cls))
