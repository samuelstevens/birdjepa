"""Parity tests comparing JAX Transformer to PyTorch Transformer.

Tests both forward and backward passes to identify gradient flow issues.
"""

import numpy as np
import jax
import jax.numpy as jnp
import torch
import equinox as eqx

import birdjepa.nn.transformer as transformer_jax
import birdjepa.nn.transformer_pt as transformer_pt

RTOL = 1e-3
ATOL = 1e-3


def _copy_linear(jax_linear: eqx.nn.Linear, pt_linear: torch.nn.Linear):
    """Copy PyTorch linear weights to JAX linear (in-place modification of JAX)."""
    # PyTorch: weight (out, in), bias (out,)
    # JAX eqx.nn.Linear: weight (out, in), bias (out,)
    weight = pt_linear.weight.detach().numpy()
    jax_linear = eqx.tree_at(lambda m: m.weight, jax_linear, jnp.array(weight))
    if pt_linear.bias is not None:
        bias = pt_linear.bias.detach().numpy()
        jax_linear = eqx.tree_at(lambda m: m.bias, jax_linear, jnp.array(bias))
    return jax_linear


def _copy_layernorm(jax_ln: eqx.nn.LayerNorm, pt_ln: torch.nn.LayerNorm):
    """Copy PyTorch LayerNorm weights to JAX LayerNorm."""
    weight = pt_ln.weight.detach().numpy()
    bias = pt_ln.bias.detach().numpy()
    jax_ln = eqx.tree_at(lambda m: m.weight, jax_ln, jnp.array(weight))
    jax_ln = eqx.tree_at(lambda m: m.bias, jax_ln, jnp.array(bias))
    return jax_ln


def _copy_attention(
    jax_attn: transformer_jax.Attention, pt_attn: transformer_pt.Attention
):
    """Copy PyTorch Attention weights to JAX Attention."""
    jax_attn = eqx.tree_at(
        lambda m: m.qkv, jax_attn, _copy_linear(jax_attn.qkv, pt_attn.qkv)
    )
    jax_attn = eqx.tree_at(
        lambda m: m.proj, jax_attn, _copy_linear(jax_attn.proj, pt_attn.proj)
    )
    return jax_attn


def _copy_mlp(jax_mlp: transformer_jax.MLP, pt_mlp: transformer_pt.MLP):
    """Copy PyTorch MLP weights to JAX MLP."""
    if not jax_mlp.use_swiglu:
        jax_mlp = eqx.tree_at(
            lambda m: m.fc1, jax_mlp, _copy_linear(jax_mlp.fc1, pt_mlp.fc1)
        )
        jax_mlp = eqx.tree_at(
            lambda m: m.fc2, jax_mlp, _copy_linear(jax_mlp.fc2, pt_mlp.fc2)
        )
    return jax_mlp


def _copy_block(jax_block: transformer_jax.Block, pt_block: transformer_pt.Block):
    """Copy PyTorch Block weights to JAX Block."""
    jax_block = eqx.tree_at(
        lambda m: m.norm1, jax_block, _copy_layernorm(jax_block.norm1, pt_block.norm1)
    )
    jax_block = eqx.tree_at(
        lambda m: m.attn, jax_block, _copy_attention(jax_block.attn, pt_block.attn)
    )
    jax_block = eqx.tree_at(
        lambda m: m.norm2, jax_block, _copy_layernorm(jax_block.norm2, pt_block.norm2)
    )
    jax_block = eqx.tree_at(
        lambda m: m.mlp, jax_block, _copy_mlp(jax_block.mlp, pt_block.mlp)
    )
    return jax_block


def _copy_transformer(
    jax_model: transformer_jax.Transformer, pt_model: transformer_pt.Transformer
):
    """Copy all PyTorch Transformer weights to JAX Transformer."""
    # Patch embedding
    jax_model = eqx.tree_at(
        lambda m: m.patch_embed.proj,
        jax_model,
        _copy_linear(jax_model.patch_embed.proj, pt_model.patch_embed.proj),
    )

    # CLS tokens
    cls_tokens = pt_model.cls_tokens.detach().numpy()
    jax_model = eqx.tree_at(lambda m: m.cls_tokens, jax_model, jnp.array(cls_tokens))

    # Reg tokens
    if pt_model.reg_tokens is not None:
        reg_tokens = pt_model.reg_tokens.detach().numpy()
        jax_model = eqx.tree_at(
            lambda m: m.reg_tokens, jax_model, jnp.array(reg_tokens)
        )

    # Position embeddings
    if pt_model.pos_embed_hw is not None:
        pos_embed = pt_model.pos_embed_hw.detach().numpy()
        jax_model = eqx.tree_at(
            lambda m: m.pos_embed_hw, jax_model, jnp.array(pos_embed)
        )

    # Blocks - JAX uses tuple when use_scan=False
    assert isinstance(jax_model.blocks, tuple), "Use use_scan=False for parity testing"
    new_blocks = []
    for i, (jax_block, pt_block) in enumerate(zip(jax_model.blocks, pt_model.blocks)):
        new_blocks.append(_copy_block(jax_block, pt_block))
    jax_model = eqx.tree_at(lambda m: m.blocks, jax_model, tuple(new_blocks))

    # Final norm
    jax_model = eqx.tree_at(
        lambda m: m.norm, jax_model, _copy_layernorm(jax_model.norm, pt_model.norm)
    )

    return jax_model


def test_transformer_forward_parity():
    """Test that JAX and PyTorch transformers produce identical forward outputs."""
    cfg_jax = transformer_jax.Config(
        input_h=32,
        input_w=32,
        patch_h=4,
        patch_w=4,
        embed_dim=64,
        depth=2,
        n_heads=4,
        dropout=0.0,
        use_scan=False,  # Use explicit loop for easier debugging
    )
    cfg_pt = transformer_pt.Config(
        input_h=32,
        input_w=32,
        patch_h=4,
        patch_w=4,
        embed_dim=64,
        depth=2,
        n_heads=4,
        dropout=0.0,
    )

    # Create models
    key = jax.random.key(42)
    jax_model = transformer_jax.Transformer(cfg_jax, key=key)
    pt_model = transformer_pt.Transformer(cfg_pt)
    pt_model.eval()  # Disable dropout

    # Copy weights from PyTorch to JAX
    jax_model = _copy_transformer(jax_model, pt_model)

    # Create identical input
    np.random.seed(123)
    x_np = np.random.randn(2, 32, 32).astype(np.float32)

    # Patchify with both
    x_jax = jnp.array(x_np)
    x_pt = torch.from_numpy(x_np)

    x_jax_patches, grid_jax = transformer_jax.patchify(x_jax, cfg_jax)
    x_pt_patches, grid_pt = transformer_pt.patchify(x_pt, cfg_pt)

    # Forward pass
    key = jax.random.key(0)
    jax_out = jax_model(x_jax_patches, grid=grid_jax, key=key)
    with torch.no_grad():
        pt_out = pt_model(x_pt_patches, grid=grid_pt)

    # Compare outputs
    np.testing.assert_allclose(
        np.asarray(jax_out.cls),
        pt_out["cls"].numpy(),
        rtol=RTOL,
        atol=ATOL,
        err_msg="CLS token mismatch",
    )
    np.testing.assert_allclose(
        np.asarray(jax_out.patches),
        pt_out["patches"].numpy(),
        rtol=RTOL,
        atol=ATOL,
        err_msg="Patch embeddings mismatch",
    )


def test_transformer_backward_parity():
    """Test that JAX and PyTorch transformers produce identical gradients.

    This is the critical test - if gradients differ, training will diverge.
    """
    cfg_jax = transformer_jax.Config(
        input_h=32,
        input_w=32,
        patch_h=4,
        patch_w=4,
        embed_dim=64,
        depth=2,
        n_heads=4,
        dropout=0.0,
        use_scan=False,
    )
    cfg_pt = transformer_pt.Config(
        input_h=32,
        input_w=32,
        patch_h=4,
        patch_w=4,
        embed_dim=64,
        depth=2,
        n_heads=4,
        dropout=0.0,
    )

    # Create models
    key = jax.random.key(42)
    jax_model = transformer_jax.Transformer(cfg_jax, key=key)
    pt_model = transformer_pt.Transformer(cfg_pt)
    pt_model.train()

    # Copy weights from PyTorch to JAX
    jax_model = _copy_transformer(jax_model, pt_model)

    # Create identical input
    np.random.seed(123)
    x_np = np.random.randn(2, 32, 32).astype(np.float32)

    x_jax = jnp.array(x_np)
    x_pt = torch.from_numpy(x_np).requires_grad_(True)

    x_jax_patches, grid_jax = transformer_jax.patchify(x_jax, cfg_jax)
    x_pt_patches, grid_pt = transformer_pt.patchify(x_pt, cfg_pt)
    x_pt_patches = x_pt_patches.detach().requires_grad_(True)

    # JAX: compute gradient of mean CLS output w.r.t. input
    def jax_loss_fn(model, x, grid, key):
        out = model(x, grid=grid, key=key)
        return out.cls.mean()

    key = jax.random.key(0)
    jax_loss, jax_grads = eqx.filter_value_and_grad(jax_loss_fn)(
        jax_model, x_jax_patches, grid_jax, key
    )

    # PyTorch: compute gradient
    pt_out = pt_model(x_pt_patches, grid=grid_pt)
    pt_loss = pt_out["cls"].mean()
    pt_loss.backward()

    # Compare loss values first
    np.testing.assert_allclose(
        float(jax_loss), pt_loss.item(), rtol=RTOL, atol=ATOL, err_msg="Loss mismatch"
    )

    # Compare gradients on patch embedding projection
    jax_patch_embed_grad = np.asarray(jax_grads.patch_embed.proj.weight)
    pt_patch_embed_grad = pt_model.patch_embed.proj.weight.grad.numpy()
    np.testing.assert_allclose(
        jax_patch_embed_grad,
        pt_patch_embed_grad,
        rtol=1e-3,
        atol=1e-3,
        err_msg="Patch embed weight gradient mismatch",
    )

    # Compare gradients on final norm
    jax_norm_weight_grad = np.asarray(jax_grads.norm.weight)
    pt_norm_weight_grad = pt_model.norm.weight.grad.numpy()
    np.testing.assert_allclose(
        jax_norm_weight_grad,
        pt_norm_weight_grad,
        rtol=1e-3,
        atol=1e-3,
        err_msg="Final norm weight gradient mismatch",
    )

    # Compare gradients on first block's attention QKV
    jax_qkv_grad = np.asarray(jax_grads.blocks[0].attn.qkv.weight)
    pt_qkv_grad = pt_model.blocks[0].attn.qkv.weight.grad.numpy()
    np.testing.assert_allclose(
        jax_qkv_grad,
        pt_qkv_grad,
        rtol=1e-3,
        atol=1e-3,
        err_msg="Block 0 QKV weight gradient mismatch",
    )


def test_attention_forward_parity():
    """Test attention module in isolation."""
    cfg_jax = transformer_jax.Config(embed_dim=64, n_heads=4, dropout=0.0)
    cfg_pt = transformer_pt.Config(embed_dim=64, n_heads=4, dropout=0.0)

    key = jax.random.key(42)
    jax_attn = transformer_jax.Attention(cfg_jax, key=key)
    pt_attn = transformer_pt.Attention(cfg_pt)
    pt_attn.eval()

    # Copy weights
    jax_attn = _copy_attention(jax_attn, pt_attn)

    # Input
    np.random.seed(123)
    x_np = np.random.randn(2, 16, 64).astype(np.float32)

    jax_out = jax_attn(jnp.array(x_np))
    with torch.no_grad():
        pt_out = pt_attn(torch.from_numpy(x_np))

    np.testing.assert_allclose(
        np.asarray(jax_out),
        pt_out.numpy(),
        rtol=RTOL,
        atol=ATOL,
        err_msg="Attention output mismatch",
    )


def test_attention_qkv_computation():
    """Debug test: check QKV computation step by step."""
    cfg_jax = transformer_jax.Config(embed_dim=64, n_heads=4, dropout=0.0)
    cfg_pt = transformer_pt.Config(embed_dim=64, n_heads=4, dropout=0.0)

    key = jax.random.key(42)
    jax_attn = transformer_jax.Attention(cfg_jax, key=key)
    pt_attn = transformer_pt.Attention(cfg_pt)
    pt_attn.eval()

    # Copy weights
    jax_attn = _copy_attention(jax_attn, pt_attn)

    # Input
    np.random.seed(123)
    x_np = np.random.randn(2, 16, 64).astype(np.float32)
    x_jax = jnp.array(x_np)
    x_pt = torch.from_numpy(x_np)

    b, n, d = x_np.shape
    n_heads = 4
    head_dim = d // n_heads

    # Step 1: QKV projection
    jax_qkv = jax.vmap(jax.vmap(jax_attn.qkv))(x_jax)
    with torch.no_grad():
        pt_qkv = pt_attn.qkv(x_pt)

    np.testing.assert_allclose(
        np.asarray(jax_qkv),
        pt_qkv.numpy(),
        rtol=1e-4,
        atol=1e-4,
        err_msg="QKV projection mismatch",
    )

    # Step 2: Reshape and transpose
    # JAX uses (B, seq_len, n_heads, head_dim), PyTorch uses (B, n_heads, seq_len, head_dim)
    jax_qkv = jax_qkv.reshape(b, n, 3, n_heads, head_dim)
    jax_qkv = jax_qkv.transpose(2, 0, 1, 3, 4)  # (3, B, N, n_heads, head_dim)
    jax_q, jax_k, jax_v = jax_qkv[0], jax_qkv[1], jax_qkv[2]

    pt_qkv = pt_qkv.reshape(b, n, 3, n_heads, head_dim)
    pt_qkv = pt_qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, N, head_dim)
    pt_q, pt_k, pt_v = pt_qkv.unbind(0)

    # Compare Q (need to transpose JAX to match PyTorch shape)
    np.testing.assert_allclose(
        np.asarray(jax_q.transpose(0, 2, 1, 3)),
        pt_q.numpy(),
        rtol=1e-4,
        atol=1e-4,
        err_msg="Q reshape mismatch",
    )

    # Step 3: Scaled dot-product attention
    scale = head_dim**-0.5

    # JAX dot_product_attention with correct shape (B, seq_len, n_heads, head_dim)
    jax_attn_out = jax.nn.dot_product_attention(jax_q, jax_k, jax_v, scale=scale)

    # PyTorch sdpa with its shape (B, n_heads, seq_len, head_dim)
    with torch.no_grad():
        pt_attn_out = torch.nn.functional.scaled_dot_product_attention(pt_q, pt_k, pt_v)

    # Compare (transpose JAX to match PyTorch shape)
    np.testing.assert_allclose(
        np.asarray(jax_attn_out.transpose(0, 2, 1, 3)),
        pt_attn_out.numpy(),
        rtol=1e-4,
        atol=1e-4,
        err_msg="JAX vs PyTorch sdpa mismatch",
    )

    # Step 4: Merge heads
    jax_merged = jax_attn_out.reshape(b, n, d)
    pt_merged = pt_attn_out.transpose(1, 2).reshape(b, n, d)

    np.testing.assert_allclose(
        np.asarray(jax_merged),
        pt_merged.numpy(),
        rtol=1e-4,
        atol=1e-4,
        err_msg="Merged heads mismatch",
    )

    # Step 5: Output projection
    jax_out = jax.vmap(jax.vmap(jax_attn.proj))(jax_merged)
    with torch.no_grad():
        pt_out = pt_attn.proj(pt_merged)

    np.testing.assert_allclose(
        np.asarray(jax_out),
        pt_out.numpy(),
        rtol=1e-4,
        atol=1e-4,
        err_msg="Output projection mismatch",
    )


def test_mlp_forward_parity():
    """Test MLP module in isolation."""
    cfg_jax = transformer_jax.Config(embed_dim=64, mlp_ratio=4.0, dropout=0.0)
    cfg_pt = transformer_pt.Config(embed_dim=64, mlp_ratio=4.0, dropout=0.0)

    key = jax.random.key(42)
    jax_mlp = transformer_jax.MLP(cfg_jax, key=key)
    pt_mlp = transformer_pt.MLP(cfg_pt)
    pt_mlp.eval()

    # Copy weights
    jax_mlp = _copy_mlp(jax_mlp, pt_mlp)

    # Input
    np.random.seed(123)
    x_np = np.random.randn(2, 16, 64).astype(np.float32)

    jax_out = jax_mlp(jnp.array(x_np), key=None)
    with torch.no_grad():
        pt_out = pt_mlp(torch.from_numpy(x_np))

    np.testing.assert_allclose(
        np.asarray(jax_out),
        pt_out.numpy(),
        rtol=RTOL,
        atol=ATOL,
        err_msg="MLP output mismatch",
    )


def test_block_forward_parity():
    """Test transformer block in isolation."""
    cfg_jax = transformer_jax.Config(
        embed_dim=64, n_heads=4, mlp_ratio=4.0, dropout=0.0
    )
    cfg_pt = transformer_pt.Config(embed_dim=64, n_heads=4, mlp_ratio=4.0, dropout=0.0)

    key = jax.random.key(42)
    jax_block = transformer_jax.Block(cfg_jax, key=key)
    pt_block = transformer_pt.Block(cfg_pt)
    pt_block.eval()

    # Copy weights
    jax_block = _copy_block(jax_block, pt_block)

    # Input
    np.random.seed(123)
    x_np = np.random.randn(2, 16, 64).astype(np.float32)

    jax_out = jax_block(jnp.array(x_np), key=None)
    with torch.no_grad():
        pt_out = pt_block(torch.from_numpy(x_np))

    np.testing.assert_allclose(
        np.asarray(jax_out),
        pt_out.numpy(),
        rtol=RTOL,
        atol=ATOL,
        err_msg="Block output mismatch",
    )
