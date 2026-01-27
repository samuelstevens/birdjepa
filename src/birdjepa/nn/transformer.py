"""Simple bidirectional transformer encoder (JAX/Equinox)."""

import dataclasses
import typing

import beartype
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Int, PRNGKeyArray, jaxtyped

internal_rope_embedding_cache: dict[
    tuple[int, int, int, float, typing.Any], tuple[Array, Array, Array, Array]
] = {}


class EncoderOutput(typing.NamedTuple):
    """Encoder outputs."""

    cls: Float[Array, "... n_cls d"]
    patches: Float[Array, "... n_patches d"]
    reg: Float[Array, "... n_reg d"]


# -------
# Config
# -------


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Transformer:
    """Transformer configuration."""

    input_h: int = 512
    """Input height (e.g., time frames for audio, image height for vision)."""
    input_w: int = 128
    """Input width (e.g., mel bins for audio, image width for vision)."""
    patch_h: int = 16
    """Patch size in height dimension."""
    patch_w: int = 16
    """Patch size in width dimension."""
    embed_dim: int = 768
    """Transformer embedding dimension."""
    depth: int = 12
    """Number of transformer blocks."""
    n_heads: int = 12
    """Number of attention heads."""
    mlp_ratio: float = 4.0
    """MLP hidden dim = embed_dim * mlp_ratio."""
    dropout: float = 0.0
    """Dropout rate."""
    # "Free wins" from recent papers
    use_rope: bool = False
    """Use rotary positional embeddings instead of absolute."""
    rope_base: float = 100.0
    """Base frequency for RoPE. Higher = longer-range position sensitivity."""
    use_qk_norm: bool = False
    """Apply LayerNorm to queries and keys for attention stability."""
    use_swiglu: bool = False
    """Use SwiGLU activation instead of GELU in MLP."""
    use_layerscale: bool = False
    """Use learnable per-layer residual scaling (from CaiT)."""
    layerscale_init: float = 1e-4
    """Initial value for LayerScale parameters."""
    n_cls_tokens: int = 1
    """Number of CLS tokens (Pixio uses 4-8)."""
    n_reg_tokens: int = 0
    """Number of register tokens (discarded at inference)."""
    use_scan: bool = True
    """Use jax.lax.scan for blocks (faster compile). False = explicit loop (for debugging)."""
    grad_ckpt: bool = True
    """Use gradient checkpointing to reduce memory (recompute activations during backward)."""

    @property
    def n_patches_h(self) -> int:
        return self.input_h // self.patch_h

    @property
    def n_patches_w(self) -> int:
        return self.input_w // self.patch_w

    @property
    def n_patches(self) -> int:
        return self.n_patches_h * self.n_patches_w


# --------------
# Debug Encoder
# --------------


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Debug:
    """Config for DebugModel (minimal linear projection)."""

    input_h: int = 32
    input_w: int = 32
    patch_h: int = 4
    patch_w: int = 4
    embed_dim: int = 64

    @property
    def n_patches_h(self) -> int:
        return self.input_h // self.patch_h

    @property
    def n_patches_w(self) -> int:
        return self.input_w // self.patch_w

    @property
    def n_patches(self) -> int:
        return self.n_patches_h * self.n_patches_w


@jaxtyped(typechecker=beartype.beartype)
class DebugModel(eqx.Module):
    """Minimal encoder for debugging: just linear projection, no transformer.

    Takes patches, projects to embed_dim, returns mean as CLS token.
    If this doesn't learn, bug is in data/optimizer. If this learns but
    Transformer doesn't, bug is in transformer architecture.
    """

    cfg: Debug = eqx.field(static=True)
    proj: eqx.nn.Linear

    def __init__(self, cfg: Debug, *, key: PRNGKeyArray):
        self.cfg = cfg
        kernel_size = cfg.patch_h * cfg.patch_w
        self.proj = eqx.nn.Linear(kernel_size, cfg.embed_dim, key=key)

    def __call__(
        self,
        x_btk: Float[Array, "b t k"],
        *,
        grid: Int[Array, "b t 2"],
        key: PRNGKeyArray | None,
    ) -> EncoderOutput:
        """Forward pass matching Transformer interface."""
        del grid, key
        # Project patches: (B, T, K) -> (B, T, D)
        x = jax.vmap(jax.vmap(self.proj))(x_btk)

        # CLS = mean of patches, no actual CLS token
        cls = x.mean(axis=1, keepdims=True)  # (B, 1, D)

        return EncoderOutput(
            cls=cls,
            patches=x,
            reg=jnp.zeros((x.shape[0], 0, self.cfg.embed_dim)),
        )


# Union type for encoder configs
Config = Transformer | Debug

# ----------------------------
# Patchify/Unpatchify Helpers
# ----------------------------


@jaxtyped(typechecker=beartype.beartype)
def patchify(
    x_bhw: Float[Array, "b h w"], cfg: "Config"
) -> tuple[Float[Array, "b n k"], Int[Array, "b n 2"]]:
    """Convert 2D input to flattened patches with grid coordinates.

    Args:
        x_bhw: Input array of shape (B, H, W).
        cfg: Transformer config with patch sizes.

    Returns:
        x_bnk: Patches of shape (B, n_patches, patch_h * patch_w).
        grid_bn2: Grid coordinates (row, col) for each patch.
    """
    b, h, w = x_bhw.shape
    ph, pw = cfg.patch_h, cfg.patch_w
    n_h, n_w = h // ph, w // pw

    # Reshape to patches: (B, n_h, patch_h, n_w, patch_w) -> (B, n_h*n_w, patch_h*patch_w)
    x = x_bhw.reshape(b, n_h, ph, n_w, pw)
    x = x.transpose(0, 1, 3, 2, 4)  # (B, n_h, n_w, patch_h, patch_w)
    x_bnk = x.reshape(b, n_h * n_w, ph * pw)

    # Generate grid coordinates
    rows = jnp.arange(n_h)
    cols = jnp.arange(n_w)
    grid_row, grid_col = jnp.meshgrid(rows, cols, indexing="ij")
    grid = jnp.stack([grid_row.flatten(), grid_col.flatten()], axis=-1)  # (n, 2)
    grid_bn2 = jnp.broadcast_to(grid[None, :, :], (b, n_h * n_w, 2))  # (B, n, 2)

    return x_bnk, grid_bn2


@jaxtyped(typechecker=beartype.beartype)
def unpatchify(
    x_bnk: Float[Array, "b n k"], grid_bn2: Int[Array, "b n 2"], cfg: "Config"
) -> Float[Array, "b h w"]:
    """Reconstruct 2D input from patches using grid coordinates.

    Args:
        x_bnk: Patches of shape (B, n_patches, patch_h * patch_w).
        grid_bn2: Grid coordinates (row, col) for each patch.
        cfg: Transformer config with patch sizes.

    Returns:
        x_bhw: Reconstructed array of shape (B, H, W).
    """
    b, n, k = x_bnk.shape
    ph, pw = cfg.patch_h, cfg.patch_w
    n_h, n_w = cfg.n_patches_h, cfg.n_patches_w

    # Create output array
    x_bhw = jnp.zeros((b, n_h * ph, n_w * pw), dtype=x_bnk.dtype)

    # Place each patch at its grid position using scatter
    for i in range(n):
        row = grid_bn2[0, i, 0]  # Same grid for all batch elements
        col = grid_bn2[0, i, 1]
        patch = x_bnk[:, i].reshape(b, ph, pw)
        x_bhw = x_bhw.at[:, row * ph : (row + 1) * ph, col * pw : (col + 1) * pw].set(
            patch
        )

    return x_bhw


# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------


class PatchEmbed(eqx.Module):
    """Linear projection of flattened patches to embeddings."""

    proj: eqx.nn.Linear

    def __init__(self, cfg: Transformer, *, key: PRNGKeyArray):
        kernel_size = cfg.patch_h * cfg.patch_w
        self.proj = eqx.nn.Linear(kernel_size, cfg.embed_dim, key=key)

    @jaxtyped(typechecker=beartype.beartype)
    def __call__(self, x_btk: Float[Array, "b t k"]) -> Float[Array, "b t d"]:
        """Project flattened patches to embeddings."""
        weight = self.proj.weight
        bias = self.proj.bias
        out = jax.lax.dot_general(
            x_btk,
            weight,
            dimension_numbers=(((2,), (1,)), ((), ())),
            precision=jax.lax.Precision.HIGHEST,
        )
        if bias is not None:
            out = out + bias
        return out


class RopePositionEmbedding(eqx.Module):
    """2D Rotary Position Embedding for vision.

    Generates sin/cos embeddings for 2D grid positions. Applies separate rotations for H and W axes (axial RoPE).
    """

    d_head: int = eqx.field(static=True)
    theta: float = eqx.field(static=True)
    dtype: typing.Any = eqx.field(static=True)
    n_patches_h: int = eqx.field(static=True)
    n_patches_w: int = eqx.field(static=True)

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        base: float,
        n_patches_h: int,
        n_patches_w: int,
        *,
        dtype: typing.Any | None = None,
    ):
        self.d_head = embed_dim // n_heads
        assert self.d_head % 4 == 0, (
            f"head_dim must be divisible by 4, got {self.d_head}"
        )
        assert n_patches_h > 0, "n_patches_h must be positive"
        assert n_patches_w > 0, "n_patches_w must be positive"
        self.theta = float(base)
        self.dtype = jnp.float32 if dtype is None else dtype
        self.n_patches_h = n_patches_h
        self.n_patches_w = n_patches_w

    @staticmethod
    def precompute_freqs_cis(
        axis_dim: int, end: int, theta: float, dtype: typing.Any
    ) -> tuple[Float[Array, "end half_axis_dim"], Float[Array, "end half_axis_dim"]]:
        freqs = 1.0 / (
            theta ** (jnp.arange(0.0, axis_dim, 2)[jnp.newaxis, :] / axis_dim)
        )
        t = jnp.arange(float(end))
        freqs_outer = jnp.outer(t, freqs)
        return jnp.cos(freqs_outer).astype(dtype), jnp.sin(freqs_outer).astype(dtype)

    @jaxtyped(typechecker=beartype.beartype)
    def __call__(
        self, grid: Int[Array, "b t 2"]
    ) -> tuple[Float[Array, "b t d_head"], Float[Array, "b t d_head"]]:
        """Compute sin/cos embeddings for grid positions.

        Args:
            grid: Grid coordinates (row, col) for each patch, shape (B, T, 2).

        Returns:
            (sin, cos) each of shape (B, T, d_head).
        """
        axis_dim = self.d_head // 2

        with jax.ensure_compile_time_eval():
            cache_key = (
                axis_dim,
                self.n_patches_h,
                self.n_patches_w,
                self.theta,
                self.dtype,
            )
            if cache_key not in internal_rope_embedding_cache:
                cos_h, sin_h = self.precompute_freqs_cis(
                    axis_dim, self.n_patches_h, self.theta, self.dtype
                )
                cos_w, sin_w = self.precompute_freqs_cis(
                    axis_dim, self.n_patches_w, self.theta, self.dtype
                )
                internal_rope_embedding_cache[cache_key] = (cos_h, sin_h, cos_w, sin_w)

            cos_h_all, sin_h_all, cos_w_all, sin_w_all = internal_rope_embedding_cache[
                cache_key
            ]

        h_idx = grid[..., 0]
        w_idx = grid[..., 1]
        cos_h = cos_h_all[h_idx]
        sin_h = sin_h_all[h_idx]
        cos_w = cos_w_all[w_idx]
        sin_w = sin_w_all[w_idx]

        sin_hw = jnp.concatenate([sin_h, sin_w], axis=-1)
        cos_hw = jnp.concatenate([cos_h, cos_w], axis=-1)
        sin = jnp.tile(sin_hw, 2)
        cos = jnp.tile(cos_hw, 2)

        return sin, cos


@jaxtyped(typechecker=beartype.beartype)
def _rotate_half(x: Float[Array, "... d"]) -> Float[Array, "... d"]:
    """Rotate the second half of x to the first, negated."""
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


@jaxtyped(typechecker=beartype.beartype)
def _apply_rope(
    q: Float[Array, "b n h d"],
    k: Float[Array, "b n h d"],
    sin: Float[Array, "b t d"],
    cos: Float[Array, "b t d"],
    n_prefix: int,
) -> tuple[Float[Array, "b n h d"], Float[Array, "b n h d"]]:
    """Apply rotary embeddings to Q and K, skipping prefix tokens (CLS, etc).

    Args:
        q, k: Query and key tensors, shape (B, N, n_heads, head_dim).
        sin, cos: Sin/cos from RopePositionEmbedding, shape (B, T, head_dim).
        n_prefix: Number of prefix tokens to skip (CLS + register tokens).
    """
    # Split prefix and position-encoded parts
    q_prefix, q_pos = q[:, :n_prefix], q[:, n_prefix:]
    k_prefix, k_pos = k[:, :n_prefix], k[:, n_prefix:]

    # Apply RoPE: x * cos + rotate_half(x) * sin
    # sin/cos have shape (B, T, d), need to broadcast over heads
    sin_expanded = sin[:, :, None, :]  # (B, T, 1, d)
    cos_expanded = cos[:, :, None, :]  # (B, T, 1, d)

    q_pos = q_pos * cos_expanded + _rotate_half(q_pos) * sin_expanded
    k_pos = k_pos * cos_expanded + _rotate_half(k_pos) * sin_expanded

    q_out = jnp.concatenate([q_prefix, q_pos], axis=1)
    k_out = jnp.concatenate([k_prefix, k_pos], axis=1)
    return q_out, k_out


class Attention(eqx.Module):
    """Multi-head self-attention."""

    n_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    scale: float = eqx.field(static=True)
    use_qk_norm: bool = eqx.field(static=True)
    use_rope: bool = eqx.field(static=True)

    qkv: eqx.nn.Linear
    proj: eqx.nn.Linear
    q_norm: eqx.nn.LayerNorm | None
    k_norm: eqx.nn.LayerNorm | None

    def __init__(self, cfg: Transformer, *, key: PRNGKeyArray):
        key1, key2 = jr.split(key)

        self.n_heads = cfg.n_heads
        self.head_dim = cfg.embed_dim // cfg.n_heads
        self.scale = self.head_dim**-0.5
        self.use_qk_norm = cfg.use_qk_norm
        self.use_rope = cfg.use_rope

        self.qkv = eqx.nn.Linear(cfg.embed_dim, 3 * cfg.embed_dim, key=key1)
        self.proj = eqx.nn.Linear(cfg.embed_dim, cfg.embed_dim, key=key2)

        if cfg.use_qk_norm:
            self.q_norm = eqx.nn.LayerNorm(self.head_dim)
            self.k_norm = eqx.nn.LayerNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

    @jaxtyped(typechecker=beartype.beartype)
    def __call__(
        self,
        x_bnd: Float[Array, "b n d"],
        *,
        rope: tuple[Float[Array, "b t d_head"], Float[Array, "b t d_head"]]
        | None = None,
        n_prefix: int = 0,
    ) -> Float[Array, "b n d"]:
        b, n, d = x_bnd.shape

        # Compute QKV
        qkv = jax.vmap(jax.vmap(self.qkv))(x_bnd)  # (B, N, 3*D)
        qkv = qkv.reshape(b, n, 3, self.n_heads, self.head_dim)
        # JAX dot_product_attention expects (B, T, N, H) = (batch, seq_len, n_heads, head_dim)
        qkv = qkv.transpose(2, 0, 1, 3, 4)  # (3, B, N, n_heads, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, N, n_heads, head_dim)

        # QK-norm (applied per head)
        if self.q_norm is not None and self.k_norm is not None:
            q = jax.vmap(jax.vmap(jax.vmap(self.q_norm)))(q)
            k = jax.vmap(jax.vmap(jax.vmap(self.k_norm)))(k)

        # RoPE
        if self.use_rope:
            assert rope is not None, "use_rope=True but rope not provided"
            sin, cos = rope
            q, k = _apply_rope(q, k, sin, cos, n_prefix)

        # Use JAX's memory-efficient attention (flash attention on GPU)
        out = jax.nn.dot_product_attention(q, k, v, scale=self.scale)

        # Merge heads - already (B, N, n_heads, head_dim), just reshape
        out_bnd = out.reshape(b, n, d)

        # Output projection
        out_bnd = jax.vmap(jax.vmap(self.proj))(out_bnd)
        return out_bnd


class MLP(eqx.Module):
    """Feedforward network with GELU or SwiGLU activation."""

    use_swiglu: bool = eqx.field(static=True)
    dropout_rate: float = eqx.field(static=True)

    # For standard MLP
    fc1: eqx.nn.Linear | None
    fc2: eqx.nn.Linear | None
    # For SwiGLU
    w1: eqx.nn.Linear | None
    w2: eqx.nn.Linear | None
    w3: eqx.nn.Linear | None

    def __init__(self, cfg: Transformer, *, key: PRNGKeyArray):
        self.use_swiglu = cfg.use_swiglu
        self.dropout_rate = cfg.dropout
        hidden_dim = int(cfg.embed_dim * cfg.mlp_ratio)

        if cfg.use_swiglu:
            # SwiGLU: needs 2/3 of hidden dim for gate
            hidden_dim = int(2 * hidden_dim / 3)
            key1, key2, key3 = jr.split(key, 3)
            self.w1 = eqx.nn.Linear(cfg.embed_dim, hidden_dim, use_bias=False, key=key1)
            self.w2 = eqx.nn.Linear(hidden_dim, cfg.embed_dim, use_bias=False, key=key2)
            self.w3 = eqx.nn.Linear(cfg.embed_dim, hidden_dim, use_bias=False, key=key3)
            self.fc1 = None
            self.fc2 = None
        else:
            key1, key2 = jr.split(key)
            self.fc1 = eqx.nn.Linear(cfg.embed_dim, hidden_dim, key=key1)
            self.fc2 = eqx.nn.Linear(hidden_dim, cfg.embed_dim, key=key2)
            self.w1 = None
            self.w2 = None
            self.w3 = None

    @jaxtyped(typechecker=beartype.beartype)
    def __call__(
        self, x_bnd: Float[Array, "b n d"], *, key: PRNGKeyArray | None = None
    ) -> Float[Array, "b n d"]:
        if self.use_swiglu:
            assert self.w1 is not None and self.w2 is not None and self.w3 is not None
            x1 = jax.vmap(jax.vmap(self.w1))(x_bnd)
            x3 = jax.vmap(jax.vmap(self.w3))(x_bnd)
            x = jax.nn.silu(x1) * x3
            x = jax.vmap(jax.vmap(self.w2))(x)
        else:
            assert self.fc1 is not None and self.fc2 is not None
            x = jax.vmap(jax.vmap(self.fc1))(x_bnd)
            x = jax.nn.gelu(x)
            x = jax.vmap(jax.vmap(self.fc2))(x)

        # Dropout
        if key is not None and self.dropout_rate > 0:
            keep = jr.bernoulli(key, 1.0 - self.dropout_rate, x.shape)
            x = jnp.where(keep, x / (1.0 - self.dropout_rate), 0.0)

        return x


@jaxtyped(typechecker=beartype.beartype)
class Block(eqx.Module):
    """Transformer block with pre-norm."""

    use_layerscale: bool = eqx.field(static=True)

    norm1: eqx.nn.LayerNorm
    attn: Attention
    norm2: eqx.nn.LayerNorm
    mlp: MLP
    gamma1: Float[Array, " d"] | None
    gamma2: Float[Array, " d"] | None

    def __init__(self, cfg: Transformer, *, key: PRNGKeyArray):
        key1, key2 = jr.split(key)

        self.use_layerscale = cfg.use_layerscale
        self.norm1 = eqx.nn.LayerNorm(cfg.embed_dim)
        self.attn = Attention(cfg, key=key1)
        self.norm2 = eqx.nn.LayerNorm(cfg.embed_dim)
        self.mlp = MLP(cfg, key=key2)

        if cfg.use_layerscale:
            self.gamma1 = jnp.full(cfg.embed_dim, cfg.layerscale_init)
            self.gamma2 = jnp.full(cfg.embed_dim, cfg.layerscale_init)
        else:
            self.gamma1 = None
            self.gamma2 = None

    def __call__(
        self,
        x_bnd: Float[Array, "b n d"],
        *,
        key: PRNGKeyArray | None = None,
        rope: tuple[Float[Array, "b t d_head"], Float[Array, "b t d_head"]]
        | None = None,
        n_prefix: int = 0,
    ) -> Float[Array, "b n d"]:
        # Attention block (no dropout - uses flash attention)
        normed = jax.vmap(jax.vmap(self.norm1))(x_bnd)
        attn_out = self.attn(normed, rope=rope, n_prefix=n_prefix)
        if self.gamma1 is not None:
            attn_out = self.gamma1 * attn_out
        x = x_bnd + attn_out

        # MLP block
        normed = jax.vmap(jax.vmap(self.norm2))(x)
        mlp_out = self.mlp(normed, key=key)
        if self.gamma2 is not None:
            mlp_out = self.gamma2 * mlp_out
        x = x + mlp_out

        return x


@jaxtyped(typechecker=beartype.beartype)
class TransformerModel(eqx.Module):
    """Simple bidirectional transformer encoder.

    Takes pre-patchified input (B, T, kernel) + grid coordinates. This allows encoder to process any subset of patches (for MAE efficiency).

    Returns EncoderOutput for flexible pooling.
    """

    cfg: Transformer = eqx.field(static=True)

    patch_embed: PatchEmbed
    cls_tokens: Float[Array, "1 n_cls d"]
    reg_tokens: Float[Array, "1 n_reg d"] | None
    pos_embed_hw: Float[Array, "1 nh nw d"] | None
    rope_embed: RopePositionEmbedding | None
    blocks: Block | tuple[Block, ...]  # Stacked (scan) or tuple (loop)
    norm: eqx.nn.LayerNorm

    def __init__(self, cfg: Transformer, *, key: PRNGKeyArray):
        keys = jr.split(key, cfg.depth + 4)
        patch_key, cls_key, reg_key, pos_key, *block_keys = keys

        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg, key=patch_key)

        # CLS tokens (initialized with truncated normal)
        self.cls_tokens = (
            jr.truncated_normal(cls_key, -2, 2, (1, cfg.n_cls_tokens, cfg.embed_dim))
            * 0.02
        )

        # Register tokens (None if n_reg_tokens=0 to avoid zero-size array serialization issues)
        self.reg_tokens = (
            jr.truncated_normal(reg_key, -2, 2, (1, cfg.n_reg_tokens, cfg.embed_dim))
            * 0.02
            if cfg.n_reg_tokens > 0
            else None
        )

        # Positional embeddings
        if cfg.use_rope:
            self.rope_embed = RopePositionEmbedding(
                cfg.embed_dim,
                cfg.n_heads,
                cfg.rope_base,
                cfg.n_patches_h,
                cfg.n_patches_w,
            )
            self.pos_embed_hw = None
        else:
            self.rope_embed = None
            self.pos_embed_hw = (
                jr.truncated_normal(
                    pos_key, -2, 2, (1, cfg.n_patches_h, cfg.n_patches_w, cfg.embed_dim)
                )
                * 0.02
            )

        # Transformer blocks
        if cfg.use_scan:
            # Stacked via filter_vmap for scan optimization
            # Must stack keys into array - filter_vmap doesn't unpack Python lists
            self.blocks = eqx.filter_vmap(lambda k: Block(cfg, key=k))(
                jnp.stack(block_keys)
            )
        else:
            # Tuple of blocks for explicit loop (debugging)
            self.blocks = tuple(Block(cfg, key=k) for k in block_keys)
        self.norm = eqx.nn.LayerNorm(cfg.embed_dim)

    def __call__(
        self,
        x_btk: Float[Array, "b t k"],
        *,
        grid: Int[Array, "b t 2"],
        key: PRNGKeyArray | None,
    ) -> EncoderOutput:
        """Forward pass with pre-patchified input.

        Args:
            x_btk: Flattened patches (B, T, patch_h * patch_w). T can be any subset.
            grid: Grid coordinates (row, col) for each patch (B, T, 2).
            key: PRNG key for dropout (optional, None = no dropout).

        Returns:
            EncoderOutput with fields:
                cls: (B, n_cls_tokens, D) - CLS token embeddings
                patches: (B, T, D) - patch embeddings
                reg: (B, n_reg_tokens, D) - register tokens (empty if n_reg_tokens=0)
        """
        b, t, _ = x_btk.shape
        n_cls = self.cfg.n_cls_tokens
        n_reg = self.cfg.n_reg_tokens

        # Patch embedding
        x = self.patch_embed(x_btk)  # (B, T, D)

        # Add positional embeddings based on grid coordinates (absolute pos embed)
        if self.pos_embed_hw is not None:
            pos_embed = self.pos_embed_hw[0, grid[..., 0], grid[..., 1]]  # (B, T, D)
            x = x + pos_embed

        # Compute RoPE sin/cos if using rotary embeddings
        rope = self.rope_embed(grid) if self.rope_embed is not None else None

        # Prepend CLS tokens
        cls = jnp.broadcast_to(self.cls_tokens, (b, n_cls, self.cfg.embed_dim))
        x = jnp.concatenate([cls, x], axis=1)  # (B, n_cls+T, D)

        # Append register tokens
        if self.reg_tokens is not None:
            reg = jnp.broadcast_to(self.reg_tokens, (b, n_reg, self.cfg.embed_dim))
            x = jnp.concatenate([x, reg], axis=1)  # (B, n_cls+T+n_reg, D)

        # Transformer blocks
        if self.cfg.use_scan:
            # Scan implementation (faster compilation, better memory)
            assert not isinstance(self.blocks, tuple)
            block_arrays, block_static = eqx.partition(self.blocks, eqx.is_array)

            if key is None:

                def scan_fn_no_key(
                    x: Float[Array, "b n d"], arrays_i
                ) -> tuple[Float[Array, "b n d"], None]:
                    block = eqx.combine(arrays_i, block_static)
                    if self.cfg.grad_ckpt:
                        return jax.checkpoint(
                            lambda x: block(x, key=None, rope=rope, n_prefix=n_cls)
                        )(x), None
                    return block(x, key=None, rope=rope, n_prefix=n_cls), None

                x, _ = jax.lax.scan(scan_fn_no_key, x, block_arrays)
            else:
                block_keys = jr.split(key, self.cfg.depth)

                def scan_fn(
                    x: Float[Array, "b n d"], inputs: tuple
                ) -> tuple[Float[Array, "b n d"], None]:
                    arrays_i, key_i = inputs
                    block = eqx.combine(arrays_i, block_static)
                    if self.cfg.grad_ckpt:
                        return jax.checkpoint(
                            lambda x: block(x, key=key_i, rope=rope, n_prefix=n_cls)
                        )(x), None
                    return block(x, key=key_i, rope=rope, n_prefix=n_cls), None

                x, _ = jax.lax.scan(scan_fn, x, (block_arrays, block_keys))
        else:
            # Explicit loop (for debugging gradient flow)
            assert isinstance(self.blocks, tuple)
            if key is None:
                block_keys = [None] * len(self.blocks)
            else:
                block_keys = jr.split(key, self.cfg.depth)
            for block, blk_key in zip(self.blocks, block_keys):
                if self.cfg.grad_ckpt:
                    x = jax.checkpoint(
                        lambda x, b=block, k=blk_key: b(
                            x, key=k, rope=rope, n_prefix=n_cls
                        )
                    )(x)
                else:
                    x = block(x, key=blk_key, rope=rope, n_prefix=n_cls)

        # Final norm
        x = jax.vmap(jax.vmap(self.norm))(x)

        # Split output: [CLS tokens | patches | register tokens]
        cls_out = x[:, :n_cls]  # (B, n_cls, D)
        patches_out = x[:, n_cls : n_cls + t]  # (B, T, D)
        if n_reg > 0:
            reg_out = x[:, n_cls + t :]  # (B, n_reg, D)
        else:
            reg_out = jnp.zeros((b, 0, self.cfg.embed_dim))

        return EncoderOutput(cls=cls_out, patches=patches_out, reg=reg_out)
