"""Simple bidirectional transformer encoder."""

import dataclasses

import beartype
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, jaxtyped
from torch import Tensor


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
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
    use_qk_norm: bool = False
    """Apply LayerNorm to queries and keys for attention stability."""
    use_swiglu: bool = False
    """Use SwiGLU activation instead of GELU in MLP."""
    use_layerscale: bool = False
    """Use learnable per-layer residual scaling (from CaiT)."""
    layerscale_init: float = 1e-4
    """Initial value for LayerScale parameters."""
    n_registers: int = 0
    """Number of register tokens (0 = none)."""

    @property
    def n_patches_h(self) -> int:
        return self.input_h // self.patch_h

    @property
    def n_patches_w(self) -> int:
        return self.input_w // self.patch_w

    @property
    def n_patches(self) -> int:
        return self.n_patches_h * self.n_patches_w


class PatchEmbed(nn.Module):
    """2D input to patch embeddings via convolution."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Conv2d(
            1,
            cfg.embed_dim,
            kernel_size=(cfg.patch_h, cfg.patch_w),
            stride=(cfg.patch_h, cfg.patch_w),
        )

    @jaxtyped(typechecker=beartype.beartype)
    def forward(self, x_bhw: Float[Tensor, "b h w"]) -> Float[Tensor, "b n d"]:
        x = x_bhw.unsqueeze(1)  # (B, 1, H, W)
        x_bnd = self.proj(x).flatten(2).transpose(1, 2)  # (B, N, D)
        return x_bnd


class Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.embed_dim // cfg.n_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(cfg.embed_dim, 3 * cfg.embed_dim, bias=True)
        self.proj = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=True)
        self.dropout = nn.Dropout(cfg.dropout)

        # Optional QK-norm
        self.q_norm = nn.LayerNorm(self.head_dim) if cfg.use_qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if cfg.use_qk_norm else nn.Identity()

        # RoPE placeholder
        self.use_rope = cfg.use_rope

    @jaxtyped(typechecker=beartype.beartype)
    def forward(self, x_bnd: Float[Tensor, "b n d"]) -> Float[Tensor, "b n d"]:
        b, n, d = x_bnd.shape

        qkv = self.qkv(x_bnd).reshape(b, n, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, head_dim)
        q, k, v = qkv.unbind(0)  # each (B, H, N, head_dim)

        # QK-norm (identity if disabled)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE would be applied here
        if self.use_rope:
            raise NotImplementedError("RoPE not yet implemented")

        # Scaled dot-product attention
        attn = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout.p if self.training else 0.0
        )

        # Merge heads
        out_bnd = attn.transpose(1, 2).reshape(b, n, d)
        out_bnd = self.proj(out_bnd)
        out_bnd = self.dropout(out_bnd)
        return out_bnd


class MLP(nn.Module):
    """Feedforward network with GELU or SwiGLU activation."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        hidden_dim = int(cfg.embed_dim * cfg.mlp_ratio)

        if self.cfg.use_swiglu:
            # SwiGLU: needs 2/3 of hidden dim for gate, scales back to same param count
            hidden_dim = int(2 * hidden_dim / 3)
            self.w1 = nn.Linear(cfg.embed_dim, hidden_dim, bias=False)
            self.w2 = nn.Linear(hidden_dim, cfg.embed_dim, bias=False)
            self.w3 = nn.Linear(cfg.embed_dim, hidden_dim, bias=False)  # gate
        else:
            self.fc1 = nn.Linear(cfg.embed_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, cfg.embed_dim)

        self.dropout = nn.Dropout(cfg.dropout)

    @jaxtyped(typechecker=beartype.beartype)
    def forward(self, x_bnd: Float[Tensor, "b n d"]) -> Float[Tensor, "b n d"]:
        if self.cfg.use_swiglu:
            x = F.silu(self.w1(x_bnd)) * self.w3(x_bnd)
            x = self.w2(x)
        else:
            x = self.fc1(x_bnd)
            x = F.gelu(x)
            x = self.fc2(x)
        return self.dropout(x)


class Block(nn.Module):
    """Transformer block with pre-norm."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.embed_dim)
        self.attn = Attention(cfg)
        self.norm2 = nn.LayerNorm(cfg.embed_dim)
        self.mlp = MLP(cfg)

        # LayerScale: learnable per-channel scaling for residuals
        if cfg.use_layerscale:
            self.gamma1 = nn.Parameter(cfg.layerscale_init * torch.ones(cfg.embed_dim))
            self.gamma2 = nn.Parameter(cfg.layerscale_init * torch.ones(cfg.embed_dim))
        else:
            self.gamma1 = None
            self.gamma2 = None

    @jaxtyped(typechecker=beartype.beartype)
    def forward(self, x_bnd: Float[Tensor, "b n d"]) -> Float[Tensor, "b n d"]:
        attn_out = self.attn(self.norm1(x_bnd))
        if self.gamma1 is not None:
            attn_out = self.gamma1 * attn_out
        x = x_bnd + attn_out

        mlp_out = self.mlp(self.norm2(x))
        if self.gamma2 is not None:
            mlp_out = self.gamma2 * mlp_out
        x = x + mlp_out

        return x


class Transformer(nn.Module):
    """Simple bidirectional transformer encoder.

    Takes 2D inputs of shape (B, H, W) and returns:
    - cls: (B, D) CLS token representation
    - patches: (B, N, D) patch representations
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        if cfg.n_registers > 0:
            raise NotImplementedError("Register tokens not yet implemented")

        self.patch_embed = PatchEmbed(cfg)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))

        # Positional embeddings (only used if not using RoPE)
        if cfg.use_rope:
            raise NotImplementedError("RoPE not yet implemented")
        else:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, 1 + cfg.n_patches, cfg.embed_dim)
            )

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.depth)])
        self.norm = nn.LayerNorm(cfg.embed_dim)

        self._init_weights()

    def _init_weights(self):
        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    @jaxtyped(typechecker=beartype.beartype)
    def forward(
        self, x_bhw: Float[Tensor, "b h w"]
    ) -> tuple[Float[Tensor, "b d"], Float[Tensor, "b n d"]]:
        """Forward pass."""
        b = x_bhw.shape[0]
        x = self.patch_embed(x_bhw)  # (B, N, D)

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 1+N, D)

        if self.pos_embed is not None:
            x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_bd: Float[Tensor, "b d"] = x[:, 0]
        patches_bnd: Float[Tensor, "b n d"] = x[:, 1:]
        return cls_bd, patches_bnd
