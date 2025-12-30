"""Simple bidirectional transformer encoder."""

import dataclasses

import beartype
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor


# -----------------------------------------------------------------------------
# Patchify/Unpatchify Helpers
# -----------------------------------------------------------------------------


@jaxtyped(typechecker=beartype.beartype)
def patchify(
    x_bhw: Float[Tensor, "b h w"], cfg: "Config"
) -> tuple[Float[Tensor, "b n k"], Int[Tensor, "b n 2"]]:
    """Convert 2D input to flattened patches with grid coordinates.

    Args:
        x_bhw: Input tensor of shape (B, H, W).
        cfg: Transformer config with patch sizes.

    Returns:
        x_bnk: Patches of shape (B, n_patches, patch_h * patch_w).
        grid_bn2: Grid coordinates (row, col) for each patch.
    """
    b, h, w = x_bhw.shape
    ph, pw = cfg.patch_h, cfg.patch_w
    n_h, n_w = h // ph, w // pw

    # Reshape to patches: (B, n_h, patch_h, n_w, patch_w) -> (B, n_h*n_w, patch_h*patch_w)
    x = x_bhw.view(b, n_h, ph, n_w, pw)
    x = x.permute(0, 1, 3, 2, 4)  # (B, n_h, n_w, patch_h, patch_w)
    x_bnk = x.reshape(b, n_h * n_w, ph * pw)

    # Generate grid coordinates
    rows = torch.arange(n_h, device=x_bhw.device)
    cols = torch.arange(n_w, device=x_bhw.device)
    grid_row, grid_col = torch.meshgrid(rows, cols, indexing="ij")
    grid = torch.stack([grid_row.flatten(), grid_col.flatten()], dim=-1)  # (n, 2)
    grid_bn2 = grid.unsqueeze(0).expand(b, -1, -1)  # (B, n, 2)

    return x_bnk, grid_bn2


@jaxtyped(typechecker=beartype.beartype)
def unpatchify(
    x_bnk: Float[Tensor, "b n k"], grid_bn2: Int[Tensor, "b n 2"], cfg: "Config"
) -> Float[Tensor, "b h w"]:
    """Reconstruct 2D input from patches using grid coordinates.

    Args:
        x_bnk: Patches of shape (B, n_patches, patch_h * patch_w).
        grid_bn2: Grid coordinates (row, col) for each patch.
        cfg: Transformer config with patch sizes.

    Returns:
        x_bhw: Reconstructed tensor of shape (B, H, W).
    """
    b, n, k = x_bnk.shape
    ph, pw = cfg.patch_h, cfg.patch_w
    n_h, n_w = cfg.n_patches_h, cfg.n_patches_w

    # Create output tensor
    x_bhw = torch.zeros(b, n_h * ph, n_w * pw, device=x_bnk.device, dtype=x_bnk.dtype)

    # Place each patch at its grid position
    for i in range(n):
        row = grid_bn2[0, i, 0].item()  # Same grid for all batch elements
        col = grid_bn2[0, i, 1].item()
        patch = x_bnk[:, i].view(b, ph, pw)
        x_bhw[:, row * ph : (row + 1) * ph, col * pw : (col + 1) * pw] = patch

    return x_bhw


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
    n_cls_tokens: int = 1
    """Number of CLS tokens (Pixio uses 4-8)."""
    n_reg_tokens: int = 0
    """Number of register tokens (discarded at inference)."""

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
    """Linear projection of flattened patches to embeddings."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        kernel_size = cfg.patch_h * cfg.patch_w
        self.proj = nn.Linear(kernel_size, cfg.embed_dim)

    @jaxtyped(typechecker=beartype.beartype)
    def forward(self, x_btk: Float[Tensor, "b t k"]) -> Float[Tensor, "b t d"]:
        """Project flattened patches to embeddings.

        Args:
            x_btk: Flattened patches (B, T, patch_h * patch_w).

        Returns:
            x_btd: Patch embeddings (B, T, embed_dim).
        """
        return self.proj(x_btk)


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

    New API: Takes pre-patchified input (B, T, kernel) + grid coordinates.
    This allows encoder to process any subset of patches (for MAE efficiency).

    Returns a dict with "cls", "patches", and "reg" keys for flexible pooling.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.patch_embed = PatchEmbed(cfg)

        # CLS tokens (can be multiple for Pixio-style training)
        self.cls_tokens = nn.Parameter(torch.zeros(1, cfg.n_cls_tokens, cfg.embed_dim))

        # Register tokens (optional, discarded at inference)
        if cfg.n_reg_tokens > 0:
            self.reg_tokens = nn.Parameter(
                torch.zeros(1, cfg.n_reg_tokens, cfg.embed_dim)
            )
        else:
            self.reg_tokens = None

        # Positional embeddings: 2D grid that we index into based on grid coords
        # Shape: (1, n_patches_h, n_patches_w, embed_dim) for easy indexing
        if cfg.use_rope:
            raise NotImplementedError("RoPE not yet implemented")
        else:
            self.pos_embed_hw = nn.Parameter(
                torch.zeros(1, cfg.n_patches_h, cfg.n_patches_w, cfg.embed_dim)
            )

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.depth)])
        self.norm = nn.LayerNorm(cfg.embed_dim)

        self._init_weights()

    def _init_weights(self):
        if self.pos_embed_hw is not None:
            nn.init.trunc_normal_(self.pos_embed_hw, std=0.02)
        nn.init.trunc_normal_(self.cls_tokens, std=0.02)
        if self.reg_tokens is not None:
            nn.init.trunc_normal_(self.reg_tokens, std=0.02)

    @beartype.beartype
    def forward(
        self, x_btk: Float[Tensor, "b t k"], *, grid: Int[Tensor, "b t 2"]
    ) -> dict[str, Tensor]:
        """Forward pass with pre-patchified input.

        Args:
            x_btk: Flattened patches (B, T, patch_h * patch_w). T can be any subset.
            grid: Grid coordinates (row, col) for each patch (B, T, 2).

        Returns:
            Dict with keys:
                "cls": (B, n_cls_tokens, D) - CLS token embeddings
                "patches": (B, T, D) - patch embeddings
                "reg": (B, n_reg_tokens, D) - register tokens (empty if n_reg_tokens=0)
        """
        b, t, _ = x_btk.shape
        n_cls = self.cfg.n_cls_tokens
        n_reg = self.cfg.n_reg_tokens
        x = self.patch_embed(x_btk)  # (B, T, D)

        # Get positional embeddings for each patch based on grid coordinates
        if self.pos_embed_hw is not None:
            pos_embed = self.pos_embed_hw[0, grid[..., 0], grid[..., 1]]  # (B, T, D)
            x = x + pos_embed

        # Prepend CLS tokens
        cls = self.cls_tokens.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, n_cls+T, D)

        # Append register tokens (no positional embedding)
        if self.reg_tokens is not None:
            reg = self.reg_tokens.expand(b, -1, -1)
            x = torch.cat([x, reg], dim=1)  # (B, n_cls+T+n_reg, D)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Split output: [CLS tokens | patches | register tokens]
        cls_out = x[:, :n_cls]  # (B, n_cls, D)
        patches_out = x[:, n_cls : n_cls + t]  # (B, T, D)
        if n_reg > 0:
            reg_out = x[:, n_cls + t :]  # (B, n_reg, D)
        else:
            reg_out = x.new_empty(b, 0, self.cfg.embed_dim)

        return {"cls": cls_out, "patches": patches_out, "reg": reg_out}
