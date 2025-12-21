"""Simple bidirectional vision transformer for audio spectrograms."""

import dataclasses

import torch
import torch.nn as nn


@dataclasses.dataclass(frozen=True)
class Config:
    """Transformer configuration for audio spectrograms."""

    spec_t: int = 512
    """Spectrogram time dimension (frames)."""
    spec_m: int = 128
    """Spectrogram mel dimension."""
    patch_t: int = 16
    """Patch size in time dimension."""
    patch_m: int = 16
    """Patch size in mel dimension."""
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

    @property
    def n_patches_t(self) -> int:
        return self.spec_t // self.patch_t

    @property
    def n_patches_m(self) -> int:
        return self.spec_m // self.patch_m

    @property
    def n_patches(self) -> int:
        return self.n_patches_t * self.n_patches_m


class PatchEmbed(nn.Module):
    """Spectrogram to patch embeddings via 2D convolution."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Conv2d(
            1,
            cfg.embed_dim,
            kernel_size=(cfg.patch_t, cfg.patch_m),
            stride=(cfg.patch_t, cfg.patch_m),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, M) -> (B, 1, T, M)
        x = x.unsqueeze(1)
        # -> (B, D, T', M') -> (B, T'*M', D)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Transformer(nn.Module):
    """Simple bidirectional transformer encoder for spectrograms.

    Takes spectrograms of shape (B, T, M) and returns:
    - cls: (B, D) CLS token representation
    - patches: (B, N, D) patch representations
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.patch_embed = PatchEmbed(cfg)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + cfg.n_patches, cfg.embed_dim))

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=cfg.embed_dim,
                nhead=cfg.n_heads,
                dim_feedforward=int(cfg.embed_dim * cfg.mlp_ratio),
                dropout=cfg.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(cfg.depth)
        ])
        self.norm = nn.LayerNorm(cfg.embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Spectrogram of shape (B, T, M).

        Returns:
            cls: CLS token of shape (B, D).
            patches: Patch embeddings of shape (B, N, D).
        """
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, N, D)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 1+N, D)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls = x[:, 0]  # (B, D)
        patches = x[:, 1:]  # (B, N, D)
        return cls, patches
