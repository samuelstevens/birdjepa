"""JEPA-specific modules: SIGReg loss and encoder with projection head."""

import torch
import torch.nn as nn

import birdjepa.nn.transformer


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization from LeJEPA.

    Enforces isotropic Gaussian structure on embeddings, which is provably optimal
    for downstream tasks. See https://arxiv.org/abs/2511.08544
    """

    def __init__(self, knots: int = 17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """Compute SIGReg loss.

        Args:
            proj: Projected embeddings of shape (V, N, D) where V=views, N=batch, D=proj_dim.

        Returns:
            Scalar loss.
        """
        A = torch.randn(proj.size(-1), 256, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class Encoder(nn.Module):
    """Transformer encoder with projection head for LeJEPA."""

    def __init__(self, cfg: birdjepa.nn.transformer.Config, proj_dim: int = 256):
        super().__init__()
        self.encoder = birdjepa.nn.transformer.Transformer(cfg)
        embed_dim = cfg.embed_dim

        # 3-layer projection head (LeJEPA uses BatchNorm + GELU)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.BatchNorm1d(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.BatchNorm1d(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Spectrogram of shape (B, T, M).

        Returns:
            cls: CLS token of shape (B, D).
            proj: Projected embedding of shape (B, proj_dim).
        """
        cls, _ = self.encoder(x)  # (B, D)
        proj = self.proj(cls)  # (B, proj_dim)
        return cls, proj
