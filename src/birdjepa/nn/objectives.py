"""Training objectives: Supervised, LeJEPA, Pixio."""

import abc
import dataclasses

import beartype
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor
from torch.utils.data import Dataset

import birdjepa.nn.transformer


@beartype.beartype
class MultiViewDataset(Dataset):
    """Wraps a base dataset to return multiple views per sample.

    Each call to base_ds[idx] should return a different random crop/augmentation.
    """

    def __init__(self, base_ds: Dataset, n_views: int):
        self.ds = base_ds
        self.n_views = n_views
        self.n_classes = base_ds.n_classes

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        items = [self.ds[idx] for _ in range(self.n_views)]
        views = torch.stack([item["data"] for item in items])
        target = items[0]["target"]
        return {"views": views, "target": target}


# -----------------------------------------------------------------------------
# SIGReg (for LeJEPA)
# -----------------------------------------------------------------------------


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization.

    Encourages embeddings to follow a standard Gaussian distribution
    using characteristic function matching.
    """

    def __init__(self, n_knots: int = 17):
        super().__init__()
        t = torch.linspace(0, 3, n_knots, dtype=torch.float32)
        dt = 3 / (n_knots - 1)
        weights = torch.full((n_knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    @jaxtyped(typechecker=beartype.beartype)
    def forward(self, proj_vbd: Float[Tensor, "v b d"]) -> Float[Tensor, ""]:
        """Compute SIGReg loss."""
        device = proj_vbd.device
        A_dk = torch.randn(proj_vbd.size(-1), 256, device=device)
        A_dk = A_dk.div_(A_dk.norm(p=2, dim=0))
        x_t = (proj_vbd @ A_dk).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj_vbd.size(-2)
        return statistic.mean()


# -----------------------------------------------------------------------------
# Base Objective
# -----------------------------------------------------------------------------


class Objective(nn.Module, abc.ABC):
    """Abstract base class for training objectives."""

    @abc.abstractmethod
    def wrap_dataset(self, base_ds: Dataset) -> Dataset:
        """Wrap base dataset with objective-specific behavior."""
        pass

    @abc.abstractmethod
    def forward(
        self, batch: dict, encoder: nn.Module
    ) -> tuple[dict[str, Tensor], Tensor, Tensor]:
        """Compute loss terms and return data for online probe.

        Args:
            batch: Dict from dataloader.
            encoder: The encoder module (not owned by objective).

        Returns:
            Tuple of (losses, embeddings, targets) where:
            - losses: Dict of named loss terms (e.g., {"ce": ..., "sigreg": ...})
            - embeddings: Detached embeddings for probe training
            - targets: Corresponding targets (may be repeated for multi-view)
        """
        pass


# -----------------------------------------------------------------------------
# Supervised
# -----------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SupervisedConfig:
    """Config for supervised objective."""

    pass  # No extra config needed


class Supervised(Objective):
    """Supervised classification objective."""

    def __init__(self, embed_dim: int, n_classes: int):
        super().__init__()
        self.head = nn.Linear(embed_dim, n_classes)

    def wrap_dataset(self, base_ds: Dataset) -> Dataset:
        return base_ds

    def forward(
        self, batch: dict, encoder: nn.Module
    ) -> tuple[dict[str, Tensor], Tensor, Tensor]:
        data_bhw: Float[Tensor, "b h w"] = batch["data"]
        target_b: Int[Tensor, " b"] = batch["target"]
        emb_bd, _ = encoder(data_bhw)
        logits_bc = self.head(emb_bd)
        return {"ce": F.cross_entropy(logits_bc, target_b)}, emb_bd.detach(), target_b


# -----------------------------------------------------------------------------
# LeJEPA
# -----------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LeJEPAConfig:
    """Config for LeJEPA objective."""

    n_views: int = 4
    """Number of augmented views per sample."""
    proj_dim: int = 16
    """Projection head output dimension."""
    lamb: float = 0.02
    """Weight for SIGReg loss (1 - lamb for invariance)."""


class LeJEPA(Objective):
    """LeJEPA: Joint-Embedding Predictive Architecture with SIGReg."""

    def __init__(self, embed_dim: int, cfg: LeJEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, cfg.proj_dim),
        )
        self.sigreg = SIGReg()

    def wrap_dataset(self, base_ds: Dataset) -> Dataset:
        return MultiViewDataset(base_ds, n_views=self.cfg.n_views)

    def forward(
        self, batch: dict, encoder: nn.Module
    ) -> tuple[dict[str, Tensor], Tensor, Tensor]:
        views_bvhw: Float[Tensor, "b v h w"] = batch["views"]
        target_b: Int[Tensor, " b"] = batch["target"]
        b, v, h, w = views_bvhw.shape

        # Flatten views for encoder
        views_flat = views_bvhw.view(b * v, h, w)
        emb_nd, _ = encoder(views_flat)  # n = b*v
        proj_np = self.proj(emb_nd)
        proj_vbp: Float[Tensor, "v b p"] = proj_np.view(b, v, -1).permute(1, 0, 2)

        # LeJEPA losses
        inv_loss = (proj_vbp.mean(0) - proj_vbp).square().mean()
        sigreg_loss = self.sigreg(proj_vbp)

        # Repeat targets for each view
        target_n = target_b.repeat_interleave(v)

        losses = {
            "inv": inv_loss * (1 - self.cfg.lamb),
            "sigreg": sigreg_loss * self.cfg.lamb,
        }
        return losses, emb_nd.detach(), target_n


# -----------------------------------------------------------------------------
# Pixio (MAE-style)
# -----------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PixioConfig:
    """Config for Pixio (MAE) objective."""

    decoder_depth: int = 8
    """Number of decoder transformer blocks."""
    decoder_dim: int = 512
    """Decoder embedding dimension."""
    mask_ratio: float = 0.75
    """Fraction of patches to mask."""
    block_size: int = 4
    """Mask in blocks of this many patches (1 = random patch masking)."""


class Pixio(Objective):
    """Pixio: Enhanced MAE with deeper decoder and block masking.

    TODO: Implement block masking and decoder.
    """

    def __init__(
        self,
        encoder_cfg: birdjepa.nn.transformer.Config,
        cfg: PixioConfig,
    ):
        super().__init__()
        self.encoder_cfg = encoder_cfg
        self.cfg = cfg

        # Decoder: maps encoder outputs back to pixels
        # TODO: Implement proper transformer decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoder_cfg.embed_dim, cfg.decoder_dim),
            nn.GELU(),
            nn.Linear(cfg.decoder_dim, encoder_cfg.patch_h * encoder_cfg.patch_w),
        )

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, encoder_cfg.embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def wrap_dataset(self, base_ds: Dataset) -> Dataset:
        return base_ds

    def forward(
        self, batch: dict, encoder: nn.Module
    ) -> tuple[dict[str, Tensor], Tensor, Tensor]:
        # TODO: Implement proper MAE forward with block masking
        # For now, placeholder that computes dummy loss
        data_bhw: Float[Tensor, "b h w"] = batch["data"]
        target_b: Int[Tensor, " b"] = batch["target"]
        emb_bd, _ = encoder(data_bhw)

        # Placeholder: predict patch pixels from embeddings
        pred = self.decoder(emb_bd)

        # Dummy MSE loss (not actually doing masking yet)
        loss = pred.mean() * 0  # Zero loss placeholder

        return {"mse": loss}, emb_bd.detach(), target_b


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------

Config = SupervisedConfig | LeJEPAConfig | PixioConfig


@beartype.beartype
def make_objective(
    cfg: Config, model_cfg: birdjepa.nn.transformer.Config, n_classes: int
) -> Objective:
    """Create an objective from config."""
    if isinstance(cfg, SupervisedConfig):
        return Supervised(model_cfg.embed_dim, n_classes)
    elif isinstance(cfg, LeJEPAConfig):
        return LeJEPA(model_cfg.embed_dim, cfg)
    elif isinstance(cfg, PixioConfig):
        return Pixio(model_cfg, cfg)
    else:
        raise ValueError(f"Unknown objective config: {type(cfg)}")
