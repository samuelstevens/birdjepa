"""Training objectives: Supervised, LeJEPA, Pixio."""

import abc
import dataclasses

import beartype
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int, jaxtyped
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
        first = items[0]
        return {"views": views, "target": first["target"], "index": first["index"]}


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
# Source Prediction (DIET-style)
# -----------------------------------------------------------------------------


class SourceHead(nn.Module):
    """Low-rank source prediction head for DIET-style self-supervision.

    Predicts which source recording a sample came from using concatenated CLS tokens.
    Uses low-rank projection to handle large number of sources efficiently.

    Reference: Perch 2.0 (https://arxiv.org/abs/2508.04665)
    """

    def __init__(self, input_dim: int, n_sources: int, rank: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, rank),
            nn.ReLU(),
            nn.Linear(rank, n_sources),
        )

    def forward(self, cls_bcd: Float[Tensor, "b c d"]) -> Float[Tensor, "b s"]:
        """Predict source from concatenated CLS tokens.

        Args:
            cls_bcd: CLS tokens (B, n_cls, embed_dim).

        Returns:
            logits_bs: Source prediction logits (B, n_sources).
        """
        # Flatten CLS tokens: (B, n_cls, D) -> (B, n_cls * D)
        b = cls_bcd.shape[0]
        cls_flat = cls_bcd.view(b, -1)
        return self.head(cls_flat)


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
    """Supervised classification objective with optional source prediction."""

    def __init__(
        self,
        encoder_cfg: birdjepa.nn.transformer.Config,
        n_classes: int,
        n_sources: int = 0,
        source_rank: int = 0,
        source_weight: float = 1.0,
    ):
        super().__init__()
        self.encoder_cfg = encoder_cfg
        self.source_weight = source_weight
        self.head = nn.Linear(encoder_cfg.embed_dim, n_classes)

        # Optional source prediction head
        if source_rank > 0 and n_sources > 0:
            input_dim = encoder_cfg.n_cls_tokens * encoder_cfg.embed_dim
            self.source_head = SourceHead(input_dim, n_sources, source_rank)
        else:
            self.source_head = None

    def wrap_dataset(self, base_ds: Dataset) -> Dataset:
        return base_ds

    def forward(
        self, batch: dict, encoder: nn.Module
    ) -> tuple[dict[str, Tensor], Tensor, Tensor]:
        data_bhw: Float[Tensor, "b h w"] = batch["data"]
        target_b: Int[Tensor, " b"] = batch["target"]

        # Patchify and encode
        x_bnk, grid_bn2 = birdjepa.nn.transformer.patchify(data_bhw, self.encoder_cfg)
        out = encoder(x_bnk, grid=grid_bn2)

        # Use mean of CLS tokens for classification
        emb_bd = out["cls"].mean(dim=1)  # (B, D)

        losses = {"ce": F.cross_entropy(self.head(emb_bd), target_b)}

        # Optional source prediction loss (uses index as source ID)
        if self.source_head is not None:
            source_logits = self.source_head(out["cls"])
            losses["source"] = self.source_weight * F.cross_entropy(
                source_logits, batch["index"]
            )

        return losses, emb_bd.detach(), target_b


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

    def __init__(self, encoder_cfg: birdjepa.nn.transformer.Config, cfg: LeJEPAConfig):
        super().__init__()
        self.encoder_cfg = encoder_cfg
        self.cfg = cfg
        self.proj = nn.Sequential(
            nn.Linear(encoder_cfg.embed_dim, 2048),
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

        # Patchify and encode
        x_bnk, grid_bn2 = birdjepa.nn.transformer.patchify(views_flat, self.encoder_cfg)
        out = encoder(x_bnk, grid=grid_bn2)

        # Use mean of CLS tokens for LeJEPA
        emb_nd = out["cls"].mean(dim=1)  # (n, D) where n = b*v

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


@jaxtyped(typechecker=beartype.beartype)
def make_block_mask(
    batch_size: int,
    n_h: int,
    n_w: int,
    block_size: int,
    mask_ratio: float,
    *,
    generator: torch.Generator,
) -> Bool[Tensor, "batch n"]:
    """Generate block mask with adjustment for exact mask count.

    1. Divide grid into blocks and randomly mask blocks
    2. Adjust to hit exact target count (add/remove individual patches)

    Args:
        batch_size: Number of samples in batch.
        n_h: Number of patches in height.
        n_w: Number of patches in width.
        block_size: Size of masking blocks (2 = 2x2 blocks).
        mask_ratio: Target fraction of patches to mask.
        generator: Random number generator for reproducibility.

    Returns:
        mask_bn: Boolean tensor (B, n_h * n_w) where True = masked.
    """
    device = generator.device
    n_patches = n_h * n_w
    n_masked_target = int(n_patches * mask_ratio)

    # Number of blocks in each dimension
    n_blocks_h = (n_h + block_size - 1) // block_size
    n_blocks_w = (n_w + block_size - 1) // block_size
    n_blocks = n_blocks_h * n_blocks_w

    # Compute max blocks to mask such that we never exceed target.
    # This ensures we only ever ADD patches in adjustment, never remove them.
    max_patches_per_block = block_size * block_size
    n_blocks_masked = min(n_masked_target // max_patches_per_block, n_blocks)

    masks = []
    for _ in range(batch_size):
        # Create block-level mask
        block_indices = torch.randperm(n_blocks, device=device, generator=generator)[
            :n_blocks_masked
        ]
        block_mask = torch.zeros(
            n_blocks_h, n_blocks_w, dtype=torch.bool, device=device
        )
        for idx in block_indices:
            bh = idx // n_blocks_w
            bw = idx % n_blocks_w
            block_mask[bh, bw] = True

        # Expand blocks to patch-level mask
        patch_mask = torch.zeros(n_h, n_w, dtype=torch.bool, device=device)
        for bh in range(n_blocks_h):
            for bw in range(n_blocks_w):
                if block_mask[bh, bw]:
                    h_start = bh * block_size
                    h_end = min(h_start + block_size, n_h)
                    w_start = bw * block_size
                    w_end = min(w_start + block_size, n_w)
                    patch_mask[h_start:h_end, w_start:w_end] = True

        patch_mask = patch_mask.flatten()

        # Add individual patches to reach exact target count
        n_masked_current = patch_mask.sum().item()
        assert n_masked_current <= n_masked_target
        if n_masked_current < n_masked_target:
            unmasked, _ = (~patch_mask).nonzero(as_tuple=True)
            n_to_mask = n_masked_target - n_masked_current
            extra = unmasked[
                torch.randperm(len(unmasked), device=device, generator=generator)[
                    :n_to_mask
                ]
            ]
            patch_mask[extra] = True

        masks.append(patch_mask)

    return torch.stack(masks)


@dataclasses.dataclass(frozen=True)
class PixioConfig:
    """Config for Pixio (MAE) objective."""

    decoder_depth: int = 8
    """Number of decoder transformer blocks."""
    decoder_dim: int = 512
    """Decoder embedding dimension."""
    decoder_heads: int = 8
    """Number of decoder attention heads."""
    mask_ratio: float = 0.75
    """Fraction of patches to mask."""
    block_size: int = 2
    """Mask in blocks of this many patches (1 = random patch masking)."""
    seed: int = 42
    """Base seed for masking RNG. Combined with worker_id for per-worker seeding."""


class PixioDecoder(nn.Module):
    """Lightweight decoder for MAE reconstruction.

    CLS token participates in decoder attention to learn better representations,
    as recommended by Pixio paper.
    """

    def __init__(self, encoder_cfg: birdjepa.nn.transformer.Config, cfg: PixioConfig):
        super().__init__()
        self.encoder_cfg = encoder_cfg
        self.cfg = cfg

        # Project encoder dim to decoder dim
        self.embed = nn.Linear(encoder_cfg.embed_dim, cfg.decoder_dim)

        # Decoder blocks (simpler than encoder, shared architecture)
        decoder_cfg = birdjepa.nn.transformer.Config(
            input_h=encoder_cfg.input_h,
            input_w=encoder_cfg.input_w,
            patch_h=encoder_cfg.patch_h,
            patch_w=encoder_cfg.patch_w,
            embed_dim=cfg.decoder_dim,
            depth=cfg.decoder_depth,
            n_heads=cfg.decoder_heads,
        )
        self.blocks = nn.ModuleList([
            birdjepa.nn.transformer.Block(decoder_cfg) for _ in range(cfg.decoder_depth)
        ])
        self.norm = nn.LayerNorm(cfg.decoder_dim)

        # Project to pixel space
        kernel_size = encoder_cfg.patch_h * encoder_cfg.patch_w
        self.head = nn.Linear(cfg.decoder_dim, kernel_size)

        # Positional embeddings for decoder (patches only, CLS has no position)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, encoder_cfg.n_patches, cfg.decoder_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    @jaxtyped(typechecker=beartype.beartype)
    def forward(
        self, cls_be: Float[Tensor, "b e"], x_bne: Float[Tensor, "b n e"]
    ) -> tuple[Float[Tensor, "b d"], Float[Tensor, "b n k"]]:
        """Decode to pixel predictions with CLS participating in attention.

        Args:
            cls_be: CLS token from encoder (B, encoder_dim).
            x_bne: Encoder output + mask tokens (B, n_patches, encoder_dim).

        Returns:
            cls_out: Decoded CLS token (B, decoder_dim) for downstream use.
            pred_bnk: Pixel predictions (B, n_patches, patch_h * patch_w).
        """
        # Project to decoder dim
        cls = self.embed(cls_be.unsqueeze(1))  # (B, 1, decoder_dim)
        x = self.embed(x_bne)  # (B, N, decoder_dim)

        # Add positional embeddings (patches only, CLS has no position)
        x = x + self.pos_embed

        # Concatenate CLS + patches for attention
        x = torch.cat([cls, x], dim=1)  # (B, 1+N, decoder_dim)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Split CLS and patches
        cls_out = x[:, 0]  # (B, decoder_dim)
        patches = x[:, 1:]  # (B, N, decoder_dim)

        return cls_out, self.head(patches)


class Pixio(Objective):
    """Pixio: Enhanced MAE with deeper decoder and block masking.

    Reference: "In Pursuit of Pixel Supervision for Visual Pre-training"
    https://arxiv.org/abs/2512.15715
    """

    def __init__(
        self,
        encoder_cfg: birdjepa.nn.transformer.Config,
        cfg: PixioConfig,
        probe_pooling: str = "cls",
        n_sources: int = 0,
        source_rank: int = 0,
        source_weight: float = 1.0,
    ):
        super().__init__()
        self.encoder_cfg = encoder_cfg
        self.cfg = cfg
        self.probe_pooling = probe_pooling
        self.source_weight = source_weight

        # Decoder
        self.decoder = PixioDecoder(encoder_cfg, cfg)

        # Mask token (in encoder dim, will be projected by decoder)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, encoder_cfg.embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Optional source prediction head
        if source_rank > 0 and n_sources > 0:
            input_dim = encoder_cfg.n_cls_tokens * encoder_cfg.embed_dim
            self.source_head = SourceHead(input_dim, n_sources, source_rank)
        else:
            self.source_head = None

        # RNG for masking (will be moved to correct device on first forward)
        self._rng: torch.Generator | None = None

    def _get_rng(self, device: torch.device) -> torch.Generator:
        """Get or create generator on the correct device."""
        if self._rng is None:
            self._rng = torch.Generator(device=device)
            self._rng.manual_seed(self.cfg.seed)

        assert self._rng.device == device
        return self._rng

    def wrap_dataset(self, base_ds: Dataset) -> Dataset:
        return base_ds

    def forward(
        self, batch: dict, encoder: nn.Module
    ) -> tuple[dict[str, Tensor], Tensor, Tensor]:
        data_bhw: Float[Tensor, "b h w"] = batch["data"]
        target_b: Int[Tensor, " b"] = batch["target"]
        b = data_bhw.shape[0]

        # 1. Patchify
        x_bnk, grid_bn2 = birdjepa.nn.transformer.patchify(data_bhw, self.encoder_cfg)
        n_patches = x_bnk.shape[1]

        # 2. Generate mask (True = masked)
        mask_bn = make_block_mask(
            b,
            self.encoder_cfg.n_patches_h,
            self.encoder_cfg.n_patches_w,
            self.cfg.block_size,
            self.cfg.mask_ratio,
            generator=self._get_rng(data_bhw.device),
        )

        # 3. Select visible patches for encoder
        visible_mask = ~mask_bn  # (B, N), True = visible
        n_visible_per_sample = visible_mask.sum(dim=1)
        assert (n_visible_per_sample == n_visible_per_sample[0]).all()
        n_visible = n_visible_per_sample[0].item()
        n_masked = n_patches - n_visible
        assert n_masked == int(n_patches * self.cfg.mask_ratio)

        # Gather visible patches (same number for all samples due to adjustment)
        x_visible = torch.zeros(b, n_visible, x_bnk.shape[2], device=x_bnk.device)
        grid_visible = torch.zeros(
            b, n_visible, 2, dtype=torch.long, device=x_bnk.device
        )
        for i in range(b):
            vis_idx = visible_mask[i].nonzero(as_tuple=True)[0]
            x_visible[i] = x_bnk[i, vis_idx]
            grid_visible[i] = grid_bn2[i, vis_idx]

        # 4. Encode visible patches only
        out = encoder(x_visible, grid=grid_visible)
        enc_visible = out["patches"]
        assert enc_visible.shape == (b, n_visible, self.encoder_cfg.embed_dim)

        # Mean of CLS tokens for decoder (handles n_cls_tokens >= 1)
        cls_bd = out["cls"].mean(dim=1)  # (B, D)

        # 5. Prepare decoder input: insert mask tokens at masked positions
        dec_input = torch.zeros(
            b, n_patches, self.encoder_cfg.embed_dim, device=x_bnk.device
        )
        for i in range(b):
            vis_idx = visible_mask[i].nonzero(as_tuple=True)[0]
            mask_idx = mask_bn[i].nonzero(as_tuple=True)[0]
            dec_input[i, vis_idx] = enc_visible[i]
            dec_input[i, mask_idx] = self.mask_token[0, 0]

        # 6. Decode to pixel predictions (CLS participates in attention)
        # Reconstruction loss flows back to encoder CLS via decoder attention
        _, pred_bnk = self.decoder(cls_bd, dec_input)
        assert pred_bnk.shape == x_bnk.shape

        # 7. Compute MSE loss on masked patches only (with per-patch normalization)
        target_bnk = x_bnk.clone()
        mean = target_bnk.mean(dim=-1, keepdim=True)
        var = target_bnk.var(dim=-1, keepdim=True)
        target_bnk = (target_bnk - mean) / (var + 1e-6).sqrt()
        loss = F.mse_loss(pred_bnk[mask_bn], target_bnk[mask_bn])

        # 8. Get embedding for probe based on pooling config
        if self.probe_pooling == "cls":
            emb_bd = out["cls"].mean(dim=1)  # Mean of CLS tokens
        elif self.probe_pooling == "patches":
            emb_bd = out["patches"].mean(dim=1)  # Mean of visible patches
        else:
            raise ValueError(f"Unknown probe_pooling: {self.probe_pooling}")

        losses = {"mse": loss}

        # 9. Optional source prediction loss (uses index as source ID)
        if self.source_head is not None:
            source_logits = self.source_head(out["cls"])
            losses["source"] = self.source_weight * F.cross_entropy(
                source_logits, batch["index"]
            )

        return losses, emb_bd.detach(), target_b


###########
# Factory #
###########

Config = SupervisedConfig | LeJEPAConfig | PixioConfig


@beartype.beartype
def make_objective(
    cfg: Config,
    model_cfg: birdjepa.nn.transformer.Config,
    n_classes: int,
    probe_pooling: str = "cls",
    n_sources: int = 0,
    source_rank: int = 0,
    source_weight: float = 1.0,
) -> Objective:
    """Create an objective from config."""
    if isinstance(cfg, SupervisedConfig):
        return Supervised(
            model_cfg,
            n_classes,
            n_sources=n_sources,
            source_rank=source_rank,
            source_weight=source_weight,
        )
    elif isinstance(cfg, LeJEPAConfig):
        return LeJEPA(model_cfg, cfg)
    elif isinstance(cfg, PixioConfig):
        return Pixio(
            model_cfg,
            cfg,
            probe_pooling=probe_pooling,
            n_sources=n_sources,
            source_rank=source_rank,
            source_weight=source_weight,
        )
    else:
        raise ValueError(f"Unknown objective config: {type(cfg)}")
