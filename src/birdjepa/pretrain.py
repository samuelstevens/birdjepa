"""
BirdJEPA pretraining with LeJEPA objective.

Based on https://github.com/rbalestr-lab/lejepa/blob/main/MINIMAL.md
"""

import dataclasses
import logging
import pathlib

import torch
import birdjepa.helpers
import wandb
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

import birdjepa.data
import birdjepa.nn.jepa
import birdjepa.nn.transformer

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logger = logging.getLogger("birdjepa.pretrain")


@dataclasses.dataclass(frozen=True)
class Config:
    """Training configuration for BirdJEPA."""

    # Data
    data: birdjepa.data.XenoCanto = birdjepa.data.XenoCanto()
    """XenoCanto dataset configuration."""
    # Model
    model: birdjepa.nn.transformer.Config = birdjepa.nn.transformer.Config()
    """Vision transformer configuration."""
    # LeJEPA
    lamb: float = 0.02
    """Lambda for loss weighting between sigreg and invariance."""
    n_views: int = 4
    """Number of augmented views per spectrogram."""
    proj_dim: int = 256
    """Projection head output dimension."""
    # Training
    lr: float = 1e-4
    """Learning rate."""
    weight_decay: float = 0.05
    """Weight decay."""
    batch_size: int = 64
    """Batch size."""
    n_workers: int = 4
    """Number of dataloader workers."""
    epochs: int = 100
    """Number of training epochs."""
    log_every: int = 50
    """Log training metrics every N steps."""
    device: str = "cuda"
    """Device to train on."""
    # Checkpointing
    ckpt_dir: pathlib.Path = pathlib.Path("./checkpoints")
    """Directory for checkpoints."""
    save_every: int = 10
    """Save checkpoint every N epochs."""
    resume_from: str = ""
    """Path to checkpoint to resume from."""
    # Slurm
    slurm_acct: str = ""
    """Slurm account string. Empty means run locally."""
    slurm_partition: str = ""
    """Slurm partition."""
    n_hours: float = 24.0
    """Slurm job length in hours."""
    log_to: pathlib.Path = pathlib.Path("./logs")
    """Where to log Slurm job stdout/stderr."""


class MultiViewDataset(torch.utils.data.Dataset):
    """Wraps XenoCantoDataset to return multiple views per sample."""

    def __init__(self, base_dataset, n_views: int = 4):
        self.ds = base_dataset
        self.n_views = n_views

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # Each call to ds[idx] gives a different random crop
        views = torch.stack([self.ds[idx] for _ in range(self.n_views)])
        return views  # (V, T, M)


def save_checkpoint(path, epoch, model, optimizer, scheduler, scaler):
    """Save training checkpoint."""
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        },
        path,
    )
    logger.info("Saved checkpoint to %s", path)


def load_checkpoint(path, model, optimizer, scheduler, scaler):
    """Load training checkpoint. Returns starting epoch."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    logger.info("Loaded checkpoint from %s (epoch %d)", path, ckpt["epoch"])
    return ckpt["epoch"] + 1


def worker_fn(cfg: Config):
    """Main training function."""
    logging.basicConfig(level=logging.INFO, format=log_format)

    if cfg.device == "cuda":
        assert torch.cuda.is_available(), "CUDA not available"

    wandb.init(project="birdjepa", config=dataclasses.asdict(cfg))
    torch.manual_seed(42)

    # Data
    base_ds = birdjepa.data.XenoCantoDataset(cfg.data)
    train_ds = MultiViewDataset(base_ds, n_views=cfg.n_views)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.n_workers,
        pin_memory=True,
    )

    # Model
    encoder = birdjepa.nn.jepa.Encoder(cfg.model, cfg.proj_dim).to(cfg.device)
    sigreg = birdjepa.nn.jepa.SIGReg().to(cfg.device)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        encoder.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    warmup_steps = len(train_loader)
    total_steps = len(train_loader) * cfg.epochs
    s1 = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[s1, s2], milestones=[warmup_steps])

    scaler = GradScaler(enabled=cfg.device == "cuda")

    # Resume from checkpoint
    start_epoch = 0
    if cfg.resume_from:
        start_epoch = load_checkpoint(
            cfg.resume_from, encoder, optimizer, scheduler, scaler
        )

    # Ensure checkpoint directory exists
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    step = 0
    epochs = birdjepa.helpers.progress(
        range(start_epoch, cfg.epochs), every=1, desc="pretrain"
    )
    for epoch in epochs:
        encoder.train()
        epoch_loss = 0.0
        epoch_sigreg = 0.0
        epoch_inv = 0.0

        for views in train_loader:
            # views: (B, V, T, M)
            B, V, T, M = views.shape
            views = views.to(cfg.device, non_blocking=True)

            with autocast(cfg.device, dtype=torch.bfloat16):
                # Flatten views for encoder
                views_flat = views.view(B * V, T, M)
                _, proj = encoder(views_flat)  # (B*V, proj_dim)
                proj = proj.view(V, B, -1)  # (V, B, proj_dim)

                # LeJEPA losses
                inv_loss = (proj.mean(0) - proj).square().mean()
                sigreg_loss = sigreg(proj)
                loss = sigreg_loss * cfg.lamb + inv_loss * (1 - cfg.lamb)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1

            loss_val = loss.item()
            sigreg_val = sigreg_loss.item()
            inv_val = inv_loss.item()

            epoch_loss += loss_val
            epoch_sigreg += sigreg_val
            epoch_inv += inv_val

            if step % cfg.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    "step %d | loss %.4f | sigreg %.4f | inv %.4f | lr %.2e",
                    step,
                    loss_val,
                    sigreg_val,
                    inv_val,
                    lr,
                )
                wandb.log({
                    "step": step,
                    "train/loss": loss_val,
                    "train/sigreg": sigreg_val,
                    "train/inv": inv_val,
                    "lr": lr,
                })

        # Log epoch summary
        n_batches = len(train_loader)
        logger.info(
            "epoch %d | avg_loss %.4f | avg_sigreg %.4f | avg_inv %.4f",
            epoch,
            epoch_loss / n_batches,
            epoch_sigreg / n_batches,
            epoch_inv / n_batches,
        )

        # Save checkpoint
        if (epoch + 1) % cfg.save_every == 0:
            ckpt_path = cfg.ckpt_dir / f"epoch_{epoch:04d}.pt"
            save_checkpoint(ckpt_path, epoch, encoder, optimizer, scheduler, scaler)

    # Save final checkpoint
    save_checkpoint(
        cfg.ckpt_dir / "final.pt", cfg.epochs - 1, encoder, optimizer, scheduler, scaler
    )
    wandb.finish()
