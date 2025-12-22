"""
BirdJEPA pretraining.

Based on https://github.com/rbalestr-lab/lejepa/blob/main/MINIMAL.md
"""

import dataclasses
import logging
import pathlib

import beartype
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

import birdjepa.data
import birdjepa.helpers
import birdjepa.nn.objectives
import birdjepa.nn.transformer

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logger = logging.getLogger("birdjepa.pretrain")


@dataclasses.dataclass(frozen=True)
class Config:
    """Training configuration for BirdJEPA."""

    # Data
    train_data: birdjepa.data.Config = birdjepa.data.XenoCanto()
    """Training dataset configuration."""
    test_data: birdjepa.data.Config = birdjepa.data.XenoCanto()
    """Test dataset configuration. Required; will fail if not provided."""
    # Model
    model: birdjepa.nn.transformer.Config = birdjepa.nn.transformer.Config()
    """Vision transformer configuration."""
    # Objective
    objective: birdjepa.nn.objectives.Config = birdjepa.nn.objectives.SupervisedConfig()
    """Training objective configuration."""
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
    ckpt_to: pathlib.Path = pathlib.Path("./checkpoints")
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


@beartype.beartype
def make_dataset(cfg: birdjepa.data.Config):
    """Create dataset from config."""
    if isinstance(cfg, birdjepa.data.XenoCanto):
        return birdjepa.data.XenoCantoDataset(cfg)
    elif isinstance(cfg, birdjepa.data.Cifar100):
        return birdjepa.data.Cifar100Dataset(cfg)
    else:
        raise ValueError(f"Unknown data config type: {type(cfg)}")


def save_checkpoint(path, epoch, encoder, objective, optimizer, scheduler, scaler):
    """Save training checkpoint."""
    ckpt = {
        "epoch": epoch,
        "encoder": encoder.state_dict(),
        "objective": objective.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
    }
    torch.save(ckpt, path)
    logger.info("Saved checkpoint to %s", path)


def load_checkpoint(path, encoder, objective, optimizer, scheduler, scaler):
    """Load training checkpoint. Returns starting epoch."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    encoder.load_state_dict(ckpt["encoder"])
    objective.load_state_dict(ckpt["objective"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    logger.info("Loaded checkpoint from %s (epoch %d)", path, ckpt["epoch"])
    return ckpt["epoch"] + 1


@beartype.beartype
def worker_fn(cfg: Config):
    """Main training function."""
    logging.basicConfig(level=logging.INFO, format=log_format)

    if cfg.device == "cuda":
        assert torch.cuda.is_available(), "CUDA not available"

    wandb.init(project="birdjepa", config=dataclasses.asdict(cfg))
    torch.manual_seed(42)

    # Data
    base_ds = make_dataset(cfg.train_data)
    assert cfg.test_data is not None, "test_data is required for evaluation"
    test_ds = make_dataset(cfg.test_data)

    # Model and objective
    n_classes = base_ds.n_classes
    encoder = birdjepa.nn.transformer.Transformer(cfg.model).to(cfg.device)
    objective = birdjepa.nn.objectives.make_objective(
        cfg.objective, cfg.model, n_classes
    ).to(cfg.device)

    # Wrap dataset for objective (e.g., multi-view for LeJEPA)
    train_ds = objective.wrap_dataset(base_ds)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.n_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.n_workers,
        pin_memory=True,
    )

    # Online linear probe
    probe = nn.Sequential(
        nn.LayerNorm(cfg.model.embed_dim),
        nn.Linear(cfg.model.embed_dim, n_classes),
    ).to(cfg.device)

    # Optimizer and scheduler
    param_groups = [
        {
            "params": encoder.parameters(),
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": objective.parameters(),
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
        },
        {"params": probe.parameters(), "lr": 1e-3, "weight_decay": 1e-7},
    ]
    optimizer = torch.optim.AdamW(param_groups)
    warmup_steps = len(train_loader)
    total_steps = len(train_loader) * cfg.epochs
    s1 = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[s1, s2], milestones=[warmup_steps])

    scaler = GradScaler(enabled=cfg.device == "cuda")

    # Resume from checkpoint
    start_epoch = 0
    if cfg.resume_from:
        raise NotImplementedError("Resuming from checkpoint not yet supported")

    # Ensure checkpoint directory exists
    cfg.ckpt_to.mkdir(parents=True, exist_ok=True)

    # Training loop
    step = 0
    epochs = birdjepa.helpers.progress(
        range(start_epoch, cfg.epochs), every=1, desc="pretrain"
    )
    for epoch in epochs:
        encoder.train()
        objective.train()
        probe.train()
        epoch_losses: dict[str, float] = {}
        epoch_probe_loss = 0.0

        for batch in train_loader:
            batch = {k: v.to(cfg.device, non_blocking=True) for k, v in batch.items()}

            with autocast(cfg.device, dtype=torch.bfloat16):
                losses, emb, targets = objective(batch, encoder)
                obj_loss = sum(losses.values())
                probe_loss = F.cross_entropy(probe(emb), targets)
                loss = obj_loss + probe_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1

            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()
            epoch_probe_loss += probe_loss.item()

            if step % cfg.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                loss_strs = " | ".join(f"{k} {v.item():.4f}" for k, v in losses.items())
                logger.info(
                    "step %d | %s | probe %.4f | lr %.2e",
                    step,
                    loss_strs,
                    probe_loss.item(),
                    lr,
                )
                log_dict = {"step": step, "lr": lr, "train/probe": probe_loss.item()}
                for k, v in losses.items():
                    log_dict[f"train/{k}"] = v.item()
                wandb.log(log_dict)

        # Log epoch summary
        n_batches = len(train_loader)
        avg_strs = " | ".join(
            f"avg_{k} {v / n_batches:.4f}" for k, v in epoch_losses.items()
        )
        logger.info(
            "epoch %d | %s | avg_probe %.4f",
            epoch,
            avg_strs,
            epoch_probe_loss / n_batches,
        )

        # Per-epoch evaluation
        encoder.eval()
        probe.eval()
        correct = 0
        total = 0
        with torch.inference_mode():
            for batch in test_loader:
                data = batch["data"].to(cfg.device, non_blocking=True)
                targets = batch["target"].to(cfg.device, non_blocking=True)
                with autocast(cfg.device, dtype=torch.bfloat16):
                    emb, _ = encoder(data)
                    logits = probe(emb)
                    preds = logits.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        acc = correct / total
        logger.info("epoch %d | test_acc %.4f", epoch, acc)
        wandb.log({"test/epoch": epoch, "test/acc": acc})

        # Save checkpoint
        if (epoch + 1) % cfg.save_every == 0:
            ckpt_path = cfg.ckpt_to / f"epoch_{epoch:04d}.pt"
            save_checkpoint(
                ckpt_path, epoch, encoder, objective, optimizer, scheduler, scaler
            )

    # Save final checkpoint
    save_checkpoint(
        cfg.ckpt_to / "final.pt",
        cfg.epochs - 1,
        encoder,
        objective,
        optimizer,
        scheduler,
        scaler,
    )
    wandb.finish()
