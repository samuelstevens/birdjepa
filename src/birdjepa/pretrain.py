"""
BirdJEPA pretraining.

Based on https://github.com/rbalestr-lab/lejepa/blob/main/MINIMAL.md
"""

import dataclasses
import json
import logging
import pathlib
import typing as tp

import beartype
import equinox as eqx
import jax
import jax.random as jr
import jax.numpy as jnp
import optax
import wandb

import birdjepa.data
import birdjepa.helpers
import birdjepa.nn.objectives
import birdjepa.nn.transformer

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logger = logging.getLogger("birdjepa.pretrain")


@beartype.beartype
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
    seed: int = 42
    """Random seed."""
    device: tp.Literal["cpu", "gpu"] = "gpu"
    """Device to train on."""
    probe_pooling: str = "cls"
    """Pooling for probe: 'cls' (mean of CLS tokens) or 'patches' (mean of patches)."""
    source_rank: int = 0
    """Low-rank bottleneck for source prediction head. 0 = disabled."""
    source_weight: float = 1.0
    """Loss weight for source prediction."""
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


def save_checkpoint(path, epoch, encoder, objective, probe, opt_state):
    """Save training checkpoint using Equinox serialization."""
    metadata = {"epoch": epoch}
    models = {"encoder": encoder, "objective": objective, "probe": probe}
    with open(path, "wb") as f:
        f.write((json.dumps(metadata) + "\n").encode())
        eqx.tree_serialise_leaves(f, models)
        eqx.tree_serialise_leaves(f, opt_state)
    logger.info("Saved checkpoint to %s", path)


def load_checkpoint(path, encoder, objective, probe, opt_state):
    """Load training checkpoint. Returns (encoder, objective, probe, opt_state, start_epoch)."""
    with open(path, "rb") as f:
        metadata = json.loads(f.readline().decode())
        models = {"encoder": encoder, "objective": objective, "probe": probe}
        models = eqx.tree_deserialise_leaves(f, models)
        opt_state = eqx.tree_deserialise_leaves(f, opt_state)
    logger.info("Loaded checkpoint from %s (epoch %d)", path, metadata["epoch"])
    return (
        models["encoder"],
        models["objective"],
        models["probe"],
        opt_state,
        metadata["epoch"] + 1,
    )


def softmax_cross_entropy(logits, labels):
    """Cross-entropy loss for classification."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    return -jnp.sum(one_hot * log_probs, axis=-1).mean()


@eqx.filter_jit
def train_step(models, optim, opt_state, batch):
    """Single training step.

    Args:
        models: Dict with 'encoder', 'objective', 'probe'.
        optim: Optax optimizer.
        opt_state: Optimizer state.
        batch: Training batch dict.

    Returns:
        (updated_models, new_opt_state, metrics_dict)
    """

    def loss_fn(models):
        encoder, objective, probe = (
            models["encoder"],
            models["objective"],
            models["probe"],
        )
        losses, emb, targets = objective(batch, encoder)
        obj_loss = sum(losses.values())
        probe_logits = probe(emb)
        probe_loss = softmax_cross_entropy(probe_logits, targets)
        total_loss = obj_loss + probe_loss
        return total_loss, {
            "losses": losses,
            "probe_loss": probe_loss,
            "emb": emb,
            "targets": targets,
        }

    (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(models)

    # Get trainable params
    params = {k: eqx.filter(v, eqx.is_inexact_array) for k, v in models.items()}
    grad_params = {k: eqx.filter(v, eqx.is_inexact_array) for k, v in grads.items()}

    updates, new_opt_state = optim.update(grad_params, opt_state, params)

    # Apply updates to models
    new_models = {}
    for k in models:
        new_models[k] = eqx.apply_updates(models[k], updates[k])

    metrics = {"loss": loss, "probe_loss": aux["probe_loss"], **aux["losses"]}
    return new_models, new_opt_state, metrics


@beartype.beartype
def worker_fn(cfg: Config):
    """Main training function."""
    logging.basicConfig(level=logging.INFO, format=log_format)

    if cfg.device == "gpu":
        assert birdjepa.helpers.jax_has_gpu(), "GPU not available"

    wandb.init(project="birdjepa", config=dataclasses.asdict(cfg))
    key = jr.key(cfg.seed)

    # Data
    train_ds = make_dataset(cfg.train_data)
    test_ds = make_dataset(cfg.test_data)
    n_classes = train_ds.n_classes

    # Model and objective
    model_key, key = jr.split(key)
    encoder = birdjepa.nn.transformer.Transformer(cfg.model, key=model_key)
    obj_key, key = jr.split(key)
    objective = birdjepa.nn.objectives.make_objective(
        cfg.objective, cfg.model, train_ds, key=obj_key
    )

    # Wrap dataset for objective (e.g., multi-view for LeJEPA)
    train_ds = objective.wrap_dataset(train_ds)
    train_loader = birdjepa.data.make_dataloader(
        train_ds,
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        n_workers=cfg.n_workers,
        shuffle=True,
        drop_last=True,
    )
    test_loader = birdjepa.data.make_dataloader(
        test_ds,
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        n_workers=cfg.n_workers,
        shuffle=False,
        drop_last=False,
    )

    # Online linear probe
    key, probe_key = jax.random.split(key)
    probe = eqx.nn.Sequential([
        eqx.nn.LayerNorm(cfg.model.embed_dim),
        eqx.nn.Linear(cfg.model.embed_dim, n_classes, key=probe_key),
    ])

    # Optimizer and scheduler
    # TODO: Add separate param groups for probe (lr=1e-3, weight_decay=1e-7)
    steps_per_epoch = len(train_ds) // cfg.batch_size
    warmup_steps = steps_per_epoch
    total_steps = steps_per_epoch * cfg.epochs
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=cfg.lr * 0.01,
        peak_value=cfg.lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=1e-6,
    )
    optim = optax.adamw(learning_rate=schedule, weight_decay=cfg.weight_decay)

    # Combined params for optimizer
    params = {
        "encoder": eqx.filter(encoder, eqx.is_inexact_array),
        "objective": eqx.filter(objective, eqx.is_inexact_array),
        "probe": eqx.filter(probe, eqx.is_inexact_array),
    }
    opt_state = optim.init(params)

    # Resume from checkpoint
    start_epoch = 0
    if cfg.resume_from:
        raise NotImplementedError("Resuming from checkpoint not yet supported")

    # Ensure checkpoint directory exists
    cfg.ckpt_to.mkdir(parents=True, exist_ok=True)

    # Training loop
    step = 0
    models = {"encoder": encoder, "objective": objective, "probe": probe}
    n_batches = 0
    epochs = birdjepa.helpers.progress(
        range(start_epoch, cfg.epochs), every=1, desc="pretrain"
    )
    for epoch in epochs:
        epoch_losses: dict[str, float] = {}
        epoch_probe_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            # Convert batch arrays to JAX
            batch = {
                k: jnp.asarray(v) if hasattr(v, "__array__") else v
                for k, v in batch.items()
            }

            models, opt_state, metrics = train_step(models, optim, opt_state, batch)
            step += 1
            n_batches += 1

            # Accumulate epoch losses
            for k, v in metrics.items():
                if k not in ("loss", "probe_loss"):
                    epoch_losses[k] = epoch_losses.get(k, 0.0) + float(v)
            epoch_probe_loss += float(metrics["probe_loss"])

            if step % cfg.log_every == 0:
                lr = float(schedule(step))
                loss_strs = " | ".join(
                    f"{k} {float(v):.4f}"
                    for k, v in metrics.items()
                    if k not in ("loss", "probe_loss")
                )
                logger.info(
                    "step %d | %s | probe %.4f | lr %.2e",
                    step,
                    loss_strs,
                    float(metrics["probe_loss"]),
                    lr,
                )
                log_dict = {
                    "step": step,
                    "lr": lr,
                    "train/probe": float(metrics["probe_loss"]),
                }
                for k, v in metrics.items():
                    if k not in ("loss", "probe_loss"):
                        log_dict[f"train/{k}"] = float(v)
                wandb.log(log_dict)

        # Log epoch summary
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
        encoder = models["encoder"]
        probe = models["probe"]
        correct = 0
        total = 0

        for batch in test_loader:
            data = jnp.asarray(batch["data"])
            targets = jnp.asarray(batch["target"])
            x_bnk, grid = birdjepa.nn.transformer.patchify(data, cfg.model)
            out = encoder(x_bnk, grid=grid)
            if cfg.probe_pooling == "cls":
                emb = out["cls"].mean(axis=1)
            else:
                emb = out["patches"].mean(axis=1)
            logits = probe(emb)
            preds = jnp.argmax(logits, axis=1)
            correct += int(jnp.sum(preds == targets))
            total += targets.shape[0]
        acc = correct / total
        logger.info("epoch %d | test_acc %.4f", epoch, acc)
        wandb.log({"test/epoch": epoch, "test/acc": acc})

        # Save checkpoint
        if (epoch + 1) % cfg.save_every == 0:
            ckpt_path = cfg.ckpt_to / f"epoch_{epoch:04d}.eqx"
            encoder, objective, probe = (
                models["encoder"],
                models["objective"],
                models["probe"],
            )
            save_checkpoint(ckpt_path, epoch, encoder, objective, probe, opt_state)

    # Save final checkpoint
    encoder, objective, probe = models["encoder"], models["objective"], models["probe"]
    save_checkpoint(
        cfg.ckpt_to / "final.eqx", cfg.epochs - 1, encoder, objective, probe, opt_state
    )
    wandb.finish()
