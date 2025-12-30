"""
BirdJEPA pretraining.

Based on https://github.com/rbalestr-lab/lejepa/blob/main/MINIMAL.md
"""

import dataclasses
import json
import logging
import os
import pathlib
import time
import typing as tp

import beartype
import equinox as eqx
import jax
import jax.random as jr
import jax.numpy as jnp
import numpy as np
import optax
import wandb
import wandb.util as wandb_util

import birdjepa.data
import birdjepa.helpers
import birdjepa.nn.objectives
import birdjepa.nn.transformer

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logger = logging.getLogger("birdjepa.pretrain")


def _is_numeric_array(x: object) -> bool:
    """Check if x is a numeric array that can be converted to JAX."""
    if not isinstance(x, (jax.Array, np.ndarray)):
        return False
    dt = getattr(x, "dtype", None)
    if dt is None:
        return False
    return (
        np.issubdtype(dt, np.bool_)
        or np.issubdtype(dt, np.integer)
        or np.issubdtype(dt, np.floating)
    )


def _batch_to_jax(batch: dict) -> dict:
    """Convert numeric arrays in batch to JAX, drop non-numeric fields."""
    return {k: jnp.asarray(v) for k, v in batch.items() if _is_numeric_array(v)}


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
    model: birdjepa.nn.transformer.Config = birdjepa.nn.transformer.Transformer()
    """Encoder configuration (Transformer or Debug for minimal)."""
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
    grad_clip: float = 1.0
    """Gradient clipping norm. 0 = disabled."""
    # Schedule
    schedule: tp.Literal["cosine", "wsd"] = "cosine"
    """LR schedule: 'cosine' (warmup + cosine decay) or 'wsd' (warmup-stable-decay)."""
    warmup_steps: int = 0
    """Number of warmup steps for learning rate schedules."""
    decay_steps: int = 0
    """Number of decay steps for WSD."""
    # Checkpointing
    ckpt_to: pathlib.Path = pathlib.Path("./checkpoints")
    """Directory for checkpoints."""
    save_every: int = 10
    """Save checkpoint every N epochs."""
    resume_from: str = ""
    """Path to checkpoint to resume from."""
    # Multi-GPU
    n_gpus: int = 1
    """Number of GPUs for data parallel training."""
    # Slurm
    slurm_acct: str = ""
    """Slurm account string. Empty means run locally."""
    slurm_partition: str = ""
    """Slurm partition."""
    n_hours: float = 24.0
    """Slurm job length in hours."""
    log_to: pathlib.Path = pathlib.Path("./logs")
    """Where to log Slurm job stdout/stderr."""
    sweep: str = ""
    """Path to sweep file with make_cfgs() function."""
    run_id: str = ""
    """Optional W&B run id for deterministic runs and checkpoint paths."""


@beartype.beartype
def make_dataset(cfg: birdjepa.data.Config) -> birdjepa.data.Dataset:
    """Create dataset from config."""
    if isinstance(cfg, birdjepa.data.XenoCanto):
        return birdjepa.data.XenoCantoDataset(cfg)
    elif isinstance(cfg, birdjepa.data.Cifar100):
        return birdjepa.data.Cifar100Dataset(cfg)
    else:
        tp.assert_never(cfg)


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


def wsd_schedule(
    peak_value: float,
    total_steps: int,
    warmup_steps: int = 0,
    decay_steps: int = 0,
    end_value: float = 0.0,
) -> optax.Schedule:
    """Warmup-Stable-Decay (WSD) learning rate schedule.

    Args:
        peak_value: Peak learning rate after warmup.
        total_steps: Total number of training steps.
        warmup_steps: Absolute warmup steps.
        decay_steps: Absolute decay steps.
        end_value: Final learning rate after decay.

    Returns:
        Optax schedule function.
    """
    assert warmup_steps >= 0, f"{warmup_steps=} must be >= 0"
    assert decay_steps >= 0, f"{decay_steps=} must be >= 0"
    stable_steps = total_steps - warmup_steps - decay_steps

    assert stable_steps >= 0, (
        f"Negative stable steps: {warmup_steps=} + {decay_steps=} > {total_steps=}"
    )

    schedules = [
        optax.constant_schedule(peak_value),
    ]
    boundaries = []
    if warmup_steps > 0:
        schedules.insert(0, optax.linear_schedule(0.0, peak_value, warmup_steps))
        boundaries.append(warmup_steps)
    if decay_steps > 0:
        schedules.append(optax.linear_schedule(peak_value, end_value, decay_steps))
        boundaries.append(warmup_steps + stable_steps)

    if not boundaries:
        return schedules[0]
    return optax.join_schedules(schedules, boundaries)


def make_train_step(optim, data_sharding, model_sharding):
    """Create train_step function with sharding.

    Args:
        optim: Optax optimizer.
        data_sharding: NamedSharding for batch data (shard along batch dim).
        model_sharding: NamedSharding for model params (replicated).

    Returns:
        train_step function.
    """

    @eqx.filter_jit
    def train_step(models, opt_state, batch, *, key):
        """Single training step with SPMD data parallelism.

        Args:
            models: Dict with 'encoder', 'objective', 'probe'.
            opt_state: Optimizer state.
            batch: Training batch dict.
            key: PRNG key for stochastic operations.

        Returns:
            (updated_models, new_opt_state, metrics_dict)
        """
        # Sharding is handled by input sharding before calling train_step

        def loss_fn(models):
            encoder, objective, probe = (
                models["encoder"],
                models["objective"],
                models["probe"],
            )
            losses, emb, targets = objective(batch, encoder, key=key)
            obj_loss = sum(losses.values())
            probe_logits = jax.vmap(probe)(emb)
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

        # Compute gradient norm before clipping
        grad_leaves = jax.tree.leaves(grad_params)
        grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in grad_leaves))

        # Compute per-component gradient norms for debugging
        enc_grads = jax.tree.leaves(grad_params["encoder"])
        obj_grads = jax.tree.leaves(grad_params["objective"])
        probe_grads = jax.tree.leaves(grad_params["probe"])
        enc_grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in enc_grads))
        obj_grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in obj_grads))
        probe_grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in probe_grads))

        updates, new_opt_state = optim.update(grad_params, opt_state, params)

        # Apply updates to models
        new_models = {}
        for k in models:
            new_models[k] = eqx.apply_updates(models[k], updates[k])

        # Compute update norms to verify updates are being applied
        update_leaves = jax.tree.leaves(updates)
        update_norm = jnp.sqrt(sum(jnp.sum(u**2) for u in update_leaves))

        # Compute parameter norms
        param_leaves = jax.tree.leaves(params)
        param_norm = jnp.sqrt(sum(jnp.sum(p**2) for p in param_leaves))

        metrics = {
            "loss": loss,
            "probe_loss": aux["probe_loss"],
            "grad_norm": grad_norm,
            "enc_grad": enc_grad_norm,
            "obj_grad": obj_grad_norm,
            "probe_grad": probe_grad_norm,
            "update_norm": update_norm,
            "param_norm": param_norm,
            **aux["losses"],
        }
        return new_models, new_opt_state, metrics

    return train_step


@beartype.beartype
def worker_fn(cfg: Config):
    """Main training function with optional data parallel support."""
    logging.basicConfig(level=logging.INFO, format=log_format)

    # Initialize multi-process JAX distributed runtime for multi-GPU SLURM jobs
    # Each SLURM task is one process; this coordinates across them
    # Skip for single-GPU to avoid heartbeat timeout issues during long JIT
    n_tasks = int(os.environ.get("SLURM_NTASKS", "1"))
    if n_tasks > 1:
        jax.distributed.initialize()
        logger.info("Initialized JAX distributed: %d processes", jax.process_count())
    else:
        logger.info("Single-GPU mode, skipping JAX distributed initialization")

    if cfg.device == "gpu":
        assert birdjepa.helpers.jax_has_gpu(), "GPU not available"

    # Set up sharding (works for both single and multi-GPU)
    n_devices = jax.device_count()
    mesh, data_sharding, model_sharding = birdjepa.helpers.setup_sharding(n_devices)
    logger.info("Using %d device(s) for training", len(mesh.devices))
    logger.info("Devices: %s", [str(d) for d in mesh.devices])

    # Only process 0 logs to wandb in multi-process mode
    is_main = jax.process_index() == 0
    if is_main:
        assert cfg.run_id
        dct = dataclasses.asdict(cfg)
        dct["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
        wandb.init(project="birdjepa", config=dct, id=cfg.run_id, resume="allow")
        logger.info("wandb run id: %s", cfg.run_id)

    key = jr.key(cfg.seed)

    # Data
    train_ds = make_dataset(cfg.train_data)
    test_ds = make_dataset(cfg.test_data)
    n_classes = train_ds.n_classes

    # Model and objective
    model_key, key = jr.split(key)
    if isinstance(cfg.model, birdjepa.nn.transformer.Debug):
        encoder = birdjepa.nn.transformer.DebugModel(cfg.model, key=model_key)
        logger.info("Using DebugEncoder (linear only, no transformer)")
    elif isinstance(cfg.model, birdjepa.nn.transformer.Transformer):
        encoder = birdjepa.nn.transformer.TransformerModel(cfg.model, key=model_key)
    else:
        tp.assert_never(cfg.model)
    obj_key, key = jr.split(key)
    objective = birdjepa.nn.objectives.make_objective(
        cfg.objective, cfg.model, train_ds, key=obj_key
    )

    # Wrap dataset for objective (e.g., multi-view for LeJEPA)
    train_ds = objective.wrap_dataset(train_ds)
    # Different seed per process so each loads different data (2x throughput with 2 GPUs)
    train_seed = cfg.seed + jax.process_index()
    train_loader = birdjepa.data.make_dataloader(
        train_ds,
        seed=train_seed,
        batch_size=cfg.batch_size,
        n_workers=cfg.n_workers,
        shuffle=True,
        drop_last=True,
    )
    # NOTE: LeJEPA wraps the dataset to return "views" instead of "data", which will break test accuracy below.
    test_ds = objective.wrap_dataset(test_ds)
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
    steps_per_epoch = len(train_ds) // cfg.batch_size
    total_steps = steps_per_epoch * cfg.epochs
    if cfg.schedule == "wsd":
        schedule = wsd_schedule(
            peak_value=cfg.lr,
            total_steps=total_steps,
            warmup_steps=cfg.warmup_steps,
            decay_steps=cfg.decay_steps,
            end_value=0.0,
        )
    elif cfg.schedule == "cosine":
        warmup_steps = cfg.warmup_steps
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=cfg.lr,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
            end_value=0.0,
        )
    else:
        tp.assert_never(cfg.model)
    optim = optax.adamw(learning_rate=schedule, weight_decay=cfg.weight_decay)
    if cfg.grad_clip > 0:
        optim = optax.chain(
            optax.clip_by_global_norm(cfg.grad_clip),
            optim,
        )

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

    ckpt_dpath = cfg.ckpt_to / cfg.run_id

    # Ensure checkpoint directory exists
    if is_main:
        ckpt_dpath.mkdir(parents=True, exist_ok=True)
        logger.info("Checkpoint dir: %s", ckpt_dpath)

    # Create train_step with sharding
    train_step = make_train_step(optim, data_sharding, model_sharding)

    # Training loop
    step = 0
    models = {"encoder": encoder, "objective": objective, "probe": probe}

    # Initial sharding of models and optimizer state
    models = eqx.filter_shard(models, model_sharding)
    opt_state = eqx.filter_shard(opt_state, model_sharding)

    n_batches = 0
    epochs = range(start_epoch, cfg.epochs)
    for epoch in birdjepa.helpers.progress(epochs, every=1, desc="pretrain"):
        epoch_losses: dict[str, float] = {}
        epoch_probe_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = _batch_to_jax(batch)
            batch = eqx.filter_shard(batch, data_sharding)
            step_key, key = jr.split(key)

            models, opt_state, metrics = train_step(
                models, opt_state, batch, key=step_key
            )

            step += 1
            n_batches += 1

            # Log peak memory usage after first step (includes JIT compilation overhead)
            # Use local_devices() for multi-GPU - memory_stats() only works on addressable devices
            if step == 1:
                for device in jax.local_devices():
                    stats = device.memory_stats()
                    if stats:
                        peak_gb = stats.get("peak_bytes_in_use", 0) / 1e9
                        logger.info("Device %s peak memory: %.2f GB", device, peak_gb)

            # Average metrics across devices for logging
            metrics = jax.tree.map(lambda x: float(jnp.mean(x)), metrics)

            # Accumulate epoch losses
            for k, v in metrics.items():
                if k not in ("loss", "probe_loss"):
                    epoch_losses[k] = epoch_losses.get(k, 0.0) + float(v)
            epoch_probe_loss += float(metrics["probe_loss"])

            if step % cfg.log_every == 0:
                lr = float(schedule(step))
                # Log all metrics as k=v pairs
                metric_strs = " ".join(
                    f"{k}={float(v):.4g}" for k, v in sorted(metrics.items())
                )
                logger.info("step=%d lr=%.2e %s", step, lr, metric_strs)
                if is_main:
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
        objective = models["objective"]
        correct = 0
        total = 0
        test_loss = 0.0
        n_test_batches = 0

        eval_key = jr.key(0)  # Fixed key for deterministic eval
        for batch in test_loader:
            data = jnp.asarray(batch["data"])
            targets = jnp.asarray(batch["target"])
            x_bnk, grid = birdjepa.nn.transformer.patchify(data, cfg.model)
            out = encoder(x_bnk, grid=grid, key=eval_key)
            if cfg.probe_pooling == "cls":
                emb = out["cls"].mean(axis=1)
            else:
                emb = out["patches"].mean(axis=1)
            logits = jax.vmap(probe)(emb)
            preds = jnp.argmax(logits, axis=1)
            correct += int(jnp.sum(preds == targets))
            total += targets.shape[0]
            eval_key, obj_key = jr.split(eval_key)
            batch_jax = _batch_to_jax(batch)
            losses, _, _ = objective(batch_jax, encoder, key=obj_key)
            loss = sum(losses.values())
            test_loss += float(jnp.mean(loss))
            n_test_batches += 1
        acc = correct / total
        logger.info("epoch %d | test_acc %.4f", epoch, acc)
        test_loss /= max(1, n_test_batches)
        logger.info("epoch %d | test_loss %.4f", epoch, test_loss)
        if is_main:
            log_dict = {
                "test/epoch": epoch,
                "test/acc": acc,
                "test/loss": test_loss,
            }
            wandb.log(log_dict)

        # Save checkpoint (only main process)
        if is_main and (epoch + 1) % cfg.save_every == 0:
            ckpt_path = ckpt_dpath / f"epoch_{epoch:04d}.eqx"
            encoder, objective, probe = (
                models["encoder"],
                models["objective"],
                models["probe"],
            )
            save_checkpoint(ckpt_path, epoch, encoder, objective, probe, opt_state)

    # Save final checkpoint (only main process)
    if is_main:
        encoder, objective, probe = (
            models["encoder"],
            models["objective"],
            models["probe"],
        )
        save_checkpoint(
            ckpt_dpath / "final.eqx",
            cfg.epochs - 1,
            encoder,
            objective,
            probe,
            opt_state,
        )
    if is_main:
        wandb.finish()


@beartype.beartype
def cli(cfg: Config):
    """CLI entrypoint: run locally or submit to Slurm."""
    logging.basicConfig(level=logging.INFO, format=log_format)

    # Load sweep configs if specified
    if cfg.sweep:
        import birdjepa.configs

        sweep_dcts = birdjepa.configs.load_sweep(pathlib.Path(cfg.sweep))
        override = dataclasses.replace(cfg, sweep="")
        cfgs, errs = birdjepa.configs.load_cfgs(
            override, default=Config(), sweep_dcts=sweep_dcts
        )
        for err in errs:
            logger.error("Config error: %s", err)
        logger.info("Loaded %d configs from %s", len(cfgs), cfg.sweep)
    else:
        cfgs = [cfg]

    cfgs_with_ids: list[Config] = []
    for c in cfgs:
        if not c.run_id:
            run_id = wandb_util.generate_id()
            c = dataclasses.replace(c, run_id=run_id)
            logger.info("Assigned wandb run id: %s", run_id)
        cfgs_with_ids.append(c)
    cfgs = cfgs_with_ids

    if not any(c.slurm_acct for c in cfgs):
        for c in cfgs:
            worker_fn(c)
        return

    assert all(c.slurm_acct for c in cfgs), "Cannot mix slurm and local configs"
    base = cfgs[0]
    for c in cfgs[1:]:
        assert c.slurm_acct == base.slurm_acct, "slurm_acct must match all cfgs"
        assert c.slurm_partition == base.slurm_partition, (
            "slurm_partition must match all cfgs"
        )
        assert c.n_hours == base.n_hours, "n_hours must match all cfgs"
        assert c.n_gpus == base.n_gpus, "n_gpus must match all cfgs"
        assert c.n_workers == base.n_workers, "n_workers must match all cfgs"
        assert c.log_to == base.log_to, "log_to must match all cfgs"

    import submitit
    import submitit.core.utils

    executor = submitit.SlurmExecutor(folder=base.log_to)
    executor.update_parameters(
        time=int(base.n_hours * 60),
        partition=base.slurm_partition,
        gpus_per_node=base.n_gpus,
        ntasks_per_node=base.n_gpus,  # One process per GPU (stable multi-GPU mode)
        cpus_per_task=base.n_workers,
        stderr_to_stdout=True,
        account=base.slurm_acct,
        setup=[
            "module load ffmpeg/6.1.1",
            "export NCCL_IB_DISABLE=1",  # Disable InfiniBand (not available)
            "export NCCL_P2P_DISABLE=1",  # Use shared memory
        ],
    )

    with executor.batch():
        jobs = [executor.submit(worker_fn, c) for c in cfgs]

    time.sleep(5.0)
    for j, job in enumerate(jobs):
        logger.info("Job %d/%d: %s %s", j + 1, len(jobs), job.job_id, job.state)

    for j, job in enumerate(jobs):
        try:
            job.result()
            logger.info("Job %d/%d finished.", j + 1, len(jobs))
        except submitit.core.utils.UncompletedJobError:
            logger.warning("Job %s (%d) did not finish.", job.job_id, j)
