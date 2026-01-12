"""
BirdJEPA pretraining.

Based on https://github.com/rbalestr-lab/lejepa/blob/main/MINIMAL.md
"""

import dataclasses
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
import wandb.util

import birdjepa.checkpoints
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


def compute_tgt_entropy(targets: np.ndarray) -> float:
    """Compute entropy of target labels in a batch (measures shuffle quality).

    High entropy = diverse classes in batch = good shuffle.
    Low entropy = clustered classes = poor shuffle (reading from few shards).
    """
    _, counts = np.unique(targets, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log(probs + 1e-10)))


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
    n_steps: int = 10000
    """Total number of training steps."""
    log_every: int = 1
    """Log training metrics every N steps."""
    eval_every: int = 1000
    """Evaluate on test set every N steps."""
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
    optimizer: tp.Literal["adamw", "muon"] = "adamw"
    """Optimizer: 'adamw' or 'muon' (applies muon to 2D params, adamw to others)."""
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
    mem_gb: int = 0
    """Node memory in GB. If set, requests enough CPUs to get this much RAM (~10GB/CPU on OSC)."""
    log_to: pathlib.Path = pathlib.Path("./logs")
    """Where to log Slurm job stdout/stderr."""
    sweep: str = ""
    """Path to sweep file with make_cfgs() function."""
    run_id: str = ""
    """Optional W&B run id for deterministic runs and checkpoint paths."""
    # Rust dataloader settings
    window_size: int = 10000
    """Shuffle buffer size for Rust loader (samples)."""


@beartype.beartype
def make_dataset(
    cfg: birdjepa.data.Config,
) -> birdjepa.data.Dataset | birdjepa.data.ShuffledXenoCantoDataset:
    """Create dataset from config."""
    if isinstance(cfg, birdjepa.data.XenoCanto):
        return birdjepa.data.ShuffledXenoCantoDataset(cfg)
    elif isinstance(cfg, birdjepa.data.Cifar100):
        return birdjepa.data.Cifar100Dataset(cfg)
    else:
        tp.assert_never(cfg)


def softmax_cross_entropy(logits, labels):
    """Cross-entropy loss for classification."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    return -jnp.sum(one_hot * log_probs, axis=-1).mean()


@beartype.beartype
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
    # Use force=True to override any handlers set by submitit
    logging.basicConfig(level=logging.INFO, format=log_format, force=True)

    # Initialize multi-process JAX distributed runtime for multi-GPU SLURM jobs
    # Each SLURM task is one process; this coordinates across them
    # Skip for single-GPU to avoid heartbeat timeout issues during long JIT
    import multiprocessing

    n_tasks = int(os.environ.get("SLURM_NTASKS", "1"))
    if n_tasks > 1 and multiprocessing.parent_process() is None:
        # Use submitit to get distributed environment, then pass explicitly to JAX
        # This avoids relying on JAX's Slurm auto-detection
        import socket
        import time

        import submitit.helpers

        # Debug logging for heartbeat failures
        hostname = socket.gethostname()
        slurm_job_id = os.environ.get("SLURM_JOB_ID", "unknown")
        slurm_nodelist = os.environ.get("SLURM_NODELIST", "unknown")
        slurm_procid = os.environ.get("SLURM_PROCID", "unknown")
        logger.info(
            "Node debug: hostname=%s, job=%s, nodelist=%s, procid=%s",
            hostname,
            slurm_job_id,
            slurm_nodelist,
            slurm_procid,
        )

        dist_env = submitit.helpers.TorchDistributedEnvironment()
        coordinator_address = f"{dist_env.master_addr}:{dist_env.master_port}"
        logger.info(
            "Distributed env: coordinator=%s, rank=%d/%d, local_rank=%d/%d",
            coordinator_address,
            dist_env.rank,
            dist_env.world_size,
            dist_env.local_rank,
            dist_env.local_world_size,
        )

        # Try to resolve coordinator address to check reachability
        try:
            coord_ip = socket.gethostbyname(dist_env.master_addr)
            logger.info("Coordinator resolves to IP: %s", coord_ip)
        except socket.gaierror as e:
            logger.warning(
                "Failed to resolve coordinator %s: %s", dist_env.master_addr, e
            )

        init_start = time.perf_counter()
        jax.distributed.initialize(
            coordinator_address=coordinator_address,
            num_processes=dist_env.world_size,
            process_id=dist_env.rank,
            initialization_timeout=300,
            heartbeat_timeout_seconds=120,  # Short for fast failure detection
        )
        init_elapsed = time.perf_counter() - init_start
        logger.info(
            "Initialized JAX distributed: %d processes (took %.1fs)",
            jax.process_count(),
            init_elapsed,
        )
    elif n_tasks == 1:
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
    logger.info("Loading train dataset")
    train_ds = make_dataset(cfg.train_data)
    logger.info("Loading test dataset")
    test_ds = make_dataset(cfg.test_data)
    n_classes = train_ds.n_classes
    logger.info(
        "Datasets loaded: %d train, %d test, %d classes",
        len(train_ds),
        len(test_ds),
        n_classes,
    )

    # Model and objective
    logger.info("Creating encoder")
    model_key, key = jr.split(key)
    if isinstance(cfg.model, birdjepa.nn.transformer.Debug):
        encoder = birdjepa.nn.transformer.DebugModel(cfg.model, key=model_key)
        logger.info("Using DebugEncoder (linear only, no transformer)")
    elif isinstance(cfg.model, birdjepa.nn.transformer.Transformer):
        encoder = birdjepa.nn.transformer.TransformerModel(cfg.model, key=model_key)
        logger.info("Created TransformerModel")
    else:
        tp.assert_never(cfg.model)
    logger.info("Creating objective")
    obj_key, key = jr.split(key)
    objective = birdjepa.nn.objectives.make_objective(
        cfg.objective, cfg.model, train_ds, key=obj_key
    )
    logger.info("Created objective: %s", type(objective).__name__)

    # Create dataloaders using Rust loader
    # Shard data across processes so each GPU loads different samples
    logger.info("Creating train dataloader")
    assert isinstance(cfg.train_data, birdjepa.data.XenoCanto)
    train_loader = birdjepa.data.RustXenoCantoLoader(
        cfg.train_data,
        seed=cfg.seed + jax.process_index(),  # Unique seed per process
        batch_size=cfg.batch_size,
        n_workers=cfg.n_workers,
        shuffle_buffer_size=cfg.window_size,
        shuffle_min_size=cfg.window_size // 2,
        infinite=True,
    )
    # Test loader with infinite=True so we can reuse across evals
    assert isinstance(cfg.test_data, birdjepa.data.XenoCanto)
    test_loader = birdjepa.data.RustXenoCantoLoader(
        cfg.test_data,
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        n_workers=cfg.n_workers,
        shuffle_buffer_size=1000,
        shuffle_min_size=0,
        infinite=True,  # Keeps cycling - we break manually
    )
    n_eval_batches = len(test_ds) // cfg.batch_size
    logger.info("Dataloaders created (eval: %d batches)", n_eval_batches)

    # Online linear probe
    logger.info("Creating linear probe")
    key, probe_key = jax.random.split(key)
    probe = eqx.nn.Sequential([
        eqx.nn.LayerNorm(cfg.model.embed_dim),
        eqx.nn.Linear(cfg.model.embed_dim, n_classes, key=probe_key),
    ])

    # Optimizer and scheduler
    logger.info("Setting up optimizer")
    logger.info("Training %d total steps", cfg.n_steps)
    if cfg.schedule == "wsd":
        schedule = wsd_schedule(
            peak_value=cfg.lr,
            total_steps=cfg.n_steps,
            warmup_steps=cfg.warmup_steps,
            decay_steps=cfg.decay_steps,
            end_value=0.0,
        )
    elif cfg.schedule == "cosine":
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=cfg.lr,
            warmup_steps=cfg.warmup_steps,
            decay_steps=cfg.n_steps,
            end_value=0.0,
        )
    else:
        tp.assert_never(cfg.model)
    if cfg.optimizer == "muon":
        optim = optax.contrib.muon(learning_rate=schedule)
    elif cfg.optimizer == "adamw":
        optim = optax.adamw(learning_rate=schedule, weight_decay=cfg.weight_decay)
    else:
        tp.assert_never(cfg.optimizer)
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
    logger.info("Optimizer initialized")

    ckpt_dpath = cfg.ckpt_to / cfg.run_id

    # Ensure checkpoint directory exists
    if is_main:
        ckpt_dpath.mkdir(parents=True, exist_ok=True)
        logger.info("Checkpoint dir: %s", ckpt_dpath)

    # Create checkpoint manager (handles retention, async saves)
    logger.info("Creating checkpoint manager")
    ckpt_mngr = birdjepa.checkpoints.CheckpointManager(ckpt_dpath)
    logger.info("Checkpoint manager created (save_interval=500, max_to_keep=5)")

    # Resume from checkpoint (auto-resume if checkpoint exists)
    start_step = 0
    restored = ckpt_mngr.load_training(
        encoder, objective, probe, opt_state, encoder_config=cfg.model
    )
    if restored is not None:
        encoder, objective, probe, opt_state, start_step = restored
        # Note: Rust loader state and PRNG key are not checkpointed
        logger.info("Resumed from checkpoint (dataloader restarts from beginning)")

    # Create train_step with sharding
    logger.info("Creating train_step function")
    train_step = make_train_step(optim, data_sharding, model_sharding)

    # Training loop (step-based with infinite iterator)
    logger.info("Starting training from step %d (total %d)", start_step, cfg.n_steps)
    step = start_step
    models = {"encoder": encoder, "objective": objective, "probe": probe}

    # Initial sharding of models and optimizer state
    models = eqx.filter_shard(models, model_sharding)
    opt_state = eqx.filter_shard(opt_state, model_sharding)

    # TODO: total is wrong when resuming (should be cfg.n_steps - step)
    last_step_time = time.time()
    for batch in birdjepa.helpers.progress(
        train_loader, every=cfg.log_every, total=cfg.n_steps
    ):
        if step >= cfg.n_steps:
            break

        # Compute target entropy before converting to JAX (measures shuffle quality)
        tgt_entropy = compute_tgt_entropy(batch["target"])

        batch = _batch_to_jax(batch)
        batch = eqx.filter_shard(batch, data_sharding)
        step_key, key = jr.split(key)

        models, opt_state, metrics = train_step(models, opt_state, batch, key=step_key)

        step += 1

        # Checkpoint model state (orbax decides whether to actually save based on interval)
        # NOTE: All processes must call save() - orbax handles primary host logic internally
        # NOTE: Don't save JAX PRNG key - it's host-local and can't be serialized in distributed mode
        ckpt_mngr.save(
            step=step,
            encoder=models["encoder"],
            objective=models["objective"],
            probe=models["probe"],
            opt_state=opt_state,
            encoder_config=cfg.model,
        )

        # Log peak memory usage after first step (includes JIT compilation overhead)
        if step == 1:
            for device in jax.local_devices():
                stats = device.memory_stats()
                if stats:
                    peak_gb = stats.get("peak_bytes_in_use", 0) / 1e9
                    logger.info("Device %s peak memory: %.2f GB", device, peak_gb)

        # Average metrics across devices for logging
        metrics = jax.tree.map(lambda x: float(jnp.mean(x)), metrics)

        if step % cfg.log_every == 0:
            now = time.time()
            sec_per_step = now - last_step_time
            last_step_time = now

            lr = float(schedule(step))
            metric_strs = " ".join(
                f"{k}={float(v):.4g}" for k, v in sorted(metrics.items())
            )
            logger.info(
                "step=%d lr=%.2e tgt_entropy=%.2f sec/step=%.1f %s",
                step,
                lr,
                tgt_entropy,
                sec_per_step,
                metric_strs,
            )
            if is_main:
                log_dict = {
                    "step": step,
                    "lr": lr,
                    "train/probe": float(metrics["probe_loss"]),
                    "train/tgt_entropy": tgt_entropy,
                    "train/sec_per_step": sec_per_step,
                }
                for k, v in metrics.items():
                    if k not in ("loss", "probe_loss"):
                        log_dict[f"train/{k}"] = float(v)
                wandb.log(log_dict)

        # Evaluate periodically
        if step % cfg.eval_every == 0:
            encoder = models["encoder"]
            probe = models["probe"]
            objective = models["objective"]
            correct = 0
            total = 0
            test_loss = 0.0

            eval_key = jr.key(0)  # Fixed key for deterministic eval
            n_test_batches = 0
            for test_batch in test_loader:
                if n_test_batches >= n_eval_batches:
                    break
                n_test_batches += 1
                data = jnp.asarray(test_batch["data"])
                targets = jnp.asarray(test_batch["target"])
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
                batch_jax = _batch_to_jax(test_batch)
                losses, _, _ = objective(batch_jax, encoder, key=obj_key)
                loss = sum(losses.values())
                test_loss += float(jnp.mean(loss))
            acc = correct / max(1, total)
            test_loss /= max(1, n_test_batches)
            logger.info("step=%d test_acc=%.4f test_loss=%.4f", step, acc, test_loss)
            if is_main:
                wandb.log({"step": step, "test/acc": acc, "test/loss": test_loss})

    # Save final checkpoint and wait for async saves to complete
    # NOTE: All processes must call save() - orbax handles primary host logic internally
    ckpt_mngr.save(
        step=step,
        encoder=models["encoder"],
        objective=models["objective"],
        probe=models["probe"],
        opt_state=opt_state,
        encoder_config=cfg.model,
        force=True,
    )
    ckpt_mngr.wait_until_finished()
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
            run_id = wandb.util.generate_id()
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
    # OSC allocates ~10GB RAM per CPU, so request enough CPUs for desired memory
    n_cpus = (
        max(base.mem_gb // 10, base.n_workers) if base.mem_gb > 0 else base.n_workers
    )
    if n_cpus > base.n_workers:
        logger.info("Requesting %d CPUs to get %dGB RAM", n_cpus, base.mem_gb)
    tag = pathlib.Path(base.sweep).stem if base.sweep else "local"
    executor.update_parameters(
        time=int(base.n_hours * 60),
        partition=base.slurm_partition,
        gpus_per_node=base.n_gpus,
        ntasks_per_node=base.n_gpus,  # One process per GPU (stable multi-GPU mode)
        cpus_per_task=n_cpus,
        stderr_to_stdout=True,
        account=base.slurm_acct,
        job_name=f"pretrain[{tag}]",
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
