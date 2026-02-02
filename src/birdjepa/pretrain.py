import dataclasses
import logging
import os
import pathlib
import time
import typing as tp

import beartype
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
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


def compute_tgt_entropy(targets: np.ndarray) -> float:
    """Compute entropy of target labels in a batch (measures shuffle quality).

    High entropy = diverse classes in batch = good shuffle.
    Low entropy = clustered classes = poor shuffle (reading from few shards).
    """
    _, counts = np.unique(targets, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log(probs + 1e-10)))


def _none_to_nan(value: float | None) -> float:
    return float("nan") if value is None else float(value)


def _batch_to_device(batch: dict, sharding) -> dict[str, jax.Array]:
    """Copy batch arrays and place on sharded device."""
    result: dict[str, jax.Array] = {}
    for key, value in batch.items():
        if not eqx.is_array(value):
            continue
        dt = getattr(value, "dtype", None)
        if dt is None:
            continue
        if not (
            np.issubdtype(dt, np.bool_)
            or np.issubdtype(dt, np.integer)
            or np.issubdtype(dt, np.floating)
        ):
            continue
        if isinstance(value, np.ndarray):
            value = np.array(value, copy=True)
        result[key] = jax.device_put(value, sharding)
    jax.tree.map(lambda x: x.block_until_ready(), result)
    return result


def _pool_embeddings(
    out: birdjepa.nn.transformer.EncoderOutput, probe_pooling: str
) -> jax.Array:
    """Pool embeddings from encoder outputs."""
    if probe_pooling == "cls":
        emb = out.cls.mean(axis=-2)
    elif probe_pooling == "patches":
        emb = out.patches.mean(axis=-2)
    else:
        tp.assert_never(probe_pooling)

    # LeJEPA keeps a view dimension (v, b, d); flatten to (v*b, d) for probe.
    if emb.ndim == 3:
        emb = emb.reshape(-1, emb.shape[-1])
    return emb


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
    prng_mode: tp.Literal["stateful", "stateless", "checkpointed"] = "stateful"
    """PRNG mode: stateful (default), stateless (fold_in step), or checkpointed (save/restore key)."""
    disable_stochastic: bool = False
    """If True, disable stochastic ops and force deterministic data settings."""
    device: tp.Literal["cpu", "gpu"] = "gpu"
    """Device to train on."""
    probe_pooling: tp.Literal["cls", "patches"] = "cls"
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
    # Multi-GPU
    n_gpus: int = 1
    """Number of GPUs for data parallel training (single process sees all GPUs)."""
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
    tags: list[str] = dataclasses.field(default_factory=list)
    """W&B tags for organizing runs."""
    # Rust dataloader settings
    window_size: int = 10000
    """Shuffle buffer size for Rust loader (samples)."""
    reset_loader_at_step: int | None = None
    """Debug: reset dataloader at this step to test if loss jumps (None = disabled)."""


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

    segments: list[tuple[int, optax.Schedule]] = []
    if warmup_steps > 0:
        segments.append((
            warmup_steps,
            optax.linear_schedule(0.0, peak_value, warmup_steps),
        ))
    if stable_steps > 0:
        segments.append((stable_steps, optax.constant_schedule(peak_value)))
    if decay_steps > 0:
        segments.append((
            decay_steps,
            optax.linear_schedule(peak_value, end_value, decay_steps),
        ))

    if not segments:
        return optax.constant_schedule(peak_value)
    if len(segments) == 1:
        return segments[0][1]

    schedules = [segment[1] for segment in segments]
    boundaries = []
    n_steps = 0
    for n_segment_steps, _ in segments[:-1]:
        n_steps += n_segment_steps
        boundaries.append(n_steps)

    return optax.join_schedules(schedules, boundaries)


@beartype.beartype
def make_train_step(
    optim,
    data_sharding,
    model_sharding,
    *,
    probe_pooling: str,
):
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

            losses, out, targets = objective(batch, encoder, key=key, mode="train")
            emb = _pool_embeddings(out, probe_pooling)
            targets_probe = targets.reshape(-1)

            obj_loss = sum(losses.values())
            probe_logits = jax.vmap(probe)(emb)

            probe_loss = softmax_cross_entropy(probe_logits, targets_probe)
            total_loss = obj_loss + probe_loss
            return total_loss, {
                "losses": losses,
                "probe_loss": probe_loss,
                "emb": emb,
                "targets": targets_probe,
            }

        (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(models)

        # Get trainable params
        params = {k: eqx.filter(v, eqx.is_inexact_array) for k, v in models.items()}
        grad_params = {k: eqx.filter(v, eqx.is_inexact_array) for k, v in grads.items()}

        # Compute gradient norm before clipping
        grad_norm = birdjepa.helpers.tree_l2_norm(grad_params)
        enc_grad_norm = birdjepa.helpers.tree_l2_norm(grad_params["encoder"])
        obj_grad_norm = birdjepa.helpers.tree_l2_norm(grad_params["objective"])
        probe_grad_norm = birdjepa.helpers.tree_l2_norm(grad_params["probe"])

        updates, new_opt_state = optim.update(grad_params, opt_state, params)

        # Apply updates to models
        new_models = {}
        for k in models:
            new_models[k] = eqx.apply_updates(models[k], updates[k])

        # Compute update norm to verify updates are being applied
        update_norm = birdjepa.helpers.tree_l2_norm(updates)

        # Compute parameter norms (pre and post update)
        param_norm_pre = birdjepa.helpers.tree_l2_norm(params)
        post_params = {
            k: eqx.filter(v, eqx.is_inexact_array) for k, v in new_models.items()
        }
        param_norm = birdjepa.helpers.tree_l2_norm(post_params)

        metrics = {
            "loss": loss,
            "probe_loss": aux["probe_loss"],
            "grad_norm": grad_norm,
            "enc_grad": enc_grad_norm,
            "obj_grad": obj_grad_norm,
            "probe_grad": probe_grad_norm,
            "update_norm": update_norm,
            "param_norm": param_norm,
            "param_norm_pre": param_norm_pre,
            **aux["losses"],
        }
        return new_models, new_opt_state, metrics

    return train_step


@beartype.beartype
def make_eval_step(
    probe_pooling: tp.Literal["cls", "patches"],
):
    """Create eval_step function with sharding."""

    @eqx.filter_jit
    def eval_step(models, batch, *, key):
        encoder, objective, probe = (
            models["encoder"],
            models["objective"],
            models["probe"],
        )
        losses, out, targets = objective(batch, encoder, key=key, mode="eval")
        emb = _pool_embeddings(out, probe_pooling)
        targets_probe = targets.reshape(-1)

        logits = jax.vmap(probe)(emb)
        preds = jnp.argmax(logits, axis=1)
        correct = jnp.sum(preds == targets_probe)
        total = targets_probe.shape[0]

        loss = sum(losses.values())
        return correct, total, loss

    return eval_step


@beartype.beartype
def worker_fn(cfg: Config):
    """Main training function. Uses single-process multi-GPU mode."""
    logging.basicConfig(level=logging.INFO, format=log_format, force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Initialize multi-process JAX distributed runtime for multi-GPU SLURM jobs
    # Each SLURM task is one process; this coordinates across them
    # Skip for single-GPU to avoid heartbeat timeout issues during long JIT
    import multiprocessing

    n_tasks = int(os.environ.get("SLURM_NTASKS", "1"))
    if n_tasks > 1 and multiprocessing.parent_process() is None:
        # Use submitit to get distributed environment, then pass explicitly to JAX
        # This avoids relying on JAX's Slurm auto-detection
        import socket

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
        logger.info("Single-process mode, skipping JAX distributed initialization")

    if cfg.device == "gpu":
        assert birdjepa.helpers.jax_has_gpu(), "GPU not available"

    if cfg.disable_stochastic:
        train_data = cfg.train_data
        if isinstance(train_data, birdjepa.data.XenoCanto):
            train_data = dataclasses.replace(
                train_data,
                truncate="start",
                augmentations=[],
            )
        elif isinstance(train_data, birdjepa.data.Cifar100):
            train_data = dataclasses.replace(train_data, augmentations=[])
        else:
            tp.assert_never(train_data)

        test_data = cfg.test_data
        if isinstance(test_data, birdjepa.data.XenoCanto):
            test_data = dataclasses.replace(
                test_data,
                truncate="start",
                augmentations=[],
            )
        elif isinstance(test_data, birdjepa.data.Cifar100):
            test_data = dataclasses.replace(test_data, augmentations=[])
        else:
            tp.assert_never(test_data)

        cfg = dataclasses.replace(
            cfg, train_data=train_data, test_data=test_data, window_size=1
        )

    msg = f"Unexpected prng_mode: {cfg.prng_mode}"
    assert cfg.prng_mode in ("stateful", "stateless", "checkpointed"), msg

    # Set up sharding (works for both single and multi-GPU)
    n_devices = jax.device_count()
    mesh, data_sharding, model_sharding = birdjepa.helpers.setup_sharding(n_devices)
    logger.info("Using %d device(s) for training", len(mesh.devices))
    logger.info("Devices: %s", [str(d) for d in mesh.devices])

    assert cfg.run_id
    dct = dataclasses.asdict(cfg)
    dct["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    wandb.init(
        project="birdjepa",
        config=dct,
        id=cfg.run_id,
        resume="allow",
        tags=cfg.tags,
    )
    logger.info("wandb run id: %s", cfg.run_id)

    key = jr.key(cfg.seed)
    base_key = key

    # Data
    msg = "LeJEPA objective requires Python datasets; not supported with Rust loader"
    assert not isinstance(cfg.objective, birdjepa.nn.objectives.LeJEPAConfig), msg

    train_data = cfg.train_data
    assert isinstance(train_data, birdjepa.data.XenoCanto)
    assert train_data.n_samples is None, "train_data.n_samples must be None"

    test_data = cfg.test_data
    assert isinstance(test_data, birdjepa.data.XenoCanto)
    assert test_data.n_samples is not None, "test_data.n_samples is required"

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

    # Create dataloaders using Rust loader
    logger.info("Creating train dataloader")
    train_loader = birdjepa.data.RustXenoCantoLoader(
        train_data,
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        n_workers=cfg.n_workers,
        shuffle_buffer_size=cfg.window_size,
        shuffle_min_size=cfg.window_size // 2,
        infinite=True,
    )
    logger.info("Creating test dataloader")
    test_loader = birdjepa.data.RustXenoCantoLoader(
        test_data,
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        n_workers=cfg.n_workers,
        shuffle_buffer_size=1000,
        shuffle_min_size=0,
        infinite=False,
    )
    n_classes = train_loader.n_classes
    n_train = len(train_loader)
    n_test = len(test_loader)
    logger.info(
        "Dataloaders created: %d train, %d test, %d classes",
        n_train,
        n_test,
        n_classes,
    )
    n_eval_batches = (n_test + cfg.batch_size - 1) // cfg.batch_size
    logger.info("Eval: %d batches", n_eval_batches)

    logger.info("Creating objective")
    obj_key, key = jr.split(key)
    objective = birdjepa.nn.objectives.make_objective(
        cfg.objective, cfg.model, train_loader, key=obj_key
    )
    logger.info("Created objective: %s", type(objective).__name__)
    eval_step = make_eval_step(cfg.probe_pooling)

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
        tp.assert_never(cfg.schedule)

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
        encoder, objective, probe, opt_state, start_step, restored_key = restored
        if cfg.prng_mode == "checkpointed":
            msg = "checkpointed prng_mode requires prng_key in checkpoint metadata"
            assert restored_key is not None, msg
            key = restored_key
        # Note: Rust loader state is not checkpointed
        logger.info("Resumed from checkpoint (dataloader restarts from beginning)")

    # Create train_step with sharding
    logger.info("Creating train_step function")
    train_step = make_train_step(
        optim,
        data_sharding,
        model_sharding,
        probe_pooling=cfg.probe_pooling,
    )

    # Training loop (step-based with infinite iterator)
    logger.info("Starting training from step %d (total %d)", start_step, cfg.n_steps)
    step = start_step
    models = {"encoder": encoder, "objective": objective, "probe": probe}
    last_param_norm = None

    # Place models and optimizer state on sharded devices
    models, opt_state = jax.tree.map(
        lambda x: jax.device_put(x, model_sharding) if eqx.is_array(x) else x,
        (models, opt_state),
    )

    # Helper to create train loader (reused for reset test)
    def _make_train_loader():
        return birdjepa.data.RustXenoCantoLoader(
            train_data,
            seed=cfg.seed,
            batch_size=cfg.batch_size,
            n_workers=cfg.n_workers,
            shuffle_buffer_size=cfg.window_size,
            shuffle_min_size=cfg.window_size // 2,
            infinite=True,
        )

    # TODO: total is wrong when resuming (should be cfg.n_steps - step)
    last_step_time = time.time()
    train_iter = iter(train_loader)
    while step < cfg.n_steps:
        batch = next(train_iter)

        # Store targets for entropy computation (only computed when logging)
        targets_np = batch["target"]

        batch = _batch_to_device(batch, data_sharding)

        if cfg.prng_mode == "stateless":
            step_key = jr.fold_in(base_key, step)
        else:
            step_key, key = jr.split(key)

        models, opt_state, metrics = train_step(models, opt_state, batch, key=step_key)

        step += 1

        # Debug: reset dataloader mid-training to test if loss jumps
        if cfg.reset_loader_at_step and step == cfg.reset_loader_at_step:
            logger.info(
                "RESET_LOADER_TEST: Resetting dataloader at step %d (loss before: %.4f)",
                step,
                float(jnp.mean(metrics["probe_loss"])),
            )
            del train_loader
            train_loader = _make_train_loader()
            train_iter = iter(train_loader)
            logger.info(
                "RESET_LOADER_TEST: New dataloader created, continuing training"
            )

        ckpt_param_norm = None
        if ckpt_mngr.should_save(step):
            # Recompute param_norm outside JIT and compare to JIT-computed value
            params_for_norm = {
                k: eqx.filter(v, eqx.is_inexact_array) for k, v in models.items()
            }
            recomputed_norm = float(birdjepa.helpers.tree_l2_norm(params_for_norm))
            leaves = jax.tree.leaves(params_for_norm)
            jit_norm = float(jnp.mean(metrics["param_norm"]))
            ckpt_param_norm = recomputed_norm

            # Log fingerprint and norm comparison for debugging
            leaf_count = len(leaves)
            element_count = sum(p.size for p in leaves)
            logger.info(
                "step=%d CKPT_DEBUG recomputed_norm=%.6f jit_norm=%.6f delta=%.6f leaf_count=%d element_count=%d",
                step,
                recomputed_norm,
                jit_norm,
                recomputed_norm - jit_norm,
                leaf_count,
                element_count,
            )

        # Checkpoint model state (orbax decides whether to actually save based on interval)
        ckpt_mngr.save(
            step=step,
            encoder=models["encoder"],
            objective=models["objective"],
            probe=models["probe"],
            opt_state=opt_state,
            encoder_config=cfg.model,
            prng_key=key if cfg.prng_mode == "checkpointed" else None,
            param_norm=ckpt_param_norm,
        )

        # Log peak memory usage after first step (includes JIT compilation overhead)
        if step == 1:
            for device in jax.local_devices():
                stats = device.memory_stats()
                if stats:
                    peak_gb = stats.get("peak_bytes_in_use", 0) / 1e9
                    logger.info("Device %s peak memory: %.2f GB", device, peak_gb)

        if step % cfg.log_every == 0:
            now = time.time()
            sec_per_step = (now - last_step_time) / cfg.log_every
            last_step_time = now

            # Compute target entropy (measures shuffle quality)
            tgt_entropy = compute_tgt_entropy(targets_np)
            max_entropy = np.log2(cfg.batch_size)
            tgt_entropy_frac = tgt_entropy / max_entropy  # fraction of max possible

            # Average metrics across devices for logging
            metrics = jax.tree.map(lambda x: float(jnp.mean(x)), metrics)

            lr = float(schedule(step))
            step_opt_stats = birdjepa.checkpoints.get_opt_state_stats(opt_state)
            metric_str = " ".join(
                f"{k}={float(v):.4g}" for k, v in sorted(metrics.items())
            )
            logger.info(
                "step=%d lr=%.2e tgt_entropy=%.2f (%.0f%%) sec/step=%.1f %s",
                step,
                lr,
                tgt_entropy,
                tgt_entropy_frac * 100,
                sec_per_step,
                metric_str,
            )
            log_dict = {
                "step": step,
                "lr": lr,
                "train/probe": float(metrics["probe_loss"]),
                "train/opt_state_l2": _none_to_nan(step_opt_stats["l2"]),
                "train/opt_state_abs_mean": _none_to_nan(step_opt_stats["abs_mean"]),
                "train/opt_state_abs_max": _none_to_nan(step_opt_stats["abs_max"]),
                "train/tgt_entropy": tgt_entropy,
                "train/tgt_entropy_frac": tgt_entropy_frac,
                "train/sec_per_step": sec_per_step,
            }
            for k, v in metrics.items():
                if k not in ("loss", "probe_loss"):
                    log_dict[f"train/{k}"] = float(v)

            # Dataloader diagnostics
            for k, v in train_loader.diagnostics().items():
                log_dict[f"dataloader/{k}"] = v

            wandb.log(log_dict)

        last_param_norm = metrics["param_norm"]

        # Evaluate periodically
        if step % cfg.eval_every == 0:
            correct = 0
            total = 0
            test_loss = 0.0

            eval_key = jr.key(0)  # Fixed key for deterministic eval
            n_test_batches = 0
            for test_batch in test_loader:
                n_test_batches += 1
                eval_key, obj_key = jr.split(eval_key)
                batch = _batch_to_device(test_batch, data_sharding)
                correct_b, total_b, loss_b = eval_step(models, batch, key=obj_key)
                correct += int(correct_b)
                total += int(total_b)
                test_loss += float(loss_b)
            acc = correct / max(1, total)
            test_loss /= max(1, n_test_batches)
            logger.info("step=%d test_acc=%.4f test_loss=%.4f", step, acc, test_loss)
            wandb.log({"step": step, "test/acc": acc, "test/loss": test_loss})

    # Save final checkpoint and wait for async saves to complete
    final_param_norm = None
    if last_param_norm is not None:
        final_param_norm = float(jnp.mean(last_param_norm))
    else:
        final_params = {
            k: eqx.filter(v, eqx.is_inexact_array) for k, v in models.items()
        }
        final_param_norm = float(birdjepa.helpers.tree_l2_norm(final_params))

    ckpt_mngr.save(
        step=step,
        encoder=models["encoder"],
        objective=models["objective"],
        probe=models["probe"],
        opt_state=opt_state,
        encoder_config=cfg.model,
        prng_key=key if cfg.prng_mode == "checkpointed" else None,
        param_norm=final_param_norm,
        force=True,
    )
    ckpt_mngr.wait_until_finished()
    wandb.finish()


@beartype.beartype
class TrainingJob:
    """Checkpointable wrapper for submitit experiments."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def __call__(self):
        worker_fn(self.cfg)

    def checkpoint(self):
        import submitit.helpers

        return submitit.helpers.DelayedSubmission(TrainingJob(self.cfg))


@beartype.beartype
def cli(cfg: Config):
    """CLI entrypoint: run locally or submit to Slurm."""
    logging.basicConfig(level=logging.INFO, format=log_format, force=True)

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

    import birdjepa._rs
    import submitit
    import submitit.core.utils

    build_profile = getattr(birdjepa._rs, "build_profile", None)
    msg = "birdjepa._rs.build_profile is missing; rebuild the Rust extension with `uv run python -m maturin develop --release`"
    assert build_profile is not None, msg
    profile = build_profile()
    msg = f"Rust extension must be built in release mode; got {profile!r}. Rebuild with `uv run python -m maturin develop --release`"
    assert profile == "release", msg

    executor = submitit.SlurmExecutor(folder=base.log_to, max_num_timeout=10)
    # OSC allocates ~10GB RAM per CPU, so request enough CPUs for desired memory
    n_cpus = (
        max(base.mem_gb // 10, base.n_workers) if base.mem_gb > 0 else base.n_workers
    )
    if n_cpus > base.n_workers:
        logger.info("Requesting %d CPUs to get %dGB RAM", n_cpus, base.mem_gb)
    tag = pathlib.Path(base.sweep).stem if base.sweep else "local"
    # Single process sees all GPUs (no JAX distributed coordination needed)
    setup = [
        "module load ffmpeg/6.1.1",
        "export NCCL_IB_DISABLE=1",  # Disable InfiniBand (not available)
        "export NCCL_P2P_DISABLE=1",  # Use shared memory
    ]
    executor.update_parameters(
        time=int(base.n_hours * 60),
        partition=base.slurm_partition,
        gpus_per_node=base.n_gpus,
        ntasks_per_node=1,
        cpus_per_task=n_cpus,
        stderr_to_stdout=True,
        account=base.slurm_acct,
        job_name=f"pretrain[{tag}]",
        setup=setup,
    )

    jobs = [executor.submit(TrainingJob(c)) for c in cfgs]

    time.sleep(5.0)
    for j, job in enumerate(jobs):
        logger.info("Job %d/%d: %s %s", j + 1, len(jobs), job.job_id, job.state)

    for j, job in enumerate(jobs):
        try:
            job.result()
            logger.info("Job %d/%d finished.", j + 1, len(jobs))
        except submitit.core.utils.UncompletedJobError:
            logger.warning("Job %s (%d) did not finish.", job.job_id, j)
