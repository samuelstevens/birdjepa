"""
Benchmark package for evaluating audio backbones on BirdSet tasks.

Tasks: pow, per, nes, uhh, hsn, nbp, ssw, sne
Classifiers: linear, mlp, centroid
"""

import dataclasses
import importlib.util
import logging
import os
import pathlib

import beartype
import datasets
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import sklearn.metrics
from jaxtyping import Array, Float, jaxtyped

from . import registry, reporting
from .. import helpers

logger = logging.getLogger("benchmark")


# BirdSet uses this column name for multi-label class indices
BIRDSET_LABEL_COL = "ebird_code_multilabel"


def get_audio_array(audio: dict) -> np.ndarray:
    """Get audio array from HuggingFace audio dict (handles streaming and non-streaming)."""
    if "array" in audio:
        # Non-streaming: already decoded
        data = audio["array"]
    elif "bytes" in audio:
        # Streaming: decode from bytes
        import io
        import soundfile as sf

        data, _ = sf.read(io.BytesIO(audio["bytes"]))
    else:
        raise ValueError(f"Unknown audio format: {list(audio.keys())}")

    # Convert stereo to mono if needed
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data


@beartype.beartype
def sample_indices_per_class(
    labels: list[list[int]], n_per_class: int, n_classes: int, seed: int = 42
) -> set[int]:
    """Sample up to n_per_class example indices for each class."""
    rng = np.random.default_rng(seed)
    selected: set[int] = set()

    class_to_examples: dict[int, list[int]] = {c: [] for c in range(n_classes)}
    for i, ex_labels in enumerate(labels):
        for c in ex_labels:
            class_to_examples[c].append(i)

    for c in range(n_classes):
        samples = class_to_examples[c]
        if len(samples) <= n_per_class:
            selected.update(samples)
        else:
            selected.update(
                int(i) for i in rng.choice(samples, size=n_per_class, replace=False)
            )

    logger.info(
        "Sampled %d examples (%d per class, %d classes)",
        len(selected),
        n_per_class,
        n_classes,
    )
    return selected


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    """Benchmark configuration."""

    task: str = "pow"
    """BirdSet task: pow, per, nes, uhh, hsn, nbp, ssw, sne."""
    model_org: str = "bird-mae"
    """Model organization."""
    model_ckpt: str = "Bird-MAE-Base"
    """Model checkpoint."""
    ckpt_path: str = ""
    """Path to local JAX checkpoint. If set, model_org/model_ckpt are ignored."""
    classifier: str = "linear"
    """Classifier: linear, mlp, centroid."""
    n_train: int = -1
    """Number of training samples per class. -1 for all."""
    # Training (Bird-MAE defaults)
    lr: float = 4e-4
    """Learning rate."""
    weight_decay: float = 3e-4
    """AdamW weight decay."""
    epochs: int = 30
    """Training epochs."""
    batch_size: int = 64
    """Batch size."""
    grad_clip: float = 2.0
    """Gradient clipping max norm."""
    log_every: int = 5
    """Log training progress every N epochs."""
    # Paths
    report_to: pathlib.Path = pathlib.Path("./results")
    """Directory for results database."""
    # Execution
    sweep: str = ""
    """Sweep file path, e.g. 'sweeps/hsn_classifiers.py'. Runs make_cfgs()."""
    dry_run: bool = True
    """If True, only print what would run. Use --no-dry-run to execute."""
    slurm_acct: str = ""
    """Slurm account string. Empty means run locally."""
    slurm_partition: str = ""
    """Slurm partition."""
    n_hours: float = 4.0
    """Slurm job length in hours."""
    log_to: pathlib.Path = pathlib.Path("./logs")
    """Where to log Slurm job stdout/stderr."""


@beartype.beartype
def make_exp_key(cfg: Config) -> reporting.ExpKey:
    """Create an ExpKey from a Config."""
    if cfg.ckpt_path:
        model_org = "birdjepa"
        model_ckpt = pathlib.Path(cfg.ckpt_path).name
    else:
        model_org = cfg.model_org
        model_ckpt = cfg.model_ckpt
    return reporting.ExpKey(
        task_name=cfg.task,
        model_org=model_org,
        model_ckpt=model_ckpt,
        clf=cfg.classifier,
        n_train=cfg.n_train,
    )


@beartype.beartype
def worker_fn(cfg: Config) -> None:
    """Run a single benchmark experiment and write results to parquet."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Use node-local TMPDIR for HuggingFace cache to avoid NFS corruption
    tmpdir = os.environ.get("TMPDIR")
    assert tmpdir, "TMPDIR must be set for node-local caching"
    os.environ["HF_HUB_CACHE"] = tmpdir
    os.environ["HF_HOME"] = tmpdir
    logger.info("Using HF cache: %s", tmpdir)

    key = make_exp_key(cfg)
    logger.info("Benchmarking %s", key)

    # 1. Load backbone (JAX model)
    backbone = registry.load(cfg.model_org, cfg.model_ckpt)
    transform = backbone.make_audio_transform()

    # 2. Get n_classes from dataset builder (before streaming)
    builder = datasets.load_dataset_builder("samuelstevens/BirdSet", cfg.task.upper())
    n_classes = len(builder.info.features[BIRDSET_LABEL_COL].feature.names)

    # 3. Load datasets with streaming to avoid NFS cache corruption
    logger.info("Loading dataset with streaming...")
    ds = datasets.load_dataset(
        "samuelstevens/BirdSet", cfg.task.upper(), streaming=True
    )

    # 4. Extract train features
    if cfg.n_train > 0:
        # Sample n_train per class: stream twice to avoid OOM
        logger.info("First pass: collecting labels for sampling...")
        all_labels = list(ds["train"][BIRDSET_LABEL_COL])
        selected_i = sample_indices_per_class(all_labels, cfg.n_train, n_classes)

        logger.info("Second pass: collecting %d selected examples...", len(selected_i))
        ds = datasets.load_dataset(
            "samuelstevens/BirdSet", cfg.task.upper(), streaming=True
        )
        train_examples = [ex for i, ex in enumerate(ds["train"]) if i in selected_i]

        logger.info(
            "Task %s: %d train, %d classes", cfg.task, len(train_examples), n_classes
        )
        logger.info("Extracting train features...")
        train = extract_features_from_list(
            backbone, transform, train_examples, cfg.batch_size, n_classes
        )
    else:
        # Use all samples: stream feature extraction to avoid OOM
        logger.info("Task %s: all train, %d classes (streaming)", cfg.task, n_classes)
        logger.info("Extracting train features (streaming)...")
        train = extract_features_streaming(
            backbone, transform, ds["train"], cfg.batch_size, n_classes
        )

    logger.info("Extracting test features (streaming)...")
    test = extract_features_streaming(
        backbone, transform, ds["test_5s"], cfg.batch_size, n_classes
    )

    # 5. Train classifier (JAX/optax)
    logger.info("Training %s probe...", cfg.classifier)
    rng_key = jr.key(42)
    if cfg.classifier == "linear":
        probe = train_linear_probe(
            train.features, train.labels, n_classes, cfg, rng_key
        )
    elif cfg.classifier == "mlp":
        probe = train_mlp_probe(train.features, train.labels, n_classes, cfg, rng_key)
    elif cfg.classifier == "centroid":
        probe = train_centroid_probe(train.features, train.labels, n_classes)
    else:
        raise NotImplementedError(f"Classifier '{cfg.classifier}' not implemented")

    # 6. Evaluate
    logger.info("Evaluating...")
    cmap, n_classes_eval, predictions = evaluate(probe, test)
    logger.info("Results: cmAP=%.4f", cmap)

    # 7. Build and write report
    exp_cfg = dataclasses.asdict(cfg)
    for k, v in exp_cfg.items():
        if isinstance(v, pathlib.Path):
            exp_cfg[k] = str(v)

    report = reporting.Report(
        key=key,
        cmap=cmap,
        n_classes=n_classes_eval,
        predictions=predictions,
        exp_cfg=exp_cfg,
    )
    report.write(cfg.report_to)


@beartype.beartype
def get_skip_reason(report_to: pathlib.Path, cfg: Config) -> str | None:
    """Return a reason to skip this experiment, or None if it should run."""
    key = make_exp_key(cfg)
    if reporting.already_ran(report_to, key):
        return "done"
    return None


@beartype.beartype
def load_sweep_cfgs(sweep_fpath: str) -> list[dict]:
    """Load a sweep file and return make_cfgs() result."""
    spec = importlib.util.spec_from_file_location("sweep", sweep_fpath)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_cfgs()


@beartype.beartype
def print_results_table(report_to: pathlib.Path, cfgs: list[Config]):
    """Print a table of results for the given configs."""
    import duckdb

    tasks = sorted({c.task for c in cfgs})
    clfs = sorted({c.classifier for c in cfgs})
    n_trains = sorted({c.n_train for c in cfgs})
    models = sorted({c.model_ckpt for c in cfgs})

    raw_dir = report_to / "raw"
    results: dict[tuple[str, str, str, int], float] = {}
    if raw_dir.exists():
        pattern = str(raw_dir / "*.parquet")
        try:
            query = (
                "SELECT task_name, model_ckpt, clf, n_train, cmap FROM read_parquet(?)"
            )
            rows = duckdb.execute(query, [pattern]).fetchall()
            for task_name, model_ckpt, clf, n_train, cmap in rows:
                results[(task_name, model_ckpt, clf, n_train)] = cmap
        except duckdb.IOException:
            pass

    for model in models:
        print(f"\n=== {model} ===")
        header = ["clf", "n"] + tasks
        print(" | ".join(f"{h:>10}" for h in header))
        print("-" * (12 * len(header)))
        for clf in clfs:
            for n_train in n_trains:
                n_str = "all" if n_train == -1 else str(n_train)
                row = [clf, n_str]
                for task in tasks:
                    key = (task, model, clf, n_train)
                    if key in results:
                        row.append(f"{results[key]:.4f}")
                    else:
                        row.append("-")
                print(" | ".join(f"{v:>10}" for v in row))


@beartype.beartype
def cli(cfg: Config):
    """CLI entrypoint: run locally or submit to Slurm."""
    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)

    if cfg.sweep:
        sweep_cfgs = load_sweep_cfgs(cfg.sweep)
        logger.info("Loaded %d configs from %s", len(sweep_cfgs), cfg.sweep)
        cfgs = [dataclasses.replace(cfg, sweep="", **s) for s in sweep_cfgs]
    else:
        cfgs = [cfg]

    if cfg.dry_run:
        print_results_table(cfg.report_to, cfgs)
        print("\nWould run:")
        for c in cfgs:
            key = make_exp_key(c)
            reason = get_skip_reason(cfg.report_to, c)
            if reason == "done":
                continue
            elif reason:
                print(f"  [skip: {reason}] {key}")
            else:
                print(f"  [pending] {key}")
        return

    to_run: list[Config] = []
    for c in cfgs:
        key = make_exp_key(c)
        reason = get_skip_reason(cfg.report_to, c)
        if reason == "done":
            continue
        if reason:
            logger.info("Skipping %s: %s", key, reason)
            continue
        to_run.append(c)

    if not to_run:
        return

    if cfg.slurm_acct:
        import submitit

        executor = submitit.SlurmExecutor(folder=cfg.log_to)
        executor.update_parameters(
            time=int(cfg.n_hours * 60),
            partition=cfg.slurm_partition,
            gpus_per_node=1,
            ntasks_per_node=1,
            cpus_per_task=4,
            stderr_to_stdout=True,
            account=cfg.slurm_acct,
            setup=["module load ffmpeg/6.1.1"],
        )

        with executor.batch():
            jobs = [executor.submit(worker_fn, c) for c in to_run]

        for job, c in zip(jobs, to_run):
            logger.info("Submitted job %s for %s", job.job_id, make_exp_key(c))
    else:
        for c in to_run:
            worker_fn(c)


# =============================================================================
# Feature Extraction (JAX)
# =============================================================================


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass
class ExtractedFeatures:
    """Features extracted from a dataset (JAX arrays)."""

    features: Float[Array, "n d"]
    labels: Float[Array, "n c"]
    ids: list[str]

    def __post_init__(self):
        assert len(self.ids) == len(self.features), (
            f"{len(self.ids)} != {len(self.features)}"
        )


@beartype.beartype
def extract_features_from_list(
    backbone, transform, examples: list[dict], batch_size: int, n_classes: int
) -> ExtractedFeatures:
    """Extract features from a list of example dicts (JAX)."""
    all_features = []
    all_labels = []
    all_ids = []

    batches = range(0, len(examples), batch_size)
    for i in helpers.progress(batches, every=10, desc="extract"):
        batch = examples[i : i + batch_size]

        specs = []
        for ex in batch:
            audio_array = get_audio_array(ex["audio"])
            spec = transform(audio_array)
            specs.append(spec)

        specs_jax = jnp.array(np.stack(specs), dtype=jnp.float32)
        encoded = backbone.encode(specs_jax)
        all_features.append(encoded)

        labels = jnp.zeros((len(batch), n_classes), dtype=jnp.float32)
        for j, ex in enumerate(batch):
            for c in ex[BIRDSET_LABEL_COL]:
                labels = labels.at[j, c].set(1.0)
        all_labels.append(labels)

        all_ids.extend(ex["filepath"] for ex in batch)

    return ExtractedFeatures(
        features=jnp.concatenate(all_features, axis=0),
        labels=jnp.concatenate(all_labels, axis=0),
        ids=all_ids,
    )


@beartype.beartype
def extract_features_streaming(
    backbone, transform, dataset, batch_size: int, n_classes: int
) -> ExtractedFeatures:
    """Extract features from a streaming dataset (JAX)."""
    all_features = []
    all_labels = []
    all_ids = []
    n_batches = 0

    batch = []
    for ex in dataset:
        batch.append(ex)
        if len(batch) >= batch_size:
            _process_batch(
                backbone, transform, batch, n_classes, all_features, all_labels, all_ids
            )
            batch = []
            n_batches += 1
            if n_batches % 10 == 0:
                logger.info("Processed %d batches", n_batches)

    if batch:
        _process_batch(
            backbone, transform, batch, n_classes, all_features, all_labels, all_ids
        )

    return ExtractedFeatures(
        features=jnp.concatenate(all_features, axis=0),
        labels=jnp.concatenate(all_labels, axis=0),
        ids=all_ids,
    )


def _process_batch(
    backbone, transform, batch, n_classes, all_features, all_labels, all_ids
):
    """Helper to process a batch of examples (JAX)."""
    specs = []
    for ex in batch:
        audio_array = get_audio_array(ex["audio"])
        spec = transform(audio_array)
        specs.append(spec)

    specs_jax = jnp.array(np.stack(specs), dtype=jnp.float32)
    encoded = backbone.encode(specs_jax)
    all_features.append(encoded)

    labels = jnp.zeros((len(batch), n_classes), dtype=jnp.float32)
    for j, ex in enumerate(batch):
        for c in ex[BIRDSET_LABEL_COL]:
            labels = labels.at[j, c].set(1.0)
    all_labels.append(labels)

    all_ids.extend(ex["filepath"] for ex in batch)


# ===============#
# Probe Training #
# ===============#


@jaxtyped(typechecker=beartype.beartype)
def asymmetric_loss(
    logits: Float[Array, "b c"],
    targets: Float[Array, "b c"],
    gamma_neg: float = 4.0,
    gamma_pos: float = 1.0,
    clip: float = 0.05,
    eps: float = 1e-8,
) -> Float[Array, ""]:
    """Asymmetric loss for multi-label classification (JAX version)."""
    probs = jax.nn.sigmoid(logits)
    probs_pos = probs
    probs_neg = 1 - probs

    if clip > 0:
        probs_neg = jnp.clip(probs_neg + clip, max=1)

    loss_pos = targets * jnp.log(jnp.clip(probs_pos, min=eps))
    loss_neg = (1 - targets) * jnp.log(jnp.clip(probs_neg, min=eps))
    loss = loss_pos + loss_neg

    if gamma_neg > 0 or gamma_pos > 0:
        pt = probs_pos * targets + probs_neg * (1 - targets)
        gamma = gamma_pos * targets + gamma_neg * (1 - targets)
        loss = loss * jnp.power(1 - pt, gamma)

    return -loss.mean()


@jaxtyped(typechecker=beartype.beartype)
class LinearProbe(eqx.Module):
    """Linear probe: LayerNorm + Linear."""

    norm_weight: Float[Array, " dim"]
    norm_bias: Float[Array, " dim"]
    linear_weight: Float[Array, "n_classes dim"]
    linear_bias: Float[Array, " n_classes"]
    eps: float = eqx.field(static=True)

    def __init__(self, dim: int, n_classes: int, *, key):
        self.norm_weight = jnp.ones(dim)
        self.norm_bias = jnp.zeros(dim)
        self.linear_weight = jr.normal(key, (n_classes, dim)) * 0.02
        self.linear_bias = jnp.zeros(n_classes)
        self.eps = 1e-6

    def __call__(self, x: Float[Array, "batch dim"]) -> Float[Array, "batch n_classes"]:
        # LayerNorm
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x = (x - mean) / jnp.sqrt(var + self.eps) * self.norm_weight + self.norm_bias
        # Linear
        return x @ self.linear_weight.T + self.linear_bias


@jaxtyped(typechecker=beartype.beartype)
class MLPProbe(eqx.Module):
    """MLP probe: LayerNorm + Linear + GELU + Linear."""

    norm_weight: Float[Array, " dim"]
    norm_bias: Float[Array, " dim"]
    fc1_weight: Float[Array, "hidden dim"]
    fc1_bias: Float[Array, " hidden"]
    fc2_weight: Float[Array, "n_classes hidden"]
    fc2_bias: Float[Array, " n_classes"]
    eps: float = eqx.field(static=True)

    def __init__(self, dim: int, n_classes: int, *, key):
        k1, k2 = jr.split(key)
        hidden = dim * 2
        self.norm_weight = jnp.ones(dim)
        self.norm_bias = jnp.zeros(dim)
        self.fc1_weight = jr.normal(k1, (hidden, dim)) * 0.02
        self.fc1_bias = jnp.zeros(hidden)
        self.fc2_weight = jr.normal(k2, (n_classes, hidden)) * 0.02
        self.fc2_bias = jnp.zeros(n_classes)
        self.eps = 1e-6

    def __call__(self, x: Float[Array, "batch dim"]) -> Float[Array, "batch n_classes"]:
        # LayerNorm
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x = (x - mean) / jnp.sqrt(var + self.eps) * self.norm_weight + self.norm_bias
        # MLP
        x = x @ self.fc1_weight.T + self.fc1_bias
        x = jax.nn.gelu(x)
        return x @ self.fc2_weight.T + self.fc2_bias


@jaxtyped(typechecker=beartype.beartype)
class CentroidClassifier(eqx.Module):
    """Multi-label nearest-centroid classifier using cosine similarity."""

    centroids: Float[Array, "n_classes dim"]

    def __call__(self, x: Float[Array, "batch dim"]) -> Float[Array, "batch n_classes"]:
        x_norm = x / jnp.clip(jnp.linalg.norm(x, axis=1, keepdims=True), min=1e-8)
        c_norm = self.centroids / jnp.clip(
            jnp.linalg.norm(self.centroids, axis=1, keepdims=True), min=1e-8
        )
        return x_norm @ c_norm.T * 10.0  # temperature=0.1


@jaxtyped(typechecker=beartype.beartype)
def train_linear_probe(
    features: Float[Array, "n d"],
    labels: Float[Array, "n c"],
    n_classes: int,
    cfg: Config,
    key: jax.Array,
) -> LinearProbe:
    """Train a linear probe with asymmetric loss."""
    dim = features.shape[1]
    probe = LinearProbe(dim, n_classes, key=key)

    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.adamw(cfg.lr, weight_decay=cfg.weight_decay, b1=0.9, b2=0.95),
    )
    opt_state = optimizer.init(eqx.filter(probe, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(probe, opt_state, x, y):
        def loss_fn(p):
            logits = jax.vmap(p)(x)
            return asymmetric_loss(logits, y)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(probe)
        updates, opt_state_new = optimizer.update(
            grads, opt_state, eqx.filter(probe, eqx.is_inexact_array)
        )
        probe_new = eqx.apply_updates(probe, updates)
        return probe_new, opt_state_new, loss

    n_samples = len(features)
    batch_size = min(cfg.batch_size, n_samples)
    rng = np.random.default_rng(42)

    for epoch in helpers.progress(range(cfg.epochs), every=cfg.log_every, desc="train"):
        indices = rng.permutation(n_samples)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            batch_i = indices[i : i + batch_size]
            x = features[batch_i]
            y = labels[batch_i]
            probe, opt_state, loss = step(probe, opt_state, x, y)
            total_loss += float(loss)
            n_batches += 1

        if (epoch + 1) % cfg.log_every == 0:
            avg_loss = total_loss / n_batches
            logger.info("epoch %d: loss=%.4f", epoch + 1, avg_loss)

    return probe


@jaxtyped(typechecker=beartype.beartype)
def train_mlp_probe(
    features: Float[Array, "n d"],
    labels: Float[Array, "n c"],
    n_classes: int,
    cfg: Config,
    key: jax.Array,
) -> MLPProbe:
    """Train a 2-layer MLP probe."""
    dim = features.shape[1]
    probe = MLPProbe(dim, n_classes, key=key)

    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.adamw(cfg.lr, weight_decay=cfg.weight_decay, b1=0.9, b2=0.95),
    )
    opt_state = optimizer.init(eqx.filter(probe, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(probe, opt_state, x, y):
        def loss_fn(p):
            logits = jax.vmap(p)(x)
            return asymmetric_loss(logits, y)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(probe)
        updates, opt_state_new = optimizer.update(
            grads, opt_state, eqx.filter(probe, eqx.is_inexact_array)
        )
        probe_new = eqx.apply_updates(probe, updates)
        return probe_new, opt_state_new, loss

    n_samples = len(features)
    batch_size = min(cfg.batch_size, n_samples)
    rng = np.random.default_rng(42)

    for epoch in helpers.progress(range(cfg.epochs), every=cfg.log_every, desc="train"):
        indices = rng.permutation(n_samples)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            batch_i = indices[i : i + batch_size]
            x = features[batch_i]
            y = labels[batch_i]
            probe, opt_state, loss = step(probe, opt_state, x, y)
            total_loss += float(loss)
            n_batches += 1

        if (epoch + 1) % cfg.log_every == 0:
            avg_loss = total_loss / n_batches
            logger.info("epoch %d: loss=%.4f", epoch + 1, avg_loss)

    return probe


@jaxtyped(typechecker=beartype.beartype)
def train_centroid_probe(
    features: Float[Array, "n d"], labels: Float[Array, "n c"], n_classes: int
) -> CentroidClassifier:
    """Build a centroid classifier by computing per-class means."""
    dim = features.shape[1]
    centroids = jnp.zeros((n_classes, dim), dtype=jnp.float32)

    for c in range(n_classes):
        mask = labels[:, c] > 0
        if mask.sum() > 0:
            centroids = centroids.at[c].set(features[mask].mean(axis=0))
        else:
            logger.warning("Class %d has no positive examples, using zero centroid.", c)

    return CentroidClassifier(centroids=centroids)


# ==========-#
# Evaluation #
# ===========#


@beartype.beartype
def evaluate(
    probe, test: ExtractedFeatures
) -> tuple[float, int, list[reporting.Prediction]]:
    """Evaluate and return cmAP along with per-example predictions."""
    features = test.features

    # Run inference in batches to avoid OOM
    batch_size = 256
    all_probs = []
    for i in range(0, len(features), batch_size):
        batch = features[i : i + batch_size]
        logits = jax.vmap(probe)(batch)
        probs = jax.nn.sigmoid(logits)
        all_probs.append(probs)

    probs = jnp.concatenate(all_probs, axis=0)
    labels = test.labels

    # cmAP: average AP across classes (need numpy for sklearn)
    probs_np = np.array(probs)
    labels_np = np.array(labels)
    aps = []
    for c in range(labels_np.shape[1]):
        if labels_np[:, c].sum() > 0:
            ap = sklearn.metrics.average_precision_score(
                labels_np[:, c], probs_np[:, c]
            )
            aps.append(ap)

    cmap = float(np.mean(aps)) if aps else 0.0

    # Build predictions list
    predictions = []
    for i, example_id in enumerate(test.ids):
        y_true = [int(c) for c in range(labels_np.shape[1]) if labels_np[i, c] > 0]
        y_pred = [int(c) for c in range(probs_np.shape[1]) if probs_np[i, c] > 0.5]
        y_score = [float(probs_np[i, c]) for c in range(probs_np.shape[1])]
        predictions.append(reporting.Prediction(example_id, y_true, y_pred, y_score))

    return cmap, len(aps), predictions
