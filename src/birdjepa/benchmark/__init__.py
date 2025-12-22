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
import jaxtyping
import numpy as np
import sklearn.metrics
import torch
import torch.nn
import torch.utils.data
from jaxtyping import Float
from torch import Tensor

from . import registry, reporting
from .. import helpers

logger = logging.getLogger("benchmark")


@jaxtyping.jaxtyped(typechecker=beartype.beartype)
class AsymmetricLoss(torch.nn.Module):
    """Asymmetric loss for multi-label classification.

    From "Asymmetric Loss For Multi-Label Classification" (Ben-Baruch et al.).
    Designed for imbalanced multi-label problems where negatives >> positives.
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(
        self, logits: Float[Tensor, "b c"], targets: Float[Tensor, "b c"]
    ) -> Float[Tensor, ""]:
        probs = torch.sigmoid(logits)
        probs_pos = probs
        probs_neg = 1 - probs

        # Asymmetric clipping: reduce contribution of very easy negatives
        if self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1)

        # Basic cross-entropy terms
        loss_pos = targets * torch.log(probs_pos.clamp(min=self.eps))
        loss_neg = (1 - targets) * torch.log(probs_neg.clamp(min=self.eps))
        loss = loss_pos + loss_neg

        # Asymmetric focusing (focal loss style)
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt = probs_pos * targets + probs_neg * (1 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            loss *= torch.pow(1 - pt, gamma)

        return -loss.mean()


# BirdSet uses this column name for multi-label class indices
BIRDSET_LABEL_COL = "ebird_code_multilabel"


@beartype.beartype
def sample_per_class(
    dataset: datasets.Dataset,
    n_per_class: int,
    label_col: str = BIRDSET_LABEL_COL,
    seed: int = 42,
) -> datasets.Dataset:
    """Sample up to n_per_class examples for each class, return their union.

    For multi-label data, a sample can belong to multiple classes. We iterate through each class, find all samples containing that class, and randomly select up to n_per_class of them. The final dataset is the union of all selected indices.

    Args:
        dataset: HuggingFace dataset with a multi-label column.
        n_per_class: Max samples to select per class.
        label_col: Column containing list of class indices per example.
        seed: Random seed for reproducibility.
    """
    assert label_col in dataset.features, (
        f"'{label_col}' not in {list(dataset.features)}"
    )

    rng = np.random.default_rng(seed)
    n_classes = len(dataset.features[label_col].feature.names)
    selected: set[int] = set()

    # Build index: for each class, which samples contain it?
    labels = dataset[label_col]
    class_to_samples: dict[int, list[int]] = {c: [] for c in range(n_classes)}
    for i, label in enumerate(labels):
        for c in label:
            class_to_samples[c].append(i)

    # Sample from each class
    for c in range(n_classes):
        samples = class_to_samples[c]
        if len(samples) <= n_per_class:
            selected.update(samples)
        else:
            selected.update(rng.choice(samples, size=n_per_class, replace=False))

    indices = sorted(selected)
    logger.info(
        "Sampled %d examples (%d per class, %d classes)",
        len(indices),
        n_per_class,
        n_classes,
    )
    return dataset.select(indices)


@beartype.beartype
def sample_per_class_list(
    examples: list[dict], n_per_class: int, n_classes: int, seed: int = 42
) -> list[dict]:
    """Sample up to n_per_class examples for each class from a list of dicts.

    For streaming datasets that have been collected to a list.
    """
    rng = np.random.default_rng(seed)
    selected: set[int] = set()

    # Build index: for each class, which examples contain it?
    class_to_examples: dict[int, list[int]] = {c: [] for c in range(n_classes)}
    for i, ex in enumerate(examples):
        for c in ex[BIRDSET_LABEL_COL]:
            class_to_examples[c].append(i)

    # Sample from each class
    for c in range(n_classes):
        samples = class_to_examples[c]
        if len(samples) <= n_per_class:
            selected.update(samples)
        else:
            selected.update(rng.choice(samples, size=n_per_class, replace=False))

    indices = sorted(selected)
    logger.info(
        "Sampled %d examples (%d per class, %d classes)",
        len(indices),
        n_per_class,
        n_classes,
    )
    return [examples[i] for i in indices]


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
    device: str = "cuda"
    """Device."""
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
    return reporting.ExpKey(
        task_name=cfg.task,
        model_org=cfg.model_org,
        model_ckpt=cfg.model_ckpt,
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

    # 1. Load backbone
    backbone = registry.load(cfg.model_org, cfg.model_ckpt)
    backbone = backbone.to(cfg.device)
    backbone.eval()
    transform = backbone.make_audio_transform()

    # 2. Get n_classes from dataset builder (before streaming)
    builder = datasets.load_dataset_builder("samuelstevens/BirdSet", cfg.task.upper())
    n_classes = len(builder.info.features[BIRDSET_LABEL_COL].feature.names)

    # 3. Load datasets with streaming to avoid NFS cache corruption
    logger.info("Loading dataset with streaming...")
    ds = datasets.load_dataset(
        "samuelstevens/BirdSet", cfg.task.upper(), streaming=True
    )

    # Collect train examples (small enough to fit in memory)
    logger.info("Collecting train examples...")
    train_examples = list(ds["train"])
    if cfg.n_train > 0:
        train_examples = sample_per_class_list(train_examples, cfg.n_train, n_classes)

    logger.info(
        "Task %s: %d train, %d classes (test streaming)",
        cfg.task,
        len(train_examples),
        n_classes,
    )

    # 4. Extract features
    logger.info("Extracting train features...")
    train = extract_features_from_list(
        backbone, transform, train_examples, cfg.device, cfg.batch_size, n_classes
    )

    logger.info("Extracting test features (streaming)...")
    test = extract_features_streaming(
        backbone, transform, ds["test_5s"], cfg.device, cfg.batch_size, n_classes
    )

    # 5. Train classifier
    logger.info("Training %s probe...", cfg.classifier)
    if cfg.classifier == "linear":
        probe = train_linear_probe(train.features, train.labels, n_classes, cfg)
    elif cfg.classifier == "mlp":
        probe = train_mlp_probe(train.features, train.labels, n_classes, cfg)
    elif cfg.classifier == "centroid":
        probe = train_centroid_probe(train.features, train.labels, n_classes, cfg)
    else:
        raise NotImplementedError(f"Classifier '{cfg.classifier}' not implemented")

    # 6. Evaluate
    logger.info("Evaluating...")
    cmap, n_classes_eval, predictions = evaluate(probe, test, cfg.device)
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

    # Get dimensions from configs
    tasks = sorted({c.task for c in cfgs})
    clfs = sorted({c.classifier for c in cfgs})
    n_trains = sorted({c.n_train for c in cfgs})
    models = sorted({c.model_ckpt for c in cfgs})

    # Query results from parquet files
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
            pass  # No parquet files yet

    # Print table for each model
    for model in models:
        print(f"\n=== {model} ===")
        # Header
        header = ["clf", "n"] + tasks
        print(" | ".join(f"{h:>10}" for h in header))
        print("-" * (12 * len(header)))
        # Rows: clf × n_train
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

    # Collect all configs to run
    if cfg.sweep:
        sweep_cfgs = load_sweep_cfgs(cfg.sweep)
        logger.info("Loaded %d configs from %s", len(sweep_cfgs), cfg.sweep)
        cfgs = [dataclasses.replace(cfg, sweep="", **s) for s in sweep_cfgs]
    else:
        cfgs = [cfg]

    # Dry run: show table and what would run
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

    # Filter to configs that should actually run
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

        # Batch submit all jobs (workers write to parquet themselves)
        with executor.batch():
            jobs = [executor.submit(worker_fn, c) for c in to_run]

        for job, c in zip(jobs, to_run):
            logger.info("Submitted job %s for %s", job.job_id, make_exp_key(c))
    else:
        # Run locally (sequentially)
        for c in to_run:
            worker_fn(c)


@jaxtyping.jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass
class ExtractedFeatures:
    """Features extracted from a dataset."""

    features: Float[Tensor, "n d"]
    labels: Float[Tensor, "n c"]
    ids: list[str]

    def __post_init__(self):
        assert len(self.ids) == len(self.features), (
            f"{len(self.ids)} != {len(self.features)}"
        )


@jaxtyping.jaxtyped(typechecker=beartype.beartype)
@torch.inference_mode()
def extract_features(
    backbone, transform, dataset, device: str, batch_size: int, n_classes: int
) -> ExtractedFeatures:
    """Extract features from a dataset using the backbone."""
    all_features = []
    all_labels = []
    all_ids = []

    batches = range(0, len(dataset), batch_size)
    for i in helpers.progress(batches, every=10, desc="extract"):
        batch = dataset[i : i + batch_size]

        specs = []
        for audio in batch["audio"]:
            spec = transform(audio["array"])
            specs.append(spec)

        specs = torch.stack(specs).to(device)
        encoded = backbone.audio_encode(specs)
        all_features.append(encoded.features.cpu())

        batch_labels = batch[BIRDSET_LABEL_COL]
        labels = torch.zeros(len(batch_labels), n_classes, dtype=torch.float32)
        for j, class_indices in enumerate(batch_labels):
            for c in class_indices:
                labels[j, c] = 1.0
        all_labels.append(labels)

        all_ids.extend(batch["filepath"])

    return ExtractedFeatures(
        features=torch.cat(all_features, dim=0),
        labels=torch.cat(all_labels, dim=0),
        ids=all_ids,
    )


@jaxtyping.jaxtyped(typechecker=beartype.beartype)
@torch.inference_mode()
def extract_features_from_list(
    backbone,
    transform,
    examples: list[dict],
    device: str,
    batch_size: int,
    n_classes: int,
) -> ExtractedFeatures:
    """Extract features from a list of example dicts (for collected train data)."""
    all_features = []
    all_labels = []
    all_ids = []

    batches = range(0, len(examples), batch_size)
    for i in helpers.progress(batches, every=10, desc="extract"):
        batch = examples[i : i + batch_size]

        specs = []
        for ex in batch:
            audio = ex["audio"]
            spec = transform(audio["array"])
            specs.append(spec)

        specs = torch.stack(specs).to(device)
        encoded = backbone.audio_encode(specs)
        all_features.append(encoded.features.cpu())

        labels = torch.zeros(len(batch), n_classes, dtype=torch.float32)
        for j, ex in enumerate(batch):
            for c in ex[BIRDSET_LABEL_COL]:
                labels[j, c] = 1.0
        all_labels.append(labels)

        all_ids.extend(ex["filepath"] for ex in batch)

    return ExtractedFeatures(
        features=torch.cat(all_features, dim=0),
        labels=torch.cat(all_labels, dim=0),
        ids=all_ids,
    )


@jaxtyping.jaxtyped(typechecker=beartype.beartype)
@torch.inference_mode()
def extract_features_streaming(
    backbone, transform, dataset, device: str, batch_size: int, n_classes: int
) -> ExtractedFeatures:
    """Extract features from a streaming dataset (for large test sets)."""
    all_features = []
    all_labels = []
    all_ids = []
    n_batches = 0

    batch = []
    for ex in dataset:
        batch.append(ex)
        if len(batch) >= batch_size:
            _process_batch(
                backbone,
                transform,
                batch,
                n_classes,
                device,
                all_features,
                all_labels,
                all_ids,
            )
            batch = []
            n_batches += 1
            if n_batches % 10 == 0:
                logger.info("Processed %d batches", n_batches)

    # Process remaining examples
    if batch:
        _process_batch(
            backbone,
            transform,
            batch,
            n_classes,
            device,
            all_features,
            all_labels,
            all_ids,
        )

    return ExtractedFeatures(
        features=torch.cat(all_features, dim=0),
        labels=torch.cat(all_labels, dim=0),
        ids=all_ids,
    )


def _process_batch(
    backbone, transform, batch, n_classes, device, all_features, all_labels, all_ids
):
    """Helper to process a batch of examples."""
    specs = []
    for ex in batch:
        audio = ex["audio"]
        spec = transform(audio["array"])
        specs.append(spec)

    specs = torch.stack(specs).to(device)
    encoded = backbone.audio_encode(specs)
    all_features.append(encoded.features.cpu())

    labels = torch.zeros(len(batch), n_classes, dtype=torch.float32)
    for j, ex in enumerate(batch):
        for c in ex[BIRDSET_LABEL_COL]:
            labels[j, c] = 1.0
    all_labels.append(labels)

    all_ids.extend(ex["filepath"] for ex in batch)


@jaxtyping.jaxtyped(typechecker=beartype.beartype)
def train_linear_probe(
    features: Float[Tensor, "n d"],
    labels: Float[Tensor, "n c"],
    n_classes: int,
    cfg: Config,
):
    """Train a linear probe with asymmetric loss using AdamW + cosine scheduler.

    Follows Bird-MAE's linear probing setup: mini-batch training with asymmetric loss designed for imbalanced multi-label classification.
    """
    dim = features.shape[1]
    probe = torch.nn.Sequential(
        torch.nn.LayerNorm(dim), torch.nn.Linear(dim, n_classes)
    )
    probe = probe.to(cfg.device)

    # Mini-batch training with DataLoader
    dataset = torch.utils.data.TensorDataset(features, labels)
    batch_size = min(cfg.batch_size, len(features))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = AsymmetricLoss()

    for epoch in helpers.progress(range(cfg.epochs), every=cfg.log_every, desc="train"):
        probe.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            logits = probe(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm=cfg.grad_clip)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % cfg.log_every == 0:
            avg_loss = total_loss / len(loader)
            logger.info("loss=%.4f lr=%.2e", avg_loss, scheduler.get_last_lr()[0])

    return probe


@jaxtyping.jaxtyped(typechecker=beartype.beartype)
def train_mlp_probe(
    features: Float[Tensor, "n d"],
    labels: Float[Tensor, "n c"],
    n_classes: int,
    cfg: Config,
):
    """Train a 2-layer MLP probe (like DINOv3 task heads).

    Architecture: LayerNorm -> Linear -> GELU -> Dropout -> Linear
    """
    dim = features.shape[1]
    hidden_dim = dim * 2
    probe = torch.nn.Sequential(
        torch.nn.LayerNorm(dim),
        torch.nn.Linear(dim, hidden_dim),
        torch.nn.GELU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(hidden_dim, n_classes),
    )
    probe = probe.to(cfg.device)

    dataset = torch.utils.data.TensorDataset(features, labels)
    batch_size = min(cfg.batch_size, len(features))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = AsymmetricLoss()

    for epoch in helpers.progress(range(cfg.epochs), every=cfg.log_every, desc="train"):
        probe.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            logits = probe(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm=cfg.grad_clip)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % cfg.log_every == 0:
            avg_loss = total_loss / len(loader)
            logger.info("loss=%.4f lr=%.2e", avg_loss, scheduler.get_last_lr()[0])

    return probe


@jaxtyping.jaxtyped(typechecker=beartype.beartype)
class CentroidClassifier(torch.nn.Module):
    """Multi-label nearest-centroid classifier using cosine similarity.

    Like SimpleShot but adapted for multi-label: computes per-class centroids and classifies by cosine similarity independently for each class.
    """

    def __init__(self, centroids: Float[Tensor, "c d"]):
        super().__init__()
        self.register_buffer("centroids", centroids)

    def forward(self, x: Float[Tensor, "b d"]) -> Float[Tensor, "b c"]:
        # L2 normalize
        x_norm = x / x.norm(dim=1, keepdim=True).clamp(min=1e-8)
        c_norm = self.centroids / self.centroids.norm(dim=1, keepdim=True).clamp(
            min=1e-8
        )
        # Cosine similarity as logits (scale by temperature)
        return x_norm @ c_norm.T * 10.0  # temperature=0.1 -> scale by 10


@jaxtyping.jaxtyped(typechecker=beartype.beartype)
def train_centroid_probe(
    features: Float[Tensor, "n d"],
    labels: Float[Tensor, "n c"],
    n_classes: int,
    cfg: Config,
) -> CentroidClassifier:
    """Build a centroid classifier by computing per-class means.

    For multi-label, each class centroid is the mean of all examples containing that class.
    """
    dim = features.shape[1]
    centroids = torch.zeros(n_classes, dim)

    for c in range(n_classes):
        mask = labels[:, c] > 0
        if mask.sum() > 0:
            centroids[c] = features[mask].mean(dim=0)
        else:
            logger.warning("Class %d has no positive examples, using zero centroid.", c)

    return CentroidClassifier(centroids).to(cfg.device)


@jaxtyping.jaxtyped(typechecker=beartype.beartype)
@torch.inference_mode()
def evaluate(
    probe, test: ExtractedFeatures, device: str
) -> tuple[float, int, list[reporting.Prediction]]:
    """Evaluate and return cmAP along with per-example predictions."""
    probe.eval()
    features = test.features.to(device)
    logits = probe(features)
    probs = torch.sigmoid(logits).cpu().numpy()
    labels = test.labels.numpy()

    # cmAP: average AP across classes
    aps = []
    for c in range(labels.shape[1]):
        if labels[:, c].sum() > 0:
            ap = sklearn.metrics.average_precision_score(labels[:, c], probs[:, c])
            aps.append(ap)

    cmap = float(np.mean(aps)) if aps else 0.0

    # Build predictions list
    predictions = []
    for i, example_id in enumerate(test.ids):
        y_true = [int(c) for c in range(labels.shape[1]) if labels[i, c] > 0]
        y_pred = [int(c) for c in range(probs.shape[1]) if probs[i, c] > 0.5]
        y_score = [float(probs[i, c]) for c in range(probs.shape[1])]
        predictions.append(reporting.Prediction(example_id, y_true, y_pred, y_score))

    return cmap, len(aps), predictions
