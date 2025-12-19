"""
Benchmark package for evaluating audio backbones on BirdSet tasks.

Tasks: pow, per, nes, uhh, hsn, nbp, ssw, sne
Classifiers: linear, prototypical
"""

import dataclasses
import logging
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

from . import registry
from .. import helpers

logger = logging.getLogger("benchmark")


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

    @jaxtyping.jaxtyped(typechecker=beartype.beartype)
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


@beartype.beartype
def sample_per_class(dataset, n_per_class: int, seed: int = 42):
    """Sample up to n_per_class examples for each class, return their union.

    For multi-label data, a sample can belong to multiple classes. We iterate through each class, find all samples containing that class, and randomly select up to n_per_class of them. The final dataset is the union of all selected indices.
    """
    rng = np.random.default_rng(seed)
    n_classes = len(dataset.features["ebird_code_multilabel"].feature.names)
    selected = set()

    # Build index: for each class, which samples contain it?
    labels = dataset["ebird_code_multilabel"]
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
    """Classifier: linear, prototypical."""
    n_train: int = -1
    """Number of training samples per class. -1 for all."""
    # Training
    lr: float = 1e-3
    """Learning rate for linear probe."""
    epochs: int = 20
    """Training epochs for linear probe."""
    batch_size: int = 64
    """Batch size."""
    device: str = "cuda"
    """Device."""
    # Paths
    report_to: pathlib.Path = pathlib.Path("./results")
    """Directory for results database."""
    # Slurm
    slurm_acct: str = ""
    """Slurm account string. Empty means run locally."""
    slurm_partition: str = ""
    """Slurm partition."""
    n_hours: float = 4.0
    """Slurm job length in hours."""
    log_to: pathlib.Path = pathlib.Path("./logs")
    """Where to log Slurm job stdout/stderr."""


@beartype.beartype
def worker_fn(cfg: Config):
    """Run a single benchmark experiment."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    )
    logger.info(
        "Benchmarking task=%s model=%s/%s classifier=%s n_train=%d",
        cfg.task,
        cfg.model_org,
        cfg.model_ckpt,
        cfg.classifier,
        cfg.n_train,
    )

    # 1. Load backbone
    backbone = registry.load(cfg.model_org, cfg.model_ckpt)
    backbone = backbone.to(cfg.device)
    backbone.eval()
    transform = backbone.make_audio_transform()

    # 2. Load dataset
    ds = datasets.load_dataset("samuelstevens/BirdSet", cfg.task.upper())
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=32_000))

    train_ds = ds["train"]
    test_ds = ds["test_5s"]

    if cfg.n_train > 0:
        train_ds = sample_per_class(train_ds, cfg.n_train)

    # Get number of classes from the dataset
    n_classes = len(train_ds.features["ebird_code_multilabel"].feature.names)
    logger.info(
        "Task %s: %d train, %d test, %d classes",
        cfg.task,
        len(train_ds),
        len(test_ds),
        n_classes,
    )

    # 3. Extract features
    logger.info("Extracting train features...")
    train_features, train_labels = extract_features(
        backbone, transform, train_ds, cfg.device, cfg.batch_size, n_classes
    )

    logger.info("Extracting test features...")
    test_features, test_labels = extract_features(
        backbone, transform, test_ds, cfg.device, cfg.batch_size, n_classes
    )

    # 4. Train classifier
    if cfg.classifier == "linear":
        logger.info("Training linear probe...")
        probe = train_linear_probe(train_features, train_labels, n_classes, cfg)
    else:
        raise NotImplementedError(f"Classifier '{cfg.classifier}' not implemented")

    # 5. Evaluate
    logger.info("Evaluating...")
    metrics = evaluate(probe, test_features, test_labels, cfg.device)
    logger.info("Results: cmAP=%.4f", metrics["cmAP"])

    return metrics


@jaxtyping.jaxtyped(typechecker=beartype.beartype)
@torch.inference_mode()
def extract_features(
    backbone, transform, dataset, device: str, batch_size: int, n_classes: int
) -> tuple[Float[Tensor, "n d"], Float[Tensor, "n c"]]:
    """Extract features from a dataset using the backbone."""
    all_features = []
    all_labels = []

    batches = range(0, len(dataset), batch_size)
    for i in helpers.progress(batches, every=10, desc="extract"):
        batch = dataset[i : i + batch_size]

        # Transform audio to spectrograms
        specs = []
        for audio in batch["audio"]:
            waveform = audio["array"]
            spec = transform(waveform)
            specs.append(spec)

        specs = torch.stack(specs).to(device)
        encoded = backbone.audio_encode(specs)
        all_features.append(encoded.features.cpu())

        # Multi-label: convert list of class indices to binary vector
        batch_labels = batch["ebird_code_multilabel"]
        labels = torch.zeros(len(batch_labels), n_classes, dtype=torch.float32)
        for j, class_indices in enumerate(batch_labels):
            for c in class_indices:
                labels[j, c] = 1.0
        all_labels.append(labels)

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return features, labels


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
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True
    )

    # AdamW with Bird-MAE's hyperparameters
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=4e-4, weight_decay=3e-4, betas=(0.9, 0.95)
    )

    # Cosine annealing scheduler
    n_epochs = 30
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Asymmetric loss for imbalanced multi-label
    criterion = AsymmetricLoss()

    for epoch in range(n_epochs):
        probe.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            logits = probe(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping (Bird-MAE uses 2.0)
            torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / len(loader)
        if (epoch + 1) % 5 == 0:
            lr = scheduler.get_last_lr()[0]
            logger.info("Epoch %d: loss=%.4f lr=%.2e", epoch + 1, avg_loss, lr)

    return probe


@jaxtyping.jaxtyped(typechecker=beartype.beartype)
@torch.inference_mode()
def evaluate(
    probe,
    features: Float[Tensor, "n d"],
    labels: Float[Tensor, "n c"],
    device: str,
) -> dict:
    """Evaluate and compute cmAP (class mean average precision)."""
    probe.eval()
    features = features.to(device)
    logits = probe(features)
    probs = torch.sigmoid(logits).cpu().numpy()
    labels = labels.numpy()

    # cmAP: average AP across classes
    aps = []
    for c in range(labels.shape[1]):
        if labels[:, c].sum() > 0:  # skip classes with no positive samples
            ap = sklearn.metrics.average_precision_score(labels[:, c], probs[:, c])
            aps.append(ap)

    cmAP = np.mean(aps) if aps else 0.0
    return {"cmAP": cmAP, "n_classes_evaluated": len(aps)}
