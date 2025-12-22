"""
Parquet-based experiment tracking for benchmark results.

Each experiment is stored as a separate parquet file, queryable via DuckDB.
"""

import dataclasses
import datetime
import json
import logging
import os
import pathlib
import socket
import subprocess
import sys
import time

import beartype
import duckdb
import polars as pl

logger = logging.getLogger(__name__)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class ExpKey:
    """Unique key for an experiment."""

    task_name: str
    model_org: str
    model_ckpt: str
    clf: str
    n_train: int

    def as_tuple(self) -> tuple[str, str, str, str, int]:
        return (self.task_name, self.model_org, self.model_ckpt, self.clf, self.n_train)


@beartype.beartype
def already_ran(report_to: str | pathlib.Path, key: ExpKey) -> bool:
    """Check if experiment exists in any parquet file."""
    report_to = pathlib.Path(os.path.expandvars(report_to))
    raw_dir = report_to / "raw"
    if not raw_dir.exists():
        return False
    pattern = str(raw_dir / "*.parquet")
    query = """
    SELECT COUNT(*) FROM read_parquet(?)
    WHERE task_name = ? AND model_org = ? AND model_ckpt = ? AND clf = ? AND n_train = ?
    """
    try:
        result = duckdb.execute(query, [pattern, *key.as_tuple()]).fetchone()
        return result[0] > 0
    except duckdb.IOException:
        return False  # No parquet files yet


def get_git_hash() -> str:
    """Returns COMMIT or COMMIT-dirty depending on working directory state."""
    try:
        commit = (
            subprocess
            .check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        )
        if status.stdout.strip():
            return f"{commit}-dirty"
        return commit
    except subprocess.CalledProcessError:
        return "unknown"


@beartype.beartype
def get_gpu_name() -> str:
    """Returns the name of the first GPU, or empty string if none."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).name
    except ImportError:
        pass
    return ""


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Prediction:
    """An individual test prediction."""

    example_id: str
    y_true: list[int]
    y_pred: list[int]
    y_score: list[float]


@beartype.beartype
@dataclasses.dataclass
class Report:
    """Result of running a benchmark experiment."""

    key: ExpKey
    cmap: float
    n_classes: int
    predictions: list[Prediction]
    exp_cfg: dict

    # Metadata (filled in automatically)
    argv: list[str] = dataclasses.field(default_factory=lambda: sys.argv.copy())
    git_commit: str = dataclasses.field(default_factory=get_git_hash)
    posix: float = dataclasses.field(default_factory=time.time)
    gpu_name: str = dataclasses.field(default_factory=get_gpu_name)
    hostname: str = dataclasses.field(default_factory=socket.gethostname)

    def write(self, report_to: str | pathlib.Path) -> None:
        """Save report as a parquet file."""
        report_to = pathlib.Path(os.path.expandvars(report_to))
        raw_dir = report_to / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fpath = raw_dir / f"{timestamp}.parquet"

        # Build dataframe with experiment metadata + predictions
        data = {
            "task_name": self.key.task_name,
            "model_org": self.key.model_org,
            "model_ckpt": self.key.model_ckpt,
            "clf": self.key.clf,
            "n_train": self.key.n_train,
            "cmap": self.cmap,
            "n_classes": self.n_classes,
            "exp_cfg": json.dumps(self.exp_cfg),
            "argv": json.dumps(self.argv),
            "git_commit": self.git_commit,
            "posix": self.posix,
            "gpu_name": self.gpu_name,
            "hostname": self.hostname,
            # Predictions as lists
            "pred_example_ids": [[p.example_id for p in self.predictions]],
            "pred_y_true": [[p.y_true for p in self.predictions]],
            "pred_y_pred": [[p.y_pred for p in self.predictions]],
            "pred_y_score": [[p.y_score for p in self.predictions]],
        }
        df = pl.DataFrame(data)

        # Atomic write: temp file then rename
        tmp_fpath = fpath.with_suffix(".parquet.tmp")
        df.write_parquet(tmp_fpath)
        tmp_fpath.rename(fpath)

        logger.info(
            "Saved report for %s (cmAP=%.4f) to %s", self.key, self.cmap, fpath.name
        )
