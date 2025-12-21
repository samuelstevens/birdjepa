"""
SQLite-based experiment tracking for benchmark results.

Provides deduplication (skip already-run experiments) and worker coordination for parallel execution.
"""

import dataclasses
import json
import logging
import os
import pathlib
import random
import socket
import sqlite3
import subprocess
import sys
import time

import beartype

logger = logging.getLogger(__name__)

schema_fpath = pathlib.Path(__file__).parent / "schema.sql"

max_retries = 10
max_backoff_s = 30.0


@beartype.beartype
def get_db(report_to: str | pathlib.Path) -> sqlite3.Connection:
    """Get a connection to the reports database."""
    report_to = pathlib.Path(os.path.expandvars(report_to))
    report_to.mkdir(parents=True, exist_ok=True)

    db_fpath = report_to / "benchmark.sqlite"
    db = sqlite3.connect(db_fpath, autocommit=True)

    with open(schema_fpath) as fd:
        db.executescript(fd.read())
    db.autocommit = False

    return db


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
def already_ran(db: sqlite3.Connection, key: ExpKey) -> bool:
    """Check if an experiment has already been run."""
    query = """
    SELECT COUNT(*) FROM experiments
    WHERE task_name = ? AND model_org = ? AND model_ckpt = ? AND clf = ? AND n_train = ?
    """
    (count,) = db.execute(query, key.as_tuple()).fetchone()
    return count > 0


@beartype.beartype
def is_claimed(db: sqlite3.Connection, key: ExpKey) -> bool:
    """Check if a run is already claimed by another process."""
    query = """
    SELECT COUNT(*) FROM runs
    WHERE task_name = ? AND model_org = ? AND model_ckpt = ? AND clf = ? AND n_train = ?
    """
    (count,) = db.execute(query, key.as_tuple()).fetchone()
    return count > 0


@beartype.beartype
def claim_run(db: sqlite3.Connection, key: ExpKey) -> bool:
    """Try to claim an experiment. Returns True if claimed, False if already taken."""
    stmt = """
    INSERT OR IGNORE INTO runs (task_name, model_org, model_ckpt, clf, n_train, pid, posix)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    values = (*key.as_tuple(), os.getpid(), time.time())

    try:
        cur = db.execute(stmt, values)
        db.commit()
        claimed = cur.rowcount == 1
        if claimed:
            logger.info("Claimed %s.", key)
        return claimed
    except Exception:
        db.rollback()
        raise


@beartype.beartype
def release_run(db: sqlite3.Connection, key: ExpKey) -> None:
    """Release a claimed run so others may claim it."""
    stmt = """
    DELETE FROM runs
    WHERE task_name = ? AND model_org = ? AND model_ckpt = ? AND clf = ? AND n_train = ?
    """
    try:
        db.execute(stmt, key.as_tuple())
        db.commit()
        logger.info("Released %s.", key)
    except Exception:
        db.rollback()
        raise


@beartype.beartype
def clear_stale_claims(db: sqlite3.Connection, *, max_age_hours: int = 72) -> int:
    """Delete claims older than max_age_hours. Returns number deleted."""
    assert max_age_hours > 0
    cutoff = time.time() - max_age_hours * 3600
    try:
        cur = db.execute("DELETE FROM runs WHERE posix < ?", (cutoff,))
        db.commit()
        return cur.rowcount
    except Exception:
        db.rollback()
        raise


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

    def write(self, db: sqlite3.Connection) -> None:
        """Save the report to the database with exponential backoff for lock contention."""
        exp_stmt = """
        INSERT INTO experiments
        (task_name, model_org, model_ckpt, clf, n_train, cmap, n_classes,
         exp_cfg, argv, git_commit, posix, gpu_name, hostname)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        pred_stmt = """
        INSERT INTO predictions
        (example_id, y_true, y_pred, y_score, experiment_id)
        VALUES (?, ?, ?, ?, ?)
        """

        exp_values = (
            *self.key.as_tuple(),
            self.cmap,
            self.n_classes,
            json.dumps(self.exp_cfg),
            json.dumps(self.argv),
            self.git_commit,
            self.posix,
            self.gpu_name,
            self.hostname,
        )
        pred_values = [
            (
                p.example_id,
                json.dumps(p.y_true),
                json.dumps(p.y_pred),
                json.dumps(p.y_score),
                None,  # experiment_id filled in after insert
            )
            for p in self.predictions
        ]

        for attempt in range(max_retries):
            try:
                cursor = db.cursor()
                cursor.execute(exp_stmt, exp_values)
                exp_id = cursor.lastrowid
                # Update pred_values with actual experiment_id
                pred_values_with_id = [(*pv[:-1], exp_id) for pv in pred_values]
                cursor.executemany(pred_stmt, pred_values_with_id)
                db.commit()
                logger.info("Saved report for %s (cmAP=%.4f).", self.key, self.cmap)
                return
            except sqlite3.OperationalError as err:
                db.rollback()
                if "database is locked" not in str(err):
                    raise
                if attempt == max_retries - 1:
                    logger.error(
                        "Failed to write report for %s after %d attempts: %s",
                        self.key,
                        max_retries,
                        err,
                    )
                    raise
                # Exponential backoff with jitter: 2^attempt * random(0.5, 1.5), capped
                delay = min(max_backoff_s, (2**attempt) * random.uniform(0.5, 1.5))
                logger.warning(
                    "Database locked for %s, retrying in %.1fs (attempt %d/%d)",
                    self.key,
                    delay,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(delay)
            except sqlite3.Error as err:
                db.rollback()
                logger.error("Failed to write report for %s: %s", self.key, err)
                raise
