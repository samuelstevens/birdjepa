"""Scan BirdSet audio with Grain and log decode errors."""

import dataclasses
import json
import logging
import pathlib
import time
import typing as tp

import beartype
import grain.python as grain
import numpy as np
import tyro

import birdjepa.data
import birdjepa.helpers

logger = logging.getLogger(__name__)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class ScanConfig:
    subset: tp.Literal["XCL", "XCM"] = "XCL"
    split: str = "train"
    batch_size: int = 1
    n_workers: int = 0
    seed: int = 42
    truncate: tp.Literal["random", "start", "end"] = "start"
    n_samples: int | None = None
    log_every: int = 1000
    errors_fpath: pathlib.Path | None = None


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class CheckConfig:
    subset: tp.Literal["XCL", "XCM"] = "XCL"
    split: str = "train"
    seed: int = 42
    truncate: tp.Literal["random", "start", "end"] = "start"
    indices: list[int] = dataclasses.field(default_factory=list)
    retries: int = 3
    sleep_s: float = 0.1
    errors_fpath: pathlib.Path | None = None


class SafeDataset(grain.RandomAccessDataSource):
    """Wrap a dataset to catch and report per-sample errors."""

    def __init__(self, ds: birdjepa.data.Dataset):
        self.ds = ds
        self.n_classes = getattr(ds, "n_classes", 0)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, record_key: tp.SupportsIndex) -> dict:
        idx = int(record_key)
        try:
            item = self.ds[idx]
        except Exception as err:
            return {
                "ok": False,
                "error": repr(err),
                "index": idx,
            }

        result = dict(item)
        result["ok"] = True
        result["error"] = ""
        return result


class SizedIter:
    """Wrap an iterator with a fixed length for progress logging."""

    def __init__(self, it, total: int):
        self.it = it
        self.total = total

    def __iter__(self):
        yield from self.it

    def __len__(self) -> int:
        return self.total


@beartype.beartype
def _open_errors_fpath(errors_fpath: pathlib.Path | None):
    if errors_fpath is None:
        return None

    errors_fpath.parent.mkdir(parents=True, exist_ok=True)
    fd = errors_fpath.open("w")
    return fd


@beartype.beartype
def _log_error(fd, *, subset: str, split: str, index: int, error: str):
    logger.error("error index=%d error=%s", index, error)
    if fd is None:
        return

    record = {
        "subset": subset,
        "split": split,
        "index": index,
        "error": error,
    }
    fd.write(json.dumps(record) + "\n")
    fd.flush()


@beartype.beartype
def _make_dataset(cfg: ScanConfig | CheckConfig) -> birdjepa.data.XenoCantoDataset:
    data_cfg = birdjepa.data.XenoCanto(
        subset=cfg.subset,
        split=cfg.split,
        seed=cfg.seed,
        truncate=cfg.truncate,
        augmentations=[],
    )
    if isinstance(cfg, ScanConfig):
        data_cfg = dataclasses.replace(data_cfg, n_samples=cfg.n_samples)
    return birdjepa.data.XenoCantoDataset(data_cfg)


@beartype.beartype
def scan(cfg: tp.Annotated[ScanConfig, tyro.conf.arg(name="")]):
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s"
    )

    msg = "batch_size must be 1 to safely record per-sample errors"
    assert cfg.batch_size == 1, msg
    assert cfg.n_workers == 0, "n_workers must be 0 for reliable error capture"

    ds = _make_dataset(cfg)
    safe_ds = SafeDataset(ds)

    loader = birdjepa.data.make_dataloader(
        safe_ds,
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        n_workers=cfg.n_workers,
        shuffle=False,
        drop_last=False,
    )

    n_total = 0
    n_errors = 0
    fd = _open_errors_fpath(cfg.errors_fpath)
    try:
        sized_loader = SizedIter(loader, len(safe_ds))
        for batch in birdjepa.helpers.progress(
            sized_loader, every=cfg.log_every, desc="scan"
        ):
            n_total += 1
            ok_arr = batch.get("ok")
            if ok_arr is None:
                continue

            ok = bool(np.asarray(ok_arr)[0])
            if ok:
                if n_total % cfg.log_every == 0:
                    logger.info("scanned %d samples (errors=%d)", n_total, n_errors)
                continue

            index = int(np.asarray(batch["index"])[0])
            error = str(np.asarray(batch["error"])[0])
            _log_error(fd, subset=cfg.subset, split=cfg.split, index=index, error=error)
            n_errors += 1
    finally:
        if fd is not None:
            fd.close()

    logger.info("done: scanned=%d errors=%d", n_total, n_errors)


@beartype.beartype
def check_indices(cfg: tp.Annotated[CheckConfig, tyro.conf.arg(name="")]):
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s"
    )
    assert cfg.indices, "indices must be non-empty"
    assert cfg.retries > 0, "retries must be > 0"
    assert cfg.sleep_s >= 0, "sleep_s must be >= 0"

    ds = _make_dataset(cfg)
    fd = _open_errors_fpath(cfg.errors_fpath)
    try:
        for idx in cfg.indices:
            errors = []
            for _ in range(cfg.retries):
                try:
                    _ = ds[idx]
                    err = ""
                except Exception as exc:
                    err = repr(exc)
                errors.append(err)
                if cfg.sleep_s:
                    time.sleep(cfg.sleep_s)

            logger.info("index=%d errors=%s", idx, errors)
            if any(errors):
                _log_error(
                    fd,
                    subset=cfg.subset,
                    split=cfg.split,
                    index=idx,
                    error=" | ".join(errors),
                )
    finally:
        if fd is not None:
            fd.close()


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({
        "scan": scan,
        "check": check_indices,
    })
