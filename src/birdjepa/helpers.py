"""Useful helpers that don't fit anywhere else."""

import collections.abc
import logging
import time

import beartype
import jax
import jax.numpy as jnp
import jax.sharding as jshard


def jax_has_gpu() -> bool:
    """Check if JAX has GPU available.

    Uses local_devices() to work in multi-process distributed mode.
    """
    try:
        gpus = [d for d in jax.local_devices() if d.platform == "gpu"]
        if not gpus:
            return False
        _ = jax.device_put(jnp.ones(1), device=gpus[0])
        return True
    except Exception:
        return False


@beartype.beartype
def setup_sharding(
    n_devices: int,
) -> tuple[jax.sharding.Mesh, jshard.NamedSharding, jshard.NamedSharding]:
    """Set up JAX sharding for data parallel training.

    Args:
        n_devices: Number of devices to use. Always creates a mesh (even for 1 device).

    Returns:
        (mesh, data_sharding, model_sharding) tuple.
        - data_sharding: Shard along batch dimension
        - model_sharding: Replicate across all devices
    """
    n_devices = max(1, n_devices)
    available = jax.devices()
    assert len(available) >= n_devices, (
        f"Requested {n_devices} devices but only {len(available)} available"
    )
    devices = available[:n_devices]
    mesh = jax.make_mesh((len(devices),), ("batch",))
    data_sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec("batch"))
    model_sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec())
    return mesh, data_sharding, model_sharding


@beartype.beartype
class progress:
    """
    Wraps an iterable with a logger like tqdm but doesn't use any control codes to manipulate a progress bar, which doesn't work well when your output is redirected to a file. Instead, simple logging statements are used, but it includes quality-of-life features like iteration speed and predicted time to finish.

    Args:
        it: Iterable to wrap.
        every: How many iterations between logging progress.
        desc: What to name the logger.
        total: If non-zero, how long the iterable is.
    """

    def __init__(
        self,
        it: collections.abc.Iterable,
        *,
        every: int = 10,
        desc: str = "progress",
        total: int = 0,
    ):
        self.it = it
        self.every = max(every, 1)
        self.logger = logging.getLogger(desc)
        self.total = total

    def __iter__(self):
        start = time.time()

        try:
            total = len(self)
        except TypeError:
            total = None

        for i, obj in enumerate(self.it):
            yield obj

            if (i + 1) % self.every == 0:
                now = time.time()
                duration_s = now - start
                per_min = (i + 1) / (duration_s / 60)

                if total is not None:
                    pred_min = (total - (i + 1)) / per_min
                    self.logger.info(
                        "%d/%d (%.1f%%) | %.1f it/m (expected finish in %.1fm)",
                        i + 1,
                        total,
                        (i + 1) / total * 100,
                        per_min,
                        pred_min,
                    )
                else:
                    self.logger.info("%d/? | %.1f it/m", i + 1, per_min)

    def __len__(self) -> int:
        if self.total > 0:
            return self.total

        # Will throw exception.
        return len(self.it)
