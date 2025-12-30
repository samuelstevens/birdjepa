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


class progress:
    """Wraps an iterable with logging instead of tqdm (works better with log files)."""

    def __init__(self, it, *, every: int = 10, desc: str = "progress"):
        self.it = it
        self.every = every
        self.logger = logging.getLogger(desc)

    def __iter__(self):
        start = time.time()
        for i, obj in enumerate(self.it):
            yield obj
            if (i + 1) % self.every == 0:
                elapsed = time.time() - start
                rate = (i + 1) / (elapsed / 60)
                if isinstance(self.it, collections.abc.Sized):
                    remaining = (len(self) - (i + 1)) / rate
                    self.logger.info(
                        "%d/%d (%.1f%%) | %.1f it/m | ~%.1fm left",
                        i + 1,
                        len(self),
                        (i + 1) / len(self) * 100,
                        rate,
                        remaining,
                    )
                else:
                    self.logger.info("%d/? | %.1f it/m", i + 1, rate)

    def __len__(self) -> int:
        return len(self.it)
