"""Useful helpers that don't fit anywhere else."""

import collections.abc
import logging
import time

import jax
import jax.numpy as jnp


def jax_has_gpu() -> bool:
    """Check if JAX has GPU available.

    From https://github.com/jax-ml/jax/issues/971
    """
    try:
        _ = jax.device_put(jnp.ones(1), device=jax.devices("gpu")[0])
        return True
    except Exception:
        return False


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
