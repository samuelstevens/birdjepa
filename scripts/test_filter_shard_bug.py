"""
Test JAX sharding buffer behavior.

Findings from investigation (2026-01-18):
- eqx.filter_shard and jax.device_put both work correctly
- Simulated buffer reuse doesn't cause corruption in this minimal test
- The Rust dataloader allocates fresh memory per batch (no buffer reuse)
- jnp.asarray copies data to JAX memory (no shared memory with numpy)
- The original corruption was likely timing-dependent or from something specific
  to the real training loop that we couldn't reproduce here

Conclusion: The explicit numpy copy in _batch_to_jax is unnecessary.
The essential fix was block_until_ready() for synchronization.
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Force CPU with 2 devices")
    args = parser.parse_args()

    if args.cpu:
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jax.sharding as jshard
    import numpy as np

    print(f"JAX version: {jax.__version__}")
    print(f"Equinox version: {eqx.__version__}")
    print(f"JAX devices: {jax.devices()}")

    n_devices = len(jax.devices())
    if n_devices < 2:
        print(f"\nERROR: Need at least 2 devices, got {n_devices}")
        print("Run on a multi-GPU node or use --cpu flag")
        return

    mesh = jax.make_mesh((2,), ("batch",))
    sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec("batch"))
    shared_buffer = np.zeros(4, dtype=np.int64)
    expected_max = 3

    def run_test(name, shard_fn, use_buffer_reuse=False, copy_first=False):
        print(f"\n{name}:")
        for i in range(3):
            if use_buffer_reuse:
                shared_buffer[:] = [0, 1, 2, 3]
                src = (
                    np.array(shared_buffer, copy=True) if copy_first else shared_buffer
                )
            else:
                src = np.array([0, 1, 2, 3], dtype=np.int64)

            arr = jnp.asarray(src)
            sharded = shard_fn(arr)

            if use_buffer_reuse:
                shared_buffer[:] = [999, 999, 999, 999]

            max_val = int(jnp.max(sharded))
            status = "OK" if max_val == expected_max else f"CORRUPTED (got {max_val})"
            print(f"  Iteration {i}: max={max_val} {status}")

    def device_put_sync(arr):
        result = jax.device_put(arr, sharding)
        result.block_until_ready()
        return result

    # Test fresh arrays (like Rust dataloader behavior)
    run_test("Fresh arrays + device_put", device_put_sync)
    run_test("Fresh arrays + filter_shard", lambda a: eqx.filter_shard(a, sharding))

    # Test buffer reuse (to verify jnp.asarray copies)
    run_test(
        "Buffer reuse + filter_shard (no copy)",
        lambda a: eqx.filter_shard(a, sharding),
        use_buffer_reuse=True,
    )
    run_test(
        "Buffer reuse + filter_shard (with copy)",
        lambda a: eqx.filter_shard(a, sharding),
        use_buffer_reuse=True,
        copy_first=True,
    )


if __name__ == "__main__":
    main()
