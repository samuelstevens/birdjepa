"""Benchmark JIT compilation time for the training step."""

import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

import birdjepa.nn.transformer
import birdjepa.nn.objectives


def main():
    print(f"JAX devices: {jax.devices()}")

    # Create model matching the sweep config
    cfg = birdjepa.nn.transformer.Transformer(
        input_h=512,
        input_w=128,
        patch_h=16,
        patch_w=16,
        embed_dim=384,
        depth=12,
        n_heads=6,
    )

    key = jr.key(42)
    model_key, key = jr.split(key)

    print("Creating encoder...")
    t0 = time.time()
    encoder = birdjepa.nn.transformer.TransformerModel(cfg, key=model_key)
    print(f"Encoder created in {time.time() - t0:.2f}s")

    # Check jaxpr complexity for a single block
    print("\nAnalyzing model complexity...")
    block = encoder.blocks
    arrays, static = eqx.partition(block, eqx.is_array)
    single_arrays = jax.tree.map(lambda x: x[0], arrays)
    single_block = eqx.combine(single_arrays, static)

    # Dummy input matching patch embedding output
    n_patches = (512 // 16) * (128 // 16)  # 32 * 8 = 256
    dummy = jnp.zeros((1, n_patches + 1, 384))  # +1 for CLS token

    jaxpr = jax.make_jaxpr(lambda x: single_block(x, key=jr.key(0)))(dummy)
    print(f"Single block jaxpr equations: {len(jaxpr.eqns)}")

    # Create objective
    objective = birdjepa.nn.objectives.Supervised(cfg, n_classes=9736, key=key)

    # Create dummy batch - use smaller batch for CPU testing
    batch_size = 32 if jax.devices()[0].platform == "cpu" else 1024
    batch = {
        "data": jnp.zeros((batch_size, 512, 128)),
        "target": jnp.zeros((batch_size,), dtype=jnp.int32),
    }

    # Define a simple forward pass matching pretrain.py
    @eqx.filter_jit
    def forward(encoder, objective, batch, key):
        losses, emb, targets = objective(batch, encoder, key=key)
        return losses["ce"].mean()

    print(f"\nBenchmarking JIT compilation (batch_size={batch_size})...")
    print("This simulates what happens on first training step.")

    t0 = time.time()
    loss = forward(encoder, objective, batch, key)
    loss.block_until_ready()
    t1 = time.time()
    print(f"First call (JIT compile + execute): {t1 - t0:.2f}s")

    # Second call should be fast
    t0 = time.time()
    loss = forward(encoder, objective, batch, key)
    loss.block_until_ready()
    t1 = time.time()
    print(f"Second call (cached): {t1 - t0:.4f}s")

    print(f"\nLoss: {float(loss):.4f}")


if __name__ == "__main__":
    main()
