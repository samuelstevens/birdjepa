# Plan: JAX-based Benchmark for BirdSet

## Goal

Evaluate trained JAX checkpoints on BirdSet benchmark tasks (POW, PER, NES, UHH, HSN, NBP, SSW, SNE) using a pure JAX pipeline.

## Rationale

The original plan used PyTorch for benchmarking with JAX↔PyTorch conversion. A pure JAX approach is cleaner:
- **No conversion overhead** - checkpoints load directly into JAX models
- **No structure matching issues** - Orbax restoration works with partial restore
- **Consistent with training** - same model code, transforms, and tooling
- **GPU acceleration** - can use JAX GPU for feature extraction

## Current State

**Checkpoints available:**
- `dpvgra4g` at step 4000+ (running now, LR=3e-4)
- `mt5n4o8g`, `31vzhhgn`, etc. from current sweep
- Format: Orbax OCDBT (JAX checkpointing)
- Location: `/fs/ess/PAS2136/samuelstevens/birdjepa/{wandb_run_id}/`

**Checkpoint structure:**
- `state.encoder` - JAX TransformerModel weights
- `state.probe` - Linear probe weights (for online eval)
- `state.objective` - Objective function head
- `state.opt_state` - Optimizer state
- `metadata` - `{"step": N}`

**IMPORTANT**: Current checkpoints have inconsistent dimensions due to a bug:
- patch_embed, cls_tokens, norm: 192-dim
- pos_embed_hw, blocks: 384-dim
This must be fixed in pretraining before benchmarking.

## Architecture

```
src/birdjepa/
├── benchmark/              # Existing PyTorch benchmark (keep for Bird-MAE)
│   ├── __init__.py
│   ├── registry.py
│   └── reporting.py
└── jax_benchmark/          # New JAX benchmark
    ├── __init__.py         # CLI, Config, main logic
    ├── probe.py            # Linear probe training with optax
    └── eval.py             # cmAP evaluation
```

## Implementation Tasks

### 1. Fix model dimension bug in pretrain.py

The TransformerModel is being created with inconsistent embed_dim. Need to debug why:
- Config says embed_dim=384
- Some layers use 384, others use 192

This is blocking - can't benchmark until checkpoints are consistent.

### 2. Create jax_benchmark module

**File:** `src/birdjepa/jax_benchmark/__init__.py`

```python
@dataclasses.dataclass
class Config:
    ckpt_path: str
    """Path to checkpoint directory."""
    task: str = "pow"
    """BirdSet task."""
    step: int | None = None
    """Checkpoint step. None = latest."""
    classifier: str = "linear"
    """linear, mlp, or centroid."""
    # Training params
    lr: float = 4e-4
    epochs: int = 30
    batch_size: int = 64
    # Execution
    device: str = "cuda"
    report_to: str = "./results"

def run_benchmark(cfg: Config) -> float:
    """Run benchmark, return cmAP."""
    # 1. Load checkpoint (encoder only, partial restore)
    encoder = load_encoder(cfg.ckpt_path, cfg.step)

    # 2. Load BirdSet data (HuggingFace -> JAX arrays)
    train_data, test_data = load_birdset(cfg.task)

    # 3. Extract features (JAX inference)
    train_features = extract_features(encoder, train_data)
    test_features = extract_features(encoder, test_data)

    # 4. Train probe (optax + equinox)
    probe = train_probe(train_features, train_labels, cfg)

    # 5. Evaluate (sklearn for cmAP)
    cmap = evaluate(probe, test_features, test_labels)

    return cmap
```

### 3. Checkpoint loading with partial restore

**Key insight**: Use Orbax's `restore_args_from_target` or provide minimal structure.

```python
def load_encoder(ckpt_path: str, step: int | None = None):
    import orbax.checkpoint as ocp

    mngr = ocp.CheckpointManager(ckpt_path)
    load_step = step or mngr.latest_step()

    # Read checkpoint metadata to get model config
    # (Once we save it - for now, infer from sweep)
    model_cfg = infer_model_config(ckpt_path)

    # Create encoder structure
    encoder = transformer.TransformerModel(model_cfg, key=jr.key(0))

    # Restore with partial structure (only encoder)
    restored = mngr.restore(
        load_step,
        args=ocp.args.Composite(
            state=ocp.args.PyTreeRestore(
                {"encoder": encoder},
                restore_args=ocp.tree.from_flat_dict(
                    {"encoder": ocp.RestoreArgs(restore_type=ocp.RestoreType.JAX_ARRAY)}
                ),
            ),
        ),
    )
    return restored["state"]["encoder"]
```

### 4. Feature extraction

```python
@jax.jit
def extract_features_batch(encoder, specs, cfg):
    """Extract features from batch of spectrograms."""
    # Patchify
    x_bnk, grid = transformer.patchify(specs, cfg)

    # Run encoder
    out = encoder(x_bnk, grid=grid, key=jr.key(0))

    # Pool: mean over patch tokens
    return out["patches"].mean(axis=1)  # [B, D]

def extract_features(encoder, dataset, batch_size=64):
    """Extract features from entire dataset."""
    all_features = []
    for batch in dataset.batch(batch_size):
        features = extract_features_batch(encoder, batch["spec"], cfg)
        all_features.append(features)
    return jnp.concatenate(all_features, axis=0)
```

### 5. Linear probe training

```python
def train_linear_probe(features, labels, n_classes, cfg):
    """Train linear probe with asymmetric loss."""
    import equinox as eqx
    import optax

    # Model: LayerNorm + Linear
    probe = eqx.nn.Sequential([
        eqx.nn.LayerNorm(features.shape[1]),
        eqx.nn.Linear(features.shape[1], n_classes, key=jr.key(0)),
    ])

    # Optimizer
    opt = optax.adamw(cfg.lr, weight_decay=cfg.weight_decay)
    opt_state = opt.init(eqx.filter(probe, eqx.is_inexact_array))

    # Training loop
    for epoch in range(cfg.epochs):
        for x, y in batched(features, labels, cfg.batch_size):
            loss, grads = jax.value_and_grad(asymmetric_loss)(probe, x, y)
            updates, opt_state = opt.update(grads, opt_state, probe)
            probe = eqx.apply_updates(probe, updates)

    return probe
```

## Data Pipeline

Use HuggingFace datasets with streaming, convert to JAX at boundary:

```python
def load_birdset(task: str):
    ds = datasets.load_dataset("samuelstevens/BirdSet", task.upper(), streaming=True)

    # Collect train (small) and stream test
    train_examples = list(ds["train"])
    test_stream = ds["test_5s"]

    # Convert audio to spectrograms using bird_mae.transform
    # Convert to JAX arrays
    ...
```

## CLI Integration

Add to `launch.py`:

```python
@app.command()
def jax_benchmark(
    ckpt_path: str,
    task: str = "pow",
    step: int | None = None,
    ...
):
    """Benchmark JAX checkpoint on BirdSet."""
    from birdjepa import jax_benchmark
    cfg = jax_benchmark.Config(ckpt_path=ckpt_path, task=task, step=step, ...)
    cmap = jax_benchmark.run_benchmark(cfg)
    print(f"cmAP: {cmap:.4f}")
```

## Open Questions

1. **Dimension bug**: Why do checkpoints have 192 for some layers, 384 for others?
2. **Partial restore**: Does Orbax support restoring only encoder from state?
3. **GPU memory**: Can we fit feature extraction + probe training on single GPU?

## Testing Plan

1. Fix dimension bug, verify consistent checkpoint shapes
2. Test checkpoint loading with partial restore
3. Test feature extraction on small batch
4. Test probe training on extracted features
5. Run full benchmark on POW task
6. Compare results with Bird-MAE baseline
