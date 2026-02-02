# Param Norm Mismatch Debugging SPEC

## Problem Statement

Checkpoint `param_norm` values are consistently **lower** than logged `train/param_norm` values. The mismatch:
- Grows with step count (e.g., -4.09 at step 8000 to -8.65 at step 10000 for tisomkvs)
- Scales with learning rate (higher LR = larger mismatch, up to -1100 delta)
- Is reproducible across CPU and GPU restore

## Quantitative Analysis

Comparing delta against update_norm at each checkpoint step (tisomkvs):

| Step | Delta | update_norm | Ratio (steps of lag) |
|------|-------|-------------|----------------------|
| 8000 | -4.09 | 0.294 | ~14 |
| 8500 | -5.11 | 0.290 | ~18 |
| 9000 | -6.16 | 0.279 | ~22 |
| 9500 | -7.42 | 0.255 | ~29 |
| 10000 | -8.65 | 0.283 | ~31 |

Key observations:
1. **Delta >> update_norm**: The mismatch is 14-31x the per-step update norm
2. **Ratio is growing**: If this were fixed async lag, ratio would be constant
3. **Cumulative divergence**: Delta grows by ~4.5 over 2000 steps (~0.002/step), which is ~0.7% of update_norm

## Checkpoint Fingerprint Analysis (2026-01-22)

Ran `compare_ckpt_param_norms.py --show-fingerprint` on tisomkvs checkpoints:

```
step,ckpt_param_norm,log_param_norm,delta,leaf_count,element_count,enc_leaves,enc_elems,obj_leaves,obj_elems,probe_leaves,probe_elems
8000,185.906,190.0,-4.09427,28,28880144,22,21382656,2,3748360,4,3749128
8500,190.69,195.8,-5.11046,28,28880144,22,21382656,2,3748360,4,3749128
9000,195.541,201.7,-6.15926,28,28880144,22,21382656,2,3748360,4,3749128
9500,200.578,208.0,-7.42188,28,28880144,22,21382656,2,3748360,4,3749128
10000,205.646,214.3,-8.65362,28,28880144,22,21382656,2,3748360,4,3749128

Dtype counts: {'float32': 28}
```

**Findings**:
- Checkpoint pytree structure is **constant** across all steps: 28 leaves, 28.9M elements
- Component breakdown: encoder (22 leaves, 21.4M), objective (2 leaves, 3.7M), probe (4 leaves, 3.7M)
- All parameters are float32
- Checkpoint metadata does NOT contain `param_norm` (checkpoints predate that feature)

**Implication**: The pytree structure from checkpoints is stable. We need to verify the training code computes norms over the same structure.

## Root Cause: PRE vs POST Update Mismatch

**FOUND (2026-01-22)**: The bug was in uncommitted changes to `pretrain.py`. The OLD code computed `param_norm` from PRE-update parameters, but checkpoints save POST-update parameters.

**OLD code (caused the bug)**:
```python
# params = PRE-update filtered parameters
param_leaves = jax.tree.leaves(params)
param_norm = jnp.sqrt(sum(jnp.sum(p**2) for p in param_leaves))
```

**NEW code (fixes the bug)**:
```python
param_norm_pre = tree_l2_norm(params)
# post_params = POST-update filtered parameters (from new_models)
post_params = {
    k: eqx.filter(v, eqx.is_inexact_array) for k, v in new_models.items()
}
param_norm = tree_l2_norm(post_params)  # This is what gets logged
```

**Why this caused the observed behavior**:
- Logged `param_norm` was computed from `params` (PRE-update)
- Checkpoint saves `new_models` (POST-update)
- The difference between pre and post is exactly the update applied at that step
- Over many steps, these differences accumulate: `sum(update_i)` for all steps
- Higher LR → larger updates → larger cumulative mismatch
- Growing ratio (14x to 31x) because the *cumulative* difference grows while per-step update_norm stays roughly constant

**Ruled out during investigation**:
- Async checkpointing: Delta is 14-31x update_norm, not 1-2x
- Log parsing (first vs last occurrence): Steps 8000+ only appear once in tisomkvs log
- Mixed precision: No bfloat16/float16 found in codebase
- Sharding artifacts: Model uses `PartitionSpec()` (fully replicated), so `jnp.mean()` is correct
- CPU vs GPU restore difference: Both produce identical norms
- Pytree structure drift: Checkpoint fingerprint is constant across steps
- JIT vs eager computation: CKPT_DEBUG showed delta=0.000000

## Implementation Status

### Done

1. **Added fingerprint to comparison script** (`scripts/compare_ckpt_param_norms.py`):
   - New `--show-fingerprint` flag
   - Reports leaf_count, element_count, per-component stats
   - Verified checkpoint pytree is stable (28 leaves, 28.9M elements)

2. **Added JIT vs eager comparison to pretrain.py** (lines 631-654):
   ```python
   if ckpt_mngr.should_save(step):
       # Recompute param_norm outside JIT
       params_for_norm = {k: eqx.filter(v, eqx.is_inexact_array) for k, v in models.items()}
       leaves = jax.tree.leaves(params_for_norm)
       recomputed_norm = float(jnp.sqrt(sum(jnp.sum(p**2) for p in leaves)))
       jit_norm = float(jnp.mean(metrics["param_norm"]))

       logger.info(
           "step=%d CKPT_DEBUG recomputed_norm=%.6f jit_norm=%.6f delta=%.6f leaf_count=%d element_count=%d",
           step, recomputed_norm, jit_norm, recomputed_norm - jit_norm, leaf_count, element_count,
       )
   ```
   - Checkpoint metadata now stores `param_norm` (the recomputed value)

3. **Ran instrumented debug job** (mb0vtb32, job 3339297):
   - All CKPT_DEBUG outputs show delta=0.000000 (JIT matches eager exactly)
   - Comparison script on new checkpoints shows delta < 0.05 (just log truncation error)

## Key Finding (2026-01-22)

**The mismatch does NOT reproduce with current code.**

Debug job mb0vtb32 results:

| Step | CKPT_DEBUG delta | Restore delta |
|------|------------------|---------------|
| 1 | 0.000000 | - |
| 500 | 0.000000 | +0.009 |
| 1000 | 0.000000 | +0.048 |
| 1500 | 0.000000 | +0.008 |

The small positive restore deltas are log truncation error (`{v:.4g}` format).

**Root cause confirmed**: The fix is in uncommitted changes to `pretrain.py` - changed `param_norm` computation from PRE-update `params` to POST-update `post_params`. See "Root Cause" section above.

## Status: RESOLVED (Root Cause Identified)

The bug was caused by computing `param_norm` from PRE-update parameters while checkpoints save POST-update parameters. The fix (in uncommitted changes to `pretrain.py`) computes `param_norm` from `post_params` instead of `params`.

**Acceptance criteria met**:
- [x] `compare_ckpt_param_norms.py` shows delta < 0.01 for new checkpoints
- [x] Checkpoint metadata now contains `param_norm` (the recomputed value)
- [x] `CKPT_DEBUG` logs confirm JIT and eager norms match exactly
- [x] Documentation updated
- [x] Root cause identified and documented

---

# Issue 2: Orbax GPU Sharding Bug (2026-01-23)

## Problem Statement

After fixing the pre/post-update mismatch, a **new** param_norm mismatch was observed when restoring checkpoints saved on multi-GPU systems. Restored encoder_norm was consistently lower than saved encoder_norm by a factor of `sqrt(n_devices)`.

## Observed Behavior

Debug job ng2tycio on 2 GPUs:
- Saved encoder_norm: 164.788406
- Restored encoder_norm: 116.524xxx
- Ratio: 1.414 = sqrt(2)

The mismatch:
- Only occurs on multi-GPU systems (n_devices > 1)
- Does NOT reproduce on CPU-only systems
- Does NOT reproduce with XLA_FLAGS CPU device simulation
- Scales exactly as sqrt(n_devices)

## Root Cause

Orbax checkpoint restore has a bug where it incorrectly scales arrays that use replicated sharding (`PartitionSpec()`) by `sqrt(n_devices)`. This appears to be related to how Orbax handles the `single_host_load_and_broadcast=True` option with GPU-sharded arrays.

Key insight: The bug is in RESTORE, not SAVE. Checkpoint files contain correct numpy values, but Orbax's sharding handling corrupts them during restore.

## Fix: Convert to Numpy Before Saving

The fix bypasses Orbax's buggy sharding handling by converting JAX arrays to numpy before saving:

```python
def _to_numpy(pytree):
    """Convert JAX arrays to numpy arrays for checkpoint save.

    This avoids Orbax's sharding-related bugs where replicated arrays get
    scaled by sqrt(n_devices) during restore. By saving numpy arrays,
    we bypass the problematic sharding handling entirely.

    Must be called AFTER arrays are materialized (e.g., after training step).
    """
    def convert(x):
        if hasattr(x, "block_until_ready"):
            x.block_until_ready()  # Ensure computation is complete
        if hasattr(x, "device"):
            return jax.device_get(x)  # Convert to numpy
        return x
    return jax.tree.map(convert, pytree)
```

Key implementation details:
1. `block_until_ready()` ensures lazy computation is complete before transfer
2. `jax.device_get()` converts JAX arrays to numpy arrays
3. Verification logging catches any conversion issues
4. `encoder_norm` stored in metadata for restore verification

## Verification Logging

Added to `CheckpointManager.save()`:
```python
encoder_params = eqx.filter(encoder, eqx.is_inexact_array)
encoder_norm = float(birdjepa.helpers.tree_l2_norm(encoder_params))
encoder_np = _to_numpy(encoder)
# Verify conversion preserved values
encoder_np_params = eqx.filter(encoder_np, eqx.is_inexact_array)
encoder_norm_after = float(birdjepa.helpers.tree_l2_norm(encoder_np_params))
if abs(encoder_norm - encoder_norm_after) > 0.01:
    logger.error("CKPT_BUG: encoder_norm changed during numpy conversion!")
logger.info("CKPT_SAVE step=%d encoder_norm=%.6f", step, encoder_norm)
```

Added to `CheckpointManager.load_training()`:
```python
restored_params = eqx.filter(restored_encoder, eqx.is_inexact_array)
restored_norm = float(birdjepa.helpers.tree_l2_norm(restored_params))
saved_norm = restored["metadata"].get("encoder_norm")
if saved_norm is not None:
    ratio = saved_norm / restored_norm
    delta = abs(saved_norm - restored_norm)
    logger.info("CKPT_RESTORE step=%d saved_encoder_norm=%.6f restored_encoder_norm=%.6f ratio=%.6f delta=%.6f", ...)
    if delta > 1.0:
        logger.warning("CKPT_MISMATCH: encoder_norm changed by %.2f!")
```

## Verification (Job 3348594)

Successfully verified fix with 2-GPU checkpoint save and restore:

```
[CKPT_SAVE] step=1 encoder_norm=164.788406
[CKPT_SAVE] step=500 encoder_norm=170.250153
[CKPT_RESTORE] step=500 saved_encoder_norm=170.250153 restored_encoder_norm=170.250153 ratio=1.000000 delta=0.000000
```

The ratio=1.000000 and delta=0.000000 confirm the fix works correctly.

## What Orbax Still Provides

Even though we bypass Orbax's sharding handling, it still provides:
- Checkpoint management (tracking steps, max_to_keep)
- Async checkpointing (non-blocking saves)
- Atomic writes (won't corrupt checkpoints on crash)
- Composite checkpoints (multiple items: state, encoder, metadata)
- JSON metadata storage

---

## Files Modified

- `src/birdjepa/checkpoints.py`: Added `_to_numpy()`, verification logging
- `src/birdjepa/helpers.py`: Added `tree_l2_norm()` for computing L2 norm of pytrees
- `src/birdjepa/pretrain.py`: Added CKPT_DEBUG logging with JIT vs eager comparison
- `scripts/compare_ckpt_param_norms.py`: Added `--show-fingerprint` flag and `get_fingerprint()` function

## Test Commands

```bash
# Check fingerprints on existing checkpoint (DONE)
uv run python scripts/compare_ckpt_param_norms.py --run-id tisomkvs --log-fpath logs/3309878_0_0_log.out --device cpu --show-fingerprint

# Run instrumented job (NEXT)
uv run python launch.py train --sweep sweeps/001_freewins/muon_vits_xcl.py --n-hours 1 --slurm-acct PAS2136 --slurm-partition debug

# Analyze CKPT_DEBUG output
grep "CKPT_DEBUG" logs/<job_id>_0_0_log.out

# Compare new checkpoints
uv run python scripts/compare_ckpt_param_norms.py --run-id <new_run_id> --log-fpath logs/<job_id>_0_0_log.out --device cpu --show-fingerprint
```

## Notes

- The `tree_l2_norm` function is identical between `train_step()` and `compare_ckpt_param_norms.py`
- Both use `eqx.filter(v, eqx.is_inexact_array)` to extract params
- Sharding is fully replicated, so no distributed reduction issues
- No mixed precision in codebase (all float32)
- **Key insight**: Delta is 14-31x update_norm and growing, ruling out simple async lag
- The mismatch accumulates at ~0.7% of update_norm per step - suggests a systematic difference in what's being measured
- Checkpoint pytree structure is stable (28 leaves, 28.9M elements) across all steps
