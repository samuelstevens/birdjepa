# Param Norm Mismatch Between Logs and Checkpoints

## Status: RESOLVED (2026-01-23)

Two separate bugs were identified and fixed:

1. **Pre-update vs post-update mismatch** (fixed 2026-01-22): Logged param_norm was computed from pre-update parameters while checkpoints saved post-update parameters.

2. **Orbax GPU sharding bug** (fixed 2026-01-23): Orbax incorrectly scaled replicated arrays by `sqrt(n_devices)` during checkpoint restore on multi-GPU systems.

## Fix Summary

The Orbax bug was fixed in `src/birdjepa/checkpoints.py` by converting JAX arrays to numpy before saving:

```python
def _to_numpy(pytree):
    """Convert JAX arrays to numpy arrays for checkpoint save.

    This avoids Orbax's sharding-related bugs where replicated arrays get
    scaled by sqrt(n_devices) during restore. By saving numpy arrays,
    we bypass the problematic sharding handling entirely.
    """
    def convert(x):
        if hasattr(x, "block_until_ready"):
            x.block_until_ready()  # Ensure computation is complete
        if hasattr(x, "device"):
            return jax.device_get(x)  # Convert to numpy
        return x
    return jax.tree.map(convert, pytree)
```

Verification logging was added to detect any future issues:
- `CKPT_SAVE` logs encoder_norm at save time
- `CKPT_RESTORE` compares saved vs restored encoder_norm (ratio should be 1.0)
- Checkpoint metadata stores `encoder_norm` for verification

## Verification (Job 3348594)

```
CKPT_SAVE step=1 encoder_norm=164.788406
CKPT_SAVE step=500 encoder_norm=170.250153
CKPT_RESTORE step=500 saved_encoder_norm=170.250153 restored_encoder_norm=170.250153 ratio=1.000000 delta=0.000000
```

---

# Historical Investigation

## Original Summary (2026-01-22)

For sweep 3309878, checkpoint param_norm values are consistently lower than logged train/param_norm values. The gap grows with step (8000 -> 10000) and is far larger than update_norm from the same logs. CPU and GPU results match for tisomkvs, so this is not a CPU/GPU restore issue. Component norms (encoder/objective/probe) show the same downward bias.

## Context

We saw suspicious behavior after resubmits and wanted to confirm whether optimizer or parameter state was restoring correctly. We wrote scripts/compare_ckpt_param_norms.py to recompute param_norm directly from checkpoints and compare against the log file values at the same steps.

## Observed Behavior

Per-run delta range (ckpt minus log) from step 8000 to 10000:
- tisomkvs (logs/3309878_0_0_log.out): -4.09 to -8.65
- qla6ertl (logs/3309878_1_0_log.out): -18.46 to -34.03
- ojl57o9i (logs/3309878_2_0_log.out): -341.48 to -408.11
- nixdo1ql (logs/3309878_3_0_log.out): -317.11 to -545.72
- u81ltwuf (logs/3309878_4_0_log.out): -885.21 to -1118.62

## What We Verified

- CPU and GPU comparisons for tisomkvs are identical (not a device restore artifact).
- All five runs in sweep 3309878 show the same sign and growth of the mismatch.
- Encoder/objective/probe component norms show the same downward bias as total param_norm.

## Reproduction

```
uv run python scripts/compare_ckpt_param_norms.py --run-id tisomkvs --log-fpath logs/3309878_0_0_log.out --device cpu
```

## Raw Data

```
uv run python scripts/compare_ckpt_param_norms.py --run-id tisomkvs --log-fpath logs/3309878_0_0_log.out --device cpu
step,ckpt_param_norm,ckpt_param_norm_encoder,ckpt_param_norm_objective,ckpt_param_norm_probe,log_param_norms,delta_first
8000,185.906,154.291,70.014,76.5057,190,-4.09427
8500,190.69,156.542,73.5732,80.2757,195.8,-5.11046
9000,195.541,158.707,77.2701,84.129,201.7,-6.15926
9500,200.578,160.84,81.1588,88.1787,208,-7.42188
10000,205.646,162.907,85.0873,92.2596,214.3,-8.65362

uv run python scripts/compare_ckpt_param_norms.py --run-id tisomkvs --log-fpath logs/3309878_0_0_log.out --device gpu
step,ckpt_param_norm,ckpt_param_norm_encoder,ckpt_param_norm_objective,ckpt_param_norm_probe,log_param_norms,delta_first
8000,185.906,154.291,70.014,76.5057,190,-4.09427
8500,190.69,156.542,73.5732,80.2757,195.8,-5.11046
9000,195.541,158.707,77.2701,84.129,201.7,-6.15926
9500,200.578,160.84,81.1588,88.1787,208,-7.42188
10000,205.646,162.907,85.0873,92.2596,214.3,-8.65361

uv run python scripts/compare_ckpt_param_norms.py --run-id qla6ertl --log-fpath logs/3309878_1_0_log.out --device cpu
step,ckpt_param_norm,ckpt_param_norm_encoder,ckpt_param_norm_objective,ckpt_param_norm_probe,log_param_norms,delta_first
8000,355.336,262.273,166.482,172.511,373.8,-18.4645
8500,376.796,272.73,180.496,187.123,399.4,-22.6036
9000,397.452,282.879,193.79,200.981,424.1,-26.648
9500,417.671,292.843,206.682,214.418,448.2,-30.5289
10000,437.071,302.508,218.91,227.152,471.1,-34.0291

uv run python scripts/compare_ckpt_param_norms.py --run-id ojl57o9i --log-fpath logs/3309878_2_0_log.out --device cpu
step,ckpt_param_norm,ckpt_param_norm_encoder,ckpt_param_norm_objective,ckpt_param_norm_probe,log_param_norms,delta_first
8000,666.524,604.615,193.376,203.226,1008,-341.476
8500,724.235,633.969,243.183,251.916,1086,-361.765
9000,779.615,662.898,285.995,294.234,1160,-380.385
9500,832.407,692.053,323.064,331.05,1228,-395.593
10000,881.891,721.783,354.345,362.216,1290,-408.109

uv run python scripts/compare_ckpt_param_norms.py --run-id nixdo1ql --log-fpath logs/3309878_3_0_log.out --device cpu
step,ckpt_param_norm,ckpt_param_norm_encoder,ckpt_param_norm_objective,ckpt_param_norm_probe,log_param_norms,delta_first
8000,1927.89,1683.58,670.205,658.13,2245,-317.108
8500,2099.2,1788.83,782.708,770.766,2490,-390.803
9000,2247.26,1895.38,859.661,847.776,2701,-453.735
9500,2380.7,2007.84,910.453,898.566,2886,-505.296
10000,2505.28,2128.11,940.708,928.783,3051,-545.719

uv run python scripts/compare_ckpt_param_norms.py --run-id u81ltwuf --log-fpath logs/3309878_4_0_log.out --device cpu
step,ckpt_param_norm,ckpt_param_norm_encoder,ckpt_param_norm_objective,ckpt_param_norm_probe,log_param_norms,delta_first
8000,5730.79,5585.98,939.769,869.252,6616,-885.213
8500,6046.17,5901.88,966.281,888.966,7070,-1023.83
9000,6388.49,6253.24,963.446,884.073,7472,-1083.51
9500,6750.71,6626.63,949.567,870.733,7861,-1110.29
10000,7130.38,7017.03,933.661,855.501,8249,-1118.62
```

## Hypotheses

1. The logged train/param_norm is computed before a different parameter update than the checkpoint save (step alignment mismatch).
2. The logging path and the checkpoint path are using different parameter trees or filter rules.
3. Logging is using stale parameters (e.g., pre-update) while checkpoints save post-update state.

## Next Steps

- Log param_norm immediately before and after checkpoint save for the same step, using the same tree and filter rules.
- Include param_norm in checkpoint metadata at save time, then compare to on-disk state on restore.
- Add a small unit test or debug mode that recomputes param_norm from saved state and compares against the last log value.
