# TrainingJob Wrapper Causes NaN Values

## Summary

**Update (2026-01-20):** The TrainingJob wrapper no longer reproduces NaNs and is now the default submission path. The `__submitit_checkpoint__` hook on `worker_fn` was removed. We now submit `TrainingJob(cfg)` directly for checkpointable requeue behavior.

## Historical Context

The sections below document the original failure from 2026-01-19, kept for reference.

## Context

We wanted to add automatic job requeue on timeout using submitit's checkpointing feature. The implementation:

```python
class TrainingJob:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def __call__(self):
        worker_fn(self.cfg)

    def checkpoint(self):
        import submitit.helpers
        return submitit.helpers.DelayedSubmission(TrainingJob(self.cfg))
```

Submission changed from:
```python
jobs = [executor.submit(worker_fn, c) for c in cfgs]
```
to:
```python
jobs = [executor.submit(TrainingJob(c)) for c in cfgs]
```

## Observed Behavior

**Working (worker_fn directly)**:
- Job 3289717: Training proceeds normally
- Step 5: `ce=9.36 loss=18.71 param_norm=216.5` (valid values)

**Failing (TrainingJob wrapper)**:
- Job 3290581: NaN from step 5 onwards
- Step 5: `ce=nan loss=nan param_norm=nan` (all metrics NaN)
- All 4 jobs in the array failed identically

## What We Verified

1. **Config is identical**: Pickled configs have the same values (lr, optimizer, seed, model config, etc.)

2. **Initialization is identical**: Both job types show the same initialization sequence in logs:
   - Same devices detected (cuda:0, cuda:1)
   - Same model created (TransformerModel)
   - Same objective created (Supervised)
   - Same optimizer initialized

3. **Step 1 succeeds**: Both job types successfully save a checkpoint at step 1, indicating the model has valid values initially

4. **NaN appears between steps 1-5**: The model parameters become NaN somewhere between step 1 (checkpoint saved) and step 5 (first logged step)

5. **Systematic failure**: All 4 jobs failed identically at step 5, suggesting a deterministic bug rather than race condition

## What submitit Does Differently

When submitit runs a job:

**worker_fn style**:
- `DelayedSubmission(function=worker_fn, args=(cfg,))`
- Execution: `worker_fn(cfg)`

**TrainingJob style**:
- `DelayedSubmission(function=TrainingJob(cfg), args=())`
- Execution: `TrainingJob(cfg)()`  which calls `worker_fn(self.cfg)`

In both cases, `worker_fn` receives a Config object and executes identically.

## Resolution

Switched back to using the TrainingJob wrapper and removed the `worker_fn.__submitit_checkpoint__` hook. Current behavior is checkpointable and stable.

## Hypotheses

1. **cloudpickle serialization difference**: When cloudpickle serializes a class instance vs a function with args, it might capture different module state or closures.

2. **JAX array buffer issue**: Similar to the `eqx.filter_shard` bug we fixed for batch data, there might be a buffer management issue that manifests differently when code is pickled as a class instance.

3. **Module import order**: When submitit unpickles a TrainingJob, it imports `birdjepa.pretrain` to reconstruct the class. This might affect JAX initialization state.

4. **Random key state**: Although both use `seed=42` and key derivation happens inside `worker_fn`, there might be subtle differences in JAX's PRNG state.

## Reproduction (Historical)

```bash
# This section documents the old reproduction steps from when the wrapper was failing.
```

## Validation

2026-01-20: TrainingJob wrapper on XCL (2 GPU) reached step 5 with finite values (job 3301154).

## Next Steps for Investigation

If this regresses, recheck RoPE stability and compare wrapper vs function serialization again.

## Related

- Job 3290581: NaN jobs using TrainingJob (historical)
- Job 3296694: Checkpointable hook on worker_fn, no NaNs at step 5 (historical)
- Job 3301154: TrainingJob wrapper working (current)
