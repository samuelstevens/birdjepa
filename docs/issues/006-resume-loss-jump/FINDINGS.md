# Training Loss Jumps at Checkpoint Resume

Last updated: 2026-02-03

# Summary

Training runs on the preemptible partition show systematic loss jumps when resuming from checkpoint. The dataloader reset test did not reproduce the jump, so the cause is likely a resume-only state difference beyond loader order.

# Background

Resume events create large loss discontinuities and materially change final loss, even when configs are identical. This document tracks the hypothesis -> experiment -> results cycles.

## What Is Not Checkpointed

During resume:
- Model weights (encoder, objective, probe) - restored correctly
- Optimizer state - restored correctly
- Dataloader state - restarts from beginning
- JAX PRNG key - reinitialized from seed
- Shuffle buffer contents - starts empty ("cold")

# Idea 1: Seed Variance Test Shows Systematic Resume Jumps

## Hypothesis

Loss jumps at resume points are systematic and seed-dependent, and their direction correlates with final loss.

## Experiment

Run three jobs with identical configs but different seeds: 27i3idlk (seed 1), xy9gn8ek (seed 2), 7bdmyn24 (seed 3).

## Results

| Run (Seed) | Resume@3505 | Resume@7005 | Final Loss |
|------------|-------------|-------------|------------|
| 27i3idlk (1) | -2.16 (down) | +4.07 (up) | 4.87 |
| xy9gn8ek (2) | -2.79 (down) | +3.70 (up) | 2.62 |
| 7bdmyn24 (3) | +3.02 (up) | +3.25 (up) | 4.78 |

- Runs with negative jumps at early resumes performed better.
- Seed 2 had the largest negative jump (-2.79) and achieved the best final loss.
- Seed 3 had positive jumps at both resumes and performed worst.

# Idea 2: Entropy Explains Loss Jump Direction

## Hypothesis

Loss change direction at resume points correlates with changes in target entropy (tgt_entropy_frac).

## Experiment

Compare loss and target entropy around resume points for the seed variance runs.

## Results

Target entropy shows minimal change at resume points (about 0.02 to 0.05) with no consistent correlation to loss direction.

# Idea 3: Dataloader Reset Causes the Loss Jump

## Hypothesis

If the loss jump is caused by dataloader reset alone, we should be able to reproduce it by manually resetting the dataloader mid-training without any checkpoint save/load.

## Experiment

- Run 9lm0bfgu: Reset dataloader at step 1000, run to step 2000
- Run knze87mn: Control - no reset, same config

## Results

- Both runs show smooth loss and EMA around step 1000 with no meaningful discontinuity.
- EMA(0.05) change at the first point after step 1000:
  - reset: -0.0067
  - control: -0.0094
- The step-1000 loss deltas are similarly small and negative in both runs.
- Slurm array job 3371929 completed successfully.

# Idea 4: Resume-Only State Differences Cause the Loss Jump

## Hypothesis

Resume-only state differences (PRNG state, optimizer interaction, shuffle buffer warmup, or checkpoint load ordering) cause the loss discontinuity.

## Experiment

- Run PRNG mode sweep with identical configs and forced resume at step 500:
  - stateless PRNG: bumi9dy3
  - checkpointed PRNG: vt5twgr2
  - disable stochastic data (no augmentations, window_size=1): kcrqmj4f

## Results

- Both stateless and checkpointed runs show clear loss jumps at resume for loss_pr and loss_ce.
- Disabling stochastic data reduces the magnitude of the jump for both losses but does not eliminate it.
- Step-500 deltas (post - pre) from the resume window:
  - stateless: loss_pr +0.244, loss_ce +0.235
  - checkpointed: loss_pr +0.289, loss_ce +0.245
  - no_stochastic: loss_pr +0.098, loss_ce +0.050

# Idea 5: Checkpoint Roundtrip Without Dataloader Restart

## Hypothesis

If the loss jump is caused by checkpoint save/load itself (serialization errors, restore mismatches, or a subtle interaction with JIT/static state), then we should be able to reproduce the discontinuity by saving a checkpoint and immediately restoring it in-process, while continuing to consume batches from the same dataloader iterator.

Conversely, if the in-process checkpoint roundtrip is smooth, then the loss jump must come from process-level resume effects (Rust loader restart, cold shuffle buffer, multithreaded nondeterminism, etc.), not Orbax restore.

## Experiment

### Code Changes

Add `cfg.debug_roundtrip_steps: list[int] = []` to the pretrain config. When non-empty, perform an in-process checkpoint roundtrip after completing each listed step:

1. Create a debug `CheckpointManager` pointing to `outputs/<run_id>/debug_ckpt/` with `async_checkpointing=False` (synchronous saves).
2. Call `debug_mngr.save(step, ..., force=True)` to save checkpoint.
3. Call `debug_mngr.load_training(encoder, objective, probe, opt_state, encoder_config=...)` which returns `(encoder, objective, probe, opt_state, step, prng_key)`.
4. Replace in-memory state with restored values. Keep in-memory PRNG (ignore returned `prng_key`) to isolate test to model/optimizer state.
5. Call `jax.clear_caches()` to force JIT re-trace of train_step.
6. Do not restart the Rust dataloader or iterator; continue using the same `train_iter`.

Error handling:
- If save/restore raises an exception: abort the run.
- If restored state differs from pre-save state beyond tolerance (atol=1e-6, rtol=1e-5): log warning but continue training.

LR schedule is stateful: optax stores the step counter inside `opt_state`. Restoring `opt_state` automatically continues the LR schedule at the correct step.

### Sweep Design

Sweep file: `docs/issues/006-resume-loss-jump/idea5_roundtrip.py`

| Condition | PRNG Mode | debug_roundtrip_steps |
|-----------|-----------|----------------------|
| roundtrip | stateless | [500, 1000, 1500] |
| roundtrip | checkpointed | [500, 1000, 1500] |
| roundtrip | no_stochastic | [500, 1000, 1500] |
| control | stateless | [] |
| control | checkpointed | [] |
| control | no_stochastic | [] |

- 6 runs total (3 PRNG modes x 2 conditions)
- 2000 steps per run
- Base config: same as Idea 4 PRNG mode sweep
- `checkpoint_every=10000` to disable normal checkpointing
- Same seed for all runs (for fair comparison between roundtrip and control)
- Slurm: preemptible-nextgen partition, no requeue (--no-requeue)
- W&B: same project as Idea 4, tag="idea5_roundtrip"

### Analysis

Manual analysis in notebook after runs complete. Compare loss_pr/loss_ce around the roundtrip step vs control runs to determine whether checkpoint save/restore alone creates a discontinuity.

## Results

Data: `docs/issues/006-resume-loss-jump/idea5-roundtrip.csv` and `docs/issues/006-resume-loss-jump/idea5-roundtrip-step{500,1000,1500}.png`.

Main result: the in-process save->restore roundtrip does not reproduce the large loss discontinuities seen under true preemption resumes.

- Note: `idea5-roundtrip.csv` only contains the stateless + checkpointed PRNG modes (no `no_stochastic` rows in this export).
- No discrete loss discontinuity at the in-process roundtrip steps (500 and 1000). Loss changes across the 10-step window (center-5 -> center+5) are small and not consistently larger under roundtrip than control:

| Window | PRNG mode | Control d_loss_pr / d_loss_ce | Roundtrip d_loss_pr / d_loss_ce |
|---|---|---:|---:|
| 495->505 (around step 500) | stateless | +0.300 / +0.268 | +0.048 / +0.022 |
| 495->505 (around step 500) | checkpointed | +0.131 / +0.244 | +0.169 / +0.098 |
| 995->1005 (around step 1000) | stateless | +0.068 / +0.098 | +0.073 / +0.089 |
| 995->1005 (around step 1000) | checkpointed | -0.081 / -0.041 | +0.008 / -0.051 |

- By step 1500 the runs have drifted substantially (roundtrip-control at step 1500), but this looks like smooth trajectory divergence rather than a sharp jump at the roundtrip boundary:
  - stateless: loss_pr -0.969, loss_ce -0.948
  - checkpointed: loss_pr +0.273, loss_ce +0.282
- Optimizer-state diagnostics are "boring" (good sign): `opt_state_abs_max` is constant (4.775) and `opt_state_l2`/`opt_state_abs_mean` evolve smoothly with no visible kinks at the roundtrip steps.

Conclusion: the big loss jumps are unlikely to be caused by Orbax serialization/restore itself; focus should remain on process-level resume effects (Rust dataloader restart, cold shuffle buffer, PRNG/worker nondeterminism, etc.).

# Idea 6: IID Sampling Diagnostics

## Hypothesis

If the dataloader perfectly mimics sampling with replacement, resume would be seamless because there's no "state" beyond the PRNG. By logging bucket and shard histograms plus index-level collision metrics, we can diagnose whether: (1) the current shuffle buffer approximates IID sampling, (2) resume causes a measurable distribution shift, and (3) temporal correlation exists in the batch sequence.

## Motivation

With true sampling-with-replacement, the only state is the PRNG key. If diagnostics reveal the current dataloader has significant temporal correlation or non-uniform sampling, that could explain why resume (which resets the shuffle buffer) causes loss jumps. The goal is to gather enough signal to either: (a) identify a fixable issue in the Rust loader, or (b) confirm that switching to true replacement sampling would solve the problem.

## Metrics

All metrics logged to W&B under the `dataloader/` prefix. Computed every step (force `log_every=1` for this experiment to get true lag-1 correlation).

### Bucket Histogram Metrics (3 methods x 2 statistics = 6 metrics)

Bucket indices into 256 bins using three methods to compare empirically:

| Method | Formula | Rationale |
|--------|---------|-----------|
| Modulo | `index % 256` | Simple, may alias with shard structure |
| Division | `index // (dataset_size // 256)` | Contiguous ranges, may alias with data order |
| Hash | `(indices.astype(np.uint64) * 2654435761) % 256` | Golden ratio prime, randomizes structure, uint64 prevents overflow |

For each method, compute:
- `bucket_chi_sq_{mod,div,hash}`: Chi-squared statistic vs uniform. Per-batch values are noisy (expected count ~B/256 per bin), but can compute EMA post-hoc from W&B data. Note: division bucketing has slight bias when dataset_size % 256 != 0, but compare to hash empirically.
- `bucket_acf_{mod,div,hash}_lag1`: Pearson correlation between current step's histogram and the previous step's histogram (true lag-1). State (`prev_*`) updates every step, so correlation is always lag-1 regardless of `log_every`.

### Shard Histogram Metrics (2 metrics)

Use raw shard IDs (Arrow file index, ~100-1000 shards):
- `shard_chi_sq`: Chi-squared statistic vs uniform. (Note: if shards have different sizes, uniform isn't the true null, but useful for stationarity.)
- `shard_acf_lag1`: Pearson correlation with previous step's shard histogram (true lag-1).

### Index-Level Metrics (4 metrics)

- `n_unique_indices`: Number of unique sample indices in the batch. Should equal batch_size under sampling without replacement.
- `overlap_exact`: Multiset overlap between current and previous batch (counts with multiplicity). Computed as sum of `min(count_curr[i], count_prev[i])` over all indices. Expected under IID with replacement: `B^2 / N`.
- `overlap_bucket`: Bucket-level overlap using 4096 buckets (not 256) to reduce collision noise. Sum of `min(hist_t[i], hist_{t-1}[i])` over buckets. With 4096 buckets and B=1024, expected collision rate ~B^2/4096 = 256, giving more signal.
- `actual_batch_size`: Sanity check for constant batch size.

### Sliding Window Uniqueness

- `n_unique_window`: Number of unique indices seen in last W=100 batches. Compare to expected under IID with replacement: `N * (1 - (1 - 1/N)^(W*B))`. Strong diagnostic for replacement vs quasi-without-replacement.

### Resume Detection (Post-hoc)

Resume boundaries are detected post-hoc from W&B logs using the same method as previous ideas (see `notebooks/006_resumption.py`):
1. Step decreasing: `step[i] <= step[i-1]`
2. Timestamp gap > 600s: `_timestamp[i] - _timestamp[i-1] > 600`

No explicit `resume_count` metric needed since `_timestamp` is logged automatically by W&B.

### First Step Handling

On the first logged step (no previous histogram/batch available):
- ACF metrics: log 1.0 (self-correlation sentinel)
- Overlap metrics: log 0

## Implementation

### Rust Changes

Modify `Loader.__next__()` to return `shard_ids` in the batch dict:
- `shard_ids`: `np.ndarray[int64]` of shape `(batch_size,)` containing the Arrow file index for each sample

Update `_rs.pyi` type stub:
```python
def __next__(self) -> dict[str, tp.Any]: ...  # includes 'shard_ids' key
```

The Rust loader already returns `indices` in the batch (confirmed in `data/__init__.py:454`).

### Python Changes

**`src/birdjepa/data/__init__.py`**

In `RustXenoCantoLoader.__init__()`, store shard count:
```python
self._n_shards = len(arrow_fpaths)  # No Rust API change needed
```

In `RustXenoCantoLoader._postprocess()`, pass through `shard_ids` with validation:
```python
shard_ids = batch["shard_ids"]
assert shard_ids.max() < self._n_shards, f"shard_id {shard_ids.max()} >= n_shards {self._n_shards}"

return {
    "data": spec.astype(np.float32),
    "target": np.asarray(labels),
    "label": str_labels,
    "index": indices,
    "shard_id": shard_ids,
}
```

Add `n_shards` property:
```python
@property
def n_shards(self) -> int:
    return self._n_shards
```

**`src/birdjepa/pretrain.py`**

Add state variables before training loop:
```python
# IID sampling diagnostics state
n_buckets = 256
n_buckets_overlap = 4096  # More buckets for overlap to reduce collision noise
dataset_size = len(train_loader)
n_shards = train_loader.n_shards

prev_bucket_hist_mod: np.ndarray | None = None
prev_bucket_hist_div: np.ndarray | None = None
prev_bucket_hist_hash: np.ndarray | None = None
prev_bucket_hist_overlap: np.ndarray | None = None  # 4096 buckets for overlap
prev_shard_hist: np.ndarray | None = None
prev_indices_counts: dict[int, int] | None = None  # For multiset overlap

# Sliding window for n_unique_window (circular buffer of last 100 batches)
window_size = 100
index_window: collections.deque[set[int]] = collections.deque(maxlen=window_size)
```

Add helper functions:
```python
def compute_bucket_hist(indices: np.ndarray, method: str, n_buckets: int, dataset_size: int) -> np.ndarray:
    if method == "mod":
        buckets = indices % n_buckets
    elif method == "div":
        bucket_size = max(1, dataset_size // n_buckets)
        buckets = np.clip(indices // bucket_size, 0, n_buckets - 1)
    elif method == "hash":
        # Cast to uint64 to prevent overflow in hash multiply
        buckets = (indices.astype(np.uint64) * 2654435761) % n_buckets
        buckets = buckets.astype(np.int64)
    else:
        raise ValueError(f"Unknown bucket method: {method}")
    return np.bincount(buckets, minlength=n_buckets).astype(np.float64)

def chi_sq_uniform(hist: np.ndarray) -> float:
    n = hist.sum()
    if n == 0:
        return 0.0
    expected = n / len(hist)
    return float(np.sum((hist - expected) ** 2 / expected))

def hist_acf(hist: np.ndarray, prev_hist: np.ndarray | None) -> float:
    if prev_hist is None:
        return 1.0  # Self-correlation sentinel for first step
    h1 = hist - hist.mean()
    h2 = prev_hist - prev_hist.mean()
    denom = np.sqrt((h1 ** 2).sum() * (h2 ** 2).sum())
    if denom < 1e-10:
        return 1.0
    return float((h1 * h2).sum() / denom)
```

After step completes, compute diagnostics. **Important**: `prev_*` state is updated every step to maintain true lag-1 correlation, but metrics are only logged when `step % log_every == 0`.
```python
indices_np = batch_raw["index"]
shard_ids_np = batch_raw["shard_id"]

# Bucket histograms (3 methods)
bucket_hist_mod = compute_bucket_hist(indices_np, "mod", n_buckets, dataset_size)
bucket_hist_div = compute_bucket_hist(indices_np, "div", n_buckets, dataset_size)
bucket_hist_hash = compute_bucket_hist(indices_np, "hash", n_buckets, dataset_size)

# Separate 4096-bucket histogram for overlap (less collision noise)
bucket_hist_overlap = compute_bucket_hist(indices_np, "hash", n_buckets_overlap, dataset_size)

# Shard histogram
shard_hist = np.bincount(shard_ids_np, minlength=n_shards).astype(np.float64)

# Index-level metrics
indices_unique, indices_counts = np.unique(indices_np, return_counts=True)
n_unique = len(indices_unique)

# Multiset overlap: sum of min(count_curr[i], count_prev[i]) over shared indices
if prev_indices_counts is not None:
    shared_indices = np.intersect1d(indices_unique, list(prev_indices_counts.keys()))
    curr_counts = dict(zip(indices_unique, indices_counts))
    overlap_exact = sum(min(curr_counts[i], prev_indices_counts[i]) for i in shared_indices)
else:
    overlap_exact = 0
overlap_bucket = int(np.minimum(bucket_hist_overlap, prev_bucket_hist_overlap).sum()) if prev_bucket_hist_overlap is not None else 0

# Sliding window uniqueness (uses set for unique count only)
index_window.append(set(indices_unique))
all_indices_in_window = set().union(*index_window)
n_unique_window = len(all_indices_in_window)
# Expected under IID: N * (1 - (1 - 1/N)^(W*B))
w_batches = len(index_window)
expected_unique_iid = dataset_size * (1 - (1 - 1/dataset_size) ** (w_batches * cfg.batch_size))

log_dict.update({
    "dataloader/bucket_chi_sq_mod": chi_sq_uniform(bucket_hist_mod),
    "dataloader/bucket_chi_sq_div": chi_sq_uniform(bucket_hist_div),
    "dataloader/bucket_chi_sq_hash": chi_sq_uniform(bucket_hist_hash),
    "dataloader/bucket_acf_mod_lag1": hist_acf(bucket_hist_mod, prev_bucket_hist_mod),
    "dataloader/bucket_acf_div_lag1": hist_acf(bucket_hist_div, prev_bucket_hist_div),
    "dataloader/bucket_acf_hash_lag1": hist_acf(bucket_hist_hash, prev_bucket_hist_hash),
    "dataloader/shard_chi_sq": chi_sq_uniform(shard_hist),
    "dataloader/shard_acf_lag1": hist_acf(shard_hist, prev_shard_hist),
    "dataloader/n_unique_indices": n_unique,
    "dataloader/overlap_exact": overlap_exact,
    "dataloader/overlap_bucket": overlap_bucket,
    "dataloader/n_unique_window": n_unique_window,
    "dataloader/expected_unique_iid": expected_unique_iid,
    "dataloader/actual_batch_size": len(indices_np),
})

# Update state for next step
prev_bucket_hist_mod = bucket_hist_mod
prev_bucket_hist_div = bucket_hist_div
prev_bucket_hist_hash = bucket_hist_hash
prev_bucket_hist_overlap = bucket_hist_overlap
prev_shard_hist = shard_hist
prev_indices_counts = dict(zip(indices_unique, indices_counts))
```

### Required Rust Loader Changes

1. Add `shard_ids` to batch output (Arrow file index per sample)
   - Track which file each sample came from during I/O
   - Return as int64 array in batch dict

## Analysis Plan

### Experiment Setup

- Force `log_every=1` to get true lag-1 ACF measurements
- Run on preemptible partition to naturally trigger resumes
- Multiple seeds to get multiple resume events

### Baseline Characterization

Run a single training job with diagnostics enabled. Analyze:

1. **Chi-squared stability**: Per-batch `bucket_chi_sq_{mod,div,hash}` and `shard_chi_sq` will be noisy (expected count ~4 per bin with B=1024, 256 bins). Compute EMA post-hoc from W&B data for trend analysis.

2. **ACF values**: Should be close to 0 under IID. If `bucket_acf_{mod,div,hash}` or `shard_acf_lag1` is consistently positive, shuffle buffer has temporal correlation. Compare the three bucket methods to see if aliasing affects the signal.

3. **Duplicate rate**: `n_unique_indices / batch_size` should equal 1.0 under sampling without replacement.

4. **Sliding window uniqueness**: Compare `n_unique_window` to `expected_unique_iid`. If `n_unique_window >> expected_unique_iid`, dataloader is closer to without-replacement than with-replacement (good for training, but problematic for resume).

5. **Overlap metrics**: `overlap_exact` expected under IID: `B^2/N`. `overlap_bucket` (4096 buckets) should track `overlap_exact` with less noise. If consistently lower than IID expectation, confirms without-replacement behavior.

### Resume Boundary Detection

Detect resume points post-hoc from W&B logs using step decrease + timestamp gap > 600s (same method as `notebooks/006_resumption.py`).

Compare metrics in 100-step windows before vs after resume:
- Mean and variance of chi-sq metrics (compute EMA post-hoc)
- Distribution of ACF metrics across all three bucket methods
- Mean `n_unique_window` vs `expected_unique_iid` ratio
- `overlap_exact` and `overlap_bucket` distributions

Compare to "normal drift" baseline: same analysis between two arbitrary 100-step windows far from any resume.

### Null Distribution

To determine significance thresholds:
1. Simulate IID sampling from uniform distribution over dataset_size indices
2. Compute same metrics on simulated data
3. Use 95th percentile of simulated metric distributions as threshold for "anomalous"

### Success Criteria

Quantitative:
- **Distribution shift detected**: If resume causes chi-sq metrics (post-hoc EMA) to spike > 2x baseline variance, and this spike correlates with loss jump magnitude (Pearson r > 0.5), we have direct evidence.
- **Temporal correlation detected**: If ACF metrics drop from pre-resume mean by > 0.1, shuffle buffer state is materially different post-resume.
- **Bucket method comparison**: If mod/div show different patterns than hash around resume, aliasing with shard structure is a factor.
- **No signal**: If all metrics are within baseline noise, the issue is elsewhere (PRNG, worker nondeterminism, numerical precision, etc.)

Qualitative:
- Produce plots of each metric around resume boundaries for visual inspection
- Compare across seeds to check consistency

## Results

(Pending experiment)
