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
