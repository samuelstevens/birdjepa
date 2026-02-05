import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import matplotlib.pyplot as plt
    import numpy as np
    import wandb
    import beartype
    import typing as tp
    return beartype, mo, np, pl, plt, tp, wandb


@app.cell
def _(mo):
    mo.md("""
    # Training Loss Jumps at Checkpoint Resume

    Reference: `docs/issues/006-resume-loss-jump/FINDINGS.md`
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    # Summary

    - Loss jumps at resume are systematic and affect final loss.
    - A dataloader reset alone does not reproduce the loss or EMA discontinuity.
    - We are testing resume-only state differences (PRNG, optimizer interaction, shuffle buffer warmup, load ordering).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Background

    This notebook mirrors the idea breakdown in `docs/issues/006-resume-loss-jump/FINDINGS.md`.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Idea 1: Seed Variance Test Shows Systematic Resume Jumps

    ## Hypothesis

    Loss jumps at resume points are systematic and seed-dependent, and their direction correlates with final loss.

    ## Experiment

    Testing training reproducibility with 3 runs at lr=0.03, Muon optimizer, all free wins enabled.
    Each run uses a different seed (1, 2, 3) but identical config.

    Key question: Do loss spikes occur at checkpoint resume points?
    """)
    return


@app.cell
def _(pl, wandb):
    # Fetch full training history for seed variance test runs
    seed_test_runs = ["27i3idlk", "xy9gn8ek", "7bdmyn24"]


    def _fetch_history(run_id: str):
        api = wandb.Api(timeout=60)
        run = api.run(f"samuelstevens/birdjepa/{run_id}")
        history = run.scan_history(
            keys=[
                "step",
                "train/probe",
                "train/ce",
                "_timestamp",
                "lr",
            ]
        )
        rows = []
        seed = run.config.get("seed")
        seed = int(seed) if seed is not None else -1
        for row in history:
            step = row.get("step")
            loss_pr = row.get("train/probe")
            loss_ce = row.get("train/ce")
            ts = row.get("_timestamp")
            lr = row.get("lr")
            if step is not None and (loss_pr is not None or loss_ce is not None):
                rows.append(
                    {
                        "run_id": run_id,
                        "seed": seed,
                        "step": int(step),
                        "loss_pr": float(loss_pr) if loss_pr is not None else None,
                        "loss_ce": float(loss_ce) if loss_ce is not None else None,
                        "timestamp": ts,
                        "lr": float(lr) if lr is not None else None,
                    }
                )
        schema = {
            "run_id": pl.Utf8,
            "seed": pl.Int64,
            "step": pl.Int64,
            "loss_pr": pl.Float64,
            "loss_ce": pl.Float64,
            "timestamp": pl.Float64,
            "lr": pl.Float64,
        }
        if not rows:
            return pl.DataFrame(schema=schema)
        return pl.DataFrame(rows, schema=schema)


    seed_history_df = pl.concat([_fetch_history(rid) for rid in seed_test_runs])
    seed_history_df
    return seed_history_df, seed_test_runs


@app.cell
def _(np, pl, plt, seed_history_df):
    # Plot: Loss vs Step for seed variance test runs
    _fig, _ax = plt.subplots(figsize=(10, 5), dpi=150)
    _ax2 = None

    _colors = {"1": "#1f77b4", "2": "#ff7f0e", "3": "#2ca02c"}

    for _seed in ["1", "2", "3"]:
        _subset = seed_history_df.filter(pl.col("seed") == int(_seed)).sort("step")
        if len(_subset) == 0:
            continue
        _steps = _subset["step"].to_list()
        _loss_pr = _subset["loss_pr"].to_list()
        _loss_ce = _subset["loss_ce"].to_list()
        _lrs = _subset["lr"].cast(pl.Float64).to_numpy()

        _ax.plot(
            _steps,
            _loss_pr,
            "-",
            color=_colors[_seed],
            label=f"seed={_seed} loss_pr",
            alpha=0.8,
            linewidth=1,
        )
        _ax.plot(
            _steps,
            _loss_ce,
            "--",
            color=_colors[_seed],
            label=f"seed={_seed} loss_ce",
            alpha=0.8,
            linewidth=1,
        )
        if np.isfinite(_lrs).any():
            if _ax2 is None:
                _ax2 = _ax.twinx()
                _ax2.set_ylabel("LR", color="gray")
                _ax2.tick_params(axis="y", colors="gray")
                _ax2.set_yscale("log")
            _ax2.plot(
                _subset["step"].to_numpy(),
                _lrs,
                "--",
                color="gray",
                alpha=0.3,
                linewidth=1,
            )

    _ax.set_xlabel("Step")
    _ax.set_ylabel("Training Loss (probe/ce)")
    _ax.set_title("Seed Variance Test: Loss vs Step (lr=0.03, Muon + free wins)")
    _ax.legend(loc="upper right")
    _ax.grid(True, alpha=0.3)
    _ax.spines[["right", "top"]].set_visible(False)
    _fig
    return


@app.cell
def _(mo, pl, seed_history_df):
    # Detect potential checkpoint resume points (large step gaps or loss spikes)
    def _detect_anomalies(df, seed):
        subset = df.filter(pl.col("seed") == seed).sort("step")
        if len(subset) < 2:
            return []
        steps = subset["step"].to_list()
        losses = subset["loss_pr"].to_list()
        anomalies = []
        for i in range(1, len(steps)):
            step_gap = steps[i] - steps[i - 1]
            loss_change = losses[i] - losses[i - 1]
            if step_gap > 100 or abs(loss_change) > 0.5:
                anomalies.append(
                    {
                        "seed": seed,
                        "step": steps[i],
                        "prev_step": steps[i - 1],
                        "step_gap": step_gap,
                        "loss_pr": losses[i],
                        "prev_loss_pr": losses[i - 1],
                        "loss_change": loss_change,
                    }
                )
        return anomalies


    _all_anomalies = []
    for _s in [1, 2, 3]:
        _all_anomalies.extend(_detect_anomalies(seed_history_df, _s))

    if _all_anomalies:
        _anomaly_df = pl.DataFrame(_all_anomalies)
        _output = mo.vstack(
            [
                mo.md("### Detected Anomalies (step gaps > 100 or loss change > 0.5)"),
                _anomaly_df,
            ]
        )
    else:
        _output = mo.md("No anomalies detected")
    _output
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Idea 1 results: EMA view of loss around resume points
    """)
    return


@app.cell
def _(np, pl, plt, seed_history_df):
    def _ema(values, alpha=0.1):
        ema = np.zeros_like(values)
        ema[0] = values[0]
        for i in range(1, len(values)):
            ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]
        return ema


    _fig, _axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 10), dpi=150, sharex=True)
    _colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}

    for _i, _seed in enumerate([1, 2, 3]):
        _ax = _axes[_i]
        _subset = seed_history_df.filter(pl.col("seed") == _seed).sort("timestamp")
        if len(_subset) == 0:
            continue

        _steps = _subset["step"].to_numpy()
        _loss_pr = _subset["loss_pr"].to_numpy()
        _loss_ce = _subset["loss_ce"].to_numpy()
        _timestamps = _subset["timestamp"].to_numpy()

        _ema_pr = _ema(_loss_pr, alpha=0.05)
        _ema_ce = _ema(_loss_ce, alpha=0.05)

        _ax.plot(
            _steps, _ema_pr, "-", color=_colors[_seed], linewidth=2, label="EMA loss_pr"
        )
        _ax.plot(
            _steps,
            _ema_ce,
            "--",
            color=_colors[_seed],
            linewidth=2,
            label="EMA loss_ce",
        )

        for _j in range(1, len(_steps)):
            if _steps[_j] <= _steps[_j - 1] and _timestamps[_j] - _timestamps[_j - 1] > 600:
                _ax.axvline(
                    _steps[_j],
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    label="resume" if _j == 1 else None,
                )

        _ax.set_ylabel(f"Loss (seed={_seed})")
        _ax.legend(loc="upper right")
        _ax.grid(True, alpha=0.3)

    _axes[-1].set_xlabel("Step")
    _fig.suptitle("Loss EMA - Red lines = resume points")
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    # Idea 2: Entropy Explains Loss Jump Direction

    ## Hypothesis

    The loss direction (up or down) after resume depends on which files end up first in the shuffled order. Files with different class distributions lead to different batch difficulty.

    If true, `tgt_entropy_frac` should correlate with the loss change direction at resume points.
    """)
    return


@app.cell
def _(pl, seed_test_runs, wandb):
    def _fetch_entropy_history(run_id: str):
        api = wandb.Api(timeout=60)
        run = api.run(f"samuelstevens/birdjepa/{run_id}")
        history = run.scan_history(
            keys=[
                "step",
                "train/probe",
                "train/ce",
                "train/tgt_entropy_frac",
                "_timestamp",
            ]
        )
        rows = []
        seed = run.config.get("seed")
        seed = int(seed) if seed is not None else -1
        for row in history:
            step = row.get("step")
            loss_pr = row.get("train/probe")
            loss_ce = row.get("train/ce")
            entropy = row.get("train/tgt_entropy_frac")
            ts = row.get("_timestamp")
            if step is not None and (loss_pr is not None or loss_ce is not None):
                rows.append(
                    {
                        "run_id": run_id,
                        "seed": seed,
                        "step": int(step),
                        "loss_pr": float(loss_pr) if loss_pr is not None else None,
                        "loss_ce": float(loss_ce) if loss_ce is not None else None,
                        "entropy_frac": float(entropy) if entropy else None,
                        "timestamp": ts,
                    }
                )
        schema = {
            "run_id": pl.Utf8,
            "seed": pl.Int64,
            "step": pl.Int64,
            "loss_pr": pl.Float64,
            "loss_ce": pl.Float64,
            "entropy_frac": pl.Float64,
            "timestamp": pl.Float64,
        }
        if not rows:
            return pl.DataFrame(schema=schema)
        return pl.DataFrame(rows, schema=schema)


    entropy_df = pl.concat([_fetch_entropy_history(rid) for rid in seed_test_runs])
    entropy_df
    return (entropy_df,)


@app.cell
def _(entropy_df, pl, plt):
    _fig, _axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 8), dpi=150, sharex=True)

    _colors = {"1": "#1f77b4", "2": "#ff7f0e", "3": "#2ca02c"}

    for _i, _seed in enumerate([1, 2, 3]):
        _ax = _axes[_i]
        _subset = entropy_df.filter(pl.col("seed") == _seed).sort("step")
        if len(_subset) == 0:
            continue

        _steps = _subset["step"].to_list()
        _loss_pr = _subset["loss_pr"].to_list()
        _loss_ce = _subset["loss_ce"].to_list()
        _entropies = _subset["entropy_frac"].to_list()

        _ax.plot(
            _steps, _loss_pr, "-", color=_colors[str(_seed)], label="loss_pr", alpha=0.8
        )
        _ax.plot(
            _steps,
            _loss_ce,
            "--",
            color=_colors[str(_seed)],
            label="loss_ce",
            alpha=0.8,
        )
        _ax2 = _ax.twinx()
        _ax2.plot(_steps, _entropies, ":", color="gray", label="entropy_frac", alpha=0.6)
        _ax.set_ylabel(f"Loss (seed={_seed})", color=_colors[str(_seed)])
        _ax2.set_ylabel("Entropy Frac", color="gray")
        _ax.grid(True, alpha=0.3)

    _axes[-1].set_xlabel("Step")
    _fig.suptitle("Loss vs Target Entropy Fraction (dotted=entropy)")
    _fig
    return


@app.cell
def _(entropy_df, mo, pl):
    def _find_resume_entropy(df, seed):
        subset = df.filter(pl.col("seed") == seed).sort("timestamp")
        if len(subset) < 2:
            return []
        steps = subset["step"].to_list()
        loss_pr = subset["loss_pr"].to_list()
        loss_ce = subset["loss_ce"].to_list()
        entropies = subset["entropy_frac"].to_list()
        timestamps = subset["timestamp"].to_list()

        resumes = []
        for i in range(1, len(steps)):
            if steps[i] <= steps[i - 1] and timestamps[i] - timestamps[i - 1] > 600:
                resumes.append(
                    {
                        "seed": seed,
                        "step_before": steps[i - 1],
                        "step_after": steps[i],
                        "loss_pr_before": loss_pr[i - 1],
                        "loss_pr_after": loss_pr[i],
                        "loss_pr_change": loss_pr[i] - loss_pr[i - 1],
                        "loss_ce_before": loss_ce[i - 1],
                        "loss_ce_after": loss_ce[i],
                        "loss_ce_change": loss_ce[i] - loss_ce[i - 1],
                        "entropy_before": entropies[i - 1],
                        "entropy_after": entropies[i],
                        "entropy_change": (entropies[i] - entropies[i - 1])
                        if entropies[i] and entropies[i - 1]
                        else None,
                    }
                )
        return resumes


    _all_resumes = []
    for _s in [1, 2, 3]:
        _all_resumes.extend(_find_resume_entropy(entropy_df, _s))

    if _all_resumes:
        _resume_df = pl.DataFrame(_all_resumes)
        _output = mo.vstack(
            [
                mo.md("### Resume Points: Loss vs Entropy Changes"),
                mo.md(
                    "If entropy_change correlates with loss_change direction, our hypothesis is supported."
                ),
                _resume_df,
            ]
        )
    else:
        _output = mo.md("No resume points detected")
    _output
    return


@app.cell
def _(entropy_df, np, pl, plt):
    def _ema(values, alpha=0.1):
        ema = np.zeros_like(values)
        ema[0] = values[0]
        for i in range(1, len(values)):
            ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]
        return ema


    _fig, _axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 10), dpi=150, sharex=True)
    _colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}

    for _i, _seed in enumerate([1, 2, 3]):
        _ax = _axes[_i]
        _subset = entropy_df.filter(pl.col("seed") == _seed).sort("timestamp")
        if len(_subset) == 0:
            continue

        _steps = _subset["step"].to_numpy()
        _loss_pr = _subset["loss_pr"].to_numpy()
        _loss_ce = _subset["loss_ce"].to_numpy()
        _timestamps = _subset["timestamp"].to_numpy()

        _ema_pr = _ema(_loss_pr, alpha=0.05)
        _ema_ce = _ema(_loss_ce, alpha=0.05)

        _ax.plot(
            _steps, _ema_pr, "-", color=_colors[_seed], linewidth=2, label="EMA loss_pr"
        )
        _ax.plot(
            _steps,
            _ema_ce,
            "--",
            color=_colors[_seed],
            linewidth=2,
            label="EMA loss_ce",
        )

        for _j in range(1, len(_steps)):
            if _steps[_j] <= _steps[_j - 1] and _timestamps[_j] - _timestamps[_j - 1] > 600:
                _ax.axvline(
                    _steps[_j],
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    label="resume" if _j == 1 else None,
                )

        _ax.set_ylabel(f"Loss (seed={_seed})")
        _ax.legend(loc="upper right")
        _ax.grid(True, alpha=0.3)

    _axes[-1].set_xlabel("Step")
    _fig.suptitle("Loss EMA - Red lines = resume points")
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    # Idea 3: Dataloader Reset Causes the Loss Jump

    ## Experiment

    Compare a run that resets the dataloader at step 1000 vs a control run with the same seed.
    """)
    return


@app.cell
def _(pl, wandb):
    reset_run_id = "9lm0bfgu"
    control_run_id = "knze87mn"
    run_labels = {
        reset_run_id: "reset_dataloader@1000",
        control_run_id: "control",
    }


    def _fetch_loss_history(run_id: str):
        api = wandb.Api(timeout=60)
        run = api.run(f"samuelstevens/birdjepa/{run_id}")
        history = run.scan_history(
            keys=["step", "train/probe", "train/ce", "_timestamp", "lr"]
        )
        rows = []
        for row in history:
            step = row.get("step")
            loss_pr = row.get("train/probe")
            loss_ce = row.get("train/ce")
            ts = row.get("_timestamp")
            lr = row.get("lr")
            if step is not None and (loss_pr is not None or loss_ce is not None):
                rows.append(
                    {
                        "run_id": run_id,
                        "label": run_labels[run_id],
                        "step": int(step),
                        "loss_pr": float(loss_pr) if loss_pr is not None else None,
                        "loss_ce": float(loss_ce) if loss_ce is not None else None,
                        "timestamp": ts,
                        "lr": float(lr) if lr is not None else None,
                    }
                )
        schema = {
            "run_id": pl.Utf8,
            "label": pl.Utf8,
            "step": pl.Int64,
            "loss_pr": pl.Float64,
            "loss_ce": pl.Float64,
            "timestamp": pl.Float64,
            "lr": pl.Float64,
        }
        if not rows:
            return pl.DataFrame(schema=schema)
        return pl.DataFrame(rows, schema=schema)


    reset_control_df = pl.concat(
        [_fetch_loss_history(rid) for rid in [reset_run_id, control_run_id]]
    )
    reset_control_df
    return reset_control_df, run_labels


@app.cell
def _(np, pl, plt, reset_control_df, run_labels):
    def _ema_loss(values, alpha=0.05):
        ema = np.zeros_like(values)
        ema[0] = values[0]
        for i in range(1, len(values)):
            ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]
        return ema


    _fig, _ax = plt.subplots(figsize=(12, 4), dpi=150)
    _ax2 = None
    _colors = {
        "reset_dataloader@1000": "#d62728",
        "control": "#1f77b4",
    }

    for _run_id, _label in run_labels.items():
        _subset = reset_control_df.filter(pl.col("run_id") == _run_id).sort("step")
        if len(_subset) == 0:
            continue

        _steps = _subset["step"].to_numpy()
        _loss_pr = _subset["loss_pr"].to_numpy()
        _loss_ce = _subset["loss_ce"].to_numpy()
        _lrs = _subset["lr"].cast(pl.Float64).to_numpy()
        _ema_pr = _ema_loss(_loss_pr, alpha=0.05)
        _ema_ce = _ema_loss(_loss_ce, alpha=0.05)

        _ax.plot(
            _steps,
            _ema_pr,
            "-",
            color=_colors[_label],
            linewidth=2,
            label=f"{_label} EMA loss_pr",
        )
        _ax.plot(
            _steps,
            _ema_ce,
            "--",
            color=_colors[_label],
            linewidth=2,
            label=f"{_label} EMA loss_ce",
        )
        _ax.plot(
            _steps,
            _loss_pr,
            "-",
            color=_colors[_label],
            alpha=0.15,
            linewidth=0.5,
        )
        _ax.plot(
            _steps,
            _loss_ce,
            "--",
            color=_colors[_label],
            alpha=0.15,
            linewidth=0.5,
        )
        if np.isfinite(_lrs).any():
            if _ax2 is None:
                _ax2 = _ax.twinx()
                _ax2.set_ylabel("LR", color="gray")
                _ax2.tick_params(axis="y", colors="gray")
                _ax2.set_yscale("log")
            _ax2.plot(
                _subset["step"].to_numpy(),
                _lrs,
                "--",
                color="gray",
                alpha=0.4,
                linewidth=1,
            )

    _ax.axvline(1000, color="black", linestyle="--", alpha=0.5, label="step=1000")
    _ax.set_xlabel("Step")
    _ax.set_ylabel("Loss (probe/ce)")
    _ax.set_title("EMA of Losses: Dataloader Reset vs Control")
    _ax.legend(loc="upper right")
    _ax.grid(True, alpha=0.3)
    _fig
    return


@app.cell
def _(np, pl, reset_control_df, run_labels):
    _rows = []
    for _run_id, _label in run_labels.items():
        _subset = reset_control_df.filter(pl.col("run_id") == _run_id).sort("step")
        if len(_subset) < 2:
            continue

        _steps = _subset["step"].to_numpy()
        _loss_pr = _subset["loss_pr"].to_numpy()
        _loss_ce = _subset["loss_ce"].to_numpy()
        _ema_pr = np.zeros_like(_loss_pr)
        _ema_ce = np.zeros_like(_loss_ce)
        _ema_pr[0] = _loss_pr[0]
        _ema_ce[0] = _loss_ce[0]
        for _i in range(1, len(_loss_pr)):
            _ema_pr[_i] = 0.05 * _loss_pr[_i] + 0.95 * _ema_pr[_i - 1]
            _ema_ce[_i] = 0.05 * _loss_ce[_i] + 0.95 * _ema_ce[_i - 1]

        _idx_after = np.searchsorted(_steps, 1000, side="left")
        if _idx_after == 0 or _idx_after >= len(_steps):
            continue
        _idx_before = _idx_after - 1

        _rows.append(
            {
                "run_id": _run_id,
                "label": _label,
                "step_before": int(_steps[_idx_before]),
                "step_after": int(_steps[_idx_after]),
                "loss_pr_before": float(_loss_pr[_idx_before]),
                "loss_pr_after": float(_loss_pr[_idx_after]),
                "loss_pr_change": float(_loss_pr[_idx_after] - _loss_pr[_idx_before]),
                "loss_ce_before": float(_loss_ce[_idx_before]),
                "loss_ce_after": float(_loss_ce[_idx_after]),
                "loss_ce_change": float(_loss_ce[_idx_after] - _loss_ce[_idx_before]),
                "ema_pr_before": float(_ema_pr[_idx_before]),
                "ema_pr_after": float(_ema_pr[_idx_after]),
                "ema_pr_change": float(_ema_pr[_idx_after] - _ema_pr[_idx_before]),
                "ema_ce_before": float(_ema_ce[_idx_before]),
                "ema_ce_after": float(_ema_ce[_idx_after]),
                "ema_ce_change": float(_ema_ce[_idx_after] - _ema_ce[_idx_before]),
            }
        )

    pl.DataFrame(_rows)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Results

    - Both runs show smooth loss_pr and loss_ce EMA through step 1000.
    - There is no meaningful discontinuity compared to control.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Idea 4: Resume-Only State Differences Cause the Loss Jump

    ## Hypothesis

    Resume-only state differences (PRNG state, optimizer interaction, shuffle buffer warmup, or checkpoint load ordering) cause the loss discontinuity.

    ## Experiment

    - Run the PRNG mode sweep (stateless / checkpointed / no_stochastic) with forced resumes.
    - Compare loss_pr, loss_ce, and norm metrics around the resume step using EMA with a configurable band.

    ## Results

    - no_stochastic looks closest to continuous in the EMA plots, but loss jumps still appear around resume.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## PRNG mode sweep (resume window)
    """)
    return


@app.cell
def _(pl, wandb):
    prng_runs = {
        "stateless": "qci0wkul",
        "checkpointed": "m9vkye34",
        "no_stochastic": "ep5sz12i",
    }


    def _fetch_prng_history(run_id: str, label: str):
        api = wandb.Api(timeout=60)
        run = api.run(f"samuelstevens/birdjepa/{run_id}")
        keys = [
            "step",
            "_step",
            "train/probe",
            "train/ce",
            "train/update_norm",
            "train/param_norm",
            "train/grad_norm",
            "train/enc_grad",
            "train/obj_grad",
            "train/probe_grad",
            "train/opt_state_l2",
            "train/opt_state_abs_mean",
            "train/opt_state_abs_max",
            "_timestamp",
            "lr",
        ]

        def _collect_rows(history):
            rows = []
            for row in history:
                step = row.get("step")
                if step is None:
                    step = row.get("_step")
                loss_pr = row.get("train/probe")
                loss_ce = row.get("train/ce")
                update_norm = row.get("train/update_norm")
                param_norm = row.get("train/param_norm")
                grad_norm = row.get("train/grad_norm")
                enc_grad = row.get("train/enc_grad")
                obj_grad = row.get("train/obj_grad")
                probe_grad = row.get("train/probe_grad")
                opt_state_l2 = row.get("train/opt_state_l2")
                opt_state_abs_mean = row.get("train/opt_state_abs_mean")
                opt_state_abs_max = row.get("train/opt_state_abs_max")
                ts = row.get("_timestamp")
                lr = row.get("lr")
                has_metrics = any(
                    value is not None
                    for value in (
                        loss_pr,
                        loss_ce,
                        update_norm,
                        param_norm,
                        grad_norm,
                        enc_grad,
                        obj_grad,
                        probe_grad,
                        opt_state_l2,
                        opt_state_abs_mean,
                        opt_state_abs_max,
                    )
                )
                if step is None or not has_metrics:
                    continue
                rows.append(
                    {
                        "run_label": label,
                        "run_id": run_id,
                        "step": int(step),
                        "loss_pr": float(loss_pr) if loss_pr is not None else None,
                        "loss_ce": float(loss_ce) if loss_ce is not None else None,
                        "update_norm": float(update_norm)
                        if update_norm is not None
                        else None,
                        "param_norm": float(param_norm) if param_norm is not None else None,
                        "grad_norm": float(grad_norm) if grad_norm is not None else None,
                        "enc_grad": float(enc_grad) if enc_grad is not None else None,
                        "obj_grad": float(obj_grad) if obj_grad is not None else None,
                        "probe_grad": float(probe_grad) if probe_grad is not None else None,
                        "opt_state_l2": float(opt_state_l2)
                        if opt_state_l2 is not None
                        else None,
                        "opt_state_abs_mean": float(opt_state_abs_mean)
                        if opt_state_abs_mean is not None
                        else None,
                        "opt_state_abs_max": float(opt_state_abs_max)
                        if opt_state_abs_max is not None
                        else None,
                        "timestamp": ts,
                        "lr": float(lr) if lr is not None else None,
                    }
                )
            return rows

        rows = _collect_rows(run.scan_history(keys=keys))
        if not rows:
            rows = _collect_rows(run.scan_history())
        schema = {
            "run_label": pl.Utf8,
            "run_id": pl.Utf8,
            "step": pl.Int64,
            "loss_pr": pl.Float64,
            "loss_ce": pl.Float64,
            "update_norm": pl.Float64,
            "param_norm": pl.Float64,
            "grad_norm": pl.Float64,
            "enc_grad": pl.Float64,
            "obj_grad": pl.Float64,
            "probe_grad": pl.Float64,
            "opt_state_l2": pl.Float64,
            "opt_state_abs_mean": pl.Float64,
            "opt_state_abs_max": pl.Float64,
            "timestamp": pl.Float64,
            "lr": pl.Float64,
        }
        if not rows:
            return pl.DataFrame(schema=schema)
        return pl.DataFrame(rows, schema=schema)


    prng_history_df = pl.concat(
        [_fetch_prng_history(run_id, label) for label, run_id in prng_runs.items()]
    )
    prng_history_df
    return prng_history_df, prng_runs


@app.cell
def _(pl, prng_history_df):
    if len(prng_history_df) == 0:
        prng_segment_df = prng_history_df
    else:
        prng_segment_df = (
            prng_history_df.sort("timestamp")
            .with_columns(
                (pl.col("step").diff().over("run_label") < 0)
                .fill_null(False)
                .cast(pl.Int64)
                .alias("resume_flag")
            )
            .with_columns(
                pl.col("resume_flag").cum_sum().over("run_label").alias("segment")
            )
            .drop("resume_flag")
        )
    prng_segment_df
    return (prng_segment_df,)


@app.cell
def _(mo, pl, prng_segment_df):
    if len(prng_segment_df) == 0:
        prng_steps = []
    else:
        prng_steps = (
            prng_segment_df.filter(pl.col("segment") > 0)
            .group_by(["run_label", "segment"])
            .agg(pl.col("step").min().alias("resume_step"))
            .select("resume_step")
            .unique()
            .sort("resume_step")
            .to_series()
            .to_list()
        )
    prng_default_step = prng_steps[0] if prng_steps else 500
    prng_center_step = mo.ui.dropdown(
        options=prng_steps or [prng_default_step],
        value=prng_default_step,
        label="PRNG resume step",
    )
    prng_window = mo.ui.slider(
        start=50,
        stop=1000,
        value=200,
        step=50,
        label="PRNG window",
    )
    mo.hstack([prng_center_step, prng_window])
    return prng_center_step, prng_window


@app.cell
def _(pl, prng_center_step, prng_segment_df, prng_window):
    prng_center = int(prng_center_step.value)
    prng_window_size = int(prng_window.value)
    prng_start_step = max(0, prng_center - prng_window_size)
    prng_stop_step = prng_center + prng_window_size
    prng_window_df = prng_segment_df.filter(
        (pl.col("step") >= prng_start_step) & (pl.col("step") <= prng_stop_step)
    )
    prng_window_df
    return prng_start_step, prng_stop_step, prng_window_df


@app.cell
def _(mo):
    band_mode = mo.ui.radio(
        options=["quantile", "std"],
        value="quantile",
        label="Band type",
    )
    quantile_options = {
        "0-100 (min/max)": (0.0, 1.0),
        "10-90": (0.1, 0.9),
        "25-75": (0.25, 0.75),
    }
    quantile_choice = mo.ui.dropdown(
        options=quantile_options,
        value="10-90",
        label="Quantile band",
    )
    window_size = mo.ui.slider(
        start=5,
        stop=100,
        value=20,
        step=5,
        label="Band window",
    )
    mo.hstack([band_mode, quantile_choice, window_size])
    return band_mode, quantile_choice, quantile_options, window_size


@app.cell
def _(
    band_mode,
    fig,
    mo,
    np,
    pl,
    plt,
    prng_center_step,
    prng_runs,
    prng_start_step,
    prng_stop_step,
    prng_window_df,
    quantile_choice,
    quantile_options,
    window_size,
):
    def _():
        mo.stop(len(prng_window_df) == 0, mo.md("No PRNG sweep data available."))

        labels = list(prng_runs.keys())

        metric_specs = [
            ("loss_pr", "Loss (probe)"),
            ("loss_ce", "Loss (ce)"),
            ("update_norm", "Update norm"),
            ("param_norm", "Param norm"),
            ("grad_norm", "Grad norm"),
            ("enc_grad", "Encoder grad"),
            ("obj_grad", "Objective grad"),
            ("probe_grad", "Probe grad"),
            ("opt_state_l2", "Opt state L2"),
            ("opt_state_abs_mean", "Opt state abs mean"),
            ("opt_state_abs_max", "Opt state abs max"),
        ]
        metrics = []
        for key, label in metric_specs:
            if key not in prng_window_df.columns:
                print(f"Key {key} not in df.")
                continue
            has_values = prng_window_df.select(pl.col(key).is_not_null().any()).item()
            if has_values:
                metrics.append((key, label))
            else:
                print(f"No values for key {key}.")
        mo.stop(len(metrics) == 0, mo.md("No PRNG sweep metrics available."))

        n_rows = len(metrics)
        n_cols = len(labels)
        _fig, _axes = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=(4.2 * n_cols, 2.2 * n_rows),
            dpi=150,
            sharex=True,
            sharey="row",
            layout="constrained",
        )
        if n_rows == 1 and n_cols == 1:
            _axes = np.array([[_axes]])
        elif n_rows == 1:
            _axes = np.array([_axes])
        elif n_cols == 1:
            _axes = np.array([[_ax] for _ax in _axes])

        _colors = ["#1f77b4", "#ff7f0e"]

        def _ema(values, alpha=0.1):
            ema = np.zeros_like(values)
            ema[0] = values[0]
            for i in range(1, len(values)):
                ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]
            return ema

        def _rolling_quantile_band(values, window, low_q, high_q):
            low_out = np.zeros_like(values)
            high_out = np.zeros_like(values)
            for i in range(len(values)):
                start = max(0, i - window + 1)
                window_vals = values[start : i + 1]
                low_out[i] = np.quantile(window_vals, low_q)
                high_out[i] = np.quantile(window_vals, high_q)
            return low_out, high_out

        def _rolling_std(values, window):
            out = np.zeros_like(values)
            for i in range(len(values)):
                start = max(0, i - window + 1)
                out[i] = np.std(values[start : i + 1])
            return out

        use_quantile = band_mode.value == "quantile"
        if isinstance(quantile_choice.value, tuple):
            low_q, high_q = quantile_choice.value
        else:
            low_q, high_q = quantile_options[quantile_choice.value]
        band_window = int(window_size.value)

        for (
            _row,
            (metric_key, metric_label),
        ) in enumerate(metrics):
            for _col, _label in enumerate(labels):
                _ax = _axes[_row, _col]
                _subset_run = prng_window_df.filter(pl.col("run_label") == _label)
                if len(_subset_run) == 0:
                    print("No subsets.")
                    continue
                for _segment in _subset_run["segment"].unique().to_list():
                    _subset = _subset_run.filter(pl.col("segment") == _segment).sort("step")
                    if len(_subset) == 0:
                        print(f"No subset in segment {_segment}.")
                        continue
                    _subset_metric = _subset.filter(pl.col(metric_key).is_not_null())
                    if len(_subset_metric) == 0:
                        print(f"No metric for {metric_key} in subset.")
                        continue
                    _steps = _subset_metric["step"].to_numpy()
                    _values = _subset_metric[metric_key].to_numpy()
                    _ema_values = _ema(_values, alpha=0.1)
                    if use_quantile:
                        _low, _high = _rolling_quantile_band(
                            _values, window=band_window, low_q=low_q, high_q=high_q
                        )
                    else:
                        _std = _rolling_std(_values, window=band_window)
                        _low = _ema_values - _std
                        _high = _ema_values + _std
                    _color = _colors[_segment % 2]
                    _ax.fill_between(
                        _steps,
                        _low,
                        _high,
                        color=_color,
                        alpha=0.2,
                        linewidth=0,
                    )
                    _label_text = f"seg {_segment}" if _row == 0 and _col == 0 else None
                    _ax.plot(
                        _steps,
                        _ema_values,
                        "-",
                        color=_color,
                        alpha=1.0,
                        label=_label_text,
                    )
                _ax.axvline(
                    prng_center_step.value, color="black", linestyle="--", alpha=0.5
                )
                _ax.set_xlim(prng_start_step, prng_stop_step)
                _ax.grid(True, alpha=0.3)
                if _row == 0:
                    _ax.set_title(_label)
                if _col == 0:
                    _ax.set_ylabel(metric_label)
                if _row == n_rows - 1:
                    _ax.set_xlabel("Step")

        if _axes[0, 0].get_legend_handles_labels()[0]:
            _axes[0, 0].legend(loc="upper right")

        if use_quantile:
            band_label = f"quantile {int(low_q * 100)}-{int(high_q * 100)}%"
        else:
            band_label = "std"
        fig.suptitle(f"PRNG mode sweep: EMA + {band_label} band (window={band_window})")
        resume_step = int(prng_center_step.value)
        out_fpath = f"docs/issues/006-resume-loss-jump/idea4-prng-step{resume_step}.png"
        fig.savefig(out_fpath, dpi=200)
        return fig


    _()
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Idea 5: Checkpoint Roundtrip Without Dataloader Restart

    ## Hypothesis

    If the loss jump is caused by checkpoint save/load itself (serialization errors, restore mismatches, or a subtle interaction with JIT/static state), then we should be able to reproduce the discontinuity by saving a checkpoint and immediately restoring it in-process, while continuing to consume batches from the same dataloader iterator.

    Conversely, if the in-process checkpoint roundtrip is smooth, then the loss jump must come from process-level resume effects (Rust loader restart, cold shuffle buffer, multithreaded nondeterminism, etc.), not Orbax restore.

    ## Experiment

    - 6 runs total (3 PRNG modes x 2 conditions)
    - Roundtrip runs perform in-process checkpoint save/restore at steps 500, 1000, 1500
    - Control runs have no roundtrips
    - 2000 steps per run
    """)
    return


@app.cell
def _(pl, wandb):
    idea5_runs = {
        "stateless_roundtrip": "gxny1ntz",
        "stateless_control": "fqjqlszz",
        "checkpointed_roundtrip": "c2q6ird2",
        "checkpointed_control": "938qmw48",
        "no_stochastic_roundtrip": "d5oxsq0w",
        "no_stochastic_control": "55w8cml1",
    }


    def _fetch_idea5_history(run_id: str, label: str):
        api = wandb.Api(timeout=60)
        run = api.run(f"samuelstevens/birdjepa/{run_id}")
        keys = [
            "step",
            "_step",
            "train/probe",
            "train/ce",
            "train/update_norm",
            "train/param_norm",
            "train/grad_norm",
            "train/opt_state_l2",
            "train/opt_state_abs_mean",
            "train/opt_state_abs_max",
            "_timestamp",
            "lr",
        ]
        rows = []
        for row in run.scan_history(keys=keys):
            step = row.get("step")
            if step is None:
                step = row.get("_step")
            loss_pr = row.get("train/probe")
            loss_ce = row.get("train/ce")
            update_norm = row.get("train/update_norm")
            param_norm = row.get("train/param_norm")
            grad_norm = row.get("train/grad_norm")
            opt_state_l2 = row.get("train/opt_state_l2")
            opt_state_abs_mean = row.get("train/opt_state_abs_mean")
            opt_state_abs_max = row.get("train/opt_state_abs_max")
            ts = row.get("_timestamp")
            lr = row.get("lr")
            has_metrics = any(
                v is not None
                for v in (loss_pr, loss_ce, update_norm, param_norm, grad_norm)
            )
            if step is None or not has_metrics:
                continue
            prng_mode = label.rsplit("_", 1)[0]
            condition = label.rsplit("_", 1)[1]
            rows.append(
                {
                    "run_label": label,
                    "run_id": run_id,
                    "prng_mode": prng_mode,
                    "condition": condition,
                    "step": int(step),
                    "loss_pr": float(loss_pr) if loss_pr is not None else None,
                    "loss_ce": float(loss_ce) if loss_ce is not None else None,
                    "update_norm": float(update_norm) if update_norm is not None else None,
                    "param_norm": float(param_norm) if param_norm is not None else None,
                    "grad_norm": float(grad_norm) if grad_norm is not None else None,
                    "opt_state_l2": float(opt_state_l2)
                    if opt_state_l2 is not None
                    else None,
                    "opt_state_abs_mean": float(opt_state_abs_mean)
                    if opt_state_abs_mean is not None
                    else None,
                    "opt_state_abs_max": float(opt_state_abs_max)
                    if opt_state_abs_max is not None
                    else None,
                    "timestamp": ts,
                    "lr": float(lr) if lr is not None else None,
                }
            )
        schema = {
            "run_label": pl.Utf8,
            "run_id": pl.Utf8,
            "prng_mode": pl.Utf8,
            "condition": pl.Utf8,
            "step": pl.Int64,
            "loss_pr": pl.Float64,
            "loss_ce": pl.Float64,
            "update_norm": pl.Float64,
            "param_norm": pl.Float64,
            "grad_norm": pl.Float64,
            "opt_state_l2": pl.Float64,
            "opt_state_abs_mean": pl.Float64,
            "opt_state_abs_max": pl.Float64,
            "timestamp": pl.Float64,
            "lr": pl.Float64,
        }
        if not rows:
            return pl.DataFrame(schema=schema)
        return pl.DataFrame(rows, schema=schema)


    idea5_df = pl.concat(
        [_fetch_idea5_history(run_id, label) for label, run_id in idea5_runs.items()]
    )
    idea5_df
    return (idea5_df,)


@app.cell
def _(idea5_df, mo, pl):
    idea5_run_summary = (
        idea5_df.group_by(["run_label", "prng_mode", "condition"])
        .agg(
            [
                pl.col("step").min().alias("min_step"),
                pl.col("step").max().alias("max_step"),
                pl.len().alias("n_rows"),
            ]
        )
        .sort("run_label")
    )
    mo.vstack(
        [
            mo.md("### Idea 5 Run Summary"),
            idea5_run_summary,
        ]
    )
    return


@app.cell
def _(idea5_df, mo, np, pl, plt):
    mo.stop(
        len(idea5_df) == 0,
        mo.md("No Idea 5 data available yet. Runs may still be in progress."),
    )

    _window_df = idea5_df.filter((pl.col("step") >= 0) & (pl.col("step") <= 1500))


    def _ema(values, alpha=0.1):
        ema = np.zeros_like(values)
        ema[0] = values[0]
        for i in range(1, len(values)):
            ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]
        return ema


    _prng_modes = ["stateless", "checkpointed"]
    _metric_specs = [
        ("loss_pr", "Loss (probe)"),
        ("loss_ce", "Loss (ce)"),
        ("update_norm", "Update norm"),
        ("param_norm", "Param norm"),
        ("grad_norm", "Grad norm"),
        ("opt_state_l2", "Opt state L2"),
        ("opt_state_abs_mean", "Opt state abs mean"),
        ("opt_state_abs_max", "Opt state abs max"),
    ]
    _metrics = []
    for _key, _label in _metric_specs:
        if _key in _window_df.columns:
            _has_values = _window_df.select(pl.col(_key).is_not_null().any()).item()
            if _has_values:
                _metrics.append((_key, _label))

    _fig, _axes = plt.subplots(
        nrows=len(_metrics),
        ncols=len(_prng_modes),
        figsize=(4 * len(_prng_modes), 2.5 * len(_metrics)),
        dpi=150,
        sharex=True,
        sharey="row",
        layout="constrained",
    )
    if len(_metrics) == 1 and len(_prng_modes) == 1:
        _axes = np.array([[_axes]])
    elif len(_metrics) == 1:
        _axes = np.array([_axes])
    elif len(_prng_modes) == 1:
        _axes = np.array([[_ax] for _ax in _axes])

    _colors = {"roundtrip": "#d62728", "control": "#1f77b4"}

    for _row, (_metric_key, _metric_label) in enumerate(_metrics):
        for _col, _prng_mode in enumerate(_prng_modes):
            _ax = _axes[_row, _col]
            for _condition in ["roundtrip", "control"]:
                _subset = _window_df.filter(
                    (pl.col("prng_mode") == _prng_mode)
                    & (pl.col("condition") == _condition)
                ).sort("step")
                if len(_subset) == 0:
                    continue
                _subset_metric = _subset.filter(pl.col(_metric_key).is_not_null())
                if len(_subset_metric) == 0:
                    continue
                _steps = _subset_metric["step"].to_numpy()
                _values = _subset_metric[_metric_key].to_numpy()
                _ema_values = _ema(_values, alpha=0.1)
                _ax.plot(
                    _steps,
                    _values,
                    "-",
                    color=_colors[_condition],
                    alpha=0.2,
                    linewidth=0.5,
                )
                _ax.plot(
                    _steps,
                    _ema_values,
                    "-",
                    color=_colors[_condition],
                    alpha=1.0,
                    linewidth=1.5,
                    label=_condition if _row == 0 else None,
                )
            _ax.axvline(500, color="black", linestyle="--", alpha=0.5)
            _ax.axvline(1000, color="black", linestyle="--", alpha=0.5)
            _ax.set_xlim(0, 1500)
            _ax.grid(True, alpha=0.3)
            if _row == 0:
                _ax.set_title(_prng_mode)
                _ax.legend(loc="upper right")
            if _col == 0:
                _ax.set_ylabel(_metric_label)
            if _row == len(_metrics) - 1:
                _ax.set_xlabel("Step")

    _fig.suptitle("Idea 5: Roundtrip vs Control (steps 0-1500)")
    _out_fpath = "docs/issues/006-resume-loss-jump/idea5-roundtrip.png"
    _fig.savefig(_out_fpath, dpi=200)

    _metric_cols = [k for k, _ in _metrics]
    _csv_df = (
        _window_df.filter(pl.col("prng_mode").is_in(_prng_modes))
        .select(["prng_mode", "condition", "step"] + _metric_cols)
        .sort(["prng_mode", "condition", "step"])
    )
    _csv_fpath = "docs/issues/006-resume-loss-jump/idea5-roundtrip.csv"
    _csv_df.write_csv(_csv_fpath)

    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Idea 5 Results

    The in-process checkpoint roundtrip (save + immediate restore without restarting the process) shows **no discontinuity** at steps 500 and 1000. The roundtrip (red) and control (blue) curves track each other closely.

    **Conclusion:** Checkpoint save/restore itself is not causing the loss jump. The issue must be related to process-level resume effects (dataloader restart, shuffle buffer refill, batch ordering, etc.).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Idea 6: IID Dataloader Diagnostics

    ## Hypothesis

    The loss jump at checkpoint resume is caused by non-IID batch sampling behavior in the dataloader. By tracking detailed index-level and bucket-level statistics, we can detect departures from true IID sampling.

    ## Experiment

    - 3 runs with seeds 1, 2, 3
    - 14 new dataloader diagnostic metrics:
      - Bucket chi-squared and ACF (mod, div, hash)
      - Shard chi-squared and ACF
      - Index uniqueness, overlap, sliding window stats
    - Compare metric behavior around resume points
    """)
    return


@app.cell
def _(mo):
    button = mo.ui.run_button(label="Refresh DataFrame")

    idea6_runs = {
        "seed1": "eju0jfrh",
        "seed2": "hvgpkgyx",
        "seed3": "ywi8293i",
    }

    button
    return button, idea6_runs


@app.cell
def _(button, idea6_runs, mo, pl, wandb):
    mo.stop(not button.value)

    # Mapping: column_name -> wandb_key
    field_map = {
        "lr": "lr",
        "loss_pr": "train/probe",
        "loss_ce": "train/ce",
        "update_norm": "train/update_norm",
        "param_norm": "train/param_norm",
        "grad_norm": "train/grad_norm",
        "bucket_chi_sq_mod": "dataloader/bucket_chi_sq_mod",
        "bucket_chi_sq_div": "dataloader/bucket_chi_sq_div",
        "bucket_chi_sq_hash": "dataloader/bucket_chi_sq_hash",
        "bucket_acf_mod_lag1": "dataloader/bucket_acf_mod_lag1",
        "bucket_acf_div_lag1": "dataloader/bucket_acf_div_lag1",
        "bucket_acf_hash_lag1": "dataloader/bucket_acf_hash_lag1",
        "shard_chi_sq": "dataloader/shard_chi_sq",
        "shard_acf_lag1": "dataloader/shard_acf_lag1",
        "n_unique_indices": "dataloader/n_unique_indices",
        "overlap_exact": "dataloader/overlap_exact",
        "overlap_bucket": "dataloader/overlap_bucket",
        "n_unique_window": "dataloader/n_unique_window",
        "expected_unique_iid": "dataloader/expected_unique_iid",
        "actual_batch_size": "dataloader/actual_batch_size",
    }

    schema = {
        "run_label": pl.Utf8,
        "run_id": pl.Utf8,
        "seed": pl.Int64,
        "step": pl.Int64,
        "timestamp": pl.Float64,
        **{col: pl.Float64 for col in field_map},
    }


    def _get(row, key):
        v = row.get(key)
        return float(v) if v is not None else None


    _api = wandb.Api(timeout=60)


    def _fetch_idea6_history(run_id: str, label: str):
        try:
            run = _api.run(f"samuelstevens/birdjepa/{run_id}")
        except wandb.errors.errors.CommError:
            return pl.DataFrame(schema=schema)

        seed = run.config.get("seed")
        seed = int(seed) if seed is not None else -1
        keys = ["step", "_step", "_timestamp"] + list(field_map.values())
        rows = []
        for row in run.scan_history(keys=keys):
            step = row.get("step") or row.get("_step")
            if step is None:
                continue
            row_data = {
                "run_label": label,
                "run_id": run_id,
                "seed": seed,
                "step": int(step),
                "timestamp": row.get("_timestamp"),
            }
            for col, key in field_map.items():
                row_data[col] = _get(row, key)
            rows.append(row_data)
        if not rows:
            return pl.DataFrame(schema=schema)
        return pl.DataFrame(rows, schema=schema)


    def flattened(dct: dict) -> dict[str, object]:
        new = {}
        for key, value in dct.items():
            if isinstance(value, dict):
                for nested_key, nested_value in flattened(value).items():
                    new[key + "." + nested_key] = nested_value
                continue

            if isinstance(value, list):
                new[key] = str(value)
                continue

            new[key] = value

        return new


    def _fetch_idea6_cfg(run_id):
        try:
            run = _api.run(f"samuelstevens/birdjepa/{run_id}")
        except wandb.errors.errors.CommError:
            return {}

        cfg = run.config
        cfg = {"run_id": run_id, **flattened(dict(cfg))}
        for key in ("tags", "test_data.augmentations"):
            cfg.pop(key)
        return cfg


    idea6_df = pl.concat(
        [_fetch_idea6_history(run_id, label) for label, run_id in idea6_runs.items()]
    ).with_columns(
        (pl.col("bucket_chi_sq_div") / 255).alias("bucket_chi_sq_div_norm"),
        (pl.col("bucket_chi_sq_mod") / 255).alias("bucket_chi_sq_mod_norm"),
        (pl.col("bucket_chi_sq_hash") / 255).alias("bucket_chi_sq_hash_norm"),
    )

    idea6_cfg_df = pl.DataFrame(
        _fetch_idea6_cfg(run_id) for _, run_id in idea6_runs.items()
    )
    idea6_cfg_df.write_csv("docs/issues/006-resume-loss-jump/idea6-config.csv")

    # Add segment tracking: segment increments each time step goes backwards (resume)
    if len(idea6_df) > 0:
        idea6_df = (
            idea6_df.sort(["seed", "timestamp"])
            .with_columns(
                # Resume = step goes backwards AND timestamp gap > 600s
                (
                    pl.col("step").diff().over("seed") < 0
                    # & (pl.col("timestamp").diff().over("seed") > 30)
                )
                .fill_null(False)
                .cast(pl.Int64)
                .alias("_resume_flag")
            )
            .with_columns(pl.col("_resume_flag").cum_sum().over("seed").alias("segment"))
            .drop("_resume_flag")
        )

    mo.vstack([idea6_df, idea6_cfg_df])
    return (idea6_df,)


@app.cell
def _(mo):
    seed1_checkbox = mo.ui.checkbox(value=True, label="Seed 1")
    seed2_checkbox = mo.ui.checkbox(value=True, label="Seed 2")
    seed3_checkbox = mo.ui.checkbox(value=True, label="Seed 3")
    mo.hstack([seed1_checkbox, seed2_checkbox, seed3_checkbox])
    return seed1_checkbox, seed2_checkbox, seed3_checkbox


@app.cell
def _(seed1_checkbox, seed2_checkbox, seed3_checkbox):
    # Build list of enabled seeds
    enabled_seeds = []
    if seed1_checkbox.value:
        enabled_seeds.append(1)
    if seed2_checkbox.value:
        enabled_seeds.append(2)
    if seed3_checkbox.value:
        enabled_seeds.append(3)
    return (enabled_seeds,)


@app.cell
def _(beartype, tp):
    @beartype.beartype
    class MetricSpec(tp.NamedTuple):
        key: str
        label: str
        ylim: tuple[float, float] | None = None
        expected: float | int | None = None
    return (MetricSpec,)


@app.cell
def _(MetricSpec, enabled_seeds, idea6_df, mo, np, pl, plt):
    mo.stop(
        len(idea6_df) == 0,
        mo.md("No Idea 6 data available yet. Runs may still be in progress."),
    )


    def _():
        # IID baseline constants
        n_examples = 528_422  # dataset size
        bsz = 2048  # batch size
        n_shards = 885
        n_buckets_overlap = 4096
        W = 100  # sliding window length

        # Derived expected values under IID sampling
        # E[min(Pois(0.5), Pois(0.5))] ≈ 0.163 (for bucket overlap with lambda=B/K=0.5)
        e_min_pois_half = 0.163

        metric_specs: list[MetricSpec] = [
            MetricSpec("loss_pr", "Loss (probe)"),
            MetricSpec("loss_ce", "Loss (ce)"),
            MetricSpec("update_norm", "Update norm"),
            MetricSpec("param_norm", "Param norm"),
            MetricSpec("grad_norm", "Grad norm"),
            MetricSpec("actual_batch_size", "Actual batch size", expected=bsz),
            MetricSpec("bucket_chi_sq_mod_norm", "Bucket $\\chi^2$ (mod)", expected=1),
            MetricSpec("bucket_acf_mod_lag1", "Bucket ACF mod", (-1.0, 1.0), expected=0),
            MetricSpec("bucket_chi_sq_div_norm", "Bucket $\\chi^2$ (div)", expected=1),
            MetricSpec("bucket_acf_div_lag1", "Bucket ACF div", (-1.0, 1.0), expected=0),
            MetricSpec("bucket_chi_sq_hash_norm", "Bucket $\\chi^2$ (hash)", expected=1),
            MetricSpec(
                "bucket_acf_hash_lag1", "Bucket ACF hash", (-1.0, 1.0), expected=0
            ),
            MetricSpec("shard_chi_sq", "Shard $\\chi^2$", expected=float(n_shards - 1)),
            MetricSpec("shard_acf_lag1", "Shard ACF", (-1.0, 1.0), expected=0),
            MetricSpec(
                "n_unique_indices",
                "Unique indices",
                expected=n_examples * (1 - np.exp(-bsz / n_examples)),
            ),
            MetricSpec("overlap_exact", "Overlap (exact)", expected=bsz**2 / n_examples),
            MetricSpec(
                "overlap_bucket",
                "Overlap (bucket)",
                expected=n_buckets_overlap * e_min_pois_half,
            ),
            MetricSpec(
                "n_unique_window",
                "Unique (window)",
                expected=n_examples * (1 - np.exp(-W * bsz / n_examples)),
            ),
            MetricSpec("expected_unique_iid", "Expected unique (IID)"),
        ]

        idea6_metrics = []
        for metric in metric_specs:
            if metric.key in idea6_df.columns:
                has_values = idea6_df.select(pl.col(metric.key).is_not_null().any()).item()
                if has_values:
                    idea6_metrics.append(metric)

        if len(idea6_metrics) == 0:
            return mo.md("No metrics with data found.")

        n_metrics = len(idea6_metrics)
        n_cols = 2
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=(5 * n_cols, 2.5 * n_rows),
            dpi=100,
            sharex=True,
            layout="constrained",
        )
        axes = axes.flatten()

        # Two colors per seed: [even_segment, odd_segment]
        _colors = {
            1: ("#a6cee3", "#1f78b4"),
            2: ("#b2df8a", "#33a02c"),
            3: ("#fb9a99", "#e31a1c"),
        }

        # Get resume steps for each seed
        resume_steps_by_seed: dict[int, list[int]] = {}
        for seed in enabled_seeds:
            if "segment" in idea6_df.columns:
                resume_steps = (
                    idea6_df.filter((pl.col("seed") == seed) & (pl.col("segment") > 0))
                    .group_by("segment")
                    .agg(pl.col("step").min().alias("resume_step"))
                    .sort("segment")["resume_step"]
                    .to_list()
                )
                resume_steps_by_seed[seed] = resume_steps
            else:
                resume_steps_by_seed[seed] = []

        def _ema(values, alpha=0.1):
            ema = np.zeros_like(values)
            ema[0] = values[0]
            for i in range(1, len(values)):
                ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]
            return ema

        for idx, metric in enumerate(idea6_metrics):
            ax = axes[idx]
            for seed in enabled_seeds:
                _subset_seed = idea6_df.filter(pl.col("seed") == seed)
                if len(_subset_seed) == 0:
                    continue
                # Plot each segment separately
                segments = sorted(_subset_seed["segment"].unique().to_list())
                for seg_idx, segment in enumerate(segments):
                    _subset = _subset_seed.filter(pl.col("segment") == segment).sort("step")
                    _subset_metric = _subset.filter(pl.col(metric.key).is_not_null())
                    if len(_subset_metric) == 0:
                        continue
                    _steps = _subset_metric["step"].to_numpy()
                    _values = _subset_metric[metric.key].to_numpy()
                    _ema_values = _ema(_values, alpha=0.1)
                    # Alternate color for segments (darker for even, lighter for odd)
                    seg_color = _colors[seed][segment % 2]
                    ax.plot(
                        _steps,
                        _values,
                        "-",
                        color=seg_color,
                        alpha=0.15,
                        linewidth=0.5,
                    )
                    ax.plot(
                        _steps,
                        _ema_values,
                        "-",
                        color=seg_color,
                        alpha=1.0,
                        linewidth=1.5,
                        label=f"seed={seed}" if seg_idx == 0 else None,
                    )
                # Draw vertical lines at resume points (use darker color, different dash pattern per seed)
                _dashes = {
                    1: (5, 2),
                    2: (2, 2),
                    3: (5, 2, 2, 2),
                }  # solid-ish, dotted, dash-dot
                for resume_step in resume_steps_by_seed[seed]:
                    ax.axvline(
                        resume_step,
                        color=_colors[seed][0],
                        linestyle="--",
                        dashes=_dashes[seed],
                        alpha=0.8,
                        linewidth=1.5,
                    )

            ax.set_ylabel(metric.label)
            if metric.ylim is not None:
                ax.set_ylim(*metric.ylim)
            if metric.expected is not None:
                ax.axhline(
                    metric.expected,
                    color="black",
                    linestyle="--",
                    alpha=0.5,
                    linewidth=1,
                    label="Expected",
                )
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Step")
            ax.xaxis.set_tick_params(labelbottom=True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(loc="upper right")

        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle("Idea 6: IID Dataloader Diagnostics")
        fig.savefig(
            "docs/issues/006-resume-loss-jump/idea6-dataloader-metrics.png", dpi=200
        )

        # Save CSV with all metrics
        metric_cols = [m.key for m in idea6_metrics]
        csv_df = (
            idea6_df.filter(pl.col("seed").is_in(enabled_seeds))
            .select(["seed", "segment", "step", "timestamp"] + metric_cols)
            .sort(["seed", "segment", "step"])
        )
        csv_df.write_csv("docs/issues/006-resume-loss-jump/idea6-dataloader.csv")

        return fig


    _()
    return


@app.cell
def _(enabled_seeds, idea6_df, mo, np, pl, plt):
    mo.stop(
        len(idea6_df) == 0,
        mo.md("No Idea 6 data available yet."),
    )

    # Get all resume steps across enabled seeds
    all_resume_steps = set()
    for seed in enabled_seeds:
        if "segment" in idea6_df.columns:
            steps = (
                idea6_df.filter((pl.col("seed") == seed) & (pl.col("segment") > 0))
                .group_by("segment")
                .agg(pl.col("step").min().alias("resume_step"))["resume_step"]
                .to_list()
            )
            all_resume_steps.update(steps)

    if not all_resume_steps:
        mo.stop(True, mo.md("No resume points detected."))

    _colors = {
        1: ("#a6cee3", "#1f78b4"),
        2: ("#b2df8a", "#33a02c"),
        3: ("#fb9a99", "#e31a1c"),
    }

    zoom_metrics = [
        ("loss_pr", "Loss (probe)"),
        ("loss_ce", "Loss (ce)"),
        ("shard_chi_sq", "Shard $\\chi^2$"),
        ("bucket_chi_sq_div_norm", "Bucket $\\chi^2$ div (norm)"),
        ("shard_acf_lag1", "Shard ACF lag1"),
        ("bucket_acf_div_lag1", "Bucket ACF div lag1"),
    ]


    def _ema(values, alpha=0.1):
        ema = np.zeros_like(values)
        ema[0] = values[0]
        for i in range(1, len(values)):
            ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]
        return ema


    figs = []
    for r in sorted(all_resume_steps):
        fig, axes = plt.subplots(
            nrows=len(zoom_metrics),
            ncols=1,
            figsize=(6, 2.5 * len(zoom_metrics)),
            dpi=150,
            sharex=True,
            layout="constrained",
        )

        for idx, (metric_key, metric_label) in enumerate(zoom_metrics):
            ax = axes[idx]
            for seed in enabled_seeds:
                _subset_seed = idea6_df.filter(
                    (pl.col("seed") == seed)
                    & (pl.col("step") >= r - 250)
                    & (pl.col("step") <= r + 250)
                )
                if len(_subset_seed) == 0 or metric_key not in _subset_seed.columns:
                    continue
                # Plot each segment separately
                segments = sorted(_subset_seed["segment"].unique().to_list())
                for seg_idx, segment in enumerate(segments):
                    _subset = _subset_seed.filter(pl.col("segment") == segment).sort("step")
                    _subset_metric = _subset.filter(pl.col(metric_key).is_not_null())
                    if len(_subset_metric) == 0:
                        continue
                    _steps = _subset_metric["step"].to_numpy()
                    _values = _subset_metric[metric_key].to_numpy()
                    _ema_values = _ema(_values, alpha=0.1)
                    seg_color = _colors[seed][segment % 2]
                    ax.plot(
                        _steps,
                        _values,
                        "-",
                        color=seg_color,
                        alpha=0.15,
                        linewidth=0.5,
                    )
                    ax.plot(
                        _steps,
                        _ema_values,
                        "-",
                        color=seg_color,
                        alpha=1.0,
                        linewidth=1.5,
                        label=f"seed={seed}" if idx == 0 and seg_idx == 0 else None,
                    )

            ax.axvline(r, color="black", linestyle="--", alpha=0.8, linewidth=1.5)
            ax.set_xlim(r - 250, r + 250)
            ax.set_ylabel(metric_label)
            ax.grid(True, alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)

        axes[-1].set_xlabel("Step")
        axes[0].legend(loc="upper right")
        fig.suptitle(f"Idea 6: Zoom around resume step {r}")
        fig.savefig(f"docs/issues/006-resume-loss-jump/idea6-zoom-resume{r}.png", dpi=150)
        figs.append(fig)
        plt.close(fig)

    mo.md(f"Saved {len(figs)} zoom figures for resume steps: {sorted(all_resume_steps)}")
    return (fig,)


if __name__ == "__main__":
    app.run()
