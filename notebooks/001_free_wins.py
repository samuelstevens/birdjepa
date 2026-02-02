import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import polars as pl
    import glob
    from pathlib import Path
    import wandb
    import numpy as np

    return Path, glob, mo, np, pl, plt, wandb


@app.cell
def _(mo):
    mo.md("""
    # Free Wins Analysis

    This notebook compares optimizer and architecture choices:
    1. AdamW (baseline architecture) vs Muon (with "free wins": RoPE, QK-norm, SwiGLU, LayerScale)
    2. Learning rate vs training loss
    3. Learning rate vs benchmark performance (cmAP)
    """)
    return


@app.cell
def _(pl, wandb):
    # Specific run IDs from the 01/25/2026 sweeps
    adamw_runs = {
        "wexdzk6k": 3e-4,
        "lw5lzgmt": 1e-3,
        "650nxn08": 3e-3,
        "xwksdgtw": 1e-2,
        "yk68ws1y": 3e-2,
    }
    muon_runs = {
        "8c77us66": 3e-2,
        "6n9menos": 1e-1,
        "g3cz884x": 3e-1,
    }

    def _fetch_sweep_results(run_ids: dict, optimizer: str):
        api = wandb.Api()
        rows = []
        for run_id, lr in run_ids.items():
            run = api.run(f"samuelstevens/birdjepa/{run_id}")
            loss = run.summary.get("train/probe")
            step = run.summary.get("step")
            rows.append({
                "run_id": run_id,
                "optimizer": optimizer,
                "lr": lr,
                "final_loss": float(loss) if loss else None,
                "final_step": step,
            })
        return pl.DataFrame(rows)

    sweep_df = pl.concat([
        _fetch_sweep_results(adamw_runs, "adamw"),
        _fetch_sweep_results(muon_runs, "muon"),
    ])
    sweep_df
    return (sweep_df,)


@app.cell
def _(plt, sweep_df):
    # Plot: LR vs Final Loss comparison for AdamW vs Muon sweeps
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    colors = {"adamw": "#1f77b4", "muon": "#ff7f0e"}
    labels = {"adamw": "AdamW (baseline)", "muon": "Muon + free wins"}

    for opt in ["adamw", "muon"]:
        subset = sweep_df.filter(sweep_df["optimizer"] == opt).sort("lr")
        lrs = subset["lr"].to_list()
        losses = subset["final_loss"].to_list()
        ax.plot(
            lrs,
            losses,
            "o-",
            color=colors[opt],
            label=labels[opt],
            markersize=8,
            linewidth=2,
        )

        # Annotate best point
        best_i = losses.index(min(losses))
        ax.annotate(
            f"lr={lrs[best_i]:.0e}\nloss={losses[best_i]:.2f}",
            (lrs[best_i], losses[best_i]),
            textcoords="offset points",
            xytext=(10, 10),
            fontsize=9,
            color=colors[opt],
        )

    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Final Training Loss")
    ax.set_title("LR vs Final Loss: AdamW vs Muon @ 10K Steps")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines[["right", "top"]].set_visible(False)
    fig
    return


@app.cell
def _(mo, sweep_df):
    # Summary table
    best_adamw = (
        sweep_df.filter(sweep_df["optimizer"] == "adamw").sort("final_loss").head(1)
    )
    best_muon = (
        sweep_df.filter(sweep_df["optimizer"] == "muon").sort("final_loss").head(1)
    )

    mo.md(f"""
    ## Summary: Best LR per Optimizer

    | Optimizer | Best LR | Final Loss | Run ID |
    |-----------|---------|------------|--------|
    | **Muon + free wins** | **{best_muon["lr"][0]:.0e}** | **{best_muon["final_loss"][0]:.2f}** | {best_muon["run_id"][0]} |
    | AdamW (baseline) | {best_adamw["lr"][0]:.0e} | {best_adamw["final_loss"][0]:.2f} | {best_adamw["run_id"][0]} |

    **Muon optimal LR is {best_muon["lr"][0] / best_adamw["lr"][0]:.0f}x higher than AdamW optimal LR.**
    """)
    return


@app.cell
def _(Path):
    # Load checkpoint metadata to map run IDs to configs
    ckpt_dir = Path("/fs/ess/PAS2136/samuelstevens/birdjepa/checkpoints")
    return ckpt_dir


@app.cell
def _(pl, wandb):
    # Fetch all training data from wandb (batch query - replaces log parsing)
    def _make_wandb_df(runs):
        rows = []
        for run in runs:
            opt = run.config.get("optimizer")
            row = {
                "run_id": run.id,
                "optimizer": opt if opt else "adam",  # null optimizer = adam
                "lr": run.config.get("lr"),
                "n_steps": run.config.get("n_steps"),
            }
            summary = run.summary
            row["final_step"] = summary.get("step")
            _loss = summary.get("train/probe")
            row["final_loss"] = float(_loss) if _loss is not None else None
            _ce = summary.get("train/ce")
            row["final_ce"] = float(_ce) if _ce is not None else None
            rows.append(row)
        return pl.DataFrame(rows)

    wandb_df = _make_wandb_df(wandb.Api().runs(path="samuelstevens/birdjepa"))
    wandb_df = wandb_df.filter(
        pl.col("lr").is_not_null()
        & pl.col("final_step").is_not_null()
        & pl.col("final_loss").is_not_null()
    )
    wandb_df
    return (wandb_df,)


@app.cell
def _(pl, plt, wandb_df):
    # Plot: LR vs Final Loss for each optimizer, color by final step
    _optimizers = sorted(wandb_df.get_column("optimizer").unique().to_list())
    _optimizers = [o for o in _optimizers if o != "unknown"]

    _fig, _axes = plt.subplots(
        nrows=1,
        ncols=len(_optimizers),
        figsize=(5 * len(_optimizers), 4),
        dpi=150,
        squeeze=False,
        sharey=True,
        sharex=True,
    )
    _axes = _axes.reshape(-1)

    for _opt, _ax in zip(_optimizers, _axes):
        _subset = wandb_df.filter(pl.col("optimizer") == _opt)
        _lrs, _losses, _steps = (
            _subset.select("lr", "final_loss", "final_step").drop_nulls().to_numpy().T
        )
        _sc = _ax.scatter(_lrs, _losses, c=_steps, alpha=0.5, cmap="viridis")
        _ax.set_xscale("log")
        _ax.set_yscale("log")
        _ax.set_xlabel("Learning Rate")
        _ax.set_ylabel("Final Loss")
        _ax.set_title(f"{_opt} (n={len(_lrs)})")
        _ax.grid(True, alpha=0.3)
        _ax.spines[["right", "top"]].set_visible(False)

    _fig.colorbar(_sc, ax=_axes, label="Final Step")
    _fig.suptitle("LR vs Final Loss by Optimizer (color = final step)")
    _fig
    return


@app.cell
def _(glob, pl):
    # Load all benchmark results (use diagonal concat to handle schema differences)
    result_files = glob.glob("results/raw/*.parquet")
    all_results = pl.concat(
        [pl.read_parquet(f) for f in result_files], how="diagonal_relaxed"
    )

    # Filter to BirdJEPA results only and extract run ID from model_ckpt
    _birdjepa_raw = all_results.filter(
        (pl.col("model_org") == "birdjepa") & (pl.col("clf") == "linear")
    ).with_columns(pl.col("model_ckpt").str.split("/").list.last().alias("run_id"))

    # Create clean benchmark dataframe: run_id, task_name -> cmap
    benchmark_df = _birdjepa_raw.select([
        "run_id",
        "task_name",
        "cmap",
        "clf",
        "n_train",
    ])
    benchmark_df
    return (benchmark_df,)


@app.cell
def _(joined_df):
    ", ".join(str(i) for i in joined_df.get_column("lr").unique().to_list())
    return


@app.cell
def _(benchmark_df, wandb_df):
    # Join pretraining and benchmark data to analyze lr -> benchmark performance
    joined_df = benchmark_df.join(wandb_df, on="run_id", how="left")
    joined_df
    return (joined_df,)


@app.cell
def _(mo):
    mo.md("""
    ## Benchmark Performance by Learning Rate

    Once we have benchmark results for multiple checkpoints, we can plot learning rate vs mean cmAP across tasks.

    **Current status:** We need to run benchmarks on all checkpoints to populate this analysis.
    """)
    return


@app.cell
def _(joined_df, mo, pl, plt):
    def _():
        # Plot 2: Learning Rate vs Benchmark Performance (using joined data)
        by_lr = (
            joined_df
            .filter(pl.col("run_id").is_not_null())
            .group_by("run_id", "n_train")
            .agg([
                pl.col("cmap").mean().alias("mean_cmap"),
                pl.col("cmap").std().alias("std_cmap"),
                pl.col("cmap").count().alias("n_tasks"),
                pl.col("lr").first().alias("lr"),
            ])
            .sort("lr")
        )

        assert len(by_lr.filter(pl.col("n_tasks") > 8)) == 0
        by_lr = by_lr.filter(pl.col("n_tasks") == 8)

        if len(by_lr) == 0:
            return mo.md("No benchmark results with matching pretraining data yet")

        lrs = by_lr["lr"].to_list()
        mean_cmaps = by_lr["mean_cmap"].to_list()

        fig, axes = plt.subplots(
            nrows=2,
            ncols=2,
            sharey=True,
            sharex=True,
            figsize=(7, 5),
            dpi=150,
            layout="constrained",
        )
        axes = axes.reshape(-1)

        for n_train, ax in zip([1, 10, 100, -1], axes):
            lrs, mean_cmaps = (
                by_lr
                .filter(pl.col("n_train") == n_train)
                .select("lr", "mean_cmap")
                .to_numpy()
                .T
            )
            ax.scatter(lrs, mean_cmaps, alpha=0.3)
            ax.set_xscale("log")
            ax.set_xlabel("Learning Rate")
            ax.set_ylabel("Mean cmAP")
            ax.grid(True, alpha=0.3)
            ax.spines[["right", "top"]].set_visible(False)
            title = f"n={n_train}" if n_train > 0 else "All"
            ax.set_title(title)

        fig.suptitle("Fig 2. Learning Rate vs (Linear) Benchmark Performance")
        return fig

    _()
    return


@app.cell
def _(mo):
    mo.md("""
    ## Training Loss vs Benchmark Performance

    This plot directly shows the relationship between pretraining loss and downstream benchmark performance. If lower training loss led to better representations, we'd expect a negative correlation. Instead, we often see the opposite - models with higher training loss (less overfitting) transfer better.
    """)
    return


@app.cell
def _(joined_df, mo, pl, plt):
    def _():
        # Plot 3: Training Loss vs Benchmark cmAP (overfitting analysis)
        by_run = (
            joined_df
            .filter(pl.col("run_id").is_not_null() & pl.col("final_loss").is_not_null())
            .group_by("run_id", "n_train")
            .agg([
                pl.col("cmap").mean().alias("mean_cmap"),
                pl.col("final_loss").first().alias("final_loss"),
                pl.col("lr").first().alias("lr"),
            ])
        )

        if len(by_run) == 0:
            return mo.md("No data with both training loss and benchmark results")

        fig, axes = plt.subplots(
            nrows=2,
            ncols=2,
            sharey=True,
            sharex=True,
            figsize=(7, 5),
            dpi=150,
            layout="constrained",
        )
        axes = axes.reshape(-1)

        for n_train, ax in zip([1, 10, 100, -1], axes):
            subset = by_run.filter(pl.col("n_train") == n_train)
            if len(subset) == 0:
                continue
            losses, cmaps, lrs = (
                subset.select("final_loss", "mean_cmap", "lr").to_numpy().T
            )
            sc = ax.scatter(
                losses,
                cmaps,
                c=lrs,
                alpha=0.5,
                cmap="viridis",
                norm=plt.matplotlib.colors.LogNorm(),
            )
            ax.set_xlabel("Final Training Loss")
            ax.set_ylabel("Mean cmAP")
            ax.grid(True, alpha=0.3)
            ax.spines[["right", "top"]].set_visible(False)
            title = f"n={n_train}" if n_train > 0 else "All"
            ax.set_title(title)

        fig.colorbar(sc, ax=axes, label="Learning Rate")
        fig.suptitle("Fig 3. Training Loss vs Benchmark Performance (color=LR)")
        return fig

    _()
    return


@app.cell
def _(mo):
    mo.md("""
    ## Checkpoint Resume Debug: Seed Variance Test

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
        api = wandb.Api()
        run = api.run(f"samuelstevens/birdjepa/{run_id}")
        history = run.scan_history(keys=["step", "train/probe", "_timestamp"])
        rows = []
        for row in history:
            step = row.get("step")
            loss = row.get("train/probe")
            if step is not None and loss is not None:
                rows.append({
                    "run_id": run_id,
                    "seed": run.config.get("seed", "?"),
                    "step": int(step),
                    "loss": float(loss),
                })
        return pl.DataFrame(rows)

    seed_history_df = pl.concat([_fetch_history(rid) for rid in seed_test_runs])
    seed_history_df
    return seed_history_df, seed_test_runs


@app.cell
def _(pl, plt, seed_history_df):
    # Plot: Loss vs Step for seed variance test runs
    _fig, _ax = plt.subplots(figsize=(10, 5), dpi=150)

    _colors = {"1": "#1f77b4", "2": "#ff7f0e", "3": "#2ca02c"}

    for _seed in ["1", "2", "3"]:
        _subset = seed_history_df.filter(pl.col("seed") == int(_seed)).sort("step")
        if len(_subset) == 0:
            continue
        _steps = _subset["step"].to_list()
        _losses = _subset["loss"].to_list()
        _ax.plot(
            _steps,
            _losses,
            "-",
            color=_colors[_seed],
            label=f"seed={_seed}",
            alpha=0.8,
            linewidth=1,
        )

    _ax.set_xlabel("Step")
    _ax.set_ylabel("Training Loss (train/probe)")
    _ax.set_title("Seed Variance Test: Loss vs Step (lr=0.03, Muon + free wins)")
    _ax.legend()
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
        losses = subset["loss"].to_list()
        anomalies = []
        for i in range(1, len(steps)):
            step_gap = steps[i] - steps[i - 1]
            loss_change = losses[i] - losses[i - 1]
            if step_gap > 100 or abs(loss_change) > 0.5:
                anomalies.append({
                    "seed": seed,
                    "step": steps[i],
                    "prev_step": steps[i - 1],
                    "step_gap": step_gap,
                    "loss": losses[i],
                    "prev_loss": losses[i - 1],
                    "loss_change": loss_change,
                })
        return anomalies

    _all_anomalies = []
    for _s in [1, 2, 3]:
        _all_anomalies.extend(_detect_anomalies(seed_history_df, _s))

    if _all_anomalies:
        _anomaly_df = pl.DataFrame(_all_anomalies)
        _output = mo.vstack([
            mo.md("### Detected Anomalies (step gaps > 100 or loss change > 0.5)"),
            _anomaly_df,
        ])
    else:
        _output = mo.md("No anomalies detected")
    _output
    return


@app.cell
def _(mo):
    mo.md("""
    ## Loss vs Target Entropy at Resume Points

    Hypothesis: The loss direction (up or down) after resume depends on which files end up first in the shuffled order. Files with different class distributions lead to different batch difficulty.

    If true, `tgt_entropy_frac` should correlate with the loss change direction at resume points.
    """)
    return


@app.cell
def _(pl, seed_test_runs, wandb):
    # Fetch tgt_entropy_frac alongside loss for seed test runs
    def _fetch_entropy_history(run_id: str):
        api = wandb.Api()
        run = api.run(f"samuelstevens/birdjepa/{run_id}")
        history = run.scan_history(
            keys=["step", "train/probe", "train/tgt_entropy_frac", "_timestamp"]
        )
        rows = []
        for row in history:
            step = row.get("step")
            loss = row.get("train/probe")
            entropy = row.get("train/tgt_entropy_frac")
            ts = row.get("_timestamp")
            if step is not None and loss is not None:
                rows.append({
                    "run_id": run_id,
                    "seed": run.config.get("seed", "?"),
                    "step": int(step),
                    "loss": float(loss),
                    "entropy_frac": float(entropy) if entropy else None,
                    "timestamp": ts,
                })
        return pl.DataFrame(rows)

    entropy_df = pl.concat([_fetch_entropy_history(rid) for rid in seed_test_runs])
    entropy_df
    return (entropy_df,)


@app.cell
def _(entropy_df, pl, plt):
    # Plot loss and entropy side by side for one run to check correlation
    _fig, _axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 8), dpi=150, sharex=True)

    _colors = {"1": "#1f77b4", "2": "#ff7f0e", "3": "#2ca02c"}

    for _i, _seed in enumerate([1, 2, 3]):
        _ax = _axes[_i]
        _subset = entropy_df.filter(pl.col("seed") == _seed).sort("step")
        if len(_subset) == 0:
            continue

        _steps = _subset["step"].to_list()
        _losses = _subset["loss"].to_list()
        _entropies = _subset["entropy_frac"].to_list()

        _ax.plot(
            _steps, _losses, "-", color=_colors[str(_seed)], label="loss", alpha=0.8
        )
        _ax2 = _ax.twinx()
        _ax2.plot(
            _steps, _entropies, "--", color="gray", label="entropy_frac", alpha=0.6
        )
        _ax.set_ylabel(f"Loss (seed={_seed})", color=_colors[str(_seed)])
        _ax2.set_ylabel("Entropy Frac", color="gray")
        _ax.grid(True, alpha=0.3)

    _axes[-1].set_xlabel("Step")
    _fig.suptitle("Loss vs Target Entropy Fraction (dashed=entropy)")
    _fig
    return


@app.cell
def _(entropy_df, mo, pl):
    # Find resume points and check entropy before/after
    def _find_resume_entropy(df, seed):
        subset = df.filter(pl.col("seed") == seed).sort("timestamp")
        if len(subset) < 2:
            return []
        steps = subset["step"].to_list()
        losses = subset["loss"].to_list()
        entropies = subset["entropy_frac"].to_list()
        timestamps = subset["timestamp"].to_list()

        resumes = []
        for i in range(1, len(steps)):
            # Detect resume: step went backwards or stayed same but timestamp jumped
            if steps[i] <= steps[i - 1] and timestamps[i] - timestamps[i - 1] > 600:
                resumes.append({
                    "seed": seed,
                    "step_before": steps[i - 1],
                    "step_after": steps[i],
                    "loss_before": losses[i - 1],
                    "loss_after": losses[i],
                    "loss_change": losses[i] - losses[i - 1],
                    "entropy_before": entropies[i - 1],
                    "entropy_after": entropies[i],
                    "entropy_change": (entropies[i] - entropies[i - 1])
                    if entropies[i] and entropies[i - 1]
                    else None,
                })
        return resumes

    _all_resumes = []
    for _s in [1, 2, 3]:
        _all_resumes.extend(_find_resume_entropy(entropy_df, _s))

    if _all_resumes:
        _resume_df = pl.DataFrame(_all_resumes)
        _output = mo.vstack([
            mo.md("### Resume Points: Loss vs Entropy Changes"),
            mo.md(
                "If entropy_change correlates with loss_change direction, our hypothesis is supported."
            ),
            _resume_df,
        ])
    else:
        _output = mo.md("No resume points detected")
    _output
    return


@app.cell
def _(entropy_df, np, pl, plt):
    # Compute EMA of loss to see if there's a systematic shift at resume
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
        _losses = _subset["loss"].to_numpy()
        _timestamps = _subset["timestamp"].to_numpy()

        # Compute EMA
        _ema_loss = _ema(_losses, alpha=0.05)

        # Plot raw and EMA
        _ax.plot(
            _steps,
            _losses,
            "-",
            color=_colors[_seed],
            alpha=0.3,
            linewidth=0.5,
            label="raw",
        )
        _ax.plot(
            _steps, _ema_loss, "-", color=_colors[_seed], linewidth=2, label="EMA(0.05)"
        )

        # Mark resume points (where step goes backwards)
        for _j in range(1, len(_steps)):
            if (
                _steps[_j] <= _steps[_j - 1]
                and _timestamps[_j] - _timestamps[_j - 1] > 600
            ):
                _ax.axvline(
                    _steps[_j],
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    label="resume" if _j == 1 else None,
                )
                # Annotate the EMA values before/after
                _ax.annotate(
                    f"EMA: {_ema_loss[_j - 1]:.2f} → {_ema_loss[_j]:.2f}",
                    (_steps[_j], _ema_loss[_j]),
                    textcoords="offset points",
                    xytext=(10, 10),
                    fontsize=8,
                    color="red",
                )

        _ax.set_ylabel(f"Loss (seed={_seed})")
        _ax.legend(loc="upper right")
        _ax.grid(True, alpha=0.3)

    _axes[-1].set_xlabel("Step")
    _fig.suptitle("Loss with EMA - Red lines = resume points")
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## Dataloader Reset Experiment (EMA Loss)

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
        api = wandb.Api()
        run = api.run(f"samuelstevens/birdjepa/{run_id}")
        history = run.scan_history(keys=["step", "train/probe", "_timestamp"])
        rows = []
        for row in history:
            step = row.get("step")
            loss = row.get("train/probe")
            ts = row.get("_timestamp")
            if step is not None and loss is not None:
                rows.append({
                    "run_id": run_id,
                    "label": run_labels[run_id],
                    "step": int(step),
                    "loss": float(loss),
                    "timestamp": ts,
                })
        return pl.DataFrame(rows)

    reset_control_df = pl.concat([
        _fetch_loss_history(rid) for rid in [reset_run_id, control_run_id]
    ])
    reset_control_df
    return control_run_id, reset_control_df, reset_run_id, run_labels


@app.cell
def _(np, pl, plt, reset_control_df, run_labels):
    def _ema_loss(values, alpha=0.05):
        ema = np.zeros_like(values)
        ema[0] = values[0]
        for i in range(1, len(values)):
            ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]
        return ema

    _fig, _ax = plt.subplots(figsize=(12, 4), dpi=150)
    _colors = {
        "reset_dataloader@1000": "#d62728",
        "control": "#1f77b4",
    }

    for _run_id, _label in run_labels.items():
        _subset = reset_control_df.filter(pl.col("run_id") == _run_id).sort("step")
        if len(_subset) == 0:
            continue

        _steps = _subset["step"].to_numpy()
        _losses = _subset["loss"].to_numpy()
        _ema = _ema_loss(_losses, alpha=0.05)

        _ax.plot(
            _steps,
            _ema,
            "-",
            color=_colors[_label],
            linewidth=2,
            label=f"{_label} EMA(0.05)",
        )
        _ax.plot(
            _steps,
            _losses,
            "-",
            color=_colors[_label],
            alpha=0.15,
            linewidth=0.5,
        )

    _ax.axvline(1000, color="black", linestyle="--", alpha=0.5, label="step=1000")
    _ax.set_xlabel("Step")
    _ax.set_ylabel("Loss")
    _ax.set_title("EMA of Loss: Dataloader Reset vs Control")
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
        _losses = _subset["loss"].to_numpy()
        _ema = np.zeros_like(_losses)
        _ema[0] = _losses[0]
        for _i in range(1, len(_losses)):
            _ema[_i] = 0.05 * _losses[_i] + 0.95 * _ema[_i - 1]

        _idx_after = np.searchsorted(_steps, 1000, side="left")
        if _idx_after == 0 or _idx_after >= len(_steps):
            continue
        _idx_before = _idx_after - 1

        _rows.append({
            "run_id": _run_id,
            "label": _label,
            "step_before": int(_steps[_idx_before]),
            "step_after": int(_steps[_idx_after]),
            "loss_before": float(_losses[_idx_before]),
            "loss_after": float(_losses[_idx_after]),
            "loss_change": float(_losses[_idx_after] - _losses[_idx_before]),
            "ema_before": float(_ema[_idx_before]),
            "ema_after": float(_ema[_idx_after]),
            "ema_change": float(_ema[_idx_after] - _ema[_idx_before]),
        })

    pl.DataFrame(_rows)
    return


if __name__ == "__main__":
    app.run()
