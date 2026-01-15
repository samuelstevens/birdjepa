import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import polars as pl
    import glob
    import json
    import numpy as np
    import re
    from pathlib import Path
    return Path, glob, json, mo, np, pl, plt, re


@app.cell
def _(mo):
    mo.md("""
    # Free Wins Analysis

    This notebook explores the trade-offs in our AdamW learning rate sweep:
    1. Learning rate vs training loss
    2. Learning rate vs benchmark performance (cmAP)
    """)
    return


@app.cell
def _(Path, json):
    # Load checkpoint metadata to map run IDs to configs
    ckpt_dir = Path("/fs/ess/PAS2136/samuelstevens/birdjepa/checkpoints")

    run_configs = {}
    for run_dir in ckpt_dir.iterdir():
        if not run_dir.is_dir():
            continue
        # Find latest checkpoint step
        steps = [int(d.name) for d in run_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        if not steps:
            continue
        latest_step = max(steps)
        metadata_dir = run_dir / str(latest_step) / "metadata"
        if metadata_dir.exists():
            # Read the JSON metadata
            json_file = list(metadata_dir.glob("*.json"))
            if json_file:
                with open(json_file[0]) as f:
                    meta = json.load(f)
                    run_configs[run_dir.name] = {
                        "step": latest_step,
                        "config": meta.get("encoder_config", {}),
                    }

    run_configs
    return


@app.cell
def _(Path, pl, re):
    # Parse training logs to extract run_id, lr, and training statistics
    log_dir = Path("logs")


    def parse_log_file(log_path):
        """Parse a training log file and extract wandb run_id and training metrics."""
        run_id = None
        records = []
        with open(log_path) as f:
            for line in f:
                # Extract wandb run ID
                if "wandb run id:" in line:
                    run_id = line.strip().split("wandb run id:")[-1].strip()
                elif "setting up run " in line:
                    run_id = line.strip().split("setting up run ")[-1].strip()

                # Extract training metrics: step, lr, loss, ce (eval loss), probe_loss
                match = re.search(r"step=(\d+)\s+lr=([\d.e+-]+).*?loss=([\d.]+)", line)
                if match:
                    record = {
                        "step": int(match.group(1)),
                        "lr": float(match.group(2)),
                        "loss": float(match.group(3)),
                    }
                    # Try to extract ce (cross-entropy/eval loss) if present
                    ce_match = re.search(r"ce=([\d.]+)", line)
                    if ce_match:
                        record["ce"] = float(ce_match.group(1))
                    records.append(record)

        df = pl.DataFrame(records) if records else None
        return run_id, df


    # Find all training log files and parse them
    # Log format: {job_id}_{array_idx}_{rank}_log.out or {job_id}_{rank}_log.out
    training_data = {}
    for log_file in log_dir.glob("*_0_log.out"):
        # Extract job_id (handles both array and non-array jobs)
        parts = log_file.name.replace("_log.out", "").split("_")
        job_id = "_".join(parts[:-1])  # Everything except the last part (rank)
        run_id, df = parse_log_file(log_file)
        if run_id is not None and df is not None and len(df) > 0:
            training_data[run_id] = {"job_id": job_id, "df": df}  # Key by run_id instead

    training_data
    return (training_data,)


@app.cell
def _(pl, training_data):
    # Create pretraining dataframe: run_id -> lr, training stats
    _rows = []
    for _run_id, _data in training_data.items():
        _df = _data["df"]
        if len(_df) < 10:
            continue
        _row = {
            "run_id": _run_id,
            "job_id": _data["job_id"],
            "lr": _df["lr"][-1],
            "n_steps": len(_df),
            "final_step": int(_df["step"][-1]),
            "final_loss": _df["loss"].tail(100).mean(),
            "min_loss": _df["loss"].min(),
        }
        if "ce" in _df.columns:
            _row["final_ce"] = _df["ce"].tail(100).mean()
            _row["min_ce"] = _df["ce"].min()
        _rows.append(_row)

    pretraining_df = pl.DataFrame(_rows)
    pretraining_df
    return (pretraining_df,)


@app.cell
def _(np, plt, pretraining_df):
    def _():
        # Histogram of training job durations (by final step)
        steps = pretraining_df.get_column("final_step").to_list()

        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        ax.hist(
            steps,
            bins=np.linspace(0, 20_000, 21),
            alpha=0.7,
            color="tab:blue",
            edgecolor="black",
        )
        ax.set_xlabel("Final Step")
        ax.set_ylabel("Number of Jobs")
        ax.set_title(f"Distribution of Training Job Durations (n={len(steps)} jobs)")
        ax.grid(True, alpha=0.3)
        ax.spines[["right", "top"]].set_visible(False)

        return fig


    _()
    return


@app.cell
def _(mo):
    min_steps_slider = mo.ui.slider(
        0, 20_000, value=10_000, step=1_000, label="Minimum steps"
    )
    min_steps_slider
    return (min_steps_slider,)


@app.cell
def _(min_steps_slider, pl, plt, pretraining_df):
    def _():
        # Plot 1: Learning Rate vs Final Loss (filtered by slider)
        _min_steps = min_steps_slider.value
        lrs, losses = filtered = (
            pretraining_df.filter(pl.col("final_step") >= _min_steps)
            .select("lr", "final_loss")
            .to_numpy()
            .T
        )

        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        ax.scatter(lrs, losses, alpha=0.3)
        ax.set_xscale("log")
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Final Loss (avg of last 100 steps)")
        ax.set_title(
            f"Learning Rate vs Training Loss (n={len(lrs)} jobs with {_min_steps:,}+ steps)"
        )
        ax.grid(True, alpha=0.3)
        ax.spines[["right", "top"]].set_visible(False)

        return fig


    _()
    return


@app.cell
def _(glob, pl):
    # Load all benchmark results (use diagonal concat to handle schema differences)
    result_files = glob.glob("results/raw/*.parquet")
    all_results = pl.concat(
        [pl.read_parquet(f) for f in result_files], how="diagonal_relaxed"
    )

    # Filter to BirdJEPA results only and extract run ID from model_ckpt
    _birdjepa_raw = all_results.filter((pl.col("model_org") == "birdjepa") & (pl.col('clf')  == 'linear')).with_columns(
        pl.col("model_ckpt").str.split("/").list.last().alias("run_id")
    )

    # Create clean benchmark dataframe: run_id, task_name -> cmap
    benchmark_df = _birdjepa_raw.select(["run_id", "task_name", "cmap", "clf", "n_train"])
    benchmark_df
    return (benchmark_df,)


@app.cell
def _(benchmark_df, pretraining_df):
    # Join pretraining and benchmark data to analyze lr -> benchmark performance
    joined_df = benchmark_df.join(pretraining_df, on="run_id", how="left")
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
            joined_df.filter(pl.col("run_id").is_not_null())
            .group_by("run_id", 'n_train')
            .agg(
                [
                    pl.col("cmap").mean().alias("mean_cmap"),
                    pl.col("cmap").std().alias("std_cmap"),
                    pl.col("cmap").count().alias("n_tasks"),
                    pl.col("lr").first().alias("lr"),
                ]
            )
            .sort("lr")
        )

        assert len(by_lr.filter(pl.col('n_tasks') > 8)) == 0
        by_lr = by_lr.filter(pl.col('n_tasks') == 8)

        if len(by_lr) == 0:
            return mo.md("No benchmark results with matching pretraining data yet")

        lrs = by_lr["lr"].to_list()
        mean_cmaps = by_lr["mean_cmap"].to_list()

        fig, axes = plt.subplots(nrows=2, ncols=2, sharey=True, sharex=True, figsize=(7, 5), dpi=150, layout='constrained')
        axes = axes.reshape(-1)

        for n_train, ax in zip([1, 10, 100, -1], axes):
            lrs, mean_cmaps = by_lr.filter(pl.col('n_train') == n_train).select('lr', 'mean_cmap').to_numpy().T
            ax.scatter(lrs, mean_cmaps, alpha=0.3)
            ax.set_xscale("log")
            ax.set_xlabel("Learning Rate")
            ax.set_ylabel("Mean cmAP")
            ax.grid(True, alpha=0.3)
            ax.spines[["right", "top"]].set_visible(False)
            title = f'n={n_train}' if n_train > 0 else 'All'
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
            joined_df.filter(pl.col("run_id").is_not_null() & pl.col("final_loss").is_not_null())
            .group_by("run_id", "n_train")
            .agg([
                pl.col("cmap").mean().alias("mean_cmap"),
                pl.col("final_loss").first().alias("final_loss"),
                pl.col("lr").first().alias("lr"),
            ])
        )

        if len(by_run) == 0:
            return mo.md("No data with both training loss and benchmark results")

        fig, axes = plt.subplots(nrows=2, ncols=2, sharey=True, sharex=True, figsize=(7, 5), dpi=150, layout='constrained')
        axes = axes.reshape(-1)

        for n_train, ax in zip([1, 10, 100, -1], axes):
            subset = by_run.filter(pl.col('n_train') == n_train)
            if len(subset) == 0:
                continue
            losses, cmaps, lrs = subset.select('final_loss', 'mean_cmap', 'lr').to_numpy().T
            sc = ax.scatter(losses, cmaps, c=lrs, alpha=0.5, cmap='viridis', norm=plt.matplotlib.colors.LogNorm())
            ax.set_xlabel("Final Training Loss")
            ax.set_ylabel("Mean cmAP")
            ax.grid(True, alpha=0.3)
            ax.spines[["right", "top"]].set_visible(False)
            title = f'n={n_train}' if n_train > 0 else 'All'
            ax.set_title(title)

        fig.colorbar(sc, ax=axes, label='Learning Rate')
        fig.suptitle("Fig 3. Training Loss vs Benchmark Performance (color=LR)")
        return fig

    _()
    return


if __name__ == "__main__":
    app.run()
