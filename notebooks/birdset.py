import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import duckdb
    import beartype
    import pathlib
    import os.path
    import polars as pl
    import datasets

    return beartype, datasets, duckdb, mo, os, pathlib, pl, plt


@app.cell
def _(beartype, duckdb, os, pathlib, pl):
    @beartype.beartype
    def load_df(report_to: pathlib.Path) -> pl.DataFrame:
        """Check if experiment exists in any parquet file."""
        report_to = pathlib.Path(os.path.expandvars(report_to))
        raw_dir = report_to / "raw"
        assert raw_dir.exists()
        pattern = str(raw_dir / "*.parquet")
        query = """SELECT * FROM read_parquet(?)"""
        return duckdb.execute(query, [pattern]).pl()

    df = load_df(pathlib.Path("./results")).drop("^pred.*$")
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _():
    colors = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]
    return (colors,)


@app.cell
def _(colors, df, get_n_train, pl, plt):
    def _():
        # Does more data help on each task?
        fig, axes = plt.subplots(
            figsize=(8, 4),
            nrows=2,
            ncols=4,
            sharex=True,
            sharey=True,
            layout="constrained",
            dpi=150,
        )
        axes = axes.reshape(-1)

        tasks = df.get_column("task_name").unique().sort().to_list()

        for i, (task, ax) in enumerate(zip(tasks, axes)):
            for model, color in zip(("base", "large", "huge"), colors):
                xs, ys = (
                    df
                    .filter(
                        pl.col("model_ckpt").str.to_lowercase().str.contains(model)
                        & (pl.col("task_name") == task)
                        & (pl.col("clf") == "linear")
                    )
                    .select("n_train", "cmap")
                    .to_numpy()
                    .T
                )

                if xs.size == 0:
                    continue

                xs = sorted([get_n_train(task, x) for x in xs.tolist()])
                print(task, xs)

                ax.plot(
                    xs,
                    ys,
                    marker="o",
                    color=color,
                    alpha=0.5,
                    label=f"Bird-MAE {model[0].upper()}",
                )

            ax.set_xscale("log")
            ax.set_title(f"{task.upper()}")
            ax.spines[["top", "right"]].set_visible(False)

            if i in (0, 4):
                ax.set_ylabel("cmAP")

            if i in (4, 5, 6, 7):
                ax.set_xlabel("Training Samples")

            if i in (3,):
                ax.legend()

        return fig

    _()
    return


@app.cell
def _(datasets, mo):
    @mo.cache
    def get_n_train(task_name: str, n_train: int) -> int:
        info = datasets.load_dataset_builder(
            "samuelstevens/BirdSet", task_name.upper()
        ).info

        n_classes = len(info.features["ebird_code_multilabel"].feature.names)
        if n_train > 0:
            return int(n_classes * n_train)

        return info.splits["train"].num_examples

    get_n_train("hsn", 3)
    return (get_n_train,)


if __name__ == "__main__":
    app.run()
