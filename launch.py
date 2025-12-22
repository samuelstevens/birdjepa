"""Launcher script for training and benchmarking with submitit support."""

import logging
import pathlib
import typing as tp

import tyro

from birdjepa import benchmark, configs, pretrain

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logger = logging.getLogger("birdjepa")


def launch_train(
    cfg: tp.Annotated[pretrain.Config, tyro.conf.arg(name="")],
    sweep: pathlib.Path | None = None,
):
    """Train a model, optionally from a sweep file.

    Args:
        cfg: Base config (CLI args override sweep values).
        sweep: Path to sweep file with make_cfgs() function.
    """
    logging.basicConfig(level=logging.INFO, format=log_format)

    # Load and merge sweep configs with CLI overrides
    if sweep is not None:
        sweep_dcts = configs.load_sweep(sweep)
        cfgs, errs = configs.load_cfgs(
            cfg, default=pretrain.Config(), sweep_dcts=sweep_dcts
        )
        for err in errs:
            logger.warning("Config error: %s", err)
        logger.info("Loaded %d configs from %s", len(cfgs), sweep)
    else:
        cfgs = [cfg]

    for c in cfgs:
        if c.slurm_acct:
            import submitit

            executor = submitit.SlurmExecutor(folder=c.log_to)
            executor.update_parameters(
                time=int(c.n_hours * 60),
                partition=c.slurm_partition,
                gpus_per_node=1,
                ntasks_per_node=1,
                cpus_per_task=c.n_workers,
                stderr_to_stdout=True,
                account=c.slurm_acct,
                setup=["module load ffmpeg/6.1.1"],
            )
            job = executor.submit(pretrain.worker_fn, c)
            logger.info("Submitted job %s.", job.job_id)
        else:
            pretrain.worker_fn(c)


def launch_benchmark(cfg: tp.Annotated[benchmark.Config, tyro.conf.arg(name="")]):
    benchmark.cli(cfg)


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({
        "train": launch_train,
        "benchmark": launch_benchmark,
    })
