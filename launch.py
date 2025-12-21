"""Launcher script for training and benchmarking with submitit support."""

import logging
import typing as tp

import tyro

from birdjepa import benchmark, pretrain

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logger = logging.getLogger("birdjepa")


def launch_train(cfg: tp.Annotated[pretrain.Config, tyro.conf.arg(name="")]):
    logging.basicConfig(level=logging.INFO, format=log_format)

    if cfg.slurm_acct:
        import submitit

        executor = submitit.SlurmExecutor(folder=cfg.log_to)
        executor.update_parameters(
            time=int(cfg.n_hours * 60),
            partition=cfg.slurm_partition,
            gpus_per_node=1,
            ntasks_per_node=1,
            cpus_per_task=cfg.n_workers,
            stderr_to_stdout=True,
            account=cfg.slurm_acct,
            setup=["module load ffmpeg/6.1.1"],
        )
        job = executor.submit(pretrain.worker_fn, cfg)
        logger.info("Submitted job %s.", job.job_id)
        job.result()
        logger.info("Job finished.")
    else:
        pretrain.worker_fn(cfg)


def launch_benchmark(cfg: tp.Annotated[benchmark.Config, tyro.conf.arg(name="")]):
    benchmark.cli(cfg)


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({
        "train": launch_train,
        "benchmark": launch_benchmark,
    })
