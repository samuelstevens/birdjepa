"""Launcher script for training and benchmarking with submitit support."""

import typing as tp

import tyro

from birdjepa import benchmark, pretrain


def launch_train(cfg: tp.Annotated[pretrain.Config, tyro.conf.arg(name="")]):
    """Train a model, optionally from a sweep file.

    Args:
        cfg: Config with optional sweep field for sweep file path.
    """
    pretrain.cli(cfg)


def launch_benchmark(cfg: tp.Annotated[benchmark.Config, tyro.conf.arg(name="")]):
    benchmark.cli(cfg)


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({
        "train": launch_train,
        "benchmark": launch_benchmark,
    })
