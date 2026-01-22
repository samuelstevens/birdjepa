"""Compare param_norm from checkpoints against log file values."""

import argparse
import logging
import os
import pathlib
import re
import typing as tp

import beartype
import yaml


logger = logging.getLogger("compare_ckpt_param_norms")

_RE_LOG_WANDB_DPATH = re.compile(r"Run data is saved locally in (.+)$")
_RE_LOG_CKPT_DPATH = re.compile(r"Checkpoint dir: (.+)$")
_RE_LOG_N_CLASSES = re.compile(r"Dataloaders created: .*?, .*?, (\d+) classes")
_RE_LOG_OBJECTIVE = re.compile(r"Created objective: (\S+)")
_RE_LOG_MODEL = re.compile(r"Created (TransformerModel)|Using DebugEncoder")
_RE_LOG_PARAM_NORM = re.compile(r"step=(\d+).*param_norm=([0-9.eE+-]+)")


@beartype.beartype
def get_log_lines(log_fpath: pathlib.Path) -> list[str]:
    assert log_fpath.exists(), f"Log file not found: {log_fpath}"
    return log_fpath.read_text().splitlines()


@beartype.beartype
def get_wandb_run_dpath(
    run_id: str, *, log_lines: list[str], wandb_root_dpath: pathlib.Path
) -> pathlib.Path:
    wandb_run_dpath = None
    for line in log_lines:
        match = _RE_LOG_WANDB_DPATH.search(line)
        if match is None:
            continue
        wandb_run_dpath = pathlib.Path(match.group(1))

    if wandb_run_dpath is not None:
        assert run_id in wandb_run_dpath.name, (
            f"Run id {run_id!r} not found in wandb path {wandb_run_dpath}"
        )
        return wandb_run_dpath

    assert wandb_root_dpath.exists(), f"W&B root not found: {wandb_root_dpath}"
    candidates = sorted(
        dpath
        for dpath in wandb_root_dpath.iterdir()
        if dpath.is_dir() and dpath.name.endswith(f"-{run_id}")
    )
    assert candidates, f"No W&B run directories found for run_id={run_id}"
    return candidates[-1]


@beartype.beartype
def get_ckpt_dpath(run_id: str, *, log_lines: list[str], cfg_dct: dict) -> pathlib.Path:
    for line in log_lines:
        match = _RE_LOG_CKPT_DPATH.search(line)
        if match is None:
            continue
        ckpt_dpath = pathlib.Path(match.group(1))
        assert ckpt_dpath.name == run_id, (
            f"Checkpoint dir {ckpt_dpath} does not match run_id {run_id}"
        )
        return ckpt_dpath

    ckpt_root = pathlib.Path(cfg_dct["ckpt_to"])
    ckpt_dpath = ckpt_root / run_id
    assert ckpt_dpath.exists(), f"Checkpoint dir not found: {ckpt_dpath}"
    return ckpt_dpath


@beartype.beartype
def get_n_classes(log_lines: list[str]) -> int:
    for line in log_lines:
        match = _RE_LOG_N_CLASSES.search(line)
        if match is None:
            continue
        return int(match.group(1))
    assert False, "Failed to parse n_classes from log file"


@beartype.beartype
def get_objective_name(log_lines: list[str]) -> str:
    for line in log_lines:
        match = _RE_LOG_OBJECTIVE.search(line)
        if match is None:
            continue
        return match.group(1)
    assert False, "Failed to parse objective name from log file"


@beartype.beartype
def get_model_name(log_lines: list[str]) -> str:
    for line in log_lines:
        match = _RE_LOG_MODEL.search(line)
        if match is None:
            continue
        if match.group(1) is not None:
            return "Transformer"
        return "Debug"
    assert False, "Failed to parse model type from log file"


@beartype.beartype
def get_param_norms_from_log(log_lines: list[str]) -> dict[int, list[float]]:
    norms_by_step: dict[int, list[float]] = {}
    for line in log_lines:
        match = _RE_LOG_PARAM_NORM.search(line)
        if match is None:
            continue
        step = int(match.group(1))
        value = float(match.group(2))
        norms_by_step.setdefault(step, []).append(value)
    return norms_by_step


@beartype.beartype
def get_wandb_config(wandb_run_dpath: pathlib.Path) -> dict:
    cfg_fpath = wandb_run_dpath / "files" / "config.yaml"
    assert cfg_fpath.exists(), f"W&B config not found: {cfg_fpath}"
    data = yaml.safe_load(cfg_fpath.read_text())
    cfg_dct = {}
    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        if "value" not in value:
            continue
        cfg_dct[key] = value["value"]
    return cfg_dct


@beartype.beartype
def make_schedule(cfg_dct: dict):
    import optax

    import birdjepa.pretrain

    schedule = cfg_dct["schedule"]
    if schedule == "wsd":
        return birdjepa.pretrain.wsd_schedule(
            peak_value=float(cfg_dct["lr"]),
            total_steps=int(cfg_dct["n_steps"]),
            warmup_steps=int(cfg_dct["warmup_steps"]),
            decay_steps=int(cfg_dct["decay_steps"]),
            end_value=0.0,
        )
    if schedule == "cosine":
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=float(cfg_dct["lr"]),
            warmup_steps=int(cfg_dct["warmup_steps"]),
            decay_steps=int(cfg_dct["n_steps"]),
            end_value=0.0,
        )
    assert False, f"Unsupported schedule: {schedule}"


@beartype.beartype
def make_model(cfg_dct: dict, *, model_name: str):
    import dataclasses
    import jax.random as jr

    import birdjepa.nn.transformer

    model_cfg = dict(cfg_dct["model"])
    if model_name == "Transformer":
        for field in dataclasses.fields(birdjepa.nn.transformer.Transformer):
            if field.name not in model_cfg:
                continue
            value = model_cfg[field.name]
            if field.type is float:
                model_cfg[field.name] = float(value)
            elif field.type is int:
                model_cfg[field.name] = int(value)
            elif field.type is bool:
                model_cfg[field.name] = bool(value)
    if model_name == "Transformer":
        cfg = birdjepa.nn.transformer.Transformer(**model_cfg)
        return birdjepa.nn.transformer.TransformerModel(cfg, key=jr.key(0)), cfg
    if model_name == "Debug":
        cfg = birdjepa.nn.transformer.Debug(**model_cfg)
        return birdjepa.nn.transformer.DebugModel(cfg, key=jr.key(0)), cfg
    assert False, f"Unsupported model type: {model_name}"


@beartype.beartype
def make_objective(
    *,
    objective_name: str,
    encoder_cfg: object,
    n_classes: int,
):
    import jax.random as jr

    import birdjepa.nn.objectives

    if objective_name == "Supervised":
        encoder_cfg = tp.cast("birdjepa.nn.transformer.Config", encoder_cfg)
        return birdjepa.nn.objectives.Supervised(
            encoder_cfg,
            n_classes=n_classes,
            key=jr.key(1),
        )
    assert False, f"Unsupported objective type: {objective_name}"


@beartype.beartype
def make_probe(*, embed_dim: int, n_classes: int):
    import equinox as eqx
    import jax.random as jr

    return eqx.nn.Sequential([
        eqx.nn.LayerNorm(embed_dim),
        eqx.nn.Linear(embed_dim, n_classes, key=jr.key(2)),
    ])


@beartype.beartype
def get_param_norm(models: dict[str, object]) -> float:
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    params = {k: eqx.filter(v, eqx.is_inexact_array) for k, v in models.items()}
    leaves = jax.tree.leaves(params)
    norm = jnp.sqrt(sum(jnp.sum(p**2) for p in leaves))
    return float(norm)


@beartype.beartype
def get_param_norms_by_component(models: dict[str, object]) -> dict[str, float]:
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    norms: dict[str, float] = {}
    for key in ("encoder", "objective", "probe"):
        params = eqx.filter(models[key], eqx.is_inexact_array)
        leaves = jax.tree.leaves(params)
        norm = jnp.sqrt(sum(jnp.sum(p**2) for p in leaves))
        norms[key] = float(norm)
    return norms


@beartype.beartype
def make_optimizer(cfg_dct: dict):
    import optax

    schedule = make_schedule(cfg_dct)
    optimizer = cfg_dct["optimizer"]
    if optimizer == "muon":
        optim = optax.contrib.muon(learning_rate=schedule)
    elif optimizer == "adamw":
        optim = optax.adamw(
            learning_rate=schedule, weight_decay=float(cfg_dct["weight_decay"])
        )
    else:
        assert False, f"Unsupported optimizer: {optimizer}"

    if float(cfg_dct["grad_clip"]) > 0:
        optim = optax.chain(optax.clip_by_global_norm(cfg_dct["grad_clip"]), optim)
    return optim


@beartype.beartype
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare param_norm from checkpoints vs log file."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--log-fpath", required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--wandb-root", default="wandb")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.device == "cpu":
        existing = os.environ.get("JAX_PLATFORM_NAME")
        if existing is not None and existing != "cpu":
            raise ValueError(
                f"JAX_PLATFORM_NAME={existing!r} conflicts with --device cpu"
            )
        os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
        os.environ.setdefault("JAX_PLATFORMS", "cpu")

    import equinox as eqx
    import jax
    import orbax.checkpoint as ocp
    import orbax.checkpoint.checkpoint_utils as ocp_utils

    run_id = args.run_id
    log_fpath = pathlib.Path(args.log_fpath)
    wandb_root_dpath = pathlib.Path(args.wandb_root)

    log_lines = get_log_lines(log_fpath)
    wandb_run_dpath = get_wandb_run_dpath(
        run_id, log_lines=log_lines, wandb_root_dpath=wandb_root_dpath
    )
    cfg_dct = get_wandb_config(wandb_run_dpath)
    ckpt_dpath = get_ckpt_dpath(run_id, log_lines=log_lines, cfg_dct=cfg_dct)

    n_classes = get_n_classes(log_lines)
    objective_name = get_objective_name(log_lines)
    model_name = get_model_name(log_lines)

    encoder, encoder_cfg = make_model(cfg_dct, model_name=model_name)
    objective = make_objective(
        objective_name=objective_name, encoder_cfg=encoder_cfg, n_classes=n_classes
    )
    probe = make_probe(embed_dim=encoder_cfg.embed_dim, n_classes=n_classes)
    optim = make_optimizer(cfg_dct)

    params = {
        "encoder": eqx.filter(encoder, eqx.is_inexact_array),
        "objective": eqx.filter(objective, eqx.is_inexact_array),
        "probe": eqx.filter(probe, eqx.is_inexact_array),
    }
    opt_state = optim.init(params)

    abstract_state = {
        "objective": objective,
        "probe": probe,
        "opt_state": opt_state,
    }

    restore_kwargs = None
    if args.device == "gpu":
        n_gpus = int(cfg_dct["n_gpus"])
        device_count = jax.device_count()
        assert device_count == n_gpus, (
            f"Checkpoint saved with {n_gpus} device(s); current JAX device_count={device_count}. "
            "Run this on the same GPU count as the training job, or pass --device cpu."
        )
    else:
        devices = jax.devices("cpu")
        assert devices, "No CPU devices available for JAX"
        cpu_sharding = jax.sharding.SingleDeviceSharding(devices[0])
        state_sharding = jax.tree.map(lambda _: cpu_sharding, abstract_state)
        encoder_sharding = jax.tree.map(lambda _: cpu_sharding, encoder)
        restore_kwargs = {
            "state": {
                "restore_args": ocp_utils.construct_restore_args(
                    abstract_state, sharding_tree=state_sharding
                )
            },
            "encoder": {
                "restore_args": ocp_utils.construct_restore_args(
                    encoder, sharding_tree=encoder_sharding
                )
            },
        }

    mngr = ocp.CheckpointManager(
        ckpt_dpath, item_names=("state", "encoder", "metadata")
    )
    steps = sorted(mngr.all_steps())
    assert steps, f"No checkpoints found in {ckpt_dpath}"

    log_param_norms = get_param_norms_from_log(log_lines)

    header = (
        "step,ckpt_param_norm,ckpt_param_norm_encoder,ckpt_param_norm_objective,"
        "ckpt_param_norm_probe,log_param_norms,delta_first"
    )
    print(header)
    for step in steps:
        restored = mngr.restore(
            step,
            args=ocp.args.Composite(
                state=ocp.args.StandardRestore(abstract_state),
                encoder=ocp.args.StandardRestore(encoder),
                metadata=ocp.args.JsonRestore(),
            ),
            restore_kwargs=restore_kwargs,
        )
        models = {
            "encoder": restored["encoder"],
            "objective": restored["state"]["objective"],
            "probe": restored["state"]["probe"],
        }
        ckpt_norm = get_param_norm(models)
        ckpt_norms = get_param_norms_by_component(models)
        log_vals = log_param_norms.get(step, [])
        delta = ""
        if log_vals:
            delta = f"{ckpt_norm - log_vals[0]:.6g}"
        log_vals_str = "|".join(f"{v:.6g}" for v in log_vals)
        print(
            f"{step},{ckpt_norm:.6g},{ckpt_norms['encoder']:.6g},"
            f"{ckpt_norms['objective']:.6g},{ckpt_norms['probe']:.6g},"
            f"{log_vals_str},{delta}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
