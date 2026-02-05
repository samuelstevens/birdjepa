"""Sweep: Idea 6 - IID sampling diagnostics around resume.

This sweep is meant to run on the preemptible partition so that the job is
naturally preempted and resumed from checkpoint. We then inspect W&B logs
around resume boundaries to see whether the Rust loader's sampling distribution
changes (cold shuffle buffer, shard skew, temporal correlation, etc.).

Notes:
- IID diagnostics are computed every step in src/birdjepa/pretrain.py.
- Metrics are logged every log_every steps (log_every=5 here, matching Idea 5).
  Lag-1 ACF metrics remain true lag-1 because the internal state is updated
  every step even when logging is sparse.
"""


def make_cfgs() -> list[dict]:
    cfgs: list[dict] = []

    train_data = {"__class__": "XenoCanto", "subset": "XCL"}
    test_data = {"__class__": "XenoCanto", "subset": "XCL", "n_samples": 10_000}

    model = {
        "input_h": 512,
        "input_w": 128,
        "patch_h": 16,
        "patch_w": 16,
        "embed_dim": 384,
        "depth": 12,
        "n_heads": 6,
        "use_rope": True,
        "rope_base": 100.0,
        "use_qk_norm": True,
        "use_swiglu": True,
        "use_layerscale": True,
        "layerscale_init": 1e-4,
    }

    base = {
        "train_data": train_data,
        "test_data": test_data,
        "model": model,
        "objective": {"__class__": "SupervisedConfig"},
        "batch_size": 2048,
        "lr": 0.03,
        "optimizer": "muon",
        "schedule": "wsd",
        "warmup_steps": 5000,
        "decay_steps": 0,
        "n_steps": 20_000,
        "log_every": 5,
        "eval_every": 1000,
        "n_workers": 60,
        "window_size": 10_000,
        "n_gpus": 2,
        "n_hours": 1.0,
        "mem_gb": 128,
        "slurm_acct": "PAS2136",
        "slurm_partition": "preemptible-nextgen",
        "ckpt_to": "/fs/ess/PAS2136/samuelstevens/birdjepa/checkpoints",
        "prng_mode": "stateless",
    }

    for seed in [1, 2, 3]:
        cfgs.append({
            **base,
            "seed": seed,
            "train_data": {**train_data, "seed": seed},
            "test_data": {**test_data, "seed": seed},
            "tags": ["006", "idea6_iid", "dataloader_diag", f"seed_{seed}"],
        })

    return cfgs
