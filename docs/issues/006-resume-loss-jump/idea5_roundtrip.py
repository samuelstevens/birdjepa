"""Sweep: Idea 5 - Checkpoint roundtrip without dataloader restart.

Tests whether checkpoint save/load itself causes loss discontinuity by doing
in-process roundtrips while keeping the same dataloader iterator.

If roundtrip is smooth: loss jump comes from process-level resume effects
(Rust loader restart, cold shuffle buffer, etc.), not Orbax restore.

If roundtrip produces jump: focus on checkpoint restore correctness.
"""


def make_cfgs() -> list[dict]:
    cfgs = []

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
        "train_data": {**train_data, "seed": 42},
        "test_data": {**test_data, "seed": 42},
        "model": model,
        "objective": {"__class__": "SupervisedConfig"},
        "batch_size": 2048,
        "lr": 0.03,
        "optimizer": "muon",
        "schedule": "wsd",
        "warmup_steps": 500,
        "decay_steps": 0,
        "n_steps": 2000,
        "log_every": 5,
        "eval_every": 500,
        "n_workers": 60,
        "window_size": 10_000,
        "seed": 42,  # Same seed for all runs
        "n_gpus": 2,
        "n_hours": 2.0,
        "mem_gb": 128,
        "slurm_acct": "PAS2136",
        "slurm_partition": "preemptible-nextgen",
        "ckpt_to": "/fs/ess/PAS2136/samuelstevens/birdjepa/checkpoints",
    }

    # PRNG modes to test
    prng_modes = [
        ("stateless", {}),
        ("checkpointed", {}),
        ("no_stochastic", {"debug_disable_stochastic": True}),
    ]

    for prng_name, extra_cfg in prng_modes:
        prng_mode = "stateless" if prng_name == "no_stochastic" else prng_name

        # Roundtrip condition: roundtrips at [500, 1000, 1500]
        cfgs.append({
            **base,
            **extra_cfg,
            "prng_mode": prng_mode,
            "debug_roundtrip_steps": [500, 1000, 1500],
            "tags": ["006", "idea5_roundtrip", f"prng_{prng_name}", "roundtrip"],
        })

        # Control condition: no roundtrips
        cfgs.append({
            **base,
            **extra_cfg,
            "prng_mode": prng_mode,
            "debug_roundtrip_steps": [],
            "tags": ["006", "idea5_roundtrip", f"prng_{prng_name}", "control"],
        })

    return cfgs
