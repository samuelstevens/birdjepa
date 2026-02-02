"""Sweep: checkpoint save/load roundtrip without dataloader restart.

Implements Idea 5 in `docs/issues/006-resume-loss-jump/FINDINGS.md`.

Goal: isolate whether Orbax checkpoint save/restore itself causes a loss jump.
We force an in-process save->load at a chosen step while continuing to consume
batches from the same Rust dataloader iterator.

This sweep includes a Muon run (matches Idea 4) and an AdamW run for comparison.
Edit as needed.
"""


def make_cfgs() -> list[dict]:
    base = {
        "train_data": {"__class__": "XenoCanto", "subset": "XCL"},
        "test_data": {"__class__": "XenoCanto", "subset": "XCL", "n_samples": 10_000},
        "model": {
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
        },
        "objective": {"__class__": "SupervisedConfig"},
        "batch_size": 2048,
        "schedule": "wsd",
        "warmup_steps": 5000,
        "decay_steps": 0,
        "n_steps": 10_000,
        "log_every": 5,
        "lr": 3e-3,
        # Avoid eval pauses around the roundtrip step; edit as needed.
        "eval_every": 10_000,
        "n_workers": 60,
        "window_size": 10_000,
        "seed": 1,
        "n_gpus": 2,
        "n_hours": 1.0,
        "mem_gb": 128,
        "slurm_acct": "PAS2136",
        "slurm_partition": "preemptible-nextgen",
        "ckpt_to": "/fs/ess/PAS2136/samuelstevens/birdjepa/checkpoints",
        "prng_mode": "checkpointed",
        "debug_roundtrip_ckpt_at_step": 1000,
        "tags": ["006", "ckpt-roundtrip"],
    }

    cfgs = []

    for optim in ["muon", "adamw"]:
        cfgs.append({**base, "optimizer": optim, "prng_mode": "stateless"})
        cfgs.append({**base, "optimizer": optim, "prng_mode": "checkpointed"})
        cfgs.append({**base, "optimizer": optim, "debug_disable_stochastic": True})

    return cfgs
