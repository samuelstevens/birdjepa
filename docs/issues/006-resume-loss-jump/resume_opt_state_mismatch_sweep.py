"""Sweep: PRNG and stochasticity tests for resume loss jumps.

Hypothesis: loss jumps are caused by resume-only stochastic differences. We test:
- stateless PRNG (fold_in step)
- checkpointed PRNG key restore
- disabling stochastic ops and data randomness
"""


def make_cfgs() -> list[dict]:
    base = {
        "train_data": {"__class__": "XenoCanto", "subset": "XCL"},
        "test_data": {
            "__class__": "XenoCanto",
            "subset": "XCL",
            "n_samples": 10_000,
        },
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
        "lr": 0.03,
        "optimizer": "muon",
        "schedule": "wsd",
        "warmup_steps": 5000,
        "decay_steps": 0,
        "n_steps": 10_000,
        "log_every": 5,
        "eval_every": 1000,
        "n_workers": 60,
        "window_size": 10_000,
        "seed": 1,
        "n_gpus": 2,
        "n_hours": 1.0,
        "mem_gb": 128,
        "slurm_acct": "PAS2136",
        "slurm_partition": "preemptible-nextgen",
        "ckpt_to": "/fs/ess/PAS2136/samuelstevens/birdjepa/checkpoints",
    }

    cfgs = []
    cfgs.append({
        **base,
        "prng_mode": "stateless",
        "tags": ["006", "resume-loss-jump", "prng-stateless"],
    })
    cfgs.append({
        **base,
        "prng_mode": "checkpointed",
        "tags": ["006", "resume-loss-jump", "prng-checkpointed"],
    })
    cfgs.append({
        **base,
        "disable_stochastic": True,
        "tags": ["006", "resume-loss-jump", "no-stochastic"],
    })
    return cfgs
