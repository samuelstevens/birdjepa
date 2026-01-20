"""Sweep: Muon LR sweep for ViT-S on XCL with free wins enabled."""


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

    lrs = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]

    for lr in lrs:
        cfgs.append({
            "train_data": train_data,
            "test_data": test_data,
            "model": model,
            "objective": {"__class__": "SupervisedConfig"},
            "batch_size": 2048,
            "lr": lr,
            "optimizer": "muon",
            "schedule": "wsd",
            "warmup_steps": 5000,
            "decay_steps": 0,
            "n_steps": 10_000,
            "log_every": 5,
            "eval_every": 1000,
            "n_workers": 60,
            "window_size": 10_000,
            "n_gpus": 2,
            "n_hours": 4.0,
            "mem_gb": 128,
            "slurm_acct": "PAS2136",
            "slurm_partition": "preemptible-nextgen",
            "ckpt_to": "/fs/ess/PAS2136/samuelstevens/birdjepa/checkpoints",
            "tags": ["001", "v0.1"],
        })

    return cfgs
