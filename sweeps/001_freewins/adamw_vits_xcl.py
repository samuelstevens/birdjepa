"""Sweep: AdamW LR sweep for ViT-S on XCL with free wins enabled."""


def make_cfgs() -> list[dict]:
    cfgs = []

    train_data = {"__class__": "XenoCanto", "subset": "XCL"}
    # Use 10k train samples for validation (XCL has no labeled valid split)
    test_data = {"__class__": "XenoCanto", "subset": "XCL", "n_samples": 10_000}

    model = {
        "input_h": 512,
        "input_w": 128,
        "patch_h": 16,
        "patch_w": 16,
        "embed_dim": 384,
        "depth": 12,
        "n_heads": 6,
        "use_rope": False,
        "use_qk_norm": False,
        "use_swiglu": False,
        "use_layerscale": False,
    }

    lrs = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]

    for lr in lrs:
        cfgs.append({
            "train_data": train_data,
            "test_data": test_data,
            "model": model,
            "objective": {"__class__": "SupervisedConfig"},
            "batch_size": 2048,  # 1024 per GPU
            "lr": lr,
            "optimizer": "adamw",
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
