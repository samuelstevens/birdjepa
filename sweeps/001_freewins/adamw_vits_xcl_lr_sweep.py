"""Sweep: AdamW LR sweep for ViT-S on BirdSet XCL (supervised baseline)."""


def make_cfgs() -> list[dict]:
    cfgs = []

    train_data = {"__class__": "XenoCanto", "subset": "XCL"}
    test_data = {
        "__class__": "XenoCanto",
        "subset": "XCL",
        "split": "valid",
        "truncate": "start",
    }

    model = {
        "input_h": 512,
        "input_w": 128,
        "patch_h": 16,
        "patch_w": 16,
        "embed_dim": 384,
        "depth": 12,
        "n_heads": 6,
    }

    lrs = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]

    for lr in lrs:
        cfgs.append({
            "train_data": train_data,
            "test_data": test_data,
            "model": model,
            "objective": {"__class__": "SupervisedConfig"},
            "batch_size": 2048,  # 1024 per GPU (max for A100 40GB)
            "lr": lr,
            "schedule": "wsd",
            "warmup_steps": 5000,
            "decay_steps": 0,
            "n_steps": 50_000,
            "eval_every": 1000,
            "n_workers": 4,
            "n_gpus": 2,
            "n_hours": 12.0,
            "mem_gb": 128,  # Request 128GB RAM (needs 13 CPUs on OSC)
            "slurm_acct": "PAS2136",
            "slurm_partition": "preemptible-nextgen",
            "ckpt_to": "/fs/ess/PAS2136/samuelstevens/birdjepa",
        })

    return cfgs
