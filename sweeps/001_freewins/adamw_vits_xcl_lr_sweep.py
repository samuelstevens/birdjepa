"""Sweep: AdamW LR sweep for ViT-S on BirdSet XCL (supervised baseline).

Note: XCL valid split has no labels, so we skip test eval during pretraining.
Evaluate on downstream benchmarks (POW, HSN, etc.) after training.
"""


def make_cfgs() -> list[dict]:
    cfgs = []

    train_data = {"__class__": "XenoCanto", "subset": "XCL"}
    # XCL valid has no labels (ebird_code is null), so no test eval during pretraining

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
            "model": model,
            "objective": {"__class__": "SupervisedConfig"},
            "batch_size": 2048,  # 1024 per GPU
            "lr": lr,
            "schedule": "wsd",
            "warmup_steps": 5000,
            "decay_steps": 0,
            "n_steps": 50_000,
            "log_every": 5,
            "n_workers": 60,
            "window_size": 10_000,  # Rust loader shuffle buffer
            "n_gpus": 2,
            "n_hours": 12.0,
            "mem_gb": 128,
            "slurm_acct": "PAS2136",
            "slurm_partition": "nextgen",
            "ckpt_to": "/fs/ess/PAS2136/samuelstevens/birdjepa",
        })

    return cfgs
