"""Sweep: ViT-S on XCL with all 'free wins' enabled (RoPE, QK-Norm, SwiGLU, LayerScale).

Compares AdamW vs Muon optimizer with different learning rates.
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

    configs = [
        # ("adamw", 1e-3),  # Done: 90.3% test_acc (run 64dr53yj)
        # ("muon", 1e-3),   # Done: 85.1% test_acc
        # ("muon", 3e-3),   # Done: 86.8% test_acc
        # ("muon", 1e-2),   # Done: 85.9% test_acc
        ("muon", 3e-2),
        ("muon", 1e-1),
    ]

    for optimizer, lr in configs:
        cfgs.append({
            "train_data": train_data,
            "test_data": test_data,
            "model": model,
            "objective": {"__class__": "SupervisedConfig"},
            "batch_size": 2048,
            "lr": lr,
            "optimizer": optimizer,
            "schedule": "wsd",
            "warmup_steps": 5000,
            "decay_steps": 0,
            "n_steps": 10_000,
            "log_every": 5,
            "eval_every": 1000,
            "n_workers": 60,
            "window_size": 10_000,
            "n_gpus": 2,
            "n_hours": 12.0,
            "mem_gb": 128,
            "slurm_acct": "",
            "slurm_partition": "",
            "ckpt_to": "/research/nfs_su_809/workspace/stevens.994/birdjepa/checkpoints",
        })

    return cfgs
