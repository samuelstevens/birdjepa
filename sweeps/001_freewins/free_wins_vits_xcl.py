"""Sweep: ViT-S on XCL with all 'free wins' enabled (RoPE, QK-Norm, SwiGLU, LayerScale).

Compares AdamW vs Muon optimizer with different learning rates.
"""


def make_cfgs() -> list[dict]:
    cfgs = []

    train_data = {"__class__": "XenoCanto", "subset": "XCL"}
    test_data = {"__class__": "XenoCanto", "subset": "XCL", "n_samples": 10_000}

    # ViT-S with all free wins enabled
    model = {
        "input_h": 512,
        "input_w": 128,
        "patch_h": 16,
        "patch_w": 16,
        "embed_dim": 384,
        "depth": 12,
        "n_heads": 6,
        # Free wins
        "use_rope": True,
        "rope_base": 100.0,
        "use_qk_norm": True,
        "use_swiglu": True,
        "use_layerscale": True,
        "layerscale_init": 1e-4,
    }

    # Test both optimizers at their typical LRs
    # AdamW: 1e-3 was optimal from prior sweep
    # Muon: typically needs different LR, try a few values
    configs = [
        ("adamw", 1e-3),
        ("muon", 1e-3),
        ("muon", 3e-3),
        ("muon", 1e-2),
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
            "slurm_acct": "PAS2136",
            "slurm_partition": "preemptible-nextgen",
            "ckpt_to": "/fs/ess/PAS2136/samuelstevens/birdjepa/checkpoints",
        })

    return cfgs
