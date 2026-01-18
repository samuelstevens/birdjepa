"""Quick muon test to verify single-process multi-GPU training works."""


def make_cfgs() -> list[dict]:
    train_data = {"__class__": "XenoCanto", "subset": "XCM"}
    test_data = {"__class__": "XenoCanto", "subset": "XCM", "n_samples": 1000}

    # Small ViT for quick testing
    model = {
        "input_h": 512,
        "input_w": 128,
        "patch_h": 16,
        "patch_w": 16,
        "embed_dim": 384,
        "depth": 6,
        "n_heads": 6,
        "use_rope": True,
        "use_qk_norm": True,
        "use_swiglu": True,
        "use_layerscale": True,
        "layerscale_init": 1e-4,
    }

    return [
        {
            "train_data": train_data,
            "test_data": test_data,
            "model": model,
            "objective": {"__class__": "SupervisedConfig"},
            "batch_size": 256,
            "lr": 3e-3,
            "optimizer": "muon",
            "schedule": "wsd",
            "warmup_steps": 100,
            "decay_steps": 0,
            "n_steps": 500,
            "log_every": 10,
            "eval_every": 100,
            "n_workers": 32,
            "window_size": 5000,
            "n_gpus": 2,
            "n_hours": 2.0,
            "mem_gb": 64,
            "slurm_acct": "PAS2136",
            "slurm_partition": "preemptible-nextgen",
            "ckpt_to": "/fs/ess/PAS2136/samuelstevens/birdjepa/checkpoints",
        }
    ]
