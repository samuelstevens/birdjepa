"""Test the new sequential dataloader with XCM subset."""


def make_cfgs() -> list[dict]:
    train_data = {"__class__": "XenoCanto", "subset": "XCM"}
    test_data = {
        "__class__": "XenoCanto",
        "subset": "XCM",
        "split": "train",
        "n_samples": 1000,
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

    return [
        {
            "train_data": train_data,
            "test_data": test_data,
            "model": model,
            "objective": {"__class__": "SupervisedConfig"},
            "batch_size": 64,
            "lr": 1e-3,
            "schedule": "wsd",
            "warmup_steps": 50,
            "decay_steps": 0,
            "n_steps": 200,
            "eval_every": 100,
            "log_every": 10,
            "n_workers": 2,
            "n_gpus": 2,
            "n_hours": 1.0,
            "mem_gb": 128,
            "window_size": 2000,
            "cycle_length": 2,
            "slurm_acct": "PAS2136",
            "slurm_partition": "preemptible-nextgen",
            "ckpt_to": "/fs/ess/PAS2136/samuelstevens/birdjepa",
        }
    ]
