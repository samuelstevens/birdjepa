"""Debug sweeps to isolate JAX training issues."""


def make_cfgs() -> list[dict]:
    # Test 1: Transformer without augmentations
    # If this learns, bug is in augmentations
    # If this doesn't learn, bug is in Transformer
    transformer_no_aug = {
        "train_data": {"__class__": "Cifar100", "augmentations": []},
        "test_data": {"__class__": "Cifar100", "split": "test", "augmentations": []},
        "model": {
            "__class__": "TransformerConfig",
            "input_h": 32,
            "input_w": 32,
            "patch_h": 4,
            "patch_w": 4,
            "embed_dim": 64,
            "depth": 4,
            "n_heads": 4,
        },
        "objective": {"__class__": "SupervisedConfig"},
        "batch_size": 128,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "grad_clip": 0.0,
        "schedule": "cosine",
        "warmup_steps": 0,
        "epochs": 20,
        "n_workers": 2,
        "log_every": 50,
    }

    return [transformer_no_aug]
