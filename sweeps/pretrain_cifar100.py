"""Sweep: pretrain configs for BirdJEPA on CIFAR-100."""


def make_cfgs() -> list[dict]:
    cfgs = []

    # LeJEPA-style augmentations for multi-view pretraining
    train_augs = [
        {"__class__": "RandomResizedCrop", "scale_min": 0.08, "scale_max": 1.0},
        {"__class__": "ColorJitter", "brightness": 0.8, "contrast": 0.8},
        {
            "__class__": "GaussianBlur",
            "kernel_size": 7,
            "sigma_min": 0.1,
            "sigma_max": 2.0,
        },
        {"__class__": "Solarize", "threshold": 0.5, "p": 0.2},
        {"__class__": "HorizontalFlip", "p": 0.5},
    ]

    # ViT-S on CIFAR-100 with 4x4 patches (64 patches total)
    cfgs.append({
        "train_data": {"__class__": "Cifar100", "augmentations": train_augs},
        "test_data": {"__class__": "Cifar100", "split": "test"},
        "model": {
            "input_h": 32,
            "input_w": 32,
            "patch_h": 4,
            "patch_w": 4,
            "embed_dim": 384,
            "depth": 12,
            "n_heads": 6,
        },
        "objective": {"__class__": "LeJEPAConfig", "proj_dim": 16},
        "batch_size": 256,
        "lr": 2e-3,
        "epochs": 800,
        "n_workers": 4,
    })

    return cfgs
