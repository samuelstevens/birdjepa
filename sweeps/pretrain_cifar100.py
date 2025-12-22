"""Sweep: pretrain configs for BirdJEPA on CIFAR-100."""


def make_cfgs() -> list[dict]:
    cfgs = []

    # Shared model config: ViT-S on CIFAR-100 with 4x4 patches (64 patches total)
    model = {
        "input_h": 32,
        "input_w": 32,
        "patch_h": 4,
        "patch_w": 4,
        "embed_dim": 384,
        "depth": 12,
        "n_heads": 6,
    }

    # LeJEPA-style augmentations for multi-view pretraining
    lejepa_augs = [
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

    # Strong augmentations for supervised learning (similar to LeJEPA but no Solarize)
    supervised_augs = [
        {"__class__": "RandomResizedCrop", "scale_min": 0.08, "scale_max": 1.0},
        {"__class__": "ColorJitter", "brightness": 0.4, "contrast": 0.4},
        {
            "__class__": "GaussianBlur",
            "kernel_size": 3,
            "sigma_min": 0.1,
            "sigma_max": 1.0,
        },
        {"__class__": "HorizontalFlip", "p": 0.5},
    ]

    # LeJEPA self-supervised pretraining
    # cfgs.append({
    #     "train_data": {"__class__": "Cifar100", "augmentations": lejepa_augs},
    #     "test_data": {"__class__": "Cifar100", "split": "test"},
    #     "model": model,
    #     "objective": {"__class__": "LeJEPAConfig", "proj_dim": 16},
    #     "batch_size": 256,
    #     "lr": 2e-3,
    #     "epochs": 800,
    #     "n_workers": 4,
    # })

    # Supervised baseline
    # cfgs.append({
    #     "train_data": {"__class__": "Cifar100", "augmentations": supervised_augs},
    #     "test_data": {"__class__": "Cifar100", "split": "test"},
    #     "model": model,
    #     "objective": {"__class__": "SupervisedConfig"},
    #     "batch_size": 256,
    #     "lr": 1e-3,
    #     "epochs": 200,
    #     "n_workers": 4,
    # })

    # Pixio (MAE) self-supervised pretraining
    # Light augmentations - reconstruction task provides learning signal
    pixio_augs = [
        {"__class__": "RandomResizedCrop", "scale_min": 0.2, "scale_max": 1.0},
        {"__class__": "HorizontalFlip", "p": 0.5},
    ]
    cfgs.append({
        "train_data": {"__class__": "Cifar100", "augmentations": pixio_augs},
        "test_data": {"__class__": "Cifar100", "split": "test"},
        "model": model,
        "objective": {
            "__class__": "PixioConfig",
            "decoder_depth": 4,
            "decoder_dim": 192,
            "decoder_heads": 4,
            "mask_ratio": 0.75,
            "block_size": 2,
        },
        "probe_pooling": "patches",
        "batch_size": 256,
        "lr": 1.5e-4,
        "epochs": 800,
        "n_workers": 4,
    })

    return cfgs
