"""Sweep: XenoCanto Medium (XCM) pretraining configs.

Testbed for comparing architecture decisions using ViT-S and ViT-B.
"""


def make_cfgs() -> list[dict]:
    cfgs = []

    # Shared XCM data config
    train_data = {"__class__": "XenoCanto", "subset": "XCM"}
    test_data = {
        "__class__": "XenoCanto",
        "subset": "XCM",
        "truncate": "start",
        "n_samples": 10_000,
    }

    # ViT-S: embed_dim=384, depth=12, n_heads=6
    cfgs.append({
        "train_data": train_data,
        "test_data": test_data,
        "model": {
            "input_h": 512,
            "input_w": 128,
            "patch_h": 16,
            "patch_w": 16,
            "embed_dim": 384,
            "depth": 12,
            "n_heads": 6,
        },
        "objective": {"__class__": "LeJEPAConfig", "proj_dim": 16},
        "batch_size": 256,
        "lr": 2e-3,
        "n_steps": 100_000,  # ~280 epochs on XCM (90k samples / 256 batch = 350 steps/epoch)
        "n_workers": 48,  # Rust loader threads
        "window_size": 10_000,
    })

    # ViT-B: embed_dim=768, depth=12, n_heads=12
    cfgs.append({
        "train_data": train_data,
        "test_data": test_data,
        "model": {
            "input_h": 512,
            "input_w": 128,
            "patch_h": 16,
            "patch_w": 16,
            "embed_dim": 768,
            "depth": 12,
            "n_heads": 12,
        },
        "objective": {"__class__": "LeJEPAConfig", "proj_dim": 16},
        "batch_size": 256,
        "lr": 2e-3,
        "n_steps": 100_000,
        "n_workers": 48,
        "window_size": 10_000,
    })

    return cfgs
