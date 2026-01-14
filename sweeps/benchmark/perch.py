"""Sweep: evaluate Perch 2.0 models on all BirdSet tasks.

Perch 2.0 is Google's EfficientNet-B3 bird embedding model (1536-dim).
Uses TensorFlow CPU inference to avoid GPU conflicts with JAX.
"""

TASKS = ["pow", "per", "nes", "uhh", "hsn", "nbp", "ssw", "sne"]
# CLFS = ["linear", "mlp", "centroid"]
CLFS = ["linear"]
# N_TRAINS = [1, 5, 100, -1]
N_TRAINS = [1, 5, -1]


def make_cfgs() -> list[dict]:
    cfgs = []
    for task in TASKS:
        for clf in CLFS:
            for n_train in N_TRAINS:
                cfgs.append({
                    "task": task,
                    "model_org": "perch",
                    "model_ckpt": "perch_v2",
                    "classifier": clf,
                    "n_train": n_train,
                })
    return cfgs
