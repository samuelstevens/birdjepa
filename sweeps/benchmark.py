"""Sweep: evaluate Bird-MAE models on all BirdSet tasks."""

TASKS = ["pow", "per", "nes", "uhh", "hsn", "nbp", "ssw", "sne"]
MODELS = ["Bird-MAE-Base"]
CLFS = ["linear"]
N_TRAINS = [1]


def make_cfgs() -> list[dict]:
    cfgs = []
    for model in MODELS:
        for task in TASKS:
            for clf in CLFS:
                for n_train in N_TRAINS:
                    cfgs.append({
                        "task": task,
                        "model_ckpt": model,
                        "classifier": clf,
                        "n_train": n_train,
                    })
    return cfgs
