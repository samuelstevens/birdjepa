# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "datasets>=4.0",
#     "soundfile",
#     "tyro",
# ]
# ///
"""
Resources:

https://huggingface.co/datasets/DBD-research-group/BirdSet

Run this with `datasets<4.0` to download. Then switch to >4.0 to use the cached version and push it to the hub.
"""

import datasets
import tyro


def main(subset: str):
    ds = datasets.load_dataset(
        "DBD-research-group/BirdSet", subset, trust_remote_code=True
    )
    ds.push_to_hub(
        "samuelstevens/BirdSet", subset, set_default=False, private=False, num_proc=1
    )


if __name__ == "__main__":
    tyro.cli(main)
