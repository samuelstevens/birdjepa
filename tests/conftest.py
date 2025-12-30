"""Pytest configuration for tests."""

import os

# Force JAX to use CPU for tests (avoids GPU availability issues).
# Must be set before JAX is imported.
os.environ["JAX_PLATFORMS"] = "cpu"
# Hide GPUs from CUDA to speed up initialization.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
