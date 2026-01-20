# 001 Baseline vs Modern Sweeps

This folder holds the "free wins" sweep set: a small-scale comparison of a typical Transformer baseline vs a modernized baseline on XCL. The intent is to show that the modern baseline is stronger under otherwise matched settings.

Baselines:
- Typical baseline: absolute positional embeddings, no QK-norm, GELU MLP, no LayerScale, AdamW.
- Modern baseline: RoPE, QK-norm, SwiGLU, LayerScale, Muon.

Sweeps:
- `adamw_vits_xcl.py`: AdamW LR sweep for the typical baseline.
- `muon_vits_xcl.py`: Muon LR sweep for the modern baseline.
