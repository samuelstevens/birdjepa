# JAX Migration Plan (Equinox + Optax + Grain)

Goal: make JAX the primary implementation with identical behavior to the current PT code (allowing only floating point differences). PT code should live only in `_pt.py` modules and be used solely as a reference for unit tests until we are confident enough to delete it.

Reference patterns: see `~/projects/beetle-traits` for Equinox/Optax/Grain usage and training loop structure.

## Scope

- JAX is the default implementation for models, objectives, training, and data.
- PT implementations are retained only as reference for unit tests.
- Data pipeline is real (no synthetic-only shortcuts).

## Constraints

- Parity must be identical to PT up to floating point differences.
- Use Equinox for NN modules, Optax for optimization, Grain for dataloading.
- PT shims are for testing only; runtime path should be JAX.
- Tolerances start at `atol=1e-7` and `rtol=1e-7`; adjust per-test only when required.
- Tests live alongside existing tests; PT tests may move to `_pt.py` files.
- Remove PyTorch dependency entirely; reimplement `kaldi.fbank` in JAX/NumPy.
- Checkpointing uses Equinox serialization for now; add Orbax later.

## Plan

## 1) Partition PT code into `_pt.py` reference modules

- Move PT implementations into:
  - `src/birdjepa/nn/transformer_pt.py`
  - `src/birdjepa/nn/objectives_pt.py`
  - `src/birdjepa/nn/bird_mae_pt.py`
  - `src/birdjepa/pretrain_pt.py`
- Replace current modules with JAX-first implementations:
  - `src/birdjepa/nn/transformer.py` -> JAX
  - `src/birdjepa/nn/objectives.py` -> JAX
  - `src/birdjepa/nn/bird_mae.py` -> JAX
  - `src/birdjepa/pretrain.py` -> JAX
- Add explicit PT-only imports in tests (e.g., `import birdjepa.nn.transformer_pt as transformer_pt`) so PT is never the default path.

## 2) JAX core math parity

- Port `patchify`, `unpatchify`, and `make_block_mask` to JAX with identical shapes and edge behavior.
- Confirm mask reproducibility using PRNG keys derived from fixed seeds.
- Write parity tests: PT reference vs JAX output for the same inputs.

## 3) JAX Transformer + Objective parity (Equinox)

- Port transformer blocks to Equinox with identical config fields.
- Port objectives: Supervised, LeJEPA, Pixio.
- Verify outputs for:
  - forward shapes
  - loss values (within tolerance)
  - embedding pooling behavior

## 4) JAX Bird-MAE parity

- Port Bird-MAE model and preprocessing to JAX.
- Keep HF checkpoint loading parity (weights and feature extractor behavior).
- Replicate `transform()` behavior (Mel spec) to match HF extractor within tolerance.
- Port `filter_audio()` with identical patch-to-time/freq mapping.

## 5) Real data pipeline in Grain

- Replace torch datasets and DataLoader with Grain datasets and iterators.
- Port audio preprocessing into Grain transforms (matching PT preprocessing).
- Ensure dataset outputs include `data`, `target`, `index` (per project assumption).
- Keep augmentation parity (order, randomness, parameterization).

## 6) JAX training loop (Optax + Equinox)

- Implement a JAX training loop following `beetle-traits/train.py` patterns:
  - `eqx.filter_jit`, `eqx.filter_value_and_grad`, `optax` updates
  - mixed precision if needed
  - W&B logging parity
- Keep evaluation metrics identical (cmAP, thresholds, probes) and comparable to PT.

## 7) Tests and parity strategy

- Update tests to compare JAX output to PT reference for identical inputs.
- Keep strict tolerances where possible; loosen only for known FP differences.
- Make it easy to delete PT code by avoiding JAX dependencies on PT.
- Keep PT-only tests in `_pt.py` and keep them colocated in `tests/`.

## Suggested file layout

- `src/birdjepa/nn/transformer.py` (JAX)
- `src/birdjepa/nn/transformer_pt.py` (PT reference)
- `src/birdjepa/nn/objectives.py` (JAX)
- `src/birdjepa/nn/objectives_pt.py` (PT reference)
- `src/birdjepa/nn/bird_mae.py` (JAX)
- `src/birdjepa/nn/bird_mae_pt.py` (PT reference)
- `src/birdjepa/pretrain.py` (JAX)
- `src/birdjepa/pretrain_pt.py` (PT reference)

## Future TODOs

- Evaluate Orbax checkpointing once core parity is stable.
