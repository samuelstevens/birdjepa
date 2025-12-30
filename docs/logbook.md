# 12/30/2025

## Bug fixed: JAX Transformer attention shape

Root cause: `jax.nn.dot_product_attention` expects shape `(B, seq_len, n_heads, head_dim)` but we were passing `(B, n_heads, seq_len, head_dim)` (PyTorch convention).

Also missing: `cls_pos_embed` - CLS tokens had no positional embedding in JAX version.

Fixes:
1. Changed QKV transpose from `(2, 0, 3, 1, 4)` to `(2, 0, 1, 3, 4)` to get correct shape
2. Added `cls_pos_embed` parameter to Transformer class
3. Created parity tests (tests/test_transformer_parity.py) for forward AND backward passes

Result: Transformer now learns on CIFAR-100!
- Epoch 0 test_acc: 7.14% (was stuck at 1%)
- Epoch 1 test_acc: 12.28%
- Epoch 2 test_acc: 16.01%
- Epoch 3 test_acc: 19.17%
- Loss decreasing: 4.37 -> 3.95 in first two epochs

## Earlier debugging (morning)

The JAX Transformer was stuck at 1% test accuracy (random for 100 classes) on CIFAR-100 supervised pretraining. Loss stuck at ~4.6 = ln(100). Gradients flowed, updates were applied, but no learning.

Isolation tests:
1. **DebugEncoder (linear projection only)**: test_acc improved 2.7% -> 3.5% over 10 epochs. Loss decreased.
2. **Transformer without augmentations**: test_acc stuck at 1% after 4 epochs.
3. **Transformer with scan vs for-loop**: identical behavior (scan is NOT the issue).

Conclusion: Bug is in the JAX Transformer architecture (transformer.py), NOT in:
- Data pipeline (CIFAR-100 loading works)
- Optimizer (AdamW works)
- Augmentations (removed, still doesn't learn)
- Training loop (DebugEncoder learns)
- jax.lax.scan (for-loop has same issue)


# 12/18/2025

Evaluation methods from Bird MAE and Song MAE papers

Bird-MAE (https://arxiv.org/pdf/2504.12880, April 2025)
- Pretrains on XCL-1.6M (1.6M bird vocalizations from Xeno-Canto)
- Evaluates on BirdSet benchmark (8 soundscape datasets)
- Datasets: POW (48 classes), PER (132), NES (89), UHH (27), HSN (21), NBP (51), SSW (81), SNE (56)
- Protocols: linear probing, prototypical probing, full fine-tuning, few-shot (1/5/10-shot)
- Metrics: cmAP (primary), AUROC, top-1 accuracy
- Input: 5-second segments, 128-dim log-mel spectrograms
- Key result: prototypical probing outperforms linear probing by up to 37% MAP

SongMAE (https://openreview.net/pdf?id=8mluzLyvyV, October 2025)
- Focus: syllable-level analysis, not species classification
- 14M param MAE-ViT, 2ms temporal resolution, 2s context
- Qualitative eval only (syllable clustering on canary, zebra finch, bengalese finch)
- Authors note lack of quantitative benchmarks

BirdSet benchmark (https://arxiv.org/pdf/2403.10380)
- Training scenarios: large (10k species), medium (409 species), dedicated (per-dataset)
- Key challenge: domain shift from focal recordings (train) to soundscapes (test)
- POW used as validation set

Evaluations we should run:
- BirdSet multi-label classification on all 8 datasets
- Linear probing, prototypical probing, fine-tuning, few-shot
- Compare against Audio-MAE, Bird-MAE, Perch, BirdNET
- Use XCL for pretraining (same as Bird-MAE)


Benchmark system plan (based on [biobench](https://github.com/samuelstevens/biobench))

Core design:
- Each BirdSet dataset is its own task module with `benchmark(cfg) -> Report`
- Registry pattern for audio backbones
- SQLite database for deduplication (skip already-run experiments)
- No distributed execution needed

Directory structure:
```
src/birdjepa/
  benchmark/
    __init__.py          # registry setup
    registry.py          # AudioBackbone interface
    reporting.py         # Report, Prediction, SQLite helpers
    config.py            # dataclasses for config
    linear_probing.py    # shared classifier
    pow.py               # Powdermill
    per.py               # Amazon Basin
    nes.py               # Colombia/Costa Rica
    uhh.py               # Hawaiian Islands
    hsn.py               # High Sierra Nevada
    nbp.py               # NIPS4BPlus
    ssw.py               # Sapsucker Woods
    sne.py               # Sierra Nevada
```

Each task module just loads data and extracts features. Classifier logic is shared.

Experiment dimensions:
- task: pow, per, nes, uhh, hsn, nbp, ssw, sne
- model: birdjepa, audio-mae, bird-mae, perch, birdnet
- classifier: linear, prototypical
- n_train: varies (e.g. 10, 50, 100, -1 for all)

SQLite deduplication keys on (task, model, classifier, n_train).

Metrics: cmAP (primary), AUROC, top-1

Entry point: `uv run python -m birdjepa.benchmark --task pow --model birdjepa --classifier linear --n_train 100`


Long-term todos:
- Investigate XCL dataset for large-scale pretraining (Bird-MAE uses XCL-1.6M)
- Plan multi-GPU training with torch.distributed (keep multi-node in mind)


# 12/19/2025

Linear probing experiments on HSN with Bird-MAE-Base

We ran systematic experiments on the HSN (High Sierra Nevada) task to understand linear probing performance. Key finding: our results (8.74% cmAP) are much lower than reported in the foundation models paper (16.37% cmAP).

Pooling strategy comparison:
- Mean pooling (default): 8.74% cmAP
- CLS token pooling: 5.68% cmAP

Mean pooling is better for Bird-MAE, which makes sense since MAE reconstruction doesn't use CLS token.

Training setup experiments (with BCE loss, no scheduler):
| Config | cmAP |
|--------|------|
| Mini-batch (bs=64, lr=1e-3, 100 epochs) | 0.0900 |
| Full-batch (lr=0.1, 100 epochs) | 0.0799 |
| Full-batch (lr=3e-2, 100 epochs) | 0.0688 |

Bird-MAE style training (asymmetric loss, cosine scheduler):
- Mini-batch (bs=64, lr=4e-4, 30 epochs): 8.74% cmAP

Key insight from Bird-MAE paper: "Standard approaches like global average pooling were inadequate for bird sounds... vocalizations typically occupy small regions of the spectrogram, where global averaging can dilute this information." They specifically designed prototypical probing because "reconstruction-focused representations don't work well with simple linear probing."

This explains the gap with reported results. The foundation models paper (arxiv.org/abs/2508.01277) reports 16.37% cmAP for Bird-MAE linear probing on HSN. Our 8.74% suggests either:
1. Different preprocessing/feature extraction
2. Different linear probe training (they use early stopping based on validation)
3. Different evaluation protocol

JEPA evaluation protocols: linear vs attentive probing

Reviewed I-JEPA, V-JEPA, and V-JEPA 2 papers to understand what probes work with JEPA architectures.

| Model | Probe Type | Notes |
|-------|-----------|-------|
| I-JEPA | Linear | Works well for images, outperforms MAE on ImageNet linear probing |
| V-JEPA | Attentive (4-layer) | +16 pts over avg pooling on SSv2 |
| V-JEPA 2 | Attentive (4-layer) | Same approach as V-JEPA |

Key findings:
- I-JEPA (images): Linear probes work well
- V-JEPA (video): Attentive probes are essential, especially for motion/temporal tasks
- V-JEPA 2 ablation: "Using adaptive pooling with a learnable cross-attention layer leads to +16.1 points on SSv2" vs average pooling
- Single-layer vs 4-layer attentive probe: only +1-1.4 points difference
- The big gain is attentive pooling vs average pooling, not probe depth

The attentive probe architecture (from V-JEPA 2):
- 4 transformer blocks with 16 attention heads
- First 3 blocks: standard self-attention
- Last block: cross-attention with learnable query token
- Output fed to linear classifier

Implications for BirdJEPA: Since bird vocalizations have temporal structure (syllables, phrases), we may face similar issues to V-JEPA where attentive probing is important. Simple mean pooling + linear probe may lose temporal information.

Sources:
- I-JEPA: https://arxiv.org/abs/2301.08243
- V-JEPA: https://openreview.net/forum?id=WFYbBOEOtv
- V-JEPA 2: https://arxiv.org/abs/2506.09985


BEANS benchmark (https://arxiv.org/abs/2210.12300)

BEANS (BEnchmark of ANimal Sounds) from Earth Species Project. Broader than BirdSet - covers birds, marine mammals, bats, dogs, mosquitoes.

Classification tasks (7):
| Dataset | Classes | Sample Rate | Duration | Species |
|---------|---------|-------------|----------|---------|
| cbi | 264 | 44100 Hz | 10s | Cornell Bird ID |
| watkins | 31 | 44100 Hz | 3s | Marine mammals (dolphins, whales, seals) |
| bats | 10 | 250000 Hz | 5s | Egyptian fruit bats |
| dogs | 10 | 44100 Hz | 10s | Individual dog ID |
| humbugdb | 14 | 44100 Hz | 3s | Mosquito species |
| esc50 | 50 | 44100 Hz | 5s | Environmental sounds |
| speech-commands | 35 | 16000 Hz | 1s | Human speech |

Detection tasks (5):
| Dataset | Classes | Sample Rate | Window |
|---------|---------|-------------|--------|
| dcase | 18 | 16000 Hz | 2s |
| enabirds | 34 | 32000 Hz | 2s |
| hiceas | 1 | 22050 Hz | 10s |
| rfcx | 24 | 48000 Hz | 10s |
| hainan-gibbons | 3 | 9600 Hz | 4s |

Metrics:
- Classification: Accuracy
- Detection: mAP

Key differences from BirdSet:
- Single-label classification (not multi-label)
- Includes detection tasks (temporal localization)
- More diverse taxa (not just birds)
- CBI (264 classes) is the most challenging - no overlapping recordists between train/test

Source: https://github.com/earthspecies/beans


# 12/20/2025

Pretraining data: BirdSet XCM

XCM is the medium-sized pretraining split from BirdSet, suitable for training BirdJEPA:
- ~90k focal recordings (directional mic at bird, not soundscapes)
- 409 species (matching all test dataset species)
- 89.3 GB, 32kHz audio in .ogg format
- Train split only (no test)

Focal recordings are cleaner than soundscapes - single bird vocalization with less background noise. This is good for pretraining since the model learns from clear examples.

Loading: `datasets.load_dataset("samuelstevens/BirdSet", "XCM")`

Data pipeline plan:
1. Random 5s crop from variable-length recordings
2. Compute log-mel spectrogram on-the-fly (no caching)
3. Use Bird-MAE spectrogram config: 32kHz, 512 time frames x 128 mel bins, 10ms frame shift
4. Return (512, 128) tensor ready for patching

Asymmetric patch sizes for fine temporal resolution

Bird-MAE uses 16x16 square patches (32 time x 8 mel = 256 patches). This gives ~160ms temporal resolution per patch, which is coarse for syllable-level structure.

SongMAE uses asymmetric patches - narrow in time, wide in frequency - to capture fine-grained temporal patterns while each patch sees the full frequency band at that time slice.

Patch size experiments to run:
| Patch (time x mel) | Time patches | Mel patches | Total | Temporal resolution |
|--------------------|--------------|-------------|-------|---------------------|
| 16 x 16 | 32 | 8 | 256 | ~160ms |
| 8 x 32 | 64 | 4 | 256 | ~80ms |
| 4 x 64 | 128 | 2 | 256 | ~40ms |

All three configurations give 256 patches total (same sequence length for transformer), but with progressively finer temporal resolution. The 4x64 patches each span half the frequency range but only 40ms of time, which should better capture rapid frequency modulations in bird syllables.

Note: 4x64 means 4 time frames x 64 mel bins per patch, so each patch is tall (covers more frequency) but narrow (covers less time).

Pixio paper notes (arxiv.org/abs/2512.15715)

Key improvements over vanilla MAE:
1. Deeper decoder: 32 blocks instead of 8. The shallow MAE decoder forced encoder's later blocks to handle reconstruction details instead of learning good representations.
2. Block masking: 4x4 patch blocks instead of single patches (75% mask ratio). Prevents reconstruction shortcuts from nearby visible patches.
3. Multiple class tokens: 8 instead of 1. Captures diverse global properties (scene type, style, camera pose). Unlike register tokens, these are used directly for downstream tasks.

For BirdJEPA, we could adapt:
- Deeper decoder for richer reconstruction signal
- Block masking in time dimension (mask contiguous time regions, not scattered patches)
- Multiple class tokens could capture different acoustic properties (species, call type, recording quality)

Other papers to read:
- NEPA (sihanxu.me/nepa)
- Perception Encoder (arxiv.org/abs/2504.13181)


# 12/21/2025

Experimental plan: systematically comparing architecture and training decisions

Using XCM with ViT-S and ViT-B as testbeds to validate architecture decisions from recent papers (Pixio, Perception Encoder, etc).

Baseline: Supervised pretraining on XCM with AdamW and standard parameterization (SP). Tune learning rate on ViT-S first.

"Free wins" (adopt without extensive ablation):
- RoPE instead of absolute positional embeddings (better length generalization)
- QK-Norm (attention stability at scale)
- SwiGLU activation (consistently better than GELU)
- Register tokens (absorb artifacts, prevent attention collapse)

Factorial comparison (need to test combinations):

| Factor | Levels |
|--------|--------|
| Objective | Supervised, LeJEPA, Pixio |
| Optimizer | AdamW, Muon |
| Parameterization | SP, muP |

3 x 2 x 2 = 12 combinations on ViT-S.

Pixio-style changes (for MAE baseline):
- Deeper decoder (8-32 blocks instead of shallow)
- Block masking (contiguous time regions, not random patches)
- Multiple CLS/register tokens (8 tokens capturing different acoustic properties)

Experimental phases:

1. Phase 1: Supervised baseline on ViT-S, tune LR
2. Phase 2: Add "free wins" (RoPE, QK-Norm, SwiGLU), verify no regression
3. Phase 3: Factorial on objectives x optimizers x parameterization (12 configs on ViT-S)
4. Phase 4: Transfer best config to ViT-B (if muP works, LR should transfer directly)

Evaluation: Online linear probe accuracy during pretraining, then downstream BirdSet/BEANS.

Note on multiple CLS tokens vs register tokens:
- Register tokens (DINOv2): Extra tokens that absorb artifacts, discarded at inference
- Multiple CLS tokens (Pixio): All 8 tokens used for representation, concatenated or pooled for downstream
- For our purposes, likely similar effect - both add learnable tokens to the sequence


# 12/22/2025

## Pixio/MAE Implementation Design

Reference: "In Pursuit of Pixel Supervision for Visual Pre-training" (https://arxiv.org/abs/2512.15715)

Pixio = MAE + 3 key changes:

1. Deeper decoder: 32 blocks (vs MAE's 8), 512 dim, 2048 hidden, 16 heads. The shallow MAE decoder forces the encoder to sacrifice representation quality for reconstruction details.
2. Block masking: 4x4 patch blocks at 75% ratio (vs single-patch masking). Prevents reconstruction shortcuts from nearby visible patches.
3. Multiple CLS tokens: 8 tokens averaged for global embedding (vs 1). Captures diverse global properties.

The current Transformer takes `(B, H, W)` images and handles patchification internally. For MAE/Pixio, we need the encoder to accept variable-length sequences (only visible patches).

New API (following DINOv3 pattern from saev/biobench):

```python
def forward(
    self, x_btk: Float[Tensor, "b t kernel"], *, grid: Int[Tensor, "b t 2"]
) -> tuple[Float[Tensor, "b d"], Float[Tensor, "b t d"]]:
```

Where:

- `x_btk`: Pre-patchified input, `(B, T, patch_h * patch_w)`. T can vary (visible patches only for encoder, all patches for decoder).
- `grid`: `(B, T, 2)` containing (row, col) coordinates for each token. Used for positional embeddings.
- Returns: `(cls_embedding, patch_embeddings)`

Why this design:

- Caller handles patchification, masking, and token selection
- Transformer just processes whatever tokens it receives
- Positional embeddings indexed by grid coordinates (works for any subset of patches)
- Same Transformer class works for both encoder (visible only) and decoder (all patches)

Masking Strategy: block masking + adjustment for fixed sequence length.

1. Divide patch grid into non-overlapping 2x2 blocks
2. Randomly mask blocks (creates contiguous masked regions = harder pretext task)
3. Adjust to hit exact target count:
   - If too few masked -> randomly mask additional individual patches
   - If too many masked -> randomly unmask some patches

This gives both:
- Harder reconstruction from contiguous masked regions
- Fixed T for efficient batching (no padding needed)

Example for 8x8 patch grid (64 patches) with 75% masking:
- 16 visible patches always (fixed T)
- Block masking ensures visible patches aren't all scattered

PixioConfig

```python
@dataclasses.dataclass(frozen=True)
class PixioConfig:
    decoder_depth: int = 8      # Number of decoder transformer blocks
    decoder_dim: int = 512      # Decoder embedding dimension
    decoder_heads: int = 8      # Decoder attention heads
    mask_ratio: float = 0.75    # Fraction of patches to mask
    block_size: int = 2         # Block masking granularity (1 = random patch)
    n_cls_tokens: int = 4       # Multiple CLS tokens (averaged for global repr)
```

Pixio Forward Pass

```
1. Patchify input
   x_bhw -> x_bnk where n = n_patches, k = patch_h * patch_w

2. Generate block mask + adjust to exact count
   mask_bn: bool tensor, True = masked
   n_visible = n * (1 - mask_ratio)

3. Select visible patches
   ```
   x_visible = x_bnk[~mask]  -> (B, n_visible, k)
   grid_visible = full_grid[~mask]  -> (B, n_visible, 2)
   ```
4. Encode visible patches only (efficiency win)
   cls_bd, enc_bvd = encoder(x_visible, grid=grid_visible)
5. Prepare decoder input: insert mask tokens at masked positions
   dec_input_bnd = empty(B, n, D)
   dec_input[~mask] = enc_bvd  # visible patch embeddings
   dec_input[mask] = mask_token  # learnable mask token
6. Add CLS tokens and run decoder
   dec_out_bnd = decoder(dec_input_bnd, grid=full_grid)
7. Project to pixels and compute loss on masked only
   pred_bnk = pixel_head(dec_out_bnd)  # (B, n, patch_h * patch_w)
   target_bnk = original patches
   loss = MSE(pred_bnk[mask], target_bnk[mask])
```

Implementation steps:

1. Modify `nn/transformer.py`:
   - Add `PatchEmbed` that takes `(B, T, kernel)` not `(B, H, W)`
   - Change `Transformer.forward` signature to accept `(x_btk, grid)`
   - Index positional embeddings by grid coordinates
   - Keep backward compatibility: add helper `patchify(x_bhw) -> x_bnk, grid`

2. Update `nn/objectives.py`:
   - Implement `Pixio.forward` with masking logic
   - Add decoder (can reuse Transformer blocks with different config)
   - Add mask token, pixel prediction head

3. Update other objectives:
   - `Supervised` and `LeJEPA` need to patchify before calling encoder
   - Or add a wrapper that handles the old API

Loss function

MSE on masked patches only (same as MAE):
```python
loss = F.mse_loss(pred_bnk[mask], target_bnk[mask])
```

Target is raw pixel values (already normalized by dataset preprocessing).

- MAE: "Masked Autoencoders Are Scalable Vision Learners" (He et al., 2021) https://arxiv.org/abs/2111.06377
- Pixio: "In Pursuit of Pixel Supervision for Visual Pre-training" https://arxiv.org/abs/2512.15715
- GitHub: https://github.com/facebookresearch/pixio

---

First Pixio run on CIFAR-100, job 3108556. Observations after ~13 epochs:

- MSE loss decreasing: 0.94 -> 0.77 (good sign)
- Probe accuracy: 2.9% (epoch 0) -> 5.8% (epoch 4) -> 5.7% (epoch 13)
- Per-patch normalization implemented (target has ~unit variance, so MSE ~0.77 means model explains ~23% of variance)

CIFAR-100 has 100 classes, so random chance is 1%. Our 5-7% is better than random but:

1. Accuracy plateauing: Improved from 2.9% to 5.8% in first 5 epochs, then stalled around 5-7%. Could indicate:
   - CLS token not learning useful representations (MAE doesn't train CLS explicitly)
   - Learning rate too low for probe
   - Need more epochs (MAE typically trains 800-1600 epochs)
2. CLS token vs mean pooling: We probe the CLS token, but MAE reconstruction doesn't use CLS. The original MAE paper uses mean pooling of all patch tokens for linear probing. This is likely a significant issue.3. Expected behavior: MAE representations are known to be weak early in training. The Bird-MAE paper notes that linear probing on MAE features underperforms until late in training. 5-7% at epoch 13/800 might be normal.4. Comparison needed: Should compare with supervised baseline at same epoch count to calibrate expectations.

### Potential Bugs / Issues to Investigate

1. Decoder input mismatch: Visible patches go through encoder but skip decoder's embedding projection. They only get decoder positional embeddings added. This matches MAE but worth verifying the dimensions align.
2. Block masking dilution: We compute `n_blocks_masked = n_masked_target // max_patches_per_block` which is conservative. For 8x8 grid with 2x2 blocks and 75% masking: target=48 masked, but 48//4=12 blocks = 48 patches exactly. But for non-divisible cases, we add individual patches which dilutes the block structure.
3. Weight decay on all params: MAE excludes bias and LayerNorm from weight decay. We apply it uniformly. Could affect optimization.
4. No gradient clipping: MAE implementations often use gradient clipping (max_norm=1.0 or similar). We don't.
5. Mask token squeeze: We use `self.mask_token.squeeze(0).squeeze(0)` which is fragile. Should use indexing `self.mask_token[0, 0]`.
6. Decoder positional embeddings: We use absolute 1D positional embeddings for decoder (flattened grid). Pixio might use 2D or learned differently.
7. Online probe during MAE training: The probe sees CLS token which MAE doesn't explicitly train for reconstruction. MAE papers often use mean pooling of patch tokens instead.
8. Learning rate: Using 1.5e-4 which is standard, but MAE often uses layer-wise LR decay which we don't implement.

### Things That Look Correct

- Per-patch normalization matches Pixio implementation
- MSE only on masked patches (not visible)
- Encoder only sees visible patches (efficiency win)
- Mask token inserted at masked positions for decoder
- Block masking with adjustment to exact count
- Reproducible masking via seeded generator

---

## Fix: CLS token in decoder (job 3108582)

The Pixio paper states: "We observe that feeding class tokens to the decoder yields slightly better performance, suggesting that allowing them to participate in reconstruction helps learn more informative global representations."

Our original implementation did NOT feed CLS to the decoder - the encoder CLS was returned directly for probing without participating in reconstruction. This meant CLS only got gradients from the (detached) probe loss, not from reconstruction.

Fix applied:
1. PixioDecoder now takes `(cls_be, x_bne)` as input
2. CLS is projected to decoder dim and concatenated with patches
3. CLS participates in all decoder attention layers
4. Reconstruction loss flows back to encoder CLS via decoder attention

Note: CLS has no positional embedding in the decoder (it's a global token, not a spatial one). Patches get positional embeddings.

The probe still uses encoder CLS (not decoder CLS), which is consistent with how Pixio evaluates - they use mean of encoder CLS tokens for kNN. Since we have 1 CLS token, this is just the encoder CLS.

---

## HuggingFace Datasets Cache Corruption on NFS

Problem: Running parallel benchmark jobs on Slurm with HuggingFace datasets causes cache corruption on NFS.

When multiple jobs start simultaneously, they race to download and prepare the same dataset files:

1. `.incomplete` file errors: Job A creates a temp file, Job B looks for it, Job A renames it, Job B crashes with `FileNotFoundError`
2. Corrupted parquet files: Partial writes result in "Parquet magic bytes not found" or "Corrupt snappy compressed data" errors that persist in cache
3. Arrow preparation races: Even after download, "Generating train/test split" writes to shared cache, causing similar corruption

What doesn't work:
- `HF_HUB_OFFLINE=1`: Only helps if cache is already clean; doesn't prevent reading corrupted files
- `download_mode` parameter: No "fail if not cached" option; `REUSE_DATASET_IF_EXISTS` still attempts downloads
- Sequential cache population then parallel reads: Works but defeats the purpose of parallel jobs; cache can still get corrupted if any job tries to re-download

Proposed solution: Use `streaming=True` with node-local TMPDIR.

```python
os.environ["HF_HUB_CACHE"] = os.environ.get("TMPDIR", "/tmp")
ds = datasets.load_dataset("samuelstevens/BirdSet", task, streaming=True)
```

Benefits:
- Data streams directly from HuggingFace, minimal local caching
- TMPDIR is node-local fast SSD, no NFS contention
- Each job is completely isolated - no race conditions possible
- Very little code change required

Code changes needed:
1. Set `HF_HUB_CACHE` to `TMPDIR` at start of worker
2. Add `streaming=True` to `load_dataset()`
3. Replace `len(dataset)` and slicing with `.iter(batch_size=...)` in `extract_features()`
4. Get `n_classes` from `datasets.load_dataset_builder(...).info` instead of `dataset.features`

Tradeoff: Streaming may be slightly slower than cached reads for repeated iterations, but for benchmarking where we iterate once, this is acceptable.

Hardcoded decisions that should be configurable

High value to make configurable:
- Spectrogram preprocessing in `src/birdjepa/data/__init__.py` and `src/birdjepa/nn/bird_mae.py`: sample rate, n_mels, frame_shift, mean/std, padding strategy, htk_compat, dither
- Dataset sources/columns/splits fixed to BirdSet: dataset name, label columns, streaming on/off, `test_5s` split, 32kHz cast
- W&B always enabled with fixed project name in `src/birdjepa/pretrain.py`
- Training seed fixed in `src/birdjepa/pretrain.py`
- Scheduler details fixed in `src/birdjepa/pretrain.py` (warmup length, start factor, eta_min)
- Online probe always on and always added to loss in `src/birdjepa/pretrain.py` (no toggle/weighting)
- Probe optimizer hyperparams fixed in `src/birdjepa/pretrain.py`
- Mixed precision fixed to bf16 in `src/birdjepa/pretrain.py`
- DataLoader `drop_last` and `pin_memory` fixed in `src/birdjepa/pretrain.py`

Benchmarking/reporting knobs:
- AsymmetricLoss hyperparams fixed in `src/birdjepa/benchmark/__init__.py`
- Centroid temperature fixed in `src/birdjepa/benchmark/__init__.py`
- Evaluation threshold fixed at 0.5 and cmAP-only reporting in `src/birdjepa/benchmark/__init__.py`
- Prediction saving cutoff fixed at 50k in `src/birdjepa/benchmark/reporting.py`

Model/weights plumbing:
- Bird-MAE checkpoint list fixed to three variants and download URL is fixed in `src/birdjepa/nn/bird_mae.py`
- Slurm resource defaults fixed in `launch.py` and `src/birdjepa/benchmark/__init__.py`

# 12/24/2025

I am going to train a couple runs to justify the "modern" training stack.

Fixed

- Dataset: BirdSet XCL, same split, same augmentations, same global batch size.
- Model: regular 12-layer ViT-S, patch = 16x16
- Budget: define a fixed number of steps that fits in 12h on 4xA100 / 6h on 4xH100, then keep that constant for every run.
- LR schedule: WSD with the same S (total steps) for every run in this comparison.

Run 3–5 learning rates (log-spaced) with:

- Optimizer: AdamW
- Architecture: baseline (no RoPE, no QK-norm, no SwiGLU, no register tokens)

Output: pick best LR by validation metric at end-of-budget (and sanity-check stability).

Run 3–5 learning rates (log-spaced, independent of Adam’s range) with:

- Optimizer: Muon
- Architecture: RoPE (2D), QK-Norm, SwiGLU, register tokens
- Everything else identical (data, steps, WSD S, batch, aug)

Output: pick best LR.

If something goes wrong, we could do

- Arch-only: AdamW + (RoPE + QK-Norm + SwiGLU + registers)
- Opt-only: Muon + baseline architecture

# 12/25/2025

Merry Christmas!
I think I finished the port to jax.

# 12/29/2025

Now I have finished the port to jax.
So we can submit a sweep on AdamW as a baseline.

Batch size memory profiling (A100 40GB):

Tested maximum batch size for supervised training on single A100 40GB with ViT-S encoder.

| Batch Size | Peak Memory | Status |
|------------|-------------|--------|
| 128        | 5.65 GB     | OK     |
| 256        | 10.07 GB    | OK     |
| 384        | 12.56 GB    | OK     |
| 512        | 17.73 GB    | OK     |
| 768        | 25.62 GB    | OK     |
| 1024       | 31.32 GB    | OK     |
| 1280       | 38.19 GB[^1]   | OOM    |
| 1536       | 45.56 GB[^1]   | OOM    |

[^1]: XLA rematerialization estimate before OOM

Findings:
- Maximum batch size is 1024 (uses ~31 GB, leaves ~9 GB headroom for dynamic allocations)
- bs=1280 needs ~38 GB after rematerialization, but OOM'd trying to allocate additional 33 GB during training
- bs=1536 exceeds 40 GB even after XLA's best rematerialization effort
- Memory scales roughly linearly: ~30 MB per sample in batch

Technical notes:
- Using `jax.nn.dot_product_attention` (flash attention) - O(seq_len) memory vs O(seq_len^2)
- `jax.lax.scan` with `jax.checkpoint` for transformer blocks
- Single-GPU mode skips `jax.distributed.initialize()` to avoid heartbeat timeouts during long JIT compilation
- Jobs at bs=768 and bs=1024 hit a separate data loading bug (`RuntimeError: Failed to open input buffer: End of file`) unrelated to GPU memory
