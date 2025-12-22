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
