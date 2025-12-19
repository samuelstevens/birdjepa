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

Next steps:
- Check preprocessing against HuggingFace Bird-MAE reference
- Try prototypical probing (reported +37% over linear)
- Compare against other models (Perch, ConvNeXt)

