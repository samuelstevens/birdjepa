# BirdJEPA

Self-supervised foundation model for bird song using joint-embedding predictive architectures with fine-grained temporal resolution.

## Motivation

Standard audio models use ~120ms patches, which is too coarse for bird song where syllables contain rapid frequency modulations. We combine LeJEPA's theoretically-grounded self-supervised objective with SongMAE's 2ms temporal resolution to learn representations that capture fine-grained acoustic structure without labeled data.

## Why JEPA over MAE?

MAE reconstructs raw spectrograms, forcing the model to predict irrelevant low-level details (exact amplitude values, background noise). JEPA instead predicts in latent space, learning abstract representations without wasting capacity on reconstruction. LeJEPA adds theoretical grounding: its SIGReg objective provably produces embeddings optimal for downstream tasks by enforcing an isotropic Gaussian structure.

## References

- [LeJEPA](https://arxiv.org/abs/2511.08544): Provable and scalable JEPA training via SIGReg (sketched isotropic Gaussian regularization)
- [SongMAE](https://openreview.net/forum?id=8mluzLyvyV): Masked autoencoder for birdsong with 2ms patches enabling syllable-level analysis
- [Bird-MAE](https://arxiv.org/abs/2504.12880): Domain-specialized MAE pretrained on BirdSet for bird sound classification

## Installing

On OSC, we need ffmpeg, which can be loaded with `module load ffmpeg/6.1.1`.

## TODO

### Pretraining

- [x] Move train.py to src/birdjepa/pretrain.py
- [x] Create src/birdjepa/data/ for XCM dataloader (random 5s crops, on-the-fly spectrograms)
- [ ] Use jax.lax.scan for unpatchify to reduce JIT compilation time
- [ ] Reuse Grain dataloader across epochs instead of recreating (pool shuts down every epoch)
- [ ] Implement checkpointing (save/resume model, optimizer, scheduler, epoch)
- [ ] Auto-resubmit with submitit for long training runs (JEPA doesn't need massive batch sizes, so we can train longer with smaller batches across multiple jobs)
- [ ] Deepen understanding of LeJEPA (SIGReg, invariance loss, projection head design)
- [ ] muP
- [ ] Muon over Adam
- [ ] Read modern ViT pretraining papers for training improvements:
  - [ ] Pixio ([arxiv.org/abs/2512.15715](https://arxiv.org/abs/2512.15715)): deeper decoder (32 blocks), 4x4 block masking, multiple class tokens
  - [ ] NEPA ([sihanxu.me/nepa](https://sihanxu.me/nepa)): modern ViT training tricks
  - [ ] Perception Encoder ([arxiv.org/abs/2504.13181](https://arxiv.org/abs/2504.13181)): modern ViT training tricks
- [ ] Implement asymmetric patch sizes (16x16, 8x32, 4x64)
- [ ] Quantization (https://github.com/google/qwix)

### Objective

- [ ] Source prediction from DIET requires a huge cross-entropy vector. https://github.com/apple/ml-cross-entropy/ and https://github.com/mgmalek/efficient_cross_entropy/ and the discussion in https://github.com/pytorch/pytorch/issues/124480 would help with this problem.

### Data

- [ ] Add iNat Sounds 2024 ([github.com/visipedia/inat_sounds/tree/main/2024](https://github.com/visipedia/inat_sounds/tree/main/2024))
- [ ] Add FSD50K for non-bird sounds ([zenodo.org/record/4060432](https://zenodo.org/record/4060432))
- [ ] Full Xeno-Canto (beyond XCL/XCM subsets)

### Benchmark

- [ ] Add Perch2 to benchmark registry (requires TensorFlow, see [arxiv.org/abs/2508.04665](https://arxiv.org/abs/2508.04665))
- [ ] Add benchmark result visualization/summarization (bootstrap CIs, comparison tables, plots)
- [ ] BIRB benchmark (https://arxiv.org/pdf/2312.07439)


