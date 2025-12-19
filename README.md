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
