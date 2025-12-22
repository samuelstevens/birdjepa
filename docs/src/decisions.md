# Hardcoded Decisions

This document lists design decisions that are hardcoded in the codebase and not configurable. These are intentional choices that simplify the implementation or follow established best practices.

## Data

<details>
<summary>CIFAR-100 is converted to grayscale</summary>

RGB images are converted to single-channel grayscale using standard luminance weights (0.299R + 0.587G + 0.114B). This matches our audio spectrogram format (single-channel 2D input) and simplifies the model architecture.

</details>

<details>
<summary>CIFAR-100 normalization uses ImageNet-derived constants</summary>

We use CIFAR_MEAN=0.4914 and CIFAR_STD=0.2470, which are the grayscale equivalents of standard CIFAR-100 normalization. These are applied after converting to [0,1] range.

</details>

<details>
<summary>Audio sample rate is 32kHz</summary>

All audio is resampled to 32kHz to match Bird-MAE's spectrogram configuration. Audio at different sample rates is automatically resampled.

</details>

<details>
<summary>Spectrograms use 128 mel bins</summary>

Log-mel spectrograms are computed with 128 mel frequency bins, matching Bird-MAE.

</details>

<details>
<summary>Spectrogram frame shift is 10ms</summary>

Spectrograms use a 10ms frame shift, resulting in 100 frames per second. A 5-second clip produces 512 time frames.

</details>

<details>
<summary>Spectrogram normalization uses Bird-MAE statistics</summary>

Spectrograms are normalized with MEAN=-7.2, STD=4.43 (divided by 2), matching Bird-MAE preprocessing. This ensures compatibility with pretrained models.

</details>

<details>
<summary>Waveforms are mean-centered before spectrogram computation</summary>

Raw waveforms have their DC offset removed by subtracting the mean before computing the spectrogram.

</details>

<details>
<summary>Short spectrograms are padded with minimum value</summary>

If the computed spectrogram is shorter than the target length, it's padded with the minimum value in the spectrogram (treating it as silence).

</details>

<details>
<summary>Kaldi fbank uses Hanning window with HTK compatibility</summary>

Spectrogram computation uses torchaudio's Kaldi-compatible filterbank with htk_compat=True, Hanning window, no dithering, and no energy feature.

</details>

<details>
<summary>XenoCanto is loaded from samuelstevens/BirdSet with ebird_code labels</summary>

The XenoCanto dataset is always loaded from the HuggingFace dataset `samuelstevens/BirdSet`, and labels are read from the `ebird_code` field. The dataset index is treated as the source ID.

</details>

<details>
<summary>CIFAR-100 is stored under ./data and downloads automatically</summary>

The CIFAR-100 dataset always uses root "./data" with download=True when initialized via torchvision.

</details>

<details>
<summary>CIFAR-100 class names are hardcoded</summary>

The 100 fine-grained CIFAR-100 class names are a fixed list in the codebase.

</details>

## Model

<details>
<summary>Transformer uses pre-norm (LayerNorm before attention/MLP)</summary>

Each transformer block applies LayerNorm before the attention and MLP sublayers, not after. This is the modern standard that improves training stability.

</details>

<details>
<summary>MLP uses GELU activation by default</summary>

The feedforward network uses GELU activation unless SwiGLU is explicitly enabled via config.

</details>

<details>
<summary>Attention uses bias in QKV and output projections</summary>

The attention mechanism includes bias terms in both the QKV projection and the output projection.

</details>

<details>
<summary>Patch embedding uses linear projection (not conv)</summary>

Patches are embedded using a simple linear layer rather than a convolutional layer. The input is already patchified before the projection.

</details>

<details>
<summary>Positional embeddings use truncated normal initialization with std=0.02</summary>

All positional embeddings, CLS tokens, and register tokens are initialized with truncated normal distribution (std=0.02), matching standard ViT practice.

</details>

<details>
<summary>CLS tokens have separate positional embeddings</summary>

Each CLS token has its own learned positional embedding, separate from the patch position embeddings.

</details>

<details>
<summary>Register tokens have no positional embedding</summary>

Register tokens (if used) are appended without positional embeddings, as they serve as auxiliary computation tokens.

</details>

<details>
<summary>LeJEPA projection head is 3-layer MLP with BatchNorm and ReLU</summary>

The LeJEPA projection head uses: Linear(D, 2048) -> BatchNorm -> ReLU -> Linear(2048, 2048) -> BatchNorm -> ReLU -> Linear(2048, proj_dim). The hidden dimension is fixed at 2048.

</details>

<details>
<summary>SIGReg uses 256 random projections and 17 knots</summary>

The SIGReg regularizer projects embeddings onto 256 random unit vectors and evaluates the characteristic function at 17 knots from 0 to 3.

</details>

<details>
<summary>SourceHead uses ReLU activation</summary>

The low-rank source prediction head uses ReLU between its two linear layers: Linear(input_dim, rank) -> ReLU -> Linear(rank, n_sources).

</details>

<details>
<summary>Pixio decoder CLS token has no positional embedding</summary>

In the Pixio decoder, the CLS token participates in attention but doesn't receive a positional embedding (only patch tokens do).

</details>

<details>
<summary>Pixio mask token is initialized with normal distribution std=0.02</summary>

The learnable mask token used for masked patches is initialized from a normal distribution with std=0.02.

</details>

<details>
<summary>Pixio uses per-patch normalization for MSE loss</summary>

Before computing MSE loss, each patch is normalized to zero mean and unit variance. This prevents the model from exploiting global statistics.

</details>

## Training

<details>
<summary>Linear warmup lasts exactly one epoch</summary>

The learning rate warmup period is set to len(train_loader) steps, which equals exactly one epoch of training.

</details>

<details>
<summary>Warmup starts at 1% of target LR</summary>

Linear warmup begins at 0.01x the target learning rate and increases linearly to the full learning rate.

</details>

<details>
<summary>Cosine annealing decays to 1e-6</summary>

After warmup, the learning rate follows a cosine schedule that decays to a minimum of 1e-6.

</details>

<details>
<summary>Online probe uses LayerNorm before linear layer</summary>

The online linear probe applies LayerNorm to the embeddings before the classification linear layer.

</details>

<details>
<summary>Online probe uses fixed LR=1e-3 and weight_decay=1e-7</summary>

The probe has its own optimizer hyperparameters separate from the main model: lr=1e-3 and weight_decay=1e-7.

</details>

<details>
<summary>Training uses bfloat16 mixed precision</summary>

All forward passes use torch.autocast with bfloat16 dtype for memory efficiency and speed.

</details>

<details>
<summary>Training uses AdamW optimizer</summary>

The optimizer is AdamW with per-parameter-group learning rates and weight decay.

</details>

<details>
<summary>Random seed is fixed at 42</summary>

torch.manual_seed(42) is called at the start of training for reproducibility.

</details>

<details>
<summary>DataLoader drops last incomplete batch</summary>

Training uses drop_last=True to ensure all batches have the same size, which is required for consistent masking in MAE.

</details>

<details>
<summary>DataLoader uses pinned memory</summary>

Both train and test dataloaders use pin_memory=True for faster GPU transfer.

</details>

<details>
<summary>Weights & Biases is always enabled with project "birdjepa"</summary>

Training always initializes WandB logging and hardcodes the project name to "birdjepa".

</details>

<details>
<summary>Online probe loss is always added to objective loss</summary>

The total loss is always the sum of the objective loss terms and the online probe cross-entropy. There is no config to disable or reweight the probe loss.

</details>

<details>
<summary>Evaluation uses top-1 accuracy</summary>

Per-epoch evaluation computes top-1 accuracy by argmax over probe logits.

</details>

## Objectives

<details>
<summary>Supervised and LeJEPA pool by mean of CLS tokens</summary>

Both supervised and LeJEPA objectives always use the mean of all CLS tokens (not just the first) for embeddings and classification. This is fixed even when multiple CLS tokens are configured.

</details>

<details>
<summary>LeJEPA invariance loss uses mean-over-views target</summary>

LeJEPA computes invariance loss against the mean projection over all views, not pairwise losses between views.

</details>

<details>
<summary>Pixio masking enforces exact mask count by adding patches</summary>

Pixio block masking first masks full blocks up to a floor of the target mask count, then adds individual patches to reach the exact target. It never removes patches after block masking.

</details>

## Benchmarking

<details>
<summary>BirdSet multi-label column is ebird_code_multilabel</summary>

All BirdSet benchmarking assumes the multi-label class index column is named `ebird_code_multilabel`.

</details>

<details>
<summary>Benchmarking always uses samuelstevens/BirdSet with streaming and 32kHz audio</summary>

Benchmarking loads `samuelstevens/BirdSet` with streaming, casts the audio column to 32kHz, and evaluates on the `test_5s` split.

</details>

<details>
<summary>Benchmarking requires TMPDIR for HuggingFace cache</summary>

The benchmark runner requires TMPDIR to be set and uses it for HF cache paths (HF_HUB_CACHE and HF_HOME).

</details>

<details>
<summary>AsymmetricLoss hyperparameters are fixed</summary>

The asymmetric loss uses gamma_neg=4.0, gamma_pos=1.0, clip=0.05, and eps=1e-8.

</details>

<details>
<summary>Probe training uses fixed architectures and optimizer settings</summary>

Linear probe uses LayerNorm -> Linear, MLP probe uses hidden_dim=2x and dropout=0.1, and both use AdamW with betas=(0.9, 0.95) and a cosine LR schedule.

</details>

<details>
<summary>Centroid classifier uses cosine similarity with temperature 0.1</summary>

The centroid classifier computes cosine similarity logits scaled by 10 (temperature=0.1).

</details>

<details>
<summary>Evaluation uses sigmoid threshold 0.5 and cmAP</summary>

Predictions are formed by thresholding sigmoid probabilities at 0.5, and the metric is cmAP (mean AP over classes with positives).

</details>

<details>
<summary>Benchmark Slurm params are fixed</summary>

Benchmark Slurm submission always uses gpus_per_node=1, cpus_per_task=4, and loads the ffmpeg/6.1.1 module.

</details>

## Reporting

<details>
<summary>Predictions are omitted for large test sets</summary>

Benchmark reports skip saving per-example predictions when the test set exceeds 50,000 examples.

</details>

## Bird-MAE

<details>
<summary>Only three Bird-MAE checkpoints are supported</summary>

Bird-MAE loading is limited to Bird-MAE-Base, Bird-MAE-Large, and Bird-MAE-Huge with a fixed config mapping.

</details>

<details>
<summary>Bird-MAE weights download URL and cache path are fixed</summary>

Weights are always downloaded from the DBD-research-group HuggingFace URLs and cached under BIRDJEPA_CACHE (default ~/.cache/birdjepa).

</details>

<details>
<summary>Benchmark registry only registers Bird-MAE and uses pooled features</summary>

The benchmark registry registers only "bird-mae", and BirdMAEBackbone returns the pooled features from the model.

</details>

<details>
<summary>filter_audio uses fixed STFT params and limited modes</summary>

filter_audio uses fixed STFT parameters (n_fft, hop_length, win_length) and only supports "time" and "time+freq" modes.

</details>

## Launcher

<details>
<summary>Training Slurm params are fixed in the launcher</summary>

Training Slurm submission always uses gpus_per_node=1, ntasks_per_node=1, and loads the ffmpeg/6.1.1 module.

</details>
