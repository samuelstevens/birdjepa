# Hardcoded Decisions

This document lists design decisions that are hardcoded in the codebase and not configurable. These are intentional choices that simplify the implementation or follow established best practices.

## Data

<details>
<summary>CIFAR-100 is converted to grayscale</summary>

RGB images are converted to single-channel grayscale using standard luminance weights (0.299R + 0.587G + 0.114B). This matches our audio spectrogram format (single-channel 2D input) and simplifies the model architecture.

Location: `data/__init__.py:317-318`
</details>

<details>
<summary>CIFAR-100 normalization uses ImageNet-derived constants</summary>

We use CIFAR_MEAN=0.4914 and CIFAR_STD=0.2470, which are the grayscale equivalents of standard CIFAR-100 normalization. These are applied after converting to [0,1] range.

Location: `data/__init__.py:184-185, 320-321`
</details>

<details>
<summary>Audio sample rate is 32kHz</summary>

All audio is resampled to 32kHz to match Bird-MAE's spectrogram configuration. Audio at different sample rates is automatically resampled.

Location: `data/__init__.py:15`
</details>

<details>
<summary>Spectrograms use 128 mel bins</summary>

Log-mel spectrograms are computed with 128 mel frequency bins, matching Bird-MAE.

Location: `data/__init__.py:17`
</details>

<details>
<summary>Spectrogram frame shift is 10ms</summary>

Spectrograms use a 10ms frame shift, resulting in 100 frames per second. A 5-second clip produces 512 time frames.

Location: `data/__init__.py:94`
</details>

<details>
<summary>Spectrogram normalization uses Bird-MAE statistics</summary>

Spectrograms are normalized with MEAN=-7.2, STD=4.43 (divided by 2), matching Bird-MAE preprocessing. This ensures compatibility with pretrained models.

Location: `data/__init__.py:18-19, 107`
</details>

<details>
<summary>Waveforms are mean-centered before spectrogram computation</summary>

Raw waveforms have their DC offset removed by subtracting the mean before computing the spectrogram.

Location: `data/__init__.py:83`
</details>

<details>
<summary>Short spectrograms are padded with minimum value</summary>

If the computed spectrogram is shorter than the target length, it's padded with the minimum value in the spectrogram (treating it as silence).

Location: `data/__init__.py:101-102`
</details>

<details>
<summary>Kaldi fbank uses Hanning window with HTK compatibility</summary>

Spectrogram computation uses torchaudio's Kaldi-compatible filterbank with htk_compat=True, Hanning window, no dithering, and no energy feature.

Location: `data/__init__.py:86-95`
</details>

## Model

<details>
<summary>Transformer uses pre-norm (LayerNorm before attention/MLP)</summary>

Each transformer block applies LayerNorm before the attention and MLP sublayers, not after. This is the modern standard that improves training stability.

Location: `nn/transformer.py:256, 261`
</details>

<details>
<summary>MLP uses GELU activation by default</summary>

The feedforward network uses GELU activation unless SwiGLU is explicitly enabled via config.

Location: `nn/transformer.py:231`
</details>

<details>
<summary>Attention uses bias in QKV and output projections</summary>

The attention mechanism includes bias terms in both the QKV projection and the output projection.

Location: `nn/transformer.py:165-166`
</details>

<details>
<summary>Patch embedding uses linear projection (not conv)</summary>

Patches are embedded using a simple linear layer rather than a convolutional layer. The input is already patchified before the projection.

Location: `nn/transformer.py:141`
</details>

<details>
<summary>Positional embeddings use truncated normal initialization with std=0.02</summary>

All positional embeddings, CLS tokens, and register tokens are initialized with truncated normal distribution (std=0.02), matching standard ViT practice.

Location: `nn/transformer.py:315-319`
</details>

<details>
<summary>CLS tokens have separate positional embeddings</summary>

Each CLS token has its own learned positional embedding, separate from the patch position embeddings.

Location: `nn/transformer.py:304-306`
</details>

<details>
<summary>Register tokens have no positional embedding</summary>

Register tokens (if used) are appended without positional embeddings, as they serve as auxiliary computation tokens.

Location: `nn/transformer.py:354-356`
</details>

<details>
<summary>LeJEPA projection head is 3-layer MLP with BatchNorm and ReLU</summary>

The LeJEPA projection head uses: Linear(D, 2048) -> BatchNorm -> ReLU -> Linear(2048, 2048) -> BatchNorm -> ReLU -> Linear(2048, proj_dim). The hidden dimension is fixed at 2048.

Location: `nn/objectives.py:230-238`
</details>

<details>
<summary>SIGReg uses 256 random projections and 17 knots</summary>

The SIGReg regularizer projects embeddings onto 256 random unit vectors and evaluates the characteristic function at 17 knots from 0 to 3.

Location: `nn/objectives.py:51-53, 66`
</details>

<details>
<summary>SourceHead uses ReLU activation</summary>

The low-rank source prediction head uses ReLU between its two linear layers: Linear(input_dim, rank) -> ReLU -> Linear(rank, n_sources).

Location: `nn/objectives.py:90-94`
</details>

<details>
<summary>Pixio decoder CLS token has no positional embedding</summary>

In the Pixio decoder, the CLS token participates in attention but doesn't receive a positional embedding (only patch tokens do).

Location: `nn/objectives.py:444-448`
</details>

<details>
<summary>Pixio mask token is initialized with normal distribution std=0.02</summary>

The learnable mask token used for masked patches is initialized from a normal distribution with std=0.02.

Location: `nn/objectives.py:489`
</details>

<details>
<summary>Pixio uses per-patch normalization for MSE loss</summary>

Before computing MSE loss, each patch is normalized to zero mean and unit variance. This prevents the model from exploiting global statistics.

Location: `nn/objectives.py:576-580`
</details>

## Training

<details>
<summary>Linear warmup lasts exactly one epoch</summary>

The learning rate warmup period is set to len(train_loader) steps, which equals exactly one epoch of training.

Location: `pretrain.py:189`
</details>

<details>
<summary>Warmup starts at 1% of target LR</summary>

Linear warmup begins at 0.01x the target learning rate and increases linearly to the full learning rate.

Location: `pretrain.py:191`
</details>

<details>
<summary>Cosine annealing decays to 1e-6</summary>

After warmup, the learning rate follows a cosine schedule that decays to a minimum of 1e-6.

Location: `pretrain.py:192`
</details>

<details>
<summary>Online probe uses LayerNorm before linear layer</summary>

The online linear probe applies LayerNorm to the embeddings before the classification linear layer.

Location: `pretrain.py:169-171`
</details>

<details>
<summary>Online probe uses fixed LR=1e-3 and weight_decay=1e-7</summary>

The probe has its own optimizer hyperparameters separate from the main model: lr=1e-3 and weight_decay=1e-7.

Location: `pretrain.py:186`
</details>

<details>
<summary>Training uses bfloat16 mixed precision</summary>

All forward passes use torch.autocast with bfloat16 dtype for memory efficiency and speed.

Location: `pretrain.py:225, 279`
</details>

<details>
<summary>Training uses AdamW optimizer</summary>

The optimizer is AdamW with per-parameter-group learning rates and weight decay.

Location: `pretrain.py:188`
</details>

<details>
<summary>Random seed is fixed at 42</summary>

torch.manual_seed(42) is called at the start of training for reproducibility.

Location: `pretrain.py:129`
</details>

<details>
<summary>DataLoader drops last incomplete batch</summary>

Training uses drop_last=True to ensure all batches have the same size, which is required for consistent masking in MAE.

Location: `pretrain.py:156`
</details>

<details>
<summary>DataLoader uses pinned memory</summary>

Both train and test dataloaders use pin_memory=True for faster GPU transfer.

Location: `pretrain.py:158, 165`
</details>
