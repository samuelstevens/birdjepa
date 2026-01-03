"""
Bird-MAE audio preprocessing.

This module provides a torch-free implementation of the Bird-MAE transform pipeline, using numpy for FFT and array operations. Returns numpy arrays.
"""

import beartype
import numpy as np
from jaxtyping import Float, jaxtyped

# Bird-MAE audio preprocessing constants

BIRDMAE_SR_HZ = 32_000
BIRDMAE_CLIP_SEC = 5
BIRDMAE_TARGET_T = 512
BIRDMAE_N_MELS = 128

BIRDMAE_MEAN = -7.2
BIRDMAE_STD = 4.43

BIRDMAE_FRAMES_PER_PATCH = 16
BIRDMAE_MELS_PER_PATCH = 16
BIRDMAE_N_TIME_PATCHES = BIRDMAE_TARGET_T // BIRDMAE_FRAMES_PER_PATCH
BIRDMAE_N_MEL_PATCHES = BIRDMAE_N_MELS // BIRDMAE_MELS_PER_PATCH

BIRDMAE_SAMPLES_PER_FRAME = 320  # 10ms frame shift at 32kHz.
BIRDMAE_SAMPLES_PER_TIME_PATCH = BIRDMAE_FRAMES_PER_PATCH * BIRDMAE_SAMPLES_PER_FRAME

BIRDMAE_STFT_N_FFT = 1024
BIRDMAE_STFT_HOP_LENGTH = BIRDMAE_SAMPLES_PER_FRAME
BIRDMAE_STFT_WIN_LENGTH = 800  # 25ms at 32kHz.
BIRDMAE_STFT_LOW_FREQ_HZ = 20.0


def hz_to_mel(hz: float | np.ndarray) -> float | np.ndarray:
    """Convert Hz to mel scale using Kaldi/Slaney formula (natural log)."""
    return 1127.0 * np.log(1.0 + hz / 700.0)


def mel_to_hz(mel: float | np.ndarray) -> float | np.ndarray:
    """Convert mel to Hz using Kaldi/Slaney formula."""
    return 700.0 * (np.exp(mel / 1127.0) - 1.0)


def _make_mel_filterbank(
    padded_window_size: int,
    n_mels: int,
    sample_rate: int | float,
    low_freq: float = 0.0,
    high_freq: float | None = None,
) -> np.ndarray:
    """
    Create a mel filterbank matrix matching Kaldi's get_mel_banks.

    Args:
        padded_window_size: FFT size (padded window size, must be even)
        n_mels: Number of mel bins
        sample_rate: Sample rate in Hz
        low_freq: Lowest frequency in Hz
        high_freq: Highest frequency in Hz (default: Nyquist)

    Returns:
        Mel filterbank matrix of shape [n_mels, padded_window_size // 2]
    """
    if high_freq is None or high_freq <= 0:
        high_freq = sample_rate / 2.0

    num_fft_bins = padded_window_size // 2
    fft_bin_width = sample_rate / padded_window_size

    mel_low = hz_to_mel(low_freq)
    mel_high = hz_to_mel(high_freq)
    mel_freq_delta = (mel_high - mel_low) / (n_mels + 1)

    # Compute left, center, right mel frequencies for each bin
    bins = np.arange(n_mels).reshape(-1, 1)
    left_mel = mel_low + bins * mel_freq_delta
    center_mel = mel_low + (bins + 1) * mel_freq_delta
    right_mel = mel_low + (bins + 2) * mel_freq_delta

    # FFT bin frequencies in mel scale
    fft_bin_mels = hz_to_mel(fft_bin_width * np.arange(num_fft_bins)).reshape(1, -1)

    # Compute triangular filter weights
    up_slope = (fft_bin_mels - left_mel) / (center_mel - left_mel)
    down_slope = (right_mel - fft_bin_mels) / (right_mel - center_mel)

    filterbank = np.maximum(0, np.minimum(up_slope, down_slope))

    return filterbank


def _next_power_of_2(x: int) -> int:
    """Returns the smallest power of 2 that is >= x."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _get_strided(
    waveform: np.ndarray, window_size: int, window_shift: int
) -> np.ndarray:
    """Extract frames from waveform with snip_edges=True."""
    n_samples = len(waveform)
    if n_samples < window_size:
        return np.empty((0, window_size), dtype=waveform.dtype)
    n_frames = 1 + (n_samples - window_size) // window_shift
    # Use stride tricks for efficiency
    strides = (window_shift * waveform.strides[0], waveform.strides[0])
    return np.lib.stride_tricks.as_strided(
        waveform, shape=(n_frames, window_size), strides=strides
    ).copy()


def _hann_window(size: int, dtype=np.float64) -> np.ndarray:
    """Create a symmetric Hanning window matching torch.hann_window(periodic=False)."""
    return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(size) / (size - 1))).astype(dtype)


@jaxtyped(typechecker=beartype.beartype)
def kaldi_fbank(
    waveform: Float[np.ndarray, " samples"],
    *,
    sample_frequency: int | float = 16000.0,
    frame_length: int | float = 25.0,
    frame_shift: int | float = 10.0,
    num_mel_bins: int = 23,
    low_freq: int | float = 20.0,
    high_freq: int | float = 0.0,
    preemphasis_coefficient: float = 0.97,
    use_energy: bool = False,
    htk_compat: bool = False,
    dither: float = 0.0,
    remove_dc_offset: bool = True,
    raw_energy: bool = True,
) -> Float[np.ndarray, "time mels"]:
    """
    Compute log-mel filterbank features matching torchaudio.compliance.kaldi.fbank.

    This matches the torchaudio implementation with snip_edges=True, use_power=True, use_log_fbank=True.

    Args:
        waveform: 1D audio samples (numpy array)
        sample_frequency: Sample rate in Hz
        frame_length: Frame length in milliseconds
        frame_shift: Frame shift in milliseconds
        num_mel_bins: Number of mel filterbank bins
        low_freq: Low cutoff frequency in Hz
        high_freq: High cutoff frequency in Hz (0 = Nyquist)
        preemphasis_coefficient: Pre-emphasis coefficient
        use_energy: If True, add energy dimension
        htk_compat: If True, put energy last (only affects use_energy=True)
        dither: Dithering constant (0 = no dithering)
        remove_dc_offset: Subtract mean from each frame
        raw_energy: Compute energy before preemphasis and windowing

    Returns:
        Numpy array of log-mel filterbank features, shape [n_frames, num_mel_bins (+ 1 if use_energy)]
    """
    sample_rate = float(sample_frequency)
    dtype = waveform.dtype
    # torchaudio always uses float32 epsilon as floor, regardless of input dtype
    eps = np.finfo(np.float32).eps

    # Convert ms to samples
    window_shift = int(sample_rate * frame_shift * 0.001)
    window_size = int(sample_rate * frame_length * 0.001)
    padded_window_size = _next_power_of_2(window_size)

    # Extract frames (snip_edges=True) - uses numpy stride tricks
    frames_np = _get_strided(waveform, window_size, window_shift)
    if frames_np.size == 0:
        n_out = num_mel_bins + (1 if use_energy else 0)
        return np.empty((0, n_out), dtype=dtype)

    # Dithering (numpy for random generation)
    if dither != 0.0:
        frames_np = frames_np + dither * np.random.randn(*frames_np.shape).astype(dtype)

    # Remove DC offset per frame
    if remove_dc_offset:
        frames_np = frames_np - np.mean(frames_np, axis=1, keepdims=True)

    # Compute raw energy before preemphasis and windowing
    if raw_energy:
        signal_log_energy = np.log(np.maximum(np.sum(frames_np**2, axis=1), eps))

    # Apply preemphasis per frame
    if preemphasis_coefficient != 0.0:
        offset = np.pad(frames_np, ((0, 0), (1, 0)), mode="edge")[:, :-1]
        frames_np = frames_np - preemphasis_coefficient * offset

    # Apply window
    window = _hann_window(window_size, dtype=dtype)
    frames_np = frames_np * window

    # Zero-pad to power of 2
    if padded_window_size != window_size:
        frames_np = np.pad(
            frames_np, ((0, 0), (0, padded_window_size - window_size)), mode="constant"
        )

    # Compute energy after windowing if not raw_energy
    if not raw_energy:
        signal_log_energy = np.log(np.maximum(np.sum(frames_np**2, axis=1), eps))

    # FFT and power spectrum (use numpy to avoid JAX GPU init in dataloader workers)
    spectrum = np.abs(np.fft.rfft(frames_np, axis=1))
    power_spectrum = spectrum**2

    # Mel filterbank
    mel_filterbank = _make_mel_filterbank(
        padded_window_size, num_mel_bins, sample_rate, low_freq, high_freq
    ).astype(dtype)
    mel_filterbank = np.pad(mel_filterbank, ((0, 0), (0, 1)), mode="constant")

    # Apply filterbank: [n_frames, n_fft_bins] @ [n_fft_bins, num_mel_bins]
    mel_energies = np.dot(power_spectrum, mel_filterbank.T)

    # Log
    mel_energies = np.log(np.maximum(mel_energies, eps))

    # Add energy if requested
    if use_energy:
        energy = signal_log_energy.reshape(-1, 1)
        if htk_compat:
            mel_energies = np.concatenate([mel_energies, energy], axis=1)
        else:
            mel_energies = np.concatenate([energy, mel_energies], axis=1)

    return mel_energies


@jaxtyped(typechecker=beartype.beartype)
def transform(
    waveform: Float[np.ndarray, " samples"],
) -> Float[np.ndarray, "time mels"]:
    """
    Transform waveform to log-mel spectrogram for Bird-MAE.

    Args:
        waveform: 1D numpy array of audio samples

    Returns:
        Numpy array of shape [512, 128] matching HF's feature extractor output
    """
    (n_samples,) = waveform.shape
    waveform = waveform.astype(np.float32)

    # 1) pad/truncate to exactly 5s
    max_len = BIRDMAE_SR_HZ * BIRDMAE_CLIP_SEC
    if n_samples < max_len:
        pad = max_len - n_samples
        waveform = np.pad(waveform, (0, pad), mode="constant", constant_values=0)
    else:
        waveform = waveform[:max_len]

    # 2) mean-center (per clip)
    waveform = waveform - np.mean(waveform)

    # 3) Kaldi fbank: [T, 128]
    fb = kaldi_fbank(
        waveform,
        htk_compat=True,
        sample_frequency=BIRDMAE_SR_HZ,
        use_energy=False,
        num_mel_bins=BIRDMAE_N_MELS,
        dither=0.0,
        frame_shift=10.0,
    )

    # 4) pad to 512 frames with min value
    t = fb.shape[0]
    if t < BIRDMAE_TARGET_T:
        diff = BIRDMAE_TARGET_T - t
        min_val = float(fb.min())
        fb = np.pad(fb, ((0, diff), (0, 0)), mode="constant", constant_values=min_val)
    elif t > BIRDMAE_TARGET_T:
        fb = fb[:BIRDMAE_TARGET_T]

    fb = (fb - BIRDMAE_MEAN) / (BIRDMAE_STD * 2.0)

    assert fb.shape == (BIRDMAE_TARGET_T, BIRDMAE_N_MELS), fb.shape

    return fb
