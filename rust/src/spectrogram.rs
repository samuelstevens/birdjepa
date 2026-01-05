//! Spectrogram computation: STFT -> Mel filterbank -> log.
//!
//! Matches the Python implementation in bird_mae.kaldi_fbank().

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::{Bound, PyResult, Python, pyclass, pymethods};
use realfft::{RealFftPlanner, RealToComplex};
use std::f32::consts::PI;
use std::sync::Arc;

/// Sparse mel filter: (bin_index, weight) pairs for non-zero weights.
type SparseMelFilter = Vec<(usize, f32)>;

/// Precomputed spectrogram transform matching kaldi_fbank.
#[pyclass]
pub struct SpectrogramTransform {
    win_length: usize,
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
    preemphasis: f32,
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    mel_filterbank: Vec<SparseMelFilter>,
}

#[pymethods]
impl SpectrogramTransform {
    #[new]
    #[pyo3(signature = (
        sample_rate = 32000,
        n_fft = 1024,
        hop_length = 320,
        n_mels = 128,
        f_min = 20.0,
        f_max = 0.0,
        win_length = 800,
        preemphasis = 0.97,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn py_new(
        sample_rate: u32,
        n_fft: usize,
        hop_length: usize,
        n_mels: usize,
        f_min: f32,
        f_max: f32,
        win_length: usize,
        preemphasis: f32,
    ) -> Self {
        let f_max = if f_max <= 0.0 {
            sample_rate as f32 / 2.0
        } else {
            f_max
        };
        Self::new(
            sample_rate,
            n_fft,
            hop_length,
            n_mels,
            f_min,
            f_max,
            win_length,
            preemphasis,
        )
    }

    /// Compute log-mel spectrogram from waveform.
    /// Input: 1D numpy array of mono audio samples
    /// Output: 2D numpy array [n_frames, n_mels] (transposed from internal [n_mels, n_frames])
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        samples: numpy::PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let samples_slice = samples.as_slice()?;
        let spec = self.transform(samples_slice);
        let (n_mels, n_frames) = self.output_shape(samples_slice.len());

        // Transpose from [n_mels, n_frames] to [n_frames, n_mels] to match kaldi_fbank
        let mut transposed = vec![0.0f32; n_mels * n_frames];
        for m in 0..n_mels {
            for t in 0..n_frames {
                transposed[t * n_mels + m] = spec[m * n_frames + t];
            }
        }

        let arr = PyArray1::from_vec(py, transposed)
            .reshape([n_frames, n_mels])
            .expect("reshape failed");
        Ok(arr)
    }
}

impl SpectrogramTransform {
    pub fn new(
        sample_rate: u32,
        n_fft: usize,
        hop_length: usize,
        n_mels: usize,
        f_min: f32,
        f_max: f32,
        win_length: usize,
        preemphasis: f32,
    ) -> Self {
        assert!(win_length <= n_fft, "win_length must be <= n_fft");

        let mut planner = RealFftPlanner::new();
        let fft = planner.plan_fft_forward(n_fft);

        // Symmetric Hann window (matches torch.hann_window(periodic=False))
        // Formula: 0.5 - 0.5 * cos(2 * pi * i / (N - 1))
        let window: Vec<f32> = (0..win_length)
            .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / (win_length - 1) as f32).cos())
            .collect();

        // Mel filterbank using Kaldi/Slaney formula
        let mel_filterbank =
            create_kaldi_mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max);

        Self {
            win_length,
            n_fft,
            hop_length,
            n_mels,
            preemphasis,
            fft,
            window,
            mel_filterbank,
        }
    }

    /// Compute log-mel spectrogram from waveform.
    /// Returns flattened [n_mels, n_frames] in row-major order.
    pub fn transform(&self, samples: &[f32]) -> Vec<f32> {
        let n_frames = if samples.len() >= self.win_length {
            (samples.len() - self.win_length) / self.hop_length + 1
        } else {
            0
        };

        if n_frames == 0 {
            return Vec::new();
        }

        let n_bins = self.n_fft / 2 + 1;
        let eps = f32::EPSILON;
        let mut spectrogram = vec![0.0f32; self.n_mels * n_frames];

        // Pre-allocated scratch buffers
        let mut frame_buf = vec![0.0f32; self.win_length];
        let mut fft_input = vec![0.0f32; self.n_fft];
        let mut spectrum = self.fft.make_output_vec();
        let mut power = vec![0.0f32; n_bins];

        for frame_idx in 0..n_frames {
            let start = frame_idx * self.hop_length;

            // Extract frame
            for i in 0..self.win_length {
                frame_buf[i] = if start + i < samples.len() {
                    samples[start + i]
                } else {
                    0.0
                };
            }

            // Remove DC offset (subtract mean)
            let mean: f32 = frame_buf.iter().sum::<f32>() / self.win_length as f32;
            for x in &mut frame_buf {
                *x -= mean;
            }

            // Apply preemphasis: y[i] = x[i] - coef * x[i-1]
            // Process in reverse to avoid overwriting needed values
            if self.preemphasis != 0.0 {
                for i in (1..self.win_length).rev() {
                    frame_buf[i] -= self.preemphasis * frame_buf[i - 1];
                }
                // First sample: use edge padding (x[-1] = x[0])
                // So y[0] = x[0] - coef * x[0] = x[0] * (1 - coef)
                frame_buf[0] *= 1.0 - self.preemphasis;
            }

            // Apply window
            for i in 0..self.win_length {
                frame_buf[i] *= self.window[i];
            }

            // Zero-pad to n_fft
            fft_input[..self.win_length].copy_from_slice(&frame_buf);
            for i in self.win_length..self.n_fft {
                fft_input[i] = 0.0;
            }

            // FFT
            self.fft
                .process(&mut fft_input, &mut spectrum)
                .expect("FFT failed");

            // Power spectrum: |FFT|^2
            for (i, c) in spectrum.iter().enumerate() {
                power[i] = c.norm_sqr();
            }

            // Apply sparse mel filterbank and log
            for (mel_idx, mel_filter) in self.mel_filterbank.iter().enumerate() {
                let mut mel_energy = 0.0f32;
                for &(bin, weight) in mel_filter {
                    mel_energy += power[bin] * weight;
                }
                spectrogram[mel_idx * n_frames + frame_idx] = mel_energy.max(eps).ln();
            }
        }

        spectrogram
    }

    /// Get output shape for a given input length.
    pub fn output_shape(&self, n_samples: usize) -> (usize, usize) {
        let n_frames = if n_samples >= self.win_length {
            (n_samples - self.win_length) / self.hop_length + 1
        } else {
            0
        };
        (self.n_mels, n_frames)
    }
}

/// Convert Hz to mel using Kaldi/Slaney formula (natural log).
fn hz_to_mel(hz: f32) -> f32 {
    1127.0 * (1.0 + hz / 700.0).ln()
}

/// Convert mel to Hz using Kaldi/Slaney formula.
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * ((mel / 1127.0).exp() - 1.0)
}

/// Create mel filterbank matching Kaldi's get_mel_banks.
fn create_kaldi_mel_filterbank(
    sample_rate: u32,
    n_fft: usize,
    n_mels: usize,
    f_min: f32,
    f_max: f32,
) -> Vec<SparseMelFilter> {
    let num_fft_bins = n_fft / 2; // Kaldi uses n_fft/2, not n_fft/2+1
    let fft_bin_width = sample_rate as f32 / n_fft as f32;

    let mel_low = hz_to_mel(f_min);
    let mel_high = hz_to_mel(f_max);
    let mel_freq_delta = (mel_high - mel_low) / (n_mels + 1) as f32;

    let mut filterbank = Vec::with_capacity(n_mels);

    for m in 0..n_mels {
        let left_mel = mel_low + m as f32 * mel_freq_delta;
        let center_mel = mel_low + (m + 1) as f32 * mel_freq_delta;
        let right_mel = mel_low + (m + 2) as f32 * mel_freq_delta;

        let mut filter = Vec::new();

        // For each FFT bin, compute triangular filter weight
        for k in 0..num_fft_bins {
            let freq_hz = fft_bin_width * k as f32;
            let freq_mel = hz_to_mel(freq_hz);

            let weight = if freq_mel > left_mel && freq_mel < center_mel {
                // Rising edge
                (freq_mel - left_mel) / (center_mel - left_mel)
            } else if freq_mel >= center_mel && freq_mel < right_mel {
                // Falling edge
                (right_mel - freq_mel) / (right_mel - center_mel)
            } else {
                0.0
            };

            if weight > 0.0 {
                filter.push((k, weight));
            }
        }

        filterbank.push(filter);
    }

    filterbank
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectrogram_output_shape() {
        let transform = SpectrogramTransform::new(32000, 1024, 320, 128, 20.0, 16000.0, 800, 0.97);

        // 5 seconds at 32kHz = 160000 samples
        // n_frames = (160000 - 800) / 320 + 1 = 498
        let (n_mels, n_frames) = transform.output_shape(160000);
        assert_eq!(n_mels, 128);
        assert_eq!(n_frames, 498);
    }

    #[test]
    fn test_spectrogram_basic() {
        let transform = SpectrogramTransform::new(32000, 1024, 320, 128, 20.0, 16000.0, 800, 0.97);

        // Generate a simple sine wave
        let samples: Vec<f32> = (0..32000)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 32000.0).sin())
            .collect();

        let spec = transform.transform(&samples);
        assert!(!spec.is_empty());
    }

    #[test]
    fn test_mel_conversion_roundtrip() {
        let hz = 1000.0;
        let mel = hz_to_mel(hz);
        let hz_back = mel_to_hz(mel);
        assert!((hz - hz_back).abs() < 0.01);
    }
}
