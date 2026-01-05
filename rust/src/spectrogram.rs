//! Spectrogram computation: STFT -> Mel filterbank -> log.
//!
//! Matches the Python implementation in bird_mae.transform().

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::{pyclass, pymethods, Bound, PyResult, Python};
use realfft::{RealFftPlanner, RealToComplex};
use std::f32::consts::PI;
use std::sync::Arc;

/// Sparse mel filter: (bin_index, weight) pairs for non-zero weights.
type SparseMelFilter = Vec<(usize, f32)>;

/// Precomputed spectrogram transform.
#[pyclass]
pub struct SpectrogramTransform {
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    mel_filterbank: Vec<SparseMelFilter>,
}

#[pymethods]
impl SpectrogramTransform {
    #[new]
    #[pyo3(signature = (sample_rate=32000, n_fft=1024, hop_length=320, n_mels=128, f_min=0.0, f_max=16000.0))]
    fn py_new(
        sample_rate: u32,
        n_fft: usize,
        hop_length: usize,
        n_mels: usize,
        f_min: f32,
        f_max: f32,
    ) -> Self {
        Self::new(sample_rate, n_fft, hop_length, n_mels, f_min, f_max)
    }

    /// Compute log-mel spectrogram from waveform.
    /// Input: 1D numpy array of mono audio samples
    /// Output: 2D numpy array [n_mels, n_frames]
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        samples: numpy::PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let samples_slice = samples.as_slice()?;
        let spec = self.transform(samples_slice);
        let (n_mels, n_frames) = self.output_shape(samples_slice.len());
        let arr = PyArray1::from_vec(py, spec)
            .reshape([n_mels, n_frames])
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
    ) -> Self {
        let mut planner = RealFftPlanner::new();
        let fft = planner.plan_fft_forward(n_fft);

        // Hann window
        let window: Vec<f32> = (0..n_fft)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / n_fft as f32).cos()))
            .collect();

        // Mel filterbank
        let mel_filterbank = create_mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max);

        Self {
            n_fft,
            hop_length,
            n_mels,
            fft,
            window,
            mel_filterbank,
        }
    }

    /// Compute log-mel spectrogram from waveform.
    pub fn transform(&self, samples: &[f32]) -> Vec<f32> {
        let n_frames = if samples.len() >= self.n_fft {
            (samples.len() - self.n_fft) / self.hop_length + 1
        } else {
            0
        };

        if n_frames == 0 {
            return Vec::new();
        }

        let n_bins = self.n_fft / 2 + 1;
        let mut spectrogram = vec![0.0f32; self.n_mels * n_frames];

        // Pre-allocated scratch buffers (reused across frames)
        let mut input = vec![0.0f32; self.n_fft];
        let mut spectrum = self.fft.make_output_vec();
        let mut power = vec![0.0f32; n_bins];

        for frame in 0..n_frames {
            let start = frame * self.hop_length;

            // Apply window
            for i in 0..self.n_fft {
                input[i] = if start + i < samples.len() {
                    samples[start + i] * self.window[i]
                } else {
                    0.0
                };
            }

            // FFT
            self.fft
                .process(&mut input, &mut spectrum)
                .expect("FFT failed");

            // Power spectrum (reuse buffer)
            for (i, c) in spectrum.iter().enumerate() {
                power[i] = c.norm_sqr();
            }

            // Apply sparse mel filterbank
            for (mel_idx, mel_filter) in self.mel_filterbank.iter().enumerate() {
                let mut mel_energy = 0.0f32;
                for &(bin, weight) in mel_filter {
                    mel_energy += power[bin] * weight;
                }
                spectrogram[mel_idx * n_frames + frame] = (mel_energy.max(1e-10)).ln();
            }
        }

        spectrogram
    }

    /// Get output shape for a given input length.
    pub fn output_shape(&self, n_samples: usize) -> (usize, usize) {
        let n_frames = if n_samples >= self.n_fft {
            (n_samples - self.n_fft) / self.hop_length + 1
        } else {
            0
        };
        (self.n_mels, n_frames)
    }
}

/// Create sparse mel filterbank (only store non-zero weights).
fn create_mel_filterbank(
    sample_rate: u32,
    n_fft: usize,
    n_mels: usize,
    f_min: f32,
    f_max: f32,
) -> Vec<SparseMelFilter> {
    let n_bins = n_fft / 2 + 1;

    // Convert Hz to Mel
    let hz_to_mel = |hz: f32| 2595.0 * (1.0 + hz / 700.0).log10();
    let mel_to_hz = |mel: f32| 700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0);

    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);

    // Mel points (n_mels + 2 for edges)
    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();

    // Convert back to Hz
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Convert to FFT bin indices
    let bin_points: Vec<usize> = hz_points
        .iter()
        .map(|&hz| ((n_fft as f32 * hz / sample_rate as f32).round() as usize).min(n_bins - 1))
        .collect();

    // Create sparse filterbank
    let mut filterbank = Vec::with_capacity(n_mels);

    for m in 0..n_mels {
        let left = bin_points[m];
        let center = bin_points[m + 1];
        let right = bin_points[m + 2];

        let mut filter = Vec::with_capacity(right - left + 1);

        // Rising edge
        for k in left..center {
            if center > left {
                let weight = (k - left) as f32 / (center - left) as f32;
                if weight > 0.0 {
                    filter.push((k, weight));
                }
            }
        }

        // Falling edge
        for k in center..=right {
            if right > center {
                let weight = (right - k) as f32 / (right - center) as f32;
                if weight > 0.0 {
                    filter.push((k, weight));
                }
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
        let transform = SpectrogramTransform::new(32000, 1024, 320, 128, 0.0, 16000.0);

        // 5 seconds at 32kHz = 160000 samples
        let (n_mels, n_frames) = transform.output_shape(160000);
        assert_eq!(n_mels, 128);
        // (160000 - 1024) / 320 + 1 = 497
        assert_eq!(n_frames, 497);
    }

    #[test]
    fn test_spectrogram_basic() {
        let transform = SpectrogramTransform::new(32000, 1024, 320, 128, 0.0, 16000.0);

        // Generate a simple sine wave
        let samples: Vec<f32> = (0..32000)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 32000.0).sin())
            .collect();

        let spec = transform.transform(&samples);
        assert!(!spec.is_empty());
    }
}
