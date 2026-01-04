//! Main data loader that ties everything together.
//!
//! Provides the Python-facing Loader class that:
//! 1. Reads Arrow files sequentially
//! 2. Shuffles samples in a bounded buffer
//! 3. Decodes audio in parallel (Rayon)
//! 4. Computes spectrograms in parallel
//! 5. Assembles batches and returns owned NumPy arrays

use crate::arrow::{count_samples, ArrowError, ArrowReader, Sample};
use crate::decode::{decode_audio, DecodeError};
use crate::shuffle::ShuffleBuffer;
use crate::spectrogram::{SpectrogramConfig, SpectrogramTransform};

use numpy::{PyArray1, PyArrayMethods};
use pyo3::types::{PyDict, PyDictMethods};
use pyo3::{pyclass, pymethods, Bound, PyResult, Python};
use rayon::prelude::*;
use std::sync::Mutex;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum LoaderError {
    #[error("Arrow error: {0}")]
    Arrow(#[from] ArrowError),
    #[error("Decode error: {0}")]
    Decode(#[from] DecodeError),
    #[error("Python error: {0}")]
    Python(#[from] pyo3::PyErr),
    #[error("epoch exhausted")]
    Exhausted,
}

/// Internal state for iteration.
struct LoaderState {
    // Current file index
    file_idx: usize,
    // Current Arrow reader (if any)
    reader: Option<ArrowReader>,
    // Shuffle buffer
    shuffle_buffer: ShuffleBuffer<Sample>,
    // Whether we've finished all files
    files_exhausted: bool,
    // Epoch seed
    seed: u64,
    // File order for this epoch (shuffled)
    file_order: Vec<usize>,
}

/// Python-facing data loader.
#[pyclass]
pub struct Loader {
    arrow_files: Vec<String>,
    /// Canonical start index for each file (stable across epochs).
    file_offsets: Vec<i64>,
    batch_size: usize,
    sample_rate: u32,
    clip_samples: usize,
    state: Mutex<LoaderState>,
    spectrogram: SpectrogramTransform,
}

#[pymethods]
impl Loader {
    #[new]
    #[pyo3(signature = (arrow_files, seed=0, batch_size=64, shuffle_buffer_size=10000, sample_rate=32000, clip_seconds=5.0, n_mels=128, n_fft=1024, hop_length=320, n_threads=8))]
    fn new(
        arrow_files: Vec<String>,
        seed: u64,
        batch_size: usize,
        shuffle_buffer_size: usize,
        sample_rate: u32,
        clip_seconds: f32,
        n_mels: usize,
        n_fft: usize,
        hop_length: usize,
        n_threads: usize,
    ) -> PyResult<Self> {
        if arrow_files.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "arrow_files must not be empty",
            ));
        }

        let clip_samples = (clip_seconds * sample_rate as f32) as usize;
        let n_files = arrow_files.len();

        // Pre-compute file offsets for stable cross-epoch indices
        let mut file_offsets = Vec::with_capacity(n_files);
        let mut offset: i64 = 0;
        for path in &arrow_files {
            file_offsets.push(offset);
            match count_samples(path) {
                Ok(count) => offset += count,
                Err(e) => {
                    return Err(pyo3::exceptions::PyIOError::new_err(format!(
                        "Failed to count samples in {}: {}",
                        path, e
                    )));
                }
            }
        }

        let spec_config = SpectrogramConfig {
            sample_rate,
            n_fft,
            hop_length,
            n_mels,
            f_min: 0.0,
            f_max: sample_rate as f32 / 2.0,
        };

        // Configure rayon thread pool
        rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build_global()
            .ok(); // Ignore error if already initialized

        let state = LoaderState {
            file_idx: 0,
            reader: None,
            shuffle_buffer: ShuffleBuffer::new(shuffle_buffer_size, seed),
            files_exhausted: false,
            seed,
            file_order: (0..n_files).collect(),
        };

        let loader = Self {
            arrow_files,
            file_offsets,
            batch_size,
            sample_rate,
            clip_samples,
            state: Mutex::new(state),
            spectrogram: SpectrogramTransform::new(spec_config),
        };

        loader.reset(seed)?;
        Ok(loader)
    }

    /// Reset for a new epoch with a new seed.
    fn reset(&self, seed: u64) -> PyResult<()> {
        let mut state = self.state.lock().unwrap();

        // Shuffle file order
        use rand::prelude::*;
        use rand::rngs::StdRng;
        let mut rng = StdRng::seed_from_u64(seed);
        state.file_order = (0..self.arrow_files.len()).collect();
        state.file_order.shuffle(&mut rng);

        state.file_idx = 0;
        state.reader = None;
        state.shuffle_buffer.reset(seed);
        state.files_exhausted = false;
        state.seed = seed;

        Ok(())
    }

    /// Number of Arrow files.
    fn n_files(&self) -> usize {
        self.arrow_files.len()
    }

    /// Batch size.
    fn batch_size(&self) -> usize {
        self.batch_size
    }

    fn __iter__(slf: pyo3::PyRef<'_, Self>) -> pyo3::PyRef<'_, Self> {
        slf
    }

    /// Infinite iterator: auto-resets with incremented seed when epoch ends.
    fn __next__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        match self.next_batch(py) {
            Ok(batch) => Ok(batch),
            Err(LoaderError::Exhausted) => {
                // Epoch done, auto-reset with next seed
                let new_seed = {
                    let state = self.state.lock().unwrap();
                    state.seed.wrapping_add(1)
                };
                self.reset(new_seed)?;

                // Try once more - if still exhausted, data is broken
                self.next_batch(py).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "error after reset: {e}"
                    ))
                })
            }
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!("{e}"))),
        }
    }
}

impl Loader {
    /// Get the next batch. Returns Exhausted when epoch is complete.
    fn next_batch<'py>(&self, py: Python<'py>) -> Result<Bound<'py, PyDict>, LoaderError> {
        // Retry loop - if all samples in a batch fail to decode, try again
        // Limit retries to prevent infinite loop if data is corrupt
        const MAX_RETRIES: usize = 10;

        for attempt in 0..MAX_RETRIES {
            let samples = self.collect_batch_samples();

            if samples.is_empty() {
                return Err(LoaderError::Exhausted);
            }

            // Process samples in parallel
            let processed: Vec<ProcessedSample> = samples
                .par_iter()
                .filter_map(|sample| self.process_sample(sample).ok())
                .collect();

            if !processed.is_empty() {
                return self.assemble_batch(py, processed);
            }

            eprintln!(
                "loader.rs: all {} samples in batch failed to decode (attempt {}/{})",
                samples.len(),
                attempt + 1,
                MAX_RETRIES
            );
        }

        panic!(
            "Failed to decode any samples after {} attempts - data may be corrupt",
            MAX_RETRIES
        );
    }

    /// Assemble processed samples into a Python dict batch.
    fn assemble_batch<'py>(
        &self,
        py: Python<'py>,
        processed: Vec<ProcessedSample>,
    ) -> Result<Bound<'py, PyDict>, LoaderError> {
        // Assemble into batch arrays
        let batch_size = processed.len();
        let (n_mels, n_frames) = self.spectrogram.output_shape(self.clip_samples);
        let spec_len = n_mels * n_frames;

        // Build flat vectors
        let mut spec_data = vec![0.0f32; batch_size * spec_len];
        let mut label_data = Vec::with_capacity(batch_size);
        let mut index_data = Vec::with_capacity(batch_size);

        for (i, sample) in processed.iter().enumerate() {
            let src_len = sample.spectrogram.len().min(spec_len);
            spec_data[i * spec_len..i * spec_len + src_len]
                .copy_from_slice(&sample.spectrogram[..src_len]);
            label_data.push(sample.label);
            index_data.push(sample.index);
        }

        // Create numpy arrays from owned vecs
        let spectrogram = PyArray1::from_vec(py, spec_data)
            .reshape([batch_size, spec_len])
            .expect("reshape failed");
        let labels = PyArray1::from_vec(py, label_data);
        let indices = PyArray1::from_vec(py, index_data);

        // Return as dict
        let dict = PyDict::new(py);
        dict.set_item("spectrogram", spectrogram)?;
        dict.set_item("labels", labels)?;
        dict.set_item("indices", indices)?;
        dict.set_item("n_mels", n_mels)?;
        dict.set_item("n_frames", n_frames)?;

        Ok(dict)
    }

    /// Collect enough samples for a batch from the shuffle buffer.
    fn collect_batch_samples(&self) -> Vec<Sample> {
        let mut state = self.state.lock().unwrap();
        let mut samples = Vec::with_capacity(self.batch_size);

        while samples.len() < self.batch_size {
            // Try to get from shuffle buffer (draining phase after finish())
            if let Some(sample) = state.shuffle_buffer.pop() {
                samples.push(sample);
                continue;
            }

            // Feed shuffle buffer and get evicted sample (filling/eviction phase)
            match self.feed_shuffle_buffer(&mut state) {
                Some(sample) => samples.push(sample),
                None => {
                    // Input exhausted - finish() was called, try pop() once more
                    if let Some(sample) = state.shuffle_buffer.pop() {
                        samples.push(sample);
                        continue;
                    }
                    break; // Truly exhausted
                }
            }
        }

        samples
    }

    /// Feed one sample into the shuffle buffer.
    /// Returns Some(evicted) when buffer is full, None when input exhausted.
    fn feed_shuffle_buffer(&self, state: &mut LoaderState) -> Option<Sample> {
        loop {
            if let Some(ref mut reader) = state.reader {
                match reader.next() {
                    Some(Ok(sample)) => {
                        if let Some(evicted) = state.shuffle_buffer.push(sample) {
                            // Buffer full - return evicted sample
                            return Some(evicted);
                        }
                        // Buffer not full yet, keep feeding
                        continue;
                    }
                    Some(Err(e)) => {
                        eprintln!("loader.rs: error reading Arrow file: {}", e);
                        state.reader = None;
                    }
                    None => {
                        // File exhausted
                        state.reader = None;
                    }
                }
            }

            // Need to open next file
            if state.file_idx >= self.arrow_files.len() {
                // All files exhausted - drain remaining buffer
                if !state.files_exhausted {
                    state.files_exhausted = true;
                    state.shuffle_buffer.finish();
                }
                return None; // pop() will drain pending_output
            }

            let file_idx = state.file_order[state.file_idx];
            let path = &self.arrow_files[file_idx];
            let start_index = self.file_offsets[file_idx];
            state.file_idx += 1;

            match ArrowReader::open(path, start_index) {
                Ok(reader) => state.reader = Some(reader),
                Err(e) => eprintln!("loader.rs: error opening {}: {}", path, e),
            }
        }
    }

    /// Process a single sample: decode + resample + truncate/pad + spectrogram.
    fn process_sample(&self, sample: &Sample) -> Result<ProcessedSample, LoaderError> {
        // Decode audio
        let audio = decode_audio(&sample.audio_bytes)?;

        // Resample to target rate
        let mut waveform = audio.resample(self.sample_rate);

        // Truncate or pad to clip_samples
        if waveform.len() > self.clip_samples {
            // Truncate from start (could randomize later)
            waveform.truncate(self.clip_samples);
        } else if waveform.len() < self.clip_samples {
            // Pad with zeros
            waveform.resize(self.clip_samples, 0.0);
        }

        // Compute spectrogram
        let spectrogram = self.spectrogram.transform(&waveform);

        Ok(ProcessedSample {
            spectrogram,
            label: sample.label,
            index: sample.index,
        })
    }
}

struct ProcessedSample {
    spectrogram: Vec<f32>,
    label: i64,
    index: i64,
}
