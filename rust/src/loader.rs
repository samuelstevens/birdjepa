//! Pipelined data loader.
//!
//! Architecture:
//! I/O Thread -> Raw Channel -> Worker Threads -> Shuffle Buffer -> Main Thread
//!
//! - I/O thread reads Arrow files sequentially from GPFS
//! - Workers decode audio, compute spectrograms in parallel
//! - Shuffle buffer provides randomness with configurable minimum fill
//! - Main thread assembles batches and returns to Python

use crate::arrow::{ArrowReader, Sample, count_samples};
use crate::decode::decode_audio;
use crate::shuffle::ConcurrentShuffleBuffer;
use crate::spectrogram::SpectrogramTransform;

use crossbeam::channel::{Receiver, RecvTimeoutError, SendTimeoutError, Sender, bounded};
use numpy::{PyArray1, PyArrayMethods};
use pyo3::types::{PyDict, PyDictMethods};
use pyo3::{Bound, PyResult, Python, pyclass, pymethods};
use rand::prelude::*;
use rand::rngs::StdRng;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// Raw sample from Arrow (bytes not yet decoded).
struct RawSample {
    audio_bytes: Vec<u8>,
    label: i64,
    index: i64,
    seed: u64,
}

/// Processed sample ready for batching.
struct ProcessedSample {
    spectrogram: Vec<f32>,
    label: i64,
    index: i64,
}

/// Python-facing pipelined data loader.
#[pyclass]
pub struct Loader {
    // Config
    batch_size: usize,
    clip_samples: usize,
    spectrogram: Arc<SpectrogramTransform>,

    // Pipeline state
    shuffle_buffer: Arc<ConcurrentShuffleBuffer<ProcessedSample>>,
    shutdown: Arc<AtomicBool>,
    io_handle: Option<JoinHandle<()>>,
    worker_handles: Vec<JoinHandle<()>>,
    monitor_handle: Option<JoinHandle<()>>,
}

#[pymethods]
impl Loader {
    #[new]
    #[pyo3(signature = (
        arrow_files,
        seed = 0,
        batch_size = 64,
        shuffle_buffer_size = 10000,
        shuffle_min_size = 8000,
        sample_rate = 32000,
        clip_seconds = 5.0,
        n_mels = 128,
        n_fft = 1024,
        hop_length = 320,
        n_workers = 8,
        raw_channel_size = 256,
        infinite = true,
        augment = true,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        arrow_files: Vec<String>,
        seed: u64,
        batch_size: usize,
        shuffle_buffer_size: usize,
        shuffle_min_size: usize,
        sample_rate: u32,
        clip_seconds: f32,
        n_mels: usize,
        n_fft: usize,
        hop_length: usize,
        n_workers: usize,
        raw_channel_size: usize,
        infinite: bool,
        augment: bool,
    ) -> PyResult<Self> {
        if arrow_files.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "arrow_files must not be empty",
            ));
        }
        if shuffle_buffer_size == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "shuffle_buffer_size must be > 0",
            ));
        }
        if raw_channel_size == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "raw_channel_size must be > 0",
            ));
        }
        if batch_size == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "batch_size must be > 0",
            ));
        }
        if n_workers == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "n_workers must be > 0",
            ));
        }

        let clip_samples = (clip_seconds * sample_rate as f32) as usize;
        let n_files = arrow_files.len();
        let f_max = sample_rate as f32 / 2.0;

        // Pre-compute file offsets for globally unique indices
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

        let spectrogram = Arc::new(SpectrogramTransform::new(
            sample_rate,
            n_fft,
            hop_length,
            n_mels,
            0.0,
            f_max,
        ));

        let shuffle_buffer = Arc::new(ConcurrentShuffleBuffer::new(
            shuffle_buffer_size,
            shuffle_min_size,
            seed,
        ));
        let shutdown = Arc::new(AtomicBool::new(false));

        let mut loader = Self {
            batch_size,
            clip_samples,
            spectrogram: Arc::clone(&spectrogram),
            shuffle_buffer: Arc::clone(&shuffle_buffer),
            shutdown: Arc::clone(&shutdown),
            io_handle: None,
            worker_handles: Vec::new(),
            monitor_handle: None,
        };

        loader.start_pipeline(
            arrow_files,
            file_offsets,
            seed,
            sample_rate,
            n_workers,
            raw_channel_size,
            infinite,
            augment,
        );

        Ok(loader)
    }

    fn __iter__(slf: pyo3::PyRef<'_, Self>) -> pyo3::PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let mut samples = Vec::with_capacity(self.batch_size);

        while samples.len() < self.batch_size {
            // Release GIL for blocking operation
            let sample = py.detach(|| self.shuffle_buffer.try_pop(Duration::from_millis(100)));

            match sample {
                Some(s) => samples.push(s),
                None => {
                    if self.shuffle_buffer.is_closed() {
                        break; // Buffer closed and empty
                    }
                    // Timeout - check for Python signals (Ctrl-C)
                    py.check_signals()?;
                }
            }
        }

        if samples.is_empty() {
            return Ok(None); // StopIteration
        }

        Ok(Some(self.assemble_batch(py, samples)?))
    }
}

impl Loader {
    #[allow(clippy::too_many_arguments)]
    fn start_pipeline(
        &mut self,
        arrow_files: Vec<String>,
        file_offsets: Vec<i64>,
        seed: u64,
        sample_rate: u32,
        n_workers: usize,
        raw_channel_size: usize,
        infinite: bool,
        augment: bool,
    ) {
        let (raw_tx, raw_rx) = bounded(raw_channel_size);

        // Worker completion counter for monitor thread
        let workers_alive = Arc::new(AtomicUsize::new(n_workers));

        // Spawn I/O thread
        self.io_handle = Some({
            let shutdown = Arc::clone(&self.shutdown);
            thread::spawn(move || {
                io_thread_main(
                    arrow_files,
                    file_offsets,
                    seed,
                    raw_tx,
                    shutdown,
                    infinite,
                    augment,
                );
            })
        });

        // Spawn workers
        self.worker_handles = (0..n_workers)
            .map(|_| {
                let raw_rx = raw_rx.clone();
                let shuffle_buffer = Arc::clone(&self.shuffle_buffer);
                let spectrogram = Arc::clone(&self.spectrogram);
                let shutdown = Arc::clone(&self.shutdown);
                let workers_alive = Arc::clone(&workers_alive);
                let clip_samples = self.clip_samples;

                thread::spawn(move || {
                    worker_thread_main(
                        raw_rx,
                        shuffle_buffer,
                        spectrogram,
                        sample_rate,
                        clip_samples,
                        shutdown,
                        augment,
                        workers_alive,
                    );
                })
            })
            .collect();

        // Spawn monitor thread: waits for workers to finish, then closes buffer
        self.monitor_handle = Some({
            let shuffle_buffer = Arc::clone(&self.shuffle_buffer);
            let shutdown = Arc::clone(&self.shutdown);

            thread::spawn(move || {
                // Wait for all workers to finish
                while workers_alive.load(Ordering::SeqCst) > 0 {
                    thread::sleep(Duration::from_millis(50));
                    if shutdown.load(Ordering::Relaxed) {
                        return; // Shutdown in progress, stop_pipeline will close buffer
                    }
                }

                // All workers done - close buffer so main thread knows we're finished
                shuffle_buffer.close();
            })
        });
    }

    fn stop_pipeline(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        self.shuffle_buffer.close();

        if let Some(h) = self.io_handle.take() {
            let _ = h.join();
        }
        for h in self.worker_handles.drain(..) {
            let _ = h.join();
        }
        if let Some(h) = self.monitor_handle.take() {
            let _ = h.join();
        }
    }

    fn assemble_batch<'py>(
        &self,
        py: Python<'py>,
        processed: Vec<ProcessedSample>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let batch_size = processed.len();
        let (n_mels, n_frames) = self.spectrogram.output_shape(self.clip_samples);
        let spec_len = n_mels * n_frames;

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

        let spectrogram = PyArray1::from_vec(py, spec_data)
            .reshape([batch_size, spec_len])
            .expect("reshape failed");
        let labels = PyArray1::from_vec(py, label_data);
        let indices = PyArray1::from_vec(py, index_data);

        let dict = PyDict::new(py);
        dict.set_item("spectrogram", spectrogram)?;
        dict.set_item("labels", labels)?;
        dict.set_item("indices", indices)?;
        dict.set_item("n_mels", n_mels)?;
        dict.set_item("n_frames", n_frames)?;

        Ok(dict)
    }
}

impl Drop for Loader {
    fn drop(&mut self) {
        self.stop_pipeline();
    }
}

// --- I/O Thread ---

fn io_thread_main(
    arrow_files: Vec<String>,
    file_offsets: Vec<i64>,
    initial_seed: u64,
    raw_tx: Sender<RawSample>,
    shutdown: Arc<AtomicBool>,
    infinite: bool,
    augment: bool,
) {
    let mut epoch_seed = initial_seed;

    loop {
        // Shuffle file order for this pass
        let mut file_order: Vec<usize> = (0..arrow_files.len()).collect();
        file_order.shuffle(&mut StdRng::seed_from_u64(epoch_seed));

        for &file_idx in &file_order {
            if shutdown.load(Ordering::Relaxed) {
                return;
            }

            let path = &arrow_files[file_idx];
            let start_index = file_offsets[file_idx];

            let reader = match ArrowReader::open(path, start_index) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("io_thread: error opening {}: {}", path, e);
                    continue;
                }
            };

            for sample_result in reader {
                if shutdown.load(Ordering::Relaxed) {
                    return;
                }

                match sample_result {
                    Ok(sample) => {
                        if !send_sample(&raw_tx, &shutdown, sample, epoch_seed, augment) {
                            return;
                        }
                    }
                    Err(e) => eprintln!("io_thread: error reading sample: {}", e),
                }
            }
        }

        if !infinite {
            return; // Single pass for eval
        }

        epoch_seed = epoch_seed.wrapping_add(1);
    }
}

/// Send sample with retry on timeout, checking shutdown between attempts.
fn send_sample(
    raw_tx: &Sender<RawSample>,
    shutdown: &Arc<AtomicBool>,
    sample: Sample,
    epoch_seed: u64,
    augment: bool,
) -> bool {
    let mut raw = RawSample {
        audio_bytes: sample.audio_bytes,
        label: sample.label,
        index: sample.index,
        seed: if augment {
            epoch_seed.wrapping_add(sample.index as u64)
        } else {
            0
        },
    };

    loop {
        if shutdown.load(Ordering::Relaxed) {
            return false;
        }
        match raw_tx.send_timeout(raw, Duration::from_millis(100)) {
            Ok(()) => return true,
            Err(SendTimeoutError::Timeout(returned)) => {
                raw = returned; // Retry with same sample
            }
            Err(SendTimeoutError::Disconnected(_)) => return false,
        }
    }
}

// --- Worker Thread ---

fn worker_thread_main(
    raw_rx: Receiver<RawSample>,
    shuffle_buffer: Arc<ConcurrentShuffleBuffer<ProcessedSample>>,
    spectrogram: Arc<SpectrogramTransform>,
    sample_rate: u32,
    clip_samples: usize,
    shutdown: Arc<AtomicBool>,
    augment: bool,
    workers_alive: Arc<AtomicUsize>,
) {
    // Ensure counter is decremented even on panic
    struct CounterGuard(Arc<AtomicUsize>);
    impl Drop for CounterGuard {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::SeqCst);
        }
    }
    let _guard = CounterGuard(workers_alive);

    loop {
        let raw = match raw_rx.recv_timeout(Duration::from_millis(100)) {
            Ok(r) => r,
            Err(RecvTimeoutError::Timeout) => {
                if shutdown.load(Ordering::Relaxed) {
                    return;
                }
                continue;
            }
            Err(RecvTimeoutError::Disconnected) => return,
        };

        // Decode audio
        let audio = match decode_audio(&raw.audio_bytes) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("worker: decode error for index {}: {}", raw.index, e);
                continue;
            }
        };

        // Resample
        let mut waveform = audio.resample(sample_rate);

        // Crop/pad: random for training, center for eval
        waveform = if augment {
            random_crop_or_pad(waveform, clip_samples, raw.seed, raw.index)
        } else {
            center_crop_or_pad(waveform, clip_samples)
        };

        let spec = spectrogram.transform(&waveform);

        let processed = ProcessedSample {
            spectrogram: spec,
            label: raw.label,
            index: raw.index,
        };

        if !shuffle_buffer.push(processed) {
            return;
        }
    }
}

// --- Cropping utilities ---

/// Center crop (if too long) or zero-pad (if too short). Deterministic.
fn center_crop_or_pad(waveform: Vec<f32>, clip_len: usize) -> Vec<f32> {
    let len = waveform.len();
    if len < clip_len {
        // Pad equally on both sides
        let pad_total = clip_len - len;
        let pad_left = pad_total / 2;
        let mut padded = vec![0.0; clip_len];
        padded[pad_left..pad_left + len].copy_from_slice(&waveform);
        padded
    } else if len == clip_len {
        waveform
    } else {
        // Center crop
        let start = (len - clip_len) / 2;
        waveform[start..start + clip_len].to_vec()
    }
}

/// Random crop (if too long) or zero-pad (if too short). Deterministic per seed+index.
fn random_crop_or_pad(mut waveform: Vec<f32>, clip_len: usize, seed: u64, index: i64) -> Vec<f32> {
    if waveform.len() < clip_len {
        waveform.resize(clip_len, 0.0);
        return waveform;
    }
    if waveform.len() == clip_len {
        return waveform;
    }
    let max_start = waveform.len() - clip_len;
    let r = splitmix64(seed ^ (index as u64).wrapping_mul(0xD6E8FEB86659FD93));
    let start = (r as usize) % (max_start + 1);
    waveform[start..start + clip_len].to_vec()
}

fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}
