# Pipelined Loader Design

## Architecture Overview

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  I/O Thread │────▶│ Raw Channel │────▶│ Worker Pool │────▶│  Concurrent │
│ (Arrow/GPFS)│     │  (bounded)  │     │(decode+spec)│     │ShuffleBuffer│
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                   │
                                                                   ▼
                                                            ┌─────────────┐
                                                            │ Main Thread │
                                                            │ (batch asm) │
                                                            └─────────────┘
```

**Key design choices:**
- Infinite iteration for training (loops forever until shutdown)
- Finite iteration for evaluation (single pass, each sample exactly once)
- Training uses random crop augmentation; eval uses center crop (no augmentation)
- Clean Ctrl-C handling via timeout-based polling and Drop impl
- Per-sample seed for varying augmentation across epoch loops (training only)

## Data Structures

### RawSample (I/O thread -> workers)

```rust
struct RawSample {
    audio_bytes: Vec<u8>,
    label: i64,
    index: i64,  // Globally unique (includes file offset)
    seed: u64,   // Per-sample seed for augmentation (training only)
}
```

### ProcessedSample (workers -> shuffle buffer)

```rust
struct ProcessedSample {
    spectrogram: Vec<f32>,
    label: i64,
    index: i64,
}
```

## ConcurrentShuffleBuffer

Thread-safe shuffle buffer with configurable minimum fill level for entropy.

```rust
// src/shuffle.rs

use std::sync::{Condvar, Mutex};
use std::time::Duration;
use rand::prelude::*;
use rand::rngs::StdRng;

pub struct ConcurrentShuffleBuffer<T> {
    inner: Mutex<ShuffleInner<T>>,
    capacity: usize,
    min_size: usize,  // 0 means no minimum
    not_full: Condvar,
    ready: Condvar,
}

struct ShuffleInner<T> {
    buffer: Vec<T>,
    rng: StdRng,
    closed: bool,
}

impl<T> ConcurrentShuffleBuffer<T> {
    pub fn new(capacity: usize, min_size: usize, seed: u64) -> Self {
        assert!(capacity > 0, "capacity must be > 0");
        assert!(min_size <= capacity, "min_size must be <= capacity");
        Self {
            inner: Mutex::new(ShuffleInner {
                buffer: Vec::with_capacity(capacity),
                rng: StdRng::seed_from_u64(seed),
                closed: false,
            }),
            capacity,
            min_size,
            not_full: Condvar::new(),
            ready: Condvar::new(),
        }
    }

    /// Push a sample. Blocks if buffer is at capacity.
    /// Returns false if buffer is closed (shutdown).
    pub fn push(&self, sample: T) -> bool {
        let mut guard = self.inner.lock().unwrap();
        loop {
            if guard.closed {
                return false;
            }
            if guard.buffer.len() < self.capacity {
                guard.buffer.push(sample);
                // Notify if we've reached min_size (or min_size is 0 and we have data)
                let dominated = self.min_size == 0 || guard.buffer.len() >= self.min_size;
                if dominated {
                    self.ready.notify_one();
                }
                return true;
            }
            let (new_guard, _) = self.not_full
                .wait_timeout(guard, Duration::from_millis(100))
                .unwrap();
            guard = new_guard;
        }
    }

    /// Try to pop a random sample with timeout.
    /// Returns Some(sample) if available, None if timed out or closed+empty.
    /// Use is_closed() to distinguish timeout vs closed.
    pub fn try_pop(&self, timeout: Duration) -> Option<T> {
        let mut guard = self.inner.lock().unwrap();

        // First try without waiting
        if let Some(sample) = self.try_pop_inner(&mut guard) {
            return Some(sample);
        }

        if guard.closed {
            return None;
        }

        // Wait once with timeout
        let (mut new_guard, _) = self.ready.wait_timeout(guard, timeout).unwrap();

        // Try again after wait
        self.try_pop_inner(&mut new_guard)
    }

    /// Internal helper: try to pop if conditions are met.
    fn try_pop_inner(&self, guard: &mut std::sync::MutexGuard<ShuffleInner<T>>) -> Option<T> {
        let len = guard.buffer.len();

        // Can pop if: min_size is 0 and we have data, OR we have >= min_size
        let can_pop = len > 0 && (self.min_size == 0 || len >= self.min_size);
        if can_pop {
            let idx = guard.rng.gen_range(0..len);
            let sample = guard.buffer.swap_remove(idx);
            self.not_full.notify_one();
            return Some(sample);
        }

        // Shutdown: drain remaining regardless of min_size
        if guard.closed && len > 0 {
            let idx = guard.rng.gen_range(0..len);
            let sample = guard.buffer.swap_remove(idx);
            self.not_full.notify_one();
            return Some(sample);
        }

        None
    }

    /// Check if buffer is closed.
    pub fn is_closed(&self) -> bool {
        self.inner.lock().unwrap().closed
    }

    /// Close the buffer. Wakes all waiters, causes push() to return false.
    pub fn close(&self) {
        let mut guard = self.inner.lock().unwrap();
        guard.closed = true;
        self.not_full.notify_all();
        self.ready.notify_all();
    }
}
```

## Channels

Using `crossbeam::channel` for bounded MPSC channels.

```rust
use crossbeam::channel::{bounded, Sender, Receiver, RecvTimeoutError, SendTimeoutError};

let (raw_tx, raw_rx) = bounded::<RawSample>(raw_channel_size);
```

**Why crossbeam?**
- `bounded()` blocks senders when full (backpressure)
- Thread-safe: workers share `raw_rx` via `Clone`
- Clean shutdown: dropping `raw_tx` causes `raw_rx.recv()` to return `Err`
- `recv_timeout()` and `send_timeout()` allow periodic shutdown checks
- `SendTimeoutError::Timeout(T)` returns ownership, avoiding clones on retry

## Thread Implementation

### I/O Thread

Supports both infinite (training) and finite (eval) modes.

```rust
fn io_thread_main(
    arrow_files: Vec<String>,
    file_offsets: Vec<i64>,
    initial_seed: u64,
    raw_tx: Sender<RawSample>,
    shutdown: Arc<AtomicBool>,
    infinite: bool,
    augment: bool,
) {
    assert!(!arrow_files.is_empty(), "arrow_files must not be empty");

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
                        let mut raw = RawSample {
                            audio_bytes: sample.audio_bytes,
                            label: sample.label,
                            index: sample.index,
                            // Seed only matters for training (augment=true)
                            seed: if augment {
                                epoch_seed.wrapping_add(sample.index as u64)
                            } else {
                                0
                            },
                        };

                        // Send with retry, reusing returned sample (no clone)
                        loop {
                            if shutdown.load(Ordering::Relaxed) {
                                return;
                            }
                            match raw_tx.send_timeout(raw, Duration::from_millis(100)) {
                                Ok(()) => break,
                                Err(SendTimeoutError::Timeout(returned)) => {
                                    raw = returned;  // Retry with same sample
                                }
                                Err(SendTimeoutError::Disconnected(_)) => return,
                            }
                        }
                    }
                    Err(e) => eprintln!("io_thread: error reading sample: {}", e),
                }
            }
        }

        if !infinite {
            // Finite mode: just exit. Monitor thread will close buffer after workers drain.
            // (raw_tx dropped here signals workers that input is done)
            return;
        }

        // Next pass with different shuffle and augmentation seeds
        epoch_seed = epoch_seed.wrapping_add(1);
    }
}
```

### Worker Thread

```rust
fn worker_thread_main(
    raw_rx: Receiver<RawSample>,
    shuffle_buffer: Arc<ConcurrentShuffleBuffer<ProcessedSample>>,
    spectrogram: Arc<SpectrogramTransform>,  // Must be Send + Sync
    sample_rate: u32,
    clip_samples: usize,
    shutdown: Arc<AtomicBool>,
    augment: bool,
    workers_alive: Arc<AtomicUsize>,  // Decrement on exit for monitor thread
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
```

**Thread safety note**: `SpectrogramTransform` must be `Send + Sync`. The current implementation uses `Arc<dyn RealToComplex<f32>>` from realfft which is thread-safe for read-only use.

## Loader Structure

```rust
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

#[pyclass]
pub struct Loader {
    // Config (kept for batch assembly)
    batch_size: usize,
    clip_samples: usize,
    spectrogram: Arc<SpectrogramTransform>,

    // Pipeline state (always running after construction)
    shuffle_buffer: Arc<ConcurrentShuffleBuffer<ProcessedSample>>,
    shutdown: Arc<AtomicBool>,
    io_handle: Option<JoinHandle<()>>,
    worker_handles: Vec<JoinHandle<()>>,
    monitor_handle: Option<JoinHandle<()>>,  // Closes buffer after workers finish (finite mode)
}

impl Loader {
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
            let workers_alive = Arc::clone(&workers_alive);

            thread::spawn(move || {
                // Wait for all workers to finish
                while workers_alive.load(Ordering::SeqCst) > 0 {
                    thread::sleep(Duration::from_millis(50));
                    if shutdown.load(Ordering::Relaxed) {
                        return;  // Shutdown in progress, stop_pipeline will close buffer
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
}

impl Drop for Loader {
    fn drop(&mut self) {
        self.stop_pipeline();
    }
}
```

## Public API

```rust
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
    fn new(...) -> PyResult<Self> {
        if arrow_files.is_empty() {
            return Err(PyValueError::new_err("arrow_files must not be empty"));
        }
        if shuffle_buffer_size == 0 {
            return Err(PyValueError::new_err("shuffle_buffer_size must be > 0"));
        }
        if raw_channel_size == 0 {
            return Err(PyValueError::new_err("raw_channel_size must be > 0"));
        }
        // Validate, compute file_offsets, create spectrogram transform
        // Call start_pipeline(seed)
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let mut samples = Vec::with_capacity(self.batch_size);

        while samples.len() < self.batch_size {
            // Release GIL for blocking operation (100ms timeout for signal responsiveness)
            let sample = py.detach(|| {
                self.shuffle_buffer.try_pop(Duration::from_millis(100))
            });

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
            return Ok(None);  // StopIteration
        }

        Ok(Some(self.assemble_batch(py, samples)?))
    }
}
```

## Ctrl-C Handling

**Realistic latency bounds:**
- Workers: up to 100ms (recv_timeout) + decode time (~15ms) ≈ 115ms
- I/O thread: up to 100ms (send_timeout) + GPFS read latency
- Main thread `__next__`: ~100ms (single try_pop timeout) + py.check_signals() overhead

**Shutdown sequence:**
1. User presses Ctrl-C during `__next__`
2. `py.check_signals()` raises `KeyboardInterrupt`
3. Python exception propagates, `Loader` is dropped
4. `Drop::drop()` calls `stop_pipeline()`
5. `shutdown` flag set, `shuffle_buffer.close()` called
6. Threads see shutdown within ~100ms each, exit
7. `join()` completes, clean termination

**Note**: With `try_pop(100ms)` + `py.check_signals()` in the loop, Ctrl-C response is ~100ms regardless of batch_size.

## Training vs Eval Mode

| Parameter | Training | Eval |
|-----------|----------|------|
| `infinite` | `true` | `false` |
| `augment` | `true` | `false` |
| `shuffle_buffer_size` | 10000 | 1000 |
| `shuffle_min_size` | 8000 | 0 |
| `n_workers` | 8 | 4 |

**Training mode** (`infinite=true, augment=true`):
- Loops forever through files with shuffled order
- Random crop augmentation (varies each epoch via per-sample seed)
- Shuffle buffer enforces min_size for good entropy

**Eval mode** (`infinite=false, augment=false`):
- Single pass through all files (each sample exactly once)
- Center crop (deterministic, no augmentation)
- min_size=0 means samples pass through immediately
- I/O thread calls `shuffle_buffer.close()` when done

## Configuration Recommendations

| Parameter | Training | Eval | Rationale |
|-----------|----------|------|-----------|
| `shuffle_buffer_size` | 10000 | 1000 | Eval needs less buffering |
| `shuffle_min_size` | 8000 | 0 | Eval: no min, drain immediately |
| `infinite` | true | false | Training loops, eval doesn't |
| `augment` | true | false | Eval uses center crop |
| `n_workers` | 8 | 4 | Eval can use fewer workers |

## Memory Budget

- Raw channel: 256 samples × ~500KB = ~128 MB
- Shuffle buffer: 10000 samples × ~250KB = ~2.5 GB
- Total: ~2.6 GB

Adjust `shuffle_capacity` and `raw_channel_size` to fit memory constraints.

## Files to Modify

1. **`src/shuffle.rs`** - Replace with `ConcurrentShuffleBuffer`
2. **`src/loader.rs`** - Rewrite with pipeline architecture
3. **`Cargo.toml`** - Already has `crossbeam = "0.8"`

## Review Feedback Addressed

| Issue | Resolution |
|-------|------------|
| Epoch seed not varying for workers | Pass `seed` per-sample in `RawSample` |
| Ctrl-C latency overstated | Documented realistic bounds (`batch_size * 100ms`) |
| I/O thread blocking send | Use `send_timeout` with retry, reuse returned sample |
| Empty arrow_files deadlock | Assert non-empty at construction |
| SpectrogramTransform thread safety | Documented `Send + Sync` requirement |
| Finite epochs for eval | Added `infinite` flag, I/O thread closes buffer |
| Index uniqueness | Confirmed globally unique (includes file offset) |
| Clone on send retry | Use `SendTimeoutError::Timeout(sample)` to reclaim ownership |
| min_size == 0 panics | Handle min_size == 0 as "no minimum" |
| Eval augmentation | Added `augment` flag, center crop for eval |
| Finite-mode shutdown | I/O thread calls `shuffle_buffer.close()` |
