//! Rust data loader for birdjepa.
//!
//! Single-process, multi-threaded data loading with:
//! - Sequential Arrow/Parquet reading
//! - Shuffle buffer for randomness
//! - Parallel audio decoding (symphonia)
//! - Parallel spectrogram computation (FFT -> log-mel)
//! - Batch assembly returning owned NumPy arrays

mod arrow;
mod decode;
mod loader;
mod shuffle;
mod spectrogram;

use pyo3::types::{PyModule, PyModuleMethods};
use pyo3::{pymodule, Bound, PyResult};

/// Python module entry point.
#[pymodule]
fn _rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<loader::Loader>()?;
    m.add_class::<spectrogram::SpectrogramTransform>()?;
    Ok(())
}
