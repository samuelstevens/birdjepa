//! Rust data loader for birdjepa.
//!
//! Single-process, multi-threaded data loading with:
//! - Sequential Arrow/Parquet reading
//! - Shuffle buffer for randomness
//! - Parallel audio decoding (symphonia)
//! - Parallel spectrogram computation (FFT -> log-mel)
//! - Batch assembly returning owned NumPy arrays

mod arrow;
pub mod decode;
mod loader;
mod shuffle;
pub mod spectrogram;

use numpy::PyArray1;
use pyo3::types::{PyModule, PyModuleMethods};
use pyo3::{Bound, PyResult, Python, pyfunction, pymodule, wrap_pyfunction};

/// Decode audio bytes to waveform (for benchmarking).
#[pyfunction]
fn decode_audio<'py>(py: Python<'py>, bytes: &[u8]) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let audio = decode::decode_audio(bytes)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
    let waveform = audio.resample(32000);
    Ok(PyArray1::from_vec(py, waveform))
}

/// Python module entry point.
#[pymodule]
fn _rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<loader::Loader>()?;
    m.add_class::<spectrogram::SpectrogramTransform>()?;
    m.add_function(wrap_pyfunction!(decode_audio, m)?)?;
    Ok(())
}
