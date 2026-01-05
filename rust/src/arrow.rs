//! Arrow IPC reading (HuggingFace datasets use Arrow IPC streaming format).

use arrow::array::{Array, BinaryArray, Int64Array, RecordBatch};
use arrow::ipc::reader::StreamReader;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ArrowError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    #[error("Missing column: {0}")]
    MissingColumn(String),
    #[error("Invalid column type for {0}")]
    InvalidColumnType(String),
}

/// A single sample from the Arrow file.
#[derive(Clone)]
pub struct Sample {
    pub audio_bytes: Vec<u8>,
    pub label: i64,
    pub index: i64,
}

/// Reads an Arrow IPC stream and yields samples sequentially.
pub struct ArrowReader {
    reader: StreamReader<BufReader<File>>,
    current_batch: Option<RecordBatch>,
    row_idx: usize,
    sample_idx: i64, // Running counter for unique sample indices
    audio_col_idx: usize,
    label_col_idx: usize,
}

/// Count samples in an Arrow file without fully reading it.
pub fn count_samples<P: AsRef<Path>>(path: P) -> Result<i64, ArrowError> {
    let file = File::open(path)?;
    let reader = StreamReader::try_new(BufReader::new(file), None)?;
    let mut count: i64 = 0;
    for batch in reader {
        count += batch?.num_rows() as i64;
    }
    Ok(count)
}

impl ArrowReader {
    pub fn open<P: AsRef<Path>>(path: P, start_index: i64) -> Result<Self, ArrowError> {
        let file = File::open(path)?;
        let reader = StreamReader::try_new(BufReader::new(file), None)?;

        // Find column indices
        let schema = reader.schema();
        let audio_col_idx = schema
            .fields()
            .iter()
            .position(|f| f.name() == "audio")
            .ok_or_else(|| ArrowError::MissingColumn("audio".to_string()))?;
        let label_col_idx = schema
            .fields()
            .iter()
            .position(|f| f.name() == "ebird_code")
            .ok_or_else(|| ArrowError::MissingColumn("ebird_code".to_string()))?;

        Ok(Self {
            reader,
            current_batch: None,
            row_idx: 0,
            sample_idx: start_index,
            audio_col_idx,
            label_col_idx,
        })
    }
}

impl Iterator for ArrowReader {
    type Item = Result<Sample, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Try to get from current batch
            if let Some(batch) = &self.current_batch {
                if self.row_idx < batch.num_rows() {
                    let index = self.sample_idx;
                    self.sample_idx += 1;
                    let sample = extract_sample(
                        batch,
                        self.row_idx,
                        self.audio_col_idx,
                        self.label_col_idx,
                        index,
                    );
                    self.row_idx += 1;
                    return Some(sample);
                }
            }

            // Need next batch
            match self.reader.next() {
                Some(Ok(batch)) => {
                    self.current_batch = Some(batch);
                    self.row_idx = 0;
                }
                Some(Err(e)) => return Some(Err(ArrowError::Arrow(e))),
                None => return None,
            }
        }
    }
}

fn extract_sample(
    batch: &RecordBatch,
    row: usize,
    audio_col: usize,
    label_col: usize,
    index: i64,
) -> Result<Sample, ArrowError> {
    // Audio column is a struct with "bytes" field
    // HuggingFace stores audio as struct{bytes: binary, path: string, ...}
    let audio_col = batch.column(audio_col);

    // The audio column is a StructArray containing a "bytes" field
    let audio_struct = audio_col
        .as_any()
        .downcast_ref::<arrow::array::StructArray>()
        .ok_or_else(|| ArrowError::InvalidColumnType("audio (expected struct)".to_string()))?;

    let bytes_idx = audio_struct
        .column_by_name("bytes")
        .ok_or_else(|| ArrowError::MissingColumn("audio.bytes".to_string()))?;

    let bytes_array = bytes_idx
        .as_any()
        .downcast_ref::<BinaryArray>()
        .ok_or_else(|| {
            ArrowError::InvalidColumnType("audio.bytes (expected binary)".to_string())
        })?;

    let audio_bytes = bytes_array.value(row).to_vec();

    // Label column - must be Int64 (ClassLabel feature type in HuggingFace datasets)
    let label_col = batch.column(label_col);
    let label = label_col
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| {
            ArrowError::InvalidColumnType(
                "ebird_code (expected int64, got string - dataset must use ClassLabel feature)".to_string(),
            )
        })?
        .value(row);

    Ok(Sample {
        audio_bytes,
        label,
        index,
    })
}
