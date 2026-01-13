//! Benchmarks for audio decode and spectrogram computation.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use std::f32::consts::PI;
use std::fs::File;
use std::io::BufReader;

// Import from our crate (lib name is "_rs" per Cargo.toml)
use _rs::decode::decode_audio;
use _rs::spectrogram::SpectrogramTransform;

/// Generate synthetic waveform (5 seconds at 32kHz).
fn make_waveform() -> Vec<f32> {
    let sample_rate = 32000;
    let duration_sec = 5;
    let n_samples = sample_rate * duration_sec;
    (0..n_samples)
        .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
        .collect()
}

/// Load real OGG Vorbis bytes from XenoCanto dataset.
fn load_ogg_bytes() -> Vec<u8> {
    // Try to load from Arrow file
    use arrow::array::{Array, BinaryArray};
    use arrow::ipc::reader::StreamReader;

    let pattern = "/fs/scratch/PAS2136/samuelstevens/cache/huggingface/datasets/samuelstevens___bird_set/XCL/0.0.0/*/bird_set-train-00000-*.arrow";
    let paths: Vec<_> = glob::glob(pattern)
        .expect("glob pattern")
        .filter_map(|p| p.ok())
        .collect();

    let path = paths.first().expect("no Arrow files found for benchmark");
    let file = File::open(path).expect("open Arrow file");
    let reader = StreamReader::try_new(BufReader::new(file), None).expect("read Arrow");

    for batch in reader {
        let batch = batch.expect("read batch");
        // Find audio.bytes column
        let audio_col = batch.column_by_name("audio").expect("audio column");
        let struct_arr = audio_col
            .as_any()
            .downcast_ref::<arrow::array::StructArray>()
            .expect("audio is struct");
        let bytes_col = struct_arr.column_by_name("bytes").expect("bytes column");
        let bytes_arr = bytes_col
            .as_any()
            .downcast_ref::<BinaryArray>()
            .expect("bytes is binary");

        // Return first non-empty sample
        for i in 0..bytes_arr.len() {
            let bytes = bytes_arr.value(i);
            if bytes.len() > 10000 {
                // Skip very short clips
                return bytes.to_vec();
            }
        }
    }
    panic!("no audio samples found");
}

fn bench_spectrogram(c: &mut Criterion) {
    let waveform = make_waveform();
    let transform = SpectrogramTransform::new(32000, 1024, 320, 128, 0.0, 16000.0);

    c.bench_function("spectrogram_5s", |b| {
        b.iter(|| transform.transform(black_box(&waveform)))
    });
}

fn bench_decode_ogg(c: &mut Criterion) {
    let ogg_bytes = load_ogg_bytes();
    println!("Loaded OGG: {} bytes", ogg_bytes.len());

    c.bench_function("decode_ogg", |b| {
        b.iter(|| decode_audio(black_box(&ogg_bytes)))
    });
}

fn bench_decode_and_resample_ogg(c: &mut Criterion) {
    let ogg_bytes = load_ogg_bytes();

    c.bench_function("decode_resample_ogg", |b| {
        b.iter(|| {
            let audio = decode_audio(black_box(&ogg_bytes)).unwrap();
            audio.resample(32000)
        })
    });
}

fn bench_full_pipeline_ogg(c: &mut Criterion) {
    let ogg_bytes = load_ogg_bytes();
    let transform = SpectrogramTransform::new(32000, 1024, 320, 128, 0.0, 16000.0);

    c.bench_function("full_pipeline_ogg", |b| {
        b.iter(|| {
            let audio = decode_audio(black_box(&ogg_bytes)).unwrap();
            let waveform = audio.resample(32000);
            transform.transform(&waveform)
        })
    });
}

criterion_group!(
    benches,
    bench_spectrogram,
    bench_decode_ogg,
    bench_decode_and_resample_ogg,
    bench_full_pipeline_ogg
);
criterion_main!(benches);
