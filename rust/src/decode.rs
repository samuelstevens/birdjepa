//! Audio decoding using symphonia.
//!
//! Supports MP3, FLAC, OGG, WAV formats.

use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction};
use std::io::Cursor;
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DecodeError {
    #[error("Symphonia error: {0}")]
    Symphonia(#[from] symphonia::core::errors::Error),
    #[error("No audio track found")]
    NoAudioTrack,
    #[error("Unsupported codec")]
    UnsupportedCodec,
}

/// Decoded audio samples.
pub struct DecodedAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: usize,
}

impl DecodedAudio {
    /// Convert to mono by averaging channels.
    pub fn to_mono(&self) -> Vec<f32> {
        if self.channels == 1 {
            return self.samples.clone();
        }

        let n_frames = self.samples.len() / self.channels;
        let mut mono = Vec::with_capacity(n_frames);

        for frame in 0..n_frames {
            let mut sum = 0.0;
            for ch in 0..self.channels {
                sum += self.samples[frame * self.channels + ch];
            }
            mono.push(sum / self.channels as f32);
        }

        mono
    }

    /// Resample to target sample rate using sinc interpolation (rubato).
    /// Uses high-quality anti-aliasing filter to prevent aliasing artifacts.
    pub fn resample(&self, target_rate: u32) -> Vec<f32> {
        if self.sample_rate == target_rate {
            return self.to_mono();
        }

        let mono = self.to_mono();
        if mono.is_empty() {
            return Vec::new();
        }

        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };

        let resample_ratio = target_rate as f64 / self.sample_rate as f64;
        let chunk_size = 1024;

        // max_resample_ratio_relative must cover actual ratio (e.g., 8kHz→32kHz = 4x)
        let max_ratio = resample_ratio.max(1.0 / resample_ratio) * 1.1;

        let mut resampler = SincFixedIn::<f32>::new(
            resample_ratio,
            max_ratio,
            params,
            chunk_size,
            1, // mono
        )
        .expect("Failed to create resampler");

        let mut output = Vec::new();
        let mut pos = 0;

        // Process in chunks
        while pos < mono.len() {
            let end = (pos + chunk_size).min(mono.len());
            let chunk = &mono[pos..end];

            // Pad last chunk if needed
            let input = if chunk.len() < chunk_size {
                let mut padded = chunk.to_vec();
                padded.resize(chunk_size, 0.0);
                vec![padded]
            } else {
                vec![chunk.to_vec()]
            };

            let resampled = resampler.process(&input, None).expect("Resampling failed");
            output.extend_from_slice(&resampled[0]);

            pos += chunk_size;
        }

        // Trim output to expected length
        let expected_len = (mono.len() as f64 * resample_ratio).round() as usize;
        output.truncate(expected_len);

        output
    }
}

/// Decode audio from bytes.
pub fn decode_audio(bytes: &[u8]) -> Result<DecodedAudio, DecodeError> {
    let cursor = Cursor::new(bytes.to_vec());
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

    let hint = Hint::new();
    let format_opts = FormatOptions::default();
    let metadata_opts = MetadataOptions::default();

    let probed =
        symphonia::default::get_probe().format(&hint, mss, &format_opts, &metadata_opts)?;
    let mut format = probed.format;

    // Find the first audio track
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or(DecodeError::NoAudioTrack)?;

    let decoder_opts = DecoderOptions::default();
    let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &decoder_opts)?;

    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or(DecodeError::UnsupportedCodec)?;
    let channels = track.codec_params.channels.map(|c| c.count()).unwrap_or(1);

    let track_id = track.id;
    let mut samples = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => return Err(e.into()),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = decoder.decode(&packet)?;
        append_samples(&decoded, &mut samples, channels)?;
    }

    if samples.is_empty() {
        return Err(DecodeError::NoAudioTrack);
    }

    Ok(DecodedAudio {
        samples,
        sample_rate,
        channels,
    })
}

fn append_samples(
    buffer: &AudioBufferRef,
    output: &mut Vec<f32>,
    channels: usize,
) -> Result<(), DecodeError> {
    match buffer {
        AudioBufferRef::F32(buf) => {
            for frame in 0..buf.frames() {
                for ch in 0..channels {
                    output.push(buf.chan(ch)[frame]);
                }
            }
        }
        AudioBufferRef::S16(buf) => {
            for frame in 0..buf.frames() {
                for ch in 0..channels {
                    output.push(buf.chan(ch)[frame] as f32 / 32768.0);
                }
            }
        }
        AudioBufferRef::S32(buf) => {
            for frame in 0..buf.frames() {
                for ch in 0..channels {
                    output.push(buf.chan(ch)[frame] as f32 / 2147483648.0);
                }
            }
        }
        AudioBufferRef::U8(buf) => {
            for frame in 0..buf.frames() {
                for ch in 0..channels {
                    output.push((buf.chan(ch)[frame] as f32 - 128.0) / 128.0);
                }
            }
        }
        _ => {
            return Err(DecodeError::UnsupportedCodec);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resample() {
        let audio = DecodedAudio {
            samples: (0..44100).map(|i| (i as f32 / 44100.0).sin()).collect(),
            sample_rate: 44100,
            channels: 1,
        };

        let resampled = audio.resample(22050);
        assert_eq!(resampled.len(), 22050);
    }
}
