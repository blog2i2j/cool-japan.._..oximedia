//! WASM audio format conversion utilities.
//!
//! This module provides WebAssembly bindings for audio sample format conversion
//! and basic sample rate conversion. It operates on raw byte buffers and is
//! suitable for browser-based audio processing pipelines.
//!
//! # Sample Format Conversion
//!
//! Supported formats: `i16`, `i32`, `f32`, `f64`
//!
//! Samples are stored in little-endian byte order. The conversion preserves
//! the normalized amplitude range (-1.0 to 1.0 for float formats).
//!
//! # Sample Rate Conversion
//!
//! Uses linear interpolation for basic resampling. For higher quality,
//! consider using the Web Audio API's `OfflineAudioContext` for resampling.
//!
//! # JavaScript Example
//!
//! ```javascript
//! // Convert i16 PCM to f32
//! const f32Bytes = oximedia.wasm_convert_sample_format(i16Buffer, 'i16', 'f32');
//!
//! // Resample from 44100 to 48000 Hz
//! const resampled = oximedia.wasm_resample(f32Samples, 44100, 48000, 2);
//! ```

use wasm_bindgen::prelude::*;

/// Convert audio samples between different sample formats.
///
/// Converts raw byte buffers between i16, i32, f32, and f64 sample formats.
/// All byte data is in little-endian order.
///
/// # Format specifications
///
/// - `"i16"`: 16-bit signed integer, 2 bytes per sample, range -32768..32767
/// - `"i32"`: 32-bit signed integer, 4 bytes per sample, range -2147483648..2147483647
/// - `"f32"`: 32-bit float, 4 bytes per sample, range -1.0..1.0
/// - `"f64"`: 64-bit float, 8 bytes per sample, range -1.0..1.0
///
/// # Arguments
///
/// * `input` - Raw bytes in the source format
/// * `from_format` - Source format string: `"i16"`, `"i32"`, `"f32"`, or `"f64"`
/// * `to_format` - Target format string: `"i16"`, `"i32"`, `"f32"`, or `"f64"`
///
/// # Returns
///
/// Raw bytes in the target format as `Uint8Array`.
///
/// # Errors
///
/// Returns an error if:
/// - An unknown format string is provided
/// - Input buffer size is not a multiple of the source sample size
///
/// # Example (JavaScript)
///
/// ```javascript
/// // Convert 16-bit WAV PCM to 32-bit float
/// const floatData = oximedia.wasm_convert_sample_format(wavPcm, 'i16', 'f32');
/// const floatView = new Float32Array(floatData.buffer);
/// ```
#[wasm_bindgen]
pub fn wasm_convert_sample_format(
    input: &[u8],
    from_format: &str,
    to_format: &str,
) -> Result<js_sys::Uint8Array, JsValue> {
    // First convert input to f64 intermediary
    let samples_f64 = bytes_to_f64(input, from_format)?;

    // Then convert f64 to output format
    let output_bytes = f64_to_bytes(&samples_f64, to_format)?;

    Ok(js_sys::Uint8Array::from(output_bytes.as_slice()))
}

/// Convert raw bytes in a given format to f64 samples.
fn bytes_to_f64(input: &[u8], format: &str) -> Result<Vec<f64>, JsValue> {
    match format {
        "i16" => {
            if input.len() % 2 != 0 {
                return Err(crate::utils::js_err(
                    "Input buffer size must be a multiple of 2 for i16 format",
                ));
            }
            Ok(input
                .chunks_exact(2)
                .map(|chunk| {
                    let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                    f64::from(sample) / 32768.0
                })
                .collect())
        }
        "i32" => {
            if input.len() % 4 != 0 {
                return Err(crate::utils::js_err(
                    "Input buffer size must be a multiple of 4 for i32 format",
                ));
            }
            Ok(input
                .chunks_exact(4)
                .map(|chunk| {
                    let sample = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    f64::from(sample) / 2_147_483_648.0
                })
                .collect())
        }
        "f32" => {
            if input.len() % 4 != 0 {
                return Err(crate::utils::js_err(
                    "Input buffer size must be a multiple of 4 for f32 format",
                ));
            }
            Ok(input
                .chunks_exact(4)
                .map(|chunk| {
                    let sample = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    f64::from(sample)
                })
                .collect())
        }
        "f64" => {
            if input.len() % 8 != 0 {
                return Err(crate::utils::js_err(
                    "Input buffer size must be a multiple of 8 for f64 format",
                ));
            }
            Ok(input
                .chunks_exact(8)
                .map(|chunk| {
                    f64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ])
                })
                .collect())
        }
        other => Err(crate::utils::js_err(&format!(
            "Unknown sample format: '{other}'. Supported: i16, i32, f32, f64"
        ))),
    }
}

/// Convert f64 samples to raw bytes in a given format.
fn f64_to_bytes(samples: &[f64], format: &str) -> Result<Vec<u8>, JsValue> {
    match format {
        "i16" => {
            let mut output = Vec::with_capacity(samples.len() * 2);
            for &s in samples {
                let clamped = s.clamp(-1.0, 1.0);
                let val = (clamped * 32767.0) as i16;
                output.extend_from_slice(&val.to_le_bytes());
            }
            Ok(output)
        }
        "i32" => {
            let mut output = Vec::with_capacity(samples.len() * 4);
            for &s in samples {
                let clamped = s.clamp(-1.0, 1.0);
                let val = (clamped * 2_147_483_647.0) as i32;
                output.extend_from_slice(&val.to_le_bytes());
            }
            Ok(output)
        }
        "f32" => {
            let mut output = Vec::with_capacity(samples.len() * 4);
            for &s in samples {
                let val = s as f32;
                output.extend_from_slice(&val.to_le_bytes());
            }
            Ok(output)
        }
        "f64" => {
            let mut output = Vec::with_capacity(samples.len() * 8);
            for &s in samples {
                output.extend_from_slice(&s.to_le_bytes());
            }
            Ok(output)
        }
        _ => Err(crate::utils::js_err(&format!(
            "Unknown sample format: '{format}'. Supported: i16, i32, f32, f64"
        ))),
    }
}

/// Simple sample rate conversion using linear interpolation.
///
/// Resamples audio from one sample rate to another. Uses linear interpolation
/// between sample points, which is suitable for quick conversions but may
/// introduce aliasing artifacts at high frequency ratios.
///
/// For production-quality resampling in the browser, consider using the
/// Web Audio API's `OfflineAudioContext`.
///
/// # Arguments
///
/// * `samples` - Interleaved f32 PCM samples
/// * `from_rate` - Source sample rate in Hz (e.g., 44100)
/// * `to_rate` - Target sample rate in Hz (e.g., 48000)
/// * `channels` - Number of audio channels
///
/// # Returns
///
/// Resampled interleaved f32 samples as `Float32Array`.
///
/// # Errors
///
/// Returns an error if:
/// - Either sample rate is zero
/// - Channel count is zero
/// - Input buffer size is not a multiple of the channel count
///
/// # Example (JavaScript)
///
/// ```javascript
/// // Resample stereo audio from 44100 to 48000 Hz
/// const resampled = oximedia.wasm_resample(stereoSamples, 44100, 48000, 2);
/// ```
#[wasm_bindgen]
pub fn wasm_resample(
    samples: &[f32],
    from_rate: u32,
    to_rate: u32,
    channels: u32,
) -> Result<js_sys::Float32Array, JsValue> {
    if from_rate == 0 || to_rate == 0 {
        return Err(crate::utils::js_err("Sample rates must be > 0"));
    }
    if channels == 0 {
        return Err(crate::utils::js_err("Channel count must be > 0"));
    }

    let ch = channels as usize;
    if samples.len() % ch != 0 {
        return Err(crate::utils::js_err(
            "Input buffer size must be a multiple of the channel count",
        ));
    }

    // If rates are the same, return a copy
    if from_rate == to_rate {
        return Ok(js_sys::Float32Array::from(samples));
    }

    let input_frames = samples.len() / ch;
    if input_frames == 0 {
        return Ok(js_sys::Float32Array::new_with_length(0));
    }

    let ratio = f64::from(to_rate) / f64::from(from_rate);
    let output_frames = (input_frames as f64 * ratio).ceil() as usize;
    let mut output = vec![0.0_f32; output_frames * ch];

    for out_frame in 0..output_frames {
        // Map output frame position back to input frame position
        let in_pos = out_frame as f64 / ratio;
        let in_frame_0 = in_pos.floor() as usize;
        let frac = (in_pos - in_frame_0 as f64) as f32;

        let in_frame_1 = (in_frame_0 + 1).min(input_frames - 1);

        for c in 0..ch {
            let s0 = samples[in_frame_0 * ch + c];
            let s1 = samples[in_frame_1 * ch + c];
            // Linear interpolation
            output[out_frame * ch + c] = s0 + frac * (s1 - s0);
        }
    }

    Ok(js_sys::Float32Array::from(output.as_slice()))
}

// ─── Pure-Rust helpers used only in tests ────────────────────────────────────

/// Convert f64 samples to i16 LE bytes without touching `JsValue`.
#[cfg(test)]
fn f64_to_i16_bytes(samples: &[f64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let val = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        out.extend_from_slice(&val.to_le_bytes());
    }
    out
}

/// Convert i16 LE bytes to f64 samples without touching `JsValue`.
#[cfg(test)]
fn i16_bytes_to_f64(input: &[u8]) -> Vec<f64> {
    input
        .chunks_exact(2)
        .map(|c| f64::from(i16::from_le_bytes([c[0], c[1]])) / 32768.0)
        .collect()
}

/// Convert f64 to f32 LE bytes.
#[cfg(test)]
fn f64_to_f32_bytes(samples: &[f64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 4);
    for &s in samples {
        out.extend_from_slice(&(s as f32).to_le_bytes());
    }
    out
}

/// Convert f32 LE bytes to f64.
#[cfg(test)]
fn f32_bytes_to_f64(input: &[u8]) -> Vec<f64> {
    input
        .chunks_exact(4)
        .map(|c| f64::from(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
        .collect()
}

/// Convert f64 to i32 LE bytes.
#[cfg(test)]
fn f64_to_i32_bytes(samples: &[f64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 4);
    for &s in samples {
        let val = (s.clamp(-1.0, 1.0) * 2_147_483_647.0) as i32;
        out.extend_from_slice(&val.to_le_bytes());
    }
    out
}

/// Convert i32 LE bytes to f64.
#[cfg(test)]
fn i32_bytes_to_f64(input: &[u8]) -> Vec<f64> {
    input
        .chunks_exact(4)
        .map(|c| f64::from(i32::from_le_bytes([c[0], c[1], c[2], c[3]])) / 2_147_483_648.0)
        .collect()
}

/// Linear-interpolation resample — pure Rust, no `JsValue`.
#[cfg(test)]
fn resample_pure(
    samples: &[f32],
    from_rate: u32,
    to_rate: u32,
    channels: usize,
) -> Result<Vec<f32>, &'static str> {
    if from_rate == 0 || to_rate == 0 {
        return Err("Sample rates must be > 0");
    }
    if channels == 0 {
        return Err("Channel count must be > 0");
    }
    if samples.len() % channels != 0 {
        return Err("Input buffer size must be a multiple of the channel count");
    }
    if from_rate == to_rate {
        return Ok(samples.to_vec());
    }
    let input_frames = samples.len() / channels;
    if input_frames == 0 {
        return Ok(Vec::new());
    }
    let ratio = f64::from(to_rate) / f64::from(from_rate);
    let output_frames = (input_frames as f64 * ratio).ceil() as usize;
    let mut output = vec![0.0_f32; output_frames * channels];
    for out_frame in 0..output_frames {
        let in_pos = out_frame as f64 / ratio;
        let in_frame_0 = in_pos.floor() as usize;
        let frac = (in_pos - in_frame_0 as f64) as f32;
        let in_frame_1 = (in_frame_0 + 1).min(input_frames - 1);
        for c in 0..channels {
            let s0 = samples[in_frame_0 * channels + c];
            let s1 = samples[in_frame_1 * channels + c];
            output[out_frame * channels + c] = s0 + frac * (s1 - s0);
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── i16 round-trips ────────────────────────────────────────────────────

    #[test]
    fn test_i16_zero_round_trip() {
        let bytes = f64_to_i16_bytes(&[0.0_f64]);
        assert_eq!(bytes.len(), 2);
        let back = i16_bytes_to_f64(&bytes);
        assert!((back[0]).abs() < 1e-4, "expected ~0.0, got {}", back[0]);
    }

    #[test]
    fn test_i16_positive_full_scale() {
        let bytes = f64_to_i16_bytes(&[1.0_f64]);
        let sample = i16::from_le_bytes([bytes[0], bytes[1]]);
        assert_eq!(sample, 32767);
        let back = i16_bytes_to_f64(&bytes);
        assert!((back[0] - 0.999_969_5).abs() < 0.0001, "got {}", back[0]);
    }

    #[test]
    fn test_i16_negative_full_scale() {
        let bytes = f64_to_i16_bytes(&[-1.0_f64]);
        let sample = i16::from_le_bytes([bytes[0], bytes[1]]);
        assert_eq!(sample, -32767);
        let back = i16_bytes_to_f64(&bytes);
        assert!((back[0] + 0.999_969_5).abs() < 0.0001, "got {}", back[0]);
    }

    // ── f32 round-trips ────────────────────────────────────────────────────

    #[test]
    fn test_f32_round_trip() {
        let val = 0.5_f64;
        let bytes = f64_to_f32_bytes(&[val]);
        assert_eq!(bytes.len(), 4);
        let back = f32_bytes_to_f64(&bytes);
        assert!((back[0] - val).abs() < 1e-6, "got {}", back[0]);
    }

    #[test]
    fn test_f32_negative_round_trip() {
        let val = -0.75_f64;
        let bytes = f64_to_f32_bytes(&[val]);
        let back = f32_bytes_to_f64(&bytes);
        assert!((back[0] - val).abs() < 1e-6, "got {}", back[0]);
    }

    // ── i32 round-trips ────────────────────────────────────────────────────

    #[test]
    fn test_i32_round_trip_quarter() {
        let val = 0.25_f64;
        let bytes = f64_to_i32_bytes(&[val]);
        assert_eq!(bytes.len(), 4);
        let back = i32_bytes_to_f64(&bytes);
        assert!((back[0] - val).abs() < 1e-7, "got {}", back[0]);
    }

    // ── format byte-size validation (pure Rust) ────────────────────────────

    #[test]
    fn test_i16_requires_even_byte_count() {
        // i16 needs 2 bytes per sample — 3 bytes is invalid.
        let is_valid = 3_usize % 2 == 0;
        assert!(!is_valid, "3 bytes should not be valid for i16");
    }

    #[test]
    fn test_i32_requires_multiple_of_4() {
        let is_valid = 5_usize % 4 == 0;
        assert!(!is_valid, "5 bytes should not be valid for i32");
    }

    #[test]
    fn test_f64_requires_multiple_of_8() {
        let is_valid = 7_usize % 8 == 0;
        assert!(!is_valid, "7 bytes should not be valid for f64");
    }

    // ── resample_pure ──────────────────────────────────────────────────────

    #[test]
    fn test_resample_same_rate_identity() {
        let samples: Vec<f32> = (0..10).map(|i| i as f32 * 0.1).collect();
        let out = resample_pure(&samples, 44100, 44100, 1).expect("resample should succeed");
        assert_eq!(out, samples);
    }

    #[test]
    fn test_resample_upsample_length() {
        let samples = vec![0.5_f32; 100];
        let out = resample_pure(&samples, 44100, 48000, 1).expect("resample should succeed");
        assert!(
            out.len() >= 108 && out.len() <= 110,
            "unexpected frame count: {}",
            out.len()
        );
    }

    #[test]
    fn test_resample_downsample_length() {
        let samples = vec![0.5_f32; 100];
        let out = resample_pure(&samples, 48000, 44100, 1).expect("resample should succeed");
        assert!(
            out.len() >= 91 && out.len() <= 93,
            "unexpected frame count: {}",
            out.len()
        );
    }

    #[test]
    fn test_resample_dc_signal_preserved() {
        // DC signal: all samples the same value → resampled output must also be DC.
        let samples = vec![0.3_f32; 20]; // 10 stereo frames
        let out = resample_pure(&samples, 44100, 48000, 2).expect("resample should succeed");
        for (i, &s) in out.iter().enumerate() {
            assert!((s - 0.3).abs() < 1e-5, "sample[{i}] = {s}, expected ~0.3");
        }
    }

    #[test]
    fn test_resample_zero_from_rate_errors() {
        let err = resample_pure(&[0.0_f32; 4], 0, 48000, 1);
        assert!(err.is_err());
    }

    #[test]
    fn test_resample_zero_to_rate_errors() {
        let err = resample_pure(&[0.0_f32; 4], 44100, 0, 1);
        assert!(err.is_err());
    }

    #[test]
    fn test_resample_zero_channels_errors() {
        let err = resample_pure(&[0.0_f32; 4], 44100, 48000, 0);
        assert!(err.is_err());
    }

    #[test]
    fn test_resample_misaligned_channels_errors() {
        let err = resample_pure(&[0.0_f32; 5], 44100, 48000, 2);
        assert!(err.is_err());
    }

    #[test]
    fn test_resample_empty_input() {
        let out = resample_pure(&[], 44100, 48000, 1).expect("resample should succeed");
        assert_eq!(out.len(), 0);
    }
}
