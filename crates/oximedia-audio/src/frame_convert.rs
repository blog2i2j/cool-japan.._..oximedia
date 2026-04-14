//! `AudioFrame` format conversion utilities.
//!
//! This module provides zero-copy-friendly conversions between interleaved and planar
//! layouts, as well as bit-depth conversions for `AudioFrame` buffers.
//!
//! # Interleaved vs Planar
//!
//! * **Interleaved** — samples for all channels are interleaved: `L R L R L R …`
//! * **Planar**      — each channel occupies its own contiguous buffer: `[L L L …] [R R R …]`
//!
//! Most DSP algorithms prefer planar layout; most network/codec APIs prefer interleaved.
//!
//! # Bit-depth conversion
//!
//! `DepthConverter` converts between `S16`, `S32`, `F32`, and `F64` sample formats using
//! correct normalisation (e.g. S16 → F32 divides by 32 768; F32 → S16 multiplies and clamps).

#![allow(dead_code)]

use bytes::Bytes;
use oximedia_core::SampleFormat;

use crate::frame::{AudioBuffer, AudioFrame};

// ---------------------------------------------------------------------------
// Layout conversion
// ---------------------------------------------------------------------------

/// Convert an `AudioFrame` from interleaved to planar layout.
///
/// If the frame is already planar the function clones it unchanged.
///
/// # Errors
///
/// Returns `None` if the frame has an unrecognised bytes-per-sample value.
#[must_use]
pub fn to_planar(frame: &AudioFrame) -> Option<AudioFrame> {
    let ch = frame.channels.count();
    if ch == 0 {
        return None;
    }
    let bps = frame.format.bytes_per_sample();
    if bps == 0 {
        return None;
    }

    match &frame.samples {
        AudioBuffer::Planar(_) => Some(frame.clone()),
        AudioBuffer::Interleaved(data) => {
            let n_frames = data.len() / (ch * bps);
            let mut planes: Vec<Vec<u8>> = vec![Vec::with_capacity(n_frames * bps); ch];

            for frame_idx in 0..n_frames {
                for ch_idx in 0..ch {
                    let src = (frame_idx * ch + ch_idx) * bps;
                    planes[ch_idx].extend_from_slice(&data[src..src + bps]);
                }
            }

            let planes_bytes: Vec<Bytes> = planes.into_iter().map(Bytes::from).collect();
            let mut out = AudioFrame::new(frame.format, frame.sample_rate, frame.channels.clone());
            out.samples = AudioBuffer::Planar(planes_bytes);
            out.timestamp = frame.timestamp;
            Some(out)
        }
    }
}

/// Convert an `AudioFrame` from planar to interleaved layout.
///
/// If the frame is already interleaved the function clones it unchanged.
///
/// # Errors
///
/// Returns `None` if the frame has an unrecognised bytes-per-sample value or
/// the planes have inconsistent sizes.
#[must_use]
pub fn to_interleaved(frame: &AudioFrame) -> Option<AudioFrame> {
    let ch = frame.channels.count();
    if ch == 0 {
        return None;
    }
    let bps = frame.format.bytes_per_sample();
    if bps == 0 {
        return None;
    }

    match &frame.samples {
        AudioBuffer::Interleaved(_) => Some(frame.clone()),
        AudioBuffer::Planar(planes) => {
            if planes.is_empty() {
                let mut out =
                    AudioFrame::new(frame.format, frame.sample_rate, frame.channels.clone());
                out.samples = AudioBuffer::Interleaved(Bytes::new());
                out.timestamp = frame.timestamp;
                return Some(out);
            }

            let n_frames = planes[0].len() / bps;
            let mut interleaved = Vec::with_capacity(n_frames * ch * bps);

            for frame_idx in 0..n_frames {
                for ch_idx in 0..ch {
                    if ch_idx >= planes.len() {
                        return None;
                    }
                    let src = frame_idx * bps;
                    if src + bps > planes[ch_idx].len() {
                        return None;
                    }
                    interleaved.extend_from_slice(&planes[ch_idx][src..src + bps]);
                }
            }

            let mut out = AudioFrame::new(frame.format, frame.sample_rate, frame.channels.clone());
            out.samples = AudioBuffer::Interleaved(Bytes::from(interleaved));
            out.timestamp = frame.timestamp;
            Some(out)
        }
    }
}

// ---------------------------------------------------------------------------
// Bit-depth conversion helpers
// ---------------------------------------------------------------------------

/// Convert a raw S16 little-endian byte slice to f32 samples in `[-1, 1]`.
#[must_use]
pub fn s16le_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|c| {
            let v = i16::from_le_bytes([c[0], c[1]]);
            v as f32 / 32_768.0
        })
        .collect()
}

/// Convert f32 samples in `[-1, 1]` to S16 little-endian bytes.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn f32_to_s16le(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32_767.0) as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Convert a raw S32 little-endian byte slice to f32 samples in `[-1, 1]`.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn s32le_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| {
            let v = i32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            v as f32 / 2_147_483_648.0
        })
        .collect()
}

/// Convert f32 samples in `[-1, 1]` to S32 little-endian bytes.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn f32_to_s32le(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 4);
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 2_147_483_647.0) as i32;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Convert a raw F32 little-endian byte slice to `Vec<f32>`.
#[must_use]
pub fn f32le_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Convert f32 samples to F64 little-endian bytes.
#[must_use]
pub fn f32_to_f64le(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 8);
    for &s in samples {
        out.extend_from_slice(&f64::from(s).to_le_bytes());
    }
    out
}

/// Convert a raw F64 little-endian byte slice to f32 samples.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn f64le_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(8)
        .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
        .collect()
}

// ---------------------------------------------------------------------------
// High-level DepthConverter
// ---------------------------------------------------------------------------

/// Convert an `AudioFrame` to a different sample bit depth.
///
/// The conversion round-trips through f32. Supported target formats are:
/// `S16`, `S32`, `F32`, and `F64` (both interleaved and planar variants).
///
/// Returns `None` when the source format cannot be decoded or the target
/// format is unsupported.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn convert_depth(frame: &AudioFrame, target: SampleFormat) -> Option<AudioFrame> {
    if frame.format == target {
        return Some(frame.clone());
    }

    // Extract as flat f32 regardless of layout
    let f32_samples = extract_as_f32(frame)?;

    // Re-encode in target format preserving layout
    let new_buf = match target {
        SampleFormat::S16 | SampleFormat::S16p => {
            let bytes = f32_to_s16le(&f32_samples);
            repack_bytes(bytes, frame, target)?
        }
        SampleFormat::S32 | SampleFormat::S32p => {
            let bytes = f32_to_s32le(&f32_samples);
            repack_bytes(bytes, frame, target)?
        }
        SampleFormat::F32 | SampleFormat::F32p => {
            let mut bytes = Vec::with_capacity(f32_samples.len() * 4);
            for &s in &f32_samples {
                bytes.extend_from_slice(&s.to_le_bytes());
            }
            repack_bytes(bytes, frame, target)?
        }
        SampleFormat::F64 | SampleFormat::F64p => {
            let bytes = f32_to_f64le(&f32_samples);
            repack_bytes(bytes, frame, target)?
        }
        _ => return None,
    };

    let mut out = AudioFrame::new(target, frame.sample_rate, frame.channels.clone());
    out.samples = new_buf;
    out.timestamp = frame.timestamp;
    Some(out)
}

/// Extract interleaved f32 samples from any `AudioFrame`, regardless of layout or depth.
#[must_use]
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
pub fn extract_as_f32(frame: &AudioFrame) -> Option<Vec<f32>> {
    let flat: Vec<u8> = match &frame.samples {
        AudioBuffer::Interleaved(data) => data.to_vec(),
        AudioBuffer::Planar(planes) => {
            // Interleave all planes
            let ch = planes.len();
            if ch == 0 {
                return Some(Vec::new());
            }
            let bps = frame.format.bytes_per_sample();
            if bps == 0 {
                return None;
            }
            let n = planes[0].len() / bps;
            let mut out = Vec::with_capacity(n * ch * bps);
            for i in 0..n {
                for p in planes.iter() {
                    let src = i * bps;
                    out.extend_from_slice(&p[src..src + bps]);
                }
            }
            out
        }
    };

    let samples = match frame.format {
        SampleFormat::S16 | SampleFormat::S16p => s16le_to_f32(&flat),
        SampleFormat::S32 | SampleFormat::S32p => s32le_to_f32(&flat),
        SampleFormat::F32 | SampleFormat::F32p => f32le_to_f32(&flat),
        SampleFormat::F64 | SampleFormat::F64p => f64le_to_f32(&flat),
        SampleFormat::U8 => flat.iter().map(|&b| (b as f32 - 128.0) / 128.0).collect(),
        _ => return None,
    };
    Some(samples)
}

/// Re-pack a flat interleaved byte buffer back into the same layout as the source frame.
fn repack_bytes(
    bytes: Vec<u8>,
    source: &AudioFrame,
    target_fmt: SampleFormat,
) -> Option<AudioBuffer> {
    let ch = source.channels.count();
    if ch == 0 {
        return None;
    }
    let new_bps = target_fmt.bytes_per_sample();
    if new_bps == 0 {
        return None;
    }

    if source.samples.is_planar() || target_fmt.is_planar() {
        // Convert the interleaved byte buffer to planar
        let n_frames = bytes.len() / (ch * new_bps);
        let mut planes: Vec<Vec<u8>> = vec![Vec::with_capacity(n_frames * new_bps); ch];
        for f_idx in 0..n_frames {
            for ch_idx in 0..ch {
                let src = (f_idx * ch + ch_idx) * new_bps;
                planes[ch_idx].extend_from_slice(&bytes[src..src + new_bps]);
            }
        }
        Some(AudioBuffer::Planar(
            planes.into_iter().map(Bytes::from).collect(),
        ))
    } else {
        Some(AudioBuffer::Interleaved(Bytes::from(bytes)))
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ChannelLayout;
    use oximedia_core::SampleFormat;

    fn make_interleaved_f32_frame(samples: &[f32], ch: usize) -> AudioFrame {
        let mut bytes = Vec::with_capacity(samples.len() * 4);
        for &s in samples {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        let mut frame = AudioFrame::new(SampleFormat::F32, 48_000, ChannelLayout::from_count(ch));
        frame.samples = AudioBuffer::Interleaved(Bytes::from(bytes));
        frame
    }

    #[test]
    fn test_to_planar_stereo() {
        // Stereo interleaved: L0 R0 L1 R1
        let samples = [0.1_f32, 0.2, 0.3, 0.4];
        let frame = make_interleaved_f32_frame(&samples, 2);
        let planar = to_planar(&frame).expect("should convert");

        if let AudioBuffer::Planar(planes) = &planar.samples {
            assert_eq!(planes.len(), 2);
            let l = f32le_to_f32(&planes[0]);
            let r = f32le_to_f32(&planes[1]);
            assert!((l[0] - 0.1).abs() < 1e-6, "L0 mismatch");
            assert!((l[1] - 0.3).abs() < 1e-6, "L1 mismatch");
            assert!((r[0] - 0.2).abs() < 1e-6, "R0 mismatch");
            assert!((r[1] - 0.4).abs() < 1e-6, "R1 mismatch");
        } else {
            panic!("expected planar buffer");
        }
    }

    #[test]
    fn test_to_interleaved_roundtrip() {
        let samples = [0.1_f32, 0.9, -0.1, -0.9, 0.5, -0.5];
        let interleaved = make_interleaved_f32_frame(&samples, 2);
        let planar = to_planar(&interleaved).expect("to planar");
        let back = to_interleaved(&planar).expect("to interleaved");

        let orig = extract_as_f32(&interleaved).expect("extract");
        let rt = extract_as_f32(&back).expect("extract rt");
        assert_eq!(orig.len(), rt.len());
        for (a, b) in orig.iter().zip(rt.iter()) {
            assert!((a - b).abs() < 1e-6, "roundtrip mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_to_planar_already_planar() {
        let samples = [0.5_f32, -0.5];
        let interleaved = make_interleaved_f32_frame(&samples, 1);
        let planar = to_planar(&interleaved).expect("ok");
        // Converting again should be a no-op
        let again = to_planar(&planar).expect("ok");
        assert!(matches!(again.samples, AudioBuffer::Planar(_)));
    }

    #[test]
    fn test_to_interleaved_already_interleaved() {
        let samples = [0.5_f32, -0.5];
        let frame = make_interleaved_f32_frame(&samples, 1);
        let out = to_interleaved(&frame).expect("ok");
        assert!(matches!(out.samples, AudioBuffer::Interleaved(_)));
    }

    #[test]
    fn test_convert_depth_f32_to_s16() {
        let samples = [0.0_f32, 1.0, -1.0, 0.5];
        let frame = make_interleaved_f32_frame(&samples, 1);
        let s16 = convert_depth(&frame, SampleFormat::S16).expect("convert");
        let back = extract_as_f32(&s16).expect("extract");
        assert_eq!(back.len(), 4);
        assert!(back[0].abs() < 1e-4, "zero maps to ~zero");
        assert!((back[1] - 1.0).abs() < 0.001, "1.0 maps to ~1.0");
        assert!((back[2] + 1.0).abs() < 0.001, "-1.0 maps to ~-1.0");
    }

    #[test]
    fn test_convert_depth_same_format_is_clone() {
        let samples = [0.3_f32, -0.3];
        let frame = make_interleaved_f32_frame(&samples, 1);
        let out = convert_depth(&frame, SampleFormat::F32).expect("convert");
        let orig = extract_as_f32(&frame).expect("orig");
        let got = extract_as_f32(&out).expect("got");
        assert_eq!(orig, got);
    }

    #[test]
    fn test_convert_depth_f32_to_s32() {
        let samples = [0.5_f32, -0.5];
        let frame = make_interleaved_f32_frame(&samples, 1);
        let s32 = convert_depth(&frame, SampleFormat::S32).expect("convert");
        let back = extract_as_f32(&s32).expect("extract");
        assert!((back[0] - 0.5).abs() < 0.001);
        assert!((back[1] + 0.5).abs() < 0.001);
    }

    #[test]
    fn test_convert_depth_f32_to_f64() {
        let samples = [0.123_f32, -0.456];
        let frame = make_interleaved_f32_frame(&samples, 1);
        let f64_frame = convert_depth(&frame, SampleFormat::F64).expect("convert");
        let back = extract_as_f32(&f64_frame).expect("extract");
        assert!((back[0] - 0.123).abs() < 1e-5);
        assert!((back[1] + 0.456).abs() < 1e-5);
    }

    #[test]
    fn test_s16le_f32_roundtrip() {
        let orig: Vec<f32> = vec![0.0, 0.5, -0.5, 1.0, -1.0];
        let encoded = f32_to_s16le(&orig);
        let decoded = s16le_to_f32(&encoded);
        for (a, b) in orig.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < 0.0002, "S16 roundtrip mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_s32le_f32_roundtrip() {
        let orig: Vec<f32> = vec![0.0, 0.5, -0.5, 0.9, -0.9];
        let encoded = f32_to_s32le(&orig);
        let decoded = s32le_to_f32(&encoded);
        for (a, b) in orig.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < 1e-6, "S32 roundtrip mismatch: {a} vs {b}");
        }
    }

    // ── Additional tests for AudioFrame format conversion (interleaved<->planar, depth) ──

    #[test]
    fn test_to_planar_mono() {
        let samples = [0.5_f32, -0.5, 0.25, -0.25];
        let frame = make_interleaved_f32_frame(&samples, 1);
        let planar = to_planar(&frame).expect("mono to planar");
        if let AudioBuffer::Planar(planes) = &planar.samples {
            assert_eq!(planes.len(), 1, "mono should have 1 plane");
            let decoded = f32le_to_f32(&planes[0]);
            assert_eq!(decoded.len(), samples.len());
            for (a, b) in samples.iter().zip(decoded.iter()) {
                assert!((a - b).abs() < 1e-6, "mono planar mismatch: {a} vs {b}");
            }
        } else {
            panic!("expected planar output");
        }
    }

    #[test]
    fn test_to_interleaved_preserves_metadata() {
        let samples = [0.1_f32, 0.2, 0.3, 0.4];
        let frame = make_interleaved_f32_frame(&samples, 2);
        let planar = to_planar(&frame).expect("to planar");
        let back = to_interleaved(&planar).expect("to interleaved");
        assert_eq!(back.sample_rate, frame.sample_rate);
        assert_eq!(back.format, frame.format);
    }

    #[test]
    fn test_extract_as_f32_from_interleaved_f32() {
        let samples = [0.5_f32, -0.5, 0.25];
        let frame = make_interleaved_f32_frame(&samples, 1);
        let extracted = extract_as_f32(&frame).expect("extract_as_f32");
        assert_eq!(extracted.len(), samples.len());
        for (a, b) in samples.iter().zip(extracted.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_extract_as_f32_from_planar() {
        let samples = [0.1_f32, 0.9, -0.1, -0.9];
        let frame = make_interleaved_f32_frame(&samples, 2);
        let planar_frame = to_planar(&frame).expect("to planar");
        let extracted = extract_as_f32(&planar_frame).expect("extract planar");
        let orig = extract_as_f32(&frame).expect("orig");
        assert_eq!(extracted.len(), orig.len());
        for (a, b) in orig.iter().zip(extracted.iter()) {
            assert!((a - b).abs() < 1e-6, "planar extract mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_convert_depth_f32_to_s16_stereo() {
        let samples = [0.5_f32, -0.5, 0.25, -0.25];
        let frame = make_interleaved_f32_frame(&samples, 2);
        let s16 = convert_depth(&frame, SampleFormat::S16).expect("convert stereo");
        let back = extract_as_f32(&s16).expect("extract stereo");
        assert_eq!(back.len(), samples.len());
        for (a, b) in samples.iter().zip(back.iter()) {
            assert!((a - b).abs() < 0.001, "stereo S16 mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_f32_to_f64le_roundtrip() {
        let orig: Vec<f32> = vec![0.0, 0.123, -0.456, 1.0, -1.0];
        let bytes = f32_to_f64le(&orig);
        assert_eq!(bytes.len(), orig.len() * 8);
        let decoded = f64le_to_f32(&bytes);
        for (a, b) in orig.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < 1e-7, "F64 roundtrip mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_convert_depth_s16_to_f32() {
        let samples_i16: Vec<i16> = vec![16384, -16384, 0, 32767];
        let mut bytes = Vec::with_capacity(samples_i16.len() * 2);
        for &s in &samples_i16 {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        let mut frame = AudioFrame::new(SampleFormat::S16, 48_000, ChannelLayout::from_count(1));
        frame.samples = AudioBuffer::Interleaved(Bytes::from(bytes));
        let f32_frame = convert_depth(&frame, SampleFormat::F32).expect("S16 to F32");
        let extracted = extract_as_f32(&f32_frame).expect("extract");
        assert_eq!(extracted.len(), samples_i16.len());
        assert!(
            (extracted[0] - 0.5).abs() < 0.001,
            "S16->F32 [0]: {}",
            extracted[0]
        );
        assert!(
            (extracted[1] + 0.5).abs() < 0.001,
            "S16->F32 [1]: {}",
            extracted[1]
        );
        assert!(
            extracted[2].abs() < 0.001,
            "zero maps to zero: {}",
            extracted[2]
        );
    }

    #[test]
    fn test_to_interleaved_from_empty_planar() {
        let mut frame = AudioFrame::new(SampleFormat::F32, 48_000, ChannelLayout::from_count(1));
        frame.samples = AudioBuffer::Planar(vec![Bytes::new()]);
        let out = to_interleaved(&frame);
        assert!(out.is_some(), "empty planar should convert without error");
    }

    #[test]
    fn test_interleaved_planar_roundtrip_3ch() {
        let n_frames = 8;
        let n_ch = 3;
        let mut samples = Vec::with_capacity(n_frames * n_ch);
        for f in 0..n_frames {
            for c in 0..n_ch {
                samples.push((f * n_ch + c) as f32 * 0.01);
            }
        }
        let frame = make_interleaved_f32_frame(&samples, n_ch);
        let planar = to_planar(&frame).expect("to_planar");
        let back = to_interleaved(&planar).expect("to_interleaved");
        let orig = extract_as_f32(&frame).expect("orig");
        let rt = extract_as_f32(&back).expect("rt");
        assert_eq!(orig.len(), rt.len());
        for (a, b) in orig.iter().zip(rt.iter()) {
            assert!((a - b).abs() < 1e-6, "3ch roundtrip: {a} vs {b}");
        }
    }

    #[test]
    fn test_convert_depth_f32_identity() {
        let samples = [0.3_f32, -0.7, 0.0, 1.0, -1.0];
        let frame = make_interleaved_f32_frame(&samples, 1);
        let out = convert_depth(&frame, SampleFormat::F32).expect("F32 identity");
        let extracted = extract_as_f32(&out).expect("extract");
        for (a, b) in samples.iter().zip(extracted.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "identity conversion mismatch: {a} vs {b}"
            );
        }
    }
}
