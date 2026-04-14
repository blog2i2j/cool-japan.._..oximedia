//! APNG (Animated Portable Network Graphics) support.
//!
//! APNG extends the PNG format with animation chunks (`acTL`, `fcTL`, `fdAT`)
//! as specified in the APNG spec (<https://wiki.mozilla.org/APNG_Spec>).
//!
//! # Format Overview
//!
//! An APNG file is a valid PNG file with extra animation chunks:
//!
//! - **`acTL`** — Animation Control: total frame count + loop count.
//! - **`fcTL`** — Frame Control: per-frame dimensions, timing, disposal.
//! - **`fdAT`** — Frame Data: compressed pixel data (same as IDAT but with sequence #).
//!
//! The first frame is optionally embedded as regular `IDAT` data so that
//! non-APNG-aware decoders show a static image.
//!
//! # This Implementation
//!
//! - **`ApngEncoder`** — builds a minimal, conformant APNG byte stream from
//!   a list of RGBA frames.
//! - **`ApngDecoder`** — detects APNG chunks and extracts frame metadata.
//!
//! # Example
//!
//! ```rust
//! use oximedia_codec::png::apng::{ApngEncoder, ApngFrame, ApngConfig, DisposeOp, BlendOp};
//!
//! let config = ApngConfig {
//!     width: 4,
//!     height: 4,
//!     loop_count: 0, // loop forever
//! };
//!
//! let frame = ApngFrame {
//!     rgba: vec![128u8; 4 * 4 * 4],
//!     delay_num: 1,
//!     delay_den: 10, // 100 ms
//!     dispose_op: DisposeOp::None,
//!     blend_op: BlendOp::Source,
//! };
//!
//! let encoder = ApngEncoder::new(config);
//! let apng_data = encoder.encode(&[frame]).expect("encode failed");
//! assert!(apng_data.starts_with(b"\x89PNG"));
//! ```

#![forbid(unsafe_code)]
#![allow(clippy::cast_possible_truncation)]

use crate::error::{CodecError, CodecResult};
use oxiarc_deflate::ZlibStreamEncoder;
use std::io::Write;

// =============================================================================
// CRC-32 (ISO 3309)
// =============================================================================

/// Compute CRC-32 for a byte slice (PNG uses ISO 3309 polynomial).
fn crc32(data: &[u8]) -> u32 {
    // Reflected poly for ISO 3309.
    const POLY: u32 = 0xEDB8_8320;
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        let mut b = u32::from(byte);
        for _ in 0..8 {
            if (crc ^ b) & 1 != 0 {
                crc = (crc >> 1) ^ POLY;
            } else {
                crc >>= 1;
            }
            b >>= 1;
        }
    }
    !crc
}

// =============================================================================
// Public types
// =============================================================================

/// Frame disposal operation (how the canvas is cleared before next frame).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DisposeOp {
    /// Do not clear (next frame composites on top).
    #[default]
    None = 0,
    /// Clear to fully transparent black.
    Background = 1,
    /// Restore to previous canvas.
    Previous = 2,
}

/// Blending operation (how the new frame is composited onto the canvas).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BlendOp {
    /// Overwrite pixels (alpha not composited — fastest).
    #[default]
    Source = 0,
    /// Alpha-composite the new frame over the existing canvas.
    Over = 1,
}

/// One frame of an APNG animation.
#[derive(Clone, Debug)]
pub struct ApngFrame {
    /// Raw RGBA pixel data: `width × height × 4` bytes.
    pub rgba: Vec<u8>,
    /// Numerator of the frame delay fraction.
    pub delay_num: u16,
    /// Denominator of the frame delay fraction (0 → treated as 100).
    pub delay_den: u16,
    /// Disposal operation.
    pub dispose_op: DisposeOp,
    /// Blend operation.
    pub blend_op: BlendOp,
}

/// APNG encoder configuration.
#[derive(Clone, Debug)]
pub struct ApngConfig {
    /// Canvas width in pixels.
    pub width: u32,
    /// Canvas height in pixels.
    pub height: u32,
    /// Number of times to loop the animation (0 = infinite).
    pub loop_count: u32,
}

// =============================================================================
// Encoder
// =============================================================================

/// APNG encoder.
///
/// Produces a binary APNG bitstream from a sequence of RGBA frames.
pub struct ApngEncoder {
    config: ApngConfig,
}

impl ApngEncoder {
    /// Create a new encoder with the given configuration.
    #[must_use]
    pub fn new(config: ApngConfig) -> Self {
        Self { config }
    }

    /// Encode `frames` into an APNG byte stream.
    ///
    /// # Errors
    ///
    /// Returns `CodecError` if any frame's pixel buffer has the wrong size
    /// or if DEFLATE compression fails.
    pub fn encode(&self, frames: &[ApngFrame]) -> CodecResult<Vec<u8>> {
        if frames.is_empty() {
            return Err(CodecError::InvalidParameter(
                "APNG requires at least one frame".to_string(),
            ));
        }

        let w = self.config.width;
        let h = self.config.height;
        let expected_len = (w as usize) * (h as usize) * 4;

        for (i, frame) in frames.iter().enumerate() {
            if frame.rgba.len() != expected_len {
                return Err(CodecError::InvalidParameter(format!(
                    "frame {i}: expected {expected_len} bytes, got {}",
                    frame.rgba.len()
                )));
            }
        }

        let mut out: Vec<u8> = Vec::new();

        // PNG signature
        out.extend_from_slice(b"\x89PNG\r\n\x1a\n");

        // IHDR (width, height, bit_depth=8, color_type=6=RGBA, compress=0, filter=0, interlace=0)
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&w.to_be_bytes());
        ihdr.extend_from_slice(&h.to_be_bytes());
        ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);
        self.write_chunk(&mut out, b"IHDR", &ihdr);

        // acTL — animation control
        let mut actl = Vec::new();
        actl.extend_from_slice(&(frames.len() as u32).to_be_bytes());
        actl.extend_from_slice(&self.config.loop_count.to_be_bytes());
        self.write_chunk(&mut out, b"acTL", &actl);

        let mut seq_num: u32 = 0;

        for (frame_idx, frame) in frames.iter().enumerate() {
            // fcTL — frame control
            let mut fctl: Vec<u8> = Vec::new();
            fctl.extend_from_slice(&seq_num.to_be_bytes());
            seq_num += 1;
            fctl.extend_from_slice(&w.to_be_bytes());
            fctl.extend_from_slice(&h.to_be_bytes());
            fctl.extend_from_slice(&0u32.to_be_bytes()); // x_offset
            fctl.extend_from_slice(&0u32.to_be_bytes()); // y_offset
            fctl.extend_from_slice(&frame.delay_num.to_be_bytes());
            fctl.extend_from_slice(&frame.delay_den.to_be_bytes());
            fctl.push(frame.dispose_op as u8);
            fctl.push(frame.blend_op as u8);
            self.write_chunk(&mut out, b"fcTL", &fctl);

            // Compress pixel data
            let raw = self.filter_rgba(&frame.rgba, w as usize, h as usize)?;

            if frame_idx == 0 {
                // First frame: IDAT (for backwards compat with non-APNG decoders)
                self.write_chunk(&mut out, b"IDAT", &raw);
            } else {
                // Subsequent frames: fdAT with sequence number prefix
                let mut fdat: Vec<u8> = Vec::new();
                fdat.extend_from_slice(&seq_num.to_be_bytes());
                seq_num += 1;
                fdat.extend_from_slice(&raw);
                self.write_chunk(&mut out, b"fdAT", &fdat);
            }
        }

        // IEND
        self.write_chunk(&mut out, b"IEND", &[]);

        Ok(out)
    }

    /// Apply PNG filter (Sub) + DEFLATE compress an RGBA frame.
    fn filter_rgba(&self, rgba: &[u8], width: usize, height: usize) -> CodecResult<Vec<u8>> {
        let row_bytes = width * 4;
        let mut filtered: Vec<u8> = Vec::with_capacity((row_bytes + 1) * height);

        for row in 0..height {
            filtered.push(1); // Sub filter type
            let base = row * row_bytes;
            for col in 0..row_bytes {
                let pixel = rgba[base + col];
                let prev = if col >= 4 { rgba[base + col - 4] } else { 0 };
                filtered.push(pixel.wrapping_sub(prev));
            }
        }

        let mut enc = ZlibStreamEncoder::new(Vec::new(), 6);
        enc.write_all(&filtered).map_err(|e| CodecError::Io(e))?;
        enc.finish().map_err(|e| CodecError::Io(e))
    }

    /// Serialise one PNG chunk: `length ++ type ++ data ++ crc`.
    fn write_chunk(&self, out: &mut Vec<u8>, chunk_type: &[u8; 4], data: &[u8]) {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        out.extend_from_slice(chunk_type);
        out.extend_from_slice(data);
        let mut crc_buf = Vec::with_capacity(4 + data.len());
        crc_buf.extend_from_slice(chunk_type);
        crc_buf.extend_from_slice(data);
        out.extend_from_slice(&crc32(&crc_buf).to_be_bytes());
    }
}

// =============================================================================
// Decoder (metadata extraction)
// =============================================================================

/// Metadata extracted from an APNG file.
#[derive(Clone, Debug)]
pub struct ApngInfo {
    /// Canvas width.
    pub width: u32,
    /// Canvas height.
    pub height: u32,
    /// Total number of animation frames.
    pub frame_count: u32,
    /// Loop count (0 = infinite).
    pub loop_count: u32,
    /// Per-frame control records.
    pub frames: Vec<FrameInfo>,
}

/// Per-frame metadata extracted from `fcTL` chunks.
#[derive(Clone, Debug)]
pub struct FrameInfo {
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// X offset on canvas.
    pub x_offset: u32,
    /// Y offset on canvas.
    pub y_offset: u32,
    /// Delay numerator.
    pub delay_num: u16,
    /// Delay denominator.
    pub delay_den: u16,
    /// Disposal op byte.
    pub dispose_op: u8,
    /// Blend op byte.
    pub blend_op: u8,
}

impl FrameInfo {
    /// Frame delay in seconds.
    #[must_use]
    pub fn delay_secs(&self) -> f64 {
        let den = if self.delay_den == 0 {
            100
        } else {
            u32::from(self.delay_den)
        };
        f64::from(self.delay_num) / f64::from(den)
    }
}

/// Minimal APNG metadata decoder.
///
/// Parses chunk headers from a raw PNG/APNG byte stream and returns
/// animation metadata. Does not decompress pixel data.
#[derive(Debug, Default)]
pub struct ApngDecoder;

impl ApngDecoder {
    /// Create a new decoder.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Parse `data` and return APNG metadata.
    ///
    /// # Errors
    ///
    /// Returns `CodecError` if the PNG signature is missing, or if chunk
    /// data is truncated.
    pub fn parse(&self, data: &[u8]) -> CodecResult<ApngInfo> {
        if data.len() < 8 || &data[..8] != b"\x89PNG\r\n\x1a\n" {
            return Err(CodecError::InvalidBitstream(
                "Not a PNG file (bad signature)".to_string(),
            ));
        }

        let mut pos = 8usize;
        let mut width = 0u32;
        let mut height = 0u32;
        let mut frame_count = 0u32;
        let mut loop_count = 0u32;
        let mut frames = Vec::new();

        while pos + 8 <= data.len() {
            let chunk_len =
                u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                    as usize;
            let chunk_type = &data[pos + 4..pos + 8];
            let chunk_data_start = pos + 8;
            let chunk_data_end = chunk_data_start + chunk_len;

            if chunk_data_end + 4 > data.len() {
                return Err(CodecError::InvalidBitstream(
                    "Truncated PNG chunk".to_string(),
                ));
            }

            let chunk_data = &data[chunk_data_start..chunk_data_end];

            match chunk_type {
                b"IHDR" if chunk_data.len() >= 8 => {
                    width = u32::from_be_bytes([
                        chunk_data[0],
                        chunk_data[1],
                        chunk_data[2],
                        chunk_data[3],
                    ]);
                    height = u32::from_be_bytes([
                        chunk_data[4],
                        chunk_data[5],
                        chunk_data[6],
                        chunk_data[7],
                    ]);
                }
                b"acTL" if chunk_data.len() >= 8 => {
                    frame_count = u32::from_be_bytes([
                        chunk_data[0],
                        chunk_data[1],
                        chunk_data[2],
                        chunk_data[3],
                    ]);
                    loop_count = u32::from_be_bytes([
                        chunk_data[4],
                        chunk_data[5],
                        chunk_data[6],
                        chunk_data[7],
                    ]);
                }
                b"fcTL" if chunk_data.len() >= 26 => {
                    // seq(4) + width(4) + height(4) + x(4) + y(4) + d_num(2) + d_den(2) + disp(1) + blend(1) = 26
                    let fw = u32::from_be_bytes([
                        chunk_data[4],
                        chunk_data[5],
                        chunk_data[6],
                        chunk_data[7],
                    ]);
                    let fh = u32::from_be_bytes([
                        chunk_data[8],
                        chunk_data[9],
                        chunk_data[10],
                        chunk_data[11],
                    ]);
                    let fx = u32::from_be_bytes([
                        chunk_data[12],
                        chunk_data[13],
                        chunk_data[14],
                        chunk_data[15],
                    ]);
                    let fy = u32::from_be_bytes([
                        chunk_data[16],
                        chunk_data[17],
                        chunk_data[18],
                        chunk_data[19],
                    ]);
                    let dn = u16::from_be_bytes([chunk_data[20], chunk_data[21]]);
                    let dd = u16::from_be_bytes([chunk_data[22], chunk_data[23]]);
                    let dispose = chunk_data[24];
                    let blend = chunk_data[25];
                    frames.push(FrameInfo {
                        width: fw,
                        height: fh,
                        x_offset: fx,
                        y_offset: fy,
                        delay_num: dn,
                        delay_den: dd,
                        dispose_op: dispose,
                        blend_op: blend,
                    });
                }
                b"IEND" => break,
                _ => {}
            }

            pos = chunk_data_end + 4; // skip CRC
        }

        Ok(ApngInfo {
            width,
            height,
            frame_count,
            loop_count,
            frames,
        })
    }

    /// Check whether `data` is an APNG (contains `acTL` chunk).
    pub fn is_apng(data: &[u8]) -> bool {
        if data.len() < 8 || &data[..8] != b"\x89PNG\r\n\x1a\n" {
            return false;
        }
        let mut pos = 8usize;
        while pos + 8 <= data.len() {
            let chunk_len =
                u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                    as usize;
            let chunk_type = &data[pos + 4..pos + 8];
            if chunk_type == b"acTL" {
                return true;
            }
            if chunk_type == b"IEND" {
                break;
            }
            pos = pos + 8 + chunk_len + 4;
        }
        false
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(w: u32, h: u32, fill: u8) -> ApngFrame {
        ApngFrame {
            rgba: vec![fill; (w * h * 4) as usize],
            delay_num: 1,
            delay_den: 10,
            dispose_op: DisposeOp::None,
            blend_op: BlendOp::Source,
        }
    }

    fn make_config(w: u32, h: u32) -> ApngConfig {
        ApngConfig {
            width: w,
            height: h,
            loop_count: 0,
        }
    }

    // --- Encoder tests ---

    #[test]
    fn test_apng_single_frame_produces_png_signature() {
        let enc = ApngEncoder::new(make_config(4, 4));
        let frame = make_frame(4, 4, 200);
        let data = enc.encode(&[frame]).expect("encode failed");
        assert!(
            data.starts_with(b"\x89PNG\r\n\x1a\n"),
            "Must start with PNG signature"
        );
    }

    #[test]
    fn test_apng_multiple_frames() {
        let enc = ApngEncoder::new(make_config(8, 8));
        let frames: Vec<_> = (0..3)
            .map(|i| make_frame(8, 8, i as u8 * 40 + 50))
            .collect();
        let data = enc.encode(&frames).expect("encode failed");
        assert!(!data.is_empty());
        // Should contain acTL chunk
        assert!(
            data.windows(4).any(|w| w == b"acTL"),
            "APNG must contain acTL chunk"
        );
    }

    #[test]
    fn test_apng_encode_empty_frames_errors() {
        let enc = ApngEncoder::new(make_config(4, 4));
        let result = enc.encode(&[]);
        assert!(result.is_err(), "Empty frame list should return error");
    }

    #[test]
    fn test_apng_wrong_frame_size_errors() {
        let enc = ApngEncoder::new(make_config(4, 4));
        let bad_frame = ApngFrame {
            rgba: vec![0u8; 100], // wrong size for 4×4
            delay_num: 1,
            delay_den: 10,
            dispose_op: DisposeOp::None,
            blend_op: BlendOp::Source,
        };
        let result = enc.encode(&[bad_frame]);
        assert!(result.is_err(), "Wrong frame size should error");
    }

    #[test]
    fn test_apng_contains_fctl_chunk() {
        let enc = ApngEncoder::new(make_config(4, 4));
        let frames = vec![make_frame(4, 4, 100), make_frame(4, 4, 200)];
        let data = enc.encode(&frames).expect("encode failed");
        let fctl_count = data.windows(4).filter(|w| *w == b"fcTL").count();
        assert_eq!(fctl_count, 2, "Should have one fcTL per frame");
    }

    #[test]
    fn test_apng_first_frame_idat() {
        let enc = ApngEncoder::new(make_config(4, 4));
        let frames = vec![make_frame(4, 4, 128)];
        let data = enc.encode(&frames).expect("encode failed");
        assert!(
            data.windows(4).any(|w| w == b"IDAT"),
            "First frame must be in IDAT"
        );
    }

    #[test]
    fn test_apng_second_frame_fdat() {
        let enc = ApngEncoder::new(make_config(4, 4));
        let frames = vec![make_frame(4, 4, 128), make_frame(4, 4, 64)];
        let data = enc.encode(&frames).expect("encode failed");
        assert!(
            data.windows(4).any(|w| w == b"fdAT"),
            "Second frame must be in fdAT"
        );
    }

    // --- Decoder tests ---

    #[test]
    fn test_apng_decoder_roundtrip_metadata() {
        let enc = ApngEncoder::new(make_config(16, 12));
        let frames: Vec<_> = (0..4).map(|_| make_frame(16, 12, 100)).collect();
        let data = enc.encode(&frames).expect("encode failed");

        let dec = ApngDecoder::new();
        let info = dec.parse(&data).expect("parse failed");

        assert_eq!(info.width, 16);
        assert_eq!(info.height, 12);
        assert_eq!(info.frame_count, 4);
        assert_eq!(info.loop_count, 0);
        assert_eq!(info.frames.len(), 4);
    }

    #[test]
    fn test_apng_decoder_frame_timing() {
        let enc = ApngEncoder::new(make_config(4, 4));
        let frame = ApngFrame {
            rgba: vec![0u8; 64],
            delay_num: 1,
            delay_den: 25, // 40 ms
            dispose_op: DisposeOp::None,
            blend_op: BlendOp::Source,
        };
        let data = enc.encode(&[frame]).expect("encode failed");
        let dec = ApngDecoder::new();
        let info = dec.parse(&data).expect("parse failed");
        assert_eq!(info.frames.len(), 1);
        let delay = info.frames[0].delay_secs();
        assert!((delay - 0.04).abs() < 1e-6, "Expected 40 ms, got {delay}s");
    }

    #[test]
    fn test_apng_is_apng_true() {
        let enc = ApngEncoder::new(make_config(4, 4));
        let data = enc.encode(&[make_frame(4, 4, 0)]).expect("encode failed");
        assert!(ApngDecoder::is_apng(&data));
    }

    #[test]
    fn test_apng_decoder_bad_signature() {
        let dec = ApngDecoder::new();
        let result = dec.parse(b"not a png file");
        assert!(result.is_err());
    }

    #[test]
    fn test_apng_loop_count() {
        let config = ApngConfig {
            width: 4,
            height: 4,
            loop_count: 3,
        };
        let enc = ApngEncoder::new(config);
        let data = enc.encode(&[make_frame(4, 4, 50)]).expect("encode");
        let dec = ApngDecoder::new();
        let info = dec.parse(&data).expect("parse");
        assert_eq!(info.loop_count, 3);
    }

    #[test]
    fn test_dispose_op_values() {
        assert_eq!(DisposeOp::None as u8, 0);
        assert_eq!(DisposeOp::Background as u8, 1);
        assert_eq!(DisposeOp::Previous as u8, 2);
    }

    #[test]
    fn test_blend_op_values() {
        assert_eq!(BlendOp::Source as u8, 0);
        assert_eq!(BlendOp::Over as u8, 1);
    }

    #[test]
    fn test_frame_info_delay_zero_den() {
        let fi = FrameInfo {
            width: 4,
            height: 4,
            x_offset: 0,
            y_offset: 0,
            delay_num: 1,
            delay_den: 0, // 0 den → treat as 100
            dispose_op: 0,
            blend_op: 0,
        };
        let delay = fi.delay_secs();
        assert!((delay - 0.01).abs() < 1e-9, "Expected 10ms, got {delay}s");
    }
}
