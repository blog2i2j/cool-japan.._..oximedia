//! Multi-bit-depth pixel format support for 360° VR image processing.
//!
//! This module provides a unified interface for reading and writing pixels
//! at 8-bit, 16-bit, and 32-bit floating-point precision.  All values are
//! normalised to and from the `[0, 1]` floating-point range so that
//! higher-level algorithms remain format-agnostic.
//!
//! ## Usage
//!
//! ```rust
//! use oximedia_360::pixel_format::{PixelFormat, sample_pixel, write_pixel};
//!
//! // A tiny 2×1 RGB image in u8
//! let data = vec![0u8, 128, 255,  64, 32, 16];
//! let pixel = sample_pixel(&data, 2, 0, 0, 3, PixelFormat::U8);
//! assert_eq!(pixel.len(), 3);
//! ```

// ─── PixelFormat enum ─────────────────────────────────────────────────────────

/// Supported pixel component formats for image buffers.
///
/// Each variant describes how individual channel values are encoded in the
/// raw byte slice.  Pixel data is always stored in native-endian byte order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    /// 8-bit unsigned integer per channel (1 byte per component).
    /// Values in `[0, 255]` are normalised to `[0.0, 1.0]`.
    U8,
    /// 16-bit unsigned integer per channel (2 bytes per component, native-endian).
    /// Values in `[0, 65535]` are normalised to `[0.0, 1.0]`.
    U16,
    /// 32-bit IEEE 754 float per channel (4 bytes per component, native-endian).
    /// Values are read/written without normalisation (assumed already in `[0, 1]`).
    F32,
}

impl PixelFormat {
    /// Number of bytes per channel component for this format.
    #[inline]
    pub fn bytes_per_component(self) -> usize {
        match self {
            PixelFormat::U8 => 1,
            PixelFormat::U16 => 2,
            PixelFormat::F32 => 4,
        }
    }
}

// ─── sample_pixel ─────────────────────────────────────────────────────────────

/// Read a single pixel from a packed row-major image buffer, normalised to `[0, 1]`.
///
/// * `data`     — raw image bytes (packed row-major, channels interleaved)
/// * `width`    — image width in pixels
/// * `x`, `y`  — zero-based pixel coordinates
/// * `channels` — number of colour channels per pixel (e.g. 3 for RGB)
/// * `format`   — pixel component format
///
/// Returns a `Vec<f32>` of length `channels`.  If the requested pixel lies
/// outside the buffer (e.g. due to wrong dimensions), returns zeros.
pub fn sample_pixel(
    data: &[u8],
    width: u32,
    x: u32,
    y: u32,
    channels: u32,
    format: PixelFormat,
) -> Vec<f32> {
    let ch = channels as usize;
    let bpc = format.bytes_per_component();
    let pixel_bytes = ch * bpc;

    // Guard: return zeros for out-of-bounds pixel coordinates
    // (Also protects against buffer being too small for the declared dimensions)
    if x >= width || (width > 0 && y as usize >= data.len() / (width as usize * pixel_bytes).max(1))
    {
        return vec![0.0f32; ch];
    }
    let byte_offset = (y as usize * width as usize + x as usize) * pixel_bytes;
    if byte_offset + pixel_bytes > data.len() {
        return vec![0.0f32; ch];
    }

    let slice = &data[byte_offset..byte_offset + pixel_bytes];
    let mut result = Vec::with_capacity(ch);

    match format {
        PixelFormat::U8 => {
            for i in 0..ch {
                result.push(slice[i] as f32 / 255.0);
            }
        }
        PixelFormat::U16 => {
            for i in 0..ch {
                let lo = slice[i * 2];
                let hi = slice[i * 2 + 1];
                let val = u16::from_ne_bytes([lo, hi]);
                result.push(val as f32 / 65535.0);
            }
        }
        PixelFormat::F32 => {
            for i in 0..ch {
                let bytes = [
                    slice[i * 4],
                    slice[i * 4 + 1],
                    slice[i * 4 + 2],
                    slice[i * 4 + 3],
                ];
                result.push(f32::from_ne_bytes(bytes));
            }
        }
    }

    result
}

// ─── write_pixel ──────────────────────────────────────────────────────────────

/// Write a single pixel into a packed row-major image buffer from normalised `[0, 1]` values.
///
/// * `data`     — raw image bytes (packed row-major, channels interleaved); mutated in-place
/// * `width`    — image width in pixels
/// * `x`, `y`  — zero-based pixel coordinates
/// * `channels` — number of colour channels per pixel
/// * `format`   — pixel component format
/// * `values`   — normalised `[0, 1]` float values to write; length must be ≥ `channels`
///
/// Values are clamped to `[0, 1]` before encoding into the destination format.
/// Out-of-bounds coordinates are silently ignored.
pub fn write_pixel(
    data: &mut [u8],
    width: u32,
    x: u32,
    y: u32,
    channels: u32,
    format: PixelFormat,
    values: &[f32],
) {
    let ch = channels as usize;
    if values.len() < ch {
        return;
    }
    let bpc = format.bytes_per_component();
    let pixel_bytes = ch * bpc;
    let byte_offset = (y as usize * width as usize + x as usize) * pixel_bytes;

    if byte_offset + pixel_bytes > data.len() {
        return;
    }

    let slice = &mut data[byte_offset..byte_offset + pixel_bytes];

    match format {
        PixelFormat::U8 => {
            for i in 0..ch {
                let v = values[i].clamp(0.0, 1.0);
                slice[i] = (v * 255.0).round() as u8;
            }
        }
        PixelFormat::U16 => {
            for i in 0..ch {
                let v = values[i].clamp(0.0, 1.0);
                let encoded = (v * 65535.0).round() as u16;
                let bytes = encoded.to_ne_bytes();
                slice[i * 2] = bytes[0];
                slice[i * 2 + 1] = bytes[1];
            }
        }
        PixelFormat::F32 => {
            for i in 0..ch {
                let bytes = values[i].to_ne_bytes();
                slice[i * 4] = bytes[0];
                slice[i * 4 + 1] = bytes[1];
                slice[i * 4 + 2] = bytes[2];
                slice[i * 4 + 3] = bytes[3];
            }
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── PixelFormat::bytes_per_component ────────────────────────────────────

    #[test]
    fn u8_bytes_per_component() {
        assert_eq!(PixelFormat::U8.bytes_per_component(), 1);
    }

    #[test]
    fn u16_bytes_per_component() {
        assert_eq!(PixelFormat::U16.bytes_per_component(), 2);
    }

    #[test]
    fn f32_bytes_per_component() {
        assert_eq!(PixelFormat::F32.bytes_per_component(), 4);
    }

    // ── sample_pixel u8 ─────────────────────────────────────────────────────

    #[test]
    fn sample_pixel_u8_black() {
        let data = vec![0u8; 4 * 4 * 3];
        let pixel = sample_pixel(&data, 4, 2, 2, 3, PixelFormat::U8);
        assert_eq!(pixel.len(), 3);
        assert!((pixel[0]).abs() < 1e-6);
        assert!((pixel[1]).abs() < 1e-6);
        assert!((pixel[2]).abs() < 1e-6);
    }

    #[test]
    fn sample_pixel_u8_white() {
        let data = vec![255u8; 2 * 2 * 3];
        let pixel = sample_pixel(&data, 2, 1, 1, 3, PixelFormat::U8);
        assert_eq!(pixel.len(), 3);
        for &v in &pixel {
            assert!((v - 1.0).abs() < 1e-6, "expected 1.0, got {v}");
        }
    }

    #[test]
    fn sample_pixel_u8_half_value() {
        // channel value 128 should normalise to ~0.502
        let data = vec![128u8, 0, 0];
        let pixel = sample_pixel(&data, 1, 0, 0, 3, PixelFormat::U8);
        assert!((pixel[0] - 128.0 / 255.0).abs() < 1e-5);
        assert!((pixel[1]).abs() < 1e-6);
    }

    #[test]
    fn sample_pixel_u8_out_of_bounds_returns_zeros() {
        let data = vec![255u8; 4 * 4 * 3];
        // x=10 is clearly out of bounds for width=4
        let pixel = sample_pixel(&data, 4, 10, 0, 3, PixelFormat::U8);
        assert_eq!(pixel.len(), 3);
        for &v in &pixel {
            assert!((v).abs() < 1e-6);
        }
    }

    // ── sample_pixel u16 ────────────────────────────────────────────────────

    #[test]
    fn sample_pixel_u16_max_normalises_to_one() {
        // u16::MAX in native-endian bytes
        let val: u16 = 65535;
        let bytes = val.to_ne_bytes();
        // A 1×1 single-channel image
        let data = vec![bytes[0], bytes[1]];
        let pixel = sample_pixel(&data, 1, 0, 0, 1, PixelFormat::U16);
        assert!(
            (pixel[0] - 1.0).abs() < 1e-5,
            "expected 1.0, got {}",
            pixel[0]
        );
    }

    #[test]
    fn sample_pixel_u16_zero_is_zero() {
        let data = vec![0u8, 0];
        let pixel = sample_pixel(&data, 1, 0, 0, 1, PixelFormat::U16);
        assert!((pixel[0]).abs() < 1e-9);
    }

    #[test]
    fn sample_pixel_u16_mid_value() {
        let val: u16 = 32768;
        let bytes = val.to_ne_bytes();
        let data = vec![bytes[0], bytes[1]];
        let pixel = sample_pixel(&data, 1, 0, 0, 1, PixelFormat::U16);
        let expected = 32768.0f32 / 65535.0;
        assert!((pixel[0] - expected).abs() < 1e-5);
    }

    // ── sample_pixel f32 ────────────────────────────────────────────────────

    #[test]
    fn sample_pixel_f32_roundtrip() {
        let value = 0.7372f32;
        let bytes = value.to_ne_bytes();
        let data = bytes.to_vec();
        let pixel = sample_pixel(&data, 1, 0, 0, 1, PixelFormat::F32);
        assert!(
            (pixel[0] - value).abs() < 1e-7,
            "expected {value}, got {}",
            pixel[0]
        );
    }

    #[test]
    fn sample_pixel_f32_multi_channel() {
        let r = 0.1f32;
        let g = 0.5f32;
        let b = 0.9f32;
        let mut data = Vec::new();
        data.extend_from_slice(&r.to_ne_bytes());
        data.extend_from_slice(&g.to_ne_bytes());
        data.extend_from_slice(&b.to_ne_bytes());
        let pixel = sample_pixel(&data, 1, 0, 0, 3, PixelFormat::F32);
        assert!((pixel[0] - r).abs() < 1e-7);
        assert!((pixel[1] - g).abs() < 1e-7);
        assert!((pixel[2] - b).abs() < 1e-7);
    }

    // ── write_pixel ─────────────────────────────────────────────────────────

    #[test]
    fn write_pixel_u8_and_read_back() {
        let mut data = vec![0u8; 3];
        write_pixel(&mut data, 1, 0, 0, 3, PixelFormat::U8, &[1.0, 0.5, 0.0]);
        assert_eq!(data[0], 255);
        // 0.5 * 255 = 127.5, rounds to 128
        assert_eq!(data[1], 128);
        assert_eq!(data[2], 0);
    }

    #[test]
    fn write_pixel_u16_and_read_back() {
        let mut data = vec![0u8; 2];
        write_pixel(&mut data, 1, 0, 0, 1, PixelFormat::U16, &[1.0]);
        let pixel = sample_pixel(&data, 1, 0, 0, 1, PixelFormat::U16);
        assert!((pixel[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn write_pixel_f32_and_read_back() {
        let value = 0.42f32;
        let mut data = vec![0u8; 4];
        write_pixel(&mut data, 1, 0, 0, 1, PixelFormat::F32, &[value]);
        let pixel = sample_pixel(&data, 1, 0, 0, 1, PixelFormat::F32);
        assert!((pixel[0] - value).abs() < 1e-7);
    }

    #[test]
    fn write_pixel_clamping() {
        let mut data = vec![0u8; 3];
        // Values outside [0,1] should be clamped
        write_pixel(&mut data, 1, 0, 0, 3, PixelFormat::U8, &[2.0, -1.0, 0.5]);
        assert_eq!(data[0], 255); // clamped to 1.0 → 255
        assert_eq!(data[1], 0); // clamped to 0.0 → 0
        assert_eq!(data[2], 128); // 0.5 → 128
    }

    #[test]
    fn write_pixel_out_of_bounds_is_safe() {
        let mut data = vec![42u8; 3];
        // Should not panic or modify anything
        write_pixel(&mut data, 1, 5, 0, 3, PixelFormat::U8, &[1.0, 1.0, 1.0]);
        assert_eq!(data, vec![42u8; 3]);
    }

    // ── sample_pixel / write_pixel roundtrip ────────────────────────────────

    #[test]
    fn roundtrip_u8_pixel_row() {
        let mut data = vec![0u8; 4 * 3]; // 4 RGB pixels
        for x in 0..4u32 {
            let r = x as f32 / 3.0;
            let g = 1.0 - r;
            let b = 0.5;
            write_pixel(&mut data, 4, x, 0, 3, PixelFormat::U8, &[r, g, b]);
        }
        for x in 0..4u32 {
            let px = sample_pixel(&data, 4, x, 0, 3, PixelFormat::U8);
            let expected_r = x as f32 / 3.0;
            // Roundtrip through u8 introduces ~1/255 ≈ 0.004 quantisation error
            assert!((px[0] - expected_r).abs() < 0.005, "x={x} R mismatch");
        }
    }
}
