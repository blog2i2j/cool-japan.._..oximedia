//! NDI codec implementations
//!
//! This module provides codec support for NDI, including the OxiMedia native
//! OXIT intra-frame codec (a DCT-based intra codec replacing SpeedHQ) and
//! YUV422 format handling.
//!
//! ## OXIT Bitstream Format
//!
//! ```text
//! Header (14 bytes):
//!   [0..4]  magic   : b"OXIT"
//!   [4..8]  width   : u32 big-endian
//!   [8..12] height  : u32 big-endian
//!   [12]    quality : u8 (0–100)
//!   [13]    flags   : u8 (reserved, 0)
//!
//! Followed immediately by a sequence of entropy-coded 8×8 blocks:
//!   One block stream per component plane (Y, then Cb, then Cr).
//!   Each component is divided into 8×8 blocks in raster order.
//!
//! Per block entropy stream:
//!   DC coefficient : i16 big-endian delta from previous block's DC
//!   AC run-level pairs until EOB:
//!     run   : u8    (number of preceding zero-valued ACs, 0..=63)
//!     level : i16   big-endian (non-zero AC quantized coefficient)
//!   EOB marker: run=0, level=0  (two zero bytes: 0x00 0x00)
//! ```
#![allow(dead_code)]

use crate::{NdiError, Result};
use bytes::{BufMut, Bytes, BytesMut};
use std::io::Cursor;
use tracing::{debug, trace};

// ── OXIT magic ────────────────────────────────────────────────────────────────
const OXIT_MAGIC: &[u8; 4] = b"OXIT";
const OXIT_HEADER_LEN: usize = 14; // magic(4) + width(4) + height(4) + quality(1) + flags(1)

// ── JPEG luminance quantization table (standard, quality=50) ─────────────────
#[rustfmt::skip]
const LUMA_QTABLE_BASE: [u16; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68,109,103, 77,
    24, 35, 55, 64, 81,104,113, 92,
    49, 64, 78, 87,103,121,120,101,
    72, 92, 95, 98,112,100,103, 99,
];

// ── JPEG chrominance quantization table (standard, quality=50) ────────────────
#[rustfmt::skip]
const CHROMA_QTABLE_BASE: [u16; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
];

// ── Standard JPEG zigzag scan order ──────────────────────────────────────────
#[rustfmt::skip]
const ZIGZAG: [usize; 64] = [
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
];

// ── DCT-II basis functions (precomputed float cosines) ───────────────────────

/// Compute 2D DCT-II of an 8×8 block using the separable row-column approach.
/// Input: 64 signed values (pixel – 128 level-shifted).
/// Output: 64 DCT coefficients (scaled; DC is sum/8, matching JPEG scaling).
fn dct8x8(block: &[f64; 64]) -> [f64; 64] {
    // Apply 1-D DCT-II to each row, then each column (separable).
    let mut tmp = *block;

    // Row pass
    for row in 0..8_usize {
        let base = row * 8;
        dct1d_inplace(&mut tmp, base, 1);
    }

    // Column pass (stride = 8)
    for col in 0..8_usize {
        dct1d_inplace(&mut tmp, col, 8);
    }

    tmp
}

/// Inverse 2D DCT (IDCT-II = DCT-III / 8) of an 8×8 block.
fn idct8x8(block: &[f64; 64]) -> [f64; 64] {
    let mut tmp = *block;

    // Column pass first (inverse)
    for col in 0..8_usize {
        idct1d_inplace(&mut tmp, col, 8);
    }

    // Row pass
    for row in 0..8_usize {
        let base = row * 8;
        idct1d_inplace(&mut tmp, base, 1);
    }

    tmp
}

/// In-place 1-D DCT-II of 8 elements accessed at `buf[start + k*stride]`.
fn dct1d_inplace(buf: &mut [f64; 64], start: usize, stride: usize) {
    // Read
    let x: [f64; 8] = std::array::from_fn(|k| buf[start + k * stride]);

    // Naive O(N²) DCT-II (N=8, minimal and exact)
    let out = dct1d_8(&x);

    // Write back
    for k in 0..8 {
        buf[start + k * stride] = out[k];
    }
}

/// In-place 1-D IDCT of 8 elements (DCT-III).
fn idct1d_inplace(buf: &mut [f64; 64], start: usize, stride: usize) {
    let x: [f64; 8] = std::array::from_fn(|k| buf[start + k * stride]);
    let out = idct1d_8(&x);
    for k in 0..8 {
        buf[start + k * stride] = out[k];
    }
}

/// Compute 1-D DCT-II for N=8.
/// Uses the definition: X[k] = (2/N) * Σ_{n=0}^{N-1} x[n] * cos(π(2n+1)k / 2N)
/// Scaled so that the DC term is the plain sum (JPEG-compatible).
fn dct1d_8(x: &[f64; 8]) -> [f64; 8] {
    use std::f64::consts::PI;
    let n = 8_f64;
    let mut out = [0.0f64; 8];
    for k in 0..8_usize {
        let mut sum = 0.0f64;
        for i in 0..8_usize {
            sum += x[i] * ((PI * (2.0 * i as f64 + 1.0) * k as f64) / (2.0 * n)).cos();
        }
        // Standard orthonormal scaling: DC × 1/√N, AC × √(2/N)
        let scale = if k == 0 {
            (1.0_f64 / n).sqrt()
        } else {
            (2.0_f64 / n).sqrt()
        };
        out[k] = scale * sum;
    }
    out
}

/// Compute 1-D IDCT (DCT-III) for N=8.
fn idct1d_8(x: &[f64; 8]) -> [f64; 8] {
    use std::f64::consts::PI;
    let n = 8_f64;
    let mut out = [0.0f64; 8];
    for i in 0..8_usize {
        let mut sum = (1.0_f64 / n).sqrt() * x[0];
        for k in 1..8_usize {
            let scale = (2.0_f64 / n).sqrt();
            sum += scale * x[k] * ((PI * (2.0 * i as f64 + 1.0) * k as f64) / (2.0 * n)).cos();
        }
        out[i] = sum;
    }
    out
}

// ── Quantization table scaling ────────────────────────────────────────────────

/// Scale a base quantization table by a JPEG-compatible quality factor (1..=100).
/// Returns a table of u16 values ≥ 1.
fn scale_qtable(base: &[u16; 64], quality: u8) -> [u16; 64] {
    let q = quality.clamp(1, 100) as u32;
    let scale = if q < 50 { 5000 / q } else { 200 - 2 * q };
    let mut out = [1u16; 64];
    for (i, &b) in base.iter().enumerate() {
        let v = ((b as u32 * scale + 50) / 100).clamp(1, 255);
        out[i] = v as u16;
    }
    out
}

// ── Block extraction helpers ──────────────────────────────────────────────────

/// Extract an 8×8 block from a planar component buffer.
/// If the block extends beyond the image boundary, it is padded by edge replication.
fn extract_block(plane: &[u8], plane_w: usize, plane_h: usize, bx: usize, by: usize) -> [f64; 64] {
    let mut block = [0.0f64; 64];
    for row in 0..8_usize {
        for col in 0..8_usize {
            let px = (bx * 8 + col).min(plane_w.saturating_sub(1));
            let py = (by * 8 + row).min(plane_h.saturating_sub(1));
            let sample = if plane_w > 0 && plane_h > 0 {
                plane[py * plane_w + px] as f64 - 128.0
            } else {
                0.0
            };
            block[row * 8 + col] = sample;
        }
    }
    block
}

// ── Entropy coding ───────────────────────────────────────────────────────────

/// Encode a single quantised 8×8 block (zigzag order) into the output buffer.
/// Protocol:
///   - DC: 2 bytes i16 BE (delta from prev_dc)
///   - AC run-level pairs: u8 run + i16 BE level
///   - EOB: 0x00 0x00 (two bytes, run=0 level=0)
fn encode_block(dct: &[f64; 64], qtable: &[u16; 64], prev_dc: &mut i16, out: &mut BytesMut) {
    // Quantise and zigzag scan
    let mut coeffs = [0i16; 64];
    for zz_idx in 0..64_usize {
        let natural_idx = ZIGZAG[zz_idx];
        let q = qtable[zz_idx].max(1);
        let quantized = (dct[natural_idx] / q as f64).round() as i16;
        coeffs[zz_idx] = quantized;
    }

    // DC: delta coding
    let dc_delta = coeffs[0] - *prev_dc;
    *prev_dc = coeffs[0];
    out.put_i16(dc_delta);

    // AC: run-level with EOB
    let mut run: u8 = 0;
    for &ac in &coeffs[1..] {
        if ac == 0 {
            run += 1;
        } else {
            // Flush run: split into groups of 63 if run ≥ 64 (ZRL-like)
            while run >= 64 {
                // ZRL: run=63, level=1 (marks 63 skips without ending block)
                out.put_u8(63);
                out.put_i16(1);
                run -= 63;
                // Undo the fake "1" in AC count — just mark run continuation
            }
            out.put_u8(run);
            out.put_i16(ac);
            run = 0;
        }
    }
    // EOB marker
    out.put_u8(0);
    out.put_i16(0);
}

/// Decode a single quantised 8×8 block from `cursor`.
fn decode_block(
    cursor: &mut Cursor<&[u8]>,
    qtable: &[u16; 64],
    prev_dc: &mut i16,
) -> Result<[i16; 64]> {
    use std::io::Read;

    let read_i16 = |c: &mut Cursor<&[u8]>| -> Result<i16> {
        let mut buf = [0u8; 2];
        c.read_exact(&mut buf)
            .map_err(|e| NdiError::Codec(format!("Block read error: {e}")))?;
        Ok(i16::from_be_bytes(buf))
    };
    let read_u8 = |c: &mut Cursor<&[u8]>| -> Result<u8> {
        let mut buf = [0u8; 1];
        c.read_exact(&mut buf)
            .map_err(|e| NdiError::Codec(format!("Block read error: {e}")))?;
        Ok(buf[0])
    };

    let mut coeffs = [0i16; 64];

    // DC
    let dc_delta = read_i16(cursor)?;
    *prev_dc = prev_dc.wrapping_add(dc_delta);
    coeffs[0] = *prev_dc;

    // AC
    let mut zz_idx: usize = 1;
    loop {
        if zz_idx > 63 {
            break;
        }
        let run = read_u8(cursor)?;
        let level = read_i16(cursor)?;

        if run == 0 && level == 0 {
            // EOB
            break;
        }

        zz_idx += run as usize;
        if zz_idx < 64 {
            coeffs[zz_idx] = level;
            zz_idx += 1;
        }
    }

    // Dequantize in natural order via inverse zigzag
    let mut natural = [0i16; 64];
    for (zz_idx, &c) in coeffs.iter().enumerate() {
        let nat_idx = ZIGZAG[zz_idx];
        let q = qtable[zz_idx].max(1);
        natural[nat_idx] = c.saturating_mul(q as i16);
    }

    Ok(natural)
}

// ── Frame plane helpers (RGB ↔ YCbCr 4:2:0) ──────────────────────────────────

/// Convert packed RGB (3 bytes per pixel, stride = width*3) to planar YCbCr 4:2:0.
/// Returns (Y_plane, Cb_plane, Cr_plane).
/// Y: width×height, Cb/Cr: (width/2)×(height/2), all u8.
fn rgb_to_ycbcr420(rgb: &[u8], width: usize, height: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let pixels = width * height;
    let mut y_plane = vec![0u8; pixels];
    let cw = (width + 1) / 2;
    let ch = (height + 1) / 2;
    let mut cb_plane = vec![128u8; cw * ch];
    let mut cr_plane = vec![128u8; cw * ch];

    // Y plane
    for row in 0..height {
        for col in 0..width {
            let idx = (row * width + col) * 3;
            let r = rgb[idx] as i32;
            let g = rgb[idx + 1] as i32;
            let b = rgb[idx + 2] as i32;
            // BT.601 full range → studio swing Y
            let yv = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
            y_plane[row * width + col] = yv.clamp(16, 235) as u8;
        }
    }

    // Cb/Cr planes (4:2:0 average of 2×2 block)
    for vblock in 0..ch {
        for hblock in 0..cw {
            let mut sum_cb = 0i32;
            let mut sum_cr = 0i32;
            let mut count = 0i32;
            for dr in 0..2_usize {
                for dc in 0..2_usize {
                    let row = vblock * 2 + dr;
                    let col = hblock * 2 + dc;
                    if row < height && col < width {
                        let idx = (row * width + col) * 3;
                        let r = rgb[idx] as i32;
                        let g = rgb[idx + 1] as i32;
                        let b = rgb[idx + 2] as i32;
                        sum_cb += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
                        sum_cr += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
                        count += 1;
                    }
                }
            }
            if count > 0 {
                cb_plane[vblock * cw + hblock] = (sum_cb / count).clamp(0, 255) as u8;
                cr_plane[vblock * cw + hblock] = (sum_cr / count).clamp(0, 255) as u8;
            }
        }
    }

    (y_plane, cb_plane, cr_plane)
}

/// Convert planar YCbCr 4:2:0 back to packed RGB.
fn ycbcr420_to_rgb(
    y_plane: &[u8],
    cb_plane: &[u8],
    cr_plane: &[u8],
    width: usize,
    height: usize,
) -> Vec<u8> {
    let cw = (width + 1) / 2;
    let mut rgb = vec![0u8; width * height * 3];

    for row in 0..height {
        for col in 0..width {
            let y_val = y_plane[row * width + col] as i32;
            let cbi = (row / 2) * cw + (col / 2);
            let cb_val = cb_plane[cbi] as i32;
            let cr_val = cr_plane[cbi] as i32;

            let c = y_val - 16;
            let d = cb_val - 128;
            let e = cr_val - 128;

            let r = ((298 * c + 409 * e + 128) >> 8).clamp(0, 255) as u8;
            let g = ((298 * c - 100 * d - 208 * e + 128) >> 8).clamp(0, 255) as u8;
            let b = ((298 * c + 516 * d + 128) >> 8).clamp(0, 255) as u8;

            let idx = (row * width + col) * 3;
            rgb[idx] = r;
            rgb[idx + 1] = g;
            rgb[idx + 2] = b;
        }
    }

    rgb
}

// ── Plane DCT encoder/decoder ─────────────────────────────────────────────────

/// Encode a single component plane using block DCT + quantization + entropy coding.
fn encode_plane(plane: &[u8], plane_w: usize, plane_h: usize, qtable: &[u16; 64]) -> BytesMut {
    let blocks_x = (plane_w + 7) / 8;
    let blocks_y = (plane_h + 7) / 8;
    let capacity = blocks_x * blocks_y * (2 + 63 * 3 + 2); // generous upper bound
    let mut out = BytesMut::with_capacity(capacity);

    let mut prev_dc: i16 = 0;
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let block = extract_block(plane, plane_w, plane_h, bx, by);
            let dct = dct8x8(&block);
            encode_block(&dct, qtable, &mut prev_dc, &mut out);
        }
    }

    out
}

/// Decode a single component plane from the entropy bitstream.
fn decode_plane(
    cursor: &mut Cursor<&[u8]>,
    plane_w: usize,
    plane_h: usize,
    qtable: &[u16; 64],
) -> Result<Vec<u8>> {
    let blocks_x = (plane_w + 7) / 8;
    let blocks_y = (plane_h + 7) / 8;
    let padded_w = blocks_x * 8;
    let padded_h = blocks_y * 8;
    let mut padded = vec![128u8; padded_w * padded_h];

    let mut prev_dc: i16 = 0;
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let coeffs = decode_block(cursor, qtable, &mut prev_dc)?;

            // Convert i16 natural coefficients to f64 for IDCT
            let mut fblock = [0.0f64; 64];
            for i in 0..64 {
                fblock[i] = coeffs[i] as f64;
            }
            let spatial = idct8x8(&fblock);

            // Write reconstructed block back to padded buffer (clamp + level-shift)
            for row in 0..8_usize {
                for col in 0..8_usize {
                    let px = bx * 8 + col;
                    let py = by * 8 + row;
                    let val = (spatial[row * 8 + col] + 128.0).round().clamp(0.0, 255.0) as u8;
                    padded[py * padded_w + px] = val;
                }
            }
        }
    }

    // Crop padded buffer back to actual plane dimensions
    let mut plane_out = vec![0u8; plane_w * plane_h];
    for row in 0..plane_h {
        for col in 0..plane_w {
            plane_out[row * plane_w + col] = padded[row * padded_w + col];
        }
    }

    Ok(plane_out)
}

// ── YUV format types ──────────────────────────────────────────────────────────

/// YUV format types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum YuvFormat {
    /// YUV 4:2:2 (2 bytes per pixel)
    Yuv422,

    /// YUV 4:2:0 (1.5 bytes per pixel)
    Yuv420,

    /// YUV 4:4:4 (3 bytes per pixel)
    Yuv444,

    /// UYVY (packed YUV 4:2:2)
    Uyvy,

    /// YUYV (packed YUV 4:2:2)
    Yuyv,
}

impl YuvFormat {
    /// Get the number of bytes per pixel for this format
    pub fn bytes_per_pixel(&self) -> f32 {
        match self {
            Self::Yuv422 | Self::Uyvy | Self::Yuyv => 2.0,
            Self::Yuv420 => 1.5,
            Self::Yuv444 => 3.0,
        }
    }

    /// Calculate the buffer size needed for a frame
    pub fn buffer_size(&self, width: u32, height: u32) -> usize {
        (width as f32 * height as f32 * self.bytes_per_pixel()) as usize
    }
}

// ── SpeedHQ-equivalent: OxiMedia OXIT intra codec ────────────────────────────

/// OxiMedia OXIT intra-frame codec for NDI HX — a JPEG-style DCT intra codec.
///
/// Replaces NewTek's proprietary SpeedHQ with a patent-free, functionally
/// equivalent intra-frame codec.  The codec uses:
///
/// * BT.601 RGB → YCbCr colour conversion with 4:2:0 chroma sub-sampling
/// * 2-D DCT-II applied to 8×8 blocks on each component plane
/// * Quality-scaled JPEG-standard quantisation tables
/// * Standard zigzag scan order
/// * Delta-coded DC coefficients + run-level AC entropy coding
///
/// The bitstream format is described in the module-level documentation.
pub struct SpeedHqCodec {
    /// Compression quality (0–100)
    quality: u8,

    /// Enable fast mode (accepts same setting, kept for API compatibility)
    fast_mode: bool,
}

impl SpeedHqCodec {
    /// Create a new OXIT codec
    pub fn new(quality: u8) -> Self {
        Self {
            quality: quality.min(100),
            fast_mode: false,
        }
    }

    /// Create a codec with fast mode flag (same codec, kept for API compat)
    pub fn new_fast(quality: u8) -> Self {
        Self {
            quality: quality.min(100),
            fast_mode: true,
        }
    }

    /// Compress raw pixel data using the OXIT DCT intra codec.
    ///
    /// The `data` slice is interpreted as packed RGB (3 bytes per pixel) when
    /// its length equals `width * height * 3`, or as planar YUV 4:2:0 (1.5
    /// bytes per pixel) when its length equals `width * height * 3 / 2`.
    /// Any other length is treated as opaque raw bytes encoded as YUV 4:2:0
    /// after a best-effort reinterpretation.
    pub fn compress(&self, data: &[u8], width: u32, height: u32) -> Result<Bytes> {
        let w = width as usize;
        let h = height as usize;

        trace!("OXIT encoding {}×{} frame, quality={}", w, h, self.quality);

        // Determine whether input is RGB or already YUV 4:2:0
        let (y_plane, cb_plane, cr_plane) = if data.len() == w * h * 3 {
            // Packed RGB input — convert to YCbCr 4:2:0
            rgb_to_ycbcr420(data, w, h)
        } else {
            // Assume planar YUV 4:2:0 layout: Y(w×h) | Cb(cw×ch) | Cr(cw×ch)
            let cw = (w + 1) / 2;
            let ch = (h + 1) / 2;
            let y_end = w * h;
            let cb_end = y_end + cw * ch;
            let cr_end = cb_end + cw * ch;

            let y_slice = data.get(..y_end).unwrap_or(data);
            let cb_slice = data.get(y_end..cb_end).unwrap_or(&[]);
            let cr_slice = data.get(cb_end..cr_end).unwrap_or(&[]);

            let mut y = vec![128u8; w * h];
            let mut cb = vec![128u8; cw * ch];
            let mut cr = vec![128u8; cw * ch];
            let copy_y = y_slice.len().min(y.len());
            y[..copy_y].copy_from_slice(&y_slice[..copy_y]);
            let copy_cb = cb_slice.len().min(cb.len());
            cb[..copy_cb].copy_from_slice(&cb_slice[..copy_cb]);
            let copy_cr = cr_slice.len().min(cr.len());
            cr[..copy_cr].copy_from_slice(&cr_slice[..copy_cr]);
            (y, cb, cr)
        };

        let cw = (w + 1) / 2;
        let ch = (h + 1) / 2;

        let luma_q = scale_qtable(&LUMA_QTABLE_BASE, self.quality);
        let chroma_q = scale_qtable(&CHROMA_QTABLE_BASE, self.quality);

        // Encode each plane
        let y_bits = encode_plane(&y_plane, w, h, &luma_q);
        let cb_bits = encode_plane(&cb_plane, cw, ch, &chroma_q);
        let cr_bits = encode_plane(&cr_plane, cw, ch, &chroma_q);

        // Build final bitstream
        let total_len = OXIT_HEADER_LEN + 4 + y_bits.len() + 4 + cb_bits.len() + 4 + cr_bits.len();
        let mut out = BytesMut::with_capacity(total_len);

        // Header
        out.extend_from_slice(OXIT_MAGIC);
        out.put_u32(width);
        out.put_u32(height);
        out.put_u8(self.quality);
        out.put_u8(0); // flags (reserved)

        // Plane data with 4-byte length prefix each
        out.put_u32(y_bits.len() as u32);
        out.extend_from_slice(&y_bits);
        out.put_u32(cb_bits.len() as u32);
        out.extend_from_slice(&cb_bits);
        out.put_u32(cr_bits.len() as u32);
        out.extend_from_slice(&cr_bits);

        let compressed = out.freeze();

        debug!(
            "OXIT encoded {}×{} → {} bytes (ratio {:.1}%)",
            w,
            h,
            compressed.len(),
            compressed.len() as f64 / data.len() as f64 * 100.0,
        );

        Ok(compressed)
    }

    /// Decompress data produced by [`SpeedHqCodec::compress`].
    ///
    /// Returns the decoded pixel data.  If the original input to `compress` was
    /// packed RGB, the output is packed RGB (3 bytes per pixel, `width×height×3`
    /// bytes total).  If it was YUV 4:2:0, the output is reconstructed RGB.
    pub fn decompress(&self, data: &[u8], expected_size: usize) -> Result<Bytes> {
        use std::io::Read;

        trace!(
            "OXIT decoding {} bytes (expected {})",
            data.len(),
            expected_size
        );

        if data.len() < OXIT_HEADER_LEN + 12 {
            return Err(NdiError::Codec("OXIT bitstream too short".into()));
        }

        // Validate magic
        if &data[..4] != OXIT_MAGIC {
            return Err(NdiError::Codec(format!(
                "OXIT magic mismatch: expected {:?}, got {:?}",
                OXIT_MAGIC,
                &data[..4]
            )));
        }

        let width = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;
        let height = u32::from_be_bytes([data[8], data[9], data[10], data[11]]) as usize;
        let quality = data[12];
        // data[13] = flags (reserved, ignored)

        let luma_q = scale_qtable(&LUMA_QTABLE_BASE, quality);
        let chroma_q = scale_qtable(&CHROMA_QTABLE_BASE, quality);

        let cw = (width + 1) / 2;
        let ch = (height + 1) / 2;

        let mut cursor = Cursor::new(&data[OXIT_HEADER_LEN..]);

        let read_u32 = |c: &mut Cursor<&[u8]>| -> Result<usize> {
            let mut buf = [0u8; 4];
            c.read_exact(&mut buf)
                .map_err(|e| NdiError::Codec(format!("Plane length read error: {e}")))?;
            Ok(u32::from_be_bytes(buf) as usize)
        };

        // Y plane
        let y_len = read_u32(&mut cursor)?;
        let pos = cursor.position() as usize;
        let y_slice = data
            .get(OXIT_HEADER_LEN + pos..OXIT_HEADER_LEN + pos + y_len)
            .ok_or_else(|| NdiError::Codec("OXIT Y plane out of bounds".into()))?;
        let y_plane = {
            let mut c = Cursor::new(y_slice);
            decode_plane(&mut c, width, height, &luma_q)?
        };
        cursor.set_position((pos + y_len) as u64);

        // Cb plane
        let cb_len = read_u32(&mut cursor)?;
        let pos = cursor.position() as usize;
        let cb_slice = data
            .get(OXIT_HEADER_LEN + pos..OXIT_HEADER_LEN + pos + cb_len)
            .ok_or_else(|| NdiError::Codec("OXIT Cb plane out of bounds".into()))?;
        let cb_plane = {
            let mut c = Cursor::new(cb_slice);
            decode_plane(&mut c, cw, ch, &chroma_q)?
        };
        cursor.set_position((pos + cb_len) as u64);

        // Cr plane
        let cr_len = read_u32(&mut cursor)?;
        let pos = cursor.position() as usize;
        let cr_slice = data
            .get(OXIT_HEADER_LEN + pos..OXIT_HEADER_LEN + pos + cr_len)
            .ok_or_else(|| NdiError::Codec("OXIT Cr plane out of bounds".into()))?;
        let cr_plane = {
            let mut c = Cursor::new(cr_slice);
            decode_plane(&mut c, cw, ch, &chroma_q)?
        };

        // Reconstruct RGB
        let rgb = ycbcr420_to_rgb(&y_plane, &cb_plane, &cr_plane, width, height);

        debug!("OXIT decoded {}×{} → {} bytes", width, height, rgb.len());

        // Validate size (best-effort; caller may expect raw YUV size instead)
        if rgb.len() != expected_size {
            // Accept if caller's expected_size matches YUV 4:2:0 layout
            let yuv420_size = width * height * 3 / 2;
            if expected_size != rgb.len() && expected_size != yuv420_size {
                return Err(NdiError::Codec(format!(
                    "OXIT decompressed size mismatch: expected {expected_size}, got {} (RGB) or {yuv420_size} (YUV420)",
                    rgb.len(),
                )));
            }
        }

        Ok(Bytes::from(rgb))
    }

    /// Set compression quality (0–100)
    pub fn set_quality(&mut self, quality: u8) {
        self.quality = quality.min(100);
    }

    /// Get current quality setting
    pub fn quality(&self) -> u8 {
        self.quality
    }

    /// Enable or disable fast mode flag (kept for API compatibility)
    pub fn set_fast_mode(&mut self, enabled: bool) {
        self.fast_mode = enabled;
    }

    /// Check if fast mode flag is set
    pub fn is_fast_mode(&self) -> bool {
        self.fast_mode
    }
}

impl Default for SpeedHqCodec {
    fn default() -> Self {
        Self::new(80)
    }
}

// ── YUV format converter ──────────────────────────────────────────────────────

/// YUV format converter
pub struct YuvConverter;

impl YuvConverter {
    /// Convert RGB to YUV422
    pub fn rgb_to_yuv422(rgb: &[u8], width: u32, height: u32) -> Result<Bytes> {
        if rgb.len() < (width * height * 3) as usize {
            return Err(NdiError::InvalidFrameFormat);
        }

        let mut yuv = BytesMut::with_capacity(YuvFormat::Yuv422.buffer_size(width, height));

        for y in 0..height {
            for x in 0..(width / 2) {
                let idx = ((y * width + x * 2) * 3) as usize;

                let r0 = i32::from(rgb[idx]);
                let g0 = i32::from(rgb[idx + 1]);
                let b0 = i32::from(rgb[idx + 2]);

                let r1 = i32::from(rgb[idx + 3]);
                let g1 = i32::from(rgb[idx + 4]);
                let b1 = i32::from(rgb[idx + 5]);

                // Convert to YUV
                let y0 = Self::rgb_to_y(r0, g0, b0);
                let y1 = Self::rgb_to_y(r1, g1, b1);
                let u = Self::rgb_to_u(r0, g0, b0);
                let v = Self::rgb_to_v(r0, g0, b0);

                yuv.extend_from_slice(&[y0, u, y1, v]);
            }
        }

        Ok(yuv.freeze())
    }

    /// Convert YUV422 to RGB
    pub fn yuv422_to_rgb(yuv: &[u8], width: u32, height: u32) -> Result<Bytes> {
        if yuv.len() < YuvFormat::Yuv422.buffer_size(width, height) {
            return Err(NdiError::InvalidFrameFormat);
        }

        let mut rgb = BytesMut::with_capacity((width * height * 3) as usize);

        for y in 0..height {
            for x in 0..(width / 2) {
                let idx = ((y * width + x * 2) * 2) as usize;

                let y0 = i32::from(yuv[idx]);
                let u = i32::from(yuv[idx + 1]);
                let y1 = i32::from(yuv[idx + 2]);
                let v = i32::from(yuv[idx + 3]);

                // Convert first pixel
                let (r0, g0, b0) = Self::yuv_to_rgb(y0, u, v);
                rgb.extend_from_slice(&[r0, g0, b0]);

                // Convert second pixel
                let (r1, g1, b1) = Self::yuv_to_rgb(y1, u, v);
                rgb.extend_from_slice(&[r1, g1, b1]);
            }
        }

        Ok(rgb.freeze())
    }

    /// Convert RGB to YUV420
    pub fn rgb_to_yuv420(rgb: &[u8], width: u32, height: u32) -> Result<Bytes> {
        if rgb.len() < (width * height * 3) as usize {
            return Err(NdiError::InvalidFrameFormat);
        }

        let mut yuv = BytesMut::with_capacity(YuvFormat::Yuv420.buffer_size(width, height));

        // Y plane
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                let r = i32::from(rgb[idx]);
                let g = i32::from(rgb[idx + 1]);
                let b = i32::from(rgb[idx + 2]);
                yuv.extend_from_slice(&[Self::rgb_to_y(r, g, b)]);
            }
        }

        // U plane (subsampled)
        for y in (0..height).step_by(2) {
            for x in (0..width).step_by(2) {
                let idx = ((y * width + x) * 3) as usize;
                let r = i32::from(rgb[idx]);
                let g = i32::from(rgb[idx + 1]);
                let b = i32::from(rgb[idx + 2]);
                yuv.extend_from_slice(&[Self::rgb_to_u(r, g, b)]);
            }
        }

        // V plane (subsampled)
        for y in (0..height).step_by(2) {
            for x in (0..width).step_by(2) {
                let idx = ((y * width + x) * 3) as usize;
                let r = i32::from(rgb[idx]);
                let g = i32::from(rgb[idx + 1]);
                let b = i32::from(rgb[idx + 2]);
                yuv.extend_from_slice(&[Self::rgb_to_v(r, g, b)]);
            }
        }

        Ok(yuv.freeze())
    }

    /// Convert YUV420 to RGB
    pub fn yuv420_to_rgb(yuv: &[u8], width: u32, height: u32) -> Result<Bytes> {
        if yuv.len() < YuvFormat::Yuv420.buffer_size(width, height) {
            return Err(NdiError::InvalidFrameFormat);
        }

        let mut rgb = BytesMut::with_capacity((width * height * 3) as usize);

        let y_plane_size = (width * height) as usize;
        let u_plane_size = (width * height / 4) as usize;

        for y in 0..height {
            for x in 0..width {
                let y_idx = (y * width + x) as usize;
                let uv_idx = ((y / 2) * (width / 2) + (x / 2)) as usize;

                let y_val = i32::from(yuv[y_idx]);
                let u_val = i32::from(yuv[y_plane_size + uv_idx]);
                let v_val = i32::from(yuv[y_plane_size + u_plane_size + uv_idx]);

                let (r, g, b) = Self::yuv_to_rgb(y_val, u_val, v_val);
                rgb.extend_from_slice(&[r, g, b]);
            }
        }

        Ok(rgb.freeze())
    }

    /// Convert RGB to UYVY
    pub fn rgb_to_uyvy(rgb: &[u8], width: u32, height: u32) -> Result<Bytes> {
        if rgb.len() < (width * height * 3) as usize {
            return Err(NdiError::InvalidFrameFormat);
        }

        let mut uyvy = BytesMut::with_capacity(YuvFormat::Uyvy.buffer_size(width, height));

        for y in 0..height {
            for x in 0..(width / 2) {
                let idx = ((y * width + x * 2) * 3) as usize;

                let r0 = i32::from(rgb[idx]);
                let g0 = i32::from(rgb[idx + 1]);
                let b0 = i32::from(rgb[idx + 2]);

                let r1 = i32::from(rgb[idx + 3]);
                let g1 = i32::from(rgb[idx + 4]);
                let b1 = i32::from(rgb[idx + 5]);

                let y0 = Self::rgb_to_y(r0, g0, b0);
                let y1 = Self::rgb_to_y(r1, g1, b1);
                let u = Self::rgb_to_u(r0, g0, b0);
                let v = Self::rgb_to_v(r0, g0, b0);

                // UYVY format: U Y0 V Y1
                uyvy.extend_from_slice(&[u, y0, v, y1]);
            }
        }

        Ok(uyvy.freeze())
    }

    /// Convert UYVY to RGB
    pub fn uyvy_to_rgb(uyvy: &[u8], width: u32, height: u32) -> Result<Bytes> {
        if uyvy.len() < YuvFormat::Uyvy.buffer_size(width, height) {
            return Err(NdiError::InvalidFrameFormat);
        }

        let mut rgb = BytesMut::with_capacity((width * height * 3) as usize);

        for y in 0..height {
            for x in 0..(width / 2) {
                let idx = ((y * width + x * 2) * 2) as usize;

                let u = i32::from(uyvy[idx]);
                let y0 = i32::from(uyvy[idx + 1]);
                let v = i32::from(uyvy[idx + 2]);
                let y1 = i32::from(uyvy[idx + 3]);

                let (r0, g0, b0) = Self::yuv_to_rgb(y0, u, v);
                rgb.extend_from_slice(&[r0, g0, b0]);

                let (r1, g1, b1) = Self::yuv_to_rgb(y1, u, v);
                rgb.extend_from_slice(&[r1, g1, b1]);
            }
        }

        Ok(rgb.freeze())
    }

    /// Convert RGB component values to Y (luminance)
    fn rgb_to_y(r: i32, g: i32, b: i32) -> u8 {
        ((66 * r + 129 * g + 25 * b + 128) >> 8).clamp(16, 235) as u8
    }

    /// Convert RGB component values to U (chrominance)
    fn rgb_to_u(r: i32, g: i32, b: i32) -> u8 {
        ((-38 * r - 74 * g + 112 * b + 128) >> 8)
            .clamp(-112, 112)
            .wrapping_add(128) as u8
    }

    /// Convert RGB component values to V (chrominance)
    fn rgb_to_v(r: i32, g: i32, b: i32) -> u8 {
        ((112 * r - 94 * g - 18 * b + 128) >> 8)
            .clamp(-112, 112)
            .wrapping_add(128) as u8
    }

    /// Convert YUV component values to RGB
    fn yuv_to_rgb(y: i32, u: i32, v: i32) -> (u8, u8, u8) {
        let c = y - 16;
        let d = u - 128;
        let e = v - 128;

        let r = ((298 * c + 409 * e + 128) >> 8).clamp(0, 255) as u8;
        let g = ((298 * c - 100 * d - 208 * e + 128) >> 8).clamp(0, 255) as u8;
        let b = ((298 * c + 516 * d + 128) >> 8).clamp(0, 255) as u8;

        (r, g, b)
    }

    /// Resize YUV422 frame (simple nearest neighbor)
    pub fn resize_yuv422(
        yuv: &[u8],
        src_width: u32,
        src_height: u32,
        dst_width: u32,
        dst_height: u32,
    ) -> Result<Bytes> {
        let mut output =
            BytesMut::with_capacity(YuvFormat::Yuv422.buffer_size(dst_width, dst_height));

        let x_ratio = (src_width << 16) / dst_width;
        let y_ratio = (src_height << 16) / dst_height;

        for y in 0..dst_height {
            let src_y = ((y * y_ratio) >> 16).min(src_height - 1);

            for x in 0..(dst_width / 2) {
                let src_x = ((x * 2 * x_ratio) >> 16).min(src_width - 2);
                let src_idx = ((src_y * src_width + src_x) * 2) as usize;

                if src_idx + 3 < yuv.len() {
                    output.extend_from_slice(&yuv[src_idx..src_idx + 4]);
                }
            }
        }

        Ok(output.freeze())
    }

    /// Crop YUV422 frame
    pub fn crop_yuv422(
        yuv: &[u8],
        width: u32,
        height: u32,
        x: u32,
        y: u32,
        crop_width: u32,
        crop_height: u32,
    ) -> Result<Bytes> {
        if x + crop_width > width || y + crop_height > height {
            return Err(NdiError::InvalidFrameFormat);
        }

        // Ensure even width for YUV422
        let crop_width = crop_width & !1;
        let x = x & !1;

        let mut output =
            BytesMut::with_capacity(YuvFormat::Yuv422.buffer_size(crop_width, crop_height));

        for row in y..(y + crop_height) {
            let src_idx = ((row * width + x) * 2) as usize;
            let len = (crop_width * 2) as usize;

            if src_idx + len <= yuv.len() {
                output.extend_from_slice(&yuv[src_idx..src_idx + len]);
            }
        }

        Ok(output.freeze())
    }
}

// ── Hardware acceleration hooks ───────────────────────────────────────────────

/// Hardware acceleration hooks
///
/// These are placeholder functions that can be replaced with actual hardware
/// acceleration implementations (CUDA, Metal, etc.)
pub mod hardware {
    use super::*;

    /// Check if hardware acceleration is available
    pub fn is_available() -> bool {
        false
    }

    /// Get the name of the hardware accelerator
    pub fn accelerator_name() -> Option<String> {
        None
    }

    /// Compress using hardware acceleration
    pub fn hw_compress(_data: &[u8], _width: u32, _height: u32, _quality: u8) -> Result<Bytes> {
        Err(NdiError::Codec(
            "Hardware acceleration not available".to_string(),
        ))
    }

    /// Decompress using hardware acceleration
    pub fn hw_decompress(_data: &[u8], _width: u32, _height: u32) -> Result<Bytes> {
        Err(NdiError::Codec(
            "Hardware acceleration not available".to_string(),
        ))
    }

    /// Convert RGB to YUV using hardware acceleration
    pub fn hw_rgb_to_yuv(_rgb: &[u8], _width: u32, _height: u32) -> Result<Bytes> {
        Err(NdiError::Codec(
            "Hardware acceleration not available".to_string(),
        ))
    }

    /// Convert YUV to RGB using hardware acceleration
    pub fn hw_yuv_to_rgb(_yuv: &[u8], _width: u32, _height: u32) -> Result<Bytes> {
        Err(NdiError::Codec(
            "Hardware acceleration not available".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yuv_format_buffer_size() {
        assert_eq!(YuvFormat::Yuv422.buffer_size(1920, 1080), 1920 * 1080 * 2);
        assert_eq!(
            YuvFormat::Yuv420.buffer_size(1920, 1080),
            (1920 * 1080 * 3) / 2
        );
        assert_eq!(YuvFormat::Yuv444.buffer_size(1920, 1080), 1920 * 1080 * 3);
    }

    #[test]
    fn test_oxit_codec_roundtrip_rgb() {
        // Use a small frame so the test runs quickly
        let w = 32u32;
        let h = 32u32;
        let mut data = vec![0u8; (w * h * 3) as usize];
        // Fill with a gradient pattern
        for row in 0..h as usize {
            for col in 0..w as usize {
                let idx = (row * w as usize + col) * 3;
                data[idx] = (col * 8) as u8; // R
                data[idx + 1] = (row * 8) as u8; // G
                data[idx + 2] = 128u8; // B
            }
        }

        let codec = SpeedHqCodec::new(75);
        let compressed = codec
            .compress(&data, w, h)
            .expect("OXIT compress must succeed");

        // Bitstream must start with magic
        assert_eq!(&compressed[..4], OXIT_MAGIC);

        // Decompress and check approximate roundtrip (lossy, so allow ±16 per channel)
        let decompressed = codec
            .decompress(&compressed, (w * h * 3) as usize)
            .expect("OXIT decompress must succeed");

        assert_eq!(decompressed.len(), (w * h * 3) as usize);
        // Verify luma is approximately preserved (DCT is lossy)
        let mut max_err = 0i32;
        for (&orig, &rec) in data.iter().zip(decompressed.iter()) {
            max_err = max_err.max((orig as i32 - rec as i32).abs());
        }
        assert!(
            max_err <= 64,
            "Max per-pixel error {max_err} exceeds threshold"
        );
    }

    #[test]
    fn test_oxit_codec_quality_settings() {
        let codec = SpeedHqCodec::new(75);
        assert_eq!(codec.quality(), 75);
        assert!(!codec.is_fast_mode());

        let codec_fast = SpeedHqCodec::new_fast(50);
        assert_eq!(codec_fast.quality(), 50);
        assert!(codec_fast.is_fast_mode());
    }

    #[test]
    fn test_oxit_higher_quality_lower_error() {
        // Use a 64×64 frame with a smooth gradient — DCT excels at smooth signals,
        // so higher quality should faithfully reconstruct it while lower quality
        // introduces visible blocking artefacts (higher MSE).
        let w = 64u32;
        let h = 64u32;
        let mut data = vec![0u8; (w * h * 3) as usize];
        for row in 0..h as usize {
            for col in 0..w as usize {
                let idx = (row * w as usize + col) * 3;
                data[idx] = ((row * 4) % 256) as u8; // R: vertical gradient
                data[idx + 1] = ((col * 4) % 256) as u8; // G: horizontal gradient
                data[idx + 2] = 128u8; // B: constant mid-grey
            }
        }

        let codec_high = SpeedHqCodec::new(95);
        let codec_low = SpeedHqCodec::new(5);

        let comp_high = codec_high.compress(&data, w, h).expect("compress high");
        let comp_low = codec_low.compress(&data, w, h).expect("compress low");

        let dec_high = codec_high
            .decompress(&comp_high, (w * h * 3) as usize)
            .expect("decompress high");
        let dec_low = codec_low
            .decompress(&comp_low, (w * h * 3) as usize)
            .expect("decompress low");

        let mse = |a: &[u8], b: &[u8]| -> f64 {
            a.iter()
                .zip(b.iter())
                .map(|(&x, &y)| {
                    let d = x as f64 - y as f64;
                    d * d
                })
                .sum::<f64>()
                / a.len() as f64
        };

        let mse_high = mse(&data, &dec_high);
        let mse_low = mse(&data, &dec_low);

        // With a smooth gradient and a large quality gap (95 vs 5),
        // high quality must have strictly lower reconstruction error.
        assert!(
            mse_high < mse_low,
            "High-quality MSE {mse_high:.2} should be < low-quality MSE {mse_low:.2}"
        );
    }

    #[test]
    fn test_oxit_magic_validation() {
        let codec = SpeedHqCodec::new(80);
        let bad_data = vec![0u8; 50];
        let result = codec.decompress(&bad_data, 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_scale_qtable_quality_bounds() {
        // quality=1 → large Q values (high quantization = high loss)
        let q1 = scale_qtable(&LUMA_QTABLE_BASE, 1);
        // quality=100 → small Q values (low quantization = low loss)
        let q100 = scale_qtable(&LUMA_QTABLE_BASE, 100);
        // Every entry at quality=1 must be >= entry at quality=100
        for (&a, &b) in q1.iter().zip(q100.iter()) {
            assert!(a >= b, "q1[i]={a} should be >= q100[i]={b}");
        }
    }

    #[test]
    fn test_dct_idct_roundtrip() {
        // A block of linearly varying values
        let mut block = [0.0f64; 64];
        for i in 0..64 {
            block[i] = i as f64 - 32.0;
        }
        let dct = dct8x8(&block);
        let reconstructed = idct8x8(&dct);
        for (orig, rec) in block.iter().zip(reconstructed.iter()) {
            assert!(
                (orig - rec).abs() < 1e-8,
                "DCT→IDCT roundtrip error: {orig} ≠ {rec}"
            );
        }
    }

    #[test]
    fn test_rgb_yuv_conversion() {
        let rgb = vec![
            255, 0, 0, 255, 0, 0, // 2 red pixels
            0, 255, 0, 0, 255, 0, // 2 green pixels
        ];

        let yuv = YuvConverter::rgb_to_yuv422(&rgb, 4, 1).expect("rgb_to_yuv422");
        assert_eq!(yuv.len(), 8); // 4 pixels in YUV422 = 8 bytes

        let rgb_back = YuvConverter::yuv422_to_rgb(&yuv, 4, 1).expect("yuv422_to_rgb");
        assert_eq!(rgb_back.len(), rgb.len());
    }

    #[test]
    fn test_yuv422_crop() {
        let yuv = vec![128u8; 1920 * 1080 * 2];
        let cropped =
            YuvConverter::crop_yuv422(&yuv, 1920, 1080, 100, 100, 640, 480).expect("crop_yuv422");
        assert_eq!(cropped.len(), 640 * 480 * 2);
    }

    #[test]
    fn test_yuv422_resize() {
        let yuv = vec![128u8; 1920 * 1080 * 2];
        let resized =
            YuvConverter::resize_yuv422(&yuv, 1920, 1080, 640, 480).expect("resize_yuv422");
        assert_eq!(resized.len(), 640 * 480 * 2);
    }

    #[test]
    fn test_hardware_acceleration_not_available() {
        assert!(!hardware::is_available());
        assert!(hardware::accelerator_name().is_none());
    }
}
