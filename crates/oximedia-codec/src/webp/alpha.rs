//! WebP ALPH chunk handler for alpha channel encoding and decoding.
//!
//! In WebP extended format, the alpha channel is stored separately in an ALPH chunk.
//! The chunk consists of a 1-byte header followed by filtered/compressed alpha data.
//!
//! # ALPH Header Byte Layout
//!
//! ```text
//! ┌─────────┬─────────┬──────────────┬────────────────────┐
//! │ bits 7:6│ bits 5:4│ bits 3:2     │ bits 1:0           │
//! │reserved │pre-proc │filter method │compression method  │
//! └─────────┴─────────┴──────────────┴────────────────────┘
//! ```

use crate::error::{CodecError, CodecResult};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Alpha compression method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlphaCompression {
    /// Raw (uncompressed) alpha data.
    NoCompression = 0,
    /// Alpha data encoded as a VP8L (WebP lossless) bitstream.
    WebPLossless = 1,
}

/// Alpha filtering method applied before compression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlphaFilter {
    /// No filtering – data stored as-is.
    None = 0,
    /// Horizontal prediction: each pixel predicted from its left neighbour.
    Horizontal = 1,
    /// Vertical prediction: each pixel predicted from the pixel above.
    Vertical = 2,
    /// Gradient prediction: `left + top - top_left`, clamped to `[0, 255]`.
    Gradient = 3,
}

/// Parsed ALPH chunk header.
#[derive(Debug, Clone)]
pub struct AlphaHeader {
    /// Compression method used for the alpha data.
    pub compression: AlphaCompression,
    /// Spatial filter applied prior to compression.
    pub filter: AlphaFilter,
    /// Pre-processing level (0 = none, 1 = level reduction).
    pub pre_processing: u8,
}

// ---------------------------------------------------------------------------
// Header parsing helpers
// ---------------------------------------------------------------------------

impl AlphaCompression {
    fn from_bits(bits: u8) -> CodecResult<Self> {
        match bits & 0x03 {
            0 => Ok(Self::NoCompression),
            1 => Ok(Self::WebPLossless),
            v => Err(CodecError::InvalidBitstream(format!(
                "unknown alpha compression method: {v}"
            ))),
        }
    }
}

impl AlphaFilter {
    fn from_bits(bits: u8) -> CodecResult<Self> {
        match bits & 0x03 {
            0 => Ok(Self::None),
            1 => Ok(Self::Horizontal),
            2 => Ok(Self::Vertical),
            3 => Ok(Self::Gradient),
            // unreachable because of the mask, but the compiler does not know
            _ => Err(CodecError::InvalidBitstream(
                "unknown alpha filter method".to_string(),
            )),
        }
    }
}

impl AlphaHeader {
    /// Parse the single header byte from the beginning of an ALPH chunk payload.
    pub fn parse(byte: u8) -> CodecResult<Self> {
        let reserved = (byte >> 6) & 0x03;
        if reserved != 0 {
            return Err(CodecError::InvalidBitstream(format!(
                "ALPH header reserved bits are non-zero: {reserved}"
            )));
        }
        let compression = AlphaCompression::from_bits(byte & 0x03)?;
        let filter = AlphaFilter::from_bits((byte >> 2) & 0x03)?;
        let pre_processing = (byte >> 4) & 0x03;

        Ok(Self {
            compression,
            filter,
            pre_processing,
        })
    }

    /// Serialize the header back to a single byte.
    pub fn to_byte(&self) -> u8 {
        let comp = self.compression as u8;
        let filt = (self.filter as u8) << 2;
        let prep = (self.pre_processing & 0x03) << 4;
        comp | filt | prep
    }
}

// ---------------------------------------------------------------------------
// Gradient prediction helper
// ---------------------------------------------------------------------------

/// Compute the gradient predictor: `left + top - top_left`, clamped to [0, 255].
#[inline]
fn gradient_predict(left: u8, top: u8, top_left: u8) -> u8 {
    let val = left as i16 + top as i16 - top_left as i16;
    val.clamp(0, 255) as u8
}

// ---------------------------------------------------------------------------
// Filtering (decode direction – reconstruction)
// ---------------------------------------------------------------------------

/// Reconstruct original alpha values from filtered residuals.
///
/// This is the *decode* direction: we read the stored residual and add the
/// prediction derived from already-reconstructed neighbours.
fn apply_filter(data: &mut [u8], width: u32, height: u32, filter: AlphaFilter) {
    let w = width as usize;
    let h = height as usize;
    let total = w * h;
    if total == 0 || data.len() < total {
        return;
    }

    match filter {
        AlphaFilter::None => { /* nothing to do */ }

        AlphaFilter::Horizontal => {
            for y in 0..h {
                let row_start = y * w;
                // First pixel in row: predicted from 0 (or left=0)
                // data[row_start] already holds the correct value (residual + 0)
                for x in 1..w {
                    let idx = row_start + x;
                    let left = data[idx - 1];
                    data[idx] = data[idx].wrapping_add(left);
                }
            }
        }

        AlphaFilter::Vertical => {
            // First row: no prediction (residual = raw)
            for y in 1..h {
                for x in 0..w {
                    let idx = y * w + x;
                    let top = data[idx - w];
                    data[idx] = data[idx].wrapping_add(top);
                }
            }
        }

        AlphaFilter::Gradient => {
            // First row: horizontal-only prediction
            for x in 1..w {
                data[x] = data[x].wrapping_add(data[x - 1]);
            }
            for y in 1..h {
                let row_start = y * w;
                // First pixel of row: predict from top only
                data[row_start] = data[row_start].wrapping_add(data[row_start - w]);

                for x in 1..w {
                    let idx = row_start + x;
                    let left = data[idx - 1];
                    let top = data[idx - w];
                    let top_left = data[idx - w - 1];
                    let pred = gradient_predict(left, top, top_left);
                    data[idx] = data[idx].wrapping_add(pred);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Inverse filtering (encode direction – produce residuals)
// ---------------------------------------------------------------------------

/// Produce filtered residuals from raw alpha values.
///
/// This is the *encode* direction: given original alpha values we compute
/// `residual = original - prediction` (using wrapping arithmetic).
fn apply_inverse_filter(
    data: &[u8],
    width: u32,
    height: u32,
    filter: AlphaFilter,
) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let total = w * h;
    if total == 0 {
        return Vec::new();
    }

    match filter {
        AlphaFilter::None => data[..total].to_vec(),

        AlphaFilter::Horizontal => {
            let mut out = vec![0u8; total];
            for y in 0..h {
                let row_start = y * w;
                out[row_start] = data[row_start]; // first pixel – no prediction
                for x in 1..w {
                    let idx = row_start + x;
                    let left = data[idx - 1];
                    out[idx] = data[idx].wrapping_sub(left);
                }
            }
            out
        }

        AlphaFilter::Vertical => {
            let mut out = vec![0u8; total];
            // First row – no prediction
            out[..w].copy_from_slice(&data[..w]);
            for y in 1..h {
                for x in 0..w {
                    let idx = y * w + x;
                    let top = data[idx - w];
                    out[idx] = data[idx].wrapping_sub(top);
                }
            }
            out
        }

        AlphaFilter::Gradient => {
            let mut out = vec![0u8; total];
            out[0] = data[0]; // top-left corner

            // First row: horizontal prediction
            for x in 1..w {
                out[x] = data[x].wrapping_sub(data[x - 1]);
            }
            for y in 1..h {
                let row_start = y * w;
                // First pixel of row: predict from top only
                out[row_start] = data[row_start].wrapping_sub(data[row_start - w]);

                for x in 1..w {
                    let idx = row_start + x;
                    let left = data[idx - 1];
                    let top = data[idx - w];
                    let top_left = data[idx - w - 1];
                    let pred = gradient_predict(left, top, top_left);
                    out[idx] = data[idx].wrapping_sub(pred);
                }
            }
            out
        }
    }
}

// ---------------------------------------------------------------------------
// Heuristic: choose the best filter for encoding
// ---------------------------------------------------------------------------

/// Score a filter by summing the absolute residuals – lower is better
/// (residuals closer to zero compress better).
fn score_filter(data: &[u8], width: u32, height: u32, filter: AlphaFilter) -> u64 {
    let residuals = apply_inverse_filter(data, width, height, filter);
    residuals.iter().map(|&b| {
        // Treat residual as signed offset from zero (wrapping distance)
        let v = b as i16;
        let d = if v > 128 { 256 - v } else { v };
        d as u64
    }).sum()
}

/// Select the filter that produces the smallest total residual.
fn select_best_filter(data: &[u8], width: u32, height: u32) -> AlphaFilter {
    let filters = [
        AlphaFilter::None,
        AlphaFilter::Horizontal,
        AlphaFilter::Vertical,
        AlphaFilter::Gradient,
    ];

    let mut best_filter = AlphaFilter::None;
    let mut best_score = u64::MAX;

    for &f in &filters {
        let s = score_filter(data, width, height, f);
        if s < best_score {
            best_score = s;
            best_filter = f;
        }
    }

    best_filter
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Decode alpha channel from ALPH chunk payload.
///
/// `data` is the raw ALPH chunk payload (starting with the header byte).
/// Returns a `Vec<u8>` of length `width * height` with reconstructed alpha
/// values in row-major order.
pub fn decode_alpha(data: &[u8], width: u32, height: u32) -> CodecResult<Vec<u8>> {
    if data.is_empty() {
        return Err(CodecError::InvalidBitstream(
            "ALPH chunk is empty".to_string(),
        ));
    }

    let total = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| {
            CodecError::InvalidParameter(format!(
                "alpha plane dimensions overflow: {width} x {height}"
            ))
        })?;

    if total == 0 {
        return Ok(Vec::new());
    }

    let header = AlphaHeader::parse(data[0])?;
    let payload = &data[1..];

    match header.compression {
        AlphaCompression::NoCompression => {
            if payload.len() < total {
                return Err(CodecError::BufferTooSmall {
                    needed: total,
                    have: payload.len(),
                });
            }
            let mut alpha = payload[..total].to_vec();
            apply_filter(&mut alpha, width, height, header.filter);
            Ok(alpha)
        }
        AlphaCompression::WebPLossless => Err(CodecError::UnsupportedFeature(
            "VP8L-compressed alpha channel is not yet supported".to_string(),
        )),
    }
}

/// Encode alpha channel into an ALPH chunk payload.
///
/// `alpha` must contain exactly `width * height` bytes in row-major order.
/// Returns the complete chunk payload: 1-byte header followed by the
/// (optionally filtered) alpha data, using no compression.
pub fn encode_alpha(
    alpha: &[u8],
    width: u32,
    height: u32,
) -> CodecResult<Vec<u8>> {
    let total = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| {
            CodecError::InvalidParameter(format!(
                "alpha plane dimensions overflow: {width} x {height}"
            ))
        })?;

    if alpha.len() < total {
        return Err(CodecError::BufferTooSmall {
            needed: total,
            have: alpha.len(),
        });
    }

    if total == 0 {
        // Degenerate case: just a header byte
        let hdr = AlphaHeader {
            compression: AlphaCompression::NoCompression,
            filter: AlphaFilter::None,
            pre_processing: 0,
        };
        return Ok(vec![hdr.to_byte()]);
    }

    let input = &alpha[..total];

    // Choose the best filter heuristically
    let best_filter = select_best_filter(input, width, height);

    let header = AlphaHeader {
        compression: AlphaCompression::NoCompression,
        filter: best_filter,
        pre_processing: 0,
    };

    let residuals = apply_inverse_filter(input, width, height, best_filter);

    let mut out = Vec::with_capacity(1 + residuals.len());
    out.push(header.to_byte());
    out.extend_from_slice(&residuals);
    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Header round-trip ---------------------------------------------------

    #[test]
    fn header_roundtrip_no_compression_no_filter() {
        let hdr = AlphaHeader {
            compression: AlphaCompression::NoCompression,
            filter: AlphaFilter::None,
            pre_processing: 0,
        };
        let byte = hdr.to_byte();
        assert_eq!(byte, 0x00);
        let parsed = AlphaHeader::parse(byte).expect("parse should succeed");
        assert_eq!(parsed.compression, AlphaCompression::NoCompression);
        assert_eq!(parsed.filter, AlphaFilter::None);
        assert_eq!(parsed.pre_processing, 0);
    }

    #[test]
    fn header_roundtrip_all_fields() {
        // compression=0, filter=gradient(3), pre_processing=1
        // byte = 0 | (3 << 2) | (1 << 4) = 0 | 12 | 16 = 28
        let hdr = AlphaHeader {
            compression: AlphaCompression::NoCompression,
            filter: AlphaFilter::Gradient,
            pre_processing: 1,
        };
        let byte = hdr.to_byte();
        assert_eq!(byte, 0x1C);
        let parsed = AlphaHeader::parse(byte).expect("parse should succeed");
        assert_eq!(parsed.compression, AlphaCompression::NoCompression);
        assert_eq!(parsed.filter, AlphaFilter::Gradient);
        assert_eq!(parsed.pre_processing, 1);
    }

    #[test]
    fn header_roundtrip_webp_lossless_horizontal() {
        // compression=1, filter=horizontal(1), pre_processing=0
        // byte = 1 | (1 << 2) = 5
        let hdr = AlphaHeader {
            compression: AlphaCompression::WebPLossless,
            filter: AlphaFilter::Horizontal,
            pre_processing: 0,
        };
        let byte = hdr.to_byte();
        assert_eq!(byte, 0x05);
        let parsed = AlphaHeader::parse(byte).expect("parse should succeed");
        assert_eq!(parsed.compression, AlphaCompression::WebPLossless);
        assert_eq!(parsed.filter, AlphaFilter::Horizontal);
    }

    #[test]
    fn header_reserved_bits_rejected() {
        // Set reserved bits (bit 6)
        let result = AlphaHeader::parse(0x40);
        assert!(result.is_err());
    }

    // -- Filter round-trips --------------------------------------------------

    #[test]
    fn filter_none_roundtrip() {
        let original: Vec<u8> = (0..12).collect();
        let w = 4u32;
        let h = 3u32;

        let residuals = apply_inverse_filter(&original, w, h, AlphaFilter::None);
        assert_eq!(residuals, original);

        let mut reconstructed = residuals;
        apply_filter(&mut reconstructed, w, h, AlphaFilter::None);
        assert_eq!(reconstructed, original);
    }

    #[test]
    fn filter_horizontal_roundtrip() {
        let original: Vec<u8> = vec![
            10, 20, 30, 40,
            50, 60, 70, 80,
            90, 100, 110, 120,
        ];
        let w = 4u32;
        let h = 3u32;

        let residuals = apply_inverse_filter(&original, w, h, AlphaFilter::Horizontal);

        // Verify residuals manually for first row:
        // res[0] = 10 (first pixel, no prediction)
        // res[1] = 20 - 10 = 10
        // res[2] = 30 - 20 = 10
        // res[3] = 40 - 30 = 10
        assert_eq!(residuals[0], 10);
        assert_eq!(residuals[1], 10);
        assert_eq!(residuals[2], 10);
        assert_eq!(residuals[3], 10);

        let mut reconstructed = residuals;
        apply_filter(&mut reconstructed, w, h, AlphaFilter::Horizontal);
        assert_eq!(reconstructed, original);
    }

    #[test]
    fn filter_vertical_roundtrip() {
        let original: Vec<u8> = vec![
            10, 20, 30, 40,
            15, 25, 35, 45,
            20, 30, 40, 50,
        ];
        let w = 4u32;
        let h = 3u32;

        let residuals = apply_inverse_filter(&original, w, h, AlphaFilter::Vertical);

        // First row: unchanged
        assert_eq!(&residuals[0..4], &[10, 20, 30, 40]);
        // Second row: res[i] = orig[i] - orig[i-w]
        // 15-10=5, 25-20=5, 35-30=5, 45-40=5
        assert_eq!(&residuals[4..8], &[5, 5, 5, 5]);

        let mut reconstructed = residuals;
        apply_filter(&mut reconstructed, w, h, AlphaFilter::Vertical);
        assert_eq!(reconstructed, original);
    }

    #[test]
    fn filter_gradient_roundtrip() {
        let original: Vec<u8> = vec![
            100, 110, 120, 130,
            105, 115, 125, 135,
            110, 120, 130, 140,
        ];
        let w = 4u32;
        let h = 3u32;

        let residuals = apply_inverse_filter(&original, w, h, AlphaFilter::Gradient);
        let mut reconstructed = residuals;
        apply_filter(&mut reconstructed, w, h, AlphaFilter::Gradient);
        assert_eq!(reconstructed, original);
    }

    #[test]
    fn filter_gradient_known_vector() {
        // 2x2 image
        // [100, 150]
        // [120, 170]
        let original: Vec<u8> = vec![100, 150, 120, 170];
        let w = 2u32;
        let h = 2u32;

        let residuals = apply_inverse_filter(&original, w, h, AlphaFilter::Gradient);

        // Pixel (0,0): no prediction => 100
        assert_eq!(residuals[0], 100);
        // Pixel (1,0): horizontal prediction from left => 150 - 100 = 50
        assert_eq!(residuals[1], 50);
        // Pixel (0,1): vertical prediction from top => 120 - 100 = 20
        assert_eq!(residuals[2], 20);
        // Pixel (1,1): gradient = left(120) + top(150) - top_left(100) = 170 => 170 - 170 = 0
        assert_eq!(residuals[3], 0);

        let mut reconstructed = residuals;
        apply_filter(&mut reconstructed, w, h, AlphaFilter::Gradient);
        assert_eq!(reconstructed, original);
    }

    #[test]
    fn gradient_predict_clamp_high() {
        // left=200, top=200, top_left=0 => 200+200-0 = 400 => clamped to 255
        assert_eq!(gradient_predict(200, 200, 0), 255);
    }

    #[test]
    fn gradient_predict_clamp_low() {
        // left=0, top=0, top_left=200 => 0+0-200 = -200 => clamped to 0
        assert_eq!(gradient_predict(0, 0, 200), 0);
    }

    #[test]
    fn gradient_predict_normal() {
        assert_eq!(gradient_predict(100, 80, 60), 120);
    }

    // -- Encode / Decode round-trip ------------------------------------------

    #[test]
    fn encode_decode_roundtrip_uniform() {
        let w = 8u32;
        let h = 6u32;
        let alpha = vec![128u8; (w * h) as usize];

        let encoded = encode_alpha(&alpha, w, h).expect("encode should succeed");
        let decoded = decode_alpha(&encoded, w, h).expect("decode should succeed");
        assert_eq!(decoded, alpha);
    }

    #[test]
    fn encode_decode_roundtrip_gradient_data() {
        let w = 16u32;
        let h = 8u32;
        let mut alpha = vec![0u8; (w * h) as usize];
        for y in 0..h as usize {
            for x in 0..w as usize {
                alpha[y * w as usize + x] = ((x * 16 + y * 8) & 0xFF) as u8;
            }
        }

        let encoded = encode_alpha(&alpha, w, h).expect("encode should succeed");
        let decoded = decode_alpha(&encoded, w, h).expect("decode should succeed");
        assert_eq!(decoded, alpha);
    }

    #[test]
    fn encode_decode_roundtrip_random_like() {
        // Pseudo-random data generated from a simple LCG
        let w = 10u32;
        let h = 10u32;
        let mut alpha = vec![0u8; (w * h) as usize];
        let mut state: u32 = 0xDEAD_BEEF;
        for byte in alpha.iter_mut() {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *byte = (state >> 16) as u8;
        }

        let encoded = encode_alpha(&alpha, w, h).expect("encode should succeed");
        let decoded = decode_alpha(&encoded, w, h).expect("decode should succeed");
        assert_eq!(decoded, alpha);
    }

    #[test]
    fn encode_decode_roundtrip_single_pixel() {
        let alpha = vec![42u8];
        let encoded = encode_alpha(&alpha, 1, 1).expect("encode should succeed");
        let decoded = decode_alpha(&encoded, 1, 1).expect("decode should succeed");
        assert_eq!(decoded, alpha);
    }

    #[test]
    fn encode_decode_roundtrip_single_row() {
        let alpha: Vec<u8> = (0..=255).collect();
        let encoded = encode_alpha(&alpha, 256, 1).expect("encode should succeed");
        let decoded = decode_alpha(&encoded, 256, 1).expect("decode should succeed");
        assert_eq!(decoded, alpha);
    }

    #[test]
    fn encode_decode_roundtrip_single_column() {
        let alpha: Vec<u8> = (0..128).collect();
        let encoded = encode_alpha(&alpha, 1, 128).expect("encode should succeed");
        let decoded = decode_alpha(&encoded, 1, 128).expect("decode should succeed");
        assert_eq!(decoded, alpha);
    }

    // -- Edge cases / error paths --------------------------------------------

    #[test]
    fn decode_empty_chunk_is_error() {
        let result = decode_alpha(&[], 4, 4);
        assert!(result.is_err());
    }

    #[test]
    fn decode_truncated_payload_is_error() {
        // Header byte for no-compression, no-filter + only 3 bytes but we need 16
        let data = vec![0x00, 1, 2, 3];
        let result = decode_alpha(&data, 4, 4);
        assert!(result.is_err());
    }

    #[test]
    fn decode_vp8l_alpha_is_unsupported() {
        // Header byte with compression = 1 (WebP lossless)
        let data = vec![0x01, 0, 0, 0, 0];
        let result = decode_alpha(&data, 2, 2);
        assert!(result.is_err());
        let err_msg = format!("{}", result.expect_err("should be error"));
        assert!(err_msg.contains("not yet supported"));
    }

    #[test]
    fn encode_too_short_input_is_error() {
        let alpha = vec![0u8; 3]; // need 4 for 2x2
        let result = encode_alpha(&alpha, 2, 2);
        assert!(result.is_err());
    }

    #[test]
    fn encode_decode_zero_dimensions() {
        let alpha: Vec<u8> = Vec::new();
        let encoded = encode_alpha(&alpha, 0, 0).expect("encode 0x0 should succeed");
        let decoded = decode_alpha(&encoded, 0, 0).expect("decode 0x0 should succeed");
        assert!(decoded.is_empty());
    }

    #[test]
    fn overflow_dimensions_rejected() {
        let result = encode_alpha(&[0], u32::MAX, u32::MAX);
        assert!(result.is_err());
    }

    // -- Known ALPH chunk test vectors ---------------------------------------

    #[test]
    fn known_vector_no_filter_no_compression() {
        // Manually constructed ALPH chunk: header=0x00, followed by raw alpha
        let alpha_raw = vec![255, 128, 64, 0, 200, 100, 50, 25];
        let w = 4u32;
        let h = 2u32;

        let mut chunk = vec![0x00u8]; // no compression, no filter, no pre-processing
        chunk.extend_from_slice(&alpha_raw);

        let decoded = decode_alpha(&chunk, w, h).expect("decode should succeed");
        assert_eq!(decoded, alpha_raw);
    }

    #[test]
    fn known_vector_horizontal_filter() {
        // 4x2 image alpha: [10, 20, 30, 40, 50, 60, 70, 80]
        // Horizontal residuals:
        //   row0: [10, 10, 10, 10]  (20-10=10, 30-20=10, 40-30=10)
        //   row1: [50, 10, 10, 10]  (60-50=10, 70-60=10, 80-70=10)
        let expected = vec![10u8, 20, 30, 40, 50, 60, 70, 80];
        let residuals = vec![10u8, 10, 10, 10, 50, 10, 10, 10];

        // header byte: compression=0, filter=horizontal(1) => (1 << 2) = 0x04
        let mut chunk = vec![0x04u8];
        chunk.extend_from_slice(&residuals);

        let decoded = decode_alpha(&chunk, 4, 2).expect("decode should succeed");
        assert_eq!(decoded, expected);
    }

    #[test]
    fn known_vector_vertical_filter() {
        // 3x3 image alpha:
        // [10, 20, 30]
        // [15, 25, 35]
        // [20, 30, 40]
        // Vertical residuals:
        //   row0: [10, 20, 30] (first row unchanged)
        //   row1: [5, 5, 5]   (15-10, 25-20, 35-30)
        //   row2: [5, 5, 5]   (20-15, 30-25, 40-35)
        let expected = vec![10u8, 20, 30, 15, 25, 35, 20, 30, 40];
        let residuals = vec![10u8, 20, 30, 5, 5, 5, 5, 5, 5];

        // header byte: compression=0, filter=vertical(2) => (2 << 2) = 0x08
        let mut chunk = vec![0x08u8];
        chunk.extend_from_slice(&residuals);

        let decoded = decode_alpha(&chunk, 3, 3).expect("decode should succeed");
        assert_eq!(decoded, expected);
    }

    #[test]
    fn known_vector_gradient_filter() {
        // 2x2 image: [100, 150, 120, 170]
        // Gradient residuals (computed in filter_gradient_known_vector test):
        //   [100, 50, 20, 0]
        let expected = vec![100u8, 150, 120, 170];
        let residuals = vec![100u8, 50, 20, 0];

        // header byte: compression=0, filter=gradient(3) => (3 << 2) = 0x0C
        let mut chunk = vec![0x0Cu8];
        chunk.extend_from_slice(&residuals);

        let decoded = decode_alpha(&chunk, 2, 2).expect("decode should succeed");
        assert_eq!(decoded, expected);
    }

    // -- Filter selection heuristic ------------------------------------------

    #[test]
    fn select_best_filter_for_uniform_data() {
        // All same value => None, Horizontal, and Vertical filters produce
        // near-zero residuals. Gradient has slightly higher residuals for
        // the boundary pixels. The best filter should produce a score no
        // worse than the None filter.
        let data = vec![128u8; 64];
        let best = select_best_filter(&data, 8, 8);
        let best_score = score_filter(&data, 8, 8, best);
        let none_score = score_filter(&data, 8, 8, AlphaFilter::None);
        assert!(best_score <= none_score);
    }

    #[test]
    fn select_best_filter_for_horizontal_ramp() {
        // Each row is a horizontal ramp with constant step.
        // Gradient filter perfectly predicts this pattern (left + top - top_left
        // cancels out for linear data), so it ties with or beats horizontal.
        let mut data = vec![0u8; 64];
        for y in 0..8usize {
            for x in 0..8usize {
                data[y * 8 + x] = (x * 30) as u8;
            }
        }
        let best = select_best_filter(&data, 8, 8);
        let best_score = score_filter(&data, 8, 8, best);
        let horiz_score = score_filter(&data, 8, 8, AlphaFilter::Horizontal);
        // The chosen filter must be at least as good as horizontal
        assert!(best_score <= horiz_score);
    }

    #[test]
    fn select_best_filter_for_vertical_ramp() {
        // Each column is a vertical ramp => gradient filter also handles this well
        // since it subsumes both horizontal and vertical prediction.
        let mut data = vec![0u8; 64];
        for y in 0..8usize {
            for x in 0..8usize {
                data[y * 8 + x] = (y * 30) as u8;
            }
        }
        let best = select_best_filter(&data, 8, 8);
        let best_score = score_filter(&data, 8, 8, best);
        let vert_score = score_filter(&data, 8, 8, AlphaFilter::Vertical);
        // The chosen filter must be at least as good as vertical
        assert!(best_score <= vert_score);
    }

    // -- Wrapping arithmetic correctness -------------------------------------

    #[test]
    fn filter_horizontal_wrapping() {
        // Ensure wrapping works: 250, 10 => residual = 10 - 250 = -240 => 16 (wrapping)
        // Reconstruction: 16 + 250 = 266 => 10 (wrapping)
        let original = vec![250u8, 10];
        let residuals = apply_inverse_filter(&original, 2, 1, AlphaFilter::Horizontal);
        assert_eq!(residuals[0], 250);
        assert_eq!(residuals[1], 10u8.wrapping_sub(250));

        let mut reconstructed = residuals;
        apply_filter(&mut reconstructed, 2, 1, AlphaFilter::Horizontal);
        assert_eq!(reconstructed, original);
    }

    #[test]
    fn filter_vertical_wrapping() {
        let original = vec![5u8, 250]; // w=1, h=2
        let residuals = apply_inverse_filter(&original, 1, 2, AlphaFilter::Vertical);
        assert_eq!(residuals[0], 5);
        assert_eq!(residuals[1], 250u8.wrapping_sub(5));

        let mut reconstructed = residuals;
        apply_filter(&mut reconstructed, 1, 2, AlphaFilter::Vertical);
        assert_eq!(reconstructed, original);
    }

    // -- Larger stress test --------------------------------------------------

    #[test]
    fn encode_decode_large_plane() {
        let w = 320u32;
        let h = 240u32;
        let total = (w * h) as usize;
        let mut alpha = vec![0u8; total];
        let mut state: u64 = 42;
        for byte in alpha.iter_mut() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *byte = (state >> 33) as u8;
        }

        let encoded = encode_alpha(&alpha, w, h).expect("encode should succeed");
        // Encoded data = 1 header byte + total alpha bytes
        assert_eq!(encoded.len(), 1 + total);

        let decoded = decode_alpha(&encoded, w, h).expect("decode should succeed");
        assert_eq!(decoded, alpha);
    }

    // -- Header byte exhaustive coverage -------------------------------------

    #[test]
    fn all_valid_header_bytes_parse() {
        // Valid combinations: reserved=0, compression in {0,1},
        // filter in {0,1,2,3}, pre_processing in {0,1,2,3}
        for comp in 0..=1u8 {
            for filt in 0..=3u8 {
                for prep in 0..=3u8 {
                    let byte = comp | (filt << 2) | (prep << 4);
                    let hdr = AlphaHeader::parse(byte)
                        .unwrap_or_else(|e| panic!("valid byte {byte:#04x} failed: {e}"));
                    assert_eq!(hdr.to_byte(), byte);
                }
            }
        }
    }

    #[test]
    fn all_reserved_header_bytes_rejected() {
        for reserved in 1..=3u8 {
            let byte = reserved << 6;
            assert!(
                AlphaHeader::parse(byte).is_err(),
                "reserved={reserved} should be rejected"
            );
        }
    }
}
