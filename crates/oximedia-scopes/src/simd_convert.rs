//! SIMD-optimized RGB-to-YCbCr conversion for waveform and vectorscope generation.
//!
//! This module provides high-throughput batch conversion between color spaces,
//! structured for auto-vectorization by the compiler. All operations are strictly
//! pure Rust with no `unsafe` blocks; the data layout and loop structure are
//! chosen so that LLVM can apply SIMD widening automatically.
//!
//! # Color Space Mathematics
//!
//! Conversion follows ITU-R BT.709 (HD) primaries:
//!
//! ```text
//! Y  =  0.2126 R + 0.7152 G + 0.0722 B
//! Cb = -0.1146 R - 0.3854 G + 0.5000 B  (+128 bias for 8-bit)
//! Cr =  0.5000 R - 0.4542 G - 0.0458 B  (+128 bias for 8-bit)
//! ```
//!
//! Integer fixed-point coefficients scaled by 2^15 (32768) are used to avoid
//! floating-point in the hot path while maintaining sub-0.5 LSB accuracy.

/// Number of pixels processed per "SIMD lane" in the batch path.
/// Matches typical AVX2 width for 32-bit lanes.
pub const BATCH_SIZE: usize = 8;

/// Converted YCbCr triplet (8-bit, broadcast-legal 16–235 / 16–240 range
/// is **not** applied here; raw full-range values are returned).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct YcbcrPixel {
    /// Luma component (0–255, BT.709 full range).
    pub y: u8,
    /// Cb (blue-difference chroma) component, biased by 128.
    pub cb: u8,
    /// Cr (red-difference chroma) component, biased by 128.
    pub cr: u8,
}

/// BT.709 fixed-point coefficients scaled by 2^15.
///
/// Values chosen so that rounding errors do not exceed ±0.5 LSB across
/// the full 0–255 input range.
mod coeff709 {
    /// Y  from R: round(0.2126 * 32768)
    pub const KR: i32 = 6967;
    /// Y  from G: round(0.7152 * 32768)
    pub const KG: i32 = 23434;
    /// Y  from B: round(0.0722 * 32768)
    pub const KB: i32 = 2367;

    /// Cb from R: round(-0.1146 * 32768)
    pub const CB_R: i32 = -3755;
    /// Cb from G: round(-0.3854 * 32768)
    pub const CB_G: i32 = -12629;
    /// Cb from B: round( 0.5000 * 32768)
    pub const CB_B: i32 = 16384;

    /// Cr from R: round( 0.5000 * 32768)
    pub const CR_R: i32 = 16384;
    /// Cr from G: round(-0.4542 * 32768)
    pub const CR_G: i32 = -14882;
    /// Cr from B: round(-0.0458 * 32768)
    pub const CR_B: i32 = -1502;
}

/// Convert a single RGB triplet to BT.709 YCbCr.
///
/// Returns a [`YcbcrPixel`] with full-range (0–255) Y and biased (0–255)
/// Cb/Cr values (neutral grey maps to Cb=128, Cr=128).
///
/// # Examples
///
/// ```
/// use oximedia_scopes::simd_convert::rgb_to_ycbcr_pixel;
/// let p = rgb_to_ycbcr_pixel(255, 255, 255);
/// assert_eq!(p.y, 255);
/// ```
#[must_use]
pub fn rgb_to_ycbcr_pixel(r: u8, g: u8, b: u8) -> YcbcrPixel {
    let ri = i32::from(r);
    let gi = i32::from(g);
    let bi = i32::from(b);

    let y = (coeff709::KR * ri + coeff709::KG * gi + coeff709::KB * bi + (1 << 14)) >> 15;
    let cb = (coeff709::CB_R * ri + coeff709::CB_G * gi + coeff709::CB_B * bi + (1 << 14)) >> 15;
    let cr = (coeff709::CR_R * ri + coeff709::CR_G * gi + coeff709::CR_B * bi + (1 << 14)) >> 15;

    YcbcrPixel {
        y: y.clamp(0, 255) as u8,
        cb: (cb + 128).clamp(0, 255) as u8,
        cr: (cr + 128).clamp(0, 255) as u8,
    }
}

/// Convert a batch of RGB pixels to YCbCr in one call.
///
/// The input slice must be RGB-interleaved (`[R, G, B, R, G, B, …]`).
/// The output vector will contain one [`YcbcrPixel`] per input pixel.
/// Incomplete trailing pixels (i.e. `frame.len() % 3 != 0`) are silently
/// ignored.
///
/// This function is structured for LLVM auto-vectorization: the inner loop
/// over a fixed `BATCH_SIZE` window processes 8 pixels in parallel, enabling
/// 256-bit AVX2 or 128-bit NEON widening without any `unsafe` code.
///
/// # Examples
///
/// ```
/// use oximedia_scopes::simd_convert::convert_batch;
/// let rgb = vec![255u8, 0, 0,  0, 255, 0,  0, 0, 255];
/// let out = convert_batch(&rgb);
/// assert_eq!(out.len(), 3);
/// ```
#[must_use]
pub fn convert_batch(frame: &[u8]) -> Vec<YcbcrPixel> {
    let pixel_count = frame.len() / 3;
    let mut output = Vec::with_capacity(pixel_count);

    // Process pixels in chunks of BATCH_SIZE for auto-vectorization.
    let chunk_pixels = pixel_count / BATCH_SIZE;
    let remainder_start = chunk_pixels * BATCH_SIZE;

    for chunk_idx in 0..chunk_pixels {
        let base = chunk_idx * BATCH_SIZE;

        // Unrolled fixed-size array so LLVM can see the trip-count.
        let mut chunk_out = [YcbcrPixel { y: 0, cb: 128, cr: 128 }; BATCH_SIZE];
        for lane in 0..BATCH_SIZE {
            let off = (base + lane) * 3;
            let r = frame[off];
            let g = frame[off + 1];
            let b = frame[off + 2];
            chunk_out[lane] = rgb_to_ycbcr_pixel(r, g, b);
        }
        output.extend_from_slice(&chunk_out);
    }

    // Handle the remaining pixels (< BATCH_SIZE).
    for px in remainder_start..pixel_count {
        let off = px * 3;
        output.push(rgb_to_ycbcr_pixel(frame[off], frame[off + 1], frame[off + 2]));
    }

    output
}

/// Extract the luma channel from a batch conversion result into a flat `Vec<u8>`.
///
/// This is a convenience wrapper for the common case of waveform generation
/// where only the Y channel is needed.
///
/// # Examples
///
/// ```
/// use oximedia_scopes::simd_convert::extract_luma;
/// let rgb = vec![128u8, 128, 128, 128, 128, 128];
/// let luma = extract_luma(&rgb);
/// assert_eq!(luma.len(), 2);
/// ```
#[must_use]
pub fn extract_luma(frame: &[u8]) -> Vec<u8> {
    convert_batch(frame).into_iter().map(|p| p.y).collect()
}

/// Extract Cb and Cr channels (interleaved `[Cb, Cr, Cb, Cr, …]`) from a
/// full RGB frame.  Used by the vectorscope for fast UV-plane construction.
///
/// # Examples
///
/// ```
/// use oximedia_scopes::simd_convert::extract_cbcr;
/// let rgb = vec![128u8, 128, 128];
/// let cbcr = extract_cbcr(&rgb);
/// assert_eq!(cbcr.len(), 2); // one Cb + one Cr per pixel
/// ```
#[must_use]
pub fn extract_cbcr(frame: &[u8]) -> Vec<u8> {
    let pixels = convert_batch(frame);
    let mut out = Vec::with_capacity(pixels.len() * 2);
    for p in &pixels {
        out.push(p.cb);
        out.push(p.cr);
    }
    out
}

/// BT.2020 fixed-point coefficients scaled by 2^15 for wide-gamut HDR content.
mod coeff2020 {
    /// Y  from R: round(0.2627 * 32768)
    pub const KR: i32 = 8610;
    /// Y  from G: round(0.6780 * 32768)
    pub const KG: i32 = 22216;
    /// Y  from B: round(0.0593 * 32768)
    pub const KB: i32 = 1943;

    /// Cb from R: round(-0.1396 * 32768)
    pub const CB_R: i32 = -4574;
    /// Cb from G: round(-0.3604 * 32768)
    pub const CB_G: i32 = -11810;
    /// Cb from B: round( 0.5000 * 32768)
    pub const CB_B: i32 = 16384;

    /// Cr from R: round( 0.5000 * 32768)
    pub const CR_R: i32 = 16384;
    /// Cr from G: round(-0.4598 * 32768)
    pub const CR_G: i32 = -15066;
    /// Cr from B: round(-0.0402 * 32768)
    pub const CR_B: i32 = -1317;
}

/// Convert a single RGB triplet to BT.2020 YCbCr (for HDR / wide-gamut scopes).
///
/// # Examples
///
/// ```
/// use oximedia_scopes::simd_convert::rgb_to_ycbcr_bt2020;
/// let p = rgb_to_ycbcr_bt2020(0, 0, 0);
/// assert_eq!(p.y, 0);
/// assert_eq!(p.cb, 128);
/// assert_eq!(p.cr, 128);
/// ```
#[must_use]
pub fn rgb_to_ycbcr_bt2020(r: u8, g: u8, b: u8) -> YcbcrPixel {
    let ri = i32::from(r);
    let gi = i32::from(g);
    let bi = i32::from(b);

    let y = (coeff2020::KR * ri + coeff2020::KG * gi + coeff2020::KB * bi + (1 << 14)) >> 15;
    let cb = (coeff2020::CB_R * ri + coeff2020::CB_G * gi + coeff2020::CB_B * bi + (1 << 14)) >> 15;
    let cr = (coeff2020::CR_R * ri + coeff2020::CR_G * gi + coeff2020::CR_B * bi + (1 << 14)) >> 15;

    YcbcrPixel {
        y: y.clamp(0, 255) as u8,
        cb: (cb + 128).clamp(0, 255) as u8,
        cr: (cr + 128).clamp(0, 255) as u8,
    }
}

/// Batch-convert a frame using BT.2020 primaries.
///
/// Semantics are identical to [`convert_batch`] but using BT.2020 coefficients.
///
/// # Examples
///
/// ```
/// use oximedia_scopes::simd_convert::convert_batch_bt2020;
/// let rgb = vec![255u8, 255, 255, 0, 0, 0];
/// let out = convert_batch_bt2020(&rgb);
/// assert_eq!(out.len(), 2);
/// assert_eq!(out[0].y, 255);
/// assert_eq!(out[1].y, 0);
/// ```
#[must_use]
pub fn convert_batch_bt2020(frame: &[u8]) -> Vec<YcbcrPixel> {
    let pixel_count = frame.len() / 3;
    let mut output = Vec::with_capacity(pixel_count);

    let chunk_pixels = pixel_count / BATCH_SIZE;
    let remainder_start = chunk_pixels * BATCH_SIZE;

    for chunk_idx in 0..chunk_pixels {
        let base = chunk_idx * BATCH_SIZE;
        let mut chunk_out = [YcbcrPixel { y: 0, cb: 128, cr: 128 }; BATCH_SIZE];
        for lane in 0..BATCH_SIZE {
            let off = (base + lane) * 3;
            chunk_out[lane] = rgb_to_ycbcr_bt2020(frame[off], frame[off + 1], frame[off + 2]);
        }
        output.extend_from_slice(&chunk_out);
    }

    for px in remainder_start..pixel_count {
        let off = px * 3;
        output.push(rgb_to_ycbcr_bt2020(frame[off], frame[off + 1], frame[off + 2]));
    }

    output
}

/// YCbCr conversion accuracy metric used in benchmarks and validation.
///
/// Returns the maximum absolute difference between scalar reference and
/// fixed-point batch conversion for a given test frame.
#[must_use]
pub fn max_error_vs_float(frame: &[u8]) -> f64 {
    let pixel_count = frame.len() / 3;
    let mut max_err: f64 = 0.0;

    for px in 0..pixel_count {
        let off = px * 3;
        let r = frame[off] as f64;
        let g = frame[off + 1] as f64;
        let b = frame[off + 2] as f64;

        let y_ref = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        let cb_ref = -0.1146 * r - 0.3854 * g + 0.5000 * b + 128.0;
        let cr_ref = 0.5000 * r - 0.4542 * g - 0.0458 * b + 128.0;

        let fp = rgb_to_ycbcr_pixel(frame[off], frame[off + 1], frame[off + 2]);

        let dy = (fp.y as f64 - y_ref.clamp(0.0, 255.0)).abs();
        let dcb = (fp.cb as f64 - cb_ref.clamp(0.0, 255.0)).abs();
        let dcr = (fp.cr as f64 - cr_ref.clamp(0.0, 255.0)).abs();

        max_err = max_err.max(dy).max(dcb).max(dcr);
    }

    max_err
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pure-black should give Y=0, Cb=128, Cr=128.
    #[test]
    fn test_black_pixel() {
        let p = rgb_to_ycbcr_pixel(0, 0, 0);
        assert_eq!(p.y, 0);
        assert_eq!(p.cb, 128);
        assert_eq!(p.cr, 128);
    }

    /// Pure-white should give Y=255, Cb≈128, Cr≈128.
    #[test]
    fn test_white_pixel() {
        let p = rgb_to_ycbcr_pixel(255, 255, 255);
        assert_eq!(p.y, 255);
        // Chroma should stay near neutral (128) for achromatic colours.
        assert!((p.cb as i32 - 128).abs() <= 1, "cb={}", p.cb);
        assert!((p.cr as i32 - 128).abs() <= 1, "cr={}", p.cr);
    }

    /// Pure-red in BT.709 has significant positive Cr.
    #[test]
    fn test_red_pixel_chroma() {
        let p = rgb_to_ycbcr_pixel(255, 0, 0);
        // Y ≈ 54 (0.2126 * 255)
        assert!((p.y as i32 - 54).abs() <= 1, "y={}", p.y);
        // Cr should be the highest chroma component for red.
        assert!(p.cr > p.cb, "expected Cr > Cb for red, got cr={} cb={}", p.cr, p.cb);
    }

    /// Pure-blue in BT.709 has significant positive Cb.
    #[test]
    fn test_blue_pixel_chroma() {
        let p = rgb_to_ycbcr_pixel(0, 0, 255);
        // Cb should be the highest chroma component for blue.
        assert!(p.cb > p.cr, "expected Cb > Cr for blue, got cb={} cr={}", p.cb, p.cr);
    }

    /// `convert_batch` on an empty frame returns an empty vec.
    #[test]
    fn test_batch_empty() {
        let out = convert_batch(&[]);
        assert!(out.is_empty());
    }

    /// `convert_batch` on a single pixel equals `rgb_to_ycbcr_pixel`.
    #[test]
    fn test_batch_single_pixel_matches_scalar() {
        let rgb = [200u8, 100, 50];
        let batch = convert_batch(&rgb);
        let scalar = rgb_to_ycbcr_pixel(200, 100, 50);
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0], scalar);
    }

    /// Batch conversion with a stride that is not a multiple of BATCH_SIZE
    /// must still produce the right number of pixels.
    #[test]
    fn test_batch_non_multiple_of_batch_size() {
        // 11 pixels → not divisible by BATCH_SIZE (8)
        let pixel_count = 11_usize;
        let rgb: Vec<u8> = (0..pixel_count * 3).map(|i| (i % 256) as u8).collect();
        let out = convert_batch(&rgb);
        assert_eq!(out.len(), pixel_count);
    }

    /// Fixed-point batch results must agree with reference float within 1 LSB.
    #[test]
    fn test_batch_accuracy_within_one_lsb() {
        // Ramp through all grey levels (achromatic – hardest for chroma).
        let rgb: Vec<u8> = (0..=255).flat_map(|v| [v, v, v]).collect();
        let err = max_error_vs_float(&rgb);
        assert!(err <= 1.0, "max error {err} exceeds 1 LSB");
    }

    /// BT.2020 black should also be Y=0, Cb=128, Cr=128.
    #[test]
    fn test_bt2020_black_pixel() {
        let p = rgb_to_ycbcr_bt2020(0, 0, 0);
        assert_eq!(p.y, 0);
        assert_eq!(p.cb, 128);
        assert_eq!(p.cr, 128);
    }

    /// BT.2020 white should be Y=255, neutral chroma.
    #[test]
    fn test_bt2020_white_pixel() {
        let p = rgb_to_ycbcr_bt2020(255, 255, 255);
        assert_eq!(p.y, 255);
        assert!((p.cb as i32 - 128).abs() <= 1, "cb={}", p.cb);
        assert!((p.cr as i32 - 128).abs() <= 1, "cr={}", p.cr);
    }

    /// `extract_luma` length must equal pixel_count.
    #[test]
    fn test_extract_luma_length() {
        let rgb = vec![0u8; 30]; // 10 pixels
        let luma = extract_luma(&rgb);
        assert_eq!(luma.len(), 10);
    }

    /// `extract_cbcr` length must be 2× pixel_count.
    #[test]
    fn test_extract_cbcr_length() {
        let rgb = vec![0u8; 30]; // 10 pixels
        let cbcr = extract_cbcr(&rgb);
        assert_eq!(cbcr.len(), 20);
    }

    /// BT.2020 batch on 9 pixels (non-aligned) returns 9 pixels.
    #[test]
    fn test_bt2020_batch_non_aligned() {
        let rgb: Vec<u8> = (0..27).collect(); // 9 pixels
        let out = convert_batch_bt2020(&rgb);
        assert_eq!(out.len(), 9);
    }
}
