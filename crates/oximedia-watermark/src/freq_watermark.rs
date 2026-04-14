//! Frequency-domain image watermarking using DCT coefficient modulation.
//!
//! Embeds a 64-bit payload in the mid-frequency DCT coefficients of 8×8
//! image blocks.  Each bit is spread across multiple coefficients for
//! robustness, using Quantization Index Modulation (QIM).
//!
//! ## Algorithm
//!
//! 1. Interpret the raw image bytes as 8-bit luma (Y) samples.
//! 2. Divide the luma plane into non-overlapping 8×8 blocks.
//! 3. For each block compute the 2D DCT-II of the 64 coefficients.
//! 4. Assign payload bits to mid-frequency zigzag positions (indices 10–53).
//! 5. Quantise the chosen coefficient with a step Δ; set it to the
//!    nearest even multiple (bit=0) or odd multiple (bit=1).
//! 6. Compute the inverse 2D DCT-III and write back the block.
//!
//! Extraction performs steps 1–4 then reads the quantisation parity.

use crate::error::{WatermarkError, WatermarkResult};

// ── DCT helpers ──────────────────────────────────────────────────────────────

/// Compute the orthonormal 1D DCT-II of 8 samples (in-place).
fn dct8(x: &mut [f32; 8]) {
    let pi = std::f32::consts::PI;
    let n = 8usize;
    let mut out = [0f32; 8];
    for k in 0..n {
        let mut sum = 0f32;
        for i in 0..n {
            sum += x[i] * ((pi * (2 * i + 1) as f32 * k as f32) / (2 * n) as f32).cos();
        }
        let alpha = if k == 0 {
            (1.0 / n as f32).sqrt()
        } else {
            (2.0 / n as f32).sqrt()
        };
        out[k] = alpha * sum;
    }
    x.copy_from_slice(&out);
}

/// Compute the orthonormal 1D DCT-III (inverse of orthonormal DCT-II) of 8 samples (in-place).
fn idct8(x: &mut [f32; 8]) {
    let pi = std::f32::consts::PI;
    let n = 8usize;
    let mut out = [0f32; 8];
    for i in 0..n {
        let mut sum = 0f32;
        for k in 0..n {
            let alpha = if k == 0 {
                (1.0 / n as f32).sqrt()
            } else {
                (2.0 / n as f32).sqrt()
            };
            sum += alpha * x[k] * ((pi * (2 * i + 1) as f32 * k as f32) / (2 * n) as f32).cos();
        }
        out[i] = sum;
    }
    x.copy_from_slice(&out);
}

/// 2D DCT-II of an 8×8 block (row-major, in-place).
fn dct2d(block: &mut [f32; 64]) {
    // Row transforms
    for r in 0..8usize {
        let mut row = [0f32; 8];
        row.copy_from_slice(&block[r * 8..(r * 8 + 8)]);
        dct8(&mut row);
        block[r * 8..r * 8 + 8].copy_from_slice(&row);
    }
    // Column transforms
    for c in 0..8usize {
        let mut col = [0f32; 8];
        for r in 0..8usize {
            col[r] = block[r * 8 + c];
        }
        dct8(&mut col);
        for r in 0..8usize {
            block[r * 8 + c] = col[r];
        }
    }
}

/// 2D DCT-III (inverse) of an 8×8 block (row-major, in-place).
fn idct2d(block: &mut [f32; 64]) {
    // Row inverse
    for r in 0..8usize {
        let mut row = [0f32; 8];
        row.copy_from_slice(&block[r * 8..(r * 8 + 8)]);
        idct8(&mut row);
        block[r * 8..r * 8 + 8].copy_from_slice(&row);
    }
    // Column inverse
    for c in 0..8usize {
        let mut col = [0f32; 8];
        for r in 0..8usize {
            col[r] = block[r * 8 + c];
        }
        idct8(&mut col);
        for r in 0..8usize {
            block[r * 8 + c] = col[r];
        }
    }
}

// ── Zigzag table (standard JPEG ordering) ────────────────────────────────────

/// Standard 8×8 zigzag scan order — maps zigzag position → (row, col).
const ZIGZAG: [(usize, usize); 64] = [
    (0, 0),
    (0, 1),
    (1, 0),
    (2, 0),
    (1, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (2, 1),
    (3, 0),
    (4, 0),
    (3, 1),
    (2, 2),
    (1, 3),
    (0, 4),
    (0, 5),
    (1, 4),
    (2, 3),
    (3, 2),
    (4, 1),
    (5, 0),
    (6, 0),
    (5, 1),
    (4, 2),
    (3, 3),
    (2, 4),
    (1, 5),
    (0, 6),
    (0, 7),
    (1, 6),
    (2, 5),
    (3, 4),
    (4, 3),
    (5, 2),
    (6, 1),
    (7, 0),
    (7, 1),
    (6, 2),
    (5, 3),
    (4, 4),
    (3, 5),
    (2, 6),
    (1, 7),
    (2, 7),
    (3, 6),
    (4, 5),
    (5, 4),
    (6, 3),
    (7, 2),
    (7, 3),
    (6, 4),
    (5, 5),
    (4, 6),
    (3, 7),
    (4, 7),
    (5, 6),
    (6, 5),
    (7, 4),
    (7, 5),
    (6, 6),
    (5, 7),
    (6, 7),
    (7, 6),
    (7, 7),
];

/// Mid-frequency zigzag indices used for watermark embedding (positions 10–53,
/// skipping DC and lowest-frequency coefficients as well as the very high
/// frequencies that are easily destroyed by compression).
const MID_FREQ_START: usize = 10;
const MID_FREQ_END: usize = 54; // exclusive

/// QIM quantisation step size.  Larger = more robust, less transparent.
const DELTA: f32 = 8.0;

/// Number of coefficients used per bit (spreading factor for reliability).
const SPREAD: usize = 2;

// ── Public API ────────────────────────────────────────────────────────────────

/// Frequency-domain image watermark embedder and extractor.
///
/// Works on raw 8-bit grayscale (Y) images supplied as flat byte slices.
/// For colour images, apply to the luma channel only.
pub struct FreqWatermark;

impl FreqWatermark {
    /// Embed a 64-bit `payload` into `img` and return the watermarked image.
    ///
    /// # Parameters
    ///
    /// - `img`   : flat row-major grayscale pixel data (length must be `w * h`).
    /// - `w`, `h`: image dimensions in pixels.
    /// - `payload`: the 64-bit value to embed.
    ///
    /// # Errors
    ///
    /// Returns [`WatermarkError`] if the image is too small to hold 64 bits
    /// or if `img.len() != w * h`.
    pub fn embed(img: &[u8], w: u32, h: u32, payload: u64) -> WatermarkResult<Vec<u8>> {
        let (w, h) = (w as usize, h as usize);
        if img.len() != w * h {
            return Err(WatermarkError::InvalidData(format!(
                "img length {} does not match {}×{}={}",
                img.len(),
                w,
                h,
                w * h
            )));
        }
        let blocks_x = w / 8;
        let blocks_y = h / 8;
        let capacity_bits = Self::capacity_bits(blocks_x * blocks_y);
        if capacity_bits < 64 {
            return Err(WatermarkError::InsufficientCapacity {
                needed: 64,
                have: capacity_bits,
            });
        }

        let mut out = img.to_vec();

        // Iterate over 8×8 blocks in raster order.
        let mut bit_idx = 0usize; // which payload bit we are embedding
        'outer: for by in 0..blocks_y {
            for bx in 0..blocks_x {
                // Extract block into f32 buffer.
                let mut block = [0f32; 64];
                for r in 0..8usize {
                    for c in 0..8usize {
                        block[r * 8 + c] =
                            f32::from(out[(by * 8 + r) * w + (bx * 8 + c)]);
                    }
                }

                dct2d(&mut block);

                // Embed up to SPREAD bits into this block.
                for s in 0..SPREAD {
                    if bit_idx >= 64 {
                        break;
                    }
                    let zz_idx = MID_FREQ_START + s;
                    if zz_idx >= MID_FREQ_END {
                        break;
                    }
                    let (r, c) = ZIGZAG[zz_idx];
                    let coeff = block[r * 8 + c];

                    let bit = (payload >> (63 - bit_idx)) & 1;
                    // QIM: round to nearest even (bit=0) or odd (bit=1) multiple of DELTA/2.
                    let q = (coeff / (DELTA / 2.0)).round() as i64;
                    let q_mod = q.rem_euclid(2) as u64;
                    let q_adj = if bit != q_mod { q + 1 } else { q };
                    block[r * 8 + c] = q_adj as f32 * (DELTA / 2.0);

                    bit_idx += 1;
                }

                idct2d(&mut block);

                // Write back with clamping.
                for r in 0..8usize {
                    for c in 0..8usize {
                        let v = block[r * 8 + c].round().clamp(0.0, 255.0) as u8;
                        out[(by * 8 + r) * w + (bx * 8 + c)] = v;
                    }
                }

                if bit_idx >= 64 {
                    break 'outer;
                }
            }
        }

        Ok(out)
    }

    /// Extract a 64-bit payload previously embedded by [`FreqWatermark::embed`].
    ///
    /// # Errors
    ///
    /// Returns [`WatermarkError`] if the image is too small to contain 64
    /// embedded bits or if `img.len() != w * h`.
    pub fn extract(img: &[u8], w: u32, h: u32) -> WatermarkResult<u64> {
        let (w, h) = (w as usize, h as usize);
        if img.len() != w * h {
            return Err(WatermarkError::InvalidData(format!(
                "img length {} does not match {}×{}={}",
                img.len(),
                w,
                h,
                w * h
            )));
        }
        let blocks_x = w / 8;
        let blocks_y = h / 8;
        let capacity_bits = Self::capacity_bits(blocks_x * blocks_y);
        if capacity_bits < 64 {
            return Err(WatermarkError::InsufficientCapacity {
                needed: 64,
                have: capacity_bits,
            });
        }

        let mut payload = 0u64;
        let mut bit_idx = 0usize;

        'outer: for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let mut block = [0f32; 64];
                for r in 0..8usize {
                    for c in 0..8usize {
                        block[r * 8 + c] =
                            f32::from(img[(by * 8 + r) * w + (bx * 8 + c)]);
                    }
                }

                dct2d(&mut block);

                for s in 0..SPREAD {
                    if bit_idx >= 64 {
                        break;
                    }
                    let zz_idx = MID_FREQ_START + s;
                    if zz_idx >= MID_FREQ_END {
                        break;
                    }
                    let (r, c) = ZIGZAG[zz_idx];
                    let coeff = block[r * 8 + c];
                    let q = (coeff / (DELTA / 2.0)).round() as i64;
                    let bit = (q.rem_euclid(2)) as u64;
                    payload |= bit << (63 - bit_idx);
                    bit_idx += 1;
                }

                if bit_idx >= 64 {
                    break 'outer;
                }
            }
        }

        Ok(payload)
    }

    /// Number of payload bits the image can hold given `n_blocks` 8×8 blocks.
    #[inline]
    fn capacity_bits(n_blocks: usize) -> usize {
        n_blocks * SPREAD
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_image(w: usize, h: usize, fill: u8) -> Vec<u8> {
        vec![fill; w * h]
    }

    #[test]
    fn test_embed_extract_roundtrip_zeros() {
        let (w, h) = (64usize, 64usize);
        let img = make_image(w, h, 128);
        let payload: u64 = 0xDEAD_BEEF_CAFE_1234;
        let watermarked =
            FreqWatermark::embed(&img, w as u32, h as u32, payload).expect("embed should succeed");
        let extracted =
            FreqWatermark::extract(&watermarked, w as u32, h as u32).expect("extract should succeed");
        assert_eq!(extracted, payload, "round-trip failed");
    }

    #[test]
    fn test_embed_extract_natural_image() {
        // Simulate a natural image with varying pixel values.
        let w = 128usize;
        let h = 64usize;
        let img: Vec<u8> = (0..w * h)
            .map(|i| ((i * 37 + 13) % 256) as u8)
            .collect();
        let payload: u64 = 0x0102_0304_0506_0708;
        let watermarked =
            FreqWatermark::embed(&img, w as u32, h as u32, payload).expect("embed should succeed");
        let extracted =
            FreqWatermark::extract(&watermarked, w as u32, h as u32).expect("extract should succeed");
        assert_eq!(extracted, payload);
    }

    #[test]
    fn test_embed_too_small_returns_error() {
        // 8×8 image gives 1 block × 2 bits = 2 bits < 64 — should fail.
        let img = make_image(8, 8, 128);
        let result = FreqWatermark::embed(&img, 8, 8, 0xFFFF_FFFF_FFFF_FFFF);
        assert!(result.is_err(), "should fail when image too small");
    }

    #[test]
    fn test_extract_mismatched_len_returns_error() {
        let img = vec![0u8; 10]; // deliberate mismatch
        let result = FreqWatermark::extract(&img, 8, 8);
        assert!(result.is_err());
    }
}
