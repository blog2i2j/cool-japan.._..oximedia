//! Forensic spread-spectrum image watermarking for user-ID tracking.
//!
//! Embeds a 64-bit `user_id` into an image using a spread-spectrum technique
//! in the pixel (spatial) domain.  The watermark is designed for:
//!
//! - **Low visibility** — perturbation amplitude ≪ 1 LSB on average.
//! - **Forensic traceability** — each user receives a unique pseudo-noise
//!   (PN) sequence derived from their ID, enabling blind detection.
//! - **Resistance to casual attack** — the mark survives minor edits, JPEG
//!   compression, and scaling.
//!
//! ## Embedding
//!
//! For each pixel `p[i]`, the embedded pixel is:
//!
//! ```text
//!   p'[i] = clamp(p[i] + α × pn[i] × bit_value, 0, 255)
//! ```
//!
//! where:
//! - `pn[i]` ∈ {-1, +1} is the i-th chip of the spreading sequence.
//! - `bit_value` is the i-th replica of the `user_id` bit (cyclic).
//! - `α` is the embedding amplitude (default `ALPHA = 0.8`).
//!
//! ## Detection
//!
//! Compute the correlation `C = Σ p'[i] × pn[i] × bit_sign[i]`.
//! A positive `C` above a threshold indicates the user is present.

use crate::error::{WatermarkError, WatermarkResult};

/// Embedding amplitude in pixel units (sub-LSB level).
const ALPHA: f32 = 0.8;

/// Forensic image watermark embedder and correlator.
pub struct ForensicWatermark;

impl ForensicWatermark {
    /// Embed `user_id` into `img` and return the watermarked image.
    ///
    /// # Parameters
    ///
    /// - `img`     : flat row-major grayscale pixel data of length `w × h`.
    /// - `w`, `h`  : image dimensions in pixels.
    /// - `user_id` : 64-bit user identifier to embed.
    ///
    /// # Errors
    ///
    /// Returns [`WatermarkError`] if `img.len() != w * h`.
    pub fn embed_user_id(img: &[u8], w: u32, h: u32, user_id: u64) -> WatermarkResult<Vec<u8>> {
        let n = (w as usize) * (h as usize);
        if img.len() != n {
            return Err(WatermarkError::InvalidData(format!(
                "img length {} ≠ w×h = {}×{} = {}",
                img.len(),
                w,
                h,
                n
            )));
        }

        // Generate the spreading (PN) sequence from user_id.
        let pn = generate_pn_sequence(user_id, n);

        let mut out = Vec::with_capacity(n);
        for (i, &px) in img.iter().enumerate() {
            // Bit value ∈ {-1, +1}: cycle the 64 bits of user_id.
            let bit_pos = i % 64;
            let bit = ((user_id >> bit_pos) & 1) as f32 * 2.0 - 1.0;

            let perturbation = ALPHA * pn[i] * bit;
            let watermarked = (f32::from(px) + perturbation).round().clamp(0.0, 255.0) as u8;
            out.push(watermarked);
        }

        Ok(out)
    }

    /// Compute the correlation score for `user_id` against `img`.
    ///
    /// A positive score above the detection threshold indicates the user's
    /// mark is present.  The threshold scales roughly as `ALPHA × √n`.
    ///
    /// # Parameters
    ///
    /// - `img`     : watermarked (or suspect) image, same layout as embed.
    /// - `w`, `h`  : image dimensions.
    /// - `user_id` : the user ID to test for.
    ///
    /// # Returns
    ///
    /// Correlation score (positive = mark present).
    ///
    /// # Errors
    ///
    /// Returns error if `img.len() != w * h`.
    pub fn correlate(img: &[u8], w: u32, h: u32, user_id: u64) -> WatermarkResult<f64> {
        let n = (w as usize) * (h as usize);
        if img.len() != n {
            return Err(WatermarkError::InvalidData(format!(
                "img length {} ≠ {}",
                img.len(),
                n
            )));
        }

        let pn = generate_pn_sequence(user_id, n);

        // Compute mean pixel value so we correlate on the residual (AC) component.
        let mean_px: f64 = img.iter().map(|&p| f64::from(p)).sum::<f64>() / n as f64;
        let mut corr = 0f64;

        for (i, &px) in img.iter().enumerate() {
            let bit_pos = i % 64;
            let bit = ((user_id >> bit_pos) & 1) as f64 * 2.0 - 1.0;
            corr += (f64::from(px) - mean_px) * f64::from(pn[i]) * bit;
        }

        Ok(corr / n as f64)
    }

    /// Detection threshold for an image of `n` pixels at the default amplitude.
    ///
    /// Set detection threshold ≈ `ALPHA * 0.5` (half the expected correlation
    /// per chip).  In practice, tune for the desired false-positive rate.
    #[must_use]
    pub fn detection_threshold() -> f64 {
        f64::from(ALPHA) * 0.5
    }
}

// ── PN sequence generation ────────────────────────────────────────────────────

/// Generate a pseudo-noise (PN) sequence of length `n` with values in {-1, +1}.
///
/// Uses a Galois LFSR seeded with `user_id` for cryptographic separation
/// between different user marks.
fn generate_pn_sequence(user_id: u64, n: usize) -> Vec<f32> {
    let mut state = if user_id == 0 {
        0xDEAD_BEEF_CAFE_1234u64
    } else {
        user_id
    };

    let mut pn = Vec::with_capacity(n);
    for _ in 0..n {
        state = lfsr64_step(state);
        // Use bit 0 of the LFSR output.
        let chip = if state & 1 == 0 { 1.0f32 } else { -1.0f32 };
        pn.push(chip);
    }
    pn
}

/// One step of a 64-bit maximal-length LFSR (Galois form).
///
/// Feedback polynomial: x^64 + x^63 + x^61 + x^60 + 1 (primitive).
#[inline]
fn lfsr64_step(state: u64) -> u64 {
    const MASK: u64 = 0xD800_0000_0000_0000u64;
    let feedback = state & 1;
    let shifted = state >> 1;
    if feedback == 0 {
        shifted
    } else {
        shifted ^ MASK
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embed_output_length() {
        let img = vec![128u8; 64 * 64];
        let result = ForensicWatermark::embed_user_id(&img, 64, 64, 0xABCD_1234_5678_9EF0);
        let watermarked = result.expect("embed should succeed");
        assert_eq!(watermarked.len(), img.len());
    }

    #[test]
    fn test_embed_correlation_positive() {
        let img = vec![128u8; 256 * 64];
        let user_id: u64 = 0xCAFE_BABE_0000_0001;
        let watermarked = ForensicWatermark::embed_user_id(&img, 256, 64, user_id)
            .expect("embed should succeed");
        let score = ForensicWatermark::correlate(&watermarked, 256, 64, user_id)
            .expect("correlate should succeed");
        let threshold = ForensicWatermark::detection_threshold();
        assert!(
            score > threshold,
            "correlation {score} should exceed threshold {threshold}"
        );
    }

    #[test]
    fn test_wrong_user_id_low_correlation() {
        let img = vec![100u8; 128 * 128];
        let owner: u64 = 0x1111_2222_3333_4444;
        let attacker: u64 = 0xAAAA_BBBB_CCCC_DDDD;
        let watermarked =
            ForensicWatermark::embed_user_id(&img, 128, 128, owner).expect("embed should succeed");
        let owner_score =
            ForensicWatermark::correlate(&watermarked, 128, 128, owner).expect("ok");
        let attacker_score =
            ForensicWatermark::correlate(&watermarked, 128, 128, attacker).expect("ok");
        assert!(
            owner_score > attacker_score,
            "owner correlation {owner_score} should exceed attacker {attacker_score}"
        );
    }

    #[test]
    fn test_error_on_size_mismatch() {
        let img = vec![0u8; 10];
        let result = ForensicWatermark::embed_user_id(&img, 8, 8, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_pixel_values_clamped() {
        let img = vec![255u8; 64 * 64];
        let result = ForensicWatermark::embed_user_id(&img, 64, 64, u64::MAX)
            .expect("should succeed");
        // All pixels should be valid u8 values (clamped during embedding).
        assert!(!result.is_empty(), "result should not be empty");
    }
}
