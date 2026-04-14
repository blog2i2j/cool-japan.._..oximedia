//! Compression artifact analysis.
//!
//! Provides a blockiness measure for JPEG-compressed or similarly block-DCT-
//! encoded images.  The *blockiness score* is defined as the mean absolute
//! difference across 8×8 block boundaries relative to the mean absolute
//! difference within blocks, normalised to `[0, 1]`.
//!
//! High blockiness (score near 1.0) indicates strong 8×8 block-boundary
//! artefacts consistent with heavy JPEG compression.  A score near 0.0 means
//! either a high-quality / uncompressed image or one without visible block
//! boundaries.
//!
//! # Example
//!
//! ```
//! use oximedia_forensics::artifact_analysis::CompressionArtifactAnalyzer;
//!
//! // Solid-colour frame — no blockiness
//! let frame = vec![128u8; 64 * 64 * 3];
//! let score = CompressionArtifactAnalyzer::blockiness(&frame, 64, 64);
//! assert!(score < 0.1, "Solid frame should have very low blockiness");
//! ```

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

const BLOCK_SIZE: usize = 8;

/// Analyses compression artifacts in raw RGB/grayscale frame data.
pub struct CompressionArtifactAnalyzer;

impl CompressionArtifactAnalyzer {
    /// Compute the *blockiness score* for a raw 8-bit frame.
    ///
    /// The frame is interpreted as packed RGB bytes (3 bytes per pixel, row-major).
    /// Luminance is approximated as the average of the three channels.
    ///
    /// The metric measures how much larger the pixel differences are *at* 8-pixel
    /// block boundaries (horizontal and vertical) compared to differences at
    /// non-boundary pixel transitions.  A score close to 1.0 signals heavy DCT
    /// blocking artefacts.
    ///
    /// # Arguments
    ///
    /// * `frame` — Packed RGB bytes, `w * h * 3` elements.
    /// * `w`     — Frame width in pixels.
    /// * `h`     — Frame height in pixels.
    ///
    /// # Returns
    ///
    /// Blockiness score in `[0, 1]`.  Returns `0.0` for frames too small to
    /// contain a complete 8×8 block or for frames where `frame.len() < w * h * 3`.
    #[must_use]
    pub fn blockiness(frame: &[u8], w: u32, h: u32) -> f32 {
        let wu = w as usize;
        let hu = h as usize;
        if wu < BLOCK_SIZE * 2 || hu < BLOCK_SIZE * 2 {
            return 0.0;
        }
        let expected = wu * hu * 3;
        if frame.len() < expected {
            return 0.0;
        }

        // Build luminance plane
        let luma: Vec<f32> = frame
            .chunks_exact(3)
            .map(|px| (f32::from(px[0]) + f32::from(px[1]) + f32::from(px[2])) / 3.0)
            .collect();

        let w = wu;
        let h = hu;

        let mut boundary_diff_sum = 0.0_f64;
        let mut boundary_count = 0u64;
        let mut interior_diff_sum = 0.0_f64;
        let mut interior_count = 0u64;

        // Horizontal transitions (across columns)
        for row in 0..h {
            for col in 1..w {
                let diff = (luma[row * w + col] - luma[row * w + col - 1]).abs() as f64;
                if col % BLOCK_SIZE == 0 {
                    boundary_diff_sum += diff;
                    boundary_count += 1;
                } else {
                    interior_diff_sum += diff;
                    interior_count += 1;
                }
            }
        }

        // Vertical transitions (across rows)
        for row in 1..h {
            for col in 0..w {
                let diff = (luma[row * w + col] - luma[(row - 1) * w + col]).abs() as f64;
                if row % BLOCK_SIZE == 0 {
                    boundary_diff_sum += diff;
                    boundary_count += 1;
                } else {
                    interior_diff_sum += diff;
                    interior_count += 1;
                }
            }
        }

        if boundary_count == 0 {
            return 0.0;
        }

        let mean_boundary = boundary_diff_sum / boundary_count as f64;
        let mean_interior = if interior_count > 0 {
            interior_diff_sum / interior_count as f64
        } else {
            // No interior transitions — treat as if interior == boundary
            mean_boundary
        };

        // Score: how much higher are boundary differences compared to interior?
        let denom = mean_boundary + mean_interior;
        if denom < 1e-10 {
            return 0.0; // uniform image
        }

        // Ratio in [0, 1]: 0 = boundary == interior, 1 = all differences at boundaries
        let raw = ((mean_boundary - mean_interior) / denom).clamp(-1.0, 1.0);
        // Map [-1, 1] → [0, 1] with negative saturated to 0
        raw.max(0.0) as f32
    }

    /// Detect whether the frame likely exhibits double-JPEG compression.
    ///
    /// Double JPEG manifests as periodic peaks in the 8×8 block-boundary
    /// difference histogram.  This simplified estimator checks whether
    /// the variance of boundary-column luminance differences significantly
    /// exceeds the variance of interior-column differences (ratio > 2.0).
    ///
    /// Returns a suspicion score in `[0, 1]`.
    #[must_use]
    pub fn double_jpeg_suspicion(frame: &[u8], w: u32, h: u32) -> f32 {
        let wu = w as usize;
        let hu = h as usize;
        if wu < BLOCK_SIZE * 2 || hu < BLOCK_SIZE * 2 {
            return 0.0;
        }
        let expected = wu * hu * 3;
        if frame.len() < expected {
            return 0.0;
        }

        let luma: Vec<f32> = frame
            .chunks_exact(3)
            .map(|px| (f32::from(px[0]) + f32::from(px[1]) + f32::from(px[2])) / 3.0)
            .collect();

        let w = wu;
        let h = hu;

        let mut boundary_diffs = Vec::new();
        let mut interior_diffs = Vec::new();

        for row in 0..h {
            for col in 1..w {
                let diff = (luma[row * w + col] - luma[row * w + col - 1]).abs();
                if col % BLOCK_SIZE == 0 {
                    boundary_diffs.push(diff);
                } else {
                    interior_diffs.push(diff);
                }
            }
        }

        if boundary_diffs.is_empty() || interior_diffs.is_empty() {
            return 0.0;
        }

        let var_boundary = variance(&boundary_diffs);
        let var_interior = variance(&interior_diffs);

        if var_interior < 1e-9 {
            return if var_boundary > 1e-9 { 1.0 } else { 0.0 };
        }

        let ratio = var_boundary / var_interior;
        // Saturate: ratio ≥ 4.0 → score = 1.0
        (ratio / 4.0).min(1.0) as f32
    }
}

/// Compute variance of a slice of f32 values.
fn variance(values: &[f32]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let n = values.len() as f64;
    let mean = values.iter().map(|&v| v as f64).sum::<f64>() / n;
    values
        .iter()
        .map(|&v| {
            let d = v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_frame(w: u32, h: u32, val: u8) -> Vec<u8> {
        vec![val; (w * h * 3) as usize]
    }

    fn blocked_frame(w: u32, h: u32) -> Vec<u8> {
        // Build a frame where every 8th column is very bright and others are dark,
        // simulating strong block-boundary discontinuities.
        let mut data = vec![0u8; (w * h * 3) as usize];
        for row in 0..(h as usize) {
            for col in 0..(w as usize) {
                let lum: u8 = if col % 8 == 0 { 255 } else { 0 };
                let idx = (row * w as usize + col) * 3;
                data[idx] = lum;
                data[idx + 1] = lum;
                data[idx + 2] = lum;
            }
        }
        data
    }

    // ── blockiness ────────────────────────────────────────────────────────────

    #[test]
    fn test_blockiness_solid_frame_near_zero() {
        let frame = solid_frame(64, 64, 128);
        let score = CompressionArtifactAnalyzer::blockiness(&frame, 64, 64);
        assert!(score < 0.05, "Solid frame should have near-zero blockiness: {score}");
    }

    #[test]
    fn test_blockiness_returns_zero_for_small_frame() {
        let frame = solid_frame(4, 4, 0);
        let score = CompressionArtifactAnalyzer::blockiness(&frame, 4, 4);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_blockiness_returns_zero_for_empty_frame() {
        let score = CompressionArtifactAnalyzer::blockiness(&[], 64, 64);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_blockiness_blocked_frame_higher_than_solid() {
        let solid = solid_frame(64, 64, 128);
        let blocked = blocked_frame(64, 64);
        let score_solid = CompressionArtifactAnalyzer::blockiness(&solid, 64, 64);
        let score_blocked = CompressionArtifactAnalyzer::blockiness(&blocked, 64, 64);
        assert!(
            score_blocked > score_solid,
            "Blocked frame should score higher than solid frame: {score_blocked} vs {score_solid}"
        );
    }

    #[test]
    fn test_blockiness_in_range() {
        let frame = blocked_frame(64, 64);
        let score = CompressionArtifactAnalyzer::blockiness(&frame, 64, 64);
        assert!(score >= 0.0 && score <= 1.0, "Score out of [0,1]: {score}");
    }

    #[test]
    fn test_blockiness_short_buffer_returns_zero() {
        // buffer too short
        let frame = vec![0u8; 10];
        let score = CompressionArtifactAnalyzer::blockiness(&frame, 64, 64);
        assert_eq!(score, 0.0);
    }

    // ── double_jpeg_suspicion ─────────────────────────────────────────────────

    #[test]
    fn test_double_jpeg_solid_frame_near_zero() {
        let frame = solid_frame(64, 64, 200);
        let score = CompressionArtifactAnalyzer::double_jpeg_suspicion(&frame, 64, 64);
        assert!(score < 0.1, "Solid frame double-JPEG suspicion should be low: {score}");
    }

    #[test]
    fn test_double_jpeg_in_range() {
        let frame = blocked_frame(64, 64);
        let score = CompressionArtifactAnalyzer::double_jpeg_suspicion(&frame, 64, 64);
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_double_jpeg_returns_zero_for_small_frame() {
        let frame = solid_frame(4, 4, 128);
        let score = CompressionArtifactAnalyzer::double_jpeg_suspicion(&frame, 4, 4);
        assert_eq!(score, 0.0);
    }
}
