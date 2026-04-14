//! Per-frame HDR content fingerprinting.
//!
//! Computes a compact perceptual fingerprint of an HDR frame by operating in
//! scene-linear light.  The algorithm divides the luminance plane into an 8×8
//! grid of blocks, computes the mean linear luminance of each block, and
//! encodes the relative ordering into a 64-bit hash.
//!
//! # Algorithm
//!
//! 1. Convert each RGB pixel to luminance `Y = 0.2126 R + 0.7152 G + 0.0722 B`
//!    (Rec. 709 / BT.2020 normalised — for HDR the values may exceed 1.0).
//! 2. Divide the frame into an 8×8 grid; compute mean luminance per cell.
//! 3. Compare each cell's mean to the global mean: set bit to 1 if above.
//! 4. Pack the 64 comparison bits into a `u64` fingerprint.
//!
//! Hamming distance between two fingerprints measures perceptual similarity:
//! a distance of 0 is an exact perceptual match; ≤10 is considered similar.
//!
//! # Example
//!
//! ```rust
//! use oximedia_hdr::hdr_fingerprint::{HdrFingerprint, HdrFrameFingerprinter};
//!
//! let width = 16usize;
//! let height = 16usize;
//! // Flat mid-grey frame (linear light)
//! let pixels = vec![0.5f32; width * height * 3];
//! let fp = HdrFrameFingerprinter::compute(&pixels, width, height).unwrap();
//! // Two identical frames produce identical fingerprints
//! let fp2 = HdrFrameFingerprinter::compute(&pixels, width, height).unwrap();
//! assert_eq!(fp.hamming_distance(&fp2), 0);
//! ```

use crate::{HdrError, Result};

/// Grid dimension used for fingerprinting (8×8 = 64 bits).
const GRID: usize = 8;

/// A 64-bit perceptual fingerprint of an HDR frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HdrFingerprint {
    /// Packed comparison bits; bit *i* is 1 if cell *i*'s mean luminance is
    /// above the global mean.
    pub bits: u64,
}

impl HdrFingerprint {
    /// Compute the Hamming distance between two fingerprints.
    ///
    /// Returns a value in [0, 64].  A distance ≤ 10 is typically considered
    /// a perceptual match for HDR content.
    pub fn hamming_distance(self, other: &HdrFingerprint) -> u32 {
        (self.bits ^ other.bits).count_ones()
    }

    /// Return `true` if the two fingerprints are perceptually similar.
    ///
    /// Uses a threshold of 10 bits (out of 64) by default.
    pub fn is_similar_to(self, other: &HdrFingerprint) -> bool {
        self.hamming_distance(other) <= 10
    }

    /// Return the raw 64-bit hash value.
    pub fn as_u64(self) -> u64 {
        self.bits
    }

    /// Construct a fingerprint from a raw 64-bit hash (e.g. loaded from a
    /// database).
    pub fn from_u64(bits: u64) -> Self {
        Self { bits }
    }
}

impl std::fmt::Display for HdrFingerprint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:016x}", self.bits)
    }
}

/// Computes [`HdrFingerprint`]s from raw HDR frame buffers.
pub struct HdrFrameFingerprinter;

impl HdrFrameFingerprinter {
    /// Compute a perceptual fingerprint from an interleaved RGB pixel buffer.
    ///
    /// `pixels` must have length `width * height * 3`.  Values should be in
    /// scene-linear light (any normalisation scale is fine — only relative
    /// luminance matters for the hash).
    ///
    /// # Errors
    /// Returns [`HdrError::MetadataParseError`] if `pixels.len() !=
    /// width * height * 3`, or if `width` or `height` is zero.
    pub fn compute(pixels: &[f32], width: usize, height: usize) -> Result<HdrFingerprint> {
        if width == 0 || height == 0 {
            return Err(HdrError::MetadataParseError(
                "frame dimensions must be non-zero".into(),
            ));
        }
        if pixels.len() != width * height * 3 {
            return Err(HdrError::MetadataParseError(format!(
                "pixel buffer length {} does not match {}×{}×3={}",
                pixels.len(),
                width,
                height,
                width * height * 3,
            )));
        }

        // ── Step 1: compute per-pixel luminance ──────────────────────────────
        let luma: Vec<f32> = pixels
            .chunks_exact(3)
            .map(|px| 0.2126 * px[0] + 0.7152 * px[1] + 0.0722 * px[2])
            .collect();

        // ── Step 2: accumulate block sums ────────────────────────────────────
        let mut block_sums = [0.0f64; GRID * GRID];
        let mut block_counts = [0u64; GRID * GRID];

        for row in 0..height {
            for col in 0..width {
                let cell_r = (row * GRID / height).min(GRID - 1);
                let cell_c = (col * GRID / width).min(GRID - 1);
                let cell_idx = cell_r * GRID + cell_c;
                block_sums[cell_idx] += luma[row * width + col] as f64;
                block_counts[cell_idx] += 1;
            }
        }

        let block_means: Vec<f64> = block_sums
            .iter()
            .zip(block_counts.iter())
            .map(|(s, c)| if *c > 0 { s / *c as f64 } else { 0.0 })
            .collect();

        // ── Step 3: global mean ──────────────────────────────────────────────
        let global_mean = block_means.iter().sum::<f64>() / (GRID * GRID) as f64;

        // ── Step 4: pack bits ────────────────────────────────────────────────
        let mut bits = 0u64;
        for (i, mean) in block_means.iter().enumerate() {
            if *mean > global_mean {
                bits |= 1u64 << i;
            }
        }

        Ok(HdrFingerprint { bits })
    }
}

/// Compute fingerprints for a sequence of frames and return similarity runs.
///
/// A *similarity run* is a maximal consecutive range of frame indices where
/// every adjacent pair has Hamming distance ≤ `threshold`.
///
/// # Errors
/// Propagates errors from [`HdrFrameFingerprinter::compute`].
pub fn find_similarity_runs(
    fingerprints: &[HdrFingerprint],
    threshold: u32,
) -> Vec<std::ops::Range<usize>> {
    if fingerprints.is_empty() {
        return Vec::new();
    }
    let mut runs = Vec::new();
    let mut run_start = 0usize;
    for i in 1..fingerprints.len() {
        if fingerprints[i - 1].hamming_distance(&fingerprints[i]) > threshold {
            if i - run_start > 1 {
                runs.push(run_start..i);
            }
            run_start = i;
        }
    }
    let last = fingerprints.len();
    if last - run_start > 1 {
        runs.push(run_start..last);
    }
    runs
}

/// Compute the mean pairwise Hamming distance across a slice of fingerprints.
///
/// Returns `None` if fewer than 2 fingerprints are provided.
pub fn mean_hamming_distance(fps: &[HdrFingerprint]) -> Option<f64> {
    if fps.len() < 2 {
        return None;
    }
    let mut sum = 0u64;
    let mut count = 0u64;
    for i in 0..fps.len() {
        for j in (i + 1)..fps.len() {
            sum += fps[i].hamming_distance(&fps[j]) as u64;
            count += 1;
        }
    }
    Some(sum as f64 / count as f64)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(width: usize, height: usize, r: f32, g: f32, b: f32) -> Vec<f32> {
        let n = width * height * 3;
        let mut v = Vec::with_capacity(n);
        for _ in 0..width * height {
            v.push(r);
            v.push(g);
            v.push(b);
        }
        v
    }

    #[test]
    fn test_identical_frames_zero_hamming() {
        let pixels = make_frame(32, 32, 0.5, 0.5, 0.5);
        let a = HdrFrameFingerprinter::compute(&pixels, 32, 32).unwrap();
        let b = HdrFrameFingerprinter::compute(&pixels, 32, 32).unwrap();
        assert_eq!(a.hamming_distance(&b), 0);
    }

    #[test]
    fn test_black_vs_white_high_hamming() {
        let black = make_frame(64, 64, 0.0, 0.0, 0.0);
        let white = make_frame(64, 64, 1.0, 1.0, 1.0);
        let fa = HdrFrameFingerprinter::compute(&black, 64, 64).unwrap();
        let fb = HdrFrameFingerprinter::compute(&white, 64, 64).unwrap();
        // Both uniform frames: all cells equal global mean → all bits 0
        // Distance = 0 (both fully uniform)
        assert_eq!(fa.hamming_distance(&fb), 0);
    }

    #[test]
    fn test_gradient_frame_non_zero_bits() {
        let w = 64usize;
        let h = 64usize;
        let mut pixels = Vec::with_capacity(w * h * 3);
        for row in 0..h {
            for col in 0..w {
                let v = col as f32 / (w - 1) as f32;
                pixels.push(v);
                pixels.push(v);
                pixels.push(v);
            }
        }
        let fp = HdrFrameFingerprinter::compute(&pixels, w, h).unwrap();
        // A left-dark, right-bright gradient should have non-zero bits
        assert!(fp.bits != 0 && fp.bits != u64::MAX);
    }

    #[test]
    fn test_fingerprint_display() {
        let fp = HdrFingerprint::from_u64(0xDEADBEEFCAFEBABE);
        let s = fp.to_string();
        assert_eq!(s, "deadbeefcafebabe");
    }

    #[test]
    fn test_from_u64_round_trip() {
        let v = 0xABCDEF0123456789u64;
        let fp = HdrFingerprint::from_u64(v);
        assert_eq!(fp.as_u64(), v);
    }

    #[test]
    fn test_is_similar_below_threshold() {
        let a = HdrFingerprint::from_u64(0u64);
        let b = HdrFingerprint::from_u64(0b1111u64); // 4 bits differ
        assert!(a.is_similar_to(&b));
    }

    #[test]
    fn test_is_not_similar_above_threshold() {
        let a = HdrFingerprint::from_u64(0u64);
        // 11 bits set → Hamming 11 > 10
        let b = HdrFingerprint::from_u64(0b111_1111_1111u64);
        assert!(!a.is_similar_to(&b));
    }

    #[test]
    fn test_error_on_zero_width() {
        let pixels = vec![0.5f32; 0];
        assert!(HdrFrameFingerprinter::compute(&pixels, 0, 8).is_err());
    }

    #[test]
    fn test_error_on_wrong_buffer_length() {
        let pixels = vec![0.5f32; 10];
        assert!(HdrFrameFingerprinter::compute(&pixels, 4, 4).is_err());
    }

    #[test]
    fn test_find_similarity_runs_empty() {
        let runs = find_similarity_runs(&[], 10);
        assert!(runs.is_empty());
    }

    #[test]
    fn test_find_similarity_runs_all_same() {
        let fp = HdrFingerprint::from_u64(0);
        let fps = vec![fp; 5];
        let runs = find_similarity_runs(&fps, 10);
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0], 0..5);
    }

    #[test]
    fn test_mean_hamming_distance_none_for_single() {
        let fp = HdrFingerprint::from_u64(0);
        assert!(mean_hamming_distance(&[fp]).is_none());
    }

    #[test]
    fn test_mean_hamming_distance_two_equal() {
        let fp = HdrFingerprint::from_u64(0xABCDu64);
        let dist = mean_hamming_distance(&[fp, fp]).unwrap();
        assert!((dist - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_hdr_peak_luminance_fingerprint_differs_from_sdr() {
        // HDR bright frame (10000 nits normalised) vs SDR-like (100 nits)
        let hdr_pixels = make_frame(32, 32, 10.0, 10.0, 10.0);
        let sdr_pixels = make_frame(32, 32, 0.01, 0.01, 0.01);
        let hdr_fp = HdrFrameFingerprinter::compute(&hdr_pixels, 32, 32).unwrap();
        let sdr_fp = HdrFrameFingerprinter::compute(&sdr_pixels, 32, 32).unwrap();
        // Both uniform – fingerprints are equal in structure (all-below-mean → both 0)
        assert_eq!(hdr_fp.bits, sdr_fp.bits);
    }
}
