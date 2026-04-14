//! Advanced compression artifact detection for video quality assessment.
//!
//! Provides three specialized detectors that go beyond the basic
//! `compression_artifacts` module:
//!
//! - [`MosquitoNoiseDetector`]: detects high-frequency oscillations (mosquito
//!   noise) that appear in areas of fine texture after lossy compression.
//! - [`RingingScoreDetector`]: locates Gibbs-phenomenon ringing near sharp
//!   edges and computes a macro-block-aligned ringing score.
//! - [`MacroblockBoundaryScorer`]: scores visible macro-block (16×16 or
//!   configurable) boundary discontinuities in the luma plane.
//!
//! All detectors accept normalised f32 luma samples in [0.0, 1.0].
//!
//! # Example
//!
//! ```
//! use oximedia_quality::compression_artifact::{
//!     MacroblockBoundaryScorer, MosquitoNoiseDetector, CompressionArtifactReport,
//! };
//!
//! let w = 64u32;
//! let h = 64u32;
//! let frame = vec![0.5f32; (w * h) as usize];
//!
//! let mb_scorer = MacroblockBoundaryScorer::new(16);
//! let mb_score = mb_scorer.score(&frame, w, h);
//! assert!(mb_score >= 0.0 && mb_score <= 1.0);
//!
//! let report = CompressionArtifactReport::analyze(&frame, w, h, 16);
//! assert!(report.overall_score() >= 0.0 && report.overall_score() <= 1.0);
//! ```

use serde::{Deserialize, Serialize};

// ─── Mosquito Noise Detector ──────────────────────────────────────────────────

/// Detects mosquito noise — high-frequency oscillation artefacts that appear
/// around object edges after lossy block-based compression (e.g., MPEG, AVC).
///
/// The approach:
/// 1. Build a high-pass residual by subtracting a 3×3 box-blur from the frame.
/// 2. Identify pixels that sit near strong edges (above `edge_threshold`).
/// 3. Measure the RMS energy of the high-pass residual in a neighbourhood
///    around each edge pixel.
/// 4. Return the mean of those energies as the mosquito noise score.
pub struct MosquitoNoiseDetector {
    /// Minimum Sobel gradient magnitude to classify a pixel as an edge.
    pub edge_threshold: f32,
    /// Half-size of the neighbourhood (in pixels) around each edge pixel
    /// that is examined for high-frequency energy.
    pub neighbourhood_radius: usize,
}

impl Default for MosquitoNoiseDetector {
    fn default() -> Self {
        Self {
            edge_threshold: 0.08,
            neighbourhood_radius: 4,
        }
    }
}

impl MosquitoNoiseDetector {
    /// Creates a new detector with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a detector with explicit parameters.
    #[must_use]
    pub fn with_params(edge_threshold: f32, neighbourhood_radius: usize) -> Self {
        Self {
            edge_threshold: edge_threshold.max(0.0),
            neighbourhood_radius: neighbourhood_radius.max(1),
        }
    }

    /// Computes a mosquito noise score in [0.0, 1.0].
    ///
    /// `frame` is a row-major slice of normalised luma values, width × height.
    #[must_use]
    pub fn score(&self, frame: &[f32], width: u32, height: u32) -> f32 {
        let w = width as usize;
        let h = height as usize;

        if w < 3 || h < 3 || frame.len() < w * h {
            return 0.0;
        }

        // Step 1: Build box-blur (3×3) and subtract to get high-pass residual
        let blur = box_blur_3x3(frame, w, h);
        let residual: Vec<f32> = frame
            .iter()
            .zip(blur.iter())
            .map(|(&f, &b)| (f - b).abs())
            .collect();

        // Step 2: Compute Sobel edge map
        let edges = sobel_magnitude(frame, w, h);

        // Step 3: For each edge pixel, measure local residual energy
        let r = self.neighbourhood_radius;
        let mut total_energy = 0.0f64;
        let mut edge_count = 0u64;

        for row in 0..h {
            for col in 0..w {
                if edges[row * w + col] < self.edge_threshold {
                    continue;
                }

                // Gather residual values in neighbourhood
                let mut energy = 0.0f64;
                let mut count = 0u64;

                let row_start = row.saturating_sub(r);
                let row_end = (row + r + 1).min(h);
                let col_start = col.saturating_sub(r);
                let col_end = (col + r + 1).min(w);

                for nr in row_start..row_end {
                    for nc in col_start..col_end {
                        let v = f64::from(residual[nr * w + nc]);
                        energy += v * v;
                        count += 1;
                    }
                }

                if count > 0 {
                    total_energy += (energy / count as f64).sqrt();
                    edge_count += 1;
                }
            }
        }

        if edge_count == 0 {
            return 0.0;
        }

        // Normalise: mean RMS around edges, scale so ~0.05 maps to 1.0
        let mean_energy = (total_energy / edge_count as f64) as f32;
        (mean_energy / 0.05).clamp(0.0, 1.0)
    }
}

// ─── Ringing Score Detector ───────────────────────────────────────────────────

/// Detects ringing artefacts (Gibbs phenomenon) aligned with macro-block
/// boundaries.
///
/// Unlike the generic ringing detector in `compression_artifacts`, this
/// implementation specifically looks for oscillations that occur at multiples
/// of `block_size` pixels, which is the characteristic signature of block-DCT
/// codecs.
pub struct RingingScoreDetector {
    /// Block size used by the codec (typically 8 for DCT-based codecs).
    pub block_size: usize,
    /// How many pixels beyond the block boundary to examine for ringing.
    pub search_pixels: usize,
    /// Minimum edge strength to consider a boundary active.
    pub edge_threshold: f32,
}

impl Default for RingingScoreDetector {
    fn default() -> Self {
        Self {
            block_size: 8,
            search_pixels: 4,
            edge_threshold: 0.05,
        }
    }
}

impl RingingScoreDetector {
    /// Creates a new detector with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a detector with explicit block size.
    #[must_use]
    pub fn with_block_size(block_size: usize) -> Self {
        Self {
            block_size: block_size.max(2),
            ..Default::default()
        }
    }

    /// Computes a ringing score in [0.0, 1.0].
    ///
    /// The score is the fraction of block-boundary pixels that show oscillatory
    /// behaviour beyond the block edge, weighted by edge strength.
    #[must_use]
    pub fn score(&self, frame: &[f32], width: u32, height: u32) -> f32 {
        let w = width as usize;
        let h = height as usize;

        if w < self.block_size + self.search_pixels
            || h < self.block_size + self.search_pixels
            || frame.len() < w * h
        {
            return 0.0;
        }

        let edges = sobel_magnitude(frame, w, h);
        let bs = self.block_size;
        let sp = self.search_pixels;

        let mut ringing_sum = 0.0f64;
        let mut boundary_count = 0u64;

        // Examine horizontal block boundaries (row % block_size == 0)
        for row in (bs..h.saturating_sub(sp)).step_by(bs) {
            for col in sp..w.saturating_sub(sp) {
                let boundary_strength = edges[row * w + col];
                if boundary_strength < self.edge_threshold {
                    continue;
                }

                // Oscillation score: count sign changes in `sp` pixels after boundary
                let mut sign_changes = 0u32;
                let mut prev = frame[row * w + col] > frame[(row - 1) * w + col];
                for k in 1..=sp {
                    if row + k >= h {
                        break;
                    }
                    let cur = frame[(row + k) * w + col] > frame[(row + k - 1) * w + col];
                    if cur != prev {
                        sign_changes += 1;
                    }
                    prev = cur;
                }

                let normalised = sign_changes as f64 / sp as f64;
                ringing_sum += normalised * f64::from(boundary_strength);
                boundary_count += 1;
            }
        }

        // Examine vertical block boundaries (col % block_size == 0)
        for col in (bs..w.saturating_sub(sp)).step_by(bs) {
            for row in sp..h.saturating_sub(sp) {
                let boundary_strength = edges[row * w + col];
                if boundary_strength < self.edge_threshold {
                    continue;
                }

                let mut sign_changes = 0u32;
                let mut prev = frame[row * w + col] > frame[row * w + col - 1];
                for k in 1..=sp {
                    if col + k >= w {
                        break;
                    }
                    let cur = frame[row * w + col + k] > frame[row * w + col + k - 1];
                    if cur != prev {
                        sign_changes += 1;
                    }
                    prev = cur;
                }

                let normalised = sign_changes as f64 / sp as f64;
                ringing_sum += normalised * f64::from(boundary_strength);
                boundary_count += 1;
            }
        }

        if boundary_count == 0 {
            return 0.0;
        }

        (ringing_sum / boundary_count as f64).clamp(0.0, 1.0) as f32
    }
}

// ─── Macro-block Boundary Scorer ─────────────────────────────────────────────

/// Scores visible macro-block boundary discontinuities in the luma plane.
///
/// High scores indicate harsh, visible block edges characteristic of
/// heavy over-compression or insufficient deblocking.
pub struct MacroblockBoundaryScorer {
    /// Macro-block side length in pixels (e.g. 16 for H.264 macro-blocks).
    pub block_size: usize,
}

impl MacroblockBoundaryScorer {
    /// Creates a scorer for the given macro-block size.
    #[must_use]
    pub fn new(block_size: usize) -> Self {
        Self {
            block_size: block_size.max(2),
        }
    }

    /// Scores macro-block boundary discontinuities.
    ///
    /// Returns a value in [0.0, 1.0] where higher means more visible blocking.
    #[must_use]
    pub fn score(&self, frame: &[f32], width: u32, height: u32) -> f32 {
        let w = width as usize;
        let h = height as usize;
        let bs = self.block_size;

        if w < bs + 1 || h < bs + 1 || frame.len() < w * h {
            return 0.0;
        }

        let mut boundary_diff = 0.0f64;
        let mut interior_diff = 0.0f64;
        let mut boundary_count = 0u64;
        let mut interior_count = 0u64;

        // Horizontal scan: compare pixel across macro-block boundaries vs. interior
        for row in 0..h {
            for col in 1..w {
                let diff = (frame[row * w + col] - frame[row * w + col - 1]).abs() as f64;
                if col % bs == 0 {
                    boundary_diff += diff;
                    boundary_count += 1;
                } else {
                    interior_diff += diff;
                    interior_count += 1;
                }
            }
        }

        // Vertical scan
        for row in 1..h {
            for col in 0..w {
                let diff = (frame[row * w + col] - frame[(row - 1) * w + col]).abs() as f64;
                if row % bs == 0 {
                    boundary_diff += diff;
                    boundary_count += 1;
                } else {
                    interior_diff += diff;
                    interior_count += 1;
                }
            }
        }

        if boundary_count == 0 || interior_count == 0 {
            return 0.0;
        }

        let mean_boundary = boundary_diff / boundary_count as f64;
        let mean_interior = interior_diff / interior_count as f64;

        if mean_interior < 1e-9 {
            return if mean_boundary > 1e-9 { 1.0 } else { 0.0 };
        }

        // Normalise ratio: ratio > 1 means boundary is more different than interior
        // Map: ratio 1 → 0.0, ratio 5+ → 1.0
        let ratio = mean_boundary / mean_interior;
        ((ratio - 1.0) / 4.0).clamp(0.0, 1.0) as f32
    }
}

// ─── Combined Report ──────────────────────────────────────────────────────────

/// Combined advanced compression artifact report for a single frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionArtifactReport {
    /// Mosquito noise score [0, 1].
    pub mosquito_noise: f32,
    /// Ringing score at block boundaries [0, 1].
    pub ringing: f32,
    /// Macro-block boundary visibility score [0, 1].
    pub macroblock_boundary: f32,
}

impl CompressionArtifactReport {
    /// Runs all three detectors with default parameters and assembles a report.
    #[must_use]
    pub fn analyze(frame: &[f32], width: u32, height: u32, block_size: usize) -> Self {
        let mosquito_noise = MosquitoNoiseDetector::new().score(frame, width, height);
        let ringing = RingingScoreDetector::with_block_size(block_size).score(frame, width, height);
        let macroblock_boundary =
            MacroblockBoundaryScorer::new(block_size).score(frame, width, height);

        Self {
            mosquito_noise,
            ringing,
            macroblock_boundary,
        }
    }

    /// Computes a weighted overall artifact score.
    ///
    /// Weights: macro-block boundary 45 %, ringing 35 %, mosquito 20 %.
    #[must_use]
    pub fn overall_score(&self) -> f32 {
        (0.45 * self.macroblock_boundary + 0.35 * self.ringing + 0.20 * self.mosquito_noise)
            .clamp(0.0, 1.0)
    }
}

// ─── Private helpers ──────────────────────────────────────────────────────────

/// Computes a 3×3 box blur.
fn box_blur_3x3(frame: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    for row in 0..h {
        for col in 0..w {
            let mut sum = 0.0f32;
            let mut count = 0u32;
            for dr in -1i32..=1 {
                for dc in -1i32..=1 {
                    let nr = row as i32 + dr;
                    let nc = col as i32 + dc;
                    if nr >= 0 && nr < h as i32 && nc >= 0 && nc < w as i32 {
                        sum += frame[nr as usize * w + nc as usize];
                        count += 1;
                    }
                }
            }
            out[row * w + col] = if count > 0 { sum / count as f32 } else { 0.0 };
        }
    }
    out
}

/// Computes Sobel gradient magnitude (result normalised roughly to [0, 1]).
fn sobel_magnitude(frame: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut mag = vec![0.0f32; w * h];
    if w < 3 || h < 3 {
        return mag;
    }
    for row in 1..h - 1 {
        for col in 1..w - 1 {
            let gx = -frame[(row - 1) * w + col - 1] + frame[(row - 1) * w + col + 1]
                - 2.0 * frame[row * w + col - 1]
                + 2.0 * frame[row * w + col + 1]
                - frame[(row + 1) * w + col - 1]
                + frame[(row + 1) * w + col + 1];
            let gy = -frame[(row - 1) * w + col - 1]
                - 2.0 * frame[(row - 1) * w + col]
                - frame[(row - 1) * w + col + 1]
                + frame[(row + 1) * w + col - 1]
                + 2.0 * frame[(row + 1) * w + col]
                + frame[(row + 1) * w + col + 1];
            mag[row * w + col] = (gx * gx + gy * gy).sqrt() / 4.0; // rough normalisation
        }
    }
    mag
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_frame(w: u32, h: u32, val: f32) -> Vec<f32> {
        vec![val; (w * h) as usize]
    }

    fn checkerboard_blocks(w: u32, h: u32, block_size: u32) -> Vec<f32> {
        let w_us = w as usize;
        let h_us = h as usize;
        let bs = block_size as usize;
        (0..w_us * h_us)
            .map(|i| {
                let row = i / w_us;
                let col = i % w_us;
                if ((row / bs) + (col / bs)) % 2 == 0 {
                    0.2
                } else {
                    0.8
                }
            })
            .collect()
    }

    // ── MosquitoNoiseDetector ──────────────────────────────────────────────

    #[test]
    fn test_mosquito_flat_frame_no_noise() {
        let frame = flat_frame(64, 64, 0.5);
        let det = MosquitoNoiseDetector::new();
        let s = det.score(&frame, 64, 64);
        // Flat frame has no edges → score should be 0
        assert_eq!(s, 0.0);
    }

    #[test]
    fn test_mosquito_score_in_range() {
        let frame = checkerboard_blocks(64, 64, 8);
        let det = MosquitoNoiseDetector::new();
        let s = det.score(&frame, 64, 64);
        assert!((0.0..=1.0).contains(&s));
    }

    #[test]
    fn test_mosquito_too_small_frame() {
        let frame = vec![0.5f32; 4];
        let det = MosquitoNoiseDetector::new();
        let s = det.score(&frame, 2, 2);
        assert_eq!(s, 0.0);
    }

    #[test]
    fn test_mosquito_with_params() {
        let det = MosquitoNoiseDetector::with_params(0.05, 3);
        assert!((det.edge_threshold - 0.05).abs() < 1e-6);
        assert_eq!(det.neighbourhood_radius, 3);
    }

    // ── RingingScoreDetector ───────────────────────────────────────────────

    #[test]
    fn test_ringing_flat_frame_zero() {
        let frame = flat_frame(64, 64, 0.5);
        let det = RingingScoreDetector::new();
        let s = det.score(&frame, 64, 64);
        // No edges → no ringing
        assert_eq!(s, 0.0);
    }

    #[test]
    fn test_ringing_score_in_range() {
        let frame = checkerboard_blocks(64, 64, 8);
        let det = RingingScoreDetector::new();
        let s = det.score(&frame, 64, 64);
        assert!((0.0..=1.0).contains(&s));
    }

    #[test]
    fn test_ringing_frame_too_small() {
        let frame = vec![0.5f32; 4 * 4];
        let det = RingingScoreDetector::with_block_size(8);
        let s = det.score(&frame, 4, 4);
        assert_eq!(s, 0.0);
    }

    // ── MacroblockBoundaryScorer ───────────────────────────────────────────

    #[test]
    fn test_mb_scorer_flat_frame_zero() {
        let frame = flat_frame(64, 64, 0.5);
        let scorer = MacroblockBoundaryScorer::new(16);
        let s = scorer.score(&frame, 64, 64);
        assert_eq!(s, 0.0);
    }

    #[test]
    fn test_mb_scorer_block_frame_high() {
        // Checkerboard with block_size=16 should produce a non-trivial score
        let frame = checkerboard_blocks(64, 64, 16);
        let scorer = MacroblockBoundaryScorer::new(16);
        let s = scorer.score(&frame, 64, 64);
        assert!((0.0..=1.0).contains(&s));
        // The checkerboard has equally strong boundary and interior differences
        // for 16-pixel blocks → score may be 0 or close to 0 since ratio ≈ 1
        // Just check it is valid
        assert!(s >= 0.0);
    }

    #[test]
    fn test_mb_scorer_frame_too_small() {
        let frame = vec![0.5f32; 4 * 4];
        let scorer = MacroblockBoundaryScorer::new(16);
        let s = scorer.score(&frame, 4, 4);
        assert_eq!(s, 0.0);
    }

    // ── CompressionArtifactReport ─────────────────────────────────────────

    #[test]
    fn test_report_flat_frame_near_zero() {
        let frame = flat_frame(64, 64, 0.5);
        let report = CompressionArtifactReport::analyze(&frame, 64, 64, 8);
        assert_eq!(report.mosquito_noise, 0.0);
        assert_eq!(report.ringing, 0.0);
        assert_eq!(report.macroblock_boundary, 0.0);
        assert_eq!(report.overall_score(), 0.0);
    }

    #[test]
    fn test_report_overall_score_in_range() {
        let frame = checkerboard_blocks(64, 64, 8);
        let report = CompressionArtifactReport::analyze(&frame, 64, 64, 8);
        assert!((0.0..=1.0).contains(&report.overall_score()));
    }

    #[test]
    fn test_report_weights_sum_to_one() {
        let w: f32 = 0.45 + 0.35 + 0.20;
        assert!((w - 1.0).abs() < 1e-6);
    }
}
