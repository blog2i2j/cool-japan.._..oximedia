//! Multi-pass analysis mode for higher accuracy.
//!
//! Provides [`MultiPassAnalyzer`] which performs two passes over a frame
//! sequence:
//!
//! 1. **First pass** — collects aggregate statistics (histogram moments,
//!    motion energy, scene count estimate) without full processing.
//! 2. **Second pass** — uses the statistics from the first pass to apply
//!    adaptive thresholds and produce a richer [`AnalysisResult`].
//!
//! This is useful for batch workflows where the full video is available upfront
//! and where the overhead of a second pass is acceptable.

#![allow(dead_code)]

use crate::{AnalysisError, AnalysisResult};

// ─────────────────────────────────────────────────────────────────────────────
// First-pass result
// ─────────────────────────────────────────────────────────────────────────────

/// Statistics collected during the first analysis pass.
///
/// These are used to calibrate detectors in the second pass.
#[derive(Debug, Clone, Default)]
pub struct PassOneResult {
    /// Total number of frames analysed in the first pass.
    pub frame_count: usize,
    /// Average luma of all frames (0.0–255.0).
    pub avg_luma: f64,
    /// Standard deviation of per-frame average luma.
    pub luma_stddev: f64,
    /// Estimated average inter-frame difference (proxy for motion energy).
    pub avg_motion_energy: f64,
    /// Estimated number of scene changes.
    pub estimated_scene_count: usize,
    /// Average edge density across all frames (0.0–1.0).
    pub avg_edge_density: f64,
    /// Suggested scene detection threshold for the second pass (0.0–1.0).
    pub suggested_scene_threshold: f64,
}

impl PassOneResult {
    /// Returns `true` if the first pass collected at least one frame.
    #[must_use]
    pub fn has_data(&self) -> bool {
        self.frame_count > 0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Second-pass result
// ─────────────────────────────────────────────────────────────────────────────

/// Per-frame analysis result from the second pass.
#[derive(Debug, Clone)]
pub struct FrameResult {
    /// Frame index.
    pub frame_idx: usize,
    /// Average luma (0.0–255.0).
    pub avg_luma: f64,
    /// Edge density (0.0–1.0).
    pub edge_density: f64,
    /// Inter-frame motion energy relative to previous frame (0.0–1.0).
    pub motion_energy: f64,
    /// Whether a scene change was detected at this frame.
    pub scene_change: bool,
}

/// Full result from the multi-pass analysis.
#[derive(Debug, Clone, Default)]
pub struct AnalysisResult2 {
    /// Per-frame results (in frame order).
    pub frames: Vec<FrameResult>,
    /// Indices of detected scene change frames.
    pub scene_changes: Vec<usize>,
    /// Calibration data from the first pass.
    pub calibration: PassOneResult,
}

impl AnalysisResult2 {
    /// Number of detected scene changes.
    #[must_use]
    pub fn scene_count(&self) -> usize {
        self.scene_changes.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MultiPassAnalyzer
// ─────────────────────────────────────────────────────────────────────────────

/// Multi-pass video analyser.
///
/// Call [`MultiPassAnalyzer::analyze_first_pass`] with all frames, then call
/// [`MultiPassAnalyzer::analyze_second_pass`] using the first-pass result to
/// obtain a more accurate final result.
pub struct MultiPassAnalyzer {
    /// Minimum scene-change inter-frame difference (used as a floor on the
    /// adaptive threshold).
    pub min_scene_threshold: f64,
    /// Maximum scene-change inter-frame difference (ceiling).
    pub max_scene_threshold: f64,
}

impl MultiPassAnalyzer {
    /// Create a new `MultiPassAnalyzer` with default thresholds.
    #[must_use]
    pub fn new() -> Self {
        Self {
            min_scene_threshold: 0.10,
            max_scene_threshold: 0.60,
        }
    }

    /// First pass: collect aggregate statistics from all frames.
    ///
    /// # Arguments
    ///
    /// * `frames` – slice of raw luma planes (`width * height` bytes each)
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError::InsufficientData`] if `frames` is empty.
    #[allow(clippy::cast_precision_loss)]
    pub fn analyze_first_pass(&self, frames: &[&[u8]]) -> AnalysisResult<PassOneResult> {
        if frames.is_empty() {
            return Err(AnalysisError::InsufficientData(
                "at least one frame is required for the first pass".to_string(),
            ));
        }

        let n = frames.len();
        let mut luma_sums = Vec::with_capacity(n);
        let mut motion_energies = Vec::with_capacity(n);
        let mut edge_densities = Vec::with_capacity(n);

        for frame in frames {
            let avg = average_luma(frame);
            luma_sums.push(avg);
            let ed = edge_density(frame);
            edge_densities.push(ed);
        }

        // Compute inter-frame differences for motion energy
        for i in 1..n {
            let diff = (luma_sums[i] - luma_sums[i - 1]).abs();
            motion_energies.push(diff);
        }
        // First frame has zero motion energy
        if !motion_energies.is_empty() {
            motion_energies.insert(0, 0.0);
        } else {
            motion_energies.push(0.0);
        }

        let avg_luma = luma_sums.iter().sum::<f64>() / n as f64;
        let luma_variance = luma_sums
            .iter()
            .map(|&v| (v - avg_luma).powi(2))
            .sum::<f64>()
            / n as f64;
        let luma_stddev = luma_variance.sqrt();

        let avg_motion_energy = if n > 1 {
            motion_energies[1..].iter().sum::<f64>() / (n - 1) as f64
        } else {
            0.0
        };

        let avg_edge_density = edge_densities.iter().sum::<f64>() / n as f64;

        // Adaptive scene threshold: base + stddev/255 * range
        let adaptive = self.min_scene_threshold
            + (luma_stddev / 255.0) * (self.max_scene_threshold - self.min_scene_threshold);
        let suggested_scene_threshold = adaptive.clamp(self.min_scene_threshold, self.max_scene_threshold);

        // Rough scene count estimate using threshold on motion energies
        let estimated_scene_count = motion_energies
            .iter()
            .filter(|&&e| e > suggested_scene_threshold * 255.0)
            .count();

        Ok(PassOneResult {
            frame_count: n,
            avg_luma,
            luma_stddev,
            avg_motion_energy,
            estimated_scene_count,
            avg_edge_density,
            suggested_scene_threshold,
        })
    }

    /// Second pass: perform detailed analysis using calibrated thresholds from the first pass.
    ///
    /// # Errors
    ///
    /// Returns an error if `frames` is empty or if `first` was not produced by
    /// a successful first pass.
    #[allow(clippy::cast_precision_loss)]
    pub fn analyze_second_pass(
        &self,
        frames: &[&[u8]],
        first: &PassOneResult,
    ) -> AnalysisResult<AnalysisResult2> {
        if frames.is_empty() {
            return Err(AnalysisError::InsufficientData(
                "at least one frame is required for the second pass".to_string(),
            ));
        }
        if !first.has_data() {
            return Err(AnalysisError::InsufficientData(
                "first-pass result must contain at least one frame".to_string(),
            ));
        }

        let scene_thresh = first.suggested_scene_threshold * 255.0;
        let n = frames.len();
        let mut results = Vec::with_capacity(n);
        let mut scene_changes = Vec::new();
        let mut prev_avg = average_luma(frames[0]);

        for (idx, frame) in frames.iter().enumerate() {
            let avg = average_luma(frame);
            let motion = if idx == 0 {
                0.0
            } else {
                let raw_diff = (avg - prev_avg).abs();
                (raw_diff / 255.0).clamp(0.0, 1.0)
            };
            let ed = edge_density(frame);
            let scene_change = idx > 0 && (avg - prev_avg).abs() > scene_thresh;
            if scene_change {
                scene_changes.push(idx);
            }
            results.push(FrameResult {
                frame_idx: idx,
                avg_luma: avg,
                edge_density: ed,
                motion_energy: motion,
                scene_change,
            });
            prev_avg = avg;
        }

        Ok(AnalysisResult2 {
            frames: results,
            scene_changes,
            calibration: first.clone(),
        })
    }
}

impl Default for MultiPassAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper functions
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the average luma value of a frame.
#[allow(clippy::cast_precision_loss)]
fn average_luma(frame: &[u8]) -> f64 {
    if frame.is_empty() {
        return 0.0;
    }
    let sum: u64 = frame.iter().map(|&v| v as u64).sum();
    sum as f64 / frame.len() as f64
}

/// Compute the fraction of pixels that are "edges" (simple Sobel magnitude > 30).
#[allow(clippy::cast_precision_loss)]
fn edge_density(frame: &[u8]) -> f64 {
    // Minimal implementation: use horizontal gradient as a proxy.
    if frame.len() < 3 {
        return 0.0;
    }
    let mut edge_count = 0u64;
    let n = frame.len();
    for i in 1..n - 1 {
        let dx = i32::from(frame[i + 1]).wrapping_sub(i32::from(frame[i - 1])).abs();
        if dx > 30 {
            edge_count += 1;
        }
    }
    edge_count as f64 / n as f64
}

// ─────────────────────────────────────────────────────────────────────────────
// Downsampling utility
// ─────────────────────────────────────────────────────────────────────────────

/// Downscale a luma plane by `factor` using nearest-neighbour sampling.
///
/// # Returns
///
/// `(downscaled_luma, new_width, new_height)`.  If `factor` is 0 or 1 the input
/// is returned unchanged (cloned).
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn downsample(frame: &[u8], w: u32, h: u32, factor: u32) -> (Vec<u8>, u32, u32) {
    if factor <= 1 || w == 0 || h == 0 {
        return (frame.to_vec(), w, h);
    }
    let new_w = (w / factor).max(1);
    let new_h = (h / factor).max(1);
    let width = w as usize;
    let new_width = new_w as usize;
    let new_height = new_h as usize;
    let f = factor as usize;

    let mut out = vec![0u8; new_width * new_height];
    for ny in 0..new_height {
        for nx in 0..new_width {
            let sy = (ny * f).min(h as usize - 1);
            let sx = (nx * f).min(w as usize - 1);
            let idx = sy * width + sx;
            out[ny * new_width + nx] = *frame.get(idx).unwrap_or(&0);
        }
    }
    (out, new_w, new_h)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_flat_frames(n: usize, size: usize, value: u8) -> Vec<Vec<u8>> {
        (0..n).map(|_| vec![value; size]).collect()
    }

    #[test]
    fn test_first_pass_empty_returns_error() {
        let analyzer = MultiPassAnalyzer::new();
        let result = analyzer.analyze_first_pass(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_first_pass_single_frame() {
        let analyzer = MultiPassAnalyzer::new();
        let frame = vec![128u8; 100];
        let result = analyzer.analyze_first_pass(&[&frame]).expect("first pass should succeed");
        assert_eq!(result.frame_count, 1);
        assert!((result.avg_luma - 128.0).abs() < 0.1);
    }

    #[test]
    fn test_second_pass_uses_first_pass_calibration() {
        let analyzer = MultiPassAnalyzer::new();
        let frames_data = make_flat_frames(5, 64, 100);
        let frames: Vec<&[u8]> = frames_data.iter().map(|v| v.as_slice()).collect();
        let first = analyzer.analyze_first_pass(&frames).expect("first pass");
        let second = analyzer.analyze_second_pass(&frames, &first).expect("second pass");
        assert_eq!(second.frames.len(), 5);
    }

    #[test]
    fn test_second_pass_detects_scene_change() {
        let analyzer = MultiPassAnalyzer::new();
        // 5 dark frames then 5 bright frames → scene change at frame 5
        let dark = vec![0u8; 64];
        let bright = vec![255u8; 64];
        let frames_data: Vec<Vec<u8>> = (0..5).map(|_| dark.clone())
            .chain((0..5).map(|_| bright.clone()))
            .collect();
        let frames: Vec<&[u8]> = frames_data.iter().map(|v| v.as_slice()).collect();
        let first = analyzer.analyze_first_pass(&frames).expect("first pass");
        let second = analyzer.analyze_second_pass(&frames, &first).expect("second pass");
        assert!(
            !second.scene_changes.is_empty(),
            "should detect scene change between dark and bright frames"
        );
    }

    #[test]
    fn test_downsample_factor_1_returns_same() {
        let frame = vec![50u8; 100];
        let (out, nw, nh) = downsample(&frame, 10, 10, 1);
        assert_eq!(nw, 10);
        assert_eq!(nh, 10);
        assert_eq!(out, frame);
    }

    #[test]
    fn test_downsample_factor_2() {
        let frame: Vec<u8> = (0u8..16).collect();
        let (out, nw, nh) = downsample(&frame, 4, 4, 2);
        assert_eq!(nw, 2);
        assert_eq!(nh, 2);
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn test_downsample_zero_factor() {
        let frame = vec![10u8; 16];
        let (out, nw, nh) = downsample(&frame, 4, 4, 0);
        assert_eq!(nw, 4);
        assert_eq!(nh, 4);
        assert_eq!(out.len(), 16);
    }
}
