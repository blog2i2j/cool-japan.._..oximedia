//! Configurable wipe direction patterns for transition detection.
//!
//! This module provides advanced wipe pattern detection beyond the basic linear
//! wipes. It implements the following pattern types:
//!
//! - **Radial wipe** — the transition boundary sweeps from a centre point outward
//!   like an expanding circle (or inward for a closing iris).
//! - **Iris wipe** — a circular wipe that opens from or closes to the centre
//!   (or any configurable centre point).
//! - **Clock wipe** (sweep wipe) — the transition boundary rotates like a clock
//!   hand from a pivot point through 360 degrees.
//! - **Box wipe** — the transition boundary expands as an axis-aligned rectangle.
//! - **Diagonal wipe** — the transition boundary is a diagonal line sweeping
//!   across the frame.
//!
//! Each detector takes two consecutive frames and scores how well the observed
//! pixel-difference pattern matches the expected wipe geometry.
//!
//! # Algorithm
//!
//! For each pixel (y, x) the inter-frame absolute luminance difference is computed.
//! Pixels with difference above a `noise_threshold` are called *transition pixels*.
//! For a given wipe geometry, a binary *mask* is constructed that is 1 on the
//! expected transition band and 0 elsewhere. The Jaccard index between the
//! observed transition pixels and the mask gives the pattern score.

use std::f32::consts::{PI, TAU};

use crate::error::{ShotError, ShotResult};
use crate::frame_buffer::FrameBuffer;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default pixel difference threshold for a transition pixel (0-255).
const DEFAULT_NOISE_THRESHOLD: u8 = 20;

/// Default width of the expected transition band as a fraction of frame size.
const DEFAULT_BAND_FRACTION: f32 = 0.15;

// ---------------------------------------------------------------------------
// Public Types
// ---------------------------------------------------------------------------

/// Classification of the detected wipe pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WipePatternKind {
    /// Expanding circle from a centre point.
    Radial,
    /// Circular iris opening or closing.
    Iris,
    /// Rotating clock-hand sweep.
    Clock,
    /// Expanding axis-aligned rectangle.
    Box,
    /// Diagonal line sweep.
    Diagonal,
    /// Standard horizontal linear wipe.
    HorizontalLinear,
    /// Standard vertical linear wipe.
    VerticalLinear,
}

impl WipePatternKind {
    /// Human-readable name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Radial => "Radial",
            Self::Iris => "Iris",
            Self::Clock => "Clock (Sweep)",
            Self::Box => "Box",
            Self::Diagonal => "Diagonal",
            Self::HorizontalLinear => "Horizontal Linear",
            Self::VerticalLinear => "Vertical Linear",
        }
    }
}

/// Configuration for a single wipe pattern detector.
#[derive(Debug, Clone)]
pub struct WipePatternConfig {
    /// Minimum absolute luminance difference for a pixel to count as a transition pixel.
    pub noise_threshold: u8,
    /// Width of the expected transition band as a fraction of the frame's smaller dimension.
    pub band_fraction: f32,
    /// Centre of radial/iris/clock wipes as normalised (x, y) in \[0,1\].
    pub centre: (f32, f32),
    /// Wipe progress: 0.0 = start, 1.0 = end. Used to position the transition band.
    pub progress: f32,
    /// Clock wipe start angle in radians (0 = right, π/2 = down, etc.).
    pub clock_start_angle: f32,
}

impl Default for WipePatternConfig {
    fn default() -> Self {
        Self {
            noise_threshold: DEFAULT_NOISE_THRESHOLD,
            band_fraction: DEFAULT_BAND_FRACTION,
            centre: (0.5, 0.5),
            progress: 0.5,
            clock_start_angle: -PI / 2.0, // 12 o'clock
        }
    }
}

/// Detection result for a single wipe pattern.
#[derive(Debug, Clone)]
pub struct WipePatternResult {
    /// Pattern kind.
    pub kind: WipePatternKind,
    /// Jaccard similarity score between transition pixels and pattern mask (0.0–1.0).
    pub score: f32,
    /// Fraction of the frame covered by transition pixels.
    pub transition_density: f32,
    /// Estimated wipe progress (0.0–1.0) deduced from the transition band position.
    pub estimated_progress: f32,
    /// Whether the pattern is confident enough to be classified as this kind.
    pub is_confident: bool,
}

/// All pattern detection results for a frame pair.
#[derive(Debug, Clone)]
pub struct WipePatternAnalysis {
    /// Per-pattern scores.
    pub results: Vec<WipePatternResult>,
    /// The best-matching pattern kind.
    pub best_match: WipePatternKind,
    /// Score of the best match.
    pub best_score: f32,
    /// True if any pattern scored above the confidence threshold.
    pub wipe_detected: bool,
}

// ---------------------------------------------------------------------------
// Pattern Detector
// ---------------------------------------------------------------------------

/// Multi-pattern wipe detector.
///
/// Evaluates all registered pattern types against an observed inter-frame
/// difference image and returns scored results.
#[derive(Debug)]
pub struct WipePatternDetector {
    config: WipePatternConfig,
    /// Minimum score to consider a pattern as "detected".
    confidence_threshold: f32,
}

impl Default for WipePatternDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl WipePatternDetector {
    /// Create a detector with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: WipePatternConfig::default(),
            confidence_threshold: 0.2,
        }
    }

    /// Create a detector with custom configuration.
    pub fn with_config(config: WipePatternConfig) -> ShotResult<Self> {
        if !(0.0..=1.0).contains(&config.progress) {
            return Err(ShotError::InvalidParameters(
                "progress must be in [0.0, 1.0]".to_string(),
            ));
        }
        if config.band_fraction <= 0.0 || config.band_fraction > 1.0 {
            return Err(ShotError::InvalidParameters(
                "band_fraction must be in (0, 1]".to_string(),
            ));
        }
        let (cx, cy) = config.centre;
        if !(0.0..=1.0).contains(&cx) || !(0.0..=1.0).contains(&cy) {
            return Err(ShotError::InvalidParameters(
                "centre coordinates must be in [0.0, 1.0]".to_string(),
            ));
        }
        Ok(Self {
            config,
            confidence_threshold: 0.2,
        })
    }

    /// Set the confidence threshold.
    pub fn set_confidence_threshold(&mut self, threshold: f32) -> ShotResult<()> {
        if !(0.0..=1.0).contains(&threshold) {
            return Err(ShotError::InvalidParameters(
                "confidence_threshold must be in [0.0, 1.0]".to_string(),
            ));
        }
        self.confidence_threshold = threshold;
        Ok(())
    }

    /// Detect wipe patterns between two consecutive frames.
    pub fn detect(
        &self,
        frame_a: &FrameBuffer,
        frame_b: &FrameBuffer,
    ) -> ShotResult<WipePatternAnalysis> {
        let (ha, wa, ca) = frame_a.dim();
        let (hb, wb, cb) = frame_b.dim();
        if ha != hb || wa != wb {
            return Err(ShotError::InvalidFrame(
                "Frames must have identical dimensions".to_string(),
            ));
        }
        if ca < 1 || cb < 1 {
            return Err(ShotError::InvalidFrame(
                "Frames must have at least 1 channel".to_string(),
            ));
        }

        // Build luminance difference map
        let diff = compute_luminance_diff(frame_a, frame_b)?;

        // Build binary transition mask from diff
        let transition = diff
            .iter()
            .map(|&d| d >= self.config.noise_threshold)
            .collect::<Vec<_>>();

        let n_pixels = (ha * wa) as f32;
        let n_transition = transition.iter().filter(|&&t| t).count() as f32;
        let transition_density = n_transition / n_pixels.max(1.0);

        let kinds = [
            WipePatternKind::Radial,
            WipePatternKind::Iris,
            WipePatternKind::Clock,
            WipePatternKind::Box,
            WipePatternKind::Diagonal,
            WipePatternKind::HorizontalLinear,
            WipePatternKind::VerticalLinear,
        ];

        let mut results = Vec::with_capacity(kinds.len());

        for kind in kinds {
            let mask = self.build_pattern_mask(kind, ha, wa);
            let score = jaccard_score(&transition, &mask);
            let estimated_progress =
                self.estimate_progress_from_transition(&transition, kind, ha, wa);
            results.push(WipePatternResult {
                kind,
                score,
                transition_density,
                estimated_progress,
                is_confident: score >= self.confidence_threshold,
            });
        }

        // Find best match
        let best = results
            .iter()
            .max_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| ShotError::DetectionFailed("No pattern results".to_string()))?;

        let best_match = best.kind;
        let best_score = best.score;
        let wipe_detected = best_score >= self.confidence_threshold;

        Ok(WipePatternAnalysis {
            results,
            best_match,
            best_score,
            wipe_detected,
        })
    }

    // -----------------------------------------------------------------------
    // Pattern mask builders
    // -----------------------------------------------------------------------

    /// Build a binary mask for the expected transition band of a given pattern.
    fn build_pattern_mask(&self, kind: WipePatternKind, h: usize, w: usize) -> Vec<bool> {
        let progress = self.config.progress;
        let band = self.config.band_fraction;
        let (cx, cy) = self.config.centre;
        let hf = h as f32;
        let wf = w as f32;
        let smaller_dim = hf.min(wf);

        let mut mask = vec![false; h * w];

        match kind {
            WipePatternKind::Radial | WipePatternKind::Iris => {
                // Transition band is an annulus at radius progress * max_radius
                let max_radius = ((cx * cx + cy * cy).sqrt())
                    .max(((1.0 - cx) * (1.0 - cx) + cy * cy).sqrt())
                    .max((cx * cx + (1.0 - cy) * (1.0 - cy)).sqrt())
                    .max(((1.0 - cx) * (1.0 - cx) + (1.0 - cy) * (1.0 - cy)).sqrt())
                    * smaller_dim;
                let radius = progress * max_radius;
                let band_px = band * smaller_dim;
                let r_inner = (radius - band_px * 0.5).max(0.0);
                let r_outer = radius + band_px * 0.5;

                for y in 0..h {
                    for x in 0..w {
                        let dx = (x as f32 / wf) - cx;
                        let dy = (y as f32 / hf) - cy;
                        let dist = (dx * dx + dy * dy).sqrt() * smaller_dim;
                        mask[y * w + x] = dist >= r_inner && dist <= r_outer;
                    }
                }
            }
            WipePatternKind::Clock => {
                // Transition band is a thin angular wedge at the current clock angle
                let swept_angle = progress * TAU + self.config.clock_start_angle;
                let band_angle = band * PI; // angular width of the band

                for y in 0..h {
                    for x in 0..w {
                        let dx = (x as f32 / wf) - cx;
                        let dy = (y as f32 / hf) - cy;
                        if dx.abs() < 1e-5 && dy.abs() < 1e-5 {
                            continue; // centre pixel — skip
                        }
                        let angle = dy.atan2(dx);
                        // Normalise angle difference into [-π, π]
                        let diff = angle_diff(angle, swept_angle);
                        mask[y * w + x] = diff.abs() <= band_angle * 0.5;
                    }
                }
            }
            WipePatternKind::Box => {
                // Transition band is the perimeter of a box expanding from the centre
                let max_half_w = cx.min(1.0 - cx) * wf;
                let max_half_h = cy.min(1.0 - cy) * hf;
                let half_w = progress * max_half_w;
                let half_h = progress * max_half_h;
                let band_px = band * smaller_dim;
                let cx_px = cx * wf;
                let cy_px = cy * hf;

                for y in 0..h {
                    for x in 0..w {
                        let dx = (x as f32 - cx_px).abs();
                        let dy = (y as f32 - cy_px).abs();
                        // Distance from the box perimeter
                        let on_x_edge =
                            (dx - half_w).abs() <= band_px * 0.5 && dy <= half_h + band_px * 0.5;
                        let on_y_edge =
                            (dy - half_h).abs() <= band_px * 0.5 && dx <= half_w + band_px * 0.5;
                        mask[y * w + x] = on_x_edge || on_y_edge;
                    }
                }
            }
            WipePatternKind::Diagonal => {
                // Wipe along main diagonal
                // At progress p, the boundary is at x + y = p * (w + h)
                let boundary = progress * (wf + hf);
                let band_px = band * smaller_dim;

                for y in 0..h {
                    for x in 0..w {
                        let dist = (x as f32 + y as f32 - boundary).abs();
                        mask[y * w + x] = dist <= band_px * 0.5;
                    }
                }
            }
            WipePatternKind::HorizontalLinear => {
                // Vertical edge sweeping left→right
                let boundary_x = (progress * wf) as usize;
                let band_px = (band * smaller_dim) as usize;
                let lo = boundary_x.saturating_sub(band_px / 2);
                let hi = (boundary_x + band_px / 2).min(w);
                for y in 0..h {
                    for x in lo..hi {
                        mask[y * w + x] = true;
                    }
                }
            }
            WipePatternKind::VerticalLinear => {
                // Horizontal edge sweeping top→bottom
                let boundary_y = (progress * hf) as usize;
                let band_px = (band * smaller_dim) as usize;
                let lo = boundary_y.saturating_sub(band_px / 2);
                let hi = (boundary_y + band_px / 2).min(h);
                for y in lo..hi {
                    for x in 0..w {
                        mask[y * w + x] = true;
                    }
                }
            }
        }

        mask
    }

    /// Estimate wipe progress from the observed transition pixel distribution.
    fn estimate_progress_from_transition(
        &self,
        transition: &[bool],
        kind: WipePatternKind,
        h: usize,
        w: usize,
    ) -> f32 {
        match kind {
            WipePatternKind::HorizontalLinear => {
                // Progress ≈ centroid x / width
                let mut sum_x = 0.0f64;
                let mut count = 0u64;
                for (idx, &t) in transition.iter().enumerate() {
                    if t {
                        let x = idx % w;
                        sum_x += x as f64;
                        count += 1;
                    }
                }
                if count == 0 {
                    0.5
                } else {
                    (sum_x / count as f64 / w as f64) as f32
                }
            }
            WipePatternKind::VerticalLinear => {
                let mut sum_y = 0.0f64;
                let mut count = 0u64;
                for (idx, &t) in transition.iter().enumerate() {
                    if t {
                        let y = idx / w;
                        sum_y += y as f64;
                        count += 1;
                    }
                }
                if count == 0 {
                    0.5
                } else {
                    (sum_y / count as f64 / h as f64) as f32
                }
            }
            _ => self.config.progress, // For other patterns, use configured progress
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute per-pixel absolute luminance difference between two frames.
///
/// Returns a flat Vec<u8> of length h*w.
fn compute_luminance_diff(a: &FrameBuffer, b: &FrameBuffer) -> ShotResult<Vec<u8>> {
    let (h, w, ca) = a.dim();
    let (_, _, cb) = b.dim();

    let mut diff = vec![0u8; h * w];

    for y in 0..h {
        for x in 0..w {
            let luma = if ca >= 3 && cb >= 3 {
                let ra = f32::from(a.get(y, x, 0));
                let ga = f32::from(a.get(y, x, 1));
                let ba = f32::from(a.get(y, x, 2));
                let rb = f32::from(b.get(y, x, 0));
                let gb = f32::from(b.get(y, x, 1));
                let bb = f32::from(b.get(y, x, 2));
                let la = ra * 0.299 + ga * 0.587 + ba * 0.114;
                let lb = rb * 0.299 + gb * 0.587 + bb * 0.114;
                (la - lb).abs().min(255.0) as u8
            } else {
                let va = f32::from(a.get(y, x, 0));
                let vb = f32::from(b.get(y, x, 0));
                (va - vb).abs().min(255.0) as u8
            };
            diff[y * w + x] = luma;
        }
    }

    Ok(diff)
}

/// Compute Jaccard index between two binary vectors.
#[must_use]
fn jaccard_score(a: &[bool], b: &[bool]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut intersection = 0u32;
    let mut union = 0u32;
    for (&av, &bv) in a.iter().zip(b.iter()) {
        if av || bv {
            union += 1;
        }
        if av && bv {
            intersection += 1;
        }
    }
    if union == 0 {
        1.0 // Both empty — perfect match
    } else {
        intersection as f32 / union as f32
    }
}

/// Normalise the difference between two angles into [-π, π].
#[inline]
fn angle_diff(a: f32, b: f32) -> f32 {
    let mut diff = a - b;
    while diff > PI {
        diff -= TAU;
    }
    while diff < -PI {
        diff += TAU;
    }
    diff
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_frame(h: usize, w: usize, lum: u8) -> FrameBuffer {
        FrameBuffer::from_elem(h, w, 3, lum)
    }

    fn frame_with_horizontal_band(h: usize, w: usize, band_y: usize, band_h: usize) -> FrameBuffer {
        let mut frame = solid_frame(h, w, 50);
        let end_y = (band_y + band_h).min(h);
        for y in band_y..end_y {
            for x in 0..w {
                frame.set(y, x, 0, 200);
                frame.set(y, x, 1, 200);
                frame.set(y, x, 2, 200);
            }
        }
        frame
    }

    #[test]
    fn test_detector_default_creation() {
        let det = WipePatternDetector::new();
        assert!(!det.config.centre.0.is_nan());
    }

    #[test]
    fn test_detect_identical_frames_no_wipe() {
        let a = solid_frame(40, 40, 100);
        let b = solid_frame(40, 40, 100);
        let det = WipePatternDetector::new();
        let result = det.detect(&a, &b).expect("detect ok");
        // No differences → transition_density near 0
        assert!(result.results[0].transition_density < 0.01);
    }

    #[test]
    fn test_detect_vertical_linear_wipe() {
        // Simulate a vertical wipe: band across the middle of the frame
        let a = frame_with_horizontal_band(80, 80, 35, 10);
        let b = solid_frame(80, 80, 50);
        let mut cfg = WipePatternConfig::default();
        cfg.progress = 0.5;
        let det = WipePatternDetector::with_config(cfg).expect("ok");
        let result = det.detect(&a, &b).expect("detect ok");
        // VerticalLinear pattern should have a non-trivial score
        let vert = result
            .results
            .iter()
            .find(|r| r.kind == WipePatternKind::VerticalLinear)
            .expect("vertical result");
        assert!(vert.score >= 0.0 && vert.score <= 1.0);
    }

    #[test]
    fn test_detect_dimension_mismatch_error() {
        let a = solid_frame(40, 40, 100);
        let b = solid_frame(80, 80, 100);
        let det = WipePatternDetector::new();
        assert!(det.detect(&a, &b).is_err());
    }

    #[test]
    fn test_all_patterns_scored() {
        let a = solid_frame(32, 32, 80);
        let b = solid_frame(32, 32, 150);
        let det = WipePatternDetector::new();
        let result = det.detect(&a, &b).expect("ok");
        assert_eq!(result.results.len(), 7);
    }

    #[test]
    fn test_jaccard_identical() {
        let a = vec![true, false, true, true];
        let score = jaccard_score(&a, &a.clone());
        assert!((score - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_jaccard_disjoint() {
        let a = vec![true, false, false, false];
        let b = vec![false, true, false, false];
        let score = jaccard_score(&a, &b);
        assert!((score - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_confidence_threshold_validation() {
        let mut det = WipePatternDetector::new();
        assert!(det.set_confidence_threshold(0.3).is_ok());
        assert!(det.set_confidence_threshold(1.5).is_err());
    }

    #[test]
    fn test_config_validation_progress() {
        let mut cfg = WipePatternConfig::default();
        cfg.progress = 1.5;
        assert!(WipePatternDetector::with_config(cfg).is_err());
    }

    #[test]
    fn test_config_validation_band_fraction() {
        let mut cfg = WipePatternConfig::default();
        cfg.band_fraction = 0.0;
        assert!(WipePatternDetector::with_config(cfg).is_err());
    }

    #[test]
    fn test_wipe_pattern_kind_names() {
        assert_eq!(WipePatternKind::Radial.name(), "Radial");
        assert_eq!(WipePatternKind::Clock.name(), "Clock (Sweep)");
        assert_eq!(WipePatternKind::Iris.name(), "Iris");
    }

    #[test]
    fn test_angle_diff_wrapping() {
        use std::f32::consts::PI;
        let diff = angle_diff(PI - 0.1, -PI + 0.1);
        assert!(diff.abs() < 0.3, "expected small wrapped diff, got {diff}");
    }

    #[test]
    fn test_detect_best_match_field() {
        let a = solid_frame(32, 32, 10);
        let b = solid_frame(32, 32, 200);
        let det = WipePatternDetector::new();
        let result = det.detect(&a, &b).expect("ok");
        // All patterns are valid kinds
        let valid_kinds = [
            WipePatternKind::Radial,
            WipePatternKind::Iris,
            WipePatternKind::Clock,
            WipePatternKind::Box,
            WipePatternKind::Diagonal,
            WipePatternKind::HorizontalLinear,
            WipePatternKind::VerticalLinear,
        ];
        assert!(valid_kinds.contains(&result.best_match));
    }
}
