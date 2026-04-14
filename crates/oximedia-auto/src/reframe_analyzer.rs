//! Subject-position analysis and reframing suggestions for vertical/square crops.
//!
//! Given a time-series of detected subject (e.g., face or person) positions in a
//! 16:9 frame, this module computes smooth, rule-of-thirds-aware crop windows for
//! common social-media aspect ratios (9:16 vertical, 1:1 square, etc.).

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// SubjectPosition
// ---------------------------------------------------------------------------

/// The normalized position of the primary subject in a single video frame.
///
/// Coordinates are in `[0, 1]` where `(0, 0)` is the top-left corner and
/// `(1, 1)` is the bottom-right corner.
#[derive(Debug, Clone, PartialEq)]
pub struct SubjectPosition {
    /// Normalized horizontal centre of the subject (0 = left, 1 = right).
    pub cx: f32,
    /// Normalized vertical centre of the subject (0 = top, 1 = bottom).
    pub cy: f32,
    /// Detection confidence (0 = uncertain, 1 = certain).
    pub confidence: f32,
    /// Index of the source frame this position was detected in.
    pub frame_index: u64,
}

impl SubjectPosition {
    /// Create a new subject position.
    #[must_use]
    pub fn new(cx: f32, cy: f32, confidence: f32, frame_index: u64) -> Self {
        Self {
            cx: cx.clamp(0.0, 1.0),
            cy: cy.clamp(0.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
            frame_index,
        }
    }
}

// ---------------------------------------------------------------------------
// ReframeTarget
// ---------------------------------------------------------------------------

/// The target aspect ratio for the reframed output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReframeTarget {
    /// 9:16 portrait (YouTube Shorts, TikTok, Instagram Reels).
    VerticalShort,
    /// 1:1 square (Instagram feed, Twitter).
    Square,
    /// 16:9 landscape kept from a taller source (padded or cropped).
    Vertical169,
    /// Arbitrary aspect ratio specified by width and height ratios.
    Custom {
        /// Width component of the target ratio (e.g., 4 for 4:3).
        w_ratio: f32,
        /// Height component of the target ratio (e.g., 3 for 4:3).
        h_ratio: f32,
    },
}

impl ReframeTarget {
    /// Aspect ratio as `width / height`.
    #[must_use]
    pub fn aspect_ratio(&self) -> f32 {
        match self {
            ReframeTarget::VerticalShort => 9.0 / 16.0,
            ReframeTarget::Square => 1.0,
            ReframeTarget::Vertical169 => 16.0 / 9.0,
            ReframeTarget::Custom { w_ratio, h_ratio } => {
                if *h_ratio <= 0.0 {
                    1.0
                } else {
                    w_ratio / h_ratio
                }
            }
        }
    }

    /// Width of the crop window relative to the source width (in `[0, 1]`).
    ///
    /// The returned value is the crop-window width assuming the source is
    /// normalised to `width = 1.0`.  The crop height is always the full source
    /// height (`1.0`) for portrait/square targets, or derived from the ratio for
    /// landscape targets.
    #[must_use]
    pub fn crop_width_fraction(&self) -> f32 {
        // The source is assumed to be 16:9 (aspect ratio ≈ 1.778).
        let source_ar = 16.0_f32 / 9.0;
        let target_ar = self.aspect_ratio();
        if target_ar >= source_ar {
            // Target is wider or same as source — full width crop.
            1.0
        } else {
            // Crop a vertical slice whose width gives the target AR.
            // crop_width / source_height = target_ar
            // source_height = source_width / source_ar
            // → crop_width = target_ar * source_width / source_ar
            (target_ar / source_ar).clamp(0.0, 1.0)
        }
    }

    /// Height of the crop window relative to the source height.
    #[must_use]
    pub fn crop_height_fraction(&self) -> f32 {
        let source_ar = 16.0_f32 / 9.0;
        let target_ar = self.aspect_ratio();
        if target_ar >= source_ar {
            // Target is at least as wide: crop height, keep full width.
            // crop_height = source_width / target_ar  (in source-height units)
            // source_height = source_width / source_ar
            // → crop_height / source_height = source_ar / target_ar
            (source_ar / target_ar).clamp(0.0, 1.0)
        } else {
            1.0
        }
    }
}

// ---------------------------------------------------------------------------
// CropWindow
// ---------------------------------------------------------------------------

/// A normalised crop window within a source frame.
///
/// All fields are in `[0, 1]` relative to the source frame dimensions.
#[derive(Debug, Clone, PartialEq)]
pub struct CropWindow {
    /// Left edge of the crop (0 = source left, 1 = source right).
    pub x: f32,
    /// Top edge of the crop (0 = source top, 1 = source bottom).
    pub y: f32,
    /// Width of the crop region (fraction of source width).
    pub width: f32,
    /// Height of the crop region (fraction of source height).
    pub height: f32,
}

impl CropWindow {
    /// Create a centred crop window with the given dimensions.
    #[must_use]
    pub fn centered(width: f32, height: f32) -> Self {
        let x = ((1.0 - width) / 2.0).clamp(0.0, 1.0);
        let y = ((1.0 - height) / 2.0).clamp(0.0, 1.0);
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Validate that the crop window is fully contained within `[0, 1]`.
    #[must_use]
    pub fn validate(&self) -> bool {
        self.x >= 0.0
            && self.y >= 0.0
            && self.width > 0.0
            && self.height > 0.0
            && self.x + self.width <= 1.0 + 1e-5
            && self.y + self.height <= 1.0 + 1e-5
    }

    /// Clamp the window so it stays within `[0, 1]` bounds.
    #[must_use]
    pub fn clamped(mut self) -> Self {
        self.width = self.width.clamp(0.0, 1.0);
        self.height = self.height.clamp(0.0, 1.0);
        self.x = self.x.clamp(0.0, 1.0 - self.width);
        self.y = self.y.clamp(0.0, 1.0 - self.height);
        self
    }

    /// Interpolate (EMA blend) between `self` (previous) and `next`.
    ///
    /// `alpha` is the weight on `self`; `1 - alpha` is the weight on `next`.
    #[must_use]
    pub fn blend(&self, next: &CropWindow, alpha: f32) -> CropWindow {
        let a = alpha.clamp(0.0, 1.0);
        let b = 1.0 - a;
        CropWindow {
            x: self.x * a + next.x * b,
            y: self.y * a + next.y * b,
            width: self.width * a + next.width * b,
            height: self.height * a + next.height * b,
        }
    }
}

// ---------------------------------------------------------------------------
// ReframeAnalyzer
// ---------------------------------------------------------------------------

/// Analyses subject positions and suggests optimal crop windows per frame.
///
/// The analyzer applies:
/// 1. **Weighted centroid** — subject detections are blended by confidence.
/// 2. **Rule of thirds** — a lone subject is nudged to the nearest third line.
/// 3. **Temporal smoothing** — EMA filtering reduces jitter between frames.
#[derive(Debug, Clone)]
pub struct ReframeAnalyzer;

impl ReframeAnalyzer {
    /// Suggest per-frame crop windows for the given subject positions and target.
    ///
    /// The returned vector contains one `CropWindow` per entry in `positions`.
    /// If `positions` is empty, a single centred crop is returned.
    ///
    /// The workflow is:
    /// 1. Compute a weighted centroid over all positions (confidence-weighted).
    /// 2. Apply rule-of-thirds offset if there is exactly one detection cluster.
    /// 3. Build a crop window centred on the focal point.
    /// 4. Clamp to source bounds.
    #[must_use]
    pub fn suggest_crop(positions: &[SubjectPosition], target: ReframeTarget) -> Vec<CropWindow> {
        let cw = target.crop_width_fraction();
        let ch = target.crop_height_fraction();

        if positions.is_empty() {
            // Centre the crop when there are no subjects.
            return vec![CropWindow::centered(cw, ch)];
        }

        // Confidence-weighted centroid across all positions.
        let (focal_cx, focal_cy) = weighted_centroid(positions);

        // Apply rule-of-thirds nudge when working with a single concentration.
        let (focus_x, focus_y) = apply_rule_of_thirds(focal_cx, focal_cy, positions.len());

        // Build one crop window per position, each centred on the same focal point
        // (individual per-frame refinement could be added here in the future).
        positions
            .iter()
            .map(|_| build_crop(focus_x, focus_y, cw, ch))
            .collect()
    }

    /// Apply exponential moving average (EMA) smoothing to a trajectory of crop windows.
    ///
    /// `alpha` ∈ `[0, 1]`: weight given to the *previous* window.  Higher values
    /// mean more smoothing (more lag).  `alpha = 0` returns the input unchanged.
    #[must_use]
    pub fn smooth_trajectory(windows: &[CropWindow], alpha: f32) -> Vec<CropWindow> {
        if windows.is_empty() {
            return Vec::new();
        }
        let alpha = alpha.clamp(0.0, 1.0);
        if alpha < 1e-6 {
            return windows.to_vec();
        }

        let mut result = Vec::with_capacity(windows.len());
        let mut prev = windows[0].clone();
        result.push(prev.clone());

        for current in &windows[1..] {
            let blended = prev.blend(current, alpha).clamped();
            result.push(blended.clone());
            prev = blended;
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute a confidence-weighted centroid over subject positions.
fn weighted_centroid(positions: &[SubjectPosition]) -> (f32, f32) {
    let total_weight: f32 = positions.iter().map(|p| p.confidence).sum();
    if total_weight < 1e-6 {
        // All confidences are zero → simple mean.
        let n = positions.len() as f32;
        let cx = positions.iter().map(|p| p.cx).sum::<f32>() / n;
        let cy = positions.iter().map(|p| p.cy).sum::<f32>() / n;
        return (cx, cy);
    }
    let cx = positions.iter().map(|p| p.cx * p.confidence).sum::<f32>() / total_weight;
    let cy = positions.iter().map(|p| p.cy * p.confidence).sum::<f32>() / total_weight;
    (cx, cy)
}

/// Nudge the focal point to the nearest rule-of-thirds line when appropriate.
///
/// Rule-of-thirds lines are at 1/3 and 2/3 of the normalised axis.
/// The nudge is applied only when there is a single detection (or a tight
/// cluster), so that subjects in two-shot frames are not displaced.
fn apply_rule_of_thirds(cx: f32, cy: f32, n_positions: usize) -> (f32, f32) {
    if n_positions != 1 {
        // Multi-subject frame: keep the balanced centroid.
        return (cx, cy);
    }

    // Snap to the nearest horizontal third.
    let thirds = [1.0_f32 / 3.0, 2.0 / 3.0];
    let best_x = *thirds
        .iter()
        .min_by(|&&a, &&b| {
            (cx - a)
                .abs()
                .partial_cmp(&(cx - b).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(&cx);
    // Only nudge if the subject is reasonably close to a third line (within 25%).
    let nudge_x = if (cx - best_x).abs() < 0.25 {
        best_x
    } else {
        cx
    };

    // Keep vertical position as-is for single subjects (face-centred framing).
    (nudge_x, cy)
}

/// Build a crop window centred on `(focus_x, focus_y)` with given dimensions.
fn build_crop(focus_x: f32, focus_y: f32, width: f32, height: f32) -> CropWindow {
    let x = focus_x - width / 2.0;
    let y = focus_y - height / 2.0;
    CropWindow {
        x,
        y,
        width,
        height,
    }
    .clamped()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn pos(cx: f32, cy: f32) -> SubjectPosition {
        SubjectPosition::new(cx, cy, 1.0, 0)
    }

    fn pos_conf(cx: f32, cy: f32, conf: f32) -> SubjectPosition {
        SubjectPosition::new(cx, cy, conf, 0)
    }

    // CropWindow tests -------------------------------------------------------

    #[test]
    fn test_crop_window_centered_valid() {
        let cw = CropWindow::centered(9.0 / 16.0, 1.0);
        assert!(cw.validate(), "centred crop should be valid: {cw:?}");
    }

    #[test]
    fn test_crop_window_validate_out_of_bounds() {
        let bad = CropWindow {
            x: 0.8,
            y: 0.0,
            width: 0.5,
            height: 1.0,
        };
        assert!(!bad.validate());
    }

    #[test]
    fn test_crop_window_clamped() {
        let bad = CropWindow {
            x: 0.8,
            y: 0.0,
            width: 0.5,
            height: 1.0,
        };
        let good = bad.clamped();
        assert!(good.validate(), "clamped crop should be valid: {good:?}");
    }

    #[test]
    fn test_crop_window_blend_midpoint() {
        let a = CropWindow {
            x: 0.0,
            y: 0.0,
            width: 0.5,
            height: 1.0,
        };
        let b = CropWindow {
            x: 0.5,
            y: 0.0,
            width: 0.5,
            height: 1.0,
        };
        let mid = a.blend(&b, 0.5);
        assert!((mid.x - 0.25).abs() < 1e-5, "mid.x={}", mid.x);
    }

    // ReframeTarget tests ----------------------------------------------------

    #[test]
    fn test_vertical_short_aspect_ratio() {
        let ar = ReframeTarget::VerticalShort.aspect_ratio();
        assert!((ar - 9.0 / 16.0).abs() < 1e-5);
    }

    #[test]
    fn test_square_aspect_ratio() {
        let ar = ReframeTarget::Square.aspect_ratio();
        assert!((ar - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_custom_aspect_ratio() {
        let ar = ReframeTarget::Custom {
            w_ratio: 4.0,
            h_ratio: 3.0,
        }
        .aspect_ratio();
        assert!((ar - 4.0 / 3.0).abs() < 1e-5);
    }

    // ReframeAnalyzer::suggest_crop tests ------------------------------------

    #[test]
    fn test_empty_positions_yields_centered_crop() {
        let windows = ReframeAnalyzer::suggest_crop(&[], ReframeTarget::VerticalShort);
        assert_eq!(windows.len(), 1);
        let w = &windows[0];
        // Should be centred: x ≈ (1 - crop_width) / 2
        let cw = ReframeTarget::VerticalShort.crop_width_fraction();
        let expected_x = (1.0 - cw) / 2.0;
        assert!(
            (w.x - expected_x).abs() < 0.05,
            "x={} expected≈{expected_x}",
            w.x
        );
    }

    #[test]
    fn test_centered_subject_yields_centered_crop() {
        // Two subjects symmetrically placed so the centroid is exactly 0.5, 0.5.
        // With two subjects, rule-of-thirds nudge is not applied, so the crop
        // should be centred.
        let positions = vec![pos(0.3, 0.5), pos(0.7, 0.5)];
        let windows = ReframeAnalyzer::suggest_crop(&positions, ReframeTarget::Square);
        assert_eq!(windows.len(), 2);
        let w = &windows[0];
        let crop_centre_x = w.x + w.width / 2.0;
        assert!(
            (crop_centre_x - 0.5).abs() < 0.05,
            "crop centre={crop_centre_x}"
        );
    }

    #[test]
    fn test_off_center_subject_shifts_crop() {
        // Subject far right (0.85) vs far left (0.15) — both with 2-subject
        // multi-position inputs so rule-of-thirds is not applied, giving clean
        // centroid comparisons.
        let right_positions = vec![pos(0.6, 0.5), pos(0.9, 0.5)];
        let left_positions = vec![pos(0.1, 0.5), pos(0.4, 0.5)];
        let right_windows =
            ReframeAnalyzer::suggest_crop(&right_positions, ReframeTarget::VerticalShort);
        let left_windows =
            ReframeAnalyzer::suggest_crop(&left_positions, ReframeTarget::VerticalShort);
        // A right-leaning subject should produce a more rightward crop x than a left-leaning one.
        assert!(
            right_windows[0].x > left_windows[0].x,
            "right crop should be more rightward: right.x={} left.x={}",
            right_windows[0].x,
            left_windows[0].x
        );
    }

    #[test]
    fn test_crop_window_is_valid_for_all_positions() {
        let positions: Vec<SubjectPosition> = (0..10).map(|i| pos(i as f32 / 10.0, 0.5)).collect();
        let windows = ReframeAnalyzer::suggest_crop(&positions, ReframeTarget::VerticalShort);
        for (i, w) in windows.iter().enumerate() {
            assert!(w.validate(), "window {i} is invalid: {w:?}");
        }
    }

    #[test]
    fn test_multiple_subjects_no_rule_of_thirds_nudge() {
        // Two subjects at different positions → centroid is used directly.
        let positions = vec![pos(0.2, 0.5), pos(0.8, 0.5)];
        let windows = ReframeAnalyzer::suggest_crop(&positions, ReframeTarget::Square);
        assert_eq!(windows.len(), 2);
        // Centroid is at 0.5, 0.5 → crop should be centred.
        let cx = windows[0].x + windows[0].width / 2.0;
        assert!((cx - 0.5).abs() < 0.05, "cx={cx}");
    }

    #[test]
    fn test_confidence_weighting() {
        // High-confidence subject at right side should dominate centroid.
        let positions = vec![
            pos_conf(0.1, 0.5, 0.1), // weak left subject
            pos_conf(0.9, 0.5, 0.9), // strong right subject
        ];
        let windows = ReframeAnalyzer::suggest_crop(&positions, ReframeTarget::Square);
        let cx = windows[0].x + windows[0].width / 2.0;
        // Centroid should be skewed toward 0.9.
        assert!(cx > 0.5, "centroid should lean right, cx={cx}");
    }

    // smooth_trajectory tests ------------------------------------------------

    #[test]
    fn test_smooth_trajectory_empty() {
        let result = ReframeAnalyzer::smooth_trajectory(&[], 0.5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_smooth_trajectory_no_smoothing() {
        let windows = vec![
            CropWindow::centered(0.5, 1.0),
            CropWindow {
                x: 0.3,
                y: 0.0,
                width: 0.5,
                height: 1.0,
            },
        ];
        let result = ReframeAnalyzer::smooth_trajectory(&windows, 0.0);
        assert_eq!(result.len(), 2);
        assert!((result[1].x - 0.3).abs() < 1e-5);
    }

    #[test]
    fn test_smooth_trajectory_reduces_jump() {
        // Sudden shift from x=0.0 to x=0.5; with alpha=0.8 the smoothed x should be < 0.5.
        let windows = vec![
            CropWindow {
                x: 0.0,
                y: 0.0,
                width: 0.4,
                height: 1.0,
            },
            CropWindow {
                x: 0.4,
                y: 0.0,
                width: 0.4,
                height: 1.0,
            },
        ];
        let result = ReframeAnalyzer::smooth_trajectory(&windows, 0.8);
        assert_eq!(result.len(), 2);
        assert!(
            result[1].x < 0.4,
            "smoothed x should be less than 0.4, got {}",
            result[1].x
        );
        assert!(
            result[1].x > 0.0,
            "smoothed x should be > 0, got {}",
            result[1].x
        );
    }

    #[test]
    fn test_smooth_trajectory_preserves_length() {
        let windows: Vec<CropWindow> = (0..8)
            .map(|i| CropWindow::centered(0.5 + i as f32 * 0.01, 1.0).clamped())
            .collect();
        let result = ReframeAnalyzer::smooth_trajectory(&windows, 0.6);
        assert_eq!(result.len(), windows.len());
    }

    #[test]
    fn test_smooth_trajectory_all_valid() {
        let windows: Vec<CropWindow> = (0..5)
            .map(|i| {
                let x = i as f32 * 0.1;
                CropWindow {
                    x,
                    y: 0.0,
                    width: 0.5,
                    height: 1.0,
                }
                .clamped()
            })
            .collect();
        let result = ReframeAnalyzer::smooth_trajectory(&windows, 0.7);
        for (i, w) in result.iter().enumerate() {
            assert!(w.validate(), "smoothed window {i} invalid: {w:?}");
        }
    }
}
