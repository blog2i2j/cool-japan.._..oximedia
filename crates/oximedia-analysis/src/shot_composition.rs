#![allow(dead_code)]
//! Shot composition analysis for video frames.
//!
//! This module analyses the spatial arrangement of visual content within a
//! frame, measuring adherence to classical composition principles:
//!
//! - **Rule of Thirds** — Identifies whether salient regions align with the
//!   classic 1/3 grid intersections.
//! - **Symmetry** — Measures horizontal and vertical luminance symmetry.
//! - **Leading Lines** — Detects strong directional gradients that guide the
//!   eye through the frame.
//! - **Visual Weight Balance** — Quantifies the distribution of visual "mass"
//!   (via luminance contrast) across quadrants.
//! - **Overall Composition Score** — A weighted blend of the above metrics,
//!   normalised to [0, 1].
//!
//! All metrics operate on the Y (luma) plane only for efficiency.

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for shot composition analysis.
#[derive(Debug, Clone)]
pub struct CompositionConfig {
    /// Number of equal-width columns used to build the composition grid.
    /// 3 = rule-of-thirds.  Must be >= 2.
    pub grid_cols: usize,
    /// Number of equal-height rows used to build the composition grid.
    /// 3 = rule-of-thirds.  Must be >= 2.
    pub grid_rows: usize,
    /// Radius (as a fraction of the min frame dimension) around each thirds
    /// intersection point considered "on" the power point.  Default 0.1.
    pub power_point_radius: f64,
    /// Gradient magnitude threshold used for leading-line detection (0..255).
    pub line_threshold: u8,
    /// Weight of rule-of-thirds in the overall score (0.0..1.0).
    pub weight_thirds: f64,
    /// Weight of symmetry in the overall score (0.0..1.0).
    pub weight_symmetry: f64,
    /// Weight of leading-lines in the overall score (0.0..1.0).
    pub weight_lines: f64,
    /// Weight of visual-weight balance in the overall score (0.0..1.0).
    /// (Remaining weight after the three above if they don't sum to 1.0.)
    pub weight_balance: f64,
}

impl Default for CompositionConfig {
    fn default() -> Self {
        Self {
            grid_cols: 3,
            grid_rows: 3,
            power_point_radius: 0.10,
            line_threshold: 40,
            weight_thirds: 0.35,
            weight_symmetry: 0.25,
            weight_lines: 0.20,
            weight_balance: 0.20,
        }
    }
}

// ---------------------------------------------------------------------------
// Sub-metric results
// ---------------------------------------------------------------------------

/// Rule-of-thirds analysis result.
#[derive(Debug, Clone)]
pub struct RuleOfThirdsResult {
    /// Fraction of high-contrast pixels (above mean luminance) that lie within
    /// the power-point radius of any grid intersection.  Range [0, 1].
    pub power_point_coverage: f64,
    /// Per-cell mean luminance (row-major).
    pub cell_luminance: Vec<f64>,
    /// Number of grid columns.
    pub cols: usize,
    /// Number of grid rows.
    pub rows: usize,
}

/// Symmetry analysis result.
#[derive(Debug, Clone, Copy)]
pub struct SymmetryResult {
    /// Horizontal symmetry score (1 = perfectly symmetric, 0 = no symmetry).
    pub horizontal: f64,
    /// Vertical symmetry score (1 = perfectly symmetric, 0 = no symmetry).
    pub vertical: f64,
    /// Combined symmetry score (mean of horizontal and vertical).
    pub combined: f64,
}

/// Leading-lines analysis result.
#[derive(Debug, Clone)]
pub struct LeadingLinesResult {
    /// Fraction of strong-gradient pixels across the entire frame.
    pub edge_fraction: f64,
    /// Dominant edge orientation in radians [−π/2, π/2].
    pub dominant_angle: f64,
    /// Orientation coherence: 1 = all edges aligned, 0 = random directions.
    pub coherence: f64,
    /// Histogram of edge orientations (8 equal-width bins over [−π/2, π/2]).
    pub angle_histogram: Vec<f64>,
}

/// Visual weight balance result.
#[derive(Debug, Clone, Copy)]
pub struct BalanceResult {
    /// Fraction of total luminance-contrast energy in the left half.
    pub left_fraction: f64,
    /// Fraction in the right half.
    pub right_fraction: f64,
    /// Fraction in the top half.
    pub top_fraction: f64,
    /// Fraction in the bottom half.
    pub bottom_fraction: f64,
    /// Balance score: 1 = perfectly balanced, 0 = all energy in one quadrant.
    pub score: f64,
}

/// Complete shot composition analysis result for one frame.
#[derive(Debug, Clone)]
pub struct CompositionResult {
    /// Frame index.
    pub frame_index: usize,
    /// Rule-of-thirds analysis.
    pub rule_of_thirds: RuleOfThirdsResult,
    /// Symmetry analysis.
    pub symmetry: SymmetryResult,
    /// Leading-lines analysis.
    pub leading_lines: LeadingLinesResult,
    /// Visual weight balance.
    pub balance: BalanceResult,
    /// Overall composition quality score in [0, 1].
    pub score: f64,
}

// ---------------------------------------------------------------------------
// Analyzer
// ---------------------------------------------------------------------------

/// Shot composition analyzer.  Stateless per-frame; call [`analyze`] for each
/// frame independently, or use [`process_batch`] for a sequence.
pub struct CompositionAnalyzer {
    config: CompositionConfig,
}

impl CompositionAnalyzer {
    /// Create a new analyzer with default configuration.
    pub fn new() -> Self {
        Self::with_config(CompositionConfig::default())
    }

    /// Create a new analyzer with custom configuration.
    pub fn with_config(config: CompositionConfig) -> Self {
        let cols = config.grid_cols.max(2);
        let rows = config.grid_rows.max(2);
        Self {
            config: CompositionConfig { grid_cols: cols, grid_rows: rows, ..config },
        }
    }

    /// Analyse the composition of a single Y-plane frame.
    ///
    /// Returns `None` if the frame is too small to analyse (< 4×4 pixels).
    pub fn analyze(
        &self,
        y_plane: &[u8],
        width: usize,
        height: usize,
        frame_index: usize,
    ) -> Option<CompositionResult> {
        if width < 4 || height < 4 || y_plane.len() < width * height {
            return None;
        }

        let rule_of_thirds = compute_rule_of_thirds(y_plane, width, height, &self.config);
        let symmetry = compute_symmetry(y_plane, width, height);
        let leading_lines =
            compute_leading_lines(y_plane, width, height, self.config.line_threshold);
        let balance = compute_balance(y_plane, width, height);

        // Normalise weights
        let total_w =
            self.config.weight_thirds + self.config.weight_symmetry
            + self.config.weight_lines + self.config.weight_balance;
        let (wt, ws, wl, wb) = if total_w > 0.0 {
            (
                self.config.weight_thirds / total_w,
                self.config.weight_symmetry / total_w,
                self.config.weight_lines / total_w,
                self.config.weight_balance / total_w,
            )
        } else {
            (0.25, 0.25, 0.25, 0.25)
        };

        let score = (wt * rule_of_thirds.power_point_coverage
            + ws * symmetry.combined
            + wl * leading_lines.coherence
            + wb * balance.score)
            .clamp(0.0, 1.0);

        Some(CompositionResult {
            frame_index,
            rule_of_thirds,
            symmetry,
            leading_lines,
            balance,
            score,
        })
    }

    /// Analyse a batch of frames and return results.
    pub fn process_batch(
        &self,
        frames: &[(&[u8], usize, usize)],
    ) -> Vec<CompositionResult> {
        frames
            .iter()
            .enumerate()
            .filter_map(|(idx, &(plane, w, h))| self.analyze(plane, w, h, idx))
            .collect()
    }
}

impl Default for CompositionAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Rule of thirds
// ---------------------------------------------------------------------------

fn compute_rule_of_thirds(
    y_plane: &[u8],
    width: usize,
    height: usize,
    config: &CompositionConfig,
) -> RuleOfThirdsResult {
    let cols = config.grid_cols;
    let rows = config.grid_rows;

    // Per-cell mean luminance
    let cell_w = width as f64 / cols as f64;
    let cell_h = height as f64 / rows as f64;

    let mut cell_luminance = vec![0.0_f64; cols * rows];
    let mut cell_counts = vec![0usize; cols * rows];

    for py in 0..height {
        let r = ((py as f64 / cell_h) as usize).min(rows - 1);
        for px in 0..width {
            let c = ((px as f64 / cell_w) as usize).min(cols - 1);
            cell_luminance[r * cols + c] += f64::from(y_plane[py * width + px]);
            cell_counts[r * cols + c] += 1;
        }
    }
    for (lum, cnt) in cell_luminance.iter_mut().zip(cell_counts.iter()) {
        if *cnt > 0 {
            *lum /= *cnt as f64;
        }
    }

    // Mean luminance of the whole frame (used to identify "salient" pixels)
    let mean_luma =
        y_plane.iter().map(|&v| f64::from(v)).sum::<f64>() / (width * height) as f64;

    // Power-point intersection coordinates (normalised 0..1)
    let mut power_points: Vec<(f64, f64)> = Vec::new();
    for ri in 1..rows {
        for ci in 1..cols {
            power_points.push((ci as f64 / cols as f64, ri as f64 / rows as f64));
        }
    }

    let radius = config.power_point_radius * width.min(height) as f64;
    let radius_sq = radius * radius;

    // Count salient pixels near power points
    let mut near_count = 0usize;
    let mut salient_count = 0usize;

    for py in 0..height {
        let fy = py as f64 + 0.5;
        for px in 0..width {
            let luma = f64::from(y_plane[py * width + px]);
            if luma <= mean_luma {
                continue;
            }
            salient_count += 1;
            let fx = px as f64 + 0.5;
            for &(nx, ny) in &power_points {
                let dx = fx - nx * width as f64;
                let dy = fy - ny * height as f64;
                if dx * dx + dy * dy <= radius_sq {
                    near_count += 1;
                    break;
                }
            }
        }
    }

    let power_point_coverage = if salient_count > 0 {
        near_count as f64 / salient_count as f64
    } else {
        0.0
    };

    RuleOfThirdsResult {
        power_point_coverage,
        cell_luminance,
        cols,
        rows,
    }
}

// ---------------------------------------------------------------------------
// Symmetry
// ---------------------------------------------------------------------------

fn compute_symmetry(y_plane: &[u8], width: usize, height: usize) -> SymmetryResult {
    let mid_x = width / 2;
    let mid_y = height / 2;

    // Horizontal symmetry: compare left vs mirrored right halves
    let mut h_diff_sum = 0.0_f64;
    let mut h_mag_sum = 0.0_f64;
    for py in 0..height {
        for px in 0..mid_x {
            let mirror_x = width - 1 - px;
            let left = f64::from(y_plane[py * width + px]);
            let right = f64::from(y_plane[py * width + mirror_x]);
            h_diff_sum += (left - right).abs();
            h_mag_sum += left + right;
        }
    }
    let h_sym = if h_mag_sum > 0.0 {
        1.0 - (h_diff_sum / h_mag_sum).min(1.0)
    } else {
        1.0
    };

    // Vertical symmetry: compare top vs mirrored bottom halves
    let mut v_diff_sum = 0.0_f64;
    let mut v_mag_sum = 0.0_f64;
    for py in 0..mid_y {
        let mirror_y = height - 1 - py;
        for px in 0..width {
            let top = f64::from(y_plane[py * width + px]);
            let bot = f64::from(y_plane[mirror_y * width + px]);
            v_diff_sum += (top - bot).abs();
            v_mag_sum += top + bot;
        }
    }
    let v_sym = if v_mag_sum > 0.0 {
        1.0 - (v_diff_sum / v_mag_sum).min(1.0)
    } else {
        1.0
    };

    SymmetryResult {
        horizontal: h_sym,
        vertical: v_sym,
        combined: (h_sym + v_sym) * 0.5,
    }
}

// ---------------------------------------------------------------------------
// Leading lines
// ---------------------------------------------------------------------------

fn compute_leading_lines(
    y_plane: &[u8],
    width: usize,
    height: usize,
    threshold: u8,
) -> LeadingLinesResult {
    if width < 3 || height < 3 {
        return LeadingLinesResult {
            edge_fraction: 0.0,
            dominant_angle: 0.0,
            coherence: 0.0,
            angle_histogram: vec![0.0; 8],
        };
    }

    let thresh = f64::from(threshold);
    let num_bins = 8usize;
    let bin_width = PI / num_bins as f64;
    let mut angle_hist = vec![0.0_f64; num_bins];
    let mut edge_count = 0usize;
    let interior = (width - 2) * (height - 2);

    for py in 1..height - 1 {
        for px in 1..width - 1 {
            let get = |x: usize, y: usize| -> f64 { f64::from(y_plane[y * width + x]) };

            let tl = get(px - 1, py - 1);
            let tc = get(px, py - 1);
            let tr = get(px + 1, py - 1);
            let ml = get(px - 1, py);
            let mr = get(px + 1, py);
            let bl = get(px - 1, py + 1);
            let bc = get(px, py + 1);
            let br = get(px + 1, py + 1);

            let gx = -tl + tr - 2.0 * ml + 2.0 * mr - bl + br;
            let gy = -tl - 2.0 * tc - tr + bl + 2.0 * bc + br;
            let mag = (gx * gx + gy * gy).sqrt();

            if mag >= thresh {
                edge_count += 1;
                // atan2 in [−π, π]; fold to [−π/2, π/2]
                let mut angle = gy.atan2(gx);
                if angle > PI / 2.0 {
                    angle -= PI;
                } else if angle < -PI / 2.0 {
                    angle += PI;
                }
                // Shift to [0, π] then bin
                let shifted = angle + PI / 2.0;
                let bin = ((shifted / bin_width) as usize).min(num_bins - 1);
                angle_hist[bin] += 1.0;
            }
        }
    }

    let edge_fraction = if interior > 0 {
        edge_count as f64 / interior as f64
    } else {
        0.0
    };

    // Normalise angle histogram
    let hist_total: f64 = angle_hist.iter().sum();
    let mut norm_hist = angle_hist.clone();
    if hist_total > 0.0 {
        for v in norm_hist.iter_mut() {
            *v /= hist_total;
        }
    }

    // Dominant angle (centre of the peak bin)
    let (peak_bin, _) = norm_hist
        .iter()
        .enumerate()
        .fold((0usize, 0.0_f64), |(mi, mv), (i, &v)| {
            if v > mv { (i, v) } else { (mi, mv) }
        });
    let dominant_angle = (peak_bin as f64 + 0.5) * bin_width - PI / 2.0;

    // Coherence: circular variance proxy (1 - entropy-like spread)
    let peak_fraction = norm_hist[peak_bin];
    let coherence = (peak_fraction * num_bins as f64 - 1.0)
        .max(0.0)
        .min(num_bins as f64 - 1.0)
        / (num_bins - 1) as f64;

    LeadingLinesResult {
        edge_fraction,
        dominant_angle,
        coherence,
        angle_histogram: norm_hist,
    }
}

// ---------------------------------------------------------------------------
// Visual weight balance
// ---------------------------------------------------------------------------

fn compute_balance(y_plane: &[u8], width: usize, height: usize) -> BalanceResult {
    let mid_x = width / 2;
    let mid_y = height / 2;

    // Visual weight = luminance contrast above global mean
    let mean_luma =
        y_plane.iter().map(|&v| f64::from(v)).sum::<f64>() / (width * height) as f64;

    let mut left = 0.0_f64;
    let mut right = 0.0_f64;
    let mut top = 0.0_f64;
    let mut bottom = 0.0_f64;

    for py in 0..height {
        for px in 0..width {
            let weight = (f64::from(y_plane[py * width + px]) - mean_luma).abs();
            if px < mid_x { left += weight; } else { right += weight; }
            if py < mid_y { top += weight; } else { bottom += weight; }
        }
    }

    let lr_total = left + right;
    let tb_total = top + bottom;

    let left_fraction = if lr_total > 0.0 { left / lr_total } else { 0.5 };
    let right_fraction = if lr_total > 0.0 { right / lr_total } else { 0.5 };
    let top_fraction = if tb_total > 0.0 { top / tb_total } else { 0.5 };
    let bottom_fraction = if tb_total > 0.0 { bottom / tb_total } else { 0.5 };

    // Balance score: 1 when fractions are 0.5/0.5, 0 when all energy in one half
    let lr_balance = 1.0 - (left_fraction - 0.5).abs() * 2.0;
    let tb_balance = 1.0 - (top_fraction - 0.5).abs() * 2.0;
    let score = ((lr_balance + tb_balance) * 0.5).clamp(0.0, 1.0);

    BalanceResult {
        left_fraction,
        right_fraction,
        top_fraction,
        bottom_fraction,
        score,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_frame(w: usize, h: usize, val: u8) -> Vec<u8> {
        vec![val; w * h]
    }

    fn vertical_split(w: usize, h: usize, left_val: u8, right_val: u8) -> Vec<u8> {
        let mut data = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                data[y * w + x] = if x < w / 2 { left_val } else { right_val };
            }
        }
        data
    }

    fn horizontal_split(w: usize, h: usize, top_val: u8, bot_val: u8) -> Vec<u8> {
        let mut data = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                data[y * w + x] = if y < h / 2 { top_val } else { bot_val };
            }
        }
        data
    }

    #[test]
    fn test_too_small_frame_returns_none() {
        let analyzer = CompositionAnalyzer::new();
        assert!(analyzer.analyze(&flat_frame(2, 2, 128), 2, 2, 0).is_none());
        assert!(analyzer.analyze(&flat_frame(3, 4, 128), 3, 4, 0).is_none());
    }

    #[test]
    fn test_score_range() {
        let analyzer = CompositionAnalyzer::new();
        let frame = flat_frame(64, 48, 128);
        let result = analyzer.analyze(&frame, 64, 48, 0).expect("should analyze");
        assert!(
            result.score >= 0.0 && result.score <= 1.0,
            "score={}",
            result.score
        );
    }

    #[test]
    fn test_symmetric_frame_high_symmetry() {
        let analyzer = CompositionAnalyzer::new();
        let frame = flat_frame(64, 64, 180);
        let result = analyzer.analyze(&frame, 64, 64, 0).expect("should analyze");
        // Flat frame is perfectly symmetric
        assert!(
            result.symmetry.horizontal > 0.99,
            "h_sym={}",
            result.symmetry.horizontal
        );
        assert!(
            result.symmetry.vertical > 0.99,
            "v_sym={}",
            result.symmetry.vertical
        );
    }

    #[test]
    fn test_asymmetric_frame_low_symmetry() {
        let analyzer = CompositionAnalyzer::new();
        let frame = vertical_split(64, 64, 0, 255);
        let result = analyzer.analyze(&frame, 64, 64, 0).expect("should analyze");
        assert!(
            result.symmetry.horizontal < 0.5,
            "expected low horizontal symmetry, got {}",
            result.symmetry.horizontal
        );
    }

    #[test]
    fn test_cell_luminance_dimensions() {
        let config = CompositionConfig {
            grid_cols: 3,
            grid_rows: 3,
            ..Default::default()
        };
        let analyzer = CompositionAnalyzer::with_config(config);
        let frame = flat_frame(90, 60, 100);
        let result = analyzer.analyze(&frame, 90, 60, 0).expect("should analyze");
        assert_eq!(result.rule_of_thirds.cols, 3);
        assert_eq!(result.rule_of_thirds.rows, 3);
        assert_eq!(result.rule_of_thirds.cell_luminance.len(), 9);
    }

    #[test]
    fn test_balanced_frame_high_balance_score() {
        let analyzer = CompositionAnalyzer::new();
        let frame = flat_frame(64, 64, 128);
        let result = analyzer.analyze(&frame, 64, 64, 0).expect("should analyze");
        // Flat frame has no contrast hence perfect balance
        assert!(
            result.balance.score >= 0.9,
            "expected high balance, got {}",
            result.balance.score
        );
    }

    #[test]
    fn test_imbalanced_frame_lower_balance_score() {
        let analyzer = CompositionAnalyzer::new();
        // Create a frame where the left quarter is very bright (255) and the
        // rest is dark (0).  The mean is ~63, so the bright strip has deviation
        // ~192 while the dark region has deviation ~63.  Left weight >> right.
        let w = 64usize;
        let h = 64usize;
        let mut frame = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w / 4 {
                frame[y * w + x] = 255;
            }
        }
        let result = analyzer.analyze(&frame, w, h, 0).expect("should analyze");
        // Left quarter is much brighter → left_fraction > right_fraction
        assert!(
            result.balance.left_fraction > result.balance.right_fraction,
            "left_fraction={} right_fraction={}",
            result.balance.left_fraction,
            result.balance.right_fraction
        );
        // Balance score should be less than 0.9 (not well balanced)
        assert!(
            result.balance.score < 0.9,
            "balance score={} should reflect imbalance",
            result.balance.score
        );
    }

    #[test]
    fn test_leading_lines_flat_frame() {
        let analyzer = CompositionAnalyzer::new();
        let frame = flat_frame(64, 64, 128);
        let result = analyzer.analyze(&frame, 64, 64, 0).expect("should analyze");
        assert!(
            result.leading_lines.edge_fraction < 0.01,
            "flat frame should have no edges"
        );
    }

    #[test]
    fn test_process_batch() {
        let analyzer = CompositionAnalyzer::new();
        let f1 = flat_frame(32, 32, 100);
        let f2 = flat_frame(32, 32, 200);
        let too_small = flat_frame(2, 2, 50);
        let frames: Vec<(&[u8], usize, usize)> = vec![
            (f1.as_slice(), 32, 32),
            (f2.as_slice(), 32, 32),
            (too_small.as_slice(), 2, 2),
        ];
        let results = analyzer.process_batch(&frames);
        // too_small is filtered out
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].frame_index, 0);
        assert_eq!(results[1].frame_index, 1);
    }

    #[test]
    fn test_angle_histogram_normalised() {
        let analyzer = CompositionAnalyzer::new();
        let mut frame = vec![0u8; 64 * 64];
        // Create strong vertical edges (horizontal gradient)
        for y in 0..64usize {
            for x in 0..64usize {
                frame[y * 64 + x] = if x % 8 < 4 { 200 } else { 20 };
            }
        }
        let result = analyzer.analyze(&frame, 64, 64, 0).expect("should analyze");
        let hist_sum: f64 = result.leading_lines.angle_histogram.iter().sum();
        assert!(
            (hist_sum - 1.0).abs() < 1e-6 || result.leading_lines.edge_fraction < 1e-9,
            "angle histogram should be normalised, sum={}",
            hist_sum
        );
    }
}
