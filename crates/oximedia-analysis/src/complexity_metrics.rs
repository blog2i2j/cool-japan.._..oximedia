#![allow(dead_code)]
//! Content complexity metrics for video frames.
//!
//! This module provides multiple complementary measures of visual complexity:
//!
//! - **DCT Variance** — The variance of 8×8 DCT AC coefficients across a frame
//!   correlates strongly with encoding difficulty and perceived texture richness.
//! - **Gradient Energy** — Sum of squared Sobel gradient magnitudes, normalised
//!   per pixel; captures fine-detail density.
//! - **Temporal Complexity Trend** — Frame-to-frame change in the above metrics,
//!   tracked in a sliding window to detect editing cuts, dissolves, and
//!   panning-induced complexity spikes.
//! - **Block Variance Map** — Per-block (8×8) luminance variance, exported as a
//!   flat `Vec<f64>` for downstream use (e.g. ROI-based bitrate allocation).
//! - **Overall Complexity Score** — A weighted combination of DCT variance and
//!   gradient energy, normalised to [0, 1].

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for complexity metric computation.
#[derive(Debug, Clone)]
pub struct ComplexityConfig {
    /// Size of DCT blocks; only 8 is supported (matches MPEG/JPEG standard).
    pub dct_block_size: usize,
    /// Weight of DCT variance in the overall score (0.0..1.0).
    /// Gradient energy weight = 1.0 - dct_weight.
    pub dct_weight: f64,
    /// Window size (frames) for temporal trend computation.
    pub trend_window: usize,
    /// Normalisation constant for gradient energy (max expected value per
    /// pixel, empirically ~2500 for typical 8-bit Sobel magnitudes).
    pub gradient_norm: f64,
    /// Normalisation constant for DCT AC variance (max expected, ~4000 for
    /// high-frequency texture).
    pub dct_norm: f64,
}

impl Default for ComplexityConfig {
    fn default() -> Self {
        Self {
            dct_block_size: 8,
            dct_weight: 0.5,
            trend_window: 30,
            gradient_norm: 2500.0,
            dct_norm: 4000.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-frame result
// ---------------------------------------------------------------------------

/// Complexity metrics for a single video frame.
#[derive(Debug, Clone)]
pub struct FrameComplexity {
    /// Frame index.
    pub frame_index: usize,
    /// Mean squared AC DCT coefficient averaged over all 8×8 blocks.
    pub dct_variance: f64,
    /// Mean squared Sobel gradient magnitude per pixel.
    pub gradient_energy: f64,
    /// Per-block luminance variance map (row-major, 8×8 blocks).
    pub block_variance_map: Vec<f64>,
    /// Number of block columns in the variance map.
    pub map_cols: usize,
    /// Number of block rows in the variance map.
    pub map_rows: usize,
    /// Overall complexity score in [0, 1].
    pub score: f64,
}

// ---------------------------------------------------------------------------
// Trend result
// ---------------------------------------------------------------------------

/// Temporal complexity trend over the sliding window.
#[derive(Debug, Clone, Copy)]
pub struct ComplexityTrend {
    /// Mean overall complexity score over the window.
    pub mean_score: f64,
    /// Standard deviation of scores in the window.
    pub std_score: f64,
    /// Frame-to-frame change rate (mean |Δscore|).
    pub change_rate: f64,
    /// Maximum score in the window.
    pub max_score: f64,
    /// Minimum score in the window.
    pub min_score: f64,
}

// ---------------------------------------------------------------------------
// Analyzer
// ---------------------------------------------------------------------------

/// Stateful complexity metrics analyzer.
pub struct ComplexityAnalyzer {
    config: ComplexityConfig,
    /// Rolling window of per-frame overall scores for trend computation.
    score_window: VecDeque<f64>,
    /// Per-frame complexity records (all frames since last reset).
    records: Vec<FrameComplexity>,
}

impl ComplexityAnalyzer {
    /// Create a new analyzer with default configuration.
    pub fn new() -> Self {
        Self::with_config(ComplexityConfig::default())
    }

    /// Create a new analyzer with custom configuration.
    pub fn with_config(config: ComplexityConfig) -> Self {
        let cap = config.trend_window;
        Self {
            config,
            score_window: VecDeque::with_capacity(cap),
            records: Vec::new(),
        }
    }

    /// Process one Y-plane frame and store results.
    pub fn process_frame(
        &mut self,
        y_plane: &[u8],
        width: usize,
        height: usize,
        frame_index: usize,
    ) -> FrameComplexity {
        let result = compute_frame_complexity(y_plane, width, height, frame_index, &self.config);
        // Update sliding window
        if self.score_window.len() >= self.config.trend_window {
            self.score_window.pop_front();
        }
        self.score_window.push_back(result.score);
        self.records.push(result.clone());
        result
    }

    /// Compute temporal complexity trend from the current sliding window.
    pub fn trend(&self) -> Option<ComplexityTrend> {
        let n = self.score_window.len();
        if n == 0 {
            return None;
        }
        let scores: Vec<f64> = self.score_window.iter().copied().collect();
        let mean = scores.iter().sum::<f64>() / n as f64;
        let variance =
            scores.iter().map(|s| (s - mean) * (s - mean)).sum::<f64>() / n as f64;
        let std_score = variance.sqrt();

        let change_rate = if n > 1 {
            scores.windows(2).map(|w| (w[1] - w[0]).abs()).sum::<f64>() / (n - 1) as f64
        } else {
            0.0
        };

        let max_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let min_score = scores.iter().copied().fold(f64::INFINITY, f64::min);

        Some(ComplexityTrend {
            mean_score: mean,
            std_score,
            change_rate,
            max_score,
            min_score,
        })
    }

    /// Returns all per-frame complexity records collected so far.
    pub fn records(&self) -> &[FrameComplexity] {
        &self.records
    }

    /// Reset analyzer state.
    pub fn reset(&mut self) {
        self.score_window.clear();
        self.records.clear();
    }
}

impl Default for ComplexityAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Core computation
// ---------------------------------------------------------------------------

/// Compute all complexity metrics for one frame.
fn compute_frame_complexity(
    y_plane: &[u8],
    width: usize,
    height: usize,
    frame_index: usize,
    config: &ComplexityConfig,
) -> FrameComplexity {
    if width == 0 || height == 0 || y_plane.is_empty() {
        return FrameComplexity {
            frame_index,
            dct_variance: 0.0,
            gradient_energy: 0.0,
            block_variance_map: Vec::new(),
            map_cols: 0,
            map_rows: 0,
            score: 0.0,
        };
    }

    let bs = config.dct_block_size.max(2);
    let map_cols = width.div_ceil(bs);
    let map_rows = height.div_ceil(bs);
    let (dct_variance, block_variance_map) =
        compute_dct_variance(y_plane, width, height, bs, map_cols, map_rows);

    let gradient_energy = compute_gradient_energy(y_plane, width, height);

    // Normalise and blend into overall score
    let norm_dct = (dct_variance / config.dct_norm).min(1.0);
    let norm_grad = (gradient_energy / config.gradient_norm).min(1.0);
    let score = config.dct_weight * norm_dct + (1.0 - config.dct_weight) * norm_grad;

    FrameComplexity {
        frame_index,
        dct_variance,
        gradient_energy,
        block_variance_map,
        map_cols,
        map_rows,
        score: score.clamp(0.0, 1.0),
    }
}

/// Compute mean-squared AC DCT coefficients over 8×8 blocks.
///
/// Uses the separable 1-D DCT-II applied to each 8×8 block.
/// Returns (mean_ac_variance, per-block variance map).
fn compute_dct_variance(
    y_plane: &[u8],
    width: usize,
    height: usize,
    bs: usize,
    map_cols: usize,
    map_rows: usize,
) -> (f64, Vec<f64>) {
    let mut block_buf = [0.0_f64; 64]; // max 8×8
    let mut map = vec![0.0_f64; map_cols * map_rows];
    let mut total_ac_sum = 0.0_f64;
    let mut block_count = 0usize;

    for by in 0..map_rows {
        for bx in 0..map_cols {
            let y0 = by * bs;
            let x0 = bx * bs;
            let bh = bs.min(height - y0);
            let bw = bs.min(width - x0);
            let n = bw * bh;
            if n == 0 {
                continue;
            }

            // Fill block buffer (pad with edge values if needed)
            for r in 0..bs {
                for c in 0..bs {
                    let yr = (y0 + r.min(bh - 1)).min(height - 1);
                    let xc = (x0 + c.min(bw - 1)).min(width - 1);
                    block_buf[r * bs + c] = f64::from(y_plane[yr * width + xc]) - 128.0;
                }
            }

            // 1-D DCT on rows then columns (separable)
            let block_slice = &mut block_buf[..bs * bs];
            dct2_rows(block_slice, bs);
            dct2_cols(block_slice, bs);

            // AC variance = mean of squared non-DC coefficients
            let ac_sum: f64 = block_slice.iter().enumerate()
                .filter(|&(idx, _)| idx != 0)  // skip DC
                .map(|(_, &v)| v * v)
                .sum();
            let ac_var = ac_sum / (bs * bs - 1).max(1) as f64;

            // Per-block luminance variance (simpler metric for the map)
            let mean_luma: f64 = (0..bh)
                .flat_map(|r| (0..bw).map(move |c| (r, c)))
                .map(|(r, c)| f64::from(y_plane[(y0 + r) * width + (x0 + c)]))
                .sum::<f64>()
                / n as f64;
            let luma_var = (0..bh)
                .flat_map(|r| (0..bw).map(move |c| (r, c)))
                .map(|(r, c)| {
                    let d = f64::from(y_plane[(y0 + r) * width + (x0 + c)]) - mean_luma;
                    d * d
                })
                .sum::<f64>()
                / n as f64;

            map[by * map_cols + bx] = luma_var;
            total_ac_sum += ac_var;
            block_count += 1;
        }
    }

    let mean_ac_var = if block_count > 0 {
        total_ac_sum / block_count as f64
    } else {
        0.0
    };

    (mean_ac_var, map)
}

/// Apply DCT-II to each row of an (n×n) flat buffer.
fn dct2_rows(buf: &mut [f64], n: usize) {
    let mut tmp = vec![0.0_f64; n];
    for r in 0..n {
        let row = &buf[r * n..(r + 1) * n];
        dct2_1d(row, &mut tmp);
        buf[r * n..(r + 1) * n].copy_from_slice(&tmp);
    }
}

/// Apply DCT-II to each column of an (n×n) flat buffer.
fn dct2_cols(buf: &mut [f64], n: usize) {
    let mut col = vec![0.0_f64; n];
    let mut tmp = vec![0.0_f64; n];
    for c in 0..n {
        for r in 0..n {
            col[r] = buf[r * n + c];
        }
        dct2_1d(&col, &mut tmp);
        for r in 0..n {
            buf[r * n + c] = tmp[r];
        }
    }
}

/// Naive DCT-II for small vectors (n ≤ 8, called at most 2×16 = 32 times per
/// 8×8 block, so O(n²) is acceptable here).
fn dct2_1d(input: &[f64], output: &mut [f64]) {
    let n = input.len();
    let scale = std::f64::consts::PI / (2.0 * n as f64);
    for k in 0..n {
        let mut sum = 0.0_f64;
        for (j, &v) in input.iter().enumerate() {
            sum += v * (scale * (2 * j + 1) as f64 * k as f64).cos();
        }
        output[k] = sum;
    }
}

/// Compute mean squared Sobel gradient magnitude per pixel.
fn compute_gradient_energy(y_plane: &[u8], width: usize, height: usize) -> f64 {
    if width < 3 || height < 3 {
        return 0.0;
    }
    let mut energy_sum = 0.0_f64;
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

            energy_sum += gx * gx + gy * gy;
        }
    }

    if interior > 0 {
        energy_sum / interior as f64
    } else {
        0.0
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

    fn checkerboard(w: usize, h: usize) -> Vec<u8> {
        let mut data = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                data[y * w + x] = if (x + y) % 2 == 0 { 255 } else { 0 };
            }
        }
        data
    }

    #[test]
    fn test_flat_frame_zero_complexity() {
        let mut analyzer = ComplexityAnalyzer::new();
        let frame = flat_frame(64, 64, 128);
        let fc = analyzer.process_frame(&frame, 64, 64, 0);
        assert!(
            fc.score < 0.01,
            "flat frame should have near-zero complexity, got {}",
            fc.score
        );
        assert!(fc.gradient_energy < 0.01);
    }

    #[test]
    fn test_complex_frame_high_score() {
        let mut analyzer = ComplexityAnalyzer::new();
        let frame = checkerboard(64, 64);
        let fc = analyzer.process_frame(&frame, 64, 64, 0);
        assert!(
            fc.score > 0.1,
            "checkerboard should have significant complexity, got {}",
            fc.score
        );
    }

    #[test]
    fn test_score_range() {
        let mut analyzer = ComplexityAnalyzer::new();
        let frame = checkerboard(64, 64);
        let fc = analyzer.process_frame(&frame, 64, 64, 0);
        assert!(fc.score >= 0.0 && fc.score <= 1.0, "score={}", fc.score);
    }

    #[test]
    fn test_block_variance_map_dimensions() {
        let config = ComplexityConfig {
            dct_block_size: 8,
            ..Default::default()
        };
        let mut analyzer = ComplexityAnalyzer::with_config(config);
        let frame = flat_frame(64, 48, 100);
        let fc = analyzer.process_frame(&frame, 64, 48, 0);
        assert_eq!(fc.map_cols, 8); // 64/8
        assert_eq!(fc.map_rows, 6); // 48/8
        assert_eq!(fc.block_variance_map.len(), 48);
    }

    #[test]
    fn test_trend_single_frame() {
        let mut analyzer = ComplexityAnalyzer::new();
        let frame = flat_frame(32, 32, 128);
        analyzer.process_frame(&frame, 32, 32, 0);
        let trend = analyzer.trend().expect("should have trend after 1 frame");
        assert_eq!(trend.max_score, trend.min_score);
        assert_eq!(trend.change_rate, 0.0);
    }

    #[test]
    fn test_trend_score_increases_with_complexity() {
        let mut analyzer = ComplexityAnalyzer::new();
        let flat = flat_frame(32, 32, 128);
        let complex = checkerboard(32, 32);

        for _ in 0..10 {
            analyzer.process_frame(&flat, 32, 32, 0);
        }
        let low_trend = analyzer.trend().expect("trend expected");

        analyzer.reset();
        for _ in 0..10 {
            analyzer.process_frame(&complex, 32, 32, 0);
        }
        let high_trend = analyzer.trend().expect("trend expected");

        assert!(
            high_trend.mean_score > low_trend.mean_score,
            "complex frames should have higher mean score"
        );
    }

    #[test]
    fn test_empty_frame_handled() {
        let mut analyzer = ComplexityAnalyzer::new();
        let fc = analyzer.process_frame(&[], 0, 0, 0);
        assert_eq!(fc.score, 0.0);
        assert!(fc.block_variance_map.is_empty());
    }

    #[test]
    fn test_records_count() {
        let mut analyzer = ComplexityAnalyzer::new();
        let frame = flat_frame(16, 16, 100);
        for i in 0..5 {
            analyzer.process_frame(&frame, 16, 16, i);
        }
        assert_eq!(analyzer.records().len(), 5);
    }

    #[test]
    fn test_reset_clears_records() {
        let mut analyzer = ComplexityAnalyzer::new();
        let frame = flat_frame(16, 16, 100);
        analyzer.process_frame(&frame, 16, 16, 0);
        analyzer.reset();
        assert!(analyzer.records().is_empty());
        assert!(analyzer.trend().is_none());
    }

    #[test]
    fn test_trend_window_eviction() {
        let config = ComplexityConfig {
            trend_window: 3,
            ..Default::default()
        };
        let mut analyzer = ComplexityAnalyzer::with_config(config);
        let flat = flat_frame(32, 32, 128);
        let complex = checkerboard(32, 32);

        // Push 3 flat then 3 complex; window should only contain complex
        for i in 0..3 {
            analyzer.process_frame(&flat, 32, 32, i);
        }
        for i in 3..6 {
            analyzer.process_frame(&complex, 32, 32, i);
        }
        let trend = analyzer.trend().expect("trend expected");
        // After eviction only high-complexity frames remain in window
        assert!(trend.mean_score > 0.01, "mean_score={}", trend.mean_score);
    }

    #[test]
    fn test_dct_variance_increases_with_texture() {
        let mut analyzer = ComplexityAnalyzer::new();
        let flat = flat_frame(64, 64, 100);
        let fc_flat = analyzer.process_frame(&flat, 64, 64, 0);
        let complex = checkerboard(64, 64);
        let fc_complex = analyzer.process_frame(&complex, 64, 64, 1);
        assert!(
            fc_complex.dct_variance > fc_flat.dct_variance,
            "checkerboard dct_var={} should exceed flat dct_var={}",
            fc_complex.dct_variance,
            fc_flat.dct_variance
        );
    }
}
