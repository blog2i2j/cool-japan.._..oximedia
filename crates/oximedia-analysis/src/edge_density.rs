#![allow(dead_code)]
//! Edge density analysis for video frames.
//!
//! This module provides tools for measuring the spatial complexity of video
//! frames by analyzing edge density. Edge density is a key metric for:
//!
//! - **Encoding difficulty estimation** - Frames with more edges require more bits
//! - **Content classification** - Text overlays, graphics, and detailed scenes have high edge density
//! - **Quality assessment** - Loss of edges indicates blurring or compression artifacts
//! - **Scene complexity scoring** - Complements motion analysis for bitrate allocation

use std::collections::VecDeque;

/// Configuration for edge density analysis.
#[derive(Debug, Clone)]
pub struct EdgeDensityConfig {
    /// Gradient magnitude threshold for classifying a pixel as an edge.
    /// Range: 0-255. Lower values detect more subtle edges.
    pub gradient_threshold: u8,
    /// Size of the sliding window for temporal edge density tracking (in frames).
    pub window_size: usize,
    /// Block size for computing regional edge density maps.
    pub block_size: usize,
    /// Whether to compute directional edge statistics (horizontal vs vertical).
    pub compute_directional: bool,
}

impl Default for EdgeDensityConfig {
    fn default() -> Self {
        Self {
            gradient_threshold: 30,
            window_size: 30,
            block_size: 16,
            compute_directional: false,
        }
    }
}

/// Result of edge density analysis for a single frame.
#[derive(Debug, Clone)]
pub struct EdgeDensityResult {
    /// Frame index.
    pub frame_index: usize,
    /// Overall edge density (0.0 to 1.0).
    pub density: f64,
    /// Average gradient magnitude across all pixels.
    pub avg_gradient: f64,
    /// Maximum gradient magnitude found.
    pub max_gradient: f64,
    /// Horizontal edge density (0.0 to 1.0), if directional analysis is enabled.
    pub horizontal_density: Option<f64>,
    /// Vertical edge density (0.0 to 1.0), if directional analysis is enabled.
    pub vertical_density: Option<f64>,
    /// Regional edge density map (row-major, block-based).
    pub region_map: Vec<f64>,
    /// Number of block columns in the region map.
    pub region_cols: usize,
    /// Number of block rows in the region map.
    pub region_rows: usize,
}

/// Temporal edge density statistics over a sliding window.
#[derive(Debug, Clone)]
pub struct EdgeDensityStats {
    /// Mean edge density over the window.
    pub mean_density: f64,
    /// Standard deviation of edge density over the window.
    pub std_density: f64,
    /// Minimum edge density in the window.
    pub min_density: f64,
    /// Maximum edge density in the window.
    pub max_density: f64,
    /// Number of frames analyzed so far.
    pub frame_count: usize,
}

/// Edge density analyzer that processes video frames.
pub struct EdgeDensityAnalyzer {
    /// Configuration.
    config: EdgeDensityConfig,
    /// Sliding window of recent density values.
    density_history: VecDeque<f64>,
    /// Running sum for mean calculation.
    running_sum: f64,
    /// Running sum of squares for variance calculation.
    running_sum_sq: f64,
    /// Total frames processed.
    frame_count: usize,
}

impl EdgeDensityAnalyzer {
    /// Create a new edge density analyzer with default configuration.
    pub fn new() -> Self {
        Self::with_config(EdgeDensityConfig::default())
    }

    /// Create a new edge density analyzer with custom configuration.
    pub fn with_config(config: EdgeDensityConfig) -> Self {
        Self {
            density_history: VecDeque::with_capacity(config.window_size),
            config,
            running_sum: 0.0,
            running_sum_sq: 0.0,
            frame_count: 0,
        }
    }

    /// Compute the Sobel gradient magnitude at a given pixel.
    #[allow(clippy::cast_precision_loss)]
    fn sobel_gradient(y_plane: &[u8], width: usize, x: usize, y: usize) -> (f64, f64) {
        // Sobel kernels:
        // Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        // Gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        let get = |px: usize, py: usize| -> f64 { f64::from(y_plane[py * width + px]) };

        let tl = get(x - 1, y - 1);
        let tc = get(x, y - 1);
        let tr = get(x + 1, y - 1);
        let ml = get(x - 1, y);
        let mr = get(x + 1, y);
        let bl = get(x - 1, y + 1);
        let bc = get(x, y + 1);
        let br = get(x + 1, y + 1);

        let gx = -tl + tr - 2.0 * ml + 2.0 * mr - bl + br;
        let gy = -tl - 2.0 * tc - tr + bl + 2.0 * bc + br;

        (gx, gy)
    }

    /// Process a single video frame and return edge density results.
    #[allow(clippy::cast_precision_loss)]
    pub fn process_frame(
        &mut self,
        y_plane: &[u8],
        width: usize,
        height: usize,
        frame_index: usize,
    ) -> EdgeDensityResult {
        let threshold = f64::from(self.config.gradient_threshold);
        let interior_pixels = if width > 2 && height > 2 {
            (width - 2) * (height - 2)
        } else {
            return self.empty_result(frame_index);
        };

        let mut edge_count: usize = 0;
        let mut h_edge_count: usize = 0;
        let mut v_edge_count: usize = 0;
        let mut gradient_sum: f64 = 0.0;
        let mut max_gradient: f64 = 0.0;

        // Regional density map
        let block_size = self.config.block_size.max(1);
        let region_cols = width.div_ceil(block_size);
        let region_rows = height.div_ceil(block_size);
        let mut region_edge_counts = vec![0usize; region_cols * region_rows];
        let mut region_pixel_counts = vec![0usize; region_cols * region_rows];

        for py in 1..height - 1 {
            for px in 1..width - 1 {
                let (gx, gy) = Self::sobel_gradient(y_plane, width, px, py);
                let mag = (gx * gx + gy * gy).sqrt();

                gradient_sum += mag;
                if mag > max_gradient {
                    max_gradient = mag;
                }

                let is_edge = mag >= threshold;
                if is_edge {
                    edge_count += 1;
                }

                if self.config.compute_directional {
                    if gx.abs() >= threshold {
                        v_edge_count += 1; // Horizontal gradient => vertical edge
                    }
                    if gy.abs() >= threshold {
                        h_edge_count += 1; // Vertical gradient => horizontal edge
                    }
                }

                // Regional map
                let bx = px / block_size;
                let by = py / block_size;
                let idx = by * region_cols + bx;
                region_pixel_counts[idx] += 1;
                if is_edge {
                    region_edge_counts[idx] += 1;
                }
            }
        }

        let density = edge_count as f64 / interior_pixels as f64;
        let avg_gradient = gradient_sum / interior_pixels as f64;

        let horizontal_density = if self.config.compute_directional {
            Some(h_edge_count as f64 / interior_pixels as f64)
        } else {
            None
        };
        let vertical_density = if self.config.compute_directional {
            Some(v_edge_count as f64 / interior_pixels as f64)
        } else {
            None
        };

        let region_map: Vec<f64> = region_edge_counts
            .iter()
            .zip(region_pixel_counts.iter())
            .map(|(&edges, &pixels)| {
                if pixels > 0 {
                    edges as f64 / pixels as f64
                } else {
                    0.0
                }
            })
            .collect();

        // Update history
        self.update_history(density);
        self.frame_count += 1;

        EdgeDensityResult {
            frame_index,
            density,
            avg_gradient,
            max_gradient,
            horizontal_density,
            vertical_density,
            region_map,
            region_cols,
            region_rows,
        }
    }

    /// Produce an empty result for degenerate frames.
    fn empty_result(&mut self, frame_index: usize) -> EdgeDensityResult {
        self.update_history(0.0);
        self.frame_count += 1;
        EdgeDensityResult {
            frame_index,
            density: 0.0,
            avg_gradient: 0.0,
            max_gradient: 0.0,
            horizontal_density: if self.config.compute_directional {
                Some(0.0)
            } else {
                None
            },
            vertical_density: if self.config.compute_directional {
                Some(0.0)
            } else {
                None
            },
            region_map: Vec::new(),
            region_cols: 0,
            region_rows: 0,
        }
    }

    /// Update the sliding window history.
    fn update_history(&mut self, density: f64) {
        if self.density_history.len() >= self.config.window_size {
            if let Some(old) = self.density_history.pop_front() {
                self.running_sum -= old;
                self.running_sum_sq -= old * old;
            }
        }
        self.density_history.push_back(density);
        self.running_sum += density;
        self.running_sum_sq += density * density;
    }

    /// Get temporal statistics over the sliding window.
    #[allow(clippy::cast_precision_loss)]
    pub fn get_stats(&self) -> EdgeDensityStats {
        let n = self.density_history.len();
        if n == 0 {
            return EdgeDensityStats {
                mean_density: 0.0,
                std_density: 0.0,
                min_density: 0.0,
                max_density: 0.0,
                frame_count: self.frame_count,
            };
        }
        let mean = self.running_sum / n as f64;
        let variance = (self.running_sum_sq / n as f64 - mean * mean).max(0.0);
        let std_dev = variance.sqrt();

        let min = self
            .density_history
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let max = self
            .density_history
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        EdgeDensityStats {
            mean_density: mean,
            std_density: std_dev,
            min_density: min,
            max_density: max,
            frame_count: self.frame_count,
        }
    }

    /// Reset the analyzer state.
    pub fn reset(&mut self) {
        self.density_history.clear();
        self.running_sum = 0.0;
        self.running_sum_sq = 0.0;
        self.frame_count = 0;
    }
}

/// Classify a frame's spatial complexity based on its edge density.
#[allow(clippy::cast_precision_loss)]
pub fn classify_complexity(density: f64) -> &'static str {
    if density < 0.02 {
        "very_low"
    } else if density < 0.08 {
        "low"
    } else if density < 0.20 {
        "medium"
    } else if density < 0.40 {
        "high"
    } else {
        "very_high"
    }
}

/// Estimate encoding difficulty based on edge density and average gradient.
#[allow(clippy::cast_precision_loss)]
pub fn estimate_encoding_difficulty(density: f64, avg_gradient: f64) -> f64 {
    // Combined metric: density * 0.6 + normalized gradient * 0.4
    // Gradient is normalized assuming max ~360 (Sobel on 8-bit, sqrt(255^2 * 4 + 255^2 * 4))
    let norm_grad = (avg_gradient / 360.0).min(1.0);
    density * 0.6 + norm_grad * 0.4
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_flat_frame(width: usize, height: usize, value: u8) -> Vec<u8> {
        vec![value; width * height]
    }

    fn make_checkerboard(width: usize, height: usize, block: usize) -> Vec<u8> {
        let mut data = vec![0u8; width * height];
        for y in 0..height {
            for x in 0..width {
                let bx = x / block;
                let by = y / block;
                data[y * width + x] = if (bx + by) % 2 == 0 { 255 } else { 0 };
            }
        }
        data
    }

    #[test]
    fn test_flat_frame_zero_density() {
        let mut analyzer = EdgeDensityAnalyzer::new();
        let frame = make_flat_frame(64, 64, 128);
        let result = analyzer.process_frame(&frame, 64, 64, 0);
        assert!(
            result.density < 0.001,
            "flat frame should have near-zero density"
        );
        assert!(result.avg_gradient < 0.001);
    }

    #[test]
    fn test_checkerboard_high_density() {
        let mut analyzer = EdgeDensityAnalyzer::new();
        // Small checkerboard blocks => lots of edges
        let frame = make_checkerboard(64, 64, 2);
        let result = analyzer.process_frame(&frame, 64, 64, 0);
        assert!(
            result.density > 0.1,
            "checkerboard should have significant edge density: {}",
            result.density
        );
    }

    #[test]
    fn test_gradient_magnitude() {
        // Create a horizontal gradient: 0 to 255 across width
        let width = 64;
        let height = 64;
        let mut frame = vec![0u8; width * height];
        #[allow(clippy::cast_possible_truncation)]
        for y in 0..height {
            for x in 0..width {
                frame[y * width + x] = ((x * 255) / (width - 1)) as u8;
            }
        }
        let mut analyzer = EdgeDensityAnalyzer::new();
        let result = analyzer.process_frame(&frame, width, height, 0);
        assert!(
            result.avg_gradient > 0.0,
            "gradient frame should have non-zero avg gradient"
        );
        assert!(result.max_gradient > 0.0);
    }

    #[test]
    fn test_directional_edges() {
        let mut config = EdgeDensityConfig::default();
        config.compute_directional = true;
        config.gradient_threshold = 10;
        let mut analyzer = EdgeDensityAnalyzer::with_config(config);

        // Vertical stripes => vertical edges (detected by horizontal gradient)
        let width = 64;
        let height = 64;
        let mut frame = vec![0u8; width * height];
        for y in 0..height {
            for x in 0..width {
                frame[y * width + x] = if x % 4 < 2 { 200 } else { 20 };
            }
        }
        let result = analyzer.process_frame(&frame, width, height, 0);
        assert!(result.vertical_density.is_some());
        assert!(result.horizontal_density.is_some());
        // Vertical edges should be dominant for vertical stripes
        assert!(
            result
                .vertical_density
                .expect("expected vertical_density to be Some/Ok")
                > result
                    .horizontal_density
                    .expect("expected vertical_density to be Some/Ok"),
            "vertical stripes should produce more vertical edges"
        );
    }

    #[test]
    fn test_region_map_dimensions() {
        let config = EdgeDensityConfig {
            block_size: 16,
            ..Default::default()
        };
        let mut analyzer = EdgeDensityAnalyzer::with_config(config);
        let frame = make_flat_frame(64, 48, 128);
        let result = analyzer.process_frame(&frame, 64, 48, 0);
        assert_eq!(result.region_cols, 4); // 64 / 16
        assert_eq!(result.region_rows, 3); // 48 / 16
        assert_eq!(result.region_map.len(), 12);
    }

    #[test]
    fn test_sliding_window_stats() {
        let config = EdgeDensityConfig {
            window_size: 5,
            ..Default::default()
        };
        let mut analyzer = EdgeDensityAnalyzer::with_config(config);
        let flat = make_flat_frame(32, 32, 128);

        for i in 0..5 {
            analyzer.process_frame(&flat, 32, 32, i);
        }
        let stats = analyzer.get_stats();
        assert_eq!(stats.frame_count, 5);
        assert!(stats.mean_density < 0.001);
        assert!(stats.std_density < 0.001);
    }

    #[test]
    fn test_window_eviction() {
        let config = EdgeDensityConfig {
            window_size: 3,
            ..Default::default()
        };
        let mut analyzer = EdgeDensityAnalyzer::with_config(config);
        let flat = make_flat_frame(32, 32, 128);
        let checker = make_checkerboard(32, 32, 2);

        // Push 3 flat, then 3 checkerboard
        for i in 0..3 {
            analyzer.process_frame(&flat, 32, 32, i);
        }
        for i in 3..6 {
            analyzer.process_frame(&checker, 32, 32, i);
        }
        let stats = analyzer.get_stats();
        assert_eq!(stats.frame_count, 6);
        // Window should only have checkerboard results, so density > 0
        assert!(stats.mean_density > 0.0);
    }

    #[test]
    fn test_classify_complexity_levels() {
        assert_eq!(classify_complexity(0.0), "very_low");
        assert_eq!(classify_complexity(0.01), "very_low");
        assert_eq!(classify_complexity(0.05), "low");
        assert_eq!(classify_complexity(0.15), "medium");
        assert_eq!(classify_complexity(0.30), "high");
        assert_eq!(classify_complexity(0.50), "very_high");
    }

    #[test]
    fn test_estimate_encoding_difficulty() {
        let easy = estimate_encoding_difficulty(0.0, 0.0);
        let hard = estimate_encoding_difficulty(0.5, 180.0);
        assert!(easy < hard);
        assert!(easy >= 0.0);
        assert!(hard <= 1.0);
    }

    #[test]
    fn test_reset_analyzer() {
        let mut analyzer = EdgeDensityAnalyzer::new();
        let frame = make_flat_frame(32, 32, 128);
        analyzer.process_frame(&frame, 32, 32, 0);
        assert_eq!(analyzer.frame_count, 1);
        analyzer.reset();
        assert_eq!(analyzer.frame_count, 0);
        let stats = analyzer.get_stats();
        assert_eq!(stats.frame_count, 0);
    }

    #[test]
    fn test_empty_stats() {
        let analyzer = EdgeDensityAnalyzer::new();
        let stats = analyzer.get_stats();
        assert_eq!(stats.frame_count, 0);
        assert_eq!(stats.mean_density, 0.0);
    }

    #[test]
    fn test_small_frame_edge_case() {
        let mut analyzer = EdgeDensityAnalyzer::new();
        // 2x2 frame has no interior pixels for Sobel
        let frame = vec![128u8; 4];
        let result = analyzer.process_frame(&frame, 2, 2, 0);
        assert_eq!(result.density, 0.0);
    }
}
