//! Dense optical flow field representation and analysis.

#![allow(dead_code)]

use std::f32::consts::PI;

/// A 2-D motion vector at a single pixel location.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlowVector {
    /// Horizontal displacement in pixels.
    pub dx: f32,
    /// Vertical displacement in pixels.
    pub dy: f32,
}

impl FlowVector {
    /// Create a new [`FlowVector`].
    #[must_use]
    pub fn new(dx: f32, dy: f32) -> Self {
        Self { dx, dy }
    }

    /// Return the magnitude (Euclidean length) of the vector.
    #[must_use]
    pub fn magnitude(&self) -> f32 {
        (self.dx * self.dx + self.dy * self.dy).sqrt()
    }

    /// Return the angle of the vector in radians, in [−π, π].
    #[must_use]
    pub fn angle_rad(&self) -> f32 {
        self.dy.atan2(self.dx)
    }

    /// Return the angle of the vector in degrees, in [−180, 180].
    #[must_use]
    pub fn angle_deg(&self) -> f32 {
        self.angle_rad().to_degrees()
    }

    /// Return `true` when the magnitude exceeds `min_magnitude`.
    #[must_use]
    pub fn is_moving(&self, min_magnitude: f32) -> bool {
        self.magnitude() > min_magnitude
    }
}

/// A dense optical flow field for a single frame pair.
#[derive(Debug, Clone)]
pub struct FlowField {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Per-pixel flow vectors in row-major order.
    pub vectors: Vec<FlowVector>,
}

impl FlowField {
    /// Create a zero flow field.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        let n = (width as usize) * (height as usize);
        Self {
            width,
            height,
            vectors: vec![FlowVector::new(0.0, 0.0); n],
        }
    }

    /// Return the average magnitude over all vectors.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_magnitude(&self) -> f32 {
        if self.vectors.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.vectors.iter().map(FlowVector::magnitude).sum();
        sum / self.vectors.len() as f32
    }

    /// Return the dominant direction (angle in degrees) by circular mean of all vectors.
    ///
    /// Returns `0.0` when there are no vectors.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn dominant_direction(&self) -> f32 {
        if self.vectors.is_empty() {
            return 0.0;
        }
        let n = self.vectors.len() as f32;
        let sin_sum: f32 = self
            .vectors
            .iter()
            .map(|v| v.angle_rad().sin())
            .sum::<f32>()
            / n;
        let cos_sum: f32 = self
            .vectors
            .iter()
            .map(|v| v.angle_rad().cos())
            .sum::<f32>()
            / n;
        sin_sum.atan2(cos_sum).to_degrees()
    }

    /// Return the fraction of pixels whose magnitude exceeds `threshold`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn motion_coverage(&self, threshold: f32) -> f32 {
        if self.vectors.is_empty() {
            return 0.0;
        }
        let moving = self
            .vectors
            .iter()
            .filter(|v| v.magnitude() > threshold)
            .count();
        moving as f32 / self.vectors.len() as f32
    }

    /// Return the maximum magnitude among all vectors.
    #[must_use]
    pub fn max_magnitude(&self) -> f32 {
        self.vectors
            .iter()
            .map(FlowVector::magnitude)
            .fold(0.0_f32, f32::max)
    }
}

/// Configuration for the dense flow field analyser.
#[derive(Debug, Clone)]
pub struct FlowFieldAnalyzerConfig {
    /// Minimum motion magnitude to be considered non-trivial.
    pub min_magnitude: f32,
    /// Number of pyramid levels for multi-scale estimation (>= 1).
    pub pyramid_levels: u32,
    /// Block size for block-matching (pixels).
    pub block_size: u32,
    /// Search radius for block-matching (pixels).
    pub search_radius: u32,
}

impl Default for FlowFieldAnalyzerConfig {
    fn default() -> Self {
        Self {
            min_magnitude: 0.5,
            pyramid_levels: 3,
            block_size: 8,
            search_radius: 16,
        }
    }
}

/// Computes dense optical flow between pairs of luma frames.
pub struct FlowFieldAnalyzer {
    config: FlowFieldAnalyzerConfig,
    prev_luma: Option<Vec<u8>>,
    prev_width: u32,
    prev_height: u32,
    processed: usize,
}

impl FlowFieldAnalyzer {
    /// Create a new [`FlowFieldAnalyzer`].
    #[must_use]
    pub fn new(config: FlowFieldAnalyzerConfig) -> Self {
        Self {
            config,
            prev_luma: None,
            prev_width: 0,
            prev_height: 0,
            processed: 0,
        }
    }

    /// Compute a dense flow field between the previous and current luma frames.
    ///
    /// Returns `None` for the first frame (no previous frame available) or if the
    /// frame dimensions do not match the previous frame.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn compute_dense(&mut self, luma: &[u8], width: u32, height: u32) -> Option<FlowField> {
        let n = (width as usize) * (height as usize);

        let result = if let Some(ref prev) = self.prev_luma {
            if self.prev_width != width
                || self.prev_height != height
                || luma.len() < n
                || prev.len() < n
            {
                None
            } else {
                // Block-matching optical flow.
                let bs = self.config.block_size as usize;
                let sr = self.config.search_radius as usize;
                let w = width as usize;
                let h = height as usize;
                let mut field = FlowField::new(width, height);

                let block_cols = w.div_ceil(bs);
                let block_rows = h.div_ceil(bs);

                for brow in 0..block_rows {
                    for bcol in 0..block_cols {
                        let bx = bcol * bs;
                        let by = brow * bs;
                        let bw = bs.min(w - bx);
                        let bh_actual = bs.min(h - by);

                        // Compute block SAD against candidates in search window.
                        let mut best_sad = u32::MAX;
                        let mut best_dx = 0_i32;
                        let mut best_dy = 0_i32;

                        let sx_min = (bx as i32 - sr as i32).max(0) as usize;
                        let sy_min = (by as i32 - sr as i32).max(0) as usize;
                        let sx_max = ((bx + sr) + bw).min(w);
                        let sy_max = ((by + sr) + bh_actual).min(h);

                        // Stride through candidate positions.
                        let step = (bs / 2).max(1);
                        let mut sy = sy_min;
                        while sy + bh_actual <= sy_max {
                            let mut sx = sx_min;
                            while sx + bw <= sx_max {
                                let mut sad = 0u32;
                                'outer: for dy in 0..bh_actual {
                                    for dx in 0..bw {
                                        let cur_idx = (by + dy) * w + (bx + dx);
                                        let ref_idx = (sy + dy) * w + (sx + dx);
                                        let diff = (luma[cur_idx] as i32 - prev[ref_idx] as i32)
                                            .unsigned_abs();
                                        sad += diff;
                                        if sad >= best_sad {
                                            break 'outer;
                                        }
                                    }
                                }
                                let cdx = sx as i32 - bx as i32;
                                let cdy = sy as i32 - by as i32;
                                if sad < best_sad
                                    || (sad == best_sad
                                        && (cdx.unsigned_abs() + cdy.unsigned_abs())
                                            < (best_dx.unsigned_abs() + best_dy.unsigned_abs()))
                                {
                                    best_sad = sad;
                                    best_dx = cdx;
                                    best_dy = cdy;
                                }
                                sx += step;
                            }
                            sy += step;
                        }

                        // Fill block pixels with the found motion vector.
                        for dy in 0..bh_actual {
                            for dx in 0..bw {
                                let idx = (by + dy) * w + (bx + dx);
                                field.vectors[idx] =
                                    FlowVector::new(best_dx as f32, best_dy as f32);
                            }
                        }
                    }
                }

                Some(field)
            }
        } else {
            None
        };

        // Update state for next call.
        let mut new_prev = vec![0u8; n];
        if luma.len() >= n {
            new_prev.copy_from_slice(&luma[..n]);
        }
        self.prev_luma = Some(new_prev);
        self.prev_width = width;
        self.prev_height = height;
        self.processed += 1;

        result
    }

    /// Return the number of frames processed so far.
    #[must_use]
    pub fn processed(&self) -> usize {
        self.processed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flow_vector_magnitude_zero() {
        let v = FlowVector::new(0.0, 0.0);
        assert!((v.magnitude() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_flow_vector_magnitude_3_4_5() {
        let v = FlowVector::new(3.0, 4.0);
        assert!((v.magnitude() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_flow_vector_angle_right() {
        let v = FlowVector::new(1.0, 0.0);
        assert!((v.angle_deg() - 0.0).abs() < 1e-4);
    }

    #[test]
    fn test_flow_vector_angle_up() {
        let v = FlowVector::new(0.0, 1.0);
        assert!((v.angle_deg() - 90.0).abs() < 1e-4);
    }

    #[test]
    fn test_flow_vector_is_moving() {
        let v = FlowVector::new(2.0, 0.0);
        assert!(v.is_moving(1.0));
        assert!(!v.is_moving(3.0));
    }

    #[test]
    fn test_flow_field_avg_magnitude_zero_field() {
        let f = FlowField::new(4, 4);
        assert!((f.avg_magnitude() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_flow_field_avg_magnitude_uniform() {
        let mut f = FlowField::new(2, 2);
        for v in &mut f.vectors {
            *v = FlowVector::new(3.0, 4.0); // magnitude 5
        }
        assert!((f.avg_magnitude() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_flow_field_max_magnitude() {
        let mut f = FlowField::new(2, 1);
        f.vectors[0] = FlowVector::new(0.0, 3.0); // magnitude 3
        f.vectors[1] = FlowVector::new(4.0, 0.0); // magnitude 4
        assert!((f.max_magnitude() - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_flow_field_motion_coverage_none() {
        let f = FlowField::new(4, 4);
        assert!((f.motion_coverage(0.5) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_flow_field_motion_coverage_all() {
        let mut f = FlowField::new(2, 2);
        for v in &mut f.vectors {
            *v = FlowVector::new(5.0, 0.0);
        }
        assert!((f.motion_coverage(1.0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_flow_field_dominant_direction_right() {
        let mut f = FlowField::new(4, 1);
        for v in &mut f.vectors {
            *v = FlowVector::new(1.0, 0.0);
        }
        let dir = f.dominant_direction();
        assert!(dir.abs() < 1.0); // should be near 0°
    }

    #[test]
    fn test_flow_field_analyzer_first_frame_returns_none() {
        let mut analyzer = FlowFieldAnalyzer::new(FlowFieldAnalyzerConfig::default());
        let luma = vec![128u8; 16 * 16];
        assert!(analyzer.compute_dense(&luma, 16, 16).is_none());
        assert_eq!(analyzer.processed(), 1);
    }

    #[test]
    fn test_flow_field_analyzer_second_frame_returns_some() {
        let mut analyzer = FlowFieldAnalyzer::new(FlowFieldAnalyzerConfig::default());
        let luma = vec![128u8; 16 * 16];
        let _ = analyzer.compute_dense(&luma, 16, 16);
        let field = analyzer.compute_dense(&luma, 16, 16);
        assert!(field.is_some());
        let f = field.expect("f should be valid");
        assert_eq!(f.width, 16);
        assert_eq!(f.height, 16);
    }

    #[test]
    fn test_flow_field_analyzer_dimension_mismatch_returns_none() {
        let mut analyzer = FlowFieldAnalyzer::new(FlowFieldAnalyzerConfig::default());
        let luma_a = vec![0u8; 8 * 8];
        let luma_b = vec![0u8; 16 * 16];
        let _ = analyzer.compute_dense(&luma_a, 8, 8);
        let result = analyzer.compute_dense(&luma_b, 16, 16);
        assert!(result.is_none());
    }

    #[test]
    fn test_flow_field_analyzer_identical_frames_zero_flow() {
        let mut cfg = FlowFieldAnalyzerConfig::default();
        cfg.block_size = 4;
        cfg.search_radius = 4;
        let mut analyzer = FlowFieldAnalyzer::new(cfg);
        let luma = vec![100u8; 8 * 8];
        let _ = analyzer.compute_dense(&luma, 8, 8);
        let field = analyzer
            .compute_dense(&luma, 8, 8)
            .expect("compute_dense should succeed");
        // Identical frames → best match at (0, 0) displacement.
        assert!((field.avg_magnitude() - 0.0).abs() < 1e-5);
    }
}
