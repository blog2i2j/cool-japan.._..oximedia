#![allow(dead_code)]
//! Motion compensation for temporal video alignment.
//!
//! This module provides frame-level motion compensation used to correct for camera movement
//! and subject motion when aligning video streams temporally.
//!
//! # Features
//!
//! - **Block-based motion estimation** using full search and diamond search
//! - **Motion vector field** representation and interpolation
//! - **Frame warping** to compensate detected motion
//! - **Motion statistics** for alignment quality assessment

use crate::{AlignError, AlignResult, Point2D};

/// A motion vector representing displacement of a block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionVector {
    /// Horizontal displacement in pixels.
    pub dx: f64,
    /// Vertical displacement in pixels.
    pub dy: f64,
    /// Match quality (lower SAD means better match).
    pub cost: f64,
}

impl MotionVector {
    /// Create a new motion vector.
    #[must_use]
    pub fn new(dx: f64, dy: f64, cost: f64) -> Self {
        Self { dx, dy, cost }
    }

    /// Create a zero motion vector.
    #[must_use]
    pub fn zero() -> Self {
        Self {
            dx: 0.0,
            dy: 0.0,
            cost: 0.0,
        }
    }

    /// Compute the magnitude of this motion vector.
    #[must_use]
    pub fn magnitude(&self) -> f64 {
        (self.dx * self.dx + self.dy * self.dy).sqrt()
    }

    /// Compute the direction angle in radians.
    #[must_use]
    pub fn direction(&self) -> f64 {
        self.dy.atan2(self.dx)
    }

    /// Add two motion vectors.
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        Self {
            dx: self.dx + other.dx,
            dy: self.dy + other.dy,
            cost: (self.cost + other.cost) / 2.0,
        }
    }

    /// Scale this motion vector by a factor.
    #[must_use]
    pub fn scale(&self, factor: f64) -> Self {
        Self {
            dx: self.dx * factor,
            dy: self.dy * factor,
            cost: self.cost,
        }
    }
}

/// Search strategy for block matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchStrategy {
    /// Full (exhaustive) search within the search range.
    FullSearch,
    /// Diamond search pattern for faster estimation.
    DiamondSearch,
    /// Three-step search for moderate speed/quality.
    ThreeStepSearch,
    /// Hexagonal search pattern.
    HexagonalSearch,
}

/// Configuration for motion estimation.
#[derive(Debug, Clone)]
pub struct MotionEstimationConfig {
    /// Block size in pixels (width and height).
    pub block_size: u32,
    /// Search range in pixels.
    pub search_range: u32,
    /// Search strategy to use.
    pub search_strategy: SearchStrategy,
    /// Enable sub-pixel refinement.
    pub sub_pixel: bool,
    /// Frame width in pixels.
    pub frame_width: u32,
    /// Frame height in pixels.
    pub frame_height: u32,
}

impl Default for MotionEstimationConfig {
    fn default() -> Self {
        Self {
            block_size: 16,
            search_range: 32,
            search_strategy: SearchStrategy::DiamondSearch,
            sub_pixel: true,
            frame_width: 1920,
            frame_height: 1080,
        }
    }
}

/// A field of motion vectors covering an entire frame.
#[derive(Debug, Clone)]
pub struct MotionField {
    /// Motion vectors in row-major order.
    pub vectors: Vec<MotionVector>,
    /// Number of blocks horizontally.
    pub cols: u32,
    /// Number of blocks vertically.
    pub rows: u32,
    /// Block size used for estimation.
    pub block_size: u32,
}

impl MotionField {
    /// Create a new motion field with zero vectors.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn new(frame_width: u32, frame_height: u32, block_size: u32) -> Self {
        let cols = frame_width.div_ceil(block_size);
        let rows = frame_height.div_ceil(block_size);
        let count = (cols * rows) as usize;
        Self {
            vectors: vec![MotionVector::zero(); count],
            cols,
            rows,
            block_size,
        }
    }

    /// Get the motion vector at block position (bx, by).
    #[must_use]
    pub fn get(&self, bx: u32, by: u32) -> Option<&MotionVector> {
        if bx < self.cols && by < self.rows {
            Some(&self.vectors[(by * self.cols + bx) as usize])
        } else {
            None
        }
    }

    /// Set the motion vector at block position (bx, by).
    pub fn set(&mut self, bx: u32, by: u32, mv: MotionVector) {
        if bx < self.cols && by < self.rows {
            self.vectors[(by * self.cols + bx) as usize] = mv;
        }
    }

    /// Interpolate a motion vector at a continuous pixel position.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn interpolate(&self, x: f64, y: f64) -> MotionVector {
        let bs = f64::from(self.block_size);
        let bx = (x / bs).floor();
        let by = (y / bs).floor();

        let bxi = bx as u32;
        let byi = by as u32;

        // Bilinear interpolation
        let fx = x / bs - bx;
        let fy = y / bs - by;

        let get_mv = |cx: u32, cy: u32| -> MotionVector {
            self.get(
                cx.min(self.cols.saturating_sub(1)),
                cy.min(self.rows.saturating_sub(1)),
            )
            .copied()
            .unwrap_or_else(MotionVector::zero)
        };

        let tl = get_mv(bxi, byi);
        let tr = get_mv(bxi + 1, byi);
        let bl = get_mv(bxi, byi + 1);
        let br = get_mv(bxi + 1, byi + 1);

        let dx = tl.dx * (1.0 - fx) * (1.0 - fy)
            + tr.dx * fx * (1.0 - fy)
            + bl.dx * (1.0 - fx) * fy
            + br.dx * fx * fy;

        let dy = tl.dy * (1.0 - fx) * (1.0 - fy)
            + tr.dy * fx * (1.0 - fy)
            + bl.dy * (1.0 - fx) * fy
            + br.dy * fx * fy;

        let cost = tl.cost * (1.0 - fx) * (1.0 - fy)
            + tr.cost * fx * (1.0 - fy)
            + bl.cost * (1.0 - fx) * fy
            + br.cost * fx * fy;

        MotionVector::new(dx, dy, cost)
    }

    /// Compute global motion from the motion field (median of all vectors).
    #[must_use]
    pub fn global_motion(&self) -> MotionVector {
        if self.vectors.is_empty() {
            return MotionVector::zero();
        }

        let mut dxs: Vec<f64> = self.vectors.iter().map(|v| v.dx).collect();
        let mut dys: Vec<f64> = self.vectors.iter().map(|v| v.dy).collect();

        dxs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        dys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mid = dxs.len() / 2;
        MotionVector::new(dxs[mid], dys[mid], 0.0)
    }

    /// Compute the average magnitude of all motion vectors.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn average_magnitude(&self) -> f64 {
        if self.vectors.is_empty() {
            return 0.0;
        }
        let total: f64 = self.vectors.iter().map(MotionVector::magnitude).sum();
        total / self.vectors.len() as f64
    }

    /// Count the number of vectors exceeding a magnitude threshold.
    #[must_use]
    pub fn count_above_threshold(&self, threshold: f64) -> usize {
        self.vectors
            .iter()
            .filter(|v| v.magnitude() > threshold)
            .count()
    }
}

/// Motion statistics for a pair of frames.
#[derive(Debug, Clone)]
pub struct MotionStats {
    /// Average motion magnitude in pixels.
    pub avg_magnitude: f64,
    /// Maximum motion magnitude in pixels.
    pub max_magnitude: f64,
    /// Standard deviation of motion magnitude.
    pub std_magnitude: f64,
    /// Global horizontal motion (median dx).
    pub global_dx: f64,
    /// Global vertical motion (median dy).
    pub global_dy: f64,
    /// Fraction of blocks with significant motion (above 1 pixel).
    pub motion_fraction: f64,
}

/// Motion compensator that estimates and applies motion compensation.
#[derive(Debug, Clone)]
pub struct MotionCompensator {
    /// Configuration for motion estimation.
    config: MotionEstimationConfig,
}

impl MotionCompensator {
    /// Create a new motion compensator with the given configuration.
    #[must_use]
    pub fn new(config: MotionEstimationConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self {
            config: MotionEstimationConfig::default(),
        }
    }

    /// Estimate motion field between a reference frame and a target frame.
    ///
    /// Both frames are provided as grayscale pixel data (one byte per pixel, row-major).
    #[allow(clippy::cast_precision_loss)]
    pub fn estimate(&self, reference: &[u8], target: &[u8]) -> AlignResult<MotionField> {
        let expected_size = (self.config.frame_width * self.config.frame_height) as usize;
        if reference.len() != expected_size || target.len() != expected_size {
            return Err(AlignError::InsufficientData(format!(
                "Expected frame size {}, got ref={} target={}",
                expected_size,
                reference.len(),
                target.len()
            )));
        }

        let mut field = MotionField::new(
            self.config.frame_width,
            self.config.frame_height,
            self.config.block_size,
        );

        let bs = self.config.block_size;
        let sr = self.config.search_range as i32;
        let w = self.config.frame_width;
        let h = self.config.frame_height;

        for by in 0..field.rows {
            for bx in 0..field.cols {
                let orig_x = (bx * bs) as i32;
                let orig_y = (by * bs) as i32;

                let mv = match self.config.search_strategy {
                    SearchStrategy::FullSearch => {
                        self.full_search(reference, target, orig_x, orig_y, bs, sr, w, h)
                    }
                    _ => {
                        // Use diamond search as default fast path
                        self.diamond_search(reference, target, orig_x, orig_y, bs, sr, w, h)
                    }
                };

                field.set(bx, by, mv);
            }
        }

        Ok(field)
    }

    /// Compute motion statistics from a motion field.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn compute_stats(field: &MotionField) -> MotionStats {
        if field.vectors.is_empty() {
            return MotionStats {
                avg_magnitude: 0.0,
                max_magnitude: 0.0,
                std_magnitude: 0.0,
                global_dx: 0.0,
                global_dy: 0.0,
                motion_fraction: 0.0,
            };
        }

        let magnitudes: Vec<f64> = field.vectors.iter().map(MotionVector::magnitude).collect();
        let n = magnitudes.len() as f64;
        let avg = magnitudes.iter().sum::<f64>() / n;
        let max = magnitudes.iter().copied().fold(0.0_f64, f64::max);
        let variance = magnitudes.iter().map(|m| (m - avg).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        let global = field.global_motion();
        let motion_count = field.count_above_threshold(1.0);

        MotionStats {
            avg_magnitude: avg,
            max_magnitude: max,
            std_magnitude: std_dev,
            global_dx: global.dx,
            global_dy: global.dy,
            motion_fraction: motion_count as f64 / n,
        }
    }

    /// Apply motion compensation to warp a set of points.
    #[must_use]
    pub fn compensate_points(field: &MotionField, points: &[Point2D]) -> Vec<Point2D> {
        points
            .iter()
            .map(|p| {
                let mv = field.interpolate(p.x, p.y);
                Point2D::new(p.x + mv.dx, p.y + mv.dy)
            })
            .collect()
    }

    /// Full search block matching (exhaustive).
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::cast_precision_loss)]
    fn full_search(
        &self,
        reference: &[u8],
        target: &[u8],
        bx: i32,
        by: i32,
        bs: u32,
        sr: i32,
        w: u32,
        h: u32,
    ) -> MotionVector {
        let mut best_dx = 0i32;
        let mut best_dy = 0i32;
        let mut best_cost = f64::MAX;

        for dy in -sr..=sr {
            for dx in -sr..=sr {
                let cost = self.compute_sad(reference, target, bx, by, bx + dx, by + dy, bs, w, h);
                if cost < best_cost
                    || (cost == best_cost
                        && (dx.unsigned_abs() + dy.unsigned_abs())
                            < (best_dx.unsigned_abs() + best_dy.unsigned_abs()))
                {
                    best_cost = cost;
                    best_dx = dx;
                    best_dy = dy;
                }
            }
        }

        MotionVector::new(f64::from(best_dx), f64::from(best_dy), best_cost)
    }

    /// Diamond search pattern block matching.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::cast_precision_loss)]
    fn diamond_search(
        &self,
        reference: &[u8],
        target: &[u8],
        bx: i32,
        by: i32,
        bs: u32,
        sr: i32,
        w: u32,
        h: u32,
    ) -> MotionVector {
        let large_diamond: [(i32, i32); 9] = [
            (0, 0),
            (0, -2),
            (1, -1),
            (2, 0),
            (1, 1),
            (0, 2),
            (-1, 1),
            (-2, 0),
            (-1, -1),
        ];

        let mut cx = 0i32;
        let mut cy = 0i32;
        let mut best_cost = f64::MAX;

        for _ in 0..sr {
            let mut found_better = false;
            let mut new_cx = cx;
            let mut new_cy = cy;

            for &(ddx, ddy) in &large_diamond {
                let tx = cx + ddx;
                let ty = cy + ddy;
                if tx.abs() > sr || ty.abs() > sr {
                    continue;
                }
                let cost = self.compute_sad(reference, target, bx, by, bx + tx, by + ty, bs, w, h);
                if cost < best_cost {
                    best_cost = cost;
                    new_cx = tx;
                    new_cy = ty;
                    found_better = true;
                }
            }

            if !found_better || (new_cx == cx && new_cy == cy) {
                break;
            }
            cx = new_cx;
            cy = new_cy;
        }

        MotionVector::new(f64::from(cx), f64::from(cy), best_cost)
    }

    /// Compute Sum of Absolute Differences (SAD) for a block.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::cast_precision_loss)]
    fn compute_sad(
        &self,
        reference: &[u8],
        target: &[u8],
        rx: i32,
        ry: i32,
        tx: i32,
        ty: i32,
        bs: u32,
        w: u32,
        h: u32,
    ) -> f64 {
        let mut sad = 0u64;
        let bs_i = bs as i32;
        let w_i = w as i32;
        let h_i = h as i32;

        for row in 0..bs_i {
            for col in 0..bs_i {
                let ref_x = rx + col;
                let ref_y = ry + row;
                let tgt_x = tx + col;
                let tgt_y = ty + row;

                if ref_x < 0 || ref_x >= w_i || ref_y < 0 || ref_y >= h_i {
                    sad += 128;
                    continue;
                }
                if tgt_x < 0 || tgt_x >= w_i || tgt_y < 0 || tgt_y >= h_i {
                    sad += 128;
                    continue;
                }

                let ref_idx = (ref_y as u32 * w + ref_x as u32) as usize;
                let tgt_idx = (tgt_y as u32 * w + tgt_x as u32) as usize;

                let diff = i32::from(reference[ref_idx]) - i32::from(target[tgt_idx]);
                sad += u64::from(diff.unsigned_abs());
            }
        }

        sad as f64
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &MotionEstimationConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motion_vector_creation() {
        let mv = MotionVector::new(3.0, 4.0, 100.0);
        assert!((mv.dx - 3.0).abs() < f64::EPSILON);
        assert!((mv.dy - 4.0).abs() < f64::EPSILON);
        assert!((mv.cost - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_motion_vector_magnitude() {
        let mv = MotionVector::new(3.0, 4.0, 0.0);
        assert!((mv.magnitude() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_motion_vector_direction() {
        let mv = MotionVector::new(1.0, 0.0, 0.0);
        assert!((mv.direction()).abs() < 1e-10);

        let mv_up = MotionVector::new(0.0, 1.0, 0.0);
        assert!((mv_up.direction() - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
    }

    #[test]
    fn test_motion_vector_zero() {
        let mv = MotionVector::zero();
        assert!((mv.magnitude()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_motion_vector_add() {
        let a = MotionVector::new(1.0, 2.0, 10.0);
        let b = MotionVector::new(3.0, 4.0, 20.0);
        let c = a.add(&b);
        assert!((c.dx - 4.0).abs() < f64::EPSILON);
        assert!((c.dy - 6.0).abs() < f64::EPSILON);
        assert!((c.cost - 15.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_motion_vector_scale() {
        let mv = MotionVector::new(2.0, 3.0, 10.0);
        let scaled = mv.scale(0.5);
        assert!((scaled.dx - 1.0).abs() < f64::EPSILON);
        assert!((scaled.dy - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_motion_field_creation() {
        let field = MotionField::new(320, 240, 16);
        assert_eq!(field.cols, 20);
        assert_eq!(field.rows, 15);
        assert_eq!(field.vectors.len(), 300);
    }

    #[test]
    fn test_motion_field_get_set() {
        let mut field = MotionField::new(64, 64, 16);
        let mv = MotionVector::new(5.0, -3.0, 50.0);
        field.set(1, 2, mv);
        let retrieved = field.get(1, 2).expect("retrieved should be valid");
        assert!((retrieved.dx - 5.0).abs() < f64::EPSILON);
        assert!((retrieved.dy - (-3.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_motion_field_global_motion() {
        let mut field = MotionField::new(64, 64, 16);
        // Set all vectors to roughly (2, 1) with some noise
        for by in 0..field.rows {
            for bx in 0..field.cols {
                field.set(bx, by, MotionVector::new(2.0, 1.0, 0.0));
            }
        }
        let global = field.global_motion();
        assert!((global.dx - 2.0).abs() < f64::EPSILON);
        assert!((global.dy - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_motion_field_average_magnitude() {
        let mut field = MotionField::new(32, 32, 16);
        field.set(0, 0, MotionVector::new(3.0, 4.0, 0.0));
        field.set(1, 0, MotionVector::new(0.0, 0.0, 0.0));
        field.set(0, 1, MotionVector::new(0.0, 0.0, 0.0));
        field.set(1, 1, MotionVector::new(0.0, 0.0, 0.0));
        let avg = field.average_magnitude();
        // One vector has mag 5.0, three have 0.0 => avg = 5.0/4 = 1.25
        assert!((avg - 1.25).abs() < 1e-10);
    }

    #[test]
    fn test_estimate_static_frames() {
        let config = MotionEstimationConfig {
            block_size: 8,
            search_range: 4,
            search_strategy: SearchStrategy::FullSearch,
            sub_pixel: false,
            frame_width: 32,
            frame_height: 32,
        };
        let comp = MotionCompensator::new(config);

        // Both frames identical => all zero motion
        let frame = vec![128u8; 32 * 32];
        let field = comp
            .estimate(&frame, &frame)
            .expect("field should be valid");

        for mv in &field.vectors {
            assert!((mv.dx).abs() < f64::EPSILON);
            assert!((mv.dy).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_estimate_wrong_size() {
        let comp = MotionCompensator::new(MotionEstimationConfig {
            frame_width: 64,
            frame_height: 64,
            ..MotionEstimationConfig::default()
        });
        let small_frame = vec![0u8; 10];
        let result = comp.estimate(&small_frame, &small_frame);
        assert!(result.is_err());
    }

    #[test]
    fn test_compensate_points() {
        let mut field = MotionField::new(64, 64, 64);
        field.set(0, 0, MotionVector::new(10.0, -5.0, 0.0));

        let points = vec![Point2D::new(10.0, 20.0)];
        let compensated = MotionCompensator::compensate_points(&field, &points);
        assert!((compensated[0].x - 20.0).abs() < 1e-10);
        assert!((compensated[0].y - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_motion_stats_static() {
        let field = MotionField::new(64, 64, 16);
        let stats = MotionCompensator::compute_stats(&field);
        assert!((stats.avg_magnitude).abs() < f64::EPSILON);
        assert!((stats.max_magnitude).abs() < f64::EPSILON);
        assert!((stats.motion_fraction).abs() < f64::EPSILON);
    }
}
