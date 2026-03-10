#![allow(dead_code)]
//! Motion vector estimation between video frames.
//!
//! This module provides block-matching motion estimation using the Sum of
//! Absolute Differences (SAD) metric. It computes per-block motion vectors
//! between a reference frame and a target frame.

/// A single motion vector for a block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionVector {
    /// Horizontal displacement in pixels.
    pub dx: f64,
    /// Vertical displacement in pixels.
    pub dy: f64,
    /// SAD cost for this match (lower is better).
    pub cost: f64,
}

impl MotionVector {
    /// Create a new motion vector.
    #[must_use]
    pub fn new(dx: f64, dy: f64, cost: f64) -> Self {
        Self { dx, dy, cost }
    }

    /// Zero motion vector.
    #[must_use]
    pub fn zero() -> Self {
        Self {
            dx: 0.0,
            dy: 0.0,
            cost: 0.0,
        }
    }

    /// Magnitude of the vector.
    #[must_use]
    pub fn magnitude(&self) -> f64 {
        (self.dx * self.dx + self.dy * self.dy).sqrt()
    }

    /// Direction in radians (atan2).
    #[must_use]
    pub fn direction(&self) -> f64 {
        self.dy.atan2(self.dx)
    }
}

/// A 2-D grid of motion vectors covering an entire frame.
#[derive(Debug, Clone)]
pub struct MotionVectorField {
    /// Motion vectors in row-major order.
    pub vectors: Vec<MotionVector>,
    /// Number of block columns.
    pub cols: usize,
    /// Number of block rows.
    pub rows: usize,
    /// Block size in pixels.
    pub block_size: usize,
}

impl MotionVectorField {
    /// Create a field filled with zero vectors.
    #[must_use]
    pub fn zeros(cols: usize, rows: usize, block_size: usize) -> Self {
        Self {
            vectors: vec![MotionVector::zero(); cols * rows],
            cols,
            rows,
            block_size,
        }
    }

    /// Get the vector at grid position `(col, row)`.
    #[must_use]
    pub fn get(&self, col: usize, row: usize) -> Option<&MotionVector> {
        if col < self.cols && row < self.rows {
            Some(&self.vectors[row * self.cols + col])
        } else {
            None
        }
    }

    /// Set the vector at grid position `(col, row)`.
    pub fn set(&mut self, col: usize, row: usize, mv: MotionVector) {
        if col < self.cols && row < self.rows {
            self.vectors[row * self.cols + col] = mv;
        }
    }

    /// Average motion magnitude across the entire field.
    #[must_use]
    pub fn avg_magnitude(&self) -> f64 {
        if self.vectors.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.vectors.iter().map(MotionVector::magnitude).sum();
        sum / self.vectors.len() as f64
    }

    /// Maximum motion magnitude in the field.
    #[must_use]
    pub fn max_magnitude(&self) -> f64 {
        self.vectors
            .iter()
            .map(MotionVector::magnitude)
            .fold(0.0f64, f64::max)
    }

    /// Average SAD cost across the field.
    #[must_use]
    pub fn avg_cost(&self) -> f64 {
        if self.vectors.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.vectors.iter().map(|v| v.cost).sum();
        sum / self.vectors.len() as f64
    }
}

/// Block-matching motion analyzer.
#[derive(Debug)]
pub struct MotionAnalyzer {
    /// Block size in pixels (NxN).
    pub block_size: usize,
    /// Search range in pixels (checked in each direction).
    pub search_range: i32,
}

impl MotionAnalyzer {
    /// Create a new analyzer.
    #[must_use]
    pub fn new(block_size: usize, search_range: i32) -> Self {
        Self {
            block_size: block_size.max(4),
            search_range: search_range.max(1),
        }
    }

    /// Compute the Sum of Absolute Differences between two blocks.
    ///
    /// `ref_frame` and `tgt_frame` are row-major grayscale buffers of size
    /// `width * height`. The block in `ref_frame` starts at `(rx, ry)` and
    /// the candidate in `tgt_frame` starts at `(tx, ty)`.
    ///
    /// Returns `u64::MAX` if any pixel falls out of bounds.
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::too_many_arguments
    )]
    pub fn compute_sad(
        &self,
        ref_frame: &[u8],
        tgt_frame: &[u8],
        width: usize,
        height: usize,
        rx: usize,
        ry: usize,
        tx: usize,
        ty: usize,
    ) -> u64 {
        let bs = self.block_size;
        if rx + bs > width || ry + bs > height || tx + bs > width || ty + bs > height {
            return u64::MAX;
        }
        let mut sad: u64 = 0;
        for row in 0..bs {
            for col in 0..bs {
                let r_val = ref_frame[(ry + row) * width + (rx + col)] as i32;
                let t_val = tgt_frame[(ty + row) * width + (tx + col)] as i32;
                sad += (r_val - t_val).unsigned_abs() as u64;
            }
        }
        sad
    }

    /// Estimate the motion vector field between a reference and target frame.
    ///
    /// Both frames are row-major grayscale buffers of the same dimensions.
    /// Returns `None` if the frames are not the same size or are empty.
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_possible_wrap
    )]
    pub fn estimate(
        &self,
        ref_frame: &[u8],
        tgt_frame: &[u8],
        width: usize,
        height: usize,
    ) -> Option<MotionVectorField> {
        if ref_frame.len() != width * height
            || tgt_frame.len() != width * height
            || width == 0
            || height == 0
        {
            return None;
        }

        let bs = self.block_size;
        let cols = width / bs;
        let rows = height / bs;
        if cols == 0 || rows == 0 {
            return None;
        }

        let mut field = MotionVectorField::zeros(cols, rows, bs);

        for br in 0..rows {
            for bc in 0..cols {
                let rx = bc * bs;
                let ry = br * bs;

                let mut best_dx: i32 = 0;
                let mut best_dy: i32 = 0;
                let mut best_sad = u64::MAX;

                for sy in -self.search_range..=self.search_range {
                    for sx in -self.search_range..=self.search_range {
                        let tx = rx as i32 + sx;
                        let ty = ry as i32 + sy;
                        if tx < 0 || ty < 0 {
                            continue;
                        }
                        let tx = tx as usize;
                        let ty = ty as usize;
                        let sad =
                            self.compute_sad(ref_frame, tgt_frame, width, height, rx, ry, tx, ty);
                        if sad < best_sad
                            || (sad == best_sad
                                && (sx.unsigned_abs() + sy.unsigned_abs())
                                    < (best_dx.unsigned_abs() + best_dy.unsigned_abs()))
                        {
                            best_sad = sad;
                            best_dx = sx;
                            best_dy = sy;
                        }
                    }
                }

                let mv = MotionVector::new(f64::from(best_dx), f64::from(best_dy), best_sad as f64);
                field.set(bc, br, mv);
            }
        }

        Some(field)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motion_vector_new() {
        let mv = MotionVector::new(3.0, 4.0, 10.0);
        assert!((mv.dx - 3.0).abs() < f64::EPSILON);
        assert!((mv.dy - 4.0).abs() < f64::EPSILON);
        assert!((mv.cost - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_motion_vector_zero() {
        let mv = MotionVector::zero();
        assert!((mv.magnitude()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_magnitude() {
        let mv = MotionVector::new(3.0, 4.0, 0.0);
        assert!((mv.magnitude() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_direction() {
        let mv = MotionVector::new(1.0, 0.0, 0.0);
        assert!((mv.direction()).abs() < 1e-9); // 0 radians
        let mv2 = MotionVector::new(0.0, 1.0, 0.0);
        assert!((mv2.direction() - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
    }

    #[test]
    fn test_field_zeros() {
        let field = MotionVectorField::zeros(4, 3, 16);
        assert_eq!(field.vectors.len(), 12);
        assert!((field.avg_magnitude()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_field_get_set() {
        let mut field = MotionVectorField::zeros(4, 3, 16);
        field.set(2, 1, MotionVector::new(5.0, 0.0, 1.0));
        let v = field.get(2, 1).expect("get should succeed");
        assert!((v.dx - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_field_get_out_of_bounds() {
        let field = MotionVectorField::zeros(4, 3, 16);
        assert!(field.get(10, 10).is_none());
    }

    #[test]
    fn test_field_avg_magnitude() {
        let mut field = MotionVectorField::zeros(2, 2, 8);
        field.set(0, 0, MotionVector::new(3.0, 4.0, 0.0)); // mag = 5
        field.set(1, 1, MotionVector::new(0.0, 0.0, 0.0)); // mag = 0
        let avg = field.avg_magnitude();
        // (5 + 0 + 0 + 0) / 4 = 1.25
        assert!((avg - 1.25).abs() < 1e-9);
    }

    #[test]
    fn test_field_max_magnitude() {
        let mut field = MotionVectorField::zeros(2, 1, 8);
        field.set(0, 0, MotionVector::new(3.0, 4.0, 0.0));
        field.set(1, 0, MotionVector::new(1.0, 0.0, 0.0));
        assert!((field.max_magnitude() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_compute_sad_identical() {
        let frame = vec![128u8; 64]; // 8x8
        let analyzer = MotionAnalyzer::new(4, 2);
        let sad = analyzer.compute_sad(&frame, &frame, 8, 8, 0, 0, 0, 0);
        assert_eq!(sad, 0);
    }

    #[test]
    fn test_compute_sad_different() {
        let ref_frame = vec![100u8; 64];
        let tgt_frame = vec![110u8; 64];
        let analyzer = MotionAnalyzer::new(4, 1);
        let sad = analyzer.compute_sad(&ref_frame, &tgt_frame, 8, 8, 0, 0, 0, 0);
        // 4x4 block, each pixel differs by 10 => 16 * 10 = 160
        assert_eq!(sad, 160);
    }

    #[test]
    fn test_compute_sad_out_of_bounds() {
        let frame = vec![0u8; 64];
        let analyzer = MotionAnalyzer::new(4, 1);
        let sad = analyzer.compute_sad(&frame, &frame, 8, 8, 6, 6, 0, 0);
        assert_eq!(sad, u64::MAX);
    }

    #[test]
    fn test_estimate_identical_frames() {
        let frame = vec![50u8; 16 * 16];
        let analyzer = MotionAnalyzer::new(8, 2);
        let field = analyzer
            .estimate(&frame, &frame, 16, 16)
            .expect("estimate should succeed");
        assert_eq!(field.cols, 2);
        assert_eq!(field.rows, 2);
        for v in &field.vectors {
            assert!((v.dx).abs() < f64::EPSILON);
            assert!((v.dy).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_estimate_mismatched_sizes() {
        let analyzer = MotionAnalyzer::new(4, 1);
        let a = vec![0u8; 100];
        let b = vec![0u8; 200];
        assert!(analyzer.estimate(&a, &b, 10, 10).is_none());
    }

    #[test]
    fn test_estimate_empty() {
        let analyzer = MotionAnalyzer::new(4, 1);
        assert!(analyzer.estimate(&[], &[], 0, 0).is_none());
    }

    #[test]
    fn test_field_avg_cost() {
        let mut field = MotionVectorField::zeros(2, 1, 8);
        field.set(0, 0, MotionVector::new(0.0, 0.0, 100.0));
        field.set(1, 0, MotionVector::new(0.0, 0.0, 200.0));
        assert!((field.avg_cost() - 150.0).abs() < 1e-9);
    }

    #[test]
    fn test_analyzer_clamps_params() {
        let a = MotionAnalyzer::new(1, 0);
        assert!(a.block_size >= 4);
        assert!(a.search_range >= 1);
    }
}
