#![allow(dead_code)]
//! Motion-compensated temporal quality analysis.
//!
//! Accounts for inter-frame motion when computing temporal quality metrics.
//! Instead of naively comparing co-located pixels, this module estimates
//! block-based motion vectors and aligns frames before measurement.
//!
//! # Features
//!
//! - Block matching with configurable block size and search range
//! - Diamond search and full search strategies
//! - Motion-compensated PSNR and temporal difference
//! - Motion field statistics (magnitude histogram, coherence)
//!
//! # Example
//!
//! ```
//! use oximedia_quality::motion_compensated::{
//!     MotionCompensatedAnalyzer, MotionConfig, SearchStrategy,
//! };
//!
//! let config = MotionConfig::default();
//! let analyzer = MotionCompensatedAnalyzer::new(config);
//! ```

use serde::{Deserialize, Serialize};

/// A 2D motion vector (horizontal, vertical displacement in pixels).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MotionVector {
    /// Horizontal displacement (positive = rightward).
    pub dx: i32,
    /// Vertical displacement (positive = downward).
    pub dy: i32,
}

impl MotionVector {
    /// Creates a new motion vector.
    #[must_use]
    pub fn new(dx: i32, dy: i32) -> Self {
        Self { dx, dy }
    }

    /// Magnitude of the motion vector.
    #[must_use]
    pub fn magnitude(&self) -> f64 {
        ((self.dx as f64).powi(2) + (self.dy as f64).powi(2)).sqrt()
    }

    /// Angle of the motion vector in degrees [0, 360).
    #[must_use]
    pub fn angle_degrees(&self) -> f64 {
        let a = (self.dy as f64).atan2(self.dx as f64).to_degrees();
        if a < 0.0 {
            a + 360.0
        } else {
            a
        }
    }
}

/// Block matching search strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchStrategy {
    /// Exhaustive search within the search range.
    FullSearch,
    /// Diamond search (faster, may miss global minimum).
    DiamondSearch,
}

/// Configuration for motion-compensated analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionConfig {
    /// Block size in pixels (width = height).
    pub block_size: usize,
    /// Search range in pixels (positive integer).
    pub search_range: usize,
    /// Search strategy.
    pub strategy: SearchStrategy,
}

impl Default for MotionConfig {
    fn default() -> Self {
        Self {
            block_size: 16,
            search_range: 16,
            strategy: SearchStrategy::DiamondSearch,
        }
    }
}

impl MotionConfig {
    /// Creates config with custom block size.
    #[must_use]
    pub fn with_block_size(mut self, bs: usize) -> Self {
        self.block_size = bs.max(4);
        self
    }

    /// Sets the search range.
    #[must_use]
    pub fn with_search_range(mut self, sr: usize) -> Self {
        self.search_range = sr.max(1);
        self
    }

    /// Sets the search strategy.
    #[must_use]
    pub fn with_strategy(mut self, s: SearchStrategy) -> Self {
        self.strategy = s;
        self
    }
}

/// Result of motion-compensated temporal quality analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionCompensatedResult {
    /// Motion-compensated mean absolute difference (lower = more temporal consistency).
    pub mc_mad: f64,
    /// Non-compensated mean absolute difference for comparison.
    pub raw_mad: f64,
    /// Motion-compensated PSNR (dB).
    pub mc_psnr_db: f64,
    /// Mean motion vector magnitude across all blocks.
    pub mean_motion_magnitude: f64,
    /// Maximum motion vector magnitude.
    pub max_motion_magnitude: f64,
    /// Motion field coherence (0..1, higher = more uniform motion).
    pub coherence: f64,
    /// Number of blocks analyzed.
    pub block_count: usize,
    /// The motion field (block-level motion vectors, row-major).
    pub motion_field: Vec<MotionVector>,
    /// Block grid dimensions (columns, rows).
    pub grid_dims: (usize, usize),
}

/// Motion-compensated temporal quality analyzer.
#[derive(Debug, Clone)]
pub struct MotionCompensatedAnalyzer {
    config: MotionConfig,
}

impl MotionCompensatedAnalyzer {
    /// Creates a new analyzer with the given configuration.
    #[must_use]
    pub fn new(config: MotionConfig) -> Self {
        Self { config }
    }

    /// Analyzes temporal quality between two consecutive frames.
    ///
    /// Uses the luma plane of each frame.
    ///
    /// # Errors
    ///
    /// Returns `Err` if frames have different dimensions or are empty.
    pub fn analyze(
        &self,
        prev: &crate::Frame,
        curr: &crate::Frame,
    ) -> Result<MotionCompensatedResult, MotionError> {
        if prev.width != curr.width || prev.height != curr.height {
            return Err(MotionError::DimensionMismatch);
        }
        let w = prev.width;
        let h = prev.height;
        if w == 0 || h == 0 {
            return Err(MotionError::EmptyFrame);
        }

        let bs = self.config.block_size;
        let cols = w / bs;
        let rows = h / bs;
        if cols == 0 || rows == 0 {
            return Err(MotionError::BlockSizeTooLarge);
        }

        let prev_luma = &prev.planes[0];
        let curr_luma = &curr.planes[0];
        let prev_stride = prev.strides[0];
        let curr_stride = curr.strides[0];

        let mut motion_field = Vec::with_capacity(cols * rows);
        let mut mc_sad_sum: f64 = 0.0;
        let mut raw_sad_sum: f64 = 0.0;
        let mut mc_sse_sum: f64 = 0.0;
        let pixel_count = (cols * rows * bs * bs) as f64;

        for by in 0..rows {
            for bx in 0..cols {
                let bx_px = bx * bs;
                let by_px = by * bs;

                // Find best motion vector
                let mv = match self.config.strategy {
                    SearchStrategy::FullSearch => self.full_search(
                        prev_luma,
                        prev_stride,
                        curr_luma,
                        curr_stride,
                        w,
                        h,
                        bx_px,
                        by_px,
                        bs,
                    ),
                    SearchStrategy::DiamondSearch => self.diamond_search(
                        prev_luma,
                        prev_stride,
                        curr_luma,
                        curr_stride,
                        w,
                        h,
                        bx_px,
                        by_px,
                        bs,
                    ),
                };

                motion_field.push(mv);

                // Compute motion-compensated SAD and raw SAD
                let (mc_sad, raw_sad, mc_sse) = self.block_errors(
                    prev_luma,
                    prev_stride,
                    curr_luma,
                    curr_stride,
                    w,
                    h,
                    bx_px,
                    by_px,
                    bs,
                    &mv,
                );
                mc_sad_sum += mc_sad;
                raw_sad_sum += raw_sad;
                mc_sse_sum += mc_sse;
            }
        }

        let mc_mad = mc_sad_sum / pixel_count;
        let raw_mad = raw_sad_sum / pixel_count;

        let mc_mse = mc_sse_sum / pixel_count;
        let mc_psnr_db = if mc_mse < 1e-10 {
            100.0 // effectively infinite
        } else {
            10.0 * (255.0_f64 * 255.0 / mc_mse).log10()
        };

        let magnitudes: Vec<f64> = motion_field.iter().map(|mv| mv.magnitude()).collect();
        let mean_motion_magnitude = magnitudes.iter().sum::<f64>() / magnitudes.len().max(1) as f64;
        let max_motion_magnitude = magnitudes.iter().copied().fold(0.0_f64, f64::max);

        let coherence = compute_coherence(&motion_field, cols, rows);

        Ok(MotionCompensatedResult {
            mc_mad,
            raw_mad,
            mc_psnr_db,
            mean_motion_magnitude,
            max_motion_magnitude,
            coherence,
            block_count: cols * rows,
            motion_field,
            grid_dims: (cols, rows),
        })
    }

    /// Full search block matching (exhaustive).
    fn full_search(
        &self,
        prev: &[u8],
        prev_stride: usize,
        curr: &[u8],
        curr_stride: usize,
        w: usize,
        h: usize,
        bx: usize,
        by: usize,
        bs: usize,
    ) -> MotionVector {
        let sr = self.config.search_range as i32;
        let mut best_mv = MotionVector::new(0, 0);
        let mut best_sad = u64::MAX;

        for dy in -sr..=sr {
            for dx in -sr..=sr {
                let sad = block_sad(
                    prev,
                    prev_stride,
                    curr,
                    curr_stride,
                    w,
                    h,
                    bx,
                    by,
                    bs,
                    dx,
                    dy,
                );
                if sad < best_sad {
                    best_sad = sad;
                    best_mv = MotionVector::new(dx, dy);
                }
            }
        }
        best_mv
    }

    /// Diamond search block matching (LDSP + SDSP).
    fn diamond_search(
        &self,
        prev: &[u8],
        prev_stride: usize,
        curr: &[u8],
        curr_stride: usize,
        w: usize,
        h: usize,
        bx: usize,
        by: usize,
        bs: usize,
    ) -> MotionVector {
        let sr = self.config.search_range as i32;

        // Large diamond step pattern
        let ldsp: [(i32, i32); 9] = [
            (0, 0),
            (0, -2),
            (0, 2),
            (-2, 0),
            (2, 0),
            (-1, -1),
            (1, -1),
            (-1, 1),
            (1, 1),
        ];
        // Small diamond step pattern
        let sdsp: [(i32, i32); 5] = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)];

        let mut center_dx = 0_i32;
        let mut center_dy = 0_i32;

        // LDSP phase
        for _ in 0..sr {
            let mut best_sad = u64::MAX;
            let mut best_offset = (0_i32, 0_i32);
            let mut center_is_best = true;

            for &(odx, ody) in &ldsp {
                let dx = center_dx + odx;
                let dy = center_dy + ody;
                if dx.abs() > sr || dy.abs() > sr {
                    continue;
                }
                let sad = block_sad(
                    prev,
                    prev_stride,
                    curr,
                    curr_stride,
                    w,
                    h,
                    bx,
                    by,
                    bs,
                    dx,
                    dy,
                );
                if sad < best_sad {
                    best_sad = sad;
                    best_offset = (odx, ody);
                    center_is_best = odx == 0 && ody == 0;
                }
            }

            center_dx += best_offset.0;
            center_dy += best_offset.1;

            if center_is_best {
                break;
            }
        }

        // SDSP refinement
        let mut best_mv = MotionVector::new(center_dx, center_dy);
        let mut best_sad = block_sad(
            prev,
            prev_stride,
            curr,
            curr_stride,
            w,
            h,
            bx,
            by,
            bs,
            center_dx,
            center_dy,
        );

        for &(odx, ody) in &sdsp[1..] {
            let dx = center_dx + odx;
            let dy = center_dy + ody;
            if dx.abs() > sr || dy.abs() > sr {
                continue;
            }
            let sad = block_sad(
                prev,
                prev_stride,
                curr,
                curr_stride,
                w,
                h,
                bx,
                by,
                bs,
                dx,
                dy,
            );
            if sad < best_sad {
                best_sad = sad;
                best_mv = MotionVector::new(dx, dy);
            }
        }

        best_mv
    }

    /// Computes block-level SAD, raw SAD, and SSE for a given motion vector.
    fn block_errors(
        &self,
        prev: &[u8],
        prev_stride: usize,
        curr: &[u8],
        curr_stride: usize,
        w: usize,
        h: usize,
        bx: usize,
        by: usize,
        bs: usize,
        mv: &MotionVector,
    ) -> (f64, f64, f64) {
        let mut mc_sad = 0.0_f64;
        let mut raw_sad = 0.0_f64;
        let mut mc_sse = 0.0_f64;

        for row in 0..bs {
            for col in 0..bs {
                let cx = bx + col;
                let cy = by + row;
                if cx >= w || cy >= h {
                    continue;
                }
                let curr_val =
                    f64::from(curr[cy * curr_stride + cx.min(curr_stride.saturating_sub(1))]);

                // Raw (no compensation)
                let prev_val_raw =
                    f64::from(prev[cy * prev_stride + cx.min(prev_stride.saturating_sub(1))]);
                raw_sad += (curr_val - prev_val_raw).abs();

                // Motion-compensated
                let px = (cx as i32 + mv.dx).clamp(0, (w as i32) - 1) as usize;
                let py = (cy as i32 + mv.dy).clamp(0, (h as i32) - 1) as usize;
                let prev_val_mc =
                    f64::from(prev[py * prev_stride + px.min(prev_stride.saturating_sub(1))]);
                let diff = curr_val - prev_val_mc;
                mc_sad += diff.abs();
                mc_sse += diff * diff;
            }
        }

        (mc_sad, raw_sad, mc_sse)
    }
}

/// Computes SAD for a block match candidate.
fn block_sad(
    prev: &[u8],
    prev_stride: usize,
    curr: &[u8],
    curr_stride: usize,
    w: usize,
    h: usize,
    bx: usize,
    by: usize,
    bs: usize,
    dx: i32,
    dy: i32,
) -> u64 {
    let mut sad = 0_u64;
    for row in 0..bs {
        for col in 0..bs {
            let cx = bx + col;
            let cy = by + row;
            if cx >= w || cy >= h {
                continue;
            }
            let px = (cx as i32 + dx).clamp(0, (w as i32) - 1) as usize;
            let py = (cy as i32 + dy).clamp(0, (h as i32) - 1) as usize;

            let c = u64::from(curr[cy * curr_stride + cx.min(curr_stride.saturating_sub(1))]);
            let p = u64::from(prev[py * prev_stride + px.min(prev_stride.saturating_sub(1))]);
            sad += c.abs_diff(p);
        }
    }
    sad
}

/// Computes motion field coherence (ratio of mean vector length to mean of individual lengths).
/// A coherent field (all vectors similar) yields coherence near 1.0.
fn compute_coherence(field: &[MotionVector], _cols: usize, _rows: usize) -> f64 {
    if field.is_empty() {
        return 0.0;
    }
    let n = field.len() as f64;
    let sum_dx: f64 = field.iter().map(|mv| mv.dx as f64).sum();
    let sum_dy: f64 = field.iter().map(|mv| mv.dy as f64).sum();
    let mean_vec_mag = ((sum_dx / n).powi(2) + (sum_dy / n).powi(2)).sqrt();
    let mean_mag: f64 = field.iter().map(|mv| mv.magnitude()).sum::<f64>() / n;

    if mean_mag < 1e-10 {
        1.0 // all zero vectors => perfectly coherent (static scene)
    } else {
        (mean_vec_mag / mean_mag).clamp(0.0, 1.0)
    }
}

/// Errors specific to motion-compensated analysis.
#[derive(Debug, Clone, thiserror::Error)]
pub enum MotionError {
    /// Frame dimensions do not match.
    #[error("frame dimensions do not match")]
    DimensionMismatch,
    /// Frame is empty (0x0).
    #[error("frame is empty")]
    EmptyFrame,
    /// Block size is larger than the frame.
    #[error("block size exceeds frame dimensions")]
    BlockSizeTooLarge,
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::PixelFormat;

    fn make_gray_frame(w: usize, h: usize, fill: u8) -> crate::Frame {
        let mut f = crate::Frame::new(w, h, PixelFormat::Gray8).expect("create frame");
        for p in f.planes[0].iter_mut() {
            *p = fill;
        }
        f
    }

    fn make_gradient_frame(w: usize, h: usize) -> crate::Frame {
        let mut f = crate::Frame::new(w, h, PixelFormat::Gray8).expect("create frame");
        for y in 0..h {
            for x in 0..w {
                f.planes[0][y * w + x] = ((x + y) % 256) as u8;
            }
        }
        f
    }

    #[test]
    fn test_identical_frames_zero_motion() {
        let frame = make_gray_frame(64, 64, 128);
        let analyzer = MotionCompensatedAnalyzer::new(MotionConfig::default().with_block_size(16));
        let result = analyzer.analyze(&frame, &frame).expect("should succeed");
        assert!(
            result.mc_mad < 1e-10,
            "identical frames should have mc_mad ~0"
        );
        assert!(result.raw_mad < 1e-10);
        assert!(
            result.mc_psnr_db >= 99.0,
            "PSNR should be very high for identical frames"
        );
        assert!(result.mean_motion_magnitude < 1e-10);
    }

    #[test]
    fn test_shifted_frame_detects_motion() {
        let w = 64;
        let h = 64;
        let prev = make_gradient_frame(w, h);
        // Shift right by 2 pixels
        let mut curr = crate::Frame::new(w, h, PixelFormat::Gray8).expect("frame");
        for y in 0..h {
            for x in 0..w {
                let src_x = if x >= 2 { x - 2 } else { 0 };
                curr.planes[0][y * w + x] = prev.planes[0][y * w + src_x];
            }
        }

        let analyzer = MotionCompensatedAnalyzer::new(
            MotionConfig::default()
                .with_block_size(16)
                .with_search_range(8),
        );
        let result = analyzer.analyze(&prev, &curr).expect("should succeed");
        // MC-PSNR should be higher than raw since motion is compensated
        assert!(result.mean_motion_magnitude > 0.0, "should detect motion");
    }

    #[test]
    fn test_full_search_strategy() {
        let frame = make_gradient_frame(32, 32);
        let config = MotionConfig::default()
            .with_block_size(8)
            .with_search_range(4)
            .with_strategy(SearchStrategy::FullSearch);
        let analyzer = MotionCompensatedAnalyzer::new(config);
        let result = analyzer.analyze(&frame, &frame).expect("should succeed");
        assert!(result.mc_mad < 1e-10);
    }

    #[test]
    fn test_dimension_mismatch() {
        let f1 = make_gray_frame(64, 64, 0);
        let f2 = make_gray_frame(32, 32, 0);
        let analyzer = MotionCompensatedAnalyzer::new(MotionConfig::default());
        let result = analyzer.analyze(&f1, &f2);
        assert!(result.is_err());
    }

    #[test]
    fn test_block_size_too_large() {
        let frame = make_gray_frame(8, 8, 0);
        let analyzer = MotionCompensatedAnalyzer::new(MotionConfig::default().with_block_size(16));
        let result = analyzer.analyze(&frame, &frame);
        assert!(result.is_err());
    }

    #[test]
    fn test_motion_vector_basics() {
        let mv = MotionVector::new(3, 4);
        assert!((mv.magnitude() - 5.0).abs() < 1e-10);

        let mv0 = MotionVector::new(0, 0);
        assert!(mv0.magnitude() < 1e-10);
    }

    #[test]
    fn test_coherence_uniform_field() {
        let field: Vec<MotionVector> = (0..16).map(|_| MotionVector::new(3, 4)).collect();
        let c = compute_coherence(&field, 4, 4);
        assert!(
            (c - 1.0).abs() < 1e-10,
            "uniform field should have coherence ~1.0, got {c}"
        );
    }

    #[test]
    fn test_coherence_random_field() {
        // Opposing vectors cancel out => low coherence
        let field = vec![
            MotionVector::new(5, 0),
            MotionVector::new(-5, 0),
            MotionVector::new(0, 5),
            MotionVector::new(0, -5),
        ];
        let c = compute_coherence(&field, 2, 2);
        assert!(
            c < 0.5,
            "opposing vectors should have low coherence, got {c}"
        );
    }

    #[test]
    fn test_grid_dimensions() {
        let frame = make_gray_frame(64, 32, 100);
        let analyzer = MotionCompensatedAnalyzer::new(MotionConfig::default().with_block_size(16));
        let result = analyzer.analyze(&frame, &frame).expect("succeed");
        assert_eq!(result.grid_dims, (4, 2));
        assert_eq!(result.block_count, 8);
        assert_eq!(result.motion_field.len(), 8);
    }

    #[test]
    fn test_mc_improves_on_raw_for_shifted_content() {
        let w = 64;
        let h = 64;
        let prev = make_gradient_frame(w, h);
        // Shift down by 4
        let mut curr = crate::Frame::new(w, h, PixelFormat::Gray8).expect("frame");
        for y in 0..h {
            for x in 0..w {
                let src_y = if y >= 4 { y - 4 } else { 0 };
                curr.planes[0][y * w + x] = prev.planes[0][src_y * w + x];
            }
        }

        let analyzer = MotionCompensatedAnalyzer::new(
            MotionConfig::default()
                .with_block_size(16)
                .with_search_range(8),
        );
        let result = analyzer.analyze(&prev, &curr).expect("succeed");
        // MC should reduce error compared to raw
        assert!(
            result.mc_mad <= result.raw_mad + 1e-6,
            "mc_mad ({}) should be <= raw_mad ({})",
            result.mc_mad,
            result.raw_mad,
        );
    }
}
