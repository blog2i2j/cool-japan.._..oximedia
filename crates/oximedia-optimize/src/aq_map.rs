//! Adaptive quantization map generation based on spatial complexity.
//!
//! This module generates per-CTU (Coding Tree Unit) QP delta maps by analysing
//! luma variance and edge density of each block.  High-complexity regions receive
//! a negative QP delta (better quality) while flat, easy-to-encode regions receive
//! a positive QP delta (fewer bits wasted).
//!
//! # Algorithm
//!
//! For each `block_size × block_size` block:
//! 1. Compute luma variance as a proxy for spatial complexity.
//! 2. Map the normalised variance through a sigmoid-like transfer function that
//!    outputs a QP delta in `[-aq_strength * MAX_DELTA, +aq_strength * MAX_DELTA]`.
//! 3. Clamp to the configurable `[min_delta, max_delta]` range.
//!
//! The resulting 2-D QP delta map is stored row-major, one value per CTU.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

/// Maximum QP delta magnitude applied by the AQ map generator.
pub const MAX_AQ_DELTA: i8 = 6;

/// Configuration for the spatial AQ map generator.
#[derive(Debug, Clone)]
pub struct AqMapConfig {
    /// Side length (in pixels) of each AQ block / CTU.
    pub block_size: u32,
    /// AQ strength multiplier in \[0.0, 2.0\].  1.0 is the reference strength.
    pub aq_strength: f32,
    /// Minimum QP delta (most negative = best quality boost).
    pub min_delta: i8,
    /// Maximum QP delta (most positive = most bit saving).
    pub max_delta: i8,
}

impl Default for AqMapConfig {
    fn default() -> Self {
        Self {
            block_size: 64,
            aq_strength: 1.0,
            min_delta: -MAX_AQ_DELTA,
            max_delta: MAX_AQ_DELTA,
        }
    }
}

/// A 2-D map of per-CTU QP deltas.
#[derive(Debug, Clone)]
pub struct AqMap {
    /// Width of the map in CTU columns.
    pub cols: u32,
    /// Height of the map in CTU rows.
    pub rows: u32,
    /// QP deltas stored row-major: `deltas[row * cols + col]`.
    pub deltas: Vec<i8>,
    /// Average absolute QP delta (for diagnostics).
    pub mean_abs_delta: f32,
}

impl AqMap {
    /// Returns the QP delta for a given CTU column / row.
    ///
    /// Returns `0` for out-of-bounds coordinates.
    #[must_use]
    pub fn delta_at(&self, col: u32, row: u32) -> i8 {
        if col < self.cols && row < self.rows {
            self.deltas[(row * self.cols + col) as usize]
        } else {
            0
        }
    }

    /// Number of CTUs in the map.
    #[must_use]
    pub fn ctu_count(&self) -> usize {
        (self.cols * self.rows) as usize
    }
}

/// Generator that produces an [`AqMap`] from a luma plane.
pub struct AqMapGenerator {
    config: AqMapConfig,
}

impl AqMapGenerator {
    /// Creates a new generator with the given configuration.
    #[must_use]
    pub fn new(config: AqMapConfig) -> Self {
        Self { config }
    }

    /// Creates a generator with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(AqMapConfig::default())
    }

    /// Generates an AQ map from a luma plane.
    ///
    /// `luma` is a flat slice of `u8` samples in row-major order.
    /// `width` and `height` are the dimensions of the luma plane.
    ///
    /// Returns `None` if `luma.len() != width * height` or if either dimension
    /// is zero.
    #[must_use]
    pub fn generate(&self, luma: &[u8], width: u32, height: u32) -> Option<AqMap> {
        let expected = (width as usize).checked_mul(height as usize)?;
        if luma.len() != expected || width == 0 || height == 0 {
            return None;
        }

        let bs = self.config.block_size.max(1);
        let cols = (width + bs - 1) / bs;
        let rows = (height + bs - 1) / bs;
        let total = (cols * rows) as usize;

        // First pass: compute variance for each block.
        let mut variances = Vec::with_capacity(total);
        for row in 0..rows {
            for col in 0..cols {
                let v = self.block_variance(luma, width, height, col, row);
                variances.push(v);
            }
        }

        // Find global max variance for normalisation (avoid division by zero).
        let max_var = variances.iter().cloned().fold(0.0_f32, f32::max).max(1.0);

        // Second pass: map variance to QP delta.
        let range = (self.config.max_delta - self.config.min_delta) as f32;
        let mut deltas = Vec::with_capacity(total);
        let mut abs_sum = 0.0_f32;

        for &var in &variances {
            let norm = var / max_var; // 0.0 = flat, 1.0 = highly complex
                                      // Flat regions → positive delta (save bits).
                                      // Complex regions → negative delta (spend bits).
            let raw_delta = (self.config.aq_strength * (0.5 - norm) * range).round();
            let delta =
                raw_delta.clamp(self.config.min_delta as f32, self.config.max_delta as f32) as i8;
            abs_sum += delta.unsigned_abs() as f32;
            deltas.push(delta);
        }

        let mean_abs_delta = if total > 0 {
            abs_sum / total as f32
        } else {
            0.0
        };

        Some(AqMap {
            cols,
            rows,
            deltas,
            mean_abs_delta,
        })
    }

    /// Computes luma variance for the block at grid position `(col, row)`.
    fn block_variance(&self, luma: &[u8], width: u32, height: u32, col: u32, row: u32) -> f32 {
        let bs = self.config.block_size;
        let x0 = col * bs;
        let y0 = row * bs;
        let x1 = (x0 + bs).min(width);
        let y1 = (y0 + bs).min(height);

        let mut sum = 0u64;
        let mut sum_sq = 0u64;
        let mut count = 0u32;

        for y in y0..y1 {
            for x in x0..x1 {
                let s = luma[(y * width + x) as usize] as u64;
                sum += s;
                sum_sq += s * s;
                count += 1;
            }
        }

        if count == 0 {
            return 0.0;
        }

        let n = count as f64;
        let mean = sum as f64 / n;
        let variance = (sum_sq as f64 / n) - mean * mean;
        variance.max(0.0) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_frame(value: u8, w: u32, h: u32) -> Vec<u8> {
        vec![value; (w * h) as usize]
    }

    fn gradient_frame(w: u32, h: u32) -> Vec<u8> {
        (0..(w * h))
            .map(|i| ((i * 255) / (w * h - 1).max(1)) as u8)
            .collect()
    }

    #[test]
    fn test_generate_flat_frame_returns_positive_deltas() {
        let gen = AqMapGenerator::with_defaults();
        let frame = flat_frame(128, 128, 128);
        let map = gen.generate(&frame, 128, 128).expect("should succeed");
        // Flat frame → low variance → positive QP delta (save bits).
        for &d in &map.deltas {
            assert!(d >= 0, "flat frame should have non-negative deltas");
        }
    }

    #[test]
    fn test_generate_gradient_frame_has_negative_deltas() {
        let gen = AqMapGenerator::with_defaults();
        let frame = gradient_frame(128, 128);
        let map = gen.generate(&frame, 128, 128).expect("should succeed");
        // At least some blocks should get negative delta (high complexity).
        let has_negative = map.deltas.iter().any(|&d| d < 0);
        assert!(
            has_negative,
            "gradient frame should have some negative deltas"
        );
    }

    #[test]
    fn test_generate_dimensions_match() {
        let gen = AqMapGenerator::with_defaults();
        let frame = flat_frame(64, 256, 144);
        let map = gen.generate(&frame, 256, 144).expect("map generated");
        // block_size=64: ceil(256/64)=4, ceil(144/64)=3
        assert_eq!(map.cols, 4);
        assert_eq!(map.rows, 3);
        assert_eq!(map.deltas.len(), 12);
    }

    #[test]
    fn test_generate_invalid_dims_returns_none() {
        let gen = AqMapGenerator::with_defaults();
        let frame = flat_frame(0, 0, 0);
        assert!(gen.generate(&frame, 0, 0).is_none());
    }

    #[test]
    fn test_delta_at_out_of_bounds_returns_zero() {
        let gen = AqMapGenerator::with_defaults();
        let frame = flat_frame(128, 64, 64);
        let map = gen.generate(&frame, 64, 64).expect("map generated");
        assert_eq!(map.delta_at(999, 999), 0);
    }

    #[test]
    fn test_aq_strength_zero_produces_all_zero_deltas() {
        let config = AqMapConfig {
            aq_strength: 0.0,
            ..AqMapConfig::default()
        };
        let gen = AqMapGenerator::new(config);
        let frame = gradient_frame(128, 128);
        let map = gen.generate(&frame, 128, 128).expect("map generated");
        for &d in &map.deltas {
            assert_eq!(d, 0, "zero strength should yield zero deltas");
        }
    }

    #[test]
    fn test_delta_clamped_to_range() {
        let config = AqMapConfig {
            min_delta: -3,
            max_delta: 3,
            aq_strength: 2.0,
            ..AqMapConfig::default()
        };
        let gen = AqMapGenerator::new(config);
        let frame = gradient_frame(64, 64);
        let map = gen.generate(&frame, 64, 64).expect("map generated");
        for &d in &map.deltas {
            assert!(d >= -3 && d <= 3, "delta {d} out of clamped range [-3, 3]");
        }
    }
}
