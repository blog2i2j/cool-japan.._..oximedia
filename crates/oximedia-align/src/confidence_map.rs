#![allow(dead_code)]
//! Alignment confidence and quality mapping.
//!
//! This module provides tools for computing, storing, and querying per-pixel or
//! per-region confidence values that quantify how reliable an alignment result is.
//! Confidence maps are useful for:
//!
//! - **Blending** -- weight contributions from different views during stitching
//! - **Quality filtering** -- reject poorly-aligned regions
//! - **Diagnostics** -- visualise alignment quality as a heat map
//!
//! # Overview
//!
//! A [`ConfidenceMap`] is a 2-D grid of `f64` values in `[0.0, 1.0]` where
//! `1.0` means perfect confidence and `0.0` means no confidence.
//!
//! The module also provides aggregation helpers ([`ConfidenceStats`]) and a
//! [`ConfidenceThresholder`] for producing binary accept/reject masks.

/// A 2-D confidence map stored as a flat row-major vector.
#[derive(Debug, Clone, PartialEq)]
pub struct ConfidenceMap {
    /// Width (number of columns)
    pub width: usize,
    /// Height (number of rows)
    pub height: usize,
    /// Row-major confidence values clamped to `[0, 1]`
    pub data: Vec<f64>,
}

impl ConfidenceMap {
    /// Create a new confidence map initialised to zero.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0.0; width * height],
        }
    }

    /// Create a confidence map filled with a constant value.
    pub fn filled(width: usize, height: usize, value: f64) -> Self {
        let v = value.clamp(0.0, 1.0);
        Self {
            width,
            height,
            data: vec![v; width * height],
        }
    }

    /// Total number of pixels.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` when the map has zero pixels.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get the confidence value at `(x, y)`. Returns `None` if out of bounds.
    pub fn get(&self, x: usize, y: usize) -> Option<f64> {
        if x < self.width && y < self.height {
            Some(self.data[y * self.width + x])
        } else {
            None
        }
    }

    /// Set the confidence value at `(x, y)`. The value is clamped to `[0, 1]`.
    /// Returns `false` if out of bounds.
    pub fn set(&mut self, x: usize, y: usize, value: f64) -> bool {
        if x < self.width && y < self.height {
            self.data[y * self.width + x] = value.clamp(0.0, 1.0);
            true
        } else {
            false
        }
    }

    /// Apply a Gaussian blur to smooth the confidence map.
    /// `radius` is the kernel half-size (the full kernel is `2*radius+1`).
    #[allow(clippy::cast_precision_loss)]
    pub fn gaussian_blur(&self, radius: usize) -> Self {
        if radius == 0 || self.is_empty() {
            return self.clone();
        }
        let kernel_size = 2 * radius + 1;
        let sigma = radius as f64 / 2.0;
        let mut kernel = vec![0.0f64; kernel_size];
        let mut sum = 0.0;
        for i in 0..kernel_size {
            let x = i as f64 - radius as f64;
            let val = (-x * x / (2.0 * sigma * sigma)).exp();
            kernel[i] = val;
            sum += val;
        }
        for v in &mut kernel {
            *v /= sum;
        }

        // Horizontal pass
        let mut temp = vec![0.0f64; self.data.len()];
        for y in 0..self.height {
            for x in 0..self.width {
                let mut acc = 0.0;
                for k in 0..kernel_size {
                    let sx = (x as isize + k as isize - radius as isize)
                        .max(0)
                        .min(self.width as isize - 1) as usize;
                    acc += self.data[y * self.width + sx] * kernel[k];
                }
                temp[y * self.width + x] = acc;
            }
        }

        // Vertical pass
        let mut result = vec![0.0f64; self.data.len()];
        for y in 0..self.height {
            for x in 0..self.width {
                let mut acc = 0.0;
                for k in 0..kernel_size {
                    let sy = (y as isize + k as isize - radius as isize)
                        .max(0)
                        .min(self.height as isize - 1) as usize;
                    acc += temp[sy * self.width + x] * kernel[k];
                }
                result[y * self.width + x] = acc.clamp(0.0, 1.0);
            }
        }

        Self {
            width: self.width,
            height: self.height,
            data: result,
        }
    }

    /// Compute aggregate statistics for this map.
    #[allow(clippy::cast_precision_loss)]
    pub fn statistics(&self) -> ConfidenceStats {
        if self.data.is_empty() {
            return ConfidenceStats {
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                std_dev: 0.0,
                median: 0.0,
            };
        }
        let n = self.data.len() as f64;
        let min = self.data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = self
            .data
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let mean = self.data.iter().sum::<f64>() / n;
        let variance = self.data.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();
        let mut sorted = self.data.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if sorted.len() % 2 == 0 {
            let mid = sorted.len() / 2;
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };
        ConfidenceStats {
            min,
            max,
            mean,
            std_dev,
            median,
        }
    }
}

/// Aggregate statistics for a confidence map.
#[derive(Debug, Clone, PartialEq)]
pub struct ConfidenceStats {
    /// Minimum confidence value
    pub min: f64,
    /// Maximum confidence value
    pub max: f64,
    /// Mean confidence value
    pub mean: f64,
    /// Standard deviation of confidence values
    pub std_dev: f64,
    /// Median confidence value
    pub median: f64,
}

/// Threshold a confidence map to produce a binary mask.
#[derive(Debug, Clone)]
pub struct ConfidenceThresholder {
    /// Threshold value in `[0, 1]`
    pub threshold: f64,
}

impl ConfidenceThresholder {
    /// Create a thresholder with the given cutoff.
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold: threshold.clamp(0.0, 1.0),
        }
    }

    /// Apply the threshold and return a binary mask (`true` = accepted).
    pub fn apply(&self, map: &ConfidenceMap) -> Vec<bool> {
        map.data.iter().map(|&v| v >= self.threshold).collect()
    }

    /// Count the number of accepted pixels.
    pub fn count_accepted(&self, map: &ConfidenceMap) -> usize {
        map.data.iter().filter(|&&v| v >= self.threshold).count()
    }

    /// Return the acceptance ratio `[0, 1]`.
    #[allow(clippy::cast_precision_loss)]
    pub fn acceptance_ratio(&self, map: &ConfidenceMap) -> f64 {
        if map.is_empty() {
            return 0.0;
        }
        self.count_accepted(map) as f64 / map.data.len() as f64
    }
}

/// Merge two confidence maps by taking the element-wise maximum.
pub fn merge_max(a: &ConfidenceMap, b: &ConfidenceMap) -> Option<ConfidenceMap> {
    if a.width != b.width || a.height != b.height {
        return None;
    }
    let data: Vec<f64> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(&va, &vb)| va.max(vb))
        .collect();
    Some(ConfidenceMap {
        width: a.width,
        height: a.height,
        data,
    })
}

/// Merge two confidence maps by taking the element-wise minimum.
pub fn merge_min(a: &ConfidenceMap, b: &ConfidenceMap) -> Option<ConfidenceMap> {
    if a.width != b.width || a.height != b.height {
        return None;
    }
    let data: Vec<f64> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(&va, &vb)| va.min(vb))
        .collect();
    Some(ConfidenceMap {
        width: a.width,
        height: a.height,
        data,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_map_zeroed() {
        let m = ConfidenceMap::new(4, 3);
        assert_eq!(m.width, 4);
        assert_eq!(m.height, 3);
        assert_eq!(m.len(), 12);
        assert!(m.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_filled_clamped() {
        let m = ConfidenceMap::filled(2, 2, 1.5);
        assert!(m.data.iter().all(|&v| (v - 1.0).abs() < 1e-12));
    }

    #[test]
    fn test_get_set() {
        let mut m = ConfidenceMap::new(3, 3);
        assert!(m.set(1, 2, 0.75));
        assert!((m.get(1, 2).expect("get should succeed") - 0.75).abs() < 1e-12);
    }

    #[test]
    fn test_get_out_of_bounds() {
        let m = ConfidenceMap::new(2, 2);
        assert!(m.get(5, 0).is_none());
    }

    #[test]
    fn test_set_out_of_bounds() {
        let mut m = ConfidenceMap::new(2, 2);
        assert!(!m.set(5, 0, 0.5));
    }

    #[test]
    fn test_set_clamps_value() {
        let mut m = ConfidenceMap::new(2, 2);
        m.set(0, 0, 2.0);
        assert!((m.get(0, 0).expect("get should succeed") - 1.0).abs() < 1e-12);
        m.set(0, 0, -1.0);
        assert!(m.get(0, 0).expect("get should succeed").abs() < 1e-12);
    }

    #[test]
    fn test_is_empty() {
        let m = ConfidenceMap::new(0, 0);
        assert!(m.is_empty());
        let m2 = ConfidenceMap::new(1, 1);
        assert!(!m2.is_empty());
    }

    #[test]
    fn test_statistics_uniform() {
        let m = ConfidenceMap::filled(3, 3, 0.5);
        let s = m.statistics();
        assert!((s.min - 0.5).abs() < 1e-12);
        assert!((s.max - 0.5).abs() < 1e-12);
        assert!((s.mean - 0.5).abs() < 1e-12);
        assert!(s.std_dev < 1e-12);
        assert!((s.median - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_statistics_varied() {
        let mut m = ConfidenceMap::new(3, 1);
        m.set(0, 0, 0.0);
        m.set(1, 0, 0.5);
        m.set(2, 0, 1.0);
        let s = m.statistics();
        assert!(s.min.abs() < 1e-12);
        assert!((s.max - 1.0).abs() < 1e-12);
        assert!((s.mean - 0.5).abs() < 1e-12);
        assert!((s.median - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_thresholder() {
        let mut m = ConfidenceMap::new(2, 2);
        m.set(0, 0, 0.3);
        m.set(1, 0, 0.7);
        m.set(0, 1, 0.5);
        m.set(1, 1, 0.9);
        let t = ConfidenceThresholder::new(0.5);
        let mask = t.apply(&m);
        assert!(!mask[0]); // 0.3 < 0.5
        assert!(mask[1]); // 0.7 >= 0.5
        assert!(mask[2]); // 0.5 >= 0.5
        assert!(mask[3]); // 0.9 >= 0.5
        assert_eq!(t.count_accepted(&m), 3);
        assert!((t.acceptance_ratio(&m) - 0.75).abs() < 1e-12);
    }

    #[test]
    fn test_merge_max() {
        let a = ConfidenceMap {
            width: 2,
            height: 1,
            data: vec![0.3, 0.8],
        };
        let b = ConfidenceMap {
            width: 2,
            height: 1,
            data: vec![0.5, 0.6],
        };
        let merged = merge_max(&a, &b).expect("merged should be valid");
        assert!((merged.data[0] - 0.5).abs() < 1e-12);
        assert!((merged.data[1] - 0.8).abs() < 1e-12);
    }

    #[test]
    fn test_merge_min() {
        let a = ConfidenceMap {
            width: 2,
            height: 1,
            data: vec![0.3, 0.8],
        };
        let b = ConfidenceMap {
            width: 2,
            height: 1,
            data: vec![0.5, 0.6],
        };
        let merged = merge_min(&a, &b).expect("merged should be valid");
        assert!((merged.data[0] - 0.3).abs() < 1e-12);
        assert!((merged.data[1] - 0.6).abs() < 1e-12);
    }

    #[test]
    fn test_merge_mismatched_dimensions() {
        let a = ConfidenceMap::new(2, 2);
        let b = ConfidenceMap::new(3, 2);
        assert!(merge_max(&a, &b).is_none());
        assert!(merge_min(&a, &b).is_none());
    }

    #[test]
    fn test_gaussian_blur_preserves_uniform() {
        let m = ConfidenceMap::filled(5, 5, 0.7);
        let blurred = m.gaussian_blur(1);
        for v in &blurred.data {
            assert!((v - 0.7).abs() < 1e-6);
        }
    }
}
