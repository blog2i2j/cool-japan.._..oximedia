//! Loudness Range (LRA) calculation.
//!
//! Implements ITU-R BS.1771 loudness range measurement using percentile-based analysis.
//!
//! LRA measures the variation in loudness of a program using the difference between
//! the 95th and 10th percentiles of the loudness distribution.

/// Loudness Range measurement.
#[derive(Clone, Copy, Debug)]
pub struct LoudnessRange {
    /// Loudness range in LU.
    pub lra: f64,
    /// 10th percentile loudness.
    pub p10: f64,
    /// 95th percentile loudness.
    pub p95: f64,
    /// Number of blocks used in calculation.
    pub block_count: usize,
}

impl LoudnessRange {
    /// Create a new loudness range measurement.
    pub fn new(lra: f64, p10: f64, p95: f64, block_count: usize) -> Self {
        Self {
            lra,
            p10,
            p95,
            block_count,
        }
    }

    /// Check if LRA measurement is valid.
    pub fn is_valid(&self) -> bool {
        self.lra.is_finite() && self.lra >= 0.0 && self.block_count >= 2
    }

    /// Check if LRA is acceptable for broadcast.
    ///
    /// Most broadcast content has LRA between 5 and 20 LU.
    pub fn is_acceptable(&self) -> bool {
        self.lra >= 1.0 && self.lra <= 30.0
    }

    /// Classify dynamic range.
    pub fn classification(&self) -> &'static str {
        if self.lra < 3.0 {
            "Very Limited"
        } else if self.lra < 6.0 {
            "Limited"
        } else if self.lra < 10.0 {
            "Moderate"
        } else if self.lra < 15.0 {
            "Wide"
        } else if self.lra < 20.0 {
            "Very Wide"
        } else {
            "Extreme"
        }
    }
}

/// Loudness Range calculator.
///
/// Calculates LRA from gated loudness blocks using ITU-R BS.1771 methodology.
pub struct LraCalculator {
    histogram: Vec<f64>,
}

impl LraCalculator {
    /// Create a new LRA calculator.
    pub fn new() -> Self {
        Self {
            histogram: Vec::new(),
        }
    }

    /// Calculate loudness range from gated blocks.
    ///
    /// # Arguments
    ///
    /// * `blocks` - Loudness blocks above absolute gate (-70 LKFS)
    ///
    /// # Returns
    ///
    /// Loudness range in LU
    pub fn calculate(&mut self, blocks: &[f64]) -> f64 {
        if blocks.len() < 2 {
            return 0.0;
        }

        // Build histogram from blocks
        self.histogram.clear();
        for &loudness in blocks {
            if loudness.is_finite() {
                self.histogram.push(loudness);
            }
        }

        if self.histogram.is_empty() {
            return 0.0;
        }

        // Sort histogram
        self.histogram
            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate percentiles
        let percentiles = self.calculate_percentiles();
        percentiles.lra
    }

    /// Calculate detailed loudness range with percentiles.
    ///
    /// # Arguments
    ///
    /// * `blocks` - Loudness blocks above absolute gate
    ///
    /// # Returns
    ///
    /// Detailed loudness range measurement
    pub fn calculate_detailed(&mut self, blocks: &[f64]) -> LoudnessRange {
        if blocks.len() < 2 {
            return LoudnessRange::new(0.0, f64::NEG_INFINITY, f64::NEG_INFINITY, 0);
        }

        // Build histogram
        self.histogram.clear();
        for &loudness in blocks {
            if loudness.is_finite() {
                self.histogram.push(loudness);
            }
        }

        if self.histogram.is_empty() {
            return LoudnessRange::new(0.0, f64::NEG_INFINITY, f64::NEG_INFINITY, 0);
        }

        // Sort histogram
        self.histogram
            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        self.calculate_percentiles()
    }

    /// Calculate percentiles from sorted histogram.
    fn calculate_percentiles(&self) -> LoudnessRange {
        let count = self.histogram.len();
        if count < 2 {
            return LoudnessRange::new(0.0, f64::NEG_INFINITY, f64::NEG_INFINITY, count);
        }

        // Calculate 10th and 95th percentile indices
        let p10_idx = ((count as f64 * 0.10).floor() as usize).min(count - 1);
        let p95_idx = ((count as f64 * 0.95).floor() as usize).min(count - 1);

        let p10 = self.histogram[p10_idx];
        let p95 = self.histogram[p95_idx];

        let lra = (p95 - p10).abs();

        LoudnessRange::new(lra, p10, p95, count)
    }

    /// Get histogram for visualization.
    ///
    /// Returns sorted loudness values.
    pub fn histogram(&self) -> &[f64] {
        &self.histogram
    }

    /// Get histogram with bin counts.
    ///
    /// # Arguments
    ///
    /// * `bin_width` - Width of each histogram bin in LU
    ///
    /// # Returns
    ///
    /// Vector of (`bin_center`, count) tuples
    pub fn histogram_binned(&self, bin_width: f64) -> Vec<(f64, usize)> {
        if self.histogram.is_empty() {
            return Vec::new();
        }

        let min = self.histogram.first().copied().unwrap_or(0.0);
        let max = self.histogram.last().copied().unwrap_or(0.0);

        let num_bins = ((max - min) / bin_width).ceil() as usize + 1;
        let mut bins = vec![0_usize; num_bins];

        for &value in &self.histogram {
            let bin_idx = ((value - min) / bin_width).floor() as usize;
            if bin_idx < num_bins {
                bins[bin_idx] += 1;
            }
        }

        bins.iter()
            .enumerate()
            .filter(|(_, &count)| count > 0)
            .map(|(idx, &count)| {
                let bin_center = min + (idx as f64 + 0.5) * bin_width;
                (bin_center, count)
            })
            .collect()
    }

    /// Calculate multiple percentiles.
    ///
    /// # Arguments
    ///
    /// * `blocks` - Loudness blocks
    /// * `percentiles` - Slice of percentile values (0.0 to 1.0)
    ///
    /// # Returns
    ///
    /// Vector of loudness values at each percentile
    pub fn calculate_percentiles_custom(
        &mut self,
        blocks: &[f64],
        percentiles: &[f64],
    ) -> Vec<f64> {
        if blocks.is_empty() {
            return vec![f64::NEG_INFINITY; percentiles.len()];
        }

        // Build and sort histogram
        self.histogram.clear();
        for &loudness in blocks {
            if loudness.is_finite() {
                self.histogram.push(loudness);
            }
        }

        if self.histogram.is_empty() {
            return vec![f64::NEG_INFINITY; percentiles.len()];
        }

        self.histogram
            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let count = self.histogram.len();

        percentiles
            .iter()
            .map(|&p| {
                let idx = ((count as f64 * p).floor() as usize).min(count - 1);
                self.histogram[idx]
            })
            .collect()
    }

    /// Reset the calculator.
    pub fn reset(&mut self) {
        self.histogram.clear();
    }
}

impl Default for LraCalculator {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate LRA from blocks (convenience function).
///
/// # Arguments
///
/// * `blocks` - Gated loudness blocks in LUFS
///
/// # Returns
///
/// Loudness range in LU
pub fn calculate_lra(blocks: &[f64]) -> f64 {
    let mut calculator = LraCalculator::new();
    calculator.calculate(blocks)
}

/// Calculate detailed LRA statistics.
///
/// # Arguments
///
/// * `blocks` - Gated loudness blocks in LUFS
///
/// # Returns
///
/// Detailed loudness range measurement
pub fn calculate_lra_detailed(blocks: &[f64]) -> LoudnessRange {
    let mut calculator = LraCalculator::new();
    calculator.calculate_detailed(blocks)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lra_calculator_creates() {
        let calculator = LraCalculator::new();
        assert_eq!(calculator.histogram().len(), 0);
    }

    #[test]
    fn test_lra_empty_blocks() {
        let lra = calculate_lra(&[]);
        assert_eq!(lra, 0.0);
    }

    #[test]
    fn test_lra_single_block() {
        let lra = calculate_lra(&[-23.0]);
        assert_eq!(lra, 0.0);
    }

    #[test]
    fn test_lra_simple_range() {
        // Create blocks from -30 to -20 LUFS (10 LU range)
        let blocks: Vec<f64> = (0..100).map(|i| -30.0 + (i as f64 / 99.0) * 10.0).collect();

        let lra = calculate_lra(&blocks);
        // LRA should be approximately 8 LU (95th - 10th percentile)
        assert!(lra > 7.0 && lra < 9.0);
    }

    #[test]
    fn test_lra_detailed() {
        let blocks: Vec<f64> = vec![-25.0, -24.0, -23.0, -22.0, -21.0, -20.0];
        let result = calculate_lra_detailed(&blocks);

        assert!(result.is_valid());
        assert_eq!(result.block_count, 6);
        assert!(result.lra > 0.0);
    }

    #[test]
    fn test_lra_classification() {
        let lra_limited = LoudnessRange::new(4.0, -25.0, -21.0, 100);
        assert_eq!(lra_limited.classification(), "Limited");

        let lra_moderate = LoudnessRange::new(8.0, -28.0, -20.0, 100);
        assert_eq!(lra_moderate.classification(), "Moderate");

        let lra_wide = LoudnessRange::new(12.0, -30.0, -18.0, 100);
        assert_eq!(lra_wide.classification(), "Wide");
    }

    #[test]
    fn test_lra_acceptability() {
        let acceptable = LoudnessRange::new(10.0, -28.0, -18.0, 100);
        assert!(acceptable.is_acceptable());

        let too_high = LoudnessRange::new(35.0, -40.0, -5.0, 100);
        assert!(!too_high.is_acceptable());
    }

    #[test]
    fn test_custom_percentiles() {
        let mut calculator = LraCalculator::new();
        let blocks: Vec<f64> = (0..100).map(|i| -30.0 + (i as f64 / 99.0) * 20.0).collect();

        let percentiles = calculator.calculate_percentiles_custom(&blocks, &[0.25, 0.50, 0.75]);
        assert_eq!(percentiles.len(), 3);
        assert!(percentiles[0] < percentiles[1]);
        assert!(percentiles[1] < percentiles[2]);
    }

    #[test]
    fn test_histogram_binned() {
        let mut calculator = LraCalculator::new();
        let blocks = vec![-25.0, -24.5, -24.0, -23.5, -23.0];
        calculator.calculate(&blocks);

        let binned = calculator.histogram_binned(0.5);
        assert!(!binned.is_empty());
    }
}
