#![allow(dead_code)]
//! Multi-generation JPEG compression detection.
//!
//! This module detects whether an image has undergone multiple rounds of
//! JPEG compression (re-saves), estimates the number of compression
//! generations, and identifies quality level changes. Double/triple
//! compression is a strong forensic indicator of manipulation.

use std::collections::HashMap;

/// JPEG quality factor representation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QualityFactor {
    /// Estimated quality (1..100).
    pub quality: u8,
    /// Confidence of the estimate (0.0..1.0).
    pub confidence: f64,
}

impl QualityFactor {
    /// Create a new quality factor.
    #[must_use]
    pub fn new(quality: u8, confidence: f64) -> Self {
        Self {
            quality,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Check if this represents a low-quality compression.
    #[must_use]
    pub fn is_low_quality(&self) -> bool {
        self.quality < 50
    }

    /// Check if this represents a high-quality compression.
    #[must_use]
    pub fn is_high_quality(&self) -> bool {
        self.quality >= 85
    }
}

/// Evidence of a single compression generation.
#[derive(Debug, Clone)]
pub struct CompressionGeneration {
    /// Generation index (0 = first compression, 1 = second, etc.).
    pub generation: u32,
    /// Estimated quality factor for this generation.
    pub quality: QualityFactor,
    /// Blocking artifact strength at this level.
    pub blocking_strength: f64,
    /// Quantization table hash (for fingerprinting).
    pub qtable_hash: u64,
}

impl CompressionGeneration {
    /// Create a new compression generation record.
    #[must_use]
    pub fn new(
        generation: u32,
        quality: QualityFactor,
        blocking_strength: f64,
        qtable_hash: u64,
    ) -> Self {
        Self {
            generation,
            quality,
            blocking_strength,
            qtable_hash,
        }
    }
}

/// Result of double-compression detection.
#[derive(Debug, Clone)]
pub struct DoubleCompressionResult {
    /// Whether double compression was detected.
    pub detected: bool,
    /// Confidence of the detection (0.0..1.0).
    pub confidence: f64,
    /// Estimated primary (first) quality factor.
    pub primary_quality: Option<QualityFactor>,
    /// Estimated secondary (current) quality factor.
    pub secondary_quality: Option<QualityFactor>,
    /// Per-block probability map of double compression.
    pub block_probabilities: Vec<f64>,
}

impl DoubleCompressionResult {
    /// Create a new result indicating no double compression.
    #[must_use]
    pub fn not_detected() -> Self {
        Self {
            detected: false,
            confidence: 0.0,
            primary_quality: None,
            secondary_quality: None,
            block_probabilities: Vec::new(),
        }
    }

    /// Create a new result indicating detected double compression.
    #[must_use]
    pub fn detected_with(
        primary: QualityFactor,
        secondary: QualityFactor,
        confidence: f64,
    ) -> Self {
        Self {
            detected: true,
            confidence: confidence.clamp(0.0, 1.0),
            primary_quality: Some(primary),
            secondary_quality: Some(secondary),
            block_probabilities: Vec::new(),
        }
    }

    /// Return the quality ratio between primary and secondary compression.
    #[must_use]
    pub fn quality_ratio(&self) -> Option<f64> {
        match (self.primary_quality, self.secondary_quality) {
            (Some(p), Some(s)) if s.quality > 0 => {
                Some(f64::from(p.quality) / f64::from(s.quality))
            }
            _ => None,
        }
    }
}

/// DCT coefficient histogram for a specific frequency position.
#[derive(Debug, Clone)]
pub struct DctHistogram {
    /// Frequency position (row, col) in the 8x8 block.
    pub position: (usize, usize),
    /// Histogram bin counts (centered at 0).
    pub bins: HashMap<i32, u64>,
    /// Total number of coefficients.
    pub total: u64,
}

impl DctHistogram {
    /// Create a new empty DCT histogram.
    #[must_use]
    pub fn new(row: usize, col: usize) -> Self {
        Self {
            position: (row, col),
            bins: HashMap::new(),
            total: 0,
        }
    }

    /// Add a coefficient value to the histogram.
    pub fn add(&mut self, value: i32) {
        *self.bins.entry(value).or_insert(0) += 1;
        self.total += 1;
    }

    /// Get the count for a specific bin.
    #[must_use]
    pub fn count(&self, value: i32) -> u64 {
        self.bins.get(&value).copied().unwrap_or(0)
    }

    /// Compute the proportion of zero coefficients.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn zero_proportion(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.count(0) as f64 / self.total as f64
    }

    /// Detect periodicity in the histogram (indicator of double compression).
    ///
    /// Returns the period and its strength. A period > 1 with high strength
    /// indicates double JPEG compression.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn detect_periodicity(&self, max_period: i32) -> (i32, f64) {
        if self.bins.is_empty() {
            return (1, 0.0);
        }

        let min_key = self.bins.keys().copied().min().unwrap_or(0);
        let max_key = self.bins.keys().copied().max().unwrap_or(0);
        let range = max_key - min_key + 1;
        if range < 4 {
            return (1, 0.0);
        }

        let mut best_period = 1;
        let mut best_score = 0.0_f64;

        for period in 2..=max_period {
            let mut on_grid = 0_u64;
            let mut off_grid = 0_u64;

            for (&val, &cnt) in &self.bins {
                if val % period == 0 {
                    on_grid += cnt;
                } else {
                    off_grid += cnt;
                }
            }

            let total = on_grid + off_grid;
            if total == 0 {
                continue;
            }
            let expected_on = total as f64 / period as f64;
            let score = (on_grid as f64 - expected_on) / expected_on.max(1.0);

            if score > best_score {
                best_score = score;
                best_period = period;
            }
        }

        (best_period, best_score.max(0.0))
    }
}

/// Blocking artifact grid analyzer.
#[derive(Debug, Clone)]
pub struct BlockingAnalyzer {
    /// Block size (typically 8 for JPEG).
    pub block_size: usize,
    /// Detected blocking strength at JPEG grid positions.
    pub grid_strength: f64,
    /// Detected blocking strength at shifted positions.
    pub shifted_strength: f64,
}

impl BlockingAnalyzer {
    /// Create a new blocking analyzer.
    #[must_use]
    pub fn new(block_size: usize) -> Self {
        Self {
            block_size,
            grid_strength: 0.0,
            shifted_strength: 0.0,
        }
    }

    /// Analyze blocking artifacts from a row of pixel differences.
    #[allow(clippy::cast_precision_loss)]
    pub fn analyze_row(&mut self, diffs: &[f64]) {
        if diffs.is_empty() || self.block_size == 0 {
            return;
        }

        let mut grid_sum = 0.0;
        let mut grid_count = 0_u64;
        let mut non_grid_sum = 0.0;
        let mut non_grid_count = 0_u64;

        for (i, &d) in diffs.iter().enumerate() {
            let abs_d = d.abs();
            if (i + 1) % self.block_size == 0 {
                grid_sum += abs_d;
                grid_count += 1;
            } else {
                non_grid_sum += abs_d;
                non_grid_count += 1;
            }
        }

        self.grid_strength = if grid_count > 0 {
            grid_sum / grid_count as f64
        } else {
            0.0
        };
        self.shifted_strength = if non_grid_count > 0 {
            non_grid_sum / non_grid_count as f64
        } else {
            0.0
        };
    }

    /// Return the ratio of grid-aligned to non-grid blocking.
    ///
    /// A ratio significantly above 1.0 indicates JPEG blocking artifacts.
    #[must_use]
    pub fn blocking_ratio(&self) -> f64 {
        if self.shifted_strength > 1e-10 {
            self.grid_strength / self.shifted_strength
        } else if self.grid_strength > 1e-10 {
            f64::INFINITY
        } else {
            1.0
        }
    }

    /// Check whether blocking artifacts are present at the JPEG grid.
    #[must_use]
    pub fn has_blocking_artifacts(&self, threshold_ratio: f64) -> bool {
        self.blocking_ratio() > threshold_ratio
    }
}

/// Comprehensive compression history analysis result.
#[derive(Debug, Clone)]
pub struct CompressionHistory {
    /// Detected compression generations.
    pub generations: Vec<CompressionGeneration>,
    /// Overall number of detected compression rounds.
    pub num_generations: u32,
    /// Double compression detection result.
    pub double_compression: DoubleCompressionResult,
    /// Blocking artifact analysis.
    pub blocking_ratio: f64,
    /// Textual findings.
    pub findings: Vec<String>,
}

impl CompressionHistory {
    /// Create a new empty compression history.
    #[must_use]
    pub fn new() -> Self {
        Self {
            generations: Vec::new(),
            num_generations: 0,
            double_compression: DoubleCompressionResult::not_detected(),
            blocking_ratio: 1.0,
            findings: Vec::new(),
        }
    }

    /// Add a compression generation.
    pub fn add_generation(&mut self, gen: CompressionGeneration) {
        self.generations.push(gen);
        self.num_generations = self.generations.len() as u32;
    }

    /// Add a finding.
    pub fn add_finding(&mut self, finding: &str) {
        self.findings.push(finding.to_string());
    }

    /// Whether multiple compression rounds were detected.
    #[must_use]
    pub fn is_multi_compressed(&self) -> bool {
        self.num_generations > 1 || self.double_compression.detected
    }
}

impl Default for CompressionHistory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_factor_creation() {
        let qf = QualityFactor::new(75, 0.9);
        assert_eq!(qf.quality, 75);
        assert!((qf.confidence - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_quality_factor_clamped_confidence() {
        let qf = QualityFactor::new(50, 1.5);
        assert!((qf.confidence - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_quality_factor_low_high() {
        assert!(QualityFactor::new(30, 0.9).is_low_quality());
        assert!(!QualityFactor::new(70, 0.9).is_low_quality());
        assert!(QualityFactor::new(90, 0.9).is_high_quality());
        assert!(!QualityFactor::new(50, 0.9).is_high_quality());
    }

    #[test]
    fn test_double_compression_not_detected() {
        let r = DoubleCompressionResult::not_detected();
        assert!(!r.detected);
        assert!(r.quality_ratio().is_none());
    }

    #[test]
    fn test_double_compression_detected() {
        let r = DoubleCompressionResult::detected_with(
            QualityFactor::new(90, 0.8),
            QualityFactor::new(75, 0.9),
            0.85,
        );
        assert!(r.detected);
        let ratio = r.quality_ratio().expect("ratio should be valid");
        assert!((ratio - 1.2).abs() < 1e-10);
    }

    #[test]
    fn test_dct_histogram_basic() {
        let mut hist = DctHistogram::new(0, 0);
        hist.add(0);
        hist.add(0);
        hist.add(1);
        hist.add(-1);
        assert_eq!(hist.count(0), 2);
        assert_eq!(hist.count(1), 1);
        assert_eq!(hist.total, 4);
    }

    #[test]
    fn test_dct_histogram_zero_proportion() {
        let mut hist = DctHistogram::new(0, 0);
        hist.add(0);
        hist.add(0);
        hist.add(1);
        assert!((hist.zero_proportion() - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_dct_histogram_empty() {
        let hist = DctHistogram::new(0, 0);
        assert!((hist.zero_proportion()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dct_histogram_periodicity() {
        let mut hist = DctHistogram::new(1, 1);
        // Strong periodicity at period 2
        for v in -10..=10 {
            let count = if v % 2 == 0 { 10 } else { 1 };
            for _ in 0..count {
                hist.add(v);
            }
        }
        let (period, strength) = hist.detect_periodicity(4);
        assert_eq!(period, 2);
        assert!(strength > 0.5);
    }

    #[test]
    fn test_blocking_analyzer_ratio() {
        let mut ba = BlockingAnalyzer::new(8);
        let mut diffs = vec![0.5; 16];
        // Make grid-aligned positions have higher values
        for i in 0..diffs.len() {
            if (i + 1) % 8 == 0 {
                diffs[i] = 5.0;
            }
        }
        ba.analyze_row(&diffs);
        assert!(ba.blocking_ratio() > 1.0);
        assert!(ba.has_blocking_artifacts(2.0));
    }

    #[test]
    fn test_blocking_analyzer_no_artifacts() {
        let mut ba = BlockingAnalyzer::new(8);
        let diffs = vec![1.0; 16];
        ba.analyze_row(&diffs);
        // Uniform diffs: ratio should be close to 1
        assert!(ba.blocking_ratio() < 2.0);
    }

    #[test]
    fn test_blocking_analyzer_empty() {
        let mut ba = BlockingAnalyzer::new(8);
        ba.analyze_row(&[]);
        assert!((ba.blocking_ratio() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compression_history_multi() {
        let mut ch = CompressionHistory::new();
        ch.add_generation(CompressionGeneration::new(
            0,
            QualityFactor::new(90, 0.9),
            0.5,
            12345,
        ));
        ch.add_generation(CompressionGeneration::new(
            1,
            QualityFactor::new(75, 0.8),
            1.2,
            67890,
        ));
        assert_eq!(ch.num_generations, 2);
        assert!(ch.is_multi_compressed());
    }

    #[test]
    fn test_compression_history_single() {
        let mut ch = CompressionHistory::new();
        ch.add_generation(CompressionGeneration::new(
            0,
            QualityFactor::new(90, 0.9),
            0.5,
            12345,
        ));
        assert!(!ch.is_multi_compressed());
    }
}
