//! Compression statistics accumulator.
//!
//! `CompressionStats` tracks running totals of original vs. compressed sizes
//! and derives compression ratio / space-saving metrics from them.
//!
//! # Example
//! ```rust
//! use oximedia_archive::stats::CompressionStats;
//!
//! let mut stats = CompressionStats::new();
//! stats.record(1_000_000, 400_000); // 1 MB → 400 kB
//! stats.record(500_000, 250_000);   // 500 kB → 250 kB
//!
//! // Overall ratio: 650_000 / 1_500_000 ≈ 0.433
//! assert!((stats.ratio() - (650_000.0 / 1_500_000.0)).abs() < 1e-6);
//! ```

#![allow(dead_code)]

/// Accumulates compression statistics across multiple files or segments.
#[derive(Debug, Clone, Default)]
pub struct CompressionStats {
    /// Sum of original (uncompressed) sizes in bytes.
    total_original: u64,
    /// Sum of compressed sizes in bytes.
    total_compressed: u64,
    /// Number of entries recorded.
    entry_count: u64,
    /// Smallest compression ratio seen (min compressed/original).
    min_ratio: f32,
    /// Largest compression ratio seen (max compressed/original).
    max_ratio: f32,
}

impl CompressionStats {
    /// Create a new, empty stats accumulator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            total_original: 0,
            total_compressed: 0,
            entry_count: 0,
            min_ratio: f32::MAX,
            max_ratio: f32::MIN,
        }
    }

    /// Record a single file's original and compressed sizes.
    ///
    /// Both values should be in bytes.  If `original` is 0 the entry is
    /// counted but the per-entry ratio is not included in the min/max
    /// statistics (to avoid division by zero).
    pub fn record(&mut self, original: u64, compressed: u64) {
        self.total_original = self.total_original.saturating_add(original);
        self.total_compressed = self.total_compressed.saturating_add(compressed);
        self.entry_count += 1;

        if original > 0 {
            let ratio = compressed as f32 / original as f32;
            if ratio < self.min_ratio {
                self.min_ratio = ratio;
            }
            if ratio > self.max_ratio {
                self.max_ratio = ratio;
            }
        }
    }

    /// Overall compression ratio: `total_compressed / total_original`.
    ///
    /// Returns `0.0` if no entries have been recorded or the total original
    /// size is zero.  A ratio below `1.0` means the data was compressed;
    /// above `1.0` means the output is larger than the input (expansion).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn ratio(&self) -> f32 {
        if self.total_original == 0 {
            return 0.0;
        }
        self.total_compressed as f32 / self.total_original as f32
    }

    /// Space savings as a fraction: `1.0 - ratio()`.
    ///
    /// `0.6` means 60 % of the original space was saved.  Negative values
    /// indicate expansion.
    #[must_use]
    pub fn space_savings(&self) -> f32 {
        1.0 - self.ratio()
    }

    /// Total original (uncompressed) bytes recorded.
    #[must_use]
    pub fn total_original_bytes(&self) -> u64 {
        self.total_original
    }

    /// Total compressed bytes recorded.
    #[must_use]
    pub fn total_compressed_bytes(&self) -> u64 {
        self.total_compressed
    }

    /// Total bytes saved (`total_original - total_compressed`).
    ///
    /// Returns `0` if the compressed total is larger than the original total.
    #[must_use]
    pub fn bytes_saved(&self) -> u64 {
        self.total_original.saturating_sub(self.total_compressed)
    }

    /// Number of entries recorded.
    #[must_use]
    pub fn entry_count(&self) -> u64 {
        self.entry_count
    }

    /// Best (lowest) per-entry compression ratio seen.
    ///
    /// Returns `None` if no entries with non-zero original size have been
    /// recorded yet.
    #[must_use]
    pub fn min_ratio(&self) -> Option<f32> {
        if self.entry_count == 0 || self.min_ratio == f32::MAX {
            None
        } else {
            Some(self.min_ratio)
        }
    }

    /// Worst (highest) per-entry compression ratio seen.
    ///
    /// Returns `None` if no entries with non-zero original size have been
    /// recorded yet.
    #[must_use]
    pub fn max_ratio(&self) -> Option<f32> {
        if self.entry_count == 0 || self.max_ratio == f32::MIN {
            None
        } else {
            Some(self.max_ratio)
        }
    }

    /// Reset all statistics back to zero.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Merge another `CompressionStats` into this one.
    pub fn merge(&mut self, other: &CompressionStats) {
        self.total_original = self.total_original.saturating_add(other.total_original);
        self.total_compressed = self.total_compressed.saturating_add(other.total_compressed);
        self.entry_count += other.entry_count;

        if other.min_ratio < self.min_ratio {
            self.min_ratio = other.min_ratio;
        }
        if other.max_ratio > self.max_ratio {
            self.max_ratio = other.max_ratio;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_is_empty() {
        let s = CompressionStats::new();
        assert_eq!(s.entry_count(), 0);
        assert_eq!(s.ratio(), 0.0);
        assert!(s.min_ratio().is_none());
        assert!(s.max_ratio().is_none());
    }

    #[test]
    fn test_record_single_entry() {
        let mut s = CompressionStats::new();
        s.record(1000, 400);
        assert_eq!(s.entry_count(), 1);
        assert!((s.ratio() - 0.4).abs() < 1e-6);
        assert!((s.space_savings() - 0.6).abs() < 1e-6);
        assert_eq!(s.bytes_saved(), 600);
    }

    #[test]
    fn test_record_multiple_entries_aggregated() {
        let mut s = CompressionStats::new();
        s.record(1_000_000, 400_000);
        s.record(500_000, 250_000);
        // total_original = 1_500_000, total_compressed = 650_000
        let expected = 650_000.0_f32 / 1_500_000.0;
        assert!((s.ratio() - expected).abs() < 1e-6);
        assert_eq!(s.entry_count(), 2);
    }

    #[test]
    fn test_ratio_no_compression() {
        let mut s = CompressionStats::new();
        s.record(100, 100);
        assert!((s.ratio() - 1.0).abs() < 1e-6);
        assert!((s.space_savings()).abs() < 1e-6);
    }

    #[test]
    fn test_ratio_expansion() {
        let mut s = CompressionStats::new();
        s.record(100, 150); // output is larger
        assert!(s.ratio() > 1.0);
        assert!(s.space_savings() < 0.0);
    }

    #[test]
    fn test_zero_original_size_no_panic() {
        let mut s = CompressionStats::new();
        s.record(0, 0);
        assert_eq!(s.entry_count(), 1);
        assert_eq!(s.ratio(), 0.0);
    }

    #[test]
    fn test_min_max_ratio() {
        let mut s = CompressionStats::new();
        s.record(100, 20); // ratio 0.2
        s.record(100, 80); // ratio 0.8
        s.record(100, 50); // ratio 0.5
        let min = s.min_ratio().expect("min should be set");
        let max = s.max_ratio().expect("max should be set");
        assert!((min - 0.2).abs() < 1e-6);
        assert!((max - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_bytes_saved_no_underflow() {
        let mut s = CompressionStats::new();
        s.record(100, 200); // expansion: compressed > original
        assert_eq!(s.bytes_saved(), 0); // saturating_sub
    }

    #[test]
    fn test_reset() {
        let mut s = CompressionStats::new();
        s.record(1000, 500);
        s.reset();
        assert_eq!(s.entry_count(), 0);
        assert_eq!(s.ratio(), 0.0);
    }

    #[test]
    fn test_merge() {
        let mut s1 = CompressionStats::new();
        s1.record(1000, 400);

        let mut s2 = CompressionStats::new();
        s2.record(500, 300);

        s1.merge(&s2);
        assert_eq!(s1.entry_count(), 2);
        assert_eq!(s1.total_original_bytes(), 1500);
        assert_eq!(s1.total_compressed_bytes(), 700);
    }
}
