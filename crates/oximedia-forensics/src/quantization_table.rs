#![allow(dead_code)]
//! JPEG quantization table analysis for forensic examination.
//!
//! Every JPEG image is compressed using quantization tables that map DCT
//! coefficients to integer levels. These tables carry a fingerprint of the
//! encoding software and quality settings used. By analysing quantization
//! tables we can:
//!
//! - **Identify the JPEG encoder** (e.g. libjpeg, Photoshop, camera firmware)
//! - **Estimate quality factor** from the luminance table
//! - **Detect re-compression** by finding evidence of double quantization
//! - **Compare tables** across images to determine common origin
//!
//! This module works entirely on the 8x8 quantization matrix values and does
//! not decode or re-encode the image pixel data.

/// Standard JPEG luminance quantization table at quality 50 (ITU-T T.81 Annex K).
pub const STANDARD_LUMINANCE_Q50: [u16; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69,
    56, 14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81,
    104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];

/// Standard JPEG chrominance quantization table at quality 50.
pub const STANDARD_CHROMINANCE_Q50: [u16; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99,
    99, 47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
];

/// An 8x8 quantization table stored in row-major (raster scan) order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantTable {
    /// 64 quantization values in raster order
    pub values: [u16; 64],
}

impl QuantTable {
    /// Create a table from 64 values.
    pub fn new(values: [u16; 64]) -> Self {
        Self { values }
    }

    /// Create a table from a slice. Returns `None` if the slice length is not 64.
    pub fn from_slice(slice: &[u16]) -> Option<Self> {
        if slice.len() != 64 {
            return None;
        }
        let mut values = [0u16; 64];
        values.copy_from_slice(slice);
        Some(Self { values })
    }

    /// Get value at position `(row, col)` where `row, col` are in `[0, 8)`.
    pub fn get(&self, row: usize, col: usize) -> Option<u16> {
        if row < 8 && col < 8 {
            Some(self.values[row * 8 + col])
        } else {
            None
        }
    }

    /// Compute the sum of all 64 quantization values.
    #[allow(clippy::cast_precision_loss)]
    pub fn sum(&self) -> u64 {
        self.values.iter().map(|&v| u64::from(v)).sum()
    }

    /// Compute the mean quantization value.
    #[allow(clippy::cast_precision_loss)]
    pub fn mean(&self) -> f64 {
        self.sum() as f64 / 64.0
    }

    /// Compute the DC value (top-left).
    pub fn dc_value(&self) -> u16 {
        self.values[0]
    }

    /// Compute element-wise absolute difference from another table.
    pub fn abs_diff(&self, other: &Self) -> [u16; 64] {
        let mut diff = [0u16; 64];
        for i in 0..64 {
            diff[i] = self.values[i].abs_diff(other.values[i]);
        }
        diff
    }

    /// Compute a normalised similarity score compared to another table.
    /// Returns 1.0 for identical tables and approaches 0.0 for very different tables.
    #[allow(clippy::cast_precision_loss)]
    pub fn similarity(&self, other: &Self) -> f64 {
        let diff_sum: f64 = self
            .abs_diff(other)
            .iter()
            .map(|&d| f64::from(d))
            .sum();
        let max_sum: f64 = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(&a, &b)| f64::from(a.max(b)).max(1.0))
            .sum();
        if max_sum < 1e-12 {
            return 1.0;
        }
        1.0 - (diff_sum / max_sum)
    }
}

/// Estimate the JPEG quality factor (1..100) from a luminance quantization table
/// using the standard libjpeg scaling formula.
///
/// Returns `None` if the table does not match the standard scaling model.
#[allow(clippy::cast_precision_loss)]
pub fn estimate_quality_factor(table: &QuantTable) -> Option<u8> {
    // For each standard Q50 value, compute what quality would produce the observed value
    // quality > 50: q_val = ((2 * std - factor * std / 50) + 1) / 2  → factor = 200 - 2*q_val*100/std
    // quality <= 50: q_val = ((50 * std / factor) + 1) / 2           → factor = 50 * std / (2*q_val - 1)
    let mut quality_sum = 0.0f64;
    let mut count = 0u32;

    for i in 0..64 {
        let std_val = STANDARD_LUMINANCE_Q50[i] as f64;
        let obs_val = table.values[i] as f64;
        if std_val < 1.0 || obs_val < 1.0 {
            continue;
        }
        // Try quality > 50 formula: q = (200 - obs*100/std) / 2
        let q_high = (200.0 - obs_val * 100.0 / std_val) / 2.0;
        // Try quality <= 50 formula: q = 50*std / (2*obs - 1) / 100 * 50
        let q_low = 5000.0 / (2.0 * obs_val / std_val * 100.0 / 100.0 * 50.0).max(1.0);

        let q = if obs_val <= std_val {
            q_high
        } else {
            q_low.min(50.0)
        };
        if (1.0..=100.0).contains(&q) {
            quality_sum += q;
            count += 1;
        }
    }

    if count == 0 {
        return None;
    }
    let avg = quality_sum / f64::from(count);
    Some(avg.round().clamp(1.0, 100.0) as u8)
}

/// Generate a quantization table for a given quality factor (1..100) using
/// the standard libjpeg formula applied to the luminance base table.
#[allow(clippy::cast_precision_loss)]
pub fn generate_table_for_quality(quality: u8) -> QuantTable {
    let q = quality.max(1).min(100);
    let scale = if q < 50 {
        5000.0 / f64::from(q)
    } else {
        200.0 - 2.0 * f64::from(q)
    };

    let mut values = [0u16; 64];
    for i in 0..64 {
        let v = (f64::from(STANDARD_LUMINANCE_Q50[i]) * scale + 50.0) / 100.0;
        values[i] = v.round().max(1.0).min(255.0) as u16;
    }
    QuantTable { values }
}

/// Detect possible double JPEG compression by checking whether quantization values
/// are multiples of a coarser table.
///
/// Returns a score in `[0, 1]` where higher means more likely double-compressed.
#[allow(clippy::cast_precision_loss)]
pub fn detect_double_quantization(table: &QuantTable, coarse_quality: u8) -> f64 {
    let coarse = generate_table_for_quality(coarse_quality);
    let mut multiple_count = 0u32;

    for i in 0..64 {
        let q1 = coarse.values[i];
        let q2 = table.values[i];
        if q1 > 0 && q2 > 0 && q2 % q1 == 0 {
            multiple_count += 1;
        }
    }

    f64::from(multiple_count) / 64.0
}

/// Result of comparing two quantization tables.
#[derive(Debug, Clone)]
pub struct TableComparisonResult {
    /// Normalised similarity (0..1)
    pub similarity: f64,
    /// Number of identical entries
    pub identical_count: usize,
    /// Maximum absolute element-wise difference
    pub max_diff: u16,
    /// Mean absolute element-wise difference
    pub mean_diff: f64,
}

/// Compare two quantization tables in detail.
#[allow(clippy::cast_precision_loss)]
pub fn compare_tables(a: &QuantTable, b: &QuantTable) -> TableComparisonResult {
    let diff = a.abs_diff(b);
    let identical_count = diff.iter().filter(|&&d| d == 0).count();
    let max_diff = diff.iter().copied().max().unwrap_or(0);
    let mean_diff: f64 = diff.iter().map(|&d| f64::from(d)).sum::<f64>() / 64.0;
    let similarity = a.similarity(b);

    TableComparisonResult {
        similarity,
        identical_count,
        max_diff,
        mean_diff,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quant_table_new() {
        let t = QuantTable::new([1; 64]);
        assert_eq!(t.values[0], 1);
        assert_eq!(t.values[63], 1);
    }

    #[test]
    fn test_from_slice_valid() {
        let v: Vec<u16> = (1..=64).collect();
        let t = QuantTable::from_slice(&v).expect("t should be valid");
        assert_eq!(t.values[0], 1);
        assert_eq!(t.values[63], 64);
    }

    #[test]
    fn test_from_slice_invalid_length() {
        let v: Vec<u16> = vec![1, 2, 3];
        assert!(QuantTable::from_slice(&v).is_none());
    }

    #[test]
    fn test_get_in_bounds() {
        let t = QuantTable::new(STANDARD_LUMINANCE_Q50);
        assert_eq!(t.get(0, 0), Some(16));
        assert_eq!(t.get(7, 7), Some(99));
    }

    #[test]
    fn test_get_out_of_bounds() {
        let t = QuantTable::new(STANDARD_LUMINANCE_Q50);
        assert!(t.get(8, 0).is_none());
    }

    #[test]
    fn test_sum_and_mean() {
        let t = QuantTable::new([2; 64]);
        assert_eq!(t.sum(), 128);
        assert!((t.mean() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_dc_value() {
        let t = QuantTable::new(STANDARD_LUMINANCE_Q50);
        assert_eq!(t.dc_value(), 16);
    }

    #[test]
    fn test_abs_diff_identical() {
        let t = QuantTable::new([10; 64]);
        let diff = t.abs_diff(&t);
        assert!(diff.iter().all(|&d| d == 0));
    }

    #[test]
    fn test_similarity_identical() {
        let t = QuantTable::new(STANDARD_LUMINANCE_Q50);
        let sim = t.similarity(&t);
        assert!((sim - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_similarity_different() {
        let a = QuantTable::new([1; 64]);
        let b = QuantTable::new([100; 64]);
        let sim = a.similarity(&b);
        assert!(sim < 0.5);
    }

    #[test]
    fn test_generate_table_quality_50() {
        let t = generate_table_for_quality(50);
        // At quality 50, scale = 200 - 100 = 100, so values should match standard
        // (with rounding: (std * 100 + 50) / 100 ≈ std)
        for i in 0..64 {
            let expected = STANDARD_LUMINANCE_Q50[i];
            let diff = t.values[i].abs_diff(expected);
            assert!(diff <= 1, "index {i}: got {} expected {expected}", t.values[i]);
        }
    }

    #[test]
    fn test_generate_table_quality_100() {
        let t = generate_table_for_quality(100);
        // Quality 100 should produce all 1s
        for v in &t.values {
            assert_eq!(*v, 1);
        }
    }

    #[test]
    fn test_detect_double_quantization_same() {
        let t = generate_table_for_quality(50);
        let score = detect_double_quantization(&t, 50);
        // Same quality: all entries should be multiples (1x)
        assert!((score - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_compare_tables_identical() {
        let t = QuantTable::new(STANDARD_LUMINANCE_Q50);
        let r = compare_tables(&t, &t);
        assert_eq!(r.identical_count, 64);
        assert_eq!(r.max_diff, 0);
        assert!(r.mean_diff.abs() < 1e-12);
        assert!((r.similarity - 1.0).abs() < 1e-12);
    }
}
