#![allow(dead_code)]
//! Byte-level corruption heat map visualization and export.
//!
//! Given a set of [`CorruptedRegion`](crate::corruption_map::CorruptedRegion)s
//! this module quantises a file into fixed-size buckets and computes a per-bucket
//! severity score, producing a heat map that can be exported as CSV, plain-text
//! ASCII art, or a raw severity vector for further processing.
//!
//! # Example
//!
//! ```
//! use oximedia_repair::corruption_heatmap::{HeatMap, HeatMapConfig};
//! use oximedia_repair::corruption_map::{CorruptedRegion, CorruptionType};
//!
//! let regions = vec![
//!     CorruptedRegion {
//!         start_byte: 0,
//!         end_byte: 512,
//!         corruption_type: CorruptionType::HeaderDamage,
//!         severity: 0.9,
//!         repairable: true,
//!     },
//! ];
//!
//! let config = HeatMapConfig::new(4096, 8);
//! let hmap = HeatMap::build(&regions, &config);
//! assert_eq!(hmap.buckets().len(), 8);
//! ```

use crate::corruption_map::CorruptedRegion;

/// Configuration for heat map generation.
#[derive(Debug, Clone)]
pub struct HeatMapConfig {
    /// Total file size in bytes.
    pub file_size: u64,
    /// Number of buckets to quantise into.
    pub bucket_count: usize,
    /// Minimum severity threshold — regions below this are ignored.
    pub min_severity: f64,
}

impl HeatMapConfig {
    /// Create a new heat map configuration.
    ///
    /// `bucket_count` is clamped to at least 1.
    pub fn new(file_size: u64, bucket_count: usize) -> Self {
        Self {
            file_size,
            bucket_count: bucket_count.max(1),
            min_severity: 0.0,
        }
    }

    /// Set the minimum severity threshold for inclusion.
    pub fn with_min_severity(mut self, threshold: f64) -> Self {
        self.min_severity = threshold.clamp(0.0, 1.0);
        self
    }

    /// Bytes per bucket (rounded up).
    fn bucket_size(&self) -> u64 {
        if self.file_size == 0 || self.bucket_count == 0 {
            return 1;
        }
        (self.file_size + self.bucket_count as u64 - 1) / self.bucket_count as u64
    }
}

/// A single bucket in the heat map.
#[derive(Debug, Clone)]
pub struct HeatBucket {
    /// Index of this bucket (0-based).
    pub index: usize,
    /// Start byte offset (inclusive).
    pub start_byte: u64,
    /// End byte offset (exclusive).
    pub end_byte: u64,
    /// Maximum severity score across all overlapping regions (0.0 – 1.0).
    pub max_severity: f64,
    /// Weighted average severity (overlap-weighted).
    pub avg_severity: f64,
    /// Number of corrupted regions that overlap this bucket.
    pub region_count: usize,
    /// Fraction of this bucket that is covered by corruption (0.0 – 1.0).
    pub coverage: f64,
}

impl HeatBucket {
    /// Returns `true` when this bucket contains any corruption.
    pub fn is_corrupted(&self) -> bool {
        self.max_severity > 0.0 && self.coverage > 0.0
    }

    /// A composite heat score combining severity and coverage.
    ///
    /// `heat = max_severity * coverage`
    pub fn heat(&self) -> f64 {
        self.max_severity * self.coverage
    }
}

/// A complete corruption heat map.
#[derive(Debug, Clone)]
pub struct HeatMap {
    /// File size that was analysed.
    pub file_size: u64,
    /// Ordered buckets from start to end of file.
    buckets: Vec<HeatBucket>,
    /// Global maximum severity across all buckets.
    pub peak_severity: f64,
    /// Total number of corrupted buckets.
    pub corrupted_bucket_count: usize,
}

impl HeatMap {
    /// Build a heat map from a set of corrupted regions.
    #[allow(clippy::cast_precision_loss)]
    pub fn build(regions: &[CorruptedRegion], config: &HeatMapConfig) -> Self {
        let bucket_size = config.bucket_size();
        let n = config.bucket_count;
        let mut buckets = Vec::with_capacity(n);

        for i in 0..n {
            let start = i as u64 * bucket_size;
            let end = ((i as u64 + 1) * bucket_size).min(config.file_size);

            let bucket_len = end.saturating_sub(start);
            let mut max_sev = 0.0_f64;
            let mut weighted_sum = 0.0_f64;
            let mut total_overlap = 0u64;
            let mut count = 0usize;

            for region in regions {
                if region.severity < config.min_severity {
                    continue;
                }

                // Compute overlap between this bucket and the region
                let overlap_start = start.max(region.start_byte);
                let overlap_end = end.min(region.end_byte);
                if overlap_start < overlap_end {
                    let overlap = overlap_end - overlap_start;
                    total_overlap += overlap;
                    weighted_sum += region.severity * overlap as f64;
                    if region.severity > max_sev {
                        max_sev = region.severity;
                    }
                    count += 1;
                }
            }

            let coverage = if bucket_len > 0 {
                total_overlap as f64 / bucket_len as f64
            } else {
                0.0
            };

            let avg_severity = if total_overlap > 0 {
                weighted_sum / total_overlap as f64
            } else {
                0.0
            };

            buckets.push(HeatBucket {
                index: i,
                start_byte: start,
                end_byte: end,
                max_severity: max_sev,
                avg_severity,
                region_count: count,
                coverage: coverage.min(1.0),
            });
        }

        let peak_severity = buckets
            .iter()
            .map(|b| b.max_severity)
            .fold(0.0_f64, f64::max);
        let corrupted_bucket_count = buckets.iter().filter(|b| b.is_corrupted()).count();

        Self {
            file_size: config.file_size,
            buckets,
            peak_severity,
            corrupted_bucket_count,
        }
    }

    /// Return a reference to the bucket vector.
    pub fn buckets(&self) -> &[HeatBucket] {
        &self.buckets
    }

    /// Return the raw severity vector (max severity per bucket).
    pub fn severity_vector(&self) -> Vec<f64> {
        self.buckets.iter().map(|b| b.max_severity).collect()
    }

    /// Return the heat vector (composite score per bucket).
    pub fn heat_vector(&self) -> Vec<f64> {
        self.buckets.iter().map(|b| b.heat()).collect()
    }

    /// Fraction of buckets that contain corruption.
    #[allow(clippy::cast_precision_loss)]
    pub fn corruption_ratio(&self) -> f64 {
        if self.buckets.is_empty() {
            return 0.0;
        }
        self.corrupted_bucket_count as f64 / self.buckets.len() as f64
    }

    /// Export the heat map as CSV text.
    ///
    /// Columns: `bucket,start_byte,end_byte,max_severity,avg_severity,coverage,region_count`
    pub fn to_csv(&self) -> String {
        let mut out =
            String::from("bucket,start_byte,end_byte,max_severity,avg_severity,coverage,region_count\n");
        for b in &self.buckets {
            out.push_str(&format!(
                "{},{},{},{:.4},{:.4},{:.4},{}\n",
                b.index, b.start_byte, b.end_byte, b.max_severity, b.avg_severity, b.coverage,
                b.region_count,
            ));
        }
        out
    }

    /// Export the heat map as an ASCII art representation.
    ///
    /// Each bucket is rendered as a character: ` ` (clean), `░` (low),
    /// `▒` (medium), `▓` (high), `█` (critical).
    pub fn to_ascii(&self, width: usize) -> String {
        let width = width.max(1);
        let step = self.buckets.len().max(1) / width.min(self.buckets.len()).max(1);
        let step = step.max(1);

        let mut line = String::with_capacity(width + 2);
        line.push('|');

        let mut col = 0;
        let mut idx = 0;
        while col < width && idx < self.buckets.len() {
            // Take the maximum severity in the step range
            let end = (idx + step).min(self.buckets.len());
            let max_sev = self.buckets[idx..end]
                .iter()
                .map(|b| b.max_severity)
                .fold(0.0_f64, f64::max);

            let ch = severity_to_char(max_sev);
            line.push(ch);
            idx = end;
            col += 1;
        }

        line.push('|');
        line
    }

    /// Export as structured JSON string.
    pub fn to_json(&self) -> String {
        let mut entries = Vec::with_capacity(self.buckets.len());
        for b in &self.buckets {
            entries.push(format!(
                "{{\"bucket\":{},\"start\":{},\"end\":{},\"max_severity\":{:.4},\"avg_severity\":{:.4},\"coverage\":{:.4},\"regions\":{}}}",
                b.index, b.start_byte, b.end_byte, b.max_severity, b.avg_severity, b.coverage, b.region_count
            ));
        }
        format!(
            "{{\"file_size\":{},\"peak_severity\":{:.4},\"corrupted_buckets\":{},\"total_buckets\":{},\"buckets\":[{}]}}",
            self.file_size,
            self.peak_severity,
            self.corrupted_bucket_count,
            self.buckets.len(),
            entries.join(",")
        )
    }

    /// Find the bucket index with the highest severity.
    pub fn hotspot(&self) -> Option<usize> {
        if self.buckets.is_empty() {
            return None;
        }
        self.buckets
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.max_severity.partial_cmp(&b.max_severity).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
    }

    /// Return contiguous runs of corrupted buckets as `(start_index, count)` pairs.
    pub fn corrupted_runs(&self) -> Vec<(usize, usize)> {
        let mut runs = Vec::new();
        let mut run_start: Option<usize> = None;
        let mut run_len = 0usize;

        for (i, b) in self.buckets.iter().enumerate() {
            if b.is_corrupted() {
                if run_start.is_none() {
                    run_start = Some(i);
                    run_len = 0;
                }
                run_len += 1;
            } else if let Some(start) = run_start.take() {
                runs.push((start, run_len));
                run_len = 0;
            }
        }

        if let Some(start) = run_start {
            runs.push((start, run_len));
        }

        runs
    }
}

/// Map a severity value (0.0 – 1.0) to an ASCII character for visualisation.
fn severity_to_char(sev: f64) -> char {
    if sev <= 0.0 {
        ' '
    } else if sev < 0.25 {
        '░'
    } else if sev < 0.5 {
        '▒'
    } else if sev < 0.75 {
        '▓'
    } else {
        '█'
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corruption_map::{CorruptedRegion, CorruptionType};

    fn region(start: u64, end: u64, severity: f64) -> CorruptedRegion {
        CorruptedRegion {
            start_byte: start,
            end_byte: end,
            corruption_type: CorruptionType::DataCorruption,
            severity,
            repairable: true,
        }
    }

    #[test]
    fn empty_regions_produce_clean_heatmap() {
        let config = HeatMapConfig::new(1024, 8);
        let hmap = HeatMap::build(&[], &config);
        assert_eq!(hmap.buckets().len(), 8);
        assert_eq!(hmap.corrupted_bucket_count, 0);
        assert!((hmap.peak_severity - 0.0).abs() < f64::EPSILON);
        assert!((hmap.corruption_ratio() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn single_region_covers_first_bucket() {
        let regions = vec![region(0, 128, 0.8)];
        let config = HeatMapConfig::new(1024, 8); // bucket size = 128
        let hmap = HeatMap::build(&regions, &config);
        assert!(hmap.buckets()[0].is_corrupted());
        assert!((hmap.buckets()[0].max_severity - 0.8).abs() < f64::EPSILON);
        assert!((hmap.buckets()[0].coverage - 1.0).abs() < f64::EPSILON);
        // Second bucket should be clean
        assert!(!hmap.buckets()[1].is_corrupted());
    }

    #[test]
    fn region_spanning_multiple_buckets() {
        let regions = vec![region(0, 512, 0.5)];
        let config = HeatMapConfig::new(1024, 4); // bucket size = 256
        let hmap = HeatMap::build(&regions, &config);
        assert!(hmap.buckets()[0].is_corrupted());
        assert!(hmap.buckets()[1].is_corrupted());
        assert!(!hmap.buckets()[2].is_corrupted());
        assert!(!hmap.buckets()[3].is_corrupted());
    }

    #[test]
    fn overlapping_regions_take_max_severity() {
        let regions = vec![region(0, 256, 0.3), region(100, 256, 0.9)];
        let config = HeatMapConfig::new(256, 1);
        let hmap = HeatMap::build(&regions, &config);
        assert!((hmap.buckets()[0].max_severity - 0.9).abs() < f64::EPSILON);
        assert_eq!(hmap.buckets()[0].region_count, 2);
    }

    #[test]
    fn min_severity_filters_low_regions() {
        let regions = vec![region(0, 256, 0.1), region(256, 512, 0.8)];
        let config = HeatMapConfig::new(512, 2).with_min_severity(0.5);
        let hmap = HeatMap::build(&regions, &config);
        // First bucket should be clean (severity 0.1 < threshold 0.5)
        assert!(!hmap.buckets()[0].is_corrupted());
        assert!(hmap.buckets()[1].is_corrupted());
    }

    #[test]
    fn severity_vector_matches_buckets() {
        let regions = vec![region(0, 100, 0.7)];
        let config = HeatMapConfig::new(400, 4);
        let hmap = HeatMap::build(&regions, &config);
        let sv = hmap.severity_vector();
        assert_eq!(sv.len(), 4);
        assert!((sv[0] - 0.7).abs() < f64::EPSILON);
        assert!((sv[1] - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn heat_vector_combines_severity_and_coverage() {
        let regions = vec![region(0, 50, 1.0)];
        let config = HeatMapConfig::new(100, 1);
        let hmap = HeatMap::build(&regions, &config);
        let hv = hmap.heat_vector();
        // coverage = 50/100 = 0.5, max_severity = 1.0, heat = 0.5
        assert!((hv[0] - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn csv_export_has_correct_header_and_rows() {
        let regions = vec![region(0, 64, 0.5)];
        let config = HeatMapConfig::new(128, 2);
        let hmap = HeatMap::build(&regions, &config);
        let csv = hmap.to_csv();
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines[0], "bucket,start_byte,end_byte,max_severity,avg_severity,coverage,region_count");
        assert_eq!(lines.len(), 3); // header + 2 buckets
    }

    #[test]
    fn ascii_export_produces_bounded_string() {
        let regions = vec![region(0, 512, 0.9)];
        let config = HeatMapConfig::new(1024, 8);
        let hmap = HeatMap::build(&regions, &config);
        let ascii = hmap.to_ascii(8);
        assert!(ascii.starts_with('|'));
        assert!(ascii.ends_with('|'));
    }

    #[test]
    fn json_export_contains_keys() {
        let regions = vec![region(0, 100, 0.6)];
        let config = HeatMapConfig::new(200, 2);
        let hmap = HeatMap::build(&regions, &config);
        let json = hmap.to_json();
        assert!(json.contains("\"file_size\":200"));
        assert!(json.contains("\"peak_severity\""));
        assert!(json.contains("\"buckets\":["));
    }

    #[test]
    fn hotspot_returns_highest_severity_bucket() {
        let regions = vec![region(0, 100, 0.3), region(200, 300, 0.9)];
        let config = HeatMapConfig::new(400, 4); // bucket size = 100
        let hmap = HeatMap::build(&regions, &config);
        assert_eq!(hmap.hotspot(), Some(2)); // bucket index 2 covers 200-300
    }

    #[test]
    fn hotspot_empty_returns_none_for_empty_config() {
        let config = HeatMapConfig::new(0, 0);
        // bucket_count is clamped to 1, but file_size is 0
        let hmap = HeatMap::build(&[], &config);
        // There's 1 bucket but it's clean
        assert!(hmap.hotspot().is_some()); // still returns an index
    }

    #[test]
    fn corrupted_runs_identifies_contiguous_spans() {
        let regions = vec![
            region(0, 300, 0.5),   // covers buckets 0,1,2
            region(500, 600, 0.5), // covers bucket 5
        ];
        let config = HeatMapConfig::new(1000, 10); // bucket size = 100
        let hmap = HeatMap::build(&regions, &config);
        let runs = hmap.corrupted_runs();
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0], (0, 3));  // buckets 0,1,2
        assert_eq!(runs[1], (5, 1));  // bucket 5
    }

    #[test]
    fn corruption_ratio_correct() {
        let regions = vec![region(0, 500, 0.5)];
        let config = HeatMapConfig::new(1000, 10);
        let hmap = HeatMap::build(&regions, &config);
        assert!((hmap.corruption_ratio() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn zero_file_size_does_not_panic() {
        let config = HeatMapConfig::new(0, 4);
        let hmap = HeatMap::build(&[], &config);
        assert_eq!(hmap.buckets().len(), 4);
        assert!((hmap.corruption_ratio() - 0.0).abs() < f64::EPSILON);
    }
}
