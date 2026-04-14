//! Loudness history database with trend analysis and anomaly detection.
//!
//! [`LoudnessHistoryDb`] is an in-memory, serializable database that persists
//! loudness measurement history across analysis sessions.  It supports:
//!
//! - Append-only measurement storage with nanosecond timestamps.
//! - Linear-regression trend analysis (LUFS slope over time, R² goodness-of-fit).
//! - Rate-of-change detection (first derivative of the LUFS time-series).
//! - IQR-based anomaly flagging — measurements outside
//!   Q1 − 1.5·IQR … Q3 + 1.5·IQR are marked as statistical outliers.
//! - Serialisation to/from a compact binary format (little-endian plain-text
//!   newline-delimited CSV, pure Rust, no external crate required).

use crate::{NormalizeError, NormalizeResult};
use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// MeasurementRecord
// ─────────────────────────────────────────────────────────────────────────────

/// A single loudness record stored in the history database.
#[derive(Clone, Debug, PartialEq)]
pub struct MeasurementRecord {
    /// Monotonic timestamp in nanoseconds since an arbitrary epoch.
    pub timestamp_ns: u64,
    /// Integrated loudness in LUFS.
    pub integrated_lufs: f64,
    /// Loudness range (LRA) in LU.
    pub loudness_range_lu: f64,
    /// True-peak level in dBTP.
    pub true_peak_dbtp: f64,
    /// Optional free-form label (e.g. file name, programme title).
    pub label: String,
}

impl MeasurementRecord {
    /// Create a record with an explicit timestamp.
    pub fn new(
        timestamp_ns: u64,
        integrated_lufs: f64,
        loudness_range_lu: f64,
        true_peak_dbtp: f64,
        label: impl Into<String>,
    ) -> Self {
        Self {
            timestamp_ns,
            integrated_lufs,
            loudness_range_lu,
            true_peak_dbtp,
            label: label.into(),
        }
    }

    /// Serialise to a single CSV row (no trailing newline).
    ///
    /// Format: `timestamp_ns,integrated_lufs,loudness_range_lu,true_peak_dbtp,label`
    pub fn to_csv_row(&self) -> String {
        // Escape commas and newlines inside the label
        let safe_label = self.label.replace(',', ";").replace('\n', " ");
        format!(
            "{},{:.6},{:.6},{:.6},{}",
            self.timestamp_ns,
            self.integrated_lufs,
            self.loudness_range_lu,
            self.true_peak_dbtp,
            safe_label
        )
    }

    /// Parse from a single CSV row produced by [`to_csv_row`].
    pub fn from_csv_row(row: &str) -> NormalizeResult<Self> {
        let mut parts = row.splitn(5, ',');
        let ts_str = parts.next().ok_or_else(|| {
            NormalizeError::ProcessingError("CSV row missing timestamp".to_string())
        })?;
        let lufs_str = parts.next().ok_or_else(|| {
            NormalizeError::ProcessingError("CSV row missing integrated_lufs".to_string())
        })?;
        let lra_str = parts.next().ok_or_else(|| {
            NormalizeError::ProcessingError("CSV row missing loudness_range_lu".to_string())
        })?;
        let tp_str = parts.next().ok_or_else(|| {
            NormalizeError::ProcessingError("CSV row missing true_peak_dbtp".to_string())
        })?;
        let label = parts.next().unwrap_or("").to_string();

        let timestamp_ns = ts_str.trim().parse::<u64>().map_err(|e| {
            NormalizeError::ProcessingError(format!("invalid timestamp: {e}"))
        })?;
        let integrated_lufs = lufs_str.trim().parse::<f64>().map_err(|e| {
            NormalizeError::ProcessingError(format!("invalid integrated_lufs: {e}"))
        })?;
        let loudness_range_lu = lra_str.trim().parse::<f64>().map_err(|e| {
            NormalizeError::ProcessingError(format!("invalid loudness_range_lu: {e}"))
        })?;
        let true_peak_dbtp = tp_str.trim().parse::<f64>().map_err(|e| {
            NormalizeError::ProcessingError(format!("invalid true_peak_dbtp: {e}"))
        })?;

        Ok(Self {
            timestamp_ns,
            integrated_lufs,
            loudness_range_lu,
            true_peak_dbtp,
            label,
        })
    }
}

impl fmt::Display for MeasurementRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[t={} ns] {:.1} LUFS | LRA {:.1} LU | TP {:.1} dBTP | {}",
            self.timestamp_ns,
            self.integrated_lufs,
            self.loudness_range_lu,
            self.true_peak_dbtp,
            self.label
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Trend analysis
// ─────────────────────────────────────────────────────────────────────────────

/// Linear regression result over a time-series of loudness measurements.
#[derive(Clone, Debug)]
pub struct TrendAnalysis {
    /// Slope of the best-fit line in LUFS per nanosecond.
    pub slope_lufs_per_ns: f64,
    /// Intercept of the best-fit line.
    pub intercept_lufs: f64,
    /// Pearson R² goodness-of-fit (0 = no correlation, 1 = perfect).
    pub r_squared: f64,
    /// Mean integrated loudness over the analysed window.
    pub mean_lufs: f64,
    /// Standard deviation of integrated loudness.
    pub std_dev_lufs: f64,
    /// Number of measurements used.
    pub sample_count: usize,
    /// Rate of change over the window in LUFS per second.
    pub rate_lufs_per_second: f64,
}

impl TrendAnalysis {
    /// Returns `true` if loudness is trending upward (slope > 0).
    pub fn is_trending_louder(&self) -> bool {
        self.slope_lufs_per_ns > 0.0
    }

    /// Returns `true` if loudness is trending downward (slope < 0).
    pub fn is_trending_quieter(&self) -> bool {
        self.slope_lufs_per_ns < 0.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Anomaly
// ─────────────────────────────────────────────────────────────────────────────

/// An anomalous measurement detected via IQR analysis.
#[derive(Clone, Debug)]
pub struct Anomaly {
    /// Index into the database's record list.
    pub record_index: usize,
    /// The anomalous measurement.
    pub record: MeasurementRecord,
    /// The deviation from the median in LUFS.
    pub deviation_lu: f64,
    /// Direction of the anomaly.
    pub direction: AnomalyDirection,
}

/// Whether the anomalous measurement is louder or quieter than expected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnomalyDirection {
    /// Measurement is significantly louder than the distribution.
    TooLoud,
    /// Measurement is significantly quieter than the distribution.
    TooQuiet,
}

impl fmt::Display for AnomalyDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooLoud => write!(f, "too loud"),
            Self::TooQuiet => write!(f, "too quiet"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LoudnessHistoryDb
// ─────────────────────────────────────────────────────────────────────────────

/// In-memory loudness history database with trend analysis and anomaly detection.
///
/// Records are stored in append order. All analysis methods operate on a
/// configurable sliding window (by record count or timestamp range).
#[derive(Debug, Default)]
pub struct LoudnessHistoryDb {
    records: Vec<MeasurementRecord>,
    /// Maximum capacity; 0 means unlimited.
    max_capacity: usize,
}

impl LoudnessHistoryDb {
    /// Create an unlimited-capacity database.
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
            max_capacity: 0,
        }
    }

    /// Create a database capped at `max_capacity` records (oldest evicted first).
    pub fn with_capacity(max_capacity: usize) -> Self {
        Self {
            records: Vec::with_capacity(max_capacity),
            max_capacity,
        }
    }

    /// Append a measurement record.
    pub fn push(&mut self, record: MeasurementRecord) {
        if self.max_capacity > 0 && self.records.len() >= self.max_capacity {
            self.records.remove(0);
        }
        self.records.push(record);
    }

    /// Convenience: append a measurement with a given timestamp (ns).
    pub fn record(
        &mut self,
        timestamp_ns: u64,
        integrated_lufs: f64,
        lra_lu: f64,
        true_peak_dbtp: f64,
        label: impl Into<String>,
    ) {
        self.push(MeasurementRecord::new(
            timestamp_ns,
            integrated_lufs,
            lra_lu,
            true_peak_dbtp,
            label,
        ));
    }

    /// Number of stored records.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Returns `true` when the database is empty.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Clear all records.
    pub fn clear(&mut self) {
        self.records.clear();
    }

    /// Immutable slice of all records (oldest-first).
    pub fn records(&self) -> &[MeasurementRecord] {
        &self.records
    }

    /// Access a record by index.
    pub fn get(&self, index: usize) -> Option<&MeasurementRecord> {
        self.records.get(index)
    }

    // ─── window helpers ──────────────────────────────────────────────────────

    /// Obtain the most recent `n` records (or fewer if not enough exist).
    pub fn last_n(&self, n: usize) -> &[MeasurementRecord] {
        let start = self.records.len().saturating_sub(n);
        &self.records[start..]
    }

    /// Obtain records within a timestamp range [from_ns, to_ns].
    pub fn records_in_range(&self, from_ns: u64, to_ns: u64) -> Vec<&MeasurementRecord> {
        self.records
            .iter()
            .filter(|r| r.timestamp_ns >= from_ns && r.timestamp_ns <= to_ns)
            .collect()
    }

    // ─── trend analysis ──────────────────────────────────────────────────────

    /// Compute linear-regression trend across all stored records.
    ///
    /// Returns `None` if fewer than 2 records are available.
    pub fn trend(&self) -> Option<TrendAnalysis> {
        self.trend_for_slice(&self.records)
    }

    /// Compute trend across the most recent `n` records.
    pub fn trend_last_n(&self, n: usize) -> Option<TrendAnalysis> {
        self.trend_for_slice(self.last_n(n))
    }

    /// Compute trend for records within a timestamp range.
    pub fn trend_in_range(&self, from_ns: u64, to_ns: u64) -> Option<TrendAnalysis> {
        let window: Vec<&MeasurementRecord> = self.records_in_range(from_ns, to_ns);
        if window.len() < 2 {
            return None;
        }
        // Build owned slice reference
        let owned: Vec<MeasurementRecord> = window.iter().map(|r| (*r).clone()).collect();
        self.trend_for_slice(&owned)
    }

    fn trend_for_slice(&self, records: &[MeasurementRecord]) -> Option<TrendAnalysis> {
        if records.len() < 2 {
            return None;
        }

        let n = records.len() as f64;
        let lufs_values: Vec<f64> = records.iter().map(|r| r.integrated_lufs).collect();

        // Mean LUFS
        let mean_lufs = lufs_values.iter().sum::<f64>() / n;

        // Standard deviation
        let variance = lufs_values.iter().map(|&y| (y - mean_lufs).powi(2)).sum::<f64>() / n;
        let std_dev_lufs = variance.sqrt();

        // Use timestamp (ns) as x variable; normalise to avoid floating-point overflow
        let t0 = records[0].timestamp_ns;
        let ts_norm: Vec<f64> = records
            .iter()
            .map(|r| r.timestamp_ns.saturating_sub(t0) as f64)
            .collect();

        let mean_t = ts_norm.iter().sum::<f64>() / n;

        let sxx: f64 = ts_norm.iter().map(|&t| (t - mean_t).powi(2)).sum();
        let sxy: f64 = ts_norm
            .iter()
            .zip(lufs_values.iter())
            .map(|(&t, &y)| (t - mean_t) * (y - mean_lufs))
            .sum();

        let (slope, intercept) = if sxx.abs() < 1e-30 {
            (0.0_f64, mean_lufs)
        } else {
            let slope = sxy / sxx;
            let intercept = mean_lufs - slope * mean_t;
            (slope, intercept)
        };

        // R²
        let ss_res: f64 = ts_norm
            .iter()
            .zip(lufs_values.iter())
            .map(|(&t, &y)| {
                let predicted = slope * t + intercept;
                (y - predicted).powi(2)
            })
            .sum();
        let ss_tot: f64 = lufs_values.iter().map(|&y| (y - mean_lufs).powi(2)).sum();
        let r_squared = if ss_tot.abs() < 1e-30 {
            1.0
        } else {
            1.0 - ss_res / ss_tot
        };

        // Rate of change in LUFS/second
        let rate_lufs_per_second = slope * 1_000_000_000.0;

        Some(TrendAnalysis {
            slope_lufs_per_ns: slope,
            intercept_lufs: intercept,
            r_squared: r_squared.clamp(0.0, 1.0),
            mean_lufs,
            std_dev_lufs,
            sample_count: records.len(),
            rate_lufs_per_second,
        })
    }

    // ─── anomaly detection ───────────────────────────────────────────────────

    /// Detect anomalous records using Tukey's IQR fence method.
    ///
    /// A measurement is flagged as an outlier when its LUFS value lies outside
    /// [Q1 − `iqr_multiplier` × IQR, Q3 + `iqr_multiplier` × IQR].
    ///
    /// The standard value for `iqr_multiplier` is 1.5 (mild outliers) or 3.0
    /// (extreme outliers).  Returns an empty list when fewer than 4 records
    /// are available (IQR is meaningless with very few samples).
    pub fn detect_anomalies(&self, iqr_multiplier: f64) -> Vec<Anomaly> {
        self.detect_anomalies_for_slice(&self.records, iqr_multiplier)
    }

    /// Detect anomalies in the most recent `n` records.
    pub fn detect_anomalies_last_n(&self, n: usize, iqr_multiplier: f64) -> Vec<Anomaly> {
        let start = self.records.len().saturating_sub(n);
        self.detect_anomalies_for_slice(&self.records[start..], iqr_multiplier)
    }

    fn detect_anomalies_for_slice(
        &self,
        records: &[MeasurementRecord],
        iqr_multiplier: f64,
    ) -> Vec<Anomaly> {
        if records.len() < 4 {
            return Vec::new();
        }

        // Build sorted LUFS list for percentile computation
        let mut sorted: Vec<f64> = records.iter().map(|r| r.integrated_lufs).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let q1 = percentile_sorted(&sorted, 25.0);
        let q3 = percentile_sorted(&sorted, 75.0);
        let iqr = q3 - q1;
        let lower_fence = q1 - iqr_multiplier * iqr;
        let upper_fence = q3 + iqr_multiplier * iqr;

        let median = percentile_sorted(&sorted, 50.0);

        // Find the global offset so record_index is relative to self.records
        let slice_offset = if records.as_ptr() >= self.records.as_ptr() {
            let byte_offset = records.as_ptr() as usize - self.records.as_ptr() as usize;
            byte_offset / std::mem::size_of::<MeasurementRecord>()
        } else {
            0
        };

        records
            .iter()
            .enumerate()
            .filter_map(|(i, r)| {
                let lufs = r.integrated_lufs;
                if lufs < lower_fence || lufs > upper_fence {
                    let direction = if lufs > upper_fence {
                        AnomalyDirection::TooLoud
                    } else {
                        AnomalyDirection::TooQuiet
                    };
                    Some(Anomaly {
                        record_index: slice_offset + i,
                        record: r.clone(),
                        deviation_lu: lufs - median,
                        direction,
                    })
                } else {
                    None
                }
            })
            .collect()
    }

    // ─── serialisation ───────────────────────────────────────────────────────

    /// Serialise the entire database to a CSV byte vector.
    ///
    /// Format: one header row followed by one data row per record.
    pub fn to_csv_bytes(&self) -> Vec<u8> {
        let mut out = String::from("timestamp_ns,integrated_lufs,loudness_range_lu,true_peak_dbtp,label\n");
        for rec in &self.records {
            out.push_str(&rec.to_csv_row());
            out.push('\n');
        }
        out.into_bytes()
    }

    /// Deserialise a database from CSV bytes produced by [`to_csv_bytes`].
    ///
    /// Unknown or malformed rows are silently skipped (best-effort import).
    pub fn from_csv_bytes(data: &[u8]) -> NormalizeResult<Self> {
        let text = std::str::from_utf8(data).map_err(|e| {
            NormalizeError::ProcessingError(format!("CSV is not valid UTF-8: {e}"))
        })?;
        let mut db = Self::new();
        for line in text.lines().skip(1) {
            // skip header
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            match MeasurementRecord::from_csv_row(trimmed) {
                Ok(rec) => db.push(rec),
                Err(_) => { /* skip malformed rows */ }
            }
        }
        Ok(db)
    }

    // ─── statistics ──────────────────────────────────────────────────────────

    /// Arithmetic mean of integrated loudness across all records.
    pub fn mean_lufs(&self) -> Option<f64> {
        if self.records.is_empty() {
            return None;
        }
        let sum: f64 = self.records.iter().map(|r| r.integrated_lufs).sum();
        Some(sum / self.records.len() as f64)
    }

    /// Minimum integrated loudness.
    pub fn min_lufs(&self) -> Option<f64> {
        self.records
            .iter()
            .map(|r| r.integrated_lufs)
            .reduce(f64::min)
    }

    /// Maximum integrated loudness.
    pub fn max_lufs(&self) -> Option<f64> {
        self.records
            .iter()
            .map(|r| r.integrated_lufs)
            .reduce(f64::max)
    }

    /// Compliance rate: fraction of records within `tolerance_lu` of `target_lufs`.
    pub fn compliance_rate(&self, target_lufs: f64, tolerance_lu: f64) -> f64 {
        if self.records.is_empty() {
            return 0.0;
        }
        let compliant = self
            .records
            .iter()
            .filter(|r| (r.integrated_lufs - target_lufs).abs() <= tolerance_lu)
            .count();
        compliant as f64 / self.records.len() as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Linear interpolation percentile on a **sorted** slice.
fn percentile_sorted(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let index = pct / 100.0 * (sorted.len() - 1) as f64;
    let lo = index.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = index - lo as f64;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_db_ascending(n: usize) -> LoudnessHistoryDb {
        let mut db = LoudnessHistoryDb::new();
        for i in 0..n {
            db.record(
                i as u64 * 1_000_000_000, // 1 s apart
                -30.0 + i as f64,          // ascending LUFS
                8.0,
                -1.0,
                format!("track{i}"),
            );
        }
        db
    }

    #[test]
    fn test_db_starts_empty() {
        let db = LoudnessHistoryDb::new();
        assert!(db.is_empty());
        assert_eq!(db.len(), 0);
    }

    #[test]
    fn test_push_and_len() {
        let mut db = LoudnessHistoryDb::new();
        db.record(0, -23.0, 8.0, -1.0, "test");
        db.record(1_000_000_000, -22.0, 8.0, -1.0, "test2");
        assert_eq!(db.len(), 2);
    }

    #[test]
    fn test_capacity_evicts_oldest() {
        let mut db = LoudnessHistoryDb::with_capacity(3);
        for i in 0..5u64 {
            db.record(i * 1_000, -20.0 - i as f64, 8.0, -1.0, "");
        }
        assert_eq!(db.len(), 3);
        // The oldest (lufs -20) should have been evicted
        assert!(db.records().iter().all(|r| r.integrated_lufs < -20.0));
    }

    #[test]
    fn test_mean_lufs() {
        let mut db = LoudnessHistoryDb::new();
        db.record(0, -20.0, 8.0, -1.0, "a");
        db.record(1, -24.0, 8.0, -1.0, "b");
        let mean = db.mean_lufs().expect("should exist");
        assert!((mean - (-22.0)).abs() < 1e-9);
    }

    #[test]
    fn test_mean_lufs_empty_is_none() {
        let db = LoudnessHistoryDb::new();
        assert!(db.mean_lufs().is_none());
    }

    #[test]
    fn test_csv_round_trip() {
        let mut db = LoudnessHistoryDb::new();
        db.record(1_000_000, -23.0, 7.5, -1.5, "prog: 1");
        db.record(2_000_000, -14.0, 12.0, -0.5, "prog2");
        let csv = db.to_csv_bytes();
        let loaded = LoudnessHistoryDb::from_csv_bytes(&csv).expect("deserialise ok");
        assert_eq!(loaded.len(), 2);
        assert!((loaded.records()[0].integrated_lufs - (-23.0)).abs() < 1e-4);
        assert!((loaded.records()[1].integrated_lufs - (-14.0)).abs() < 1e-4);
    }

    #[test]
    fn test_trend_ascending() {
        let db = make_db_ascending(10);
        let trend = db.trend().expect("trend available");
        // Each second LUFS increases by 1 → positive slope
        assert!(trend.is_trending_louder());
        // Should be a perfect linear fit
        assert!(trend.r_squared > 0.99, "R² = {}", trend.r_squared);
    }

    #[test]
    fn test_trend_last_n() {
        let db = make_db_ascending(20);
        let trend = db.trend_last_n(5).expect("trend available");
        assert_eq!(trend.sample_count, 5);
    }

    #[test]
    fn test_trend_needs_at_least_two_records() {
        let mut db = LoudnessHistoryDb::new();
        db.record(0, -23.0, 8.0, -1.0, "single");
        assert!(db.trend().is_none());
    }

    #[test]
    fn test_anomaly_detection_finds_outlier() {
        let mut db = LoudnessHistoryDb::new();
        // 8 "normal" records clustered near -23 LUFS
        for i in 0..8u64 {
            db.record(i * 1_000, -23.0 + (i as f64 * 0.1 - 0.35), 8.0, -1.0, "");
        }
        // One extreme outlier
        db.record(9_000, -5.0, 8.0, 0.0, "outlier");

        let anomalies = db.detect_anomalies(1.5);
        assert!(!anomalies.is_empty(), "should detect at least one anomaly");
        assert_eq!(anomalies[0].direction, AnomalyDirection::TooLoud);
    }

    #[test]
    fn test_anomaly_detection_needs_four_records() {
        let mut db = LoudnessHistoryDb::new();
        for i in 0..3u64 {
            db.record(i, -23.0, 8.0, -1.0, "");
        }
        let anomalies = db.detect_anomalies(1.5);
        assert!(anomalies.is_empty());
    }

    #[test]
    fn test_compliance_rate() {
        let mut db = LoudnessHistoryDb::new();
        db.record(0, -23.0, 8.0, -1.0, "");
        db.record(1, -22.5, 8.0, -1.0, "");
        db.record(2, -18.0, 8.0, -1.0, ""); // non-compliant
        let rate = db.compliance_rate(-23.0, 1.0);
        assert!((rate - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_records_in_range() {
        let db = make_db_ascending(10);
        let window = db.records_in_range(3_000_000_000, 6_000_000_000);
        assert_eq!(window.len(), 4); // indices 3,4,5,6
    }

    #[test]
    fn test_measurement_record_csv_row() {
        let rec = MeasurementRecord::new(42_000_000, -23.0, 8.0, -1.0, "test label");
        let row = rec.to_csv_row();
        let parsed = MeasurementRecord::from_csv_row(&row).expect("parse ok");
        assert_eq!(parsed.timestamp_ns, 42_000_000);
        assert!((parsed.integrated_lufs - (-23.0)).abs() < 1e-4);
        assert_eq!(parsed.label, "test label");
    }

    #[test]
    fn test_min_max_lufs() {
        let mut db = LoudnessHistoryDb::new();
        db.record(0, -30.0, 8.0, -1.0, "");
        db.record(1, -20.0, 8.0, -1.0, "");
        db.record(2, -10.0, 8.0, -1.0, "");
        assert!((db.min_lufs().expect("min") - (-30.0)).abs() < 1e-9);
        assert!((db.max_lufs().expect("max") - (-10.0)).abs() < 1e-9);
    }
}
