#![allow(dead_code)]
//! Historical benchmark database for persisting and querying benchmark results
//! across multiple runs, builds, and time periods.
//!
//! # Overview
//!
//! The `historical_db` module provides:
//!
//! - **`HistoricalEntry`** — a single benchmark observation with metadata
//!   (timestamp, build id, codec, metric, value).
//! - **`HistoricalDb`** — an in-memory store of entries backed by a JSON file
//!   on disk, supporting append, query, and trend analysis.
//! - **`TrendAnalysis`** — rolling statistics and linear-regression trend over
//!   a time-ordered metric series.
//! - **`BenchHistory`** — a view of all observations for a specific
//!   `(codec, metric)` pair, with helpers for plotting-friendly output.
//!
//! The database file is newline-delimited JSON (`*.ndjson`) so that individual
//! entries can be appended without rewriting the entire file.

use crate::{BenchError, BenchResult};
use oximedia_core::types::CodecId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Entry
// ---------------------------------------------------------------------------

/// A single benchmark observation stored in the historical database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalEntry {
    /// Unix timestamp (seconds since epoch) when this entry was recorded.
    pub timestamp_secs: u64,
    /// Human-readable ISO-8601 timestamp.
    pub timestamp_str: String,
    /// Build / commit identifier (e.g. a short Git SHA or CI run ID).
    pub build_id: String,
    /// Codec this entry belongs to.
    pub codec: CodecId,
    /// Metric name (e.g. `"encoding_fps"`, `"psnr_db"`, `"file_size_bytes"`).
    pub metric: String,
    /// Measured value.
    pub value: f64,
    /// Optional test sequence name.
    pub sequence: Option<String>,
    /// Optional preset / configuration label.
    pub preset: Option<String>,
    /// Arbitrary extra context (key → value).
    pub extra: HashMap<String, String>,
}

impl HistoricalEntry {
    /// Create a minimal entry with the current wall-clock time.
    #[must_use]
    pub fn new(
        build_id: impl Into<String>,
        codec: CodecId,
        metric: impl Into<String>,
        value: f64,
    ) -> Self {
        let ts = current_unix_secs();
        Self {
            timestamp_secs: ts,
            timestamp_str: unix_secs_to_iso8601(ts),
            build_id: build_id.into(),
            codec,
            metric: metric.into(),
            value,
            sequence: None,
            preset: None,
            extra: HashMap::new(),
        }
    }

    /// Builder: attach a sequence name.
    #[must_use]
    pub fn with_sequence(mut self, seq: impl Into<String>) -> Self {
        self.sequence = Some(seq.into());
        self
    }

    /// Builder: attach a preset.
    #[must_use]
    pub fn with_preset(mut self, preset: impl Into<String>) -> Self {
        self.preset = Some(preset.into());
        self
    }

    /// Builder: add extra context.
    #[must_use]
    pub fn with_extra(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra.insert(key.into(), value.into());
        self
    }

    /// Override the timestamp (useful for testing or replaying data).
    #[must_use]
    pub fn with_timestamp(mut self, secs: u64) -> Self {
        self.timestamp_secs = secs;
        self.timestamp_str = unix_secs_to_iso8601(secs);
        self
    }
}

// ---------------------------------------------------------------------------
// Query / filter
// ---------------------------------------------------------------------------

/// A query predicate for filtering historical entries.
#[derive(Debug, Clone, Default)]
pub struct HistoricalQuery {
    /// If set, only return entries for this codec.
    pub codec: Option<CodecId>,
    /// If set, only return entries for this metric name.
    pub metric: Option<String>,
    /// If set, only return entries with `timestamp_secs >= from`.
    pub from_secs: Option<u64>,
    /// If set, only return entries with `timestamp_secs <= to`.
    pub to_secs: Option<u64>,
    /// If set, only return entries for this build ID.
    pub build_id: Option<String>,
    /// If set, only return entries for this sequence name.
    pub sequence: Option<String>,
    /// If set, only return entries for this preset.
    pub preset: Option<String>,
    /// Maximum number of entries to return (0 = unlimited).
    pub limit: usize,
}

impl HistoricalQuery {
    /// Create a new empty query (matches everything).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Filter by codec.
    #[must_use]
    pub fn with_codec(mut self, codec: CodecId) -> Self {
        self.codec = Some(codec);
        self
    }

    /// Filter by metric name.
    #[must_use]
    pub fn with_metric(mut self, metric: impl Into<String>) -> Self {
        self.metric = Some(metric.into());
        self
    }

    /// Filter by timestamp range (inclusive).
    #[must_use]
    pub fn with_time_range(mut self, from_secs: u64, to_secs: u64) -> Self {
        self.from_secs = Some(from_secs);
        self.to_secs = Some(to_secs);
        self
    }

    /// Filter by build ID.
    #[must_use]
    pub fn with_build_id(mut self, id: impl Into<String>) -> Self {
        self.build_id = Some(id.into());
        self
    }

    /// Filter by sequence name.
    #[must_use]
    pub fn with_sequence(mut self, seq: impl Into<String>) -> Self {
        self.sequence = Some(seq.into());
        self
    }

    /// Filter by preset.
    #[must_use]
    pub fn with_preset(mut self, preset: impl Into<String>) -> Self {
        self.preset = Some(preset.into());
        self
    }

    /// Limit the number of results.
    #[must_use]
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Check whether an entry matches this query.
    #[must_use]
    pub fn matches(&self, entry: &HistoricalEntry) -> bool {
        if let Some(codec) = self.codec {
            if entry.codec != codec {
                return false;
            }
        }
        if let Some(ref metric) = self.metric {
            if &entry.metric != metric {
                return false;
            }
        }
        if let Some(from) = self.from_secs {
            if entry.timestamp_secs < from {
                return false;
            }
        }
        if let Some(to) = self.to_secs {
            if entry.timestamp_secs > to {
                return false;
            }
        }
        if let Some(ref build_id) = self.build_id {
            if &entry.build_id != build_id {
                return false;
            }
        }
        if let Some(ref seq) = self.sequence {
            if entry.sequence.as_deref() != Some(seq.as_str()) {
                return false;
            }
        }
        if let Some(ref preset) = self.preset {
            if entry.preset.as_deref() != Some(preset.as_str()) {
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Historical database
// ---------------------------------------------------------------------------

/// Configuration for the historical database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalDbConfig {
    /// Path to the NDJSON file on disk.
    pub db_path: PathBuf,
    /// Maximum number of entries to keep in memory (0 = unlimited).
    pub max_in_memory: usize,
    /// Whether to flush each append to disk immediately.
    pub flush_on_append: bool,
}

impl Default for HistoricalDbConfig {
    fn default() -> Self {
        Self {
            db_path: PathBuf::from("./bench_history.ndjson"),
            max_in_memory: 0,
            flush_on_append: true,
        }
    }
}

impl HistoricalDbConfig {
    /// Create a new config.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set the database path.
    #[must_use]
    pub fn with_db_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.db_path = path.into();
        self
    }

    /// Builder: set the maximum in-memory entry count.
    #[must_use]
    pub fn with_max_in_memory(mut self, max: usize) -> Self {
        self.max_in_memory = max;
        self
    }

    /// Builder: set whether to flush on every append.
    #[must_use]
    pub fn with_flush_on_append(mut self, flush: bool) -> Self {
        self.flush_on_append = flush;
        self
    }
}

/// In-memory historical benchmark database backed by an NDJSON file.
pub struct HistoricalDb {
    config: HistoricalDbConfig,
    /// All entries currently held in memory, sorted by timestamp ascending.
    entries: Vec<HistoricalEntry>,
    /// Whether the in-memory state is dirty (has unsaved changes).
    dirty: bool,
}

impl HistoricalDb {
    /// Create a new empty database with the given configuration.
    #[must_use]
    pub fn new(config: HistoricalDbConfig) -> Self {
        Self {
            config,
            entries: Vec::new(),
            dirty: false,
        }
    }

    /// Open (or create) the database from the path specified in the config.
    ///
    /// If the file does not exist an empty database is returned.
    ///
    /// # Errors
    ///
    /// Returns [`BenchError::Io`] or [`BenchError::Serialization`] if loading
    /// fails.
    pub fn open(config: HistoricalDbConfig) -> BenchResult<Self> {
        let mut db = Self::new(config);
        db.load()?;
        Ok(db)
    }

    /// Load entries from disk into memory.
    ///
    /// # Errors
    ///
    /// Returns an error if the file exists but cannot be parsed.
    pub fn load(&mut self) -> BenchResult<()> {
        let path = &self.config.db_path;
        if !path.exists() {
            return Ok(());
        }
        let content = std::fs::read_to_string(path)?;
        let mut entries = Vec::new();
        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let entry: HistoricalEntry =
                serde_json::from_str(trimmed).map_err(BenchError::Serialization)?;
            entries.push(entry);
        }
        entries.sort_by_key(|e| e.timestamp_secs);
        self.entries = entries;
        self.dirty = false;
        Ok(())
    }

    /// Flush all in-memory entries to disk (overwrites the existing file).
    ///
    /// # Errors
    ///
    /// Returns [`BenchError::Io`] or [`BenchError::Serialization`] on failure.
    pub fn flush(&mut self) -> BenchResult<()> {
        if let Some(parent) = self.config.db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut lines = Vec::with_capacity(self.entries.len());
        for entry in &self.entries {
            lines.push(serde_json::to_string(entry).map_err(BenchError::Serialization)?);
        }
        let content = lines.join("\n");
        std::fs::write(&self.config.db_path, content.as_bytes())?;
        self.dirty = false;
        Ok(())
    }

    /// Append a single entry to the database.
    ///
    /// If `flush_on_append` is set in the config, the entry is written to disk
    /// immediately (appended to the NDJSON file).
    ///
    /// # Errors
    ///
    /// Returns an error if flushing fails.
    pub fn append(&mut self, entry: HistoricalEntry) -> BenchResult<()> {
        // Maintain max_in_memory cap
        if self.config.max_in_memory > 0 && self.entries.len() >= self.config.max_in_memory {
            let overflow = self.entries.len() - self.config.max_in_memory + 1;
            self.entries.drain(..overflow);
        }
        // Keep sorted order
        let pos = self
            .entries
            .partition_point(|e| e.timestamp_secs <= entry.timestamp_secs);
        self.entries.insert(pos, entry.clone());
        self.dirty = true;

        if self.config.flush_on_append {
            // Append single line to file
            self.append_line_to_file(&entry)?;
            self.dirty = false;
        }
        Ok(())
    }

    /// Append multiple entries at once.
    ///
    /// # Errors
    ///
    /// Returns an error if any append fails.
    pub fn append_many(&mut self, entries: Vec<HistoricalEntry>) -> BenchResult<()> {
        for entry in entries {
            self.append(entry)?;
        }
        Ok(())
    }

    /// Query entries matching the given predicate, sorted by timestamp ascending.
    #[must_use]
    pub fn query(&self, q: &HistoricalQuery) -> Vec<&HistoricalEntry> {
        let mut results: Vec<&HistoricalEntry> =
            self.entries.iter().filter(|e| q.matches(e)).collect();
        if q.limit > 0 && results.len() > q.limit {
            results.truncate(q.limit);
        }
        results
    }

    /// Return the most recent entry matching the query.
    #[must_use]
    pub fn latest(&self, q: &HistoricalQuery) -> Option<&HistoricalEntry> {
        self.entries.iter().rev().find(|e| q.matches(e))
    }

    /// Total number of entries currently in memory.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the database is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return all distinct codec IDs present in the database.
    #[must_use]
    pub fn distinct_codecs(&self) -> Vec<CodecId> {
        let mut seen = std::collections::HashSet::new();
        let mut codecs = Vec::new();
        for entry in &self.entries {
            if seen.insert(entry.codec) {
                codecs.push(entry.codec);
            }
        }
        codecs
    }

    /// Return all distinct metric names present in the database.
    #[must_use]
    pub fn distinct_metrics(&self) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        let mut metrics = Vec::new();
        for entry in &self.entries {
            if seen.insert(entry.metric.clone()) {
                metrics.push(entry.metric.clone());
            }
        }
        metrics
    }

    /// Get a history view for a specific `(codec, metric)` pair.
    #[must_use]
    pub fn history(&self, codec: CodecId, metric: &str) -> BenchHistory {
        let q = HistoricalQuery::new().with_codec(codec).with_metric(metric);
        let entries: Vec<HistoricalEntry> = self.query(&q).into_iter().cloned().collect();
        BenchHistory {
            codec,
            metric: metric.to_string(),
            entries,
        }
    }

    /// Delete entries matching a query.  Returns the number of deleted entries.
    pub fn delete(&mut self, q: &HistoricalQuery) -> usize {
        let before = self.entries.len();
        self.entries.retain(|e| !q.matches(e));
        let deleted = before - self.entries.len();
        if deleted > 0 {
            self.dirty = true;
        }
        deleted
    }

    // ---- private helpers ---------------------------------------------------

    fn append_line_to_file(&self, entry: &HistoricalEntry) -> BenchResult<()> {
        use std::io::Write as IoWrite;
        if let Some(parent) = self.config.db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let line = serde_json::to_string(entry).map_err(BenchError::Serialization)?;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.config.db_path)?;
        writeln!(file, "{line}")?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// BenchHistory — a view over entries for one (codec, metric) pair
// ---------------------------------------------------------------------------

/// A time-ordered view of all observations for a specific `(codec, metric)` pair.
#[derive(Debug, Clone)]
pub struct BenchHistory {
    /// Codec.
    pub codec: CodecId,
    /// Metric name.
    pub metric: String,
    /// Time-ordered entries (ascending by timestamp).
    pub entries: Vec<HistoricalEntry>,
}

impl BenchHistory {
    /// Number of observations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether there are no observations.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// All values in time order.
    #[must_use]
    pub fn values(&self) -> Vec<f64> {
        self.entries.iter().map(|e| e.value).collect()
    }

    /// All `(unix_timestamp_secs, value)` pairs in time order.
    #[must_use]
    pub fn time_series(&self) -> Vec<(u64, f64)> {
        self.entries
            .iter()
            .map(|e| (e.timestamp_secs, e.value))
            .collect()
    }

    /// Compute trend analysis for this history.
    ///
    /// Returns `None` when fewer than two data points are available.
    #[must_use]
    pub fn trend(&self) -> Option<TrendAnalysis> {
        TrendAnalysis::compute(self)
    }

    /// Slice the last `n` entries.
    #[must_use]
    pub fn last_n(&self, n: usize) -> Self {
        let start = self.entries.len().saturating_sub(n);
        Self {
            codec: self.codec,
            metric: self.metric.clone(),
            entries: self.entries[start..].to_vec(),
        }
    }

    /// Find the entry with the best (highest or lowest) value.
    ///
    /// `higher_is_better` controls the direction.
    #[must_use]
    pub fn best_entry(&self, higher_is_better: bool) -> Option<&HistoricalEntry> {
        if self.entries.is_empty() {
            return None;
        }
        if higher_is_better {
            self.entries.iter().max_by(|a, b| {
                a.value
                    .partial_cmp(&b.value)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        } else {
            self.entries.iter().min_by(|a, b| {
                a.value
                    .partial_cmp(&b.value)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Trend analysis
// ---------------------------------------------------------------------------

/// Rolling statistics and linear-regression trend for a metric time series.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    /// Metric name.
    pub metric: String,
    /// Number of data points.
    pub n: usize,
    /// Arithmetic mean.
    pub mean: f64,
    /// Standard deviation.
    pub std_dev: f64,
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
    /// Linear regression slope (units per second).
    pub slope: f64,
    /// Linear regression intercept.
    pub intercept: f64,
    /// Pearson correlation coefficient (−1 to +1).
    pub correlation: f64,
    /// Predicted value at the latest timestamp.
    pub predicted_latest: f64,
    /// Percent change predicted over the observation window.
    pub trend_pct: f64,
}

impl TrendAnalysis {
    /// Compute a trend analysis from a [`BenchHistory`].
    ///
    /// Returns `None` when fewer than two entries are available.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn compute(history: &BenchHistory) -> Option<Self> {
        let n = history.entries.len();
        if n < 2 {
            return None;
        }
        let ts_min = history.entries[0].timestamp_secs as f64;
        let xs: Vec<f64> = history
            .entries
            .iter()
            .map(|e| e.timestamp_secs as f64 - ts_min)
            .collect();
        let ys: Vec<f64> = history.entries.iter().map(|e| e.value).collect();

        let mean_x = xs.iter().sum::<f64>() / n as f64;
        let mean_y = ys.iter().sum::<f64>() / n as f64;

        let cov_xy: f64 = xs
            .iter()
            .zip(ys.iter())
            .map(|(&x, &y)| (x - mean_x) * (y - mean_y))
            .sum::<f64>()
            / n as f64;
        let var_x: f64 = xs.iter().map(|&x| (x - mean_x).powi(2)).sum::<f64>() / n as f64;
        let var_y: f64 = ys.iter().map(|&y| (y - mean_y).powi(2)).sum::<f64>() / n as f64;

        let slope = if var_x > 0.0 { cov_xy / var_x } else { 0.0 };
        let intercept = mean_y - slope * mean_x;

        let std_dev_x = var_x.sqrt();
        let std_dev_y = var_y.sqrt();
        let correlation = if std_dev_x > 0.0 && std_dev_y > 0.0 {
            cov_xy / (std_dev_x * std_dev_y)
        } else {
            0.0
        };

        let x_latest = xs.last().copied().unwrap_or(0.0);
        let x_first = 0.0_f64;
        let predicted_latest = slope * x_latest + intercept;
        let predicted_first = slope * x_first + intercept;
        let trend_pct = if predicted_first.abs() > 0.0 {
            (predicted_latest - predicted_first) / predicted_first.abs() * 100.0
        } else {
            0.0
        };

        let min = ys.iter().cloned().reduce(f64::min).unwrap_or(0.0);
        let max = ys.iter().cloned().reduce(f64::max).unwrap_or(0.0);

        Some(Self {
            metric: history.metric.clone(),
            n,
            mean: mean_y,
            std_dev: std_dev_y,
            min,
            max,
            slope,
            intercept,
            correlation,
            predicted_latest,
            trend_pct,
        })
    }

    /// Whether the trend is improving (positive slope).
    #[must_use]
    pub fn is_improving(&self) -> bool {
        self.slope > 0.0
    }

    /// Whether the trend is degrading (negative slope).
    #[must_use]
    pub fn is_degrading(&self) -> bool {
        self.slope < 0.0
    }

    /// Whether the trend shows a significant change (|trend_pct| > threshold).
    #[must_use]
    pub fn is_significant(&self, threshold_pct: f64) -> bool {
        self.trend_pct.abs() > threshold_pct
    }
}

// ---------------------------------------------------------------------------
// Aggregated report across the full database
// ---------------------------------------------------------------------------

/// Summary statistics for a single codec/metric combination across all builds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSummary {
    /// Codec identifier.
    pub codec: CodecId,
    /// Metric name.
    pub metric: String,
    /// Number of observations.
    pub observation_count: usize,
    /// Mean value.
    pub mean: f64,
    /// Standard deviation.
    pub std_dev: f64,
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
    /// Latest observed value.
    pub latest: f64,
    /// Trend analysis (if enough data).
    pub trend: Option<TrendAnalysis>,
}

/// A database-level report aggregating summaries across all codecs and metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalReport {
    /// When the report was generated (ISO-8601).
    pub generated_at: String,
    /// Total entry count in the database at the time of generation.
    pub total_entries: usize,
    /// Per-codec-metric summaries.
    pub summaries: Vec<MetricSummary>,
}

impl HistoricalReport {
    /// Generate a report from a populated [`HistoricalDb`].
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn generate(db: &HistoricalDb) -> Self {
        let codecs = db.distinct_codecs();
        let metrics = db.distinct_metrics();
        let mut summaries = Vec::new();

        for &codec in &codecs {
            for metric in &metrics {
                let history = db.history(codec, metric);
                if history.is_empty() {
                    continue;
                }
                let values = history.values();
                let n = values.len() as f64;
                let mean = values.iter().sum::<f64>() / n;
                let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
                let std_dev = var.sqrt();
                let min = values.iter().cloned().reduce(f64::min).unwrap_or(0.0);
                let max = values.iter().cloned().reduce(f64::max).unwrap_or(0.0);
                let latest = values.last().copied().unwrap_or(0.0);
                let trend = history.trend();

                summaries.push(MetricSummary {
                    codec,
                    metric: metric.clone(),
                    observation_count: values.len(),
                    mean,
                    std_dev,
                    min,
                    max,
                    latest,
                    trend,
                });
            }
        }

        Self {
            generated_at: unix_secs_to_iso8601(current_unix_secs()),
            total_entries: db.len(),
            summaries,
        }
    }

    /// Serialise the report to pretty JSON.
    ///
    /// # Errors
    ///
    /// Returns a [`BenchError`] if serialization fails.
    pub fn to_json(&self) -> BenchResult<String> {
        serde_json::to_string_pretty(self).map_err(BenchError::Serialization)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return the current Unix timestamp in seconds.
fn current_unix_secs() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_secs()
}

/// Format a Unix timestamp as ISO-8601 UTC string.
fn unix_secs_to_iso8601(secs: u64) -> String {
    let hours = (secs % 86400) / 3600;
    let minutes = (secs % 3600) / 60;
    let seconds = secs % 60;
    let z = (secs / 86400) as i64 + 719_468_i64;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let yr = if m <= 2 { y + 1 } else { y };
    format!("{yr:04}-{m:02}-{d:02}T{hours:02}:{minutes:02}:{seconds:02}Z")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(
        build: &str,
        codec: CodecId,
        metric: &str,
        value: f64,
        ts: u64,
    ) -> HistoricalEntry {
        HistoricalEntry::new(build, codec, metric, value).with_timestamp(ts)
    }

    // ---- HistoricalEntry ---------------------------------------------------

    #[test]
    fn test_entry_builder() {
        let entry = HistoricalEntry::new("build-abc", CodecId::Av1, "encoding_fps", 45.0)
            .with_sequence("forest_4k")
            .with_preset("medium")
            .with_extra("env", "ci");
        assert_eq!(entry.build_id, "build-abc");
        assert_eq!(entry.metric, "encoding_fps");
        assert!((entry.value - 45.0).abs() < 1e-9);
        assert_eq!(entry.sequence.as_deref(), Some("forest_4k"));
        assert_eq!(entry.preset.as_deref(), Some("medium"));
        assert_eq!(entry.extra.get("env").map(String::as_str), Some("ci"));
    }

    // ---- HistoricalQuery ---------------------------------------------------

    #[test]
    fn test_query_matches_codec() {
        let entry = make_entry("b1", CodecId::Av1, "fps", 30.0, 1000);
        let q_av1 = HistoricalQuery::new().with_codec(CodecId::Av1);
        let q_vp9 = HistoricalQuery::new().with_codec(CodecId::Vp9);
        assert!(q_av1.matches(&entry));
        assert!(!q_vp9.matches(&entry));
    }

    #[test]
    fn test_query_time_range() {
        let entry = make_entry("b1", CodecId::Av1, "fps", 30.0, 500);
        let inside = HistoricalQuery::new().with_time_range(400, 600);
        let before = HistoricalQuery::new().with_time_range(600, 700);
        assert!(inside.matches(&entry));
        assert!(!before.matches(&entry));
    }

    // ---- HistoricalDb ------------------------------------------------------

    #[test]
    fn test_db_append_and_query() {
        let db_path = std::env::temp_dir()
            .join("oximedia-bench-histdb-noop.ndjson")
            .to_string_lossy()
            .into_owned();
        let cfg = HistoricalDbConfig::new()
            .with_flush_on_append(false)
            .with_db_path(db_path);
        let mut db = HistoricalDb::new(cfg);
        db.append(make_entry("b1", CodecId::Av1, "fps", 30.0, 100))
            .expect("append should succeed");
        db.append(make_entry("b2", CodecId::Av1, "fps", 35.0, 200))
            .expect("append should succeed");
        db.append(make_entry("b3", CodecId::Vp9, "fps", 25.0, 300))
            .expect("append should succeed");
        assert_eq!(db.len(), 3);

        let q = HistoricalQuery::new().with_codec(CodecId::Av1);
        let results = db.query(&q);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_db_flush_and_reload() {
        let tmp = std::env::temp_dir().join("oximedia_historical_db_test.ndjson");
        let cfg = HistoricalDbConfig::new()
            .with_db_path(&tmp)
            .with_flush_on_append(false);
        let mut db = HistoricalDb::new(cfg.clone());
        db.append(make_entry("b1", CodecId::Av1, "fps", 40.0, 1000))
            .expect("append should succeed");
        db.flush().expect("flush should succeed");

        let db2 = HistoricalDb::open(cfg).expect("open should succeed");
        assert_eq!(db2.len(), 1);
        assert!((db2.entries[0].value - 40.0).abs() < 1e-9);

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_db_delete() {
        let cfg = HistoricalDbConfig::new().with_flush_on_append(false);
        let mut db = HistoricalDb::new(cfg);
        db.append(make_entry("b1", CodecId::Av1, "fps", 30.0, 100))
            .expect("append should succeed");
        db.append(make_entry("b2", CodecId::Vp9, "fps", 25.0, 200))
            .expect("append should succeed");
        let deleted = db.delete(&HistoricalQuery::new().with_codec(CodecId::Av1));
        assert_eq!(deleted, 1);
        assert_eq!(db.len(), 1);
    }

    #[test]
    fn test_db_distinct_codecs_and_metrics() {
        let cfg = HistoricalDbConfig::new().with_flush_on_append(false);
        let mut db = HistoricalDb::new(cfg);
        db.append(make_entry("b1", CodecId::Av1, "fps", 30.0, 100))
            .expect("append should succeed");
        db.append(make_entry("b2", CodecId::Vp9, "psnr", 42.0, 200))
            .expect("append should succeed");
        let codecs = db.distinct_codecs();
        assert_eq!(codecs.len(), 2);
        let metrics = db.distinct_metrics();
        assert_eq!(metrics.len(), 2);
    }

    // ---- BenchHistory & TrendAnalysis --------------------------------------

    #[test]
    fn test_trend_analysis_improving() {
        let cfg = HistoricalDbConfig::new().with_flush_on_append(false);
        let mut db = HistoricalDb::new(cfg);
        for (i, val) in [10.0, 20.0, 30.0, 40.0, 50.0].iter().enumerate() {
            db.append(make_entry("b", CodecId::Av1, "fps", *val, i as u64 * 86400))
                .expect("append should succeed");
        }
        let hist = db.history(CodecId::Av1, "fps");
        let trend = hist.trend().expect("trend should be computed");
        assert!(trend.is_improving());
        assert!(!trend.is_degrading());
        assert!(trend.trend_pct > 0.0);
    }

    #[test]
    fn test_trend_analysis_degrading() {
        let cfg = HistoricalDbConfig::new().with_flush_on_append(false);
        let mut db = HistoricalDb::new(cfg);
        for (i, val) in [50.0, 40.0, 30.0, 20.0, 10.0].iter().enumerate() {
            db.append(make_entry("b", CodecId::Vp9, "fps", *val, i as u64 * 86400))
                .expect("append should succeed");
        }
        let hist = db.history(CodecId::Vp9, "fps");
        let trend = hist.trend().expect("trend should be computed");
        assert!(trend.is_degrading());
    }

    #[test]
    fn test_bench_history_last_n() {
        let cfg = HistoricalDbConfig::new().with_flush_on_append(false);
        let mut db = HistoricalDb::new(cfg);
        for i in 0..10u64 {
            db.append(make_entry("b", CodecId::Av1, "fps", i as f64, i * 1000))
                .expect("append should succeed");
        }
        let hist = db.history(CodecId::Av1, "fps");
        let last3 = hist.last_n(3);
        assert_eq!(last3.len(), 3);
        assert_eq!(last3.entries[0].timestamp_secs, 7000);
    }

    #[test]
    fn test_bench_history_best_entry() {
        let cfg = HistoricalDbConfig::new().with_flush_on_append(false);
        let mut db = HistoricalDb::new(cfg);
        let vals = [10.0, 50.0, 30.0];
        for (i, &v) in vals.iter().enumerate() {
            db.append(make_entry("b", CodecId::Av1, "fps", v, i as u64 * 100))
                .expect("append should succeed");
        }
        let hist = db.history(CodecId::Av1, "fps");
        let best = hist.best_entry(true).expect("best entry should exist");
        assert!((best.value - 50.0).abs() < 1e-9);
        let worst = hist.best_entry(false).expect("worst entry should exist");
        assert!((worst.value - 10.0).abs() < 1e-9);
    }

    // ---- HistoricalReport --------------------------------------------------

    #[test]
    fn test_historical_report_generate() {
        let cfg = HistoricalDbConfig::new().with_flush_on_append(false);
        let mut db = HistoricalDb::new(cfg);
        for i in 0..5u64 {
            db.append(make_entry(
                "b",
                CodecId::Av1,
                "fps",
                30.0 + i as f64,
                i * 1000,
            ))
            .expect("append should succeed");
        }
        let report = HistoricalReport::generate(&db);
        assert_eq!(report.total_entries, 5);
        assert!(!report.summaries.is_empty());
        let json = report.to_json().expect("serialization should succeed");
        assert!(json.contains("fps"));
    }
}
