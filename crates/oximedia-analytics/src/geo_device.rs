//! Geographic and device breakdown analytics for viewer session metrics.
//!
//! Aggregates session-level data across two orthogonal dimensions — geographic
//! region and device type — and computes per-slice metrics such as view count,
//! unique viewer count, total watch time, and average watch time.
//!
//! ## Usage
//!
//! ```rust
//! use oximedia_analytics::geo_device::{SessionRecord, BreakdownAnalyzer, DeviceType, Region};
//!
//! let mut analyzer = BreakdownAnalyzer::new();
//! analyzer.ingest(SessionRecord {
//!     viewer_id: "v1".into(),
//!     region: Region::NorthAmerica,
//!     device: DeviceType::Desktop,
//!     watch_seconds: 120.0,
//! });
//! analyzer.ingest(SessionRecord {
//!     viewer_id: "v2".into(),
//!     region: Region::NorthAmerica,
//!     device: DeviceType::Mobile,
//!     watch_seconds: 60.0,
//! });
//! let by_region = analyzer.breakdown_by_region();
//! assert_eq!(by_region.len(), 1); // one region: NorthAmerica
//! ```

use crate::error::AnalyticsError;
use std::collections::HashMap;

// ─── DeviceType ──────────────────────────────────────────────────────────────

/// Broad device category inferred from the user-agent or client metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    /// Traditional laptop or desktop computer.
    Desktop,
    /// Smartphone.
    Mobile,
    /// Tablet (iPad, Android tablet, etc.).
    Tablet,
    /// Smart TV, streaming stick, or set-top box.
    SmartTv,
    /// Game console.
    Console,
    /// Device type could not be determined.
    Unknown,
}

impl DeviceType {
    /// Returns a human-readable label for the device type.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Desktop => "desktop",
            Self::Mobile => "mobile",
            Self::Tablet => "tablet",
            Self::SmartTv => "smart_tv",
            Self::Console => "console",
            Self::Unknown => "unknown",
        }
    }
}

// ─── Region ──────────────────────────────────────────────────────────────────

/// Geographic region grouping for broadcast and streaming analytics.
///
/// Regions correspond to broad audience groupings commonly used in media
/// rights and advertising markets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Region {
    /// United States, Canada, Mexico.
    NorthAmerica,
    /// Brazil, Argentina, Colombia, and other LATAM markets.
    LatinAmerica,
    /// UK, France, Germany, and other European markets.
    Europe,
    /// Russia and other CIS/Eastern European markets.
    EasternEurope,
    /// Middle East and Africa.
    Mea,
    /// India, Southeast Asia, Australia, New Zealand.
    AsiaPacific,
    /// Japan, South Korea, China.
    EastAsia,
    /// Region could not be mapped.
    Unknown,
}

impl Region {
    /// Returns a human-readable label for the region.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::NorthAmerica => "north_america",
            Self::LatinAmerica => "latin_america",
            Self::Europe => "europe",
            Self::EasternEurope => "eastern_europe",
            Self::Mea => "mea",
            Self::AsiaPacific => "asia_pacific",
            Self::EastAsia => "east_asia",
            Self::Unknown => "unknown",
        }
    }
}

// ─── SessionRecord ───────────────────────────────────────────────────────────

/// A single viewer session enriched with geographic and device metadata.
#[derive(Debug, Clone)]
pub struct SessionRecord {
    /// Opaque viewer / user identifier (used for unique viewer counting).
    pub viewer_id: String,
    /// Geographic region of the viewer.
    pub region: Region,
    /// Device category used for playback.
    pub device: DeviceType,
    /// Total watch time for this session in seconds.
    pub watch_seconds: f64,
}

// ─── SliceMetrics ────────────────────────────────────────────────────────────

/// Aggregated metrics for a single dimension slice (region or device).
#[derive(Debug, Clone, PartialEq)]
pub struct SliceMetrics {
    /// Total number of sessions in this slice.
    pub sessions: u64,
    /// Number of distinct `viewer_id` values (approximate unique viewers).
    pub unique_viewers: u64,
    /// Sum of all `watch_seconds` values in this slice.
    pub total_watch_seconds: f64,
    /// Average watch time per session (`total_watch_seconds / sessions`).
    pub avg_watch_seconds: f64,
}

impl SliceMetrics {
    fn new(sessions: u64, unique_viewers: u64, total_watch_seconds: f64) -> Self {
        let avg = if sessions == 0 {
            0.0
        } else {
            total_watch_seconds / sessions as f64
        };
        Self {
            sessions,
            unique_viewers,
            total_watch_seconds,
            avg_watch_seconds: avg,
        }
    }
}

// ─── BreakdownAnalyzer ───────────────────────────────────────────────────────

/// Accumulates session records and computes breakdowns by region and device.
#[derive(Debug, Default)]
pub struct BreakdownAnalyzer {
    records: Vec<SessionRecord>,
}

impl BreakdownAnalyzer {
    /// Creates an empty analyzer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
        }
    }

    /// Ingests a single session record.
    pub fn ingest(&mut self, record: SessionRecord) {
        self.records.push(record);
    }

    /// Ingests a batch of session records.
    pub fn ingest_batch(&mut self, records: impl IntoIterator<Item = SessionRecord>) {
        self.records.extend(records);
    }

    /// Total number of session records ingested so far.
    #[must_use]
    pub fn session_count(&self) -> usize {
        self.records.len()
    }

    /// Computes per-region aggregates.
    ///
    /// Returns a map from [`Region`] to [`SliceMetrics`].
    /// Regions with no sessions are omitted.
    #[must_use]
    pub fn breakdown_by_region(&self) -> HashMap<Region, SliceMetrics> {
        self.aggregate(|r| r.region)
    }

    /// Computes per-device aggregates.
    ///
    /// Returns a map from [`DeviceType`] to [`SliceMetrics`].
    /// Device types with no sessions are omitted.
    #[must_use]
    pub fn breakdown_by_device(&self) -> HashMap<DeviceType, SliceMetrics> {
        self.aggregate(|r| r.device)
    }

    /// Computes a cross-tab: breakdown by `(region, device)` pair.
    ///
    /// Returns a map from `(Region, DeviceType)` to [`SliceMetrics`].
    #[must_use]
    pub fn breakdown_by_region_and_device(&self) -> HashMap<(Region, DeviceType), SliceMetrics> {
        self.aggregate(|r| (r.region, r.device))
    }

    /// Finds the region with the highest total watch time.
    ///
    /// # Errors
    ///
    /// Returns [`AnalyticsError::InsufficientData`] when no records have been
    /// ingested.
    pub fn top_region_by_watch_time(&self) -> Result<Region, AnalyticsError> {
        if self.records.is_empty() {
            return Err(AnalyticsError::InsufficientData(
                "no sessions ingested".into(),
            ));
        }
        let breakdown = self.breakdown_by_region();
        breakdown
            .into_iter()
            .max_by(|(_, a), (_, b)| {
                a.total_watch_seconds
                    .partial_cmp(&b.total_watch_seconds)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(region, _)| region)
            .ok_or_else(|| AnalyticsError::InsufficientData("breakdown empty".into()))
    }

    /// Finds the device type with the highest session count.
    ///
    /// # Errors
    ///
    /// Returns [`AnalyticsError::InsufficientData`] when no records have been
    /// ingested.
    pub fn top_device_by_sessions(&self) -> Result<DeviceType, AnalyticsError> {
        if self.records.is_empty() {
            return Err(AnalyticsError::InsufficientData(
                "no sessions ingested".into(),
            ));
        }
        let breakdown = self.breakdown_by_device();
        breakdown
            .into_iter()
            .max_by_key(|(_, m)| m.sessions)
            .map(|(device, _)| device)
            .ok_or_else(|| AnalyticsError::InsufficientData("breakdown empty".into()))
    }

    // ── private helpers ──────────────────────────────────────────────────────

    /// Generic aggregation keyed by the output of `key_fn`.
    fn aggregate<K, F>(&self, key_fn: F) -> HashMap<K, SliceMetrics>
    where
        K: Eq + std::hash::Hash,
        F: Fn(&SessionRecord) -> K,
    {
        // Phase 1: accumulate totals.
        struct Acc {
            sessions: u64,
            viewers: std::collections::HashSet<String>,
            total_watch: f64,
        }

        let mut map: HashMap<K, Acc> = HashMap::new();

        for rec in &self.records {
            let key = key_fn(rec);
            let acc = map.entry(key).or_insert_with(|| Acc {
                sessions: 0,
                viewers: std::collections::HashSet::new(),
                total_watch: 0.0,
            });
            acc.sessions += 1;
            acc.viewers.insert(rec.viewer_id.clone());
            acc.total_watch += rec.watch_seconds;
        }

        // Phase 2: finalise into SliceMetrics.
        map.into_iter()
            .map(|(k, acc)| {
                (
                    k,
                    SliceMetrics::new(acc.sessions, acc.viewers.len() as u64, acc.total_watch),
                )
            })
            .collect()
    }
}

// ─── Period Comparison ───────────────────────────────────────────────────────

/// A tagged session record with an epoch timestamp for temporal comparisons.
#[derive(Debug, Clone)]
pub struct TimestampedRecord {
    /// The underlying session record.
    pub record: SessionRecord,
    /// Unix epoch timestamp in seconds when this session occurred.
    pub timestamp_s: i64,
}

impl TimestampedRecord {
    /// Wrap an existing record with a timestamp.
    pub fn new(record: SessionRecord, timestamp_s: i64) -> Self {
        Self {
            record,
            timestamp_s,
        }
    }
}

/// Comparison of a metric across two time periods (baseline vs comparison).
#[derive(Debug, Clone, PartialEq)]
pub struct PeriodDelta {
    /// Metric value in the baseline period.
    pub baseline: f64,
    /// Metric value in the comparison period.
    pub comparison: f64,
    /// Absolute change: `comparison − baseline`.
    pub absolute_change: f64,
    /// Relative (percentage) change: `(comparison − baseline) / baseline * 100`.
    ///
    /// `NaN` when `baseline` is zero.
    pub relative_change_pct: f64,
}

impl PeriodDelta {
    fn new(baseline: f64, comparison: f64) -> Self {
        let absolute_change = comparison - baseline;
        let relative_change_pct = if baseline.abs() < f64::EPSILON {
            f64::NAN
        } else {
            absolute_change / baseline * 100.0
        };
        Self {
            baseline,
            comparison,
            absolute_change,
            relative_change_pct,
        }
    }

    /// Returns `true` when the comparison period shows growth over baseline.
    pub fn is_growing(&self) -> bool {
        self.absolute_change > 0.0
    }
}

/// Period-over-period breakdown comparison for a single region or device slice.
#[derive(Debug, Clone)]
pub struct SliceComparison {
    /// Change in session count.
    pub sessions: PeriodDelta,
    /// Change in unique viewers.
    pub unique_viewers: PeriodDelta,
    /// Change in total watch seconds.
    pub total_watch_seconds: PeriodDelta,
    /// Change in average watch seconds per session.
    pub avg_watch_seconds: PeriodDelta,
}

/// A full geo/device breakdown report with optional period comparison.
#[derive(Debug, Clone)]
pub struct GeoDeviceReport {
    /// Per-region session metrics.
    pub by_region: HashMap<Region, SliceMetrics>,
    /// Per-device session metrics.
    pub by_device: HashMap<DeviceType, SliceMetrics>,
    /// Cross-tab (region × device) metrics.
    pub cross_tab: HashMap<(Region, DeviceType), SliceMetrics>,
    /// Total number of sessions in the report.
    pub total_sessions: u64,
    /// Total unique viewer count across all sessions.
    pub total_unique_viewers: u64,
    /// Total watch time in seconds.
    pub total_watch_seconds: f64,
    /// Average watch time per session across all sessions.
    pub overall_avg_watch_seconds: f64,
}

impl GeoDeviceReport {
    /// Fraction of total sessions that belong to `region` (0.0–1.0).
    ///
    /// Returns `0.0` when no sessions are recorded.
    pub fn region_share(&self, region: Region) -> f64 {
        if self.total_sessions == 0 {
            return 0.0;
        }
        self.by_region
            .get(&region)
            .map(|m| m.sessions as f64 / self.total_sessions as f64)
            .unwrap_or(0.0)
    }

    /// Fraction of total sessions that belong to `device` (0.0–1.0).
    ///
    /// Returns `0.0` when no sessions are recorded.
    pub fn device_share(&self, device: DeviceType) -> f64 {
        if self.total_sessions == 0 {
            return 0.0;
        }
        self.by_device
            .get(&device)
            .map(|m| m.sessions as f64 / self.total_sessions as f64)
            .unwrap_or(0.0)
    }

    /// Returns the region with the highest session count, or `None` when empty.
    pub fn dominant_region_by_sessions(&self) -> Option<Region> {
        self.by_region
            .iter()
            .max_by_key(|(_, m)| m.sessions)
            .map(|(&r, _)| r)
    }

    /// Returns the device type with the highest unique viewer count, or `None`
    /// when empty.
    pub fn dominant_device_by_viewers(&self) -> Option<DeviceType> {
        self.by_device
            .iter()
            .max_by_key(|(_, m)| m.unique_viewers)
            .map(|(&d, _)| d)
    }
}

impl BreakdownAnalyzer {
    /// Generate a full [`GeoDeviceReport`] from the ingested records.
    pub fn build_report(&self) -> GeoDeviceReport {
        let by_region = self.breakdown_by_region();
        let by_device = self.breakdown_by_device();
        let cross_tab = self.breakdown_by_region_and_device();

        let total_sessions = self.records.len() as u64;
        let total_watch_seconds: f64 = self.records.iter().map(|r| r.watch_seconds).sum();
        let total_unique_viewers = {
            let unique: std::collections::HashSet<&str> =
                self.records.iter().map(|r| r.viewer_id.as_str()).collect();
            unique.len() as u64
        };
        let overall_avg_watch_seconds = if total_sessions == 0 {
            0.0
        } else {
            total_watch_seconds / total_sessions as f64
        };

        GeoDeviceReport {
            by_region,
            by_device,
            cross_tab,
            total_sessions,
            total_unique_viewers,
            total_watch_seconds,
            overall_avg_watch_seconds,
        }
    }

    /// Compare metrics for a [`Region`] between two sub-sets of records split
    /// at `split_timestamp_s`.  Records at or before the split are the
    /// *baseline*; records after are the *comparison* period.
    ///
    /// Returns `None` when no records are tagged with timestamps (i.e., the
    /// [`TimestampedRecord`] API has not been used) or when the region is
    /// absent from either period.
    ///
    /// Use [`BreakdownAnalyzer::ingest_timestamped`] to build an analyzer with
    /// timestamp metadata.
    pub fn compare_region_periods(
        &self,
        timestamped: &[TimestampedRecord],
        region: Region,
        split_timestamp_s: i64,
    ) -> Option<SliceComparison> {
        let baseline: Vec<SessionRecord> = timestamped
            .iter()
            .filter(|t| t.timestamp_s <= split_timestamp_s && t.record.region == region)
            .map(|t| t.record.clone())
            .collect();
        let comparison: Vec<SessionRecord> = timestamped
            .iter()
            .filter(|t| t.timestamp_s > split_timestamp_s && t.record.region == region)
            .map(|t| t.record.clone())
            .collect();

        if baseline.is_empty() && comparison.is_empty() {
            return None;
        }

        let base_metrics = Self::compute_slice_metrics(&baseline);
        let comp_metrics = Self::compute_slice_metrics(&comparison);

        Some(SliceComparison {
            sessions: PeriodDelta::new(base_metrics.sessions as f64, comp_metrics.sessions as f64),
            unique_viewers: PeriodDelta::new(
                base_metrics.unique_viewers as f64,
                comp_metrics.unique_viewers as f64,
            ),
            total_watch_seconds: PeriodDelta::new(
                base_metrics.total_watch_seconds,
                comp_metrics.total_watch_seconds,
            ),
            avg_watch_seconds: PeriodDelta::new(
                base_metrics.avg_watch_seconds,
                comp_metrics.avg_watch_seconds,
            ),
        })
    }

    /// Compare metrics for a [`DeviceType`] between two time periods split at
    /// `split_timestamp_s`.
    pub fn compare_device_periods(
        &self,
        timestamped: &[TimestampedRecord],
        device: DeviceType,
        split_timestamp_s: i64,
    ) -> Option<SliceComparison> {
        let baseline: Vec<SessionRecord> = timestamped
            .iter()
            .filter(|t| t.timestamp_s <= split_timestamp_s && t.record.device == device)
            .map(|t| t.record.clone())
            .collect();
        let comparison: Vec<SessionRecord> = timestamped
            .iter()
            .filter(|t| t.timestamp_s > split_timestamp_s && t.record.device == device)
            .map(|t| t.record.clone())
            .collect();

        if baseline.is_empty() && comparison.is_empty() {
            return None;
        }

        let base_metrics = Self::compute_slice_metrics(&baseline);
        let comp_metrics = Self::compute_slice_metrics(&comparison);

        Some(SliceComparison {
            sessions: PeriodDelta::new(base_metrics.sessions as f64, comp_metrics.sessions as f64),
            unique_viewers: PeriodDelta::new(
                base_metrics.unique_viewers as f64,
                comp_metrics.unique_viewers as f64,
            ),
            total_watch_seconds: PeriodDelta::new(
                base_metrics.total_watch_seconds,
                comp_metrics.total_watch_seconds,
            ),
            avg_watch_seconds: PeriodDelta::new(
                base_metrics.avg_watch_seconds,
                comp_metrics.avg_watch_seconds,
            ),
        })
    }

    /// Ingest a collection of [`TimestampedRecord`]s (discards the timestamp
    /// metadata for the core analyzer, but returns the slice for use with
    /// comparison functions).
    pub fn ingest_timestamped(&mut self, records: &[TimestampedRecord]) {
        for tr in records {
            self.records.push(tr.record.clone());
        }
    }

    // ── private ──────────────────────────────────────────────────────────────

    /// Compute `SliceMetrics` directly from an arbitrary slice of records.
    fn compute_slice_metrics(records: &[SessionRecord]) -> SliceMetrics {
        if records.is_empty() {
            return SliceMetrics::new(0, 0, 0.0);
        }
        let sessions = records.len() as u64;
        let total_watch: f64 = records.iter().map(|r| r.watch_seconds).sum();
        let unique: std::collections::HashSet<&str> =
            records.iter().map(|r| r.viewer_id.as_str()).collect();
        SliceMetrics::new(sessions, unique.len() as u64, total_watch)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(viewer_id: &str, region: Region, device: DeviceType, watch: f64) -> SessionRecord {
        SessionRecord {
            viewer_id: viewer_id.into(),
            region,
            device,
            watch_seconds: watch,
        }
    }

    #[test]
    fn empty_analyzer_has_zero_sessions() {
        let a = BreakdownAnalyzer::new();
        assert_eq!(a.session_count(), 0);
    }

    #[test]
    fn single_record_breakdown_by_region() {
        let mut a = BreakdownAnalyzer::new();
        a.ingest(rec("v1", Region::Europe, DeviceType::Desktop, 300.0));
        let bd = a.breakdown_by_region();
        let m = bd.get(&Region::Europe).expect("Europe present");
        assert_eq!(m.sessions, 1);
        assert_eq!(m.unique_viewers, 1);
        assert!((m.total_watch_seconds - 300.0).abs() < 1e-9);
        assert!((m.avg_watch_seconds - 300.0).abs() < 1e-9);
    }

    #[test]
    fn multiple_regions_counted_separately() {
        let mut a = BreakdownAnalyzer::new();
        a.ingest(rec("v1", Region::Europe, DeviceType::Desktop, 100.0));
        a.ingest(rec("v2", Region::AsiaPacific, DeviceType::Mobile, 200.0));
        a.ingest(rec("v3", Region::Europe, DeviceType::Tablet, 150.0));

        let bd = a.breakdown_by_region();
        let eu = bd.get(&Region::Europe).expect("EU");
        let ap = bd.get(&Region::AsiaPacific).expect("AP");

        assert_eq!(eu.sessions, 2);
        assert_eq!(ap.sessions, 1);
        assert!((eu.total_watch_seconds - 250.0).abs() < 1e-9);
    }

    #[test]
    fn unique_viewers_deduplicated() {
        let mut a = BreakdownAnalyzer::new();
        // Same viewer, three sessions in Europe.
        a.ingest(rec("v1", Region::Europe, DeviceType::Desktop, 60.0));
        a.ingest(rec("v1", Region::Europe, DeviceType::Mobile, 45.0));
        a.ingest(rec("v2", Region::Europe, DeviceType::Desktop, 90.0));

        let bd = a.breakdown_by_region();
        let eu = bd.get(&Region::Europe).expect("EU");
        assert_eq!(eu.sessions, 3);
        assert_eq!(eu.unique_viewers, 2); // v1 and v2
    }

    #[test]
    fn breakdown_by_device_type() {
        let mut a = BreakdownAnalyzer::new();
        a.ingest(rec("v1", Region::NorthAmerica, DeviceType::Mobile, 100.0));
        a.ingest(rec("v2", Region::NorthAmerica, DeviceType::Mobile, 80.0));
        a.ingest(rec("v3", Region::Europe, DeviceType::Desktop, 200.0));

        let bd = a.breakdown_by_device();
        let mob = bd.get(&DeviceType::Mobile).expect("Mobile");
        let desk = bd.get(&DeviceType::Desktop).expect("Desktop");

        assert_eq!(mob.sessions, 2);
        assert_eq!(desk.sessions, 1);
    }

    #[test]
    fn breakdown_cross_tab() {
        let mut a = BreakdownAnalyzer::new();
        a.ingest(rec("v1", Region::Europe, DeviceType::Mobile, 60.0));
        a.ingest(rec("v2", Region::Europe, DeviceType::Desktop, 120.0));
        a.ingest(rec("v3", Region::NorthAmerica, DeviceType::Mobile, 90.0));

        let bd = a.breakdown_by_region_and_device();
        assert_eq!(
            bd.get(&(Region::Europe, DeviceType::Mobile))
                .map(|m| m.sessions),
            Some(1)
        );
        assert_eq!(
            bd.get(&(Region::Europe, DeviceType::Desktop))
                .map(|m| m.sessions),
            Some(1)
        );
        assert_eq!(
            bd.get(&(Region::NorthAmerica, DeviceType::Mobile))
                .map(|m| m.sessions),
            Some(1)
        );
    }

    #[test]
    fn top_region_by_watch_time() {
        let mut a = BreakdownAnalyzer::new();
        a.ingest(rec("v1", Region::Europe, DeviceType::Desktop, 100.0));
        a.ingest(rec("v2", Region::AsiaPacific, DeviceType::Mobile, 500.0));
        a.ingest(rec("v3", Region::AsiaPacific, DeviceType::Tablet, 300.0));

        let top = a.top_region_by_watch_time().expect("top region");
        assert_eq!(top, Region::AsiaPacific);
    }

    #[test]
    fn top_device_by_sessions() {
        let mut a = BreakdownAnalyzer::new();
        a.ingest(rec("v1", Region::Europe, DeviceType::Mobile, 60.0));
        a.ingest(rec("v2", Region::Europe, DeviceType::Mobile, 60.0));
        a.ingest(rec("v3", Region::Europe, DeviceType::SmartTv, 200.0));

        let top = a.top_device_by_sessions().expect("top device");
        assert_eq!(top, DeviceType::Mobile);
    }

    #[test]
    fn empty_analyzer_top_region_errors() {
        let a = BreakdownAnalyzer::new();
        assert!(a.top_region_by_watch_time().is_err());
    }

    #[test]
    fn empty_analyzer_top_device_errors() {
        let a = BreakdownAnalyzer::new();
        assert!(a.top_device_by_sessions().is_err());
    }

    #[test]
    fn device_type_labels_are_stable() {
        assert_eq!(DeviceType::Desktop.label(), "desktop");
        assert_eq!(DeviceType::Mobile.label(), "mobile");
        assert_eq!(DeviceType::SmartTv.label(), "smart_tv");
        assert_eq!(DeviceType::Unknown.label(), "unknown");
    }

    #[test]
    fn region_labels_are_stable() {
        assert_eq!(Region::NorthAmerica.label(), "north_america");
        assert_eq!(Region::AsiaPacific.label(), "asia_pacific");
        assert_eq!(Region::Unknown.label(), "unknown");
    }

    #[test]
    fn ingest_batch() {
        let mut a = BreakdownAnalyzer::new();
        let records = vec![
            rec("v1", Region::Europe, DeviceType::Desktop, 100.0),
            rec("v2", Region::Europe, DeviceType::Desktop, 200.0),
        ];
        a.ingest_batch(records);
        assert_eq!(a.session_count(), 2);
    }

    // ── GeoDeviceReport ───────────────────────────────────────────────────────

    #[test]
    fn build_report_totals_correct() {
        let mut a = BreakdownAnalyzer::new();
        a.ingest(rec("v1", Region::Europe, DeviceType::Desktop, 100.0));
        a.ingest(rec("v2", Region::Europe, DeviceType::Mobile, 200.0));
        a.ingest(rec("v3", Region::AsiaPacific, DeviceType::Mobile, 300.0));

        let report = a.build_report();
        assert_eq!(report.total_sessions, 3);
        assert_eq!(report.total_unique_viewers, 3);
        assert!((report.total_watch_seconds - 600.0).abs() < 1e-9);
        assert!((report.overall_avg_watch_seconds - 200.0).abs() < 1e-9);
    }

    #[test]
    fn build_report_empty_analyzer() {
        let a = BreakdownAnalyzer::new();
        let report = a.build_report();
        assert_eq!(report.total_sessions, 0);
        assert_eq!(report.total_unique_viewers, 0);
        assert_eq!(report.total_watch_seconds, 0.0);
        assert_eq!(report.overall_avg_watch_seconds, 0.0);
        assert!(report.by_region.is_empty());
    }

    #[test]
    fn region_share_sums_to_one() {
        let mut a = BreakdownAnalyzer::new();
        a.ingest(rec("v1", Region::Europe, DeviceType::Desktop, 100.0));
        a.ingest(rec("v2", Region::Europe, DeviceType::Desktop, 100.0));
        a.ingest(rec("v3", Region::AsiaPacific, DeviceType::Mobile, 100.0));

        let report = a.build_report();
        let eu_share = report.region_share(Region::Europe);
        let ap_share = report.region_share(Region::AsiaPacific);
        let unknown_share = report.region_share(Region::Unknown);
        let total = eu_share + ap_share + unknown_share;
        assert!((total - (eu_share + ap_share)).abs() < 1e-9); // unknown is 0
        assert!((eu_share + ap_share - 1.0).abs() < 1e-9);
    }

    #[test]
    fn device_share_correct() {
        let mut a = BreakdownAnalyzer::new();
        a.ingest(rec("v1", Region::Europe, DeviceType::Mobile, 100.0));
        a.ingest(rec("v2", Region::Europe, DeviceType::Mobile, 100.0));
        a.ingest(rec("v3", Region::Europe, DeviceType::Desktop, 100.0));

        let report = a.build_report();
        let mobile_share = report.device_share(DeviceType::Mobile);
        let desktop_share = report.device_share(DeviceType::Desktop);
        assert!((mobile_share - 2.0 / 3.0).abs() < 1e-9);
        assert!((desktop_share - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn dominant_region_by_sessions() {
        let mut a = BreakdownAnalyzer::new();
        a.ingest(rec("v1", Region::Europe, DeviceType::Desktop, 100.0));
        a.ingest(rec("v2", Region::Europe, DeviceType::Mobile, 100.0));
        a.ingest(rec("v3", Region::AsiaPacific, DeviceType::Mobile, 100.0));

        let report = a.build_report();
        assert_eq!(report.dominant_region_by_sessions(), Some(Region::Europe));
    }

    #[test]
    fn dominant_device_by_viewers() {
        let mut a = BreakdownAnalyzer::new();
        a.ingest(rec("v1", Region::Europe, DeviceType::SmartTv, 100.0));
        a.ingest(rec("v2", Region::Europe, DeviceType::SmartTv, 100.0));
        a.ingest(rec("v3", Region::Europe, DeviceType::Desktop, 100.0));

        let report = a.build_report();
        assert_eq!(
            report.dominant_device_by_viewers(),
            Some(DeviceType::SmartTv)
        );
    }

    // ── Period comparison ─────────────────────────────────────────────────────

    fn ts_rec(
        viewer_id: &str,
        region: Region,
        device: DeviceType,
        watch: f64,
        ts: i64,
    ) -> TimestampedRecord {
        TimestampedRecord::new(
            SessionRecord {
                viewer_id: viewer_id.into(),
                region,
                device,
                watch_seconds: watch,
            },
            ts,
        )
    }

    #[test]
    fn compare_region_periods_sessions_grow() {
        let a = BreakdownAnalyzer::new();
        let records = vec![
            ts_rec("v1", Region::Europe, DeviceType::Desktop, 60.0, 100),
            ts_rec("v2", Region::Europe, DeviceType::Mobile, 90.0, 200),
            ts_rec("v3", Region::Europe, DeviceType::Desktop, 120.0, 201),
            ts_rec("v4", Region::Europe, DeviceType::Mobile, 80.0, 300),
        ];
        // Split at 200: baseline has 2 sessions (v1,v2), comparison has 2 (v3,v4).
        let cmp = a
            .compare_region_periods(&records, Region::Europe, 200)
            .expect("comparison");
        assert_eq!(cmp.sessions.baseline as u64, 2);
        assert_eq!(cmp.sessions.comparison as u64, 2);
        assert!((cmp.sessions.absolute_change).abs() < 1e-9); // no change
    }

    #[test]
    fn compare_region_periods_watch_time_grows() {
        let a = BreakdownAnalyzer::new();
        let records = vec![
            ts_rec("v1", Region::NorthAmerica, DeviceType::Mobile, 60.0, 50),
            ts_rec("v2", Region::NorthAmerica, DeviceType::Mobile, 600.0, 150),
        ];
        // Baseline (≤100): 60s watch; comparison (>100): 600s watch.
        let cmp = a
            .compare_region_periods(&records, Region::NorthAmerica, 100)
            .expect("comparison");
        assert!((cmp.total_watch_seconds.baseline - 60.0).abs() < 1e-9);
        assert!((cmp.total_watch_seconds.comparison - 600.0).abs() < 1e-9);
        assert!(cmp.total_watch_seconds.is_growing());
    }

    #[test]
    fn compare_device_periods_absent_region_returns_none() {
        let a = BreakdownAnalyzer::new();
        let records = vec![ts_rec(
            "v1",
            Region::AsiaPacific,
            DeviceType::Mobile,
            60.0,
            100,
        )];
        // Tablet has no records in either period.
        let cmp = a.compare_device_periods(&records, DeviceType::Tablet, 50);
        assert!(cmp.is_none());
    }

    #[test]
    fn period_delta_relative_change_computed() {
        let delta = PeriodDelta::new(100.0, 150.0);
        assert!((delta.relative_change_pct - 50.0).abs() < 1e-9);
        assert!(delta.is_growing());
    }

    #[test]
    fn period_delta_zero_baseline_gives_nan_relative() {
        let delta = PeriodDelta::new(0.0, 10.0);
        assert!(delta.relative_change_pct.is_nan());
    }

    #[test]
    fn ingest_timestamped_populates_analyzer() {
        let mut a = BreakdownAnalyzer::new();
        let records = vec![
            ts_rec("v1", Region::Europe, DeviceType::Desktop, 100.0, 1),
            ts_rec("v2", Region::Europe, DeviceType::Mobile, 200.0, 2),
        ];
        a.ingest_timestamped(&records);
        assert_eq!(a.session_count(), 2);
    }
}
