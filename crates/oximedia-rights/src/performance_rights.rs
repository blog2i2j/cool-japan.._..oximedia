//! Performance rights tracking for broadcast, streaming, and live performances.
//!
//! Performance rights cover the public performance of musical works — i.e. any
//! time a protected composition (not just a recording) is played publicly.
//! Rights organisations (PROs) such as ASCAP, BMI, SOCAN, SESAC, PRS, and
//! STIM collect and distribute these royalties on behalf of songwriters and
//! publishers.
//!
//! # Features
//!
//! * [`PerformanceCategory`] distinguishes broadcast, streaming, live, and
//!   synchronisation performances.
//! * [`PerformanceRecord`] captures every public performance event with the
//!   metadata PROs require for royalty allocation.
//! * [`PerformanceReport`] aggregates records into a period-based report.
//! * [`BlanketLicense`] + [`BlanketLicenseMatcher`] determine whether a given
//!   performance is covered by an existing blanket licence.
//! * [`PerformanceRightsTracker`] provides in-memory CRUD for performance
//!   records and report generation.

#![allow(missing_docs)]
#![allow(clippy::cast_precision_loss)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{Result, RightsError};

// ── PerformanceCategory ───────────────────────────────────────────────────────

/// High-level category of a public performance.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PerformanceCategory {
    /// Terrestrial radio or television broadcast.
    Broadcast,
    /// Internet-based audio or video streaming.
    Streaming,
    /// Live concert, club, or theatrical performance.
    LivePerformance,
    /// Use in a film/TV/ad sync — the "performance" aspect of a sync licence.
    Synchronisation,
    /// Background music in a public venue (hotel, restaurant, retail).
    BackgroundMusic,
    /// Podcast or on-demand audio programme.
    Podcast,
    /// User-generated / user-uploaded content platform (e.g. YouTube UGC).
    UserGenerated,
    /// Custom / PRO-specific category.
    Custom(String),
}

impl std::fmt::Display for PerformanceCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Broadcast => write!(f, "Broadcast"),
            Self::Streaming => write!(f, "Streaming"),
            Self::LivePerformance => write!(f, "Live Performance"),
            Self::Synchronisation => write!(f, "Synchronisation"),
            Self::BackgroundMusic => write!(f, "Background Music"),
            Self::Podcast => write!(f, "Podcast"),
            Self::UserGenerated => write!(f, "User Generated"),
            Self::Custom(s) => write!(f, "{s}"),
        }
    }
}

// ── PerformanceRecord ─────────────────────────────────────────────────────────

/// A single public performance event for a musical work.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRecord {
    /// Unique identifier for this performance record.
    pub id: String,
    /// ISRC of the master recording used (if applicable).
    pub isrc: Option<String>,
    /// ISWC of the underlying musical composition.
    pub iswc: Option<String>,
    /// Title of the musical work.
    pub work_title: String,
    /// Name of the performing artist / band.
    pub artist: String,
    /// Category of performance.
    pub category: PerformanceCategory,
    /// ISO 3166-1 alpha-2 country where the performance occurred.
    pub territory: String,
    /// Unix timestamp of the performance.
    pub performed_at: u64,
    /// Duration of this performance in seconds.
    pub duration_secs: f64,
    /// Number of plays / streams (1 for live; >1 for aggregate streaming batches).
    pub play_count: u64,
    /// Name of the venue, station, or platform.
    pub venue_or_platform: String,
    /// Audience size estimate (0 = unknown).
    pub audience_size: u64,
    /// PRO affiliation expected to collect for this performance.
    pub collecting_pro: Option<String>,
    /// Whether this performance has already been reported to the PRO.
    pub reported: bool,
}

impl PerformanceRecord {
    /// Create a new performance record.
    pub fn new(
        id: impl Into<String>,
        work_title: impl Into<String>,
        artist: impl Into<String>,
        category: PerformanceCategory,
        territory: impl Into<String>,
        performed_at: u64,
        duration_secs: f64,
        play_count: u64,
        venue_or_platform: impl Into<String>,
    ) -> Result<Self> {
        if duration_secs < 0.0 {
            return Err(RightsError::InvalidOperation(
                "duration_secs must be non-negative".into(),
            ));
        }
        if play_count == 0 {
            return Err(RightsError::InvalidOperation(
                "play_count must be at least 1".into(),
            ));
        }
        Ok(Self {
            id: id.into(),
            isrc: None,
            iswc: None,
            work_title: work_title.into(),
            artist: artist.into(),
            category,
            territory: territory.into(),
            performed_at,
            duration_secs,
            play_count,
            venue_or_platform: venue_or_platform.into(),
            audience_size: 0,
            collecting_pro: None,
            reported: false,
        })
    }

    /// Mark this record as reported.
    pub fn mark_reported(&mut self) {
        self.reported = true;
    }

    /// Effective total duration in seconds (duration × play_count).
    #[must_use]
    pub fn total_duration_secs(&self) -> f64 {
        self.duration_secs * self.play_count as f64
    }
}

// ── ReportingPeriod ───────────────────────────────────────────────────────────

/// A calendar period used to scope a performance report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportingPeriod {
    /// Start of the period (Unix timestamp, inclusive).
    pub start: u64,
    /// End of the period (Unix timestamp, inclusive).
    pub end: u64,
    /// Human-readable label (e.g. "Q1 2024", "January 2024").
    pub label: String,
}

impl ReportingPeriod {
    /// Create a new reporting period.
    pub fn new(start: u64, end: u64, label: impl Into<String>) -> Result<Self> {
        if end < start {
            return Err(RightsError::InvalidOperation(
                "end must be >= start".into(),
            ));
        }
        Ok(Self {
            start,
            end,
            label: label.into(),
        })
    }

    /// Return `true` if a Unix timestamp falls within this period.
    #[must_use]
    pub fn contains(&self, ts: u64) -> bool {
        ts >= self.start && ts <= self.end
    }
}

// ── PerformanceSummary ────────────────────────────────────────────────────────

/// Aggregated statistics for a single work within a reporting period.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    /// ISWC / ISRC / title used as the grouping key.
    pub work_key: String,
    /// Title of the musical work.
    pub work_title: String,
    /// Total number of performance records contributing to this summary.
    pub record_count: u64,
    /// Total number of individual plays / streams.
    pub total_plays: u64,
    /// Total performed duration in seconds.
    pub total_duration_secs: f64,
    /// Breakdown of plays by category.
    pub plays_by_category: HashMap<String, u64>,
    /// Breakdown of plays by territory.
    pub plays_by_territory: HashMap<String, u64>,
}

// ── PerformanceReport ─────────────────────────────────────────────────────────

/// An aggregated performance report for a given period and PRO.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    /// Unique report identifier.
    pub id: String,
    /// PRO to which this report will be submitted.
    pub target_pro: String,
    /// The period covered by this report.
    pub period: ReportingPeriod,
    /// Per-work summaries.
    pub summaries: Vec<PerformanceSummary>,
    /// Total plays across all works.
    pub grand_total_plays: u64,
    /// Total performed duration across all works, in seconds.
    pub grand_total_duration_secs: f64,
    /// Unix timestamp when the report was generated.
    pub generated_at: u64,
}

impl PerformanceReport {
    /// Render the report as a CSV string suitable for PRO submission.
    #[must_use]
    pub fn to_csv(&self) -> String {
        let mut out = String::new();
        out.push_str("Work Key,Title,Records,Total Plays,Total Duration (s),Broadcast,Streaming,Live\n");
        for s in &self.summaries {
            let broadcast = s.plays_by_category.get("Broadcast").copied().unwrap_or(0);
            let streaming = s.plays_by_category.get("Streaming").copied().unwrap_or(0);
            let live = s
                .plays_by_category
                .get("Live Performance")
                .copied()
                .unwrap_or(0);
            out.push_str(&format!(
                "{},{},{},{},{:.2},{},{},{}\n",
                s.work_key,
                s.work_title,
                s.record_count,
                s.total_plays,
                s.total_duration_secs,
                broadcast,
                streaming,
                live,
            ));
        }
        out
    }
}

// ── BlanketLicense ────────────────────────────────────────────────────────────

/// A blanket licence granted by a PRO covering a broad set of performances.
///
/// A blanket licence allows the holder to perform any work in the PRO's
/// repertoire for the term and territory specified, typically in exchange for
/// a flat annual fee.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlanketLicense {
    /// Unique licence identifier.
    pub id: String,
    /// The PRO that issued this licence.
    pub issuing_pro: String,
    /// The licensee (broadcaster, venue, platform, etc.).
    pub licensee: String,
    /// ISO 3166-1 alpha-2 territories covered.  Empty = worldwide.
    pub territories: Vec<String>,
    /// Performance categories covered by this licence.
    pub categories: Vec<PerformanceCategory>,
    /// Licence start date (Unix timestamp).
    pub valid_from: u64,
    /// Licence expiry date (Unix timestamp; `None` = perpetual).
    pub valid_until: Option<u64>,
    /// Annual licence fee in the nominated currency.
    pub annual_fee: f64,
    /// ISO 4217 currency code.
    pub currency: String,
}

impl BlanketLicense {
    /// Return `true` if this licence is active at the given Unix timestamp.
    #[must_use]
    pub fn is_active_at(&self, ts: u64) -> bool {
        if ts < self.valid_from {
            return false;
        }
        match self.valid_until {
            Some(end) => ts <= end,
            None => true,
        }
    }

    /// Return `true` if this licence covers the given territory.
    #[must_use]
    pub fn covers_territory(&self, territory: &str) -> bool {
        self.territories.is_empty() || self.territories.iter().any(|t| t == territory)
    }

    /// Return `true` if this licence covers the given performance category.
    #[must_use]
    pub fn covers_category(&self, category: &PerformanceCategory) -> bool {
        self.categories.is_empty() || self.categories.iter().any(|c| c == category)
    }
}

// ── BlanketLicenseMatcher ─────────────────────────────────────────────────────

/// Checks whether a performance record is covered by any known blanket licence.
#[derive(Debug, Default)]
pub struct BlanketLicenseMatcher {
    licenses: Vec<BlanketLicense>,
}

impl BlanketLicenseMatcher {
    /// Create an empty matcher.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a blanket licence.
    pub fn add_license(&mut self, license: BlanketLicense) {
        self.licenses.push(license);
    }

    /// Return the first active blanket licence that covers the given
    /// performance record, or `None` if no match is found.
    #[must_use]
    pub fn find_covering_license<'a>(
        &'a self,
        record: &PerformanceRecord,
    ) -> Option<&'a BlanketLicense> {
        self.licenses.iter().find(|lic| {
            lic.is_active_at(record.performed_at)
                && lic.covers_territory(&record.territory)
                && lic.covers_category(&record.category)
        })
    }

    /// Return `true` if the performance is covered by any registered blanket
    /// licence from the given PRO.
    #[must_use]
    pub fn is_covered_by_pro(&self, record: &PerformanceRecord, pro: &str) -> bool {
        self.licenses.iter().any(|lic| {
            lic.issuing_pro == pro
                && lic.is_active_at(record.performed_at)
                && lic.covers_territory(&record.territory)
                && lic.covers_category(&record.category)
        })
    }
}

// ── PerformanceRightsTracker ──────────────────────────────────────────────────

/// In-memory tracker for performance records with report generation.
#[derive(Debug, Default)]
pub struct PerformanceRightsTracker {
    records: HashMap<String, PerformanceRecord>,
}

impl PerformanceRightsTracker {
    /// Create a new, empty tracker.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a performance record.  Returns an error if the ID already exists.
    pub fn add_record(&mut self, record: PerformanceRecord) -> Result<()> {
        if self.records.contains_key(&record.id) {
            return Err(RightsError::InvalidOperation(format!(
                "duplicate record id: {}",
                record.id
            )));
        }
        self.records.insert(record.id.clone(), record);
        Ok(())
    }

    /// Retrieve a record by ID.
    pub fn get_record(&self, id: &str) -> Result<&PerformanceRecord> {
        self.records
            .get(id)
            .ok_or_else(|| RightsError::NotFound(format!("performance record '{id}'")))
    }

    /// Update an existing record.
    pub fn update_record(&mut self, record: PerformanceRecord) -> Result<()> {
        if !self.records.contains_key(&record.id) {
            return Err(RightsError::NotFound(format!(
                "performance record '{}'",
                record.id
            )));
        }
        self.records.insert(record.id.clone(), record);
        Ok(())
    }

    /// Remove a record by ID.
    pub fn remove_record(&mut self, id: &str) -> Result<PerformanceRecord> {
        self.records
            .remove(id)
            .ok_or_else(|| RightsError::NotFound(format!("performance record '{id}'")))
    }

    /// Total number of records held.
    #[must_use]
    pub fn record_count(&self) -> usize {
        self.records.len()
    }

    /// Return all unreported records.
    #[must_use]
    pub fn unreported_records(&self) -> Vec<&PerformanceRecord> {
        self.records.values().filter(|r| !r.reported).collect()
    }

    /// Mark all records in the given period as reported.
    pub fn mark_period_as_reported(&mut self, period: &ReportingPeriod) {
        for record in self.records.values_mut() {
            if period.contains(record.performed_at) {
                record.reported = true;
            }
        }
    }

    /// Generate a [`PerformanceReport`] for the given period and PRO.
    ///
    /// Only records whose `collecting_pro` matches `target_pro` (or whose
    /// `collecting_pro` is `None`) and whose `performed_at` falls within the
    /// period are included.
    pub fn generate_report(
        &self,
        report_id: impl Into<String>,
        target_pro: impl Into<String>,
        period: ReportingPeriod,
        generated_at: u64,
    ) -> Result<PerformanceReport> {
        let target_pro = target_pro.into();

        // collect matching records
        let matching: Vec<&PerformanceRecord> = self
            .records
            .values()
            .filter(|r| {
                period.contains(r.performed_at)
                    && r.collecting_pro
                        .as_deref()
                        .map_or(true, |p| p == target_pro)
            })
            .collect();

        // aggregate per work (keyed by ISWC if present, else work_title)
        let mut agg: HashMap<String, PerformanceSummary> = HashMap::new();
        for record in &matching {
            let key = record
                .iswc
                .clone()
                .unwrap_or_else(|| record.work_title.clone());
            let summary = agg.entry(key.clone()).or_insert_with(|| PerformanceSummary {
                work_key: key,
                work_title: record.work_title.clone(),
                record_count: 0,
                total_plays: 0,
                total_duration_secs: 0.0,
                plays_by_category: HashMap::new(),
                plays_by_territory: HashMap::new(),
            });
            summary.record_count += 1;
            summary.total_plays += record.play_count;
            summary.total_duration_secs += record.total_duration_secs();
            *summary
                .plays_by_category
                .entry(record.category.to_string())
                .or_insert(0) += record.play_count;
            *summary
                .plays_by_territory
                .entry(record.territory.clone())
                .or_insert(0) += record.play_count;
        }

        let summaries: Vec<PerformanceSummary> = agg.into_values().collect();
        let grand_total_plays = summaries.iter().map(|s| s.total_plays).sum();
        let grand_total_duration_secs = summaries.iter().map(|s| s.total_duration_secs).sum();

        Ok(PerformanceReport {
            id: report_id.into(),
            target_pro,
            period,
            summaries,
            grand_total_plays,
            grand_total_duration_secs,
            generated_at,
        })
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_record(id: &str, category: PerformanceCategory, plays: u64) -> PerformanceRecord {
        let mut r = PerformanceRecord::new(
            id,
            "Test Song",
            "Test Artist",
            category,
            "US",
            1_000_000_u64,
            180.0,
            plays,
            "Radio Station WXYZ",
        )
        .unwrap();
        r.collecting_pro = Some("ASCAP".into());
        r.iswc = Some("T-123456789-0".into());
        r
    }

    #[test]
    fn test_performance_record_creation_valid() {
        let r = PerformanceRecord::new(
            "r1",
            "Song A",
            "Artist A",
            PerformanceCategory::Streaming,
            "GB",
            2_000_000_u64,
            210.0,
            500,
            "Spotify",
        )
        .unwrap();
        assert_eq!(r.work_title, "Song A");
        assert_eq!(r.play_count, 500);
    }

    #[test]
    fn test_performance_record_zero_plays_rejected() {
        let result = PerformanceRecord::new(
            "r2",
            "Song B",
            "Artist B",
            PerformanceCategory::Broadcast,
            "US",
            0,
            60.0,
            0, // invalid
            "Station",
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_performance_record_total_duration() {
        let r = sample_record("r3", PerformanceCategory::Streaming, 10);
        // 10 plays × 180 s = 1800 s
        assert!((r.total_duration_secs() - 1800.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_reporting_period_contains() {
        let period = ReportingPeriod::new(1_000, 2_000, "Test").unwrap();
        assert!(period.contains(1_500));
        assert!(!period.contains(999));
        assert!(!period.contains(2_001));
    }

    #[test]
    fn test_reporting_period_invalid() {
        let result = ReportingPeriod::new(2_000, 1_000, "Bad");
        assert!(result.is_err());
    }

    #[test]
    fn test_tracker_add_and_get() {
        let mut tracker = PerformanceRightsTracker::new();
        tracker
            .add_record(sample_record("r10", PerformanceCategory::Broadcast, 1))
            .unwrap();
        let r = tracker.get_record("r10").unwrap();
        assert_eq!(r.work_title, "Test Song");
    }

    #[test]
    fn test_tracker_duplicate_rejected() {
        let mut tracker = PerformanceRightsTracker::new();
        tracker
            .add_record(sample_record("dup", PerformanceCategory::Streaming, 1))
            .unwrap();
        let result = tracker.add_record(sample_record("dup", PerformanceCategory::Streaming, 1));
        assert!(result.is_err());
    }

    #[test]
    fn test_tracker_generate_report() {
        let mut tracker = PerformanceRightsTracker::new();
        // two streaming records within period
        let r1 = sample_record("g1", PerformanceCategory::Streaming, 100);
        let r2 = sample_record("g2", PerformanceCategory::Broadcast, 50);
        tracker.add_record(r1).unwrap();
        tracker.add_record(r2).unwrap();

        let period = ReportingPeriod::new(900_000, 2_000_000, "Q1").unwrap();
        let report = tracker
            .generate_report("rep1", "ASCAP", period, 3_000_000)
            .unwrap();

        assert_eq!(report.grand_total_plays, 150);
        assert_eq!(report.target_pro, "ASCAP");
        assert!(!report.summaries.is_empty());
    }

    #[test]
    fn test_blanket_license_matcher() {
        let license = BlanketLicense {
            id: "lic1".into(),
            issuing_pro: "BMI".into(),
            licensee: "Radio Corp".into(),
            territories: vec!["US".into()],
            categories: vec![PerformanceCategory::Broadcast],
            valid_from: 0,
            valid_until: None,
            annual_fee: 5000.0,
            currency: "USD".into(),
        };
        let mut matcher = BlanketLicenseMatcher::new();
        matcher.add_license(license);

        let record = PerformanceRecord::new(
            "bl1",
            "Hit Song",
            "Some Artist",
            PerformanceCategory::Broadcast,
            "US",
            500_000,
            200.0,
            1,
            "Radio Corp FM",
        )
        .unwrap();

        let found = matcher.find_covering_license(&record);
        assert!(found.is_some());
        assert_eq!(found.unwrap().id, "lic1");
        assert!(matcher.is_covered_by_pro(&record, "BMI"));
        assert!(!matcher.is_covered_by_pro(&record, "ASCAP"));
    }

    #[test]
    fn test_report_to_csv_format() {
        let mut tracker = PerformanceRightsTracker::new();
        tracker
            .add_record(sample_record("csv1", PerformanceCategory::Streaming, 200))
            .unwrap();
        let period = ReportingPeriod::new(0, u64::MAX, "All Time").unwrap();
        let report = tracker
            .generate_report("r_csv", "ASCAP", period, 0)
            .unwrap();
        let csv = report.to_csv();
        assert!(csv.contains("Work Key,Title"));
        assert!(csv.contains("Test Song"));
    }
}
