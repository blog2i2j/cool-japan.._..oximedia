//! Unified archive integrity dashboard aggregating multiple integrity dimensions.
//!
//! This module provides a comprehensive view of archive health by combining:
//! - Fixity check outcomes
//! - Replication status
//! - Format obsolescence risk
//! - Bit-rot detection events
//! - Retention compliance
//! - Access log anomalies
//!
//! It produces a scored `IntegrityReport` suitable for display, alerting, and
//! automated remediation workflows.

use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Overall health tier of the archive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HealthTier {
    /// Critical issues requiring immediate action.
    Critical,
    /// Significant issues requiring prompt attention.
    Warning,
    /// Minor issues to monitor.
    Caution,
    /// No issues detected.
    Healthy,
}

impl HealthTier {
    /// Returns a color code suitable for dashboard display (ANSI-safe label).
    #[must_use]
    pub const fn color_label(&self) -> &'static str {
        match self {
            Self::Critical => "red",
            Self::Warning => "orange",
            Self::Caution => "yellow",
            Self::Healthy => "green",
        }
    }

    /// Returns a human-readable tier name.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Critical => "CRITICAL",
            Self::Warning => "WARNING",
            Self::Caution => "CAUTION",
            Self::Healthy => "HEALTHY",
        }
    }

    /// Determines the health tier from a 0.0–1.0 score.
    #[must_use]
    pub fn from_score(score: f64) -> Self {
        if score >= 0.90 {
            Self::Healthy
        } else if score >= 0.70 {
            Self::Caution
        } else if score >= 0.40 {
            Self::Warning
        } else {
            Self::Critical
        }
    }
}

/// The result of a single fixity check on an individual object.
#[derive(Debug, Clone)]
pub struct FixityCheckRecord {
    /// Identifier of the archival object.
    pub object_id: String,
    /// When the check was performed.
    pub checked_at: SystemTime,
    /// Whether the check passed.
    pub passed: bool,
    /// Optional error message if failed.
    pub error_message: Option<String>,
}

impl FixityCheckRecord {
    /// Creates a passing fixity record.
    #[must_use]
    pub fn passed(object_id: impl Into<String>) -> Self {
        Self {
            object_id: object_id.into(),
            checked_at: SystemTime::now(),
            passed: true,
            error_message: None,
        }
    }

    /// Creates a failing fixity record with an error message.
    #[must_use]
    pub fn failed(object_id: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            object_id: object_id.into(),
            checked_at: SystemTime::now(),
            passed: false,
            error_message: Some(error.into()),
        }
    }
}

/// Replication status for an archival object.
#[derive(Debug, Clone)]
pub struct ReplicationStatus {
    /// Identifier of the archival object.
    pub object_id: String,
    /// Number of confirmed copies across all locations.
    pub confirmed_copies: u32,
    /// Minimum required copies per policy.
    pub required_copies: u32,
    /// Storage locations holding a copy.
    pub locations: Vec<String>,
}

impl ReplicationStatus {
    /// Creates a new replication status record.
    #[must_use]
    pub fn new(object_id: impl Into<String>, required_copies: u32) -> Self {
        Self {
            object_id: object_id.into(),
            confirmed_copies: 0,
            required_copies,
            locations: Vec::new(),
        }
    }

    /// Adds a confirmed copy at the given location.
    pub fn add_copy(&mut self, location: impl Into<String>) {
        self.locations.push(location.into());
        self.confirmed_copies += 1;
    }

    /// Returns true if the object meets its replication requirement.
    #[must_use]
    pub fn meets_requirement(&self) -> bool {
        self.confirmed_copies >= self.required_copies
    }

    /// Returns the copy deficit (0 if requirement is met).
    #[must_use]
    pub fn copy_deficit(&self) -> u32 {
        self.required_copies.saturating_sub(self.confirmed_copies)
    }
}

/// Format obsolescence risk level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ObsolescenceRisk {
    /// Format is widely supported and actively maintained.
    Low,
    /// Format support may decline in 5–10 years.
    Medium,
    /// Format is already deprecated or rarely supported.
    High,
    /// Format is effectively obsolete.
    Critical,
}

impl ObsolescenceRisk {
    /// Returns a numeric penalty factor (0.0 = no penalty, 1.0 = critical).
    #[must_use]
    pub const fn penalty(&self) -> f64 {
        match self {
            Self::Low => 0.0,
            Self::Medium => 0.15,
            Self::High => 0.40,
            Self::Critical => 0.70,
        }
    }

    /// Returns a human-readable label.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Critical => "critical",
        }
    }
}

/// An entry representing a detected bit-rot event.
#[derive(Debug, Clone)]
pub struct BitRotEvent {
    /// Identifier of the affected object.
    pub object_id: String,
    /// When the event was detected.
    pub detected_at: SystemTime,
    /// Byte offset of the first detected corruption (if known).
    pub byte_offset: Option<u64>,
    /// Whether the object was recovered from a replica.
    pub recovered: bool,
}

/// Dimension scores feeding into the composite integrity score.
#[derive(Debug, Clone, Default)]
pub struct DimensionScores {
    /// Fixity dimension: ratio of objects with passing fixity checks.
    pub fixity: f64,
    /// Replication dimension: ratio of objects meeting copy requirements.
    pub replication: f64,
    /// Format health dimension: 1.0 minus weighted obsolescence penalty.
    pub format_health: f64,
    /// Bit-rot dimension: 1.0 minus ratio of affected objects.
    pub bit_rot_health: f64,
    /// Staleness dimension: 1.0 if all checks are recent, decreasing with age.
    pub staleness: f64,
}

impl DimensionScores {
    /// Computes the weighted composite score.
    ///
    /// Weights: fixity=0.35, replication=0.25, format=0.20, bit_rot=0.15, staleness=0.05
    #[must_use]
    pub fn composite(&self) -> f64 {
        self.fixity * 0.35
            + self.replication * 0.25
            + self.format_health * 0.20
            + self.bit_rot_health * 0.15
            + self.staleness * 0.05
    }
}

/// An actionable recommendation generated by the dashboard.
#[derive(Debug, Clone)]
pub struct DashboardRecommendation {
    /// Short identifier/code for this recommendation.
    pub code: String,
    /// Human-readable description.
    pub description: String,
    /// Priority (1 = highest).
    pub priority: u8,
    /// Object IDs affected.
    pub affected_objects: Vec<String>,
}

impl DashboardRecommendation {
    /// Creates a new recommendation.
    #[must_use]
    pub fn new(
        code: impl Into<String>,
        description: impl Into<String>,
        priority: u8,
    ) -> Self {
        Self {
            code: code.into(),
            description: description.into(),
            priority,
            affected_objects: Vec::new(),
        }
    }

    /// Adds an affected object ID.
    pub fn with_object(mut self, id: impl Into<String>) -> Self {
        self.affected_objects.push(id.into());
        self
    }
}

/// Complete integrity report produced by the dashboard.
#[derive(Debug, Clone)]
pub struct IntegrityReport {
    /// When this report was generated.
    pub generated_at: SystemTime,
    /// Total number of objects assessed.
    pub total_objects: u32,
    /// Per-dimension scores.
    pub dimension_scores: DimensionScores,
    /// Composite integrity score (0.0–1.0).
    pub composite_score: f64,
    /// Overall health tier.
    pub health_tier: HealthTier,
    /// Actionable recommendations, sorted by priority.
    pub recommendations: Vec<DashboardRecommendation>,
    /// Object IDs with critical issues.
    pub critical_objects: Vec<String>,
    /// Object IDs with fixity failures.
    pub fixity_failures: Vec<String>,
    /// Object IDs below replication threshold.
    pub under_replicated: Vec<String>,
    /// Object IDs affected by detected bit-rot.
    pub bit_rot_affected: Vec<String>,
}

impl IntegrityReport {
    /// Returns true if the archive is in a healthy state.
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        self.health_tier == HealthTier::Healthy
    }

    /// Returns the number of recommendations by priority.
    #[must_use]
    pub fn recommendation_count_by_priority(&self, priority: u8) -> usize {
        self.recommendations
            .iter()
            .filter(|r| r.priority == priority)
            .count()
    }
}

/// The primary dashboard aggregator.
#[derive(Debug, Clone, Default)]
pub struct IntegrityDashboard {
    /// Fixity check records.
    fixity_records: Vec<FixityCheckRecord>,
    /// Replication status per object.
    replication_map: HashMap<String, ReplicationStatus>,
    /// Format obsolescence risk per object.
    format_risks: HashMap<String, ObsolescenceRisk>,
    /// Bit-rot events.
    bit_rot_events: Vec<BitRotEvent>,
    /// Maximum age of a fixity check before it is considered stale.
    max_check_age: Duration,
}

impl IntegrityDashboard {
    /// Creates a new dashboard with a 30-day staleness threshold.
    #[must_use]
    pub fn new() -> Self {
        Self {
            fixity_records: Vec::new(),
            replication_map: HashMap::new(),
            format_risks: HashMap::new(),
            bit_rot_events: Vec::new(),
            max_check_age: Duration::from_secs(30 * 24 * 3600),
        }
    }

    /// Sets the maximum age for fixity checks before they are marked stale.
    pub fn set_max_check_age(&mut self, age: Duration) {
        self.max_check_age = age;
    }

    /// Records a fixity check result.
    pub fn record_fixity(&mut self, record: FixityCheckRecord) {
        self.fixity_records.push(record);
    }

    /// Sets the replication status for an object.
    pub fn set_replication(&mut self, status: ReplicationStatus) {
        self.replication_map.insert(status.object_id.clone(), status);
    }

    /// Sets the format obsolescence risk for an object.
    pub fn set_format_risk(&mut self, object_id: impl Into<String>, risk: ObsolescenceRisk) {
        self.format_risks.insert(object_id.into(), risk);
    }

    /// Records a bit-rot event.
    pub fn record_bit_rot(&mut self, event: BitRotEvent) {
        self.bit_rot_events.push(event);
    }

    /// Generates a comprehensive integrity report.
    #[must_use]
    pub fn generate_report(&self) -> IntegrityReport {
        let now = SystemTime::now();

        // --- Fixity dimension ---
        let total_fixity = self.fixity_records.len();
        let passed_fixity = self.fixity_records.iter().filter(|r| r.passed).count();
        let fixity_score = if total_fixity == 0 {
            0.5 // Unknown → neutral
        } else {
            passed_fixity as f64 / total_fixity as f64
        };
        let fixity_failures: Vec<String> = self
            .fixity_records
            .iter()
            .filter(|r| !r.passed)
            .map(|r| r.object_id.clone())
            .collect();

        // --- Replication dimension ---
        let total_rep = self.replication_map.len();
        let met_rep = self
            .replication_map
            .values()
            .filter(|s| s.meets_requirement())
            .count();
        let replication_score = if total_rep == 0 {
            0.5
        } else {
            met_rep as f64 / total_rep as f64
        };
        let under_replicated: Vec<String> = self
            .replication_map
            .values()
            .filter(|s| !s.meets_requirement())
            .map(|s| s.object_id.clone())
            .collect();

        // --- Format health dimension ---
        let total_fmt = self.format_risks.len();
        let format_score = if total_fmt == 0 {
            1.0
        } else {
            let total_penalty: f64 = self.format_risks.values().map(|r| r.penalty()).sum();
            let avg_penalty = total_penalty / total_fmt as f64;
            (1.0 - avg_penalty).max(0.0)
        };

        // --- Bit-rot dimension ---
        let unique_affected: std::collections::HashSet<&str> = self
            .bit_rot_events
            .iter()
            .filter(|e| !e.recovered)
            .map(|e| e.object_id.as_str())
            .collect();
        let bit_rot_affected: Vec<String> = unique_affected
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        let total_monitored = total_fixity.max(total_rep).max(total_fmt).max(1);
        let bit_rot_score = if bit_rot_affected.is_empty() {
            1.0
        } else {
            let ratio = bit_rot_affected.len() as f64 / total_monitored as f64;
            (1.0 - ratio * 3.0).max(0.0) // 3x penalty for bit-rot
        };

        // --- Staleness dimension ---
        let stale_count = self
            .fixity_records
            .iter()
            .filter(|r| {
                now.duration_since(r.checked_at)
                    .map(|d| d > self.max_check_age)
                    .unwrap_or(false)
            })
            .count();
        let staleness_score = if total_fixity == 0 {
            0.5
        } else {
            let stale_ratio = stale_count as f64 / total_fixity as f64;
            (1.0 - stale_ratio).max(0.0)
        };

        let dimension_scores = DimensionScores {
            fixity: fixity_score,
            replication: replication_score,
            format_health: format_score,
            bit_rot_health: bit_rot_score,
            staleness: staleness_score,
        };
        let composite_score = dimension_scores.composite();
        let health_tier = HealthTier::from_score(composite_score);

        // --- Collect critical objects ---
        let mut critical_set = std::collections::HashSet::new();
        for id in &fixity_failures {
            critical_set.insert(id.clone());
        }
        for id in &bit_rot_affected {
            critical_set.insert(id.clone());
        }
        let critical_objects: Vec<String> = critical_set.into_iter().collect();

        // --- Build recommendations ---
        let mut recs: Vec<DashboardRecommendation> = Vec::new();

        if !fixity_failures.is_empty() {
            let mut r = DashboardRecommendation::new(
                "FIXITY_FAIL",
                format!(
                    "{} object(s) failed fixity checks — investigate and restore from replica",
                    fixity_failures.len()
                ),
                1,
            );
            for id in &fixity_failures {
                r = r.with_object(id);
            }
            recs.push(r);
        }

        if !bit_rot_affected.is_empty() {
            let mut r = DashboardRecommendation::new(
                "BIT_ROT",
                format!(
                    "{} object(s) show unrecovered bit-rot — restore immediately",
                    bit_rot_affected.len()
                ),
                1,
            );
            for id in &bit_rot_affected {
                r = r.with_object(id);
            }
            recs.push(r);
        }

        if !under_replicated.is_empty() {
            let mut r = DashboardRecommendation::new(
                "UNDER_REP",
                format!(
                    "{} object(s) below replication threshold — create additional copies",
                    under_replicated.len()
                ),
                2,
            );
            for id in &under_replicated {
                r = r.with_object(id);
            }
            recs.push(r);
        }

        let high_risk_formats: Vec<String> = self
            .format_risks
            .iter()
            .filter(|(_, r)| **r >= ObsolescenceRisk::High)
            .map(|(id, _)| id.clone())
            .collect();
        if !high_risk_formats.is_empty() {
            let mut r = DashboardRecommendation::new(
                "FORMAT_RISK",
                format!(
                    "{} object(s) use high-risk formats — plan migration",
                    high_risk_formats.len()
                ),
                2,
            );
            for id in &high_risk_formats {
                r = r.with_object(id);
            }
            recs.push(r);
        }

        if stale_count > 0 {
            recs.push(DashboardRecommendation::new(
                "STALE_FIXITY",
                format!("{stale_count} fixity check(s) are overdue — schedule new checks"),
                3,
            ));
        }

        // Sort by priority ascending
        recs.sort_by_key(|r| r.priority);

        IntegrityReport {
            generated_at: now,
            total_objects: total_monitored as u32,
            dimension_scores,
            composite_score,
            health_tier,
            recommendations: recs,
            critical_objects,
            fixity_failures,
            under_replicated,
            bit_rot_affected,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn make_rep(id: &str, confirmed: u32, required: u32) -> ReplicationStatus {
        let mut s = ReplicationStatus::new(id, required);
        for i in 0..confirmed {
            s.add_copy(format!("loc-{i}"));
        }
        s
    }

    #[test]
    fn test_health_tier_from_score() {
        assert_eq!(HealthTier::from_score(1.0), HealthTier::Healthy);
        assert_eq!(HealthTier::from_score(0.90), HealthTier::Healthy);
        assert_eq!(HealthTier::from_score(0.85), HealthTier::Caution);
        assert_eq!(HealthTier::from_score(0.70), HealthTier::Caution);
        assert_eq!(HealthTier::from_score(0.65), HealthTier::Warning);
        assert_eq!(HealthTier::from_score(0.39), HealthTier::Critical);
    }

    #[test]
    fn test_health_tier_ordering() {
        assert!(HealthTier::Critical < HealthTier::Warning);
        assert!(HealthTier::Warning < HealthTier::Caution);
        assert!(HealthTier::Caution < HealthTier::Healthy);
    }

    #[test]
    fn test_health_tier_labels() {
        assert!(!HealthTier::Critical.label().is_empty());
        assert!(!HealthTier::Healthy.color_label().is_empty());
    }

    #[test]
    fn test_replication_status_meets_requirement() {
        let mut s = ReplicationStatus::new("obj-1", 3);
        assert!(!s.meets_requirement());
        assert_eq!(s.copy_deficit(), 3);
        s.add_copy("us-east");
        s.add_copy("eu-west");
        assert!(!s.meets_requirement());
        s.add_copy("ap-south");
        assert!(s.meets_requirement());
        assert_eq!(s.copy_deficit(), 0);
    }

    #[test]
    fn test_obsolescence_risk_penalty() {
        assert_eq!(ObsolescenceRisk::Low.penalty(), 0.0);
        assert!(ObsolescenceRisk::Medium.penalty() > 0.0);
        assert!(ObsolescenceRisk::High.penalty() > ObsolescenceRisk::Medium.penalty());
        assert!(ObsolescenceRisk::Critical.penalty() > ObsolescenceRisk::High.penalty());
    }

    #[test]
    fn test_dimension_scores_composite() {
        let s = DimensionScores {
            fixity: 1.0,
            replication: 1.0,
            format_health: 1.0,
            bit_rot_health: 1.0,
            staleness: 1.0,
        };
        assert!((s.composite() - 1.0).abs() < f64::EPSILON);

        let s2 = DimensionScores::default();
        assert_eq!(s2.composite(), 0.0);
    }

    #[test]
    fn test_empty_dashboard_report() {
        let dashboard = IntegrityDashboard::new();
        let report = dashboard.generate_report();
        // With no data, composite is 0.5*0.35 + 0.5*0.25 + 1.0*0.20 + 1.0*0.15 + 0.5*0.05
        // = 0.175 + 0.125 + 0.20 + 0.15 + 0.025 = 0.675
        assert!(report.composite_score > 0.0);
        assert!(report.recommendations.is_empty());
    }

    #[test]
    fn test_dashboard_with_all_passing_fixity() {
        let mut dashboard = IntegrityDashboard::new();
        for i in 0..5 {
            dashboard.record_fixity(FixityCheckRecord::passed(format!("obj-{i}")));
        }
        let report = dashboard.generate_report();
        assert_eq!(report.dimension_scores.fixity, 1.0);
        assert!(report.fixity_failures.is_empty());
    }

    #[test]
    fn test_dashboard_with_fixity_failures() {
        let mut dashboard = IntegrityDashboard::new();
        dashboard.record_fixity(FixityCheckRecord::passed("obj-1"));
        dashboard.record_fixity(FixityCheckRecord::failed("obj-2", "hash mismatch"));
        dashboard.record_fixity(FixityCheckRecord::failed("obj-3", "file missing"));

        let report = dashboard.generate_report();
        assert!((report.dimension_scores.fixity - 1.0 / 3.0).abs() < 1e-6);
        assert_eq!(report.fixity_failures.len(), 2);
        assert!(report
            .recommendations
            .iter()
            .any(|r| r.code == "FIXITY_FAIL"));
        assert_eq!(report.recommendation_count_by_priority(1), 1);
    }

    #[test]
    fn test_dashboard_replication_under_threshold() {
        let mut dashboard = IntegrityDashboard::new();
        dashboard.record_fixity(FixityCheckRecord::passed("obj-1"));
        dashboard.set_replication(make_rep("obj-1", 1, 3));
        dashboard.set_replication(make_rep("obj-2", 3, 3));

        let report = dashboard.generate_report();
        assert_eq!(report.under_replicated.len(), 1);
        assert!(report
            .recommendations
            .iter()
            .any(|r| r.code == "UNDER_REP"));
    }

    #[test]
    fn test_dashboard_format_risk() {
        let mut dashboard = IntegrityDashboard::new();
        dashboard.record_fixity(FixityCheckRecord::passed("obj-1"));
        dashboard.set_format_risk("obj-1", ObsolescenceRisk::Critical);

        let report = dashboard.generate_report();
        assert!(report.dimension_scores.format_health < 1.0);
        assert!(report
            .recommendations
            .iter()
            .any(|r| r.code == "FORMAT_RISK"));
    }

    #[test]
    fn test_dashboard_bit_rot_event() {
        let mut dashboard = IntegrityDashboard::new();
        dashboard.record_fixity(FixityCheckRecord::passed("obj-1"));
        dashboard.record_fixity(FixityCheckRecord::passed("obj-2"));
        dashboard.record_bit_rot(BitRotEvent {
            object_id: "obj-1".to_string(),
            detected_at: SystemTime::now(),
            byte_offset: Some(4096),
            recovered: false,
        });

        let report = dashboard.generate_report();
        assert_eq!(report.bit_rot_affected.len(), 1);
        assert!(report
            .recommendations
            .iter()
            .any(|r| r.code == "BIT_ROT"));
    }

    #[test]
    fn test_dashboard_stale_fixity() {
        let mut dashboard = IntegrityDashboard::new();
        dashboard.set_max_check_age(Duration::from_secs(1)); // 1 second threshold
        let old_record = FixityCheckRecord {
            object_id: "obj-1".to_string(),
            checked_at: SystemTime::UNIX_EPOCH, // Very old
            passed: true,
            error_message: None,
        };
        dashboard.record_fixity(old_record);

        let report = dashboard.generate_report();
        assert!(report
            .recommendations
            .iter()
            .any(|r| r.code == "STALE_FIXITY"));
    }

    #[test]
    fn test_report_is_healthy_when_all_good() {
        let mut dashboard = IntegrityDashboard::new();
        for i in 0..10 {
            dashboard.record_fixity(FixityCheckRecord::passed(format!("obj-{i}")));
            let mut s = make_rep(&format!("obj-{i}"), 3, 3);
            s.add_copy(format!("extra-{i}"));
            dashboard.set_replication(s);
            dashboard.set_format_risk(format!("obj-{i}"), ObsolescenceRisk::Low);
        }
        let report = dashboard.generate_report();
        assert!(report.is_healthy(), "score={}", report.composite_score);
        assert!(report.recommendations.is_empty());
    }

    #[test]
    fn test_recommendation_sorting_by_priority() {
        let mut dashboard = IntegrityDashboard::new();
        // Create both a fixity failure (priority 1) and replication issue (priority 2)
        dashboard.record_fixity(FixityCheckRecord::failed("obj-1", "err"));
        dashboard.set_replication(make_rep("obj-2", 0, 3));

        let report = dashboard.generate_report();
        if report.recommendations.len() >= 2 {
            assert!(
                report.recommendations[0].priority <= report.recommendations[1].priority,
                "recommendations should be sorted ascending by priority"
            );
        }
    }
}
