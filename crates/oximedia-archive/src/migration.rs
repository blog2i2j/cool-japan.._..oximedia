//! Format migration planning for long-term digital preservation.
//!
//! Assesses file format risk, recommends migration paths, and estimates
//! the cost and time of migration batches.

#![allow(dead_code)]

/// NDSA / Library of Congress format-risk classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum FormatRisk {
    /// Preferred: widely supported, openly specified, actively maintained.
    Preferred,
    /// Stable: well-documented, no known risks in the near term.
    Stable,
    /// At Risk: declining tool support or unclear long-term viability.
    AtRisk,
    /// Endangered: few remaining tools; migration recommended soon.
    Endangered,
    /// Obsolete: tools no longer maintained; migration urgent.
    Obsolete,
}

impl FormatRisk {
    /// Migration priority (higher = more urgent).
    #[must_use]
    pub const fn priority(&self) -> u8 {
        match self {
            Self::Preferred => 0,
            Self::Stable => 1,
            Self::AtRisk => 2,
            Self::Endangered => 3,
            Self::Obsolete => 4,
        }
    }

    /// Human-readable label.
    #[must_use]
    pub const fn label(&self) -> &str {
        match self {
            Self::Preferred => "preferred",
            Self::Stable => "stable",
            Self::AtRisk => "at_risk",
            Self::Endangered => "endangered",
            Self::Obsolete => "obsolete",
        }
    }
}

/// A file format descriptor.
#[derive(Clone, Debug)]
pub struct FileFormat {
    /// Short name of the format (e.g., `"DPX"`).
    pub name: String,
    /// Canonical file extension without the leading dot (e.g., `"dpx"`).
    pub extension: String,
    /// MIME type (e.g., `"image/x-dpx"`).
    pub mime_type: String,
    /// Current risk status.
    pub risk: FormatRisk,
}

impl FileFormat {
    /// Create a new file format descriptor.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        extension: impl Into<String>,
        mime_type: impl Into<String>,
        risk: FormatRisk,
    ) -> Self {
        Self {
            name: name.into(),
            extension: extension.into(),
            mime_type: mime_type.into(),
            risk,
        }
    }

    // ── Common format presets ─────────────────────────────────────────────────

    /// SMPTE DPX (cinema standard, preferred for image sequences).
    #[must_use]
    pub fn dpx() -> Self {
        Self::new("DPX", "dpx", "image/x-dpx", FormatRisk::Preferred)
    }

    /// Apple ProRes 4444 (widely used, good tool support).
    #[must_use]
    pub fn prores_4444() -> Self {
        Self::new("ProRes 4444", "mov", "video/quicktime", FormatRisk::Stable)
    }

    /// Avid DNxHD (professional NLE format, good support).
    #[must_use]
    pub fn avid_dnxhd() -> Self {
        Self::new("Avid DNxHD", "mxf", "application/mxf", FormatRisk::Stable)
    }

    /// H.264 in MP4 container (delivery format; lossy).
    #[must_use]
    pub fn h264_mp4() -> Self {
        Self::new("H.264/MP4", "mp4", "video/mp4", FormatRisk::Stable)
    }

    /// MPEG-2 (older broadcast standard; declining support).
    #[must_use]
    pub fn mpeg2() -> Self {
        Self::new("MPEG-2", "mpg", "video/mpeg", FormatRisk::AtRisk)
    }

    /// DV (consumer/prosumer tape format; obsolescence risk).
    #[must_use]
    pub fn dv() -> Self {
        Self::new("DV", "dv", "video/x-dv", FormatRisk::Endangered)
    }

    /// Betacam SP (broadcast videotape; hardware-dependent).
    #[must_use]
    pub fn betacam() -> Self {
        Self::new(
            "Betacam SP",
            "betacam",
            "application/octet-stream",
            FormatRisk::Obsolete,
        )
    }
}

/// Describes a recommended migration from one format to another.
#[derive(Clone, Debug)]
pub struct MigrationPath {
    /// Source format.
    pub source: FileFormat,
    /// Recommended target format.
    pub target: FileFormat,
    /// Whether the migration involves quality loss.
    pub quality_loss: bool,
    /// Whether the original format can be recovered from the migrated file.
    pub reversible: bool,
    /// Human-readable notes about the migration.
    pub notes: String,
}

/// Plans format migrations based on format risk.
pub struct MigrationPlanner;

impl MigrationPlanner {
    /// Return a recommended migration path for `source`, or `None` if no
    /// migration is needed (format is `Preferred` or `Stable`).
    #[must_use]
    pub fn plan_migration(source: &FileFormat) -> Option<MigrationPath> {
        match source.risk {
            FormatRisk::Preferred | FormatRisk::Stable => None,
            FormatRisk::AtRisk => Some(MigrationPath {
                source: source.clone(),
                target: FileFormat::prores_4444(),
                quality_loss: false,
                reversible: false,
                notes: "Migrate to ProRes 4444 for better long-term tool support.".to_string(),
            }),
            FormatRisk::Endangered => Some(MigrationPath {
                source: source.clone(),
                target: FileFormat::dpx(),
                quality_loss: false,
                reversible: false,
                notes: "Urgent: migrate to DPX image sequence for lossless preservation."
                    .to_string(),
            }),
            FormatRisk::Obsolete => Some(MigrationPath {
                source: source.clone(),
                target: FileFormat::dpx(),
                quality_loss: false,
                reversible: false,
                notes: "Critical: format is obsolete — immediate migration to DPX required."
                    .to_string(),
            }),
        }
    }

    /// Estimate migration cost (CPU-hours per GB) for the given format.
    ///
    /// Returns a rough multiplier: higher for formats requiring complex decoding.
    #[must_use]
    pub fn estimate_cost_gb(format: &FileFormat, size_gb: f64) -> f64 {
        let cpu_hours_per_gb = match format.risk {
            FormatRisk::Preferred => 0.1,
            FormatRisk::Stable => 0.2,
            FormatRisk::AtRisk => 0.5,
            FormatRisk::Endangered => 1.5,
            FormatRisk::Obsolete => 4.0,
        };
        cpu_hours_per_gb * size_gb
    }
}

/// A planned batch of migrations.
#[derive(Clone, Debug)]
pub struct MigrationBatch {
    /// List of `(asset_id, migration_path)` pairs.
    pub items: Vec<(String, MigrationPath)>,
    /// Total size of all assets in GB.
    pub total_size_gb: f64,
    /// Estimated processing time in hours.
    pub estimated_hours: f64,
}

impl MigrationBatch {
    /// Create an empty migration batch.
    #[must_use]
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            total_size_gb: 0.0,
            estimated_hours: 0.0,
        }
    }

    /// Add an item to the batch.
    pub fn add(&mut self, asset_id: impl Into<String>, path: MigrationPath, size_gb: f64) {
        let cost = MigrationPlanner::estimate_cost_gb(&path.source, size_gb);
        self.estimated_hours += cost;
        self.total_size_gb += size_gb;
        self.items.push((asset_id.into(), path));
    }

    /// Number of assets in this batch.
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Returns `true` if the batch is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

impl Default for MigrationBatch {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_risk_priority_ordering() {
        assert!(FormatRisk::Obsolete.priority() > FormatRisk::Preferred.priority());
        assert!(FormatRisk::Endangered.priority() > FormatRisk::AtRisk.priority());
    }

    #[test]
    fn test_format_risk_labels_non_empty() {
        for risk in [
            FormatRisk::Preferred,
            FormatRisk::Stable,
            FormatRisk::AtRisk,
            FormatRisk::Endangered,
            FormatRisk::Obsolete,
        ] {
            assert!(!risk.label().is_empty());
        }
    }

    #[test]
    fn test_dpx_is_preferred() {
        assert_eq!(FileFormat::dpx().risk, FormatRisk::Preferred);
    }

    #[test]
    fn test_betacam_is_obsolete() {
        assert_eq!(FileFormat::betacam().risk, FormatRisk::Obsolete);
    }

    #[test]
    fn test_dv_is_endangered() {
        assert_eq!(FileFormat::dv().risk, FormatRisk::Endangered);
    }

    #[test]
    fn test_mpeg2_is_at_risk() {
        assert_eq!(FileFormat::mpeg2().risk, FormatRisk::AtRisk);
    }

    #[test]
    fn test_plan_migration_preferred_none() {
        let dpx = FileFormat::dpx();
        assert!(MigrationPlanner::plan_migration(&dpx).is_none());
    }

    #[test]
    fn test_plan_migration_at_risk_some() {
        let mpeg2 = FileFormat::mpeg2();
        let path = MigrationPlanner::plan_migration(&mpeg2);
        assert!(path.is_some());
        let path = path.expect("path should be valid");
        assert_eq!(path.target.risk, FormatRisk::Stable);
    }

    #[test]
    fn test_plan_migration_obsolete_to_dpx() {
        let betacam = FileFormat::betacam();
        let path = MigrationPlanner::plan_migration(&betacam).expect("path should be valid");
        assert_eq!(path.target.extension, "dpx");
    }

    #[test]
    fn test_estimate_cost_gb_preferred_cheap() {
        let dpx = FileFormat::dpx();
        let cost = MigrationPlanner::estimate_cost_gb(&dpx, 10.0);
        // 0.1 cpu-hours/GB * 10 GB = 1.0 — cheaper than at-risk formats
        assert!(cost <= 1.0, "cost = {cost}");
    }

    #[test]
    fn test_estimate_cost_gb_obsolete_expensive() {
        let betacam = FileFormat::betacam();
        let cost = MigrationPlanner::estimate_cost_gb(&betacam, 10.0);
        assert!(cost > 10.0, "cost = {cost}");
    }

    #[test]
    fn test_migration_batch_add() {
        let mut batch = MigrationBatch::new();
        let dv = FileFormat::dv();
        let path = MigrationPlanner::plan_migration(&dv).expect("path should be valid");
        batch.add("asset-001", path, 5.0);
        assert_eq!(batch.len(), 1);
        assert!((batch.total_size_gb - 5.0).abs() < 1e-9);
        assert!(batch.estimated_hours > 0.0);
    }

    #[test]
    fn test_migration_batch_is_empty() {
        let batch = MigrationBatch::new();
        assert!(batch.is_empty());
    }
}

// ── New migration-planning types ──────────────────────────────────────────────

/// Risk level of a migration operation.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MigrationRisk {
    /// Routine, well-understood migration.
    Low,
    /// Some complexity; proceed with review.
    Medium,
    /// Significant risk; expert sign-off required.
    High,
    /// Mission-critical; full risk management plan needed.
    Critical,
}

impl MigrationRisk {
    /// Numeric score (higher = more risky).
    #[must_use]
    pub const fn score(self) -> u32 {
        match self {
            Self::Low => 1,
            Self::Medium => 2,
            Self::High => 3,
            Self::Critical => 4,
        }
    }
}

/// Degree to which a given format is supported by current tools.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FormatSupport {
    /// Actively maintained; no migration needed.
    FullySupported,
    /// Works but not recommended for new projects.
    LegacySupport,
    /// Will be removed in a future tool release.
    DeprecatedSoon,
    /// Tools no longer handle this format.
    Unsupported,
}

impl FormatSupport {
    /// Derive the migration risk implied by the support level.
    #[must_use]
    pub const fn migration_risk(self) -> MigrationRisk {
        match self {
            Self::FullySupported => MigrationRisk::Low,
            Self::LegacySupport => MigrationRisk::Medium,
            Self::DeprecatedSoon => MigrationRisk::High,
            Self::Unsupported => MigrationRisk::Critical,
        }
    }
}

/// Describes a single migration job (source → target format).
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct MigrationTask {
    /// Source format identifier.
    pub source_format: String,
    /// Target format identifier.
    pub target_format: String,
    /// Number of files to migrate.
    pub file_count: u32,
    /// Estimated time in hours.
    pub estimated_hours: f32,
}

impl MigrationTask {
    /// Returns `true` if the task covers more than 1 000 files.
    #[must_use]
    pub fn is_large(&self) -> bool {
        self.file_count > 1_000
    }
}

/// A collection of migration tasks forming a coherent plan.
#[allow(dead_code)]
#[derive(Default, Debug)]
pub struct MigrationPlan {
    /// Individual tasks.
    pub tasks: Vec<MigrationTask>,
    /// Free-text description.
    pub description: String,
}

impl MigrationPlan {
    /// Create an empty plan.
    #[must_use]
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            tasks: Vec::new(),
            description: description.into(),
        }
    }

    /// Append a task to the plan.
    pub fn add(&mut self, task: MigrationTask) {
        self.tasks.push(task);
    }

    /// Total number of files across all tasks.
    #[must_use]
    pub fn total_files(&self) -> u32 {
        self.tasks.iter().map(|t| t.file_count).sum()
    }

    /// Sum of estimated hours across all tasks.
    #[must_use]
    pub fn total_hours(&self) -> f32 {
        self.tasks.iter().map(|t| t.estimated_hours).sum()
    }

    /// Tasks where the source format has `High` or `Critical` support risk.
    ///
    /// Heuristic: more than 1 000 files OR estimated > 24 h counts as high-risk.
    #[must_use]
    pub fn high_risk_tasks(&self) -> Vec<&MigrationTask> {
        self.tasks
            .iter()
            .filter(|t| t.is_large() || t.estimated_hours > 24.0)
            .collect()
    }
}

#[cfg(test)]
mod migration_plan_tests {
    use super::*;

    fn small_task(src: &str, tgt: &str, n: u32, h: f32) -> MigrationTask {
        MigrationTask {
            source_format: src.to_string(),
            target_format: tgt.to_string(),
            file_count: n,
            estimated_hours: h,
        }
    }

    #[test]
    fn test_migration_risk_score_ordering() {
        assert!(MigrationRisk::Critical.score() > MigrationRisk::High.score());
        assert!(MigrationRisk::High.score() > MigrationRisk::Medium.score());
        assert!(MigrationRisk::Medium.score() > MigrationRisk::Low.score());
    }

    #[test]
    fn test_migration_risk_score_values() {
        assert_eq!(MigrationRisk::Low.score(), 1);
        assert_eq!(MigrationRisk::Critical.score(), 4);
    }

    #[test]
    fn test_format_support_fully_supported_low_risk() {
        assert_eq!(
            FormatSupport::FullySupported.migration_risk(),
            MigrationRisk::Low
        );
    }

    #[test]
    fn test_format_support_unsupported_critical_risk() {
        assert_eq!(
            FormatSupport::Unsupported.migration_risk(),
            MigrationRisk::Critical
        );
    }

    #[test]
    fn test_format_support_deprecated_high_risk() {
        assert_eq!(
            FormatSupport::DeprecatedSoon.migration_risk(),
            MigrationRisk::High
        );
    }

    #[test]
    fn test_migration_task_is_large_false() {
        let t = small_task("DV", "DPX", 500, 5.0);
        assert!(!t.is_large());
    }

    #[test]
    fn test_migration_task_is_large_true() {
        let t = small_task("DV", "DPX", 2_000, 40.0);
        assert!(t.is_large());
    }

    #[test]
    fn test_migration_plan_total_files() {
        let mut plan = MigrationPlan::new("Q1 migration");
        plan.add(small_task("A", "B", 100, 2.0));
        plan.add(small_task("C", "D", 200, 4.0));
        assert_eq!(plan.total_files(), 300);
    }

    #[test]
    fn test_migration_plan_total_hours() {
        let mut plan = MigrationPlan::new("test");
        plan.add(small_task("A", "B", 10, 1.5));
        plan.add(small_task("C", "D", 10, 2.5));
        assert!((plan.total_hours() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_migration_plan_high_risk_by_file_count() {
        let mut plan = MigrationPlan::new("big batch");
        plan.add(small_task("DV", "DPX", 5_000, 10.0));
        plan.add(small_task("MP4", "MOV", 50, 1.0));
        let hr = plan.high_risk_tasks();
        assert_eq!(hr.len(), 1);
        assert_eq!(hr[0].source_format, "DV");
    }

    #[test]
    fn test_migration_plan_high_risk_by_hours() {
        let mut plan = MigrationPlan::new("long job");
        plan.add(small_task("old", "new", 100, 30.0));
        assert_eq!(plan.high_risk_tasks().len(), 1);
    }

    #[test]
    fn test_migration_plan_no_high_risk() {
        let mut plan = MigrationPlan::new("easy");
        plan.add(small_task("A", "B", 50, 2.0));
        assert!(plan.high_risk_tasks().is_empty());
    }
}
