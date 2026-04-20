//! Job migration: upgrade job schemas when template format changes.
//!
//! When batch job templates evolve over time (new fields, renamed keys, changed
//! defaults), existing serialised jobs in the queue or database need to be
//! transformed to match the new schema.  This module provides:
//!
//! - **SchemaVersion**: semantic version tracking for job schemas.
//! - **MigrationStep**: a single transformation from one version to the next.
//! - **MigrationChain**: an ordered list of steps that can upgrade a job from
//!   any supported version to the latest.
//! - **JobMigrator**: the entry point that detects a job's version and applies
//!   the necessary migrations.

#![allow(dead_code)]

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::{BatchError, Result};

// ---------------------------------------------------------------------------
// Schema version
// ---------------------------------------------------------------------------

/// A semantic version for job schemas.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SchemaVersion {
    /// Major version (breaking changes).
    pub major: u32,
    /// Minor version (backwards-compatible additions).
    pub minor: u32,
    /// Patch version (bug fixes in schema).
    pub patch: u32,
}

impl SchemaVersion {
    /// Create a new version.
    #[must_use]
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Whether this version is strictly older than `other`.
    #[must_use]
    pub fn is_older_than(&self, other: &Self) -> bool {
        self.to_tuple() < other.to_tuple()
    }

    /// Whether this version is the same or newer than `other`.
    #[must_use]
    pub fn is_at_least(&self, other: &Self) -> bool {
        self.to_tuple() >= other.to_tuple()
    }

    fn to_tuple(&self) -> (u32, u32, u32) {
        (self.major, self.minor, self.patch)
    }
}

impl std::fmt::Display for SchemaVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl PartialOrd for SchemaVersion {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SchemaVersion {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.to_tuple().cmp(&other.to_tuple())
    }
}

// ---------------------------------------------------------------------------
// Migration step
// ---------------------------------------------------------------------------

/// The kind of transformation applied by a migration step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MigrationAction {
    /// Add a new field with a default value.
    AddField {
        /// Field name (dot-separated path for nested fields).
        field: String,
        /// Default value as a JSON string.
        default_value: String,
    },
    /// Remove a field.
    RemoveField {
        /// Field name.
        field: String,
    },
    /// Rename a field.
    RenameField {
        /// Old field name.
        from: String,
        /// New field name.
        to: String,
    },
    /// Change the type/format of a field value.
    TransformField {
        /// Field name.
        field: String,
        /// Description of the transformation.
        description: String,
    },
    /// Set a field to a new value unconditionally.
    SetValue {
        /// Field name.
        field: String,
        /// New value as a JSON string.
        value: String,
    },
    /// A custom migration with a description.
    Custom {
        /// Human-readable description.
        description: String,
    },
}

impl std::fmt::Display for MigrationAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AddField { field, .. } => write!(f, "add_field({field})"),
            Self::RemoveField { field } => write!(f, "remove_field({field})"),
            Self::RenameField { from, to } => write!(f, "rename_field({from}->{to})"),
            Self::TransformField { field, .. } => write!(f, "transform_field({field})"),
            Self::SetValue { field, .. } => write!(f, "set_value({field})"),
            Self::Custom { description } => write!(f, "custom({description})"),
        }
    }
}

/// A single migration step from one version to the next.
#[derive(Debug, Clone)]
pub struct MigrationStep {
    /// The version this step migrates FROM.
    pub from_version: SchemaVersion,
    /// The version this step migrates TO.
    pub to_version: SchemaVersion,
    /// Human-readable description.
    pub description: String,
    /// Actions to apply in order.
    pub actions: Vec<MigrationAction>,
    /// Whether this migration is reversible.
    pub reversible: bool,
}

impl MigrationStep {
    /// Create a new migration step.
    #[must_use]
    pub fn new(from: SchemaVersion, to: SchemaVersion, description: impl Into<String>) -> Self {
        Self {
            from_version: from,
            to_version: to,
            description: description.into(),
            actions: Vec::new(),
            reversible: false,
        }
    }

    /// Add an action to this step.
    #[must_use]
    pub fn with_action(mut self, action: MigrationAction) -> Self {
        self.actions.push(action);
        self
    }

    /// Mark as reversible.
    #[must_use]
    pub fn reversible(mut self) -> Self {
        self.reversible = true;
        self
    }
}

// ---------------------------------------------------------------------------
// Migration chain
// ---------------------------------------------------------------------------

/// An ordered list of migration steps.
#[derive(Debug, Clone)]
pub struct MigrationChain {
    steps: Vec<MigrationStep>,
}

impl MigrationChain {
    /// Create a new empty chain.
    #[must_use]
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Add a step to the chain.
    pub fn add_step(&mut self, step: MigrationStep) {
        self.steps.push(step);
        // Keep sorted by from_version.
        self.steps
            .sort_by(|a, b| a.from_version.cmp(&b.from_version));
    }

    /// Number of steps in the chain.
    #[must_use]
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Whether the chain is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Get the latest version the chain can migrate to.
    #[must_use]
    pub fn latest_version(&self) -> Option<SchemaVersion> {
        self.steps.last().map(|s| s.to_version.clone())
    }

    /// Find the sequence of steps needed to migrate from `from` to `to`.
    ///
    /// Returns an empty vec if no migration is needed (versions are equal).
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::InvalidJobConfig`] if no migration path exists.
    pub fn find_path(
        &self,
        from: &SchemaVersion,
        to: &SchemaVersion,
    ) -> Result<Vec<&MigrationStep>> {
        if from == to {
            return Ok(Vec::new());
        }

        if from > to {
            return Err(BatchError::InvalidJobConfig(format!(
                "Cannot downgrade from {from} to {to}: downgrade not supported"
            )));
        }

        let mut path = Vec::new();
        let mut current = from.clone();

        while current < *to {
            let step = self
                .steps
                .iter()
                .find(|s| s.from_version == current)
                .ok_or_else(|| {
                    BatchError::InvalidJobConfig(format!(
                        "No migration path from {current} to {to}"
                    ))
                })?;

            path.push(step);
            current = step.to_version.clone();

            // Safety: prevent infinite loops.
            if path.len() > 1000 {
                return Err(BatchError::InvalidJobConfig(
                    "Migration path too long (possible loop)".to_string(),
                ));
            }
        }

        Ok(path)
    }

    /// Get all steps as a slice.
    #[must_use]
    pub fn steps(&self) -> &[MigrationStep] {
        &self.steps
    }
}

impl Default for MigrationChain {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Job data (generic representation)
// ---------------------------------------------------------------------------

/// A generic job data representation for migration purposes.
///
/// Jobs are represented as a flat map of string keys to JSON values,
/// making it easy to add/remove/rename fields during migration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobData {
    /// Schema version of this job data.
    pub schema_version: SchemaVersion,
    /// Key-value fields.
    pub fields: HashMap<String, serde_json::Value>,
}

impl JobData {
    /// Create a new job data with the given version.
    #[must_use]
    pub fn new(version: SchemaVersion) -> Self {
        Self {
            schema_version: version,
            fields: HashMap::new(),
        }
    }

    /// Set a field value.
    pub fn set(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.fields.insert(key.into(), value);
    }

    /// Get a field value.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&serde_json::Value> {
        self.fields.get(key)
    }

    /// Remove a field.
    pub fn remove(&mut self, key: &str) -> Option<serde_json::Value> {
        self.fields.remove(key)
    }

    /// Rename a field. Returns `true` if the field existed.
    pub fn rename(&mut self, from: &str, to: &str) -> bool {
        if let Some(val) = self.fields.remove(from) {
            self.fields.insert(to.to_string(), val);
            true
        } else {
            false
        }
    }

    /// Whether a field exists.
    #[must_use]
    pub fn has_field(&self, key: &str) -> bool {
        self.fields.contains_key(key)
    }

    /// Number of fields.
    #[must_use]
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }
}

// ---------------------------------------------------------------------------
// Job migrator
// ---------------------------------------------------------------------------

/// The main entry point for migrating job data.
#[derive(Debug)]
pub struct JobMigrator {
    chain: MigrationChain,
    target_version: SchemaVersion,
}

impl JobMigrator {
    /// Create a new migrator.
    #[must_use]
    pub fn new(chain: MigrationChain) -> Self {
        let target = chain
            .latest_version()
            .unwrap_or(SchemaVersion::new(1, 0, 0));
        Self {
            chain,
            target_version: target,
        }
    }

    /// Set a specific target version (default is latest in chain).
    #[must_use]
    pub fn with_target(mut self, version: SchemaVersion) -> Self {
        self.target_version = version;
        self
    }

    /// Migrate a job to the target version.
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::InvalidJobConfig`] if no migration path exists.
    pub fn migrate(&self, job: &mut JobData) -> Result<MigrationReport> {
        let path = self
            .chain
            .find_path(&job.schema_version, &self.target_version)?;

        if path.is_empty() {
            return Ok(MigrationReport {
                from_version: job.schema_version.clone(),
                to_version: job.schema_version.clone(),
                steps_applied: 0,
                actions_applied: Vec::new(),
            });
        }

        let from = job.schema_version.clone();
        let mut actions_applied = Vec::new();

        for step in &path {
            for action in &step.actions {
                self.apply_action(job, action)?;
                actions_applied.push(action.to_string());
            }
            job.schema_version = step.to_version.clone();
        }

        Ok(MigrationReport {
            from_version: from,
            to_version: job.schema_version.clone(),
            steps_applied: path.len(),
            actions_applied,
        })
    }

    /// Check if a job needs migration.
    #[must_use]
    pub fn needs_migration(&self, job: &JobData) -> bool {
        job.schema_version < self.target_version
    }

    /// Get the target version.
    #[must_use]
    pub fn target_version(&self) -> &SchemaVersion {
        &self.target_version
    }

    /// Get the migration chain.
    #[must_use]
    pub fn chain(&self) -> &MigrationChain {
        &self.chain
    }

    // ── Private ─────────────────────────────────────────────────────────

    fn apply_action(&self, job: &mut JobData, action: &MigrationAction) -> Result<()> {
        match action {
            MigrationAction::AddField {
                field,
                default_value,
            } => {
                if !job.has_field(field) {
                    let value: serde_json::Value =
                        serde_json::from_str(default_value).map_err(|e| {
                            BatchError::InvalidJobConfig(format!(
                                "Invalid default value for field '{field}': {e}"
                            ))
                        })?;
                    job.set(field.clone(), value);
                }
            }
            MigrationAction::RemoveField { field } => {
                job.remove(field);
            }
            MigrationAction::RenameField { from, to } => {
                job.rename(from, to);
            }
            MigrationAction::SetValue { field, value } => {
                let parsed: serde_json::Value = serde_json::from_str(value).map_err(|e| {
                    BatchError::InvalidJobConfig(format!("Invalid value for field '{field}': {e}"))
                })?;
                job.set(field.clone(), parsed);
            }
            MigrationAction::TransformField { .. } | MigrationAction::Custom { .. } => {
                // Custom/transform actions are handled by external logic.
                // Here we just record them as applied.
            }
        }
        Ok(())
    }
}

/// Report of a completed migration.
#[derive(Debug, Clone)]
pub struct MigrationReport {
    /// Original schema version.
    pub from_version: SchemaVersion,
    /// Resulting schema version.
    pub to_version: SchemaVersion,
    /// Number of migration steps applied.
    pub steps_applied: usize,
    /// Description of each action applied.
    pub actions_applied: Vec<String>,
}

impl MigrationReport {
    /// Whether any migrations were actually applied.
    #[must_use]
    pub fn was_migrated(&self) -> bool {
        self.steps_applied > 0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn v(major: u32, minor: u32, patch: u32) -> SchemaVersion {
        SchemaVersion::new(major, minor, patch)
    }

    fn sample_chain() -> MigrationChain {
        let mut chain = MigrationChain::new();

        // v1.0.0 -> v1.1.0: add "priority" field
        chain.add_step(
            MigrationStep::new(v(1, 0, 0), v(1, 1, 0), "Add priority field").with_action(
                MigrationAction::AddField {
                    field: "priority".to_string(),
                    default_value: "\"normal\"".to_string(),
                },
            ),
        );

        // v1.1.0 -> v1.2.0: rename "output_path" to "output_dir"
        chain.add_step(
            MigrationStep::new(v(1, 1, 0), v(1, 2, 0), "Rename output field").with_action(
                MigrationAction::RenameField {
                    from: "output_path".to_string(),
                    to: "output_dir".to_string(),
                },
            ),
        );

        // v1.2.0 -> v2.0.0: add "retry_config", remove "legacy_retries"
        chain.add_step(
            MigrationStep::new(v(1, 2, 0), v(2, 0, 0), "Restructure retry config")
                .with_action(MigrationAction::AddField {
                    field: "retry_config".to_string(),
                    default_value: r#"{"max_attempts": 3, "delay_ms": 1000}"#.to_string(),
                })
                .with_action(MigrationAction::RemoveField {
                    field: "legacy_retries".to_string(),
                }),
        );

        chain
    }

    // ── SchemaVersion ───────────────────────────────────────────────────
    #[test]
    fn test_schema_version_display() {
        assert_eq!(v(1, 2, 3).to_string(), "1.2.3");
    }

    #[test]
    fn test_schema_version_comparison() {
        assert!(v(1, 0, 0).is_older_than(&v(1, 1, 0)));
        assert!(v(1, 1, 0).is_older_than(&v(2, 0, 0)));
        assert!(!v(2, 0, 0).is_older_than(&v(1, 0, 0)));
        assert!(v(1, 0, 0).is_at_least(&v(1, 0, 0)));
        assert!(v(2, 0, 0).is_at_least(&v(1, 0, 0)));
    }

    #[test]
    fn test_schema_version_ordering() {
        let mut versions = vec![v(2, 0, 0), v(1, 0, 0), v(1, 2, 0), v(1, 1, 0)];
        versions.sort();
        assert_eq!(versions[0], v(1, 0, 0));
        assert_eq!(versions[3], v(2, 0, 0));
    }

    // ── MigrationAction display ─────────────────────────────────────────
    #[test]
    fn test_migration_action_display() {
        assert_eq!(
            MigrationAction::AddField {
                field: "priority".into(),
                default_value: "\"normal\"".into()
            }
            .to_string(),
            "add_field(priority)"
        );
        assert_eq!(
            MigrationAction::RenameField {
                from: "old".into(),
                to: "new".into()
            }
            .to_string(),
            "rename_field(old->new)"
        );
        assert_eq!(
            MigrationAction::RemoveField {
                field: "legacy".into()
            }
            .to_string(),
            "remove_field(legacy)"
        );
    }

    // ── MigrationChain ──────────────────────────────────────────────────
    #[test]
    fn test_chain_latest_version() {
        let chain = sample_chain();
        assert_eq!(chain.latest_version(), Some(v(2, 0, 0)));
    }

    #[test]
    fn test_chain_find_path_same_version() {
        let chain = sample_chain();
        let path = chain
            .find_path(&v(2, 0, 0), &v(2, 0, 0))
            .expect("should work");
        assert!(path.is_empty());
    }

    #[test]
    fn test_chain_find_path_one_step() {
        let chain = sample_chain();
        let path = chain
            .find_path(&v(1, 0, 0), &v(1, 1, 0))
            .expect("should work");
        assert_eq!(path.len(), 1);
        assert_eq!(path[0].to_version, v(1, 1, 0));
    }

    #[test]
    fn test_chain_find_path_multi_step() {
        let chain = sample_chain();
        let path = chain
            .find_path(&v(1, 0, 0), &v(2, 0, 0))
            .expect("should work");
        assert_eq!(path.len(), 3);
    }

    #[test]
    fn test_chain_find_path_no_path() {
        let chain = sample_chain();
        let result = chain.find_path(&v(0, 9, 0), &v(2, 0, 0));
        assert!(result.is_err());
    }

    #[test]
    fn test_chain_downgrade_not_supported() {
        let chain = sample_chain();
        let result = chain.find_path(&v(2, 0, 0), &v(1, 0, 0));
        assert!(result.is_err());
    }

    // ── JobData ─────────────────────────────────────────────────────────
    #[test]
    fn test_job_data_field_operations() {
        let mut data = JobData::new(v(1, 0, 0));
        data.set("name", serde_json::Value::String("test".into()));
        assert!(data.has_field("name"));
        assert_eq!(data.field_count(), 1);

        assert_eq!(
            data.get("name"),
            Some(&serde_json::Value::String("test".into()))
        );

        data.rename("name", "job_name");
        assert!(!data.has_field("name"));
        assert!(data.has_field("job_name"));

        data.remove("job_name");
        assert_eq!(data.field_count(), 0);
    }

    // ── JobMigrator ─────────────────────────────────────────────────────
    #[test]
    fn test_migrator_no_migration_needed() {
        let chain = sample_chain();
        let migrator = JobMigrator::new(chain);
        let mut job = JobData::new(v(2, 0, 0));
        let report = migrator.migrate(&mut job).expect("should work");
        assert!(!report.was_migrated());
        assert_eq!(report.steps_applied, 0);
    }

    #[test]
    fn test_migrator_single_step() {
        let chain = sample_chain();
        let migrator = JobMigrator::new(chain);
        let mut job = JobData::new(v(1, 0, 0));

        // Migrate just one step.
        let migrator = migrator.with_target(v(1, 1, 0));
        let report = migrator.migrate(&mut job).expect("should work");

        assert!(report.was_migrated());
        assert_eq!(report.steps_applied, 1);
        assert_eq!(job.schema_version, v(1, 1, 0));
        // Priority field should have been added.
        assert!(job.has_field("priority"));
        assert_eq!(
            job.get("priority"),
            Some(&serde_json::Value::String("normal".into()))
        );
    }

    #[test]
    fn test_migrator_full_migration() {
        let chain = sample_chain();
        let migrator = JobMigrator::new(chain);

        let mut job = JobData::new(v(1, 0, 0));
        job.set(
            "output_path",
            serde_json::Value::String(
                std::env::temp_dir()
                    .join("oximedia-batch-migration-output")
                    .to_string_lossy()
                    .into_owned(),
            ),
        );
        job.set("legacy_retries", serde_json::Value::Number(3.into()));

        let report = migrator.migrate(&mut job).expect("should work");

        assert!(report.was_migrated());
        assert_eq!(report.steps_applied, 3);
        assert_eq!(job.schema_version, v(2, 0, 0));

        // Check all migrations applied:
        // 1. priority added
        assert!(job.has_field("priority"));
        // 2. output_path renamed to output_dir
        assert!(!job.has_field("output_path"));
        assert!(job.has_field("output_dir"));
        // 3. retry_config added, legacy_retries removed
        assert!(job.has_field("retry_config"));
        assert!(!job.has_field("legacy_retries"));
    }

    #[test]
    fn test_migrator_add_field_does_not_overwrite_existing() {
        let chain = sample_chain();
        let migrator = JobMigrator::new(chain).with_target(v(1, 1, 0));

        let mut job = JobData::new(v(1, 0, 0));
        job.set("priority", serde_json::Value::String("high".into()));

        let _report = migrator.migrate(&mut job).expect("should work");

        // Should NOT overwrite existing "priority" field.
        assert_eq!(
            job.get("priority"),
            Some(&serde_json::Value::String("high".into()))
        );
    }

    #[test]
    fn test_migrator_needs_migration() {
        let chain = sample_chain();
        let migrator = JobMigrator::new(chain);

        let old_job = JobData::new(v(1, 0, 0));
        let current_job = JobData::new(v(2, 0, 0));

        assert!(migrator.needs_migration(&old_job));
        assert!(!migrator.needs_migration(&current_job));
    }

    #[test]
    fn test_migrator_set_value_action() {
        let mut chain = MigrationChain::new();
        chain.add_step(
            MigrationStep::new(v(1, 0, 0), v(1, 1, 0), "Set format version").with_action(
                MigrationAction::SetValue {
                    field: "format_version".to_string(),
                    value: "2".to_string(),
                },
            ),
        );

        let migrator = JobMigrator::new(chain);
        let mut job = JobData::new(v(1, 0, 0));
        job.set("format_version", serde_json::Value::Number(1.into()));

        migrator.migrate(&mut job).expect("should work");

        assert_eq!(
            job.get("format_version"),
            Some(&serde_json::Value::Number(2.into()))
        );
    }

    #[test]
    fn test_migration_report_actions() {
        let chain = sample_chain();
        let migrator = JobMigrator::new(chain).with_target(v(1, 1, 0));
        let mut job = JobData::new(v(1, 0, 0));

        let report = migrator.migrate(&mut job).expect("should work");
        assert_eq!(report.actions_applied.len(), 1);
        assert!(report.actions_applied[0].contains("priority"));
    }

    // ── MigrationStep builder ───────────────────────────────────────────
    #[test]
    fn test_migration_step_builder() {
        let step = MigrationStep::new(v(1, 0, 0), v(1, 1, 0), "Test")
            .with_action(MigrationAction::AddField {
                field: "x".into(),
                default_value: "0".into(),
            })
            .reversible();

        assert!(step.reversible);
        assert_eq!(step.actions.len(), 1);
    }

    // ── Empty chain ─────────────────────────────────────────────────────
    #[test]
    fn test_empty_chain() {
        let chain = MigrationChain::new();
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);
        assert!(chain.latest_version().is_none());
    }

    // ── Invalid JSON value in migration ─────────────────────────────────
    #[test]
    fn test_invalid_json_value_returns_error() {
        let mut chain = MigrationChain::new();
        chain.add_step(
            MigrationStep::new(v(1, 0, 0), v(1, 1, 0), "Bad value").with_action(
                MigrationAction::AddField {
                    field: "bad".into(),
                    default_value: "not valid json {{{".into(),
                },
            ),
        );

        let migrator = JobMigrator::new(chain);
        let mut job = JobData::new(v(1, 0, 0));
        let result = migrator.migrate(&mut job);
        assert!(result.is_err());
    }

    // ── Custom action is no-op ──────────────────────────────────────────
    #[test]
    fn test_custom_action_is_noop() {
        let mut chain = MigrationChain::new();
        chain.add_step(
            MigrationStep::new(v(1, 0, 0), v(1, 1, 0), "Custom migration").with_action(
                MigrationAction::Custom {
                    description: "manual review required".into(),
                },
            ),
        );

        let migrator = JobMigrator::new(chain);
        let mut job = JobData::new(v(1, 0, 0));
        let report = migrator.migrate(&mut job).expect("should work");
        assert!(report.was_migrated());
    }
}
