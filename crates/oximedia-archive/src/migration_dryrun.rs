//! Dry-run planning and rollback capability for format migrations.
//!
//! This module extends the [`migration`](crate::migration) module with two
//! critical features for safe long-term preservation workflows:
//!
//! ## Dry-run mode
//! Before performing any irreversible migration (transcoding, format conversion),
//! the operator can request a *plan* — a detailed description of every action
//! that would be taken, including:
//! - Which files would be touched
//! - Estimated output sizes and CPU time
//! - Risk assessment per format
//! - Any pre-condition failures (missing tools, insufficient disk space)
//!
//! ## Rollback
//! After a migration has been executed, the [`RollbackJournal`] records the
//! original state of each file so that the migration can be undone.  A rollback
//! replaces the migrated file(s) with the originals (which must have been
//! preserved, e.g. in the OARC sidecar or a designated backup directory).
//!
//! The journal is serialisable to JSON so that it survives process restarts.

use crate::{ArchiveError, ArchiveResult};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Dry-run plan
// ---------------------------------------------------------------------------

/// The overall status of a pre-condition check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PreconditionStatus {
    /// The pre-condition is satisfied.
    Ok,
    /// The pre-condition failed; the action cannot proceed without intervention.
    Failed,
    /// The pre-condition could not be evaluated (e.g. tool not installed).
    Skipped,
}

/// A single pre-condition check recorded in a dry-run plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreconditionCheck {
    /// Short identifier (e.g. `"disk_space"`, `"tool_available"`).
    pub name: String,
    /// Human-readable message explaining the outcome.
    pub message: String,
    /// Check outcome.
    pub status: PreconditionStatus,
}

impl PreconditionCheck {
    /// Create a new passing pre-condition.
    #[must_use]
    pub fn ok(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            message: message.into(),
            status: PreconditionStatus::Ok,
        }
    }

    /// Create a new failing pre-condition.
    #[must_use]
    pub fn failed(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            message: message.into(),
            status: PreconditionStatus::Failed,
        }
    }

    /// Create a skipped pre-condition.
    #[must_use]
    pub fn skipped(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            message: message.into(),
            status: PreconditionStatus::Skipped,
        }
    }
}

/// A single planned action for one source file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannedAction {
    /// Absolute path to the source file.
    pub source_path: PathBuf,

    /// Proposed output path after migration.
    pub target_path: PathBuf,

    /// Source format name (e.g. `"DV"`).
    pub source_format: String,

    /// Target format name (e.g. `"DPX"`).
    pub target_format: String,

    /// Whether quality loss is expected.
    pub quality_loss: bool,

    /// Whether the migration is reversible (lossless source backup exists).
    pub reversible: bool,

    /// Pre-condition checks for this action.
    pub preconditions: Vec<PreconditionCheck>,

    /// Estimated output size in bytes.
    pub estimated_output_bytes: Option<u64>,

    /// Estimated CPU time in fractional seconds.
    pub estimated_cpu_secs: Option<f64>,

    /// Human-readable notes.
    pub notes: String,
}

impl PlannedAction {
    /// Create a new planned action.
    #[must_use]
    pub fn new(
        source_path: impl Into<PathBuf>,
        target_path: impl Into<PathBuf>,
        source_format: impl Into<String>,
        target_format: impl Into<String>,
    ) -> Self {
        Self {
            source_path: source_path.into(),
            target_path: target_path.into(),
            source_format: source_format.into(),
            target_format: target_format.into(),
            quality_loss: false,
            reversible: true,
            preconditions: Vec::new(),
            estimated_output_bytes: None,
            estimated_cpu_secs: None,
            notes: String::new(),
        }
    }

    /// Add a pre-condition check.
    pub fn add_precondition(&mut self, check: PreconditionCheck) {
        self.preconditions.push(check);
    }

    /// Return `true` if all pre-conditions are `Ok` or `Skipped`.
    #[must_use]
    pub fn preconditions_satisfied(&self) -> bool {
        self.preconditions
            .iter()
            .all(|c| c.status != PreconditionStatus::Failed)
    }
}

/// A complete dry-run migration plan covering all candidate files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DryRunPlan {
    /// When the plan was generated.
    pub generated_at: DateTime<Utc>,

    /// Human-readable name for this migration campaign.
    pub campaign_name: String,

    /// All planned actions.
    pub actions: Vec<PlannedAction>,

    /// Global notes or warnings.
    pub global_notes: Vec<String>,
}

impl DryRunPlan {
    /// Create an empty plan.
    #[must_use]
    pub fn new(campaign_name: impl Into<String>) -> Self {
        Self {
            generated_at: Utc::now(),
            campaign_name: campaign_name.into(),
            actions: Vec::new(),
            global_notes: Vec::new(),
        }
    }

    /// Add an action.
    pub fn add_action(&mut self, action: PlannedAction) {
        self.actions.push(action);
    }

    /// Add a global note.
    pub fn add_note(&mut self, note: impl Into<String>) {
        self.global_notes.push(note.into());
    }

    /// Total number of actions in the plan.
    #[must_use]
    pub fn action_count(&self) -> usize {
        self.actions.len()
    }

    /// Number of actions whose pre-conditions are fully satisfied.
    #[must_use]
    pub fn ready_count(&self) -> usize {
        self.actions
            .iter()
            .filter(|a| a.preconditions_satisfied())
            .count()
    }

    /// Number of actions blocked by at least one failed pre-condition.
    #[must_use]
    pub fn blocked_count(&self) -> usize {
        self.actions
            .iter()
            .filter(|a| !a.preconditions_satisfied())
            .count()
    }

    /// Sum of estimated output sizes across all ready actions.
    #[must_use]
    pub fn total_estimated_output_bytes(&self) -> u64 {
        self.actions
            .iter()
            .filter(|a| a.preconditions_satisfied())
            .filter_map(|a| a.estimated_output_bytes)
            .fold(0u64, |acc, n| acc.saturating_add(n))
    }

    /// Sum of estimated CPU seconds across all ready actions.
    #[must_use]
    pub fn total_estimated_cpu_secs(&self) -> f64 {
        self.actions
            .iter()
            .filter(|a| a.preconditions_satisfied())
            .filter_map(|a| a.estimated_cpu_secs)
            .sum()
    }

    /// Serialize the plan to a JSON string.
    pub fn to_json(&self) -> ArchiveResult<String> {
        serde_json::to_string_pretty(self).map_err(|e| ArchiveError::Validation(e.to_string()))
    }

    /// Deserialize a plan from a JSON string.
    pub fn from_json(json: &str) -> ArchiveResult<Self> {
        serde_json::from_str(json).map_err(|e| ArchiveError::Validation(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// Rollback journal
// ---------------------------------------------------------------------------

/// The outcome of a single executed migration action, recorded in the journal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionOutcome {
    /// The action completed successfully.
    Success,
    /// The action failed; the target was not written.
    Failed,
    /// The action was skipped (pre-conditions not met).
    Skipped,
    /// The action was rolled back.
    RolledBack,
}

/// A record of one executed migration action in the rollback journal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JournalEntry {
    /// Unique entry identifier.
    pub entry_id: String,

    /// Path of the source file at the time of migration.
    pub source_path: PathBuf,

    /// Path of the migrated (output) file.
    pub target_path: PathBuf,

    /// Path where the original file was backed up before migration.
    /// `None` if the source was not preserved separately.
    pub backup_path: Option<PathBuf>,

    /// BLAKE3 checksum of the original source file (before migration).
    pub source_checksum: Option<String>,

    /// BLAKE3 checksum of the migrated output file.
    pub target_checksum: Option<String>,

    /// When the action was executed.
    pub executed_at: DateTime<Utc>,

    /// Outcome of the action.
    pub outcome: ActionOutcome,

    /// Error message if the action failed.
    pub error_message: Option<String>,
}

impl JournalEntry {
    /// Create a new journal entry in the `Success` state.
    #[must_use]
    pub fn success(
        entry_id: impl Into<String>,
        source_path: impl Into<PathBuf>,
        target_path: impl Into<PathBuf>,
    ) -> Self {
        Self {
            entry_id: entry_id.into(),
            source_path: source_path.into(),
            target_path: target_path.into(),
            backup_path: None,
            source_checksum: None,
            target_checksum: None,
            executed_at: Utc::now(),
            outcome: ActionOutcome::Success,
            error_message: None,
        }
    }

    /// Create a new journal entry in the `Failed` state.
    #[must_use]
    pub fn failed(
        entry_id: impl Into<String>,
        source_path: impl Into<PathBuf>,
        target_path: impl Into<PathBuf>,
        error: impl Into<String>,
    ) -> Self {
        Self {
            entry_id: entry_id.into(),
            source_path: source_path.into(),
            target_path: target_path.into(),
            backup_path: None,
            source_checksum: None,
            target_checksum: None,
            executed_at: Utc::now(),
            outcome: ActionOutcome::Failed,
            error_message: Some(error.into()),
        }
    }

    /// Return `true` if the migration can be rolled back for this entry.
    ///
    /// Rollback requires: outcome is `Success` and a backup path was recorded.
    #[must_use]
    pub fn can_rollback(&self) -> bool {
        self.outcome == ActionOutcome::Success && self.backup_path.is_some()
    }
}

/// Journal of executed migration actions with rollback support.
///
/// The journal is serialisable to JSON so it survives process restarts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackJournal {
    /// Name of the migration campaign.
    pub campaign_name: String,

    /// When the campaign started.
    pub started_at: DateTime<Utc>,

    /// All recorded entries.
    pub entries: Vec<JournalEntry>,

    /// Arbitrary tags for querying (e.g. `{"operator": "alice"}`).
    pub tags: HashMap<String, String>,
}

impl RollbackJournal {
    /// Create a new, empty journal.
    #[must_use]
    pub fn new(campaign_name: impl Into<String>) -> Self {
        Self {
            campaign_name: campaign_name.into(),
            started_at: Utc::now(),
            entries: Vec::new(),
            tags: HashMap::new(),
        }
    }

    /// Record a journal entry.
    pub fn record(&mut self, entry: JournalEntry) {
        self.entries.push(entry);
    }

    /// Return all entries eligible for rollback (Success + backup_path set).
    #[must_use]
    pub fn rollback_candidates(&self) -> Vec<&JournalEntry> {
        self.entries.iter().filter(|e| e.can_rollback()).collect()
    }

    /// Build a rollback plan: for each eligible entry, describe the
    /// `(target_path → backup_path)` file-move that would undo the migration.
    ///
    /// Returns a list of `(from, to)` tuples describing the restore operations.
    #[must_use]
    pub fn build_rollback_plan(&self) -> Vec<RollbackStep> {
        self.entries
            .iter()
            .filter(|e| e.can_rollback())
            .map(|e| RollbackStep {
                entry_id: e.entry_id.clone(),
                migrated_path: e.target_path.clone(),
                restore_from: e
                    .backup_path
                    .clone()
                    .expect("can_rollback guarantees backup_path is Some"),
                source_checksum: e.source_checksum.clone(),
            })
            .collect()
    }

    /// Apply a rollback by simulating the file-move operations.
    ///
    /// In a real implementation this would call `std::fs::rename` or copy
    /// bytes; here the method validates the plan and marks entries as
    /// `RolledBack` in memory.  Callers are responsible for the actual I/O.
    pub fn mark_rolled_back(&mut self, entry_id: &str) -> ArchiveResult<()> {
        let entry = self
            .entries
            .iter_mut()
            .find(|e| e.entry_id == entry_id)
            .ok_or_else(|| ArchiveError::Validation(format!("entry '{}' not found", entry_id)))?;

        if !entry.can_rollback() {
            return Err(ArchiveError::Validation(format!(
                "entry '{}' cannot be rolled back (outcome={:?}, backup={:?})",
                entry_id, entry.outcome, entry.backup_path
            )));
        }
        entry.outcome = ActionOutcome::RolledBack;
        Ok(())
    }

    /// Number of successfully rolled-back entries.
    #[must_use]
    pub fn rolled_back_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.outcome == ActionOutcome::RolledBack)
            .count()
    }

    /// Total number of entries.
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Number of successful (not yet rolled back) entries.
    #[must_use]
    pub fn success_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.outcome == ActionOutcome::Success)
            .count()
    }

    /// Serialize the journal to a pretty-printed JSON string.
    pub fn to_json(&self) -> ArchiveResult<String> {
        serde_json::to_string_pretty(self).map_err(|e| ArchiveError::Validation(e.to_string()))
    }

    /// Deserialize a journal from a JSON string.
    pub fn from_json(json: &str) -> ArchiveResult<Self> {
        serde_json::from_str(json).map_err(|e| ArchiveError::Validation(e.to_string()))
    }
}

/// A single step in a rollback plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackStep {
    /// Journal entry id.
    pub entry_id: String,
    /// The migrated output file to be removed / replaced.
    pub migrated_path: PathBuf,
    /// The backup of the original file to restore from.
    pub restore_from: PathBuf,
    /// Expected checksum of the restored file (for verification after restore).
    pub source_checksum: Option<String>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── DryRunPlan ──────────────────────────────────────────────────────────

    #[test]
    fn test_dry_run_plan_empty() {
        let plan = DryRunPlan::new("2026-Q1 DV Migration");
        assert_eq!(plan.action_count(), 0);
        assert_eq!(plan.ready_count(), 0);
        assert_eq!(plan.blocked_count(), 0);
    }

    #[test]
    fn test_dry_run_plan_ready_action() {
        let mut plan = DryRunPlan::new("test");
        let mut action = PlannedAction::new("/src/a.dv", "/dst/a.dpx", "DV", "DPX");
        action.add_precondition(PreconditionCheck::ok("disk_space", "50 GB free"));
        action.estimated_output_bytes = Some(2_000_000_000);
        action.estimated_cpu_secs = Some(120.0);
        plan.add_action(action);
        assert_eq!(plan.ready_count(), 1);
        assert_eq!(plan.blocked_count(), 0);
        assert_eq!(plan.total_estimated_output_bytes(), 2_000_000_000);
        assert!((plan.total_estimated_cpu_secs() - 120.0).abs() < 1e-9);
    }

    #[test]
    fn test_dry_run_plan_blocked_action() {
        let mut plan = DryRunPlan::new("test");
        let mut action = PlannedAction::new("/src/b.dv", "/dst/b.dpx", "DV", "DPX");
        action.add_precondition(PreconditionCheck::failed("disk_space", "Only 1 GB free"));
        plan.add_action(action);
        assert_eq!(plan.ready_count(), 0);
        assert_eq!(plan.blocked_count(), 1);
    }

    #[test]
    fn test_dry_run_plan_mixed_preconditions() {
        let mut plan = DryRunPlan::new("test");
        let mut action = PlannedAction::new("/src/c.dv", "/dst/c.dpx", "DV", "DPX");
        action.add_precondition(PreconditionCheck::ok("tool", "ffmpeg present"));
        action.add_precondition(PreconditionCheck::failed("space", "insufficient space"));
        plan.add_action(action);
        assert_eq!(
            plan.blocked_count(),
            1,
            "one failed check blocks the action"
        );
    }

    #[test]
    fn test_dry_run_plan_skipped_precondition_not_blocking() {
        let mut plan = DryRunPlan::new("test");
        let mut action = PlannedAction::new("/src/d.dv", "/dst/d.dpx", "DV", "DPX");
        action.add_precondition(PreconditionCheck::skipped("optional", "tool unavailable"));
        plan.add_action(action);
        assert_eq!(plan.ready_count(), 1, "skipped check does not block");
    }

    #[test]
    fn test_dry_run_plan_json_roundtrip() {
        let mut plan = DryRunPlan::new("json-test");
        plan.add_note("This is a test campaign");
        let json = plan.to_json().expect("serialize");
        let restored = DryRunPlan::from_json(&json).expect("deserialize");
        assert_eq!(restored.campaign_name, "json-test");
        assert_eq!(restored.global_notes.len(), 1);
    }

    // ── RollbackJournal ─────────────────────────────────────────────────────

    #[test]
    fn test_journal_empty() {
        let journal = RollbackJournal::new("empty campaign");
        assert_eq!(journal.entry_count(), 0);
        assert!(journal.rollback_candidates().is_empty());
    }

    #[test]
    fn test_journal_success_entry_with_backup_is_candidate() {
        let mut journal = RollbackJournal::new("test");
        let mut entry = JournalEntry::success("e1", "/src/a.dv", "/dst/a.dpx");
        entry.backup_path = Some(PathBuf::from("/backup/a.dv"));
        journal.record(entry);
        assert_eq!(journal.rollback_candidates().len(), 1);
    }

    #[test]
    fn test_journal_success_entry_without_backup_not_candidate() {
        let mut journal = RollbackJournal::new("test");
        let entry = JournalEntry::success("e2", "/src/b.dv", "/dst/b.dpx");
        // no backup_path set
        journal.record(entry);
        assert_eq!(journal.rollback_candidates().len(), 0);
    }

    #[test]
    fn test_journal_failed_entry_not_candidate() {
        let mut journal = RollbackJournal::new("test");
        let mut entry = JournalEntry::failed("e3", "/src/c.dv", "/dst/c.dpx", "transcode error");
        entry.backup_path = Some(PathBuf::from("/backup/c.dv"));
        journal.record(entry);
        assert_eq!(
            journal.rollback_candidates().len(),
            0,
            "failed actions cannot be rolled back"
        );
    }

    #[test]
    fn test_journal_mark_rolled_back_ok() {
        let mut journal = RollbackJournal::new("test");
        let mut entry = JournalEntry::success("e4", "/src/d.dv", "/dst/d.dpx");
        entry.backup_path = Some(PathBuf::from("/backup/d.dv"));
        journal.record(entry);

        journal.mark_rolled_back("e4").expect("should rollback");
        assert_eq!(journal.rolled_back_count(), 1);
        assert_eq!(journal.success_count(), 0);
    }

    #[test]
    fn test_journal_mark_rolled_back_unknown_id_errors() {
        let mut journal = RollbackJournal::new("test");
        let result = journal.mark_rolled_back("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_journal_build_rollback_plan() {
        let mut journal = RollbackJournal::new("test");
        let mut entry = JournalEntry::success("e5", "/src/e.dv", "/dst/e.dpx");
        entry.backup_path = Some(PathBuf::from("/backup/e.dv"));
        entry.source_checksum = Some("abc123".to_string());
        journal.record(entry);

        let plan = journal.build_rollback_plan();
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].entry_id, "e5");
        assert_eq!(plan[0].restore_from, PathBuf::from("/backup/e.dv"));
        assert_eq!(plan[0].source_checksum.as_deref(), Some("abc123"));
    }

    #[test]
    fn test_journal_json_roundtrip() {
        let mut journal = RollbackJournal::new("json-roundtrip");
        journal
            .tags
            .insert("operator".to_string(), "alice".to_string());
        let json = journal.to_json().expect("serialize");
        let restored = RollbackJournal::from_json(&json).expect("deserialize");
        assert_eq!(restored.campaign_name, "json-roundtrip");
        assert_eq!(
            restored.tags.get("operator").map(|s| s.as_str()),
            Some("alice")
        );
    }

    // --- New tests for migration dry-run and rollback (implementation items) ---

    #[test]
    fn test_planned_action_default_flags() {
        let action = PlannedAction::new("/src/a.dv", "/dst/a.mxf", "DV", "MXF");
        assert!(!action.quality_loss, "quality_loss should default to false");
        assert!(action.reversible, "reversible should default to true");
        assert!(action.notes.is_empty(), "notes should default to empty");
        assert!(
            action.preconditions.is_empty(),
            "preconditions should default to empty"
        );
    }

    #[test]
    fn test_planned_action_preconditions_satisfied_empty() {
        // No preconditions → satisfied
        let action = PlannedAction::new("/src/a.dv", "/dst/a.mxf", "DV", "MXF");
        assert!(action.preconditions_satisfied());
    }

    #[test]
    fn test_planned_action_multiple_ok_preconditions() {
        let mut action = PlannedAction::new("/src/a.dv", "/dst/a.mxf", "DV", "MXF");
        action.add_precondition(PreconditionCheck::ok("disk", "100 GB free"));
        action.add_precondition(PreconditionCheck::ok("tool", "oximedia available"));
        action.add_precondition(PreconditionCheck::skipped("gpu", "no GPU available"));
        assert!(action.preconditions_satisfied());
    }

    #[test]
    fn test_dry_run_plan_global_notes() {
        let mut plan = DryRunPlan::new("notes-test");
        plan.add_note("note 1");
        plan.add_note("note 2");
        assert_eq!(plan.global_notes.len(), 2);
        assert_eq!(plan.global_notes[0], "note 1");
    }

    #[test]
    fn test_dry_run_plan_estimated_bytes_only_ready() {
        let mut plan = DryRunPlan::new("test");
        // Blocked action — should NOT be included in estimate
        let mut blocked = PlannedAction::new("/src/b.dv", "/dst/b.mxf", "DV", "MXF");
        blocked.add_precondition(PreconditionCheck::failed("space", "no space"));
        blocked.estimated_output_bytes = Some(500_000_000);
        plan.add_action(blocked);

        // Ready action — SHOULD be included
        let mut ready = PlannedAction::new("/src/a.dv", "/dst/a.mxf", "DV", "MXF");
        ready.estimated_output_bytes = Some(200_000_000);
        plan.add_action(ready);

        assert_eq!(plan.total_estimated_output_bytes(), 200_000_000);
    }

    #[test]
    fn test_journal_entry_can_rollback_requires_success_and_backup() {
        // Failed entry with backup → cannot rollback
        let mut e = JournalEntry::failed("e1", "/src/a.dv", "/dst/a.mxf", "error");
        e.backup_path = Some(PathBuf::from("/backup/a.dv"));
        assert!(!e.can_rollback());

        // Success entry without backup → cannot rollback
        let e2 = JournalEntry::success("e2", "/src/b.dv", "/dst/b.mxf");
        assert!(!e2.can_rollback());

        // Success entry with backup → can rollback
        let mut e3 = JournalEntry::success("e3", "/src/c.dv", "/dst/c.mxf");
        e3.backup_path = Some(PathBuf::from("/backup/c.dv"));
        assert!(e3.can_rollback());
    }

    #[test]
    fn test_journal_rollback_idempotent_state() {
        let mut journal = RollbackJournal::new("test");
        let mut entry = JournalEntry::success("e1", "/src/a.dv", "/dst/a.mxf");
        entry.backup_path = Some(PathBuf::from("/backup/a.dv"));
        journal.record(entry);

        // After rolling back, the entry is marked RolledBack
        journal.mark_rolled_back("e1").expect("rollback ok");
        assert_eq!(journal.rolled_back_count(), 1);
        // Trying to rollback again should fail (outcome is now RolledBack, not Success)
        let second = journal.mark_rolled_back("e1");
        assert!(
            second.is_err(),
            "cannot rollback an already-rolled-back entry"
        );
    }

    #[test]
    fn test_journal_multiple_entries_only_eligible_in_plan() {
        let mut journal = RollbackJournal::new("test");

        // Entry 1: success + backup (eligible)
        let mut e1 = JournalEntry::success("e1", "/src/a.dv", "/dst/a.mxf");
        e1.backup_path = Some(PathBuf::from("/backup/a.dv"));
        journal.record(e1);

        // Entry 2: success, no backup (not eligible)
        journal.record(JournalEntry::success("e2", "/src/b.dv", "/dst/b.mxf"));

        // Entry 3: failed + backup (not eligible)
        let mut e3 = JournalEntry::failed("e3", "/src/c.dv", "/dst/c.mxf", "error");
        e3.backup_path = Some(PathBuf::from("/backup/c.dv"));
        journal.record(e3);

        assert_eq!(journal.rollback_candidates().len(), 1);
        let plan = journal.build_rollback_plan();
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].entry_id, "e1");
    }

    #[test]
    fn test_rollback_step_fields() {
        let mut journal = RollbackJournal::new("test");
        let mut entry = JournalEntry::success("e42", "/src/master.mxf", "/dst/master.dpx");
        entry.backup_path = Some(PathBuf::from("/vault/master.mxf.bak"));
        entry.source_checksum = Some("blake3hex".to_string());
        journal.record(entry);

        let steps = journal.build_rollback_plan();
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].entry_id, "e42");
        assert_eq!(steps[0].migrated_path, PathBuf::from("/dst/master.dpx"));
        assert_eq!(
            steps[0].restore_from,
            PathBuf::from("/vault/master.mxf.bak")
        );
        assert_eq!(steps[0].source_checksum.as_deref(), Some("blake3hex"));
    }

    #[test]
    fn test_dry_run_plan_total_cpu_secs_only_ready() {
        let mut plan = DryRunPlan::new("cpu-test");

        let mut blocked = PlannedAction::new("/src/b.dv", "/dst/b.mxf", "DV", "MXF");
        blocked.add_precondition(PreconditionCheck::failed("space", "no space"));
        blocked.estimated_cpu_secs = Some(500.0);
        plan.add_action(blocked);

        let mut ready = PlannedAction::new("/src/a.dv", "/dst/a.mxf", "DV", "MXF");
        ready.estimated_cpu_secs = Some(30.0);
        plan.add_action(ready);

        let total = plan.total_estimated_cpu_secs();
        assert!(
            (total - 30.0).abs() < 1e-9,
            "blocked action CPU secs should not be counted"
        );
    }
}
