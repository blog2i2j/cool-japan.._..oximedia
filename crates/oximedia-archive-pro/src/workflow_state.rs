//! Workflow state management for archive preservation pipelines.
//!
//! Tracks the phase of an archival workflow from ingest through distribution,
//! with support for phase advancement, rollback, and terminal state detection.

#![allow(dead_code)]

/// Phases in an archive preservation workflow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkflowPhase {
    /// Content has been received and is awaiting processing.
    Ingest,
    /// Content is undergoing quality control checks.
    Qc,
    /// Content is being packaged for long-term preservation.
    Preservation,
    /// Content metadata is being catalogued.
    Catalog,
    /// Content is being prepared for or delivered to end users.
    Distribution,
    /// Workflow completed successfully.
    Complete,
    /// Workflow failed and cannot continue.
    Failed,
}

impl WorkflowPhase {
    /// Returns the numeric index of the phase (for ordering comparisons).
    ///
    /// Terminal phases (`Complete` and `Failed`) return `usize::MAX`.
    #[must_use]
    pub const fn phase_index(&self) -> usize {
        match self {
            Self::Ingest => 0,
            Self::Qc => 1,
            Self::Preservation => 2,
            Self::Catalog => 3,
            Self::Distribution => 4,
            Self::Complete => usize::MAX,
            Self::Failed => usize::MAX,
        }
    }

    /// Returns true if this phase is a terminal (non-advanceable) phase.
    #[must_use]
    pub const fn is_terminal(&self) -> bool {
        matches!(self, Self::Complete | Self::Failed)
    }

    /// Returns the human-readable label for this phase.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Ingest => "Ingest",
            Self::Qc => "Quality Control",
            Self::Preservation => "Preservation",
            Self::Catalog => "Catalog",
            Self::Distribution => "Distribution",
            Self::Complete => "Complete",
            Self::Failed => "Failed",
        }
    }

    /// Returns the next logical phase, or `None` if terminal.
    #[must_use]
    pub const fn next(&self) -> Option<Self> {
        match self {
            Self::Ingest => Some(Self::Qc),
            Self::Qc => Some(Self::Preservation),
            Self::Preservation => Some(Self::Catalog),
            Self::Catalog => Some(Self::Distribution),
            Self::Distribution => Some(Self::Complete),
            Self::Complete | Self::Failed => None,
        }
    }
}

/// State snapshot for a single workflow item.
#[derive(Debug, Clone)]
pub struct WorkflowState {
    /// Unique identifier for the workflow item.
    pub item_id: String,
    /// Current phase of the workflow.
    pub phase: WorkflowPhase,
    /// Optional notes attached to this state snapshot.
    pub notes: Option<String>,
}

impl WorkflowState {
    /// Creates a new `WorkflowState` starting at the `Ingest` phase.
    #[must_use]
    pub fn new(item_id: impl Into<String>) -> Self {
        Self {
            item_id: item_id.into(),
            phase: WorkflowPhase::Ingest,
            notes: None,
        }
    }

    /// Returns true when the workflow has reached a terminal state.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        self.phase.is_terminal()
    }

    /// Attaches a note to this state snapshot.
    #[must_use]
    pub fn with_notes(mut self, notes: impl Into<String>) -> Self {
        self.notes = Some(notes.into());
        self
    }
}

/// Manages the lifecycle of a workflow item through its phases.
#[derive(Debug)]
pub struct WorkflowStateManager {
    history: Vec<WorkflowPhase>,
    current_phase: WorkflowPhase,
    item_id: String,
}

impl WorkflowStateManager {
    /// Creates a new manager for the given item, starting at `Ingest`.
    #[must_use]
    pub fn new(item_id: impl Into<String>) -> Self {
        Self {
            history: vec![WorkflowPhase::Ingest],
            current_phase: WorkflowPhase::Ingest,
            item_id: item_id.into(),
        }
    }

    /// Returns the current phase.
    #[must_use]
    pub fn current(&self) -> WorkflowPhase {
        self.current_phase
    }

    /// Returns the item ID managed by this instance.
    #[must_use]
    pub fn item_id(&self) -> &str {
        &self.item_id
    }

    /// Returns a slice of all phases visited (in order).
    #[must_use]
    pub fn history(&self) -> &[WorkflowPhase] {
        &self.history
    }

    /// Advances to the next phase. Returns `Ok(new_phase)` or an error string
    /// when the workflow is already in a terminal state.
    ///
    /// # Errors
    /// Returns `Err` when the workflow is already terminal.
    pub fn advance(&mut self) -> Result<WorkflowPhase, String> {
        if self.current_phase.is_terminal() {
            return Err(format!(
                "Workflow '{}' is already in terminal phase '{}'",
                self.item_id,
                self.current_phase.label()
            ));
        }
        let next = self.current_phase.next().ok_or_else(|| {
            format!(
                "No next phase after '{}' for item '{}'",
                self.current_phase.label(),
                self.item_id
            )
        })?;
        self.current_phase = next;
        self.history.push(next);
        Ok(next)
    }

    /// Rolls back to the previous phase. Returns `Ok(prev_phase)` or an error
    /// when there is no previous phase to return to.
    ///
    /// # Errors
    /// Returns `Err` when already at the initial phase.
    pub fn rollback(&mut self) -> Result<WorkflowPhase, String> {
        if self.history.len() <= 1 {
            return Err(format!(
                "Cannot rollback item '{}': already at initial phase",
                self.item_id
            ));
        }
        self.history.pop();
        let prev = *self.history.last().expect("history has at least one entry");
        self.current_phase = prev;
        Ok(prev)
    }

    /// Marks the workflow as failed.
    pub fn fail(&mut self) {
        self.current_phase = WorkflowPhase::Failed;
        self.history.push(WorkflowPhase::Failed);
    }

    /// Builds a `WorkflowState` snapshot for the current phase.
    #[must_use]
    pub fn snapshot(&self) -> WorkflowState {
        WorkflowState::new(self.item_id.clone())
            .with_notes(format!("Phase: {}", self.current_phase.label()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_index_ordering() {
        assert!(WorkflowPhase::Ingest.phase_index() < WorkflowPhase::Qc.phase_index());
        assert!(WorkflowPhase::Qc.phase_index() < WorkflowPhase::Preservation.phase_index());
        assert!(WorkflowPhase::Preservation.phase_index() < WorkflowPhase::Catalog.phase_index());
        assert!(WorkflowPhase::Catalog.phase_index() < WorkflowPhase::Distribution.phase_index());
    }

    #[test]
    fn test_terminal_phases() {
        assert!(WorkflowPhase::Complete.is_terminal());
        assert!(WorkflowPhase::Failed.is_terminal());
        assert!(!WorkflowPhase::Ingest.is_terminal());
        assert!(!WorkflowPhase::Distribution.is_terminal());
    }

    #[test]
    fn test_terminal_index_is_max() {
        assert_eq!(WorkflowPhase::Complete.phase_index(), usize::MAX);
        assert_eq!(WorkflowPhase::Failed.phase_index(), usize::MAX);
    }

    #[test]
    fn test_phase_next_chain() {
        assert_eq!(WorkflowPhase::Ingest.next(), Some(WorkflowPhase::Qc));
        assert_eq!(WorkflowPhase::Qc.next(), Some(WorkflowPhase::Preservation));
        assert_eq!(
            WorkflowPhase::Preservation.next(),
            Some(WorkflowPhase::Catalog)
        );
        assert_eq!(
            WorkflowPhase::Catalog.next(),
            Some(WorkflowPhase::Distribution)
        );
        assert_eq!(
            WorkflowPhase::Distribution.next(),
            Some(WorkflowPhase::Complete)
        );
        assert_eq!(WorkflowPhase::Complete.next(), None);
        assert_eq!(WorkflowPhase::Failed.next(), None);
    }

    #[test]
    fn test_phase_labels_nonempty() {
        let phases = [
            WorkflowPhase::Ingest,
            WorkflowPhase::Qc,
            WorkflowPhase::Preservation,
            WorkflowPhase::Catalog,
            WorkflowPhase::Distribution,
            WorkflowPhase::Complete,
            WorkflowPhase::Failed,
        ];
        for p in phases {
            assert!(!p.label().is_empty());
        }
    }

    #[test]
    fn test_workflow_state_new() {
        let ws = WorkflowState::new("item-001");
        assert_eq!(ws.phase, WorkflowPhase::Ingest);
        assert!(!ws.is_terminal());
        assert_eq!(ws.item_id, "item-001");
    }

    #[test]
    fn test_workflow_state_with_notes() {
        let ws = WorkflowState::new("item-002").with_notes("Ingested from tape");
        assert!(ws.notes.is_some());
        assert_eq!(
            ws.notes.expect("operation should succeed"),
            "Ingested from tape"
        );
    }

    #[test]
    fn test_manager_advance_full_cycle() {
        let mut mgr = WorkflowStateManager::new("item-003");
        assert_eq!(mgr.current(), WorkflowPhase::Ingest);
        mgr.advance().expect("operation should succeed");
        assert_eq!(mgr.current(), WorkflowPhase::Qc);
        mgr.advance().expect("operation should succeed");
        assert_eq!(mgr.current(), WorkflowPhase::Preservation);
        mgr.advance().expect("operation should succeed");
        assert_eq!(mgr.current(), WorkflowPhase::Catalog);
        mgr.advance().expect("operation should succeed");
        assert_eq!(mgr.current(), WorkflowPhase::Distribution);
        mgr.advance().expect("operation should succeed");
        assert_eq!(mgr.current(), WorkflowPhase::Complete);
    }

    #[test]
    fn test_manager_advance_fails_on_terminal() {
        let mut mgr = WorkflowStateManager::new("item-004");
        mgr.fail();
        assert!(mgr.advance().is_err());
    }

    #[test]
    fn test_manager_rollback() {
        let mut mgr = WorkflowStateManager::new("item-005");
        mgr.advance().expect("operation should succeed");
        mgr.advance().expect("operation should succeed");
        let prev = mgr.rollback().expect("operation should succeed");
        assert_eq!(prev, WorkflowPhase::Qc);
        assert_eq!(mgr.current(), WorkflowPhase::Qc);
    }

    #[test]
    fn test_manager_rollback_fails_at_initial() {
        let mut mgr = WorkflowStateManager::new("item-006");
        assert!(mgr.rollback().is_err());
    }

    #[test]
    fn test_manager_fail_sets_terminal() {
        let mut mgr = WorkflowStateManager::new("item-007");
        mgr.advance().expect("operation should succeed");
        mgr.fail();
        assert_eq!(mgr.current(), WorkflowPhase::Failed);
        assert!(mgr.current().is_terminal());
    }

    #[test]
    fn test_manager_history_length() {
        let mut mgr = WorkflowStateManager::new("item-008");
        mgr.advance().expect("operation should succeed");
        mgr.advance().expect("operation should succeed");
        assert_eq!(mgr.history().len(), 3); // Ingest, Qc, Preservation
    }

    #[test]
    fn test_manager_snapshot() {
        let mgr = WorkflowStateManager::new("item-009");
        let snap = mgr.snapshot();
        assert_eq!(snap.item_id, "item-009");
        assert_eq!(snap.phase, WorkflowPhase::Ingest);
        assert!(snap.notes.is_some());
    }

    #[test]
    fn test_manager_item_id() {
        let mgr = WorkflowStateManager::new("reel-42");
        assert_eq!(mgr.item_id(), "reel-42");
    }
}
