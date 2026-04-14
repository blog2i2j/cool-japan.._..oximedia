//! Multi-stage approval workflow for media assets.
//!
//! An `ApprovalWorkflow` is a named sequence of `ApprovalStage`s through which
//! an asset (or any entity identified by a UUID) must pass before it can be
//! considered approved.  Each stage has one or more assigned approvers and a
//! quorum policy (any one approver vs. all approvers).  Any reviewer can also
//! reject at any stage, which immediately halts the workflow.
//!
//! # Typical flow
//!
//! 1. Create an `ApprovalWorkflow` with `WorkflowBuilder`.
//! 2. Submit an entity (e.g. an asset) to get an `ApprovalRun`.
//! 3. Approvers cast `ApprovalDecision`s via `ApprovalRun::cast_decision`.
//! 4. The run advances through stages automatically.
//! 5. Once the final stage is approved the run reaches `RunStatus::Approved`.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Role types
// ---------------------------------------------------------------------------

/// A named approver role (e.g. "editor", "legal", "executive_producer").
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ApproverRole(pub String);

impl ApproverRole {
    /// Create a new role from a string slice.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// The role name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.0
    }
}

// ---------------------------------------------------------------------------
// Quorum policy
// ---------------------------------------------------------------------------

/// How many approvals are needed to pass a stage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuorumPolicy {
    /// Any single approver is sufficient.
    AnyOne,
    /// All designated approvers must approve.
    All,
    /// At least N approvers must approve.
    AtLeast(usize),
    /// A majority (> 50 %) must approve.
    Majority,
}

impl QuorumPolicy {
    /// Determine whether the quorum has been reached given the total number of
    /// approvers and the number who have approved.
    #[must_use]
    pub fn is_met(&self, total: usize, approved: usize) -> bool {
        match self {
            Self::AnyOne => approved >= 1,
            Self::All => approved >= total,
            Self::AtLeast(n) => approved >= *n,
            Self::Majority => {
                if total == 0 {
                    false
                } else {
                    approved * 2 > total
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Stage definition
// ---------------------------------------------------------------------------

/// A single named stage in an approval workflow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalStage {
    /// Unique stage identifier.
    pub id: Uuid,
    /// Human-readable stage name.
    pub name: String,
    /// Description of what this stage reviews.
    pub description: Option<String>,
    /// Users (by ID) who are designated approvers for this stage.
    pub approver_ids: Vec<Uuid>,
    /// Roles required to approve (supplementary to explicit user IDs).
    pub required_roles: Vec<ApproverRole>,
    /// Quorum policy for this stage.
    pub quorum: QuorumPolicy,
    /// Maximum time (in hours) allowed before this stage is considered overdue.
    pub sla_hours: Option<u32>,
    /// Whether reviewers may delegate to another user.
    pub allow_delegation: bool,
}

impl ApprovalStage {
    /// Create a new stage with a simple "any one approver" quorum.
    #[must_use]
    pub fn new(name: impl Into<String>, approver_ids: Vec<Uuid>) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            description: None,
            approver_ids,
            required_roles: vec![],
            quorum: QuorumPolicy::AnyOne,
            sla_hours: None,
            allow_delegation: false,
        }
    }

    /// Set the quorum policy.
    #[must_use]
    pub fn with_quorum(mut self, quorum: QuorumPolicy) -> Self {
        self.quorum = quorum;
        self
    }

    /// Set an SLA in hours.
    #[must_use]
    pub fn with_sla_hours(mut self, hours: u32) -> Self {
        self.sla_hours = Some(hours);
        self
    }

    /// Attach a description.
    #[must_use]
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }
}

// ---------------------------------------------------------------------------
// Workflow definition
// ---------------------------------------------------------------------------

/// A reusable approval workflow template.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalWorkflow {
    /// Unique workflow ID.
    pub id: Uuid,
    /// Human-readable workflow name.
    pub name: String,
    /// Ordered list of stages.
    pub stages: Vec<ApprovalStage>,
    /// Whether the workflow supports parallel stage processing.
    pub parallel_stages: bool,
    /// ID of the user who defined this workflow.
    pub created_by: Uuid,
    /// When the workflow definition was created.
    pub created_at: DateTime<Utc>,
}

impl ApprovalWorkflow {
    /// Create a new sequential workflow.
    #[must_use]
    pub fn new(name: impl Into<String>, stages: Vec<ApprovalStage>, created_by: Uuid) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            stages,
            parallel_stages: false,
            created_by,
            created_at: Utc::now(),
        }
    }

    /// Total number of stages.
    #[must_use]
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }
}

// ---------------------------------------------------------------------------
// Decision
// ---------------------------------------------------------------------------

/// The verdict cast by a single reviewer on a stage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecisionVerdict {
    /// The reviewer approves progression to the next stage.
    Approved,
    /// The reviewer requests changes before re-review.
    RequestChanges,
    /// The reviewer rejects the asset outright.
    Rejected,
    /// The reviewer abstains (counted neither for nor against quorum).
    Abstain,
}

/// A decision cast by a reviewer on a specific stage of a run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalDecision {
    /// Unique decision ID.
    pub id: Uuid,
    /// The run this decision belongs to.
    pub run_id: Uuid,
    /// The stage ID this decision pertains to.
    pub stage_id: Uuid,
    /// The reviewer who cast this decision.
    pub reviewer_id: Uuid,
    /// The verdict.
    pub verdict: DecisionVerdict,
    /// Optional comment from the reviewer.
    pub comment: Option<String>,
    /// When the decision was cast.
    pub decided_at: DateTime<Utc>,
    /// Optional ID of a user to whom this reviewer delegated (if delegation is enabled).
    pub delegated_to: Option<Uuid>,
}

impl ApprovalDecision {
    /// Create a new decision.
    #[must_use]
    pub fn new(
        run_id: Uuid,
        stage_id: Uuid,
        reviewer_id: Uuid,
        verdict: DecisionVerdict,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            run_id,
            stage_id,
            reviewer_id,
            verdict,
            comment: None,
            decided_at: Utc::now(),
            delegated_to: None,
        }
    }

    /// Attach a reviewer comment.
    #[must_use]
    pub fn with_comment(mut self, comment: impl Into<String>) -> Self {
        self.comment = Some(comment.into());
        self
    }
}

// ---------------------------------------------------------------------------
// Run state
// ---------------------------------------------------------------------------

/// The overall status of an approval run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunStatus {
    /// The run has been created but not yet started.
    Pending,
    /// The run is actively progressing through stages.
    InProgress,
    /// All stages have been approved.
    Approved,
    /// A reviewer rejected at some stage.
    Rejected,
    /// A reviewer requested changes; the submitter must revise and resubmit.
    ChangesRequested,
    /// The run was cancelled by an administrator.
    Cancelled,
}

impl RunStatus {
    /// Whether the run has reached a terminal state.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Approved | Self::Rejected | Self::ChangesRequested | Self::Cancelled
        )
    }
}

/// Error type for approval run operations.
#[derive(Debug, thiserror::Error)]
pub enum ApprovalError {
    /// The run is in a terminal state and cannot accept further decisions.
    #[error("Run is already in a terminal state: {0:?}")]
    AlreadyTerminal(RunStatus),
    /// The decision refers to a stage that is not the current active stage.
    #[error("Decision is for stage {decision_stage:?} but current stage is {current_stage:?}")]
    WrongStage {
        decision_stage: Uuid,
        current_stage: Uuid,
    },
    /// The reviewer is not in the list of designated approvers for this stage.
    #[error("Reviewer {0} is not a designated approver for this stage")]
    NotAnApprover(Uuid),
    /// The reviewer has already cast a decision for this stage.
    #[error("Reviewer {0} has already cast a decision for this stage")]
    AlreadyDecided(Uuid),
    /// No stages are defined in the workflow.
    #[error("Workflow has no stages")]
    NoStages,
    /// The run has no active stage (workflow completed or not started).
    #[error("No active stage")]
    NoActiveStage,
}

// ---------------------------------------------------------------------------
// ApprovalRun
// ---------------------------------------------------------------------------

/// A live instance of a workflow applied to a specific entity.
#[derive(Debug, Serialize, Deserialize)]
pub struct ApprovalRun {
    /// Unique run ID.
    pub id: Uuid,
    /// The workflow definition this run follows.
    pub workflow_id: Uuid,
    /// The entity being reviewed (e.g. asset ID).
    pub entity_id: Uuid,
    /// Current run status.
    pub status: RunStatus,
    /// Index of the current active stage within the workflow stages list.
    pub current_stage_index: usize,
    /// All decisions cast so far, keyed by stage ID.
    decisions_by_stage: HashMap<Uuid, Vec<ApprovalDecision>>,
    /// The workflow snapshot embedded for self-contained reads.
    workflow: ApprovalWorkflow,
    /// Who submitted this entity for review.
    pub submitted_by: Uuid,
    /// When the run was created.
    pub created_at: DateTime<Utc>,
    /// When the run reached a terminal state, if ever.
    pub completed_at: Option<DateTime<Utc>>,
    /// Submission comment (e.g. "ready for legal review").
    pub submission_note: Option<String>,
}

impl ApprovalRun {
    /// Start a new run for the given workflow and entity.
    ///
    /// # Errors
    ///
    /// Returns [`ApprovalError::NoStages`] if the workflow contains no stages.
    pub fn start(
        workflow: ApprovalWorkflow,
        entity_id: Uuid,
        submitted_by: Uuid,
    ) -> Result<Self, ApprovalError> {
        if workflow.stages.is_empty() {
            return Err(ApprovalError::NoStages);
        }
        Ok(Self {
            id: Uuid::new_v4(),
            workflow_id: workflow.id,
            entity_id,
            status: RunStatus::InProgress,
            current_stage_index: 0,
            decisions_by_stage: HashMap::new(),
            workflow,
            submitted_by,
            created_at: Utc::now(),
            completed_at: None,
            submission_note: None,
        })
    }

    /// Attach a submission note.
    #[must_use]
    pub fn with_submission_note(mut self, note: impl Into<String>) -> Self {
        self.submission_note = Some(note.into());
        self
    }

    /// The currently active stage definition, if the run is in progress.
    #[must_use]
    pub fn current_stage(&self) -> Option<&ApprovalStage> {
        if self.status == RunStatus::InProgress {
            self.workflow.stages.get(self.current_stage_index)
        } else {
            None
        }
    }

    /// Cast a decision on the current stage.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// * The run is already in a terminal state.
    /// * The decision stage ID does not match the current stage.
    /// * The reviewer is not a designated approver for the stage.
    /// * The reviewer has already cast a decision.
    pub fn cast_decision(&mut self, decision: ApprovalDecision) -> Result<(), ApprovalError> {
        if self.status.is_terminal() {
            return Err(ApprovalError::AlreadyTerminal(self.status.clone()));
        }

        let stage = self
            .current_stage()
            .ok_or(ApprovalError::NoActiveStage)?;

        if decision.stage_id != stage.id {
            return Err(ApprovalError::WrongStage {
                decision_stage: decision.stage_id,
                current_stage: stage.id,
            });
        }

        // Verify the reviewer is designated (unless approver_ids is empty = open review)
        if !stage.approver_ids.is_empty()
            && !stage.approver_ids.contains(&decision.reviewer_id)
        {
            return Err(ApprovalError::NotAnApprover(decision.reviewer_id));
        }

        // Check for duplicate decision
        let existing = self
            .decisions_by_stage
            .entry(stage.id)
            .or_default();
        if existing.iter().any(|d| d.reviewer_id == decision.reviewer_id) {
            return Err(ApprovalError::AlreadyDecided(decision.reviewer_id));
        }

        // Handle terminal verdicts immediately
        match decision.verdict {
            DecisionVerdict::Rejected => {
                existing.push(decision);
                self.status = RunStatus::Rejected;
                self.completed_at = Some(Utc::now());
                return Ok(());
            }
            DecisionVerdict::RequestChanges => {
                existing.push(decision);
                self.status = RunStatus::ChangesRequested;
                self.completed_at = Some(Utc::now());
                return Ok(());
            }
            _ => {}
        }

        existing.push(decision);

        // Re-borrow stage for quorum check
        let stage = &self.workflow.stages[self.current_stage_index];
        let decisions = self.decisions_by_stage.get(&stage.id).map(Vec::as_slice).unwrap_or(&[]);
        let approved_count = decisions
            .iter()
            .filter(|d| d.verdict == DecisionVerdict::Approved)
            .count();

        let total = stage.approver_ids.len().max(1);
        if stage.quorum.is_met(total, approved_count) {
            // Advance to the next stage
            let next_index = self.current_stage_index + 1;
            if next_index >= self.workflow.stages.len() {
                self.status = RunStatus::Approved;
                self.completed_at = Some(Utc::now());
            } else {
                self.current_stage_index = next_index;
            }
        }

        Ok(())
    }

    /// Cancel this run administratively.
    ///
    /// # Errors
    ///
    /// Returns an error if the run is already in a terminal state.
    pub fn cancel(&mut self) -> Result<(), ApprovalError> {
        if self.status.is_terminal() {
            return Err(ApprovalError::AlreadyTerminal(self.status.clone()));
        }
        self.status = RunStatus::Cancelled;
        self.completed_at = Some(Utc::now());
        Ok(())
    }

    /// All decisions for a given stage.
    #[must_use]
    pub fn decisions_for_stage(&self, stage_id: &Uuid) -> &[ApprovalDecision] {
        self.decisions_by_stage
            .get(stage_id)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Total decisions cast across all stages.
    #[must_use]
    pub fn total_decisions(&self) -> usize {
        self.decisions_by_stage.values().map(Vec::len).sum()
    }
}

// ---------------------------------------------------------------------------
// WorkflowBuilder
// ---------------------------------------------------------------------------

/// Fluent builder for `ApprovalWorkflow`.
#[derive(Default)]
pub struct WorkflowBuilder {
    name: String,
    stages: Vec<ApprovalStage>,
    created_by: Option<Uuid>,
}

impl WorkflowBuilder {
    /// Start a new builder.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            stages: vec![],
            created_by: None,
        }
    }

    /// Set the creator ID.
    #[must_use]
    pub fn created_by(mut self, user_id: Uuid) -> Self {
        self.created_by = Some(user_id);
        self
    }

    /// Append a stage.
    #[must_use]
    pub fn stage(mut self, stage: ApprovalStage) -> Self {
        self.stages.push(stage);
        self
    }

    /// Build the `ApprovalWorkflow`.
    ///
    /// # Errors
    ///
    /// Returns [`ApprovalError::NoStages`] if no stages were added.
    pub fn build(self) -> Result<ApprovalWorkflow, ApprovalError> {
        if self.stages.is_empty() {
            return Err(ApprovalError::NoStages);
        }
        let created_by = self.created_by.unwrap_or_else(Uuid::nil);
        Ok(ApprovalWorkflow::new(self.name, self.stages, created_by))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_user() -> Uuid {
        Uuid::new_v4()
    }

    fn make_asset() -> Uuid {
        Uuid::new_v4()
    }

    fn simple_workflow(approver: Uuid, creator: Uuid) -> ApprovalWorkflow {
        let stage = ApprovalStage::new("Review", vec![approver]);
        WorkflowBuilder::new("Simple Review")
            .created_by(creator)
            .stage(stage)
            .build()
            .expect("build should succeed")
    }

    #[test]
    fn quorum_any_one() {
        let q = QuorumPolicy::AnyOne;
        assert!(!q.is_met(3, 0));
        assert!(q.is_met(3, 1));
        assert!(q.is_met(3, 3));
    }

    #[test]
    fn quorum_all() {
        let q = QuorumPolicy::All;
        assert!(!q.is_met(3, 2));
        assert!(q.is_met(3, 3));
    }

    #[test]
    fn quorum_majority() {
        let q = QuorumPolicy::Majority;
        assert!(!q.is_met(4, 2)); // 2 out of 4 = 50 % — not a majority
        assert!(q.is_met(4, 3)); // 3 out of 4 = 75 %
        assert!(!q.is_met(0, 0)); // degenerate case
    }

    #[test]
    fn run_single_stage_approve() {
        let approver = make_user();
        let submitter = make_user();
        let entity = make_asset();

        let wf = simple_workflow(approver, submitter);
        let stage_id = wf.stages[0].id;

        let mut run = ApprovalRun::start(wf, entity, submitter).expect("start should succeed");
        assert_eq!(run.status, RunStatus::InProgress);

        let decision = ApprovalDecision::new(run.id, stage_id, approver, DecisionVerdict::Approved);
        run.cast_decision(decision).expect("cast_decision should succeed");

        assert_eq!(run.status, RunStatus::Approved);
        assert!(run.completed_at.is_some());
    }

    #[test]
    fn run_single_stage_reject() {
        let approver = make_user();
        let submitter = make_user();
        let entity = make_asset();

        let wf = simple_workflow(approver, submitter);
        let stage_id = wf.stages[0].id;

        let mut run = ApprovalRun::start(wf, entity, submitter).unwrap();
        let decision = ApprovalDecision::new(run.id, stage_id, approver, DecisionVerdict::Rejected);
        run.cast_decision(decision).unwrap();

        assert_eq!(run.status, RunStatus::Rejected);
    }

    #[test]
    fn run_multi_stage_sequential() {
        let approver_a = make_user();
        let approver_b = make_user();
        let submitter = make_user();
        let entity = make_asset();

        let stage_a = ApprovalStage::new("Stage A", vec![approver_a]);
        let stage_b = ApprovalStage::new("Stage B", vec![approver_b]);
        let wf = WorkflowBuilder::new("Two Stage")
            .created_by(submitter)
            .stage(stage_a)
            .stage(stage_b)
            .build()
            .unwrap();

        let stage_a_id = wf.stages[0].id;
        let stage_b_id = wf.stages[1].id;

        let mut run = ApprovalRun::start(wf, entity, submitter).unwrap();
        assert_eq!(run.current_stage_index, 0);

        run.cast_decision(ApprovalDecision::new(
            run.id,
            stage_a_id,
            approver_a,
            DecisionVerdict::Approved,
        ))
        .unwrap();

        assert_eq!(run.status, RunStatus::InProgress);
        assert_eq!(run.current_stage_index, 1);

        run.cast_decision(ApprovalDecision::new(
            run.id,
            stage_b_id,
            approver_b,
            DecisionVerdict::Approved,
        ))
        .unwrap();

        assert_eq!(run.status, RunStatus::Approved);
    }

    #[test]
    fn not_an_approver_returns_error() {
        let approver = make_user();
        let stranger = make_user();
        let submitter = make_user();
        let entity = make_asset();

        let wf = simple_workflow(approver, submitter);
        let stage_id = wf.stages[0].id;

        let mut run = ApprovalRun::start(wf, entity, submitter).unwrap();
        let decision = ApprovalDecision::new(run.id, stage_id, stranger, DecisionVerdict::Approved);
        let result = run.cast_decision(decision);

        assert!(matches!(result, Err(ApprovalError::NotAnApprover(_))));
    }

    #[test]
    fn already_decided_returns_error() {
        let approver = make_user();
        let submitter = make_user();
        let entity = make_asset();

        // Workflow needs two approvers so AnyOne quorum doesn't close the stage after first vote
        let approver2 = make_user();
        let stage = ApprovalStage::new("Review", vec![approver, approver2])
            .with_quorum(QuorumPolicy::All);
        let wf = WorkflowBuilder::new("All Required")
            .created_by(submitter)
            .stage(stage)
            .build()
            .unwrap();
        let stage_id = wf.stages[0].id;

        let mut run = ApprovalRun::start(wf, entity, submitter).unwrap();
        run.cast_decision(ApprovalDecision::new(
            run.id,
            stage_id,
            approver,
            DecisionVerdict::Approved,
        ))
        .unwrap();

        let duplicate =
            ApprovalDecision::new(run.id, stage_id, approver, DecisionVerdict::Approved);
        let result = run.cast_decision(duplicate);

        assert!(matches!(result, Err(ApprovalError::AlreadyDecided(_))));
    }

    #[test]
    fn cancel_run() {
        let approver = make_user();
        let submitter = make_user();
        let entity = make_asset();

        let wf = simple_workflow(approver, submitter);
        let mut run = ApprovalRun::start(wf, entity, submitter).unwrap();

        run.cancel().expect("cancel should succeed");
        assert_eq!(run.status, RunStatus::Cancelled);

        // Cancelling again should fail
        let result = run.cancel();
        assert!(matches!(result, Err(ApprovalError::AlreadyTerminal(_))));
    }

    #[test]
    fn workflow_builder_no_stages_fails() {
        let result = WorkflowBuilder::new("Empty").build();
        assert!(matches!(result, Err(ApprovalError::NoStages)));
    }
}
