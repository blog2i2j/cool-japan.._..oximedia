#![allow(dead_code)]
//! Repair strategy planning and execution scheduling.
//!
//! This module generates ordered repair plans from a set of detected issues.
//! Each plan consists of prioritized steps that respect dependencies (e.g., header
//! repair must precede index rebuild), estimated durations, and rollback points.

use std::collections::HashMap;

/// Unique identifier for a repair step.
pub type StepId = u32;

/// The kind of repair action to perform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RepairAction {
    /// Rebuild or patch the container header.
    FixHeader,
    /// Reconstruct the seek/index table.
    RebuildIndex,
    /// Correct invalid or discontinuous timestamps.
    FixTimestamps,
    /// Re-synchronize audio and video streams.
    SyncStreams,
    /// Interpolate or conceal missing frames.
    ConcealFrames,
    /// Recover or regenerate lost packets.
    RecoverPackets,
    /// Repair or regenerate metadata atoms.
    RepairMetadata,
    /// Trim a truncated tail and finalize the container.
    FinalizeTruncated,
    /// Reorder mis-sequenced frames.
    ReorderFrames,
    /// Repair audio-specific issues (clicks, dropouts).
    RepairAudio,
}

impl std::fmt::Display for RepairAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FixHeader => write!(f, "Fix Header"),
            Self::RebuildIndex => write!(f, "Rebuild Index"),
            Self::FixTimestamps => write!(f, "Fix Timestamps"),
            Self::SyncStreams => write!(f, "Sync Streams"),
            Self::ConcealFrames => write!(f, "Conceal Frames"),
            Self::RecoverPackets => write!(f, "Recover Packets"),
            Self::RepairMetadata => write!(f, "Repair Metadata"),
            Self::FinalizeTruncated => write!(f, "Finalize Truncated"),
            Self::ReorderFrames => write!(f, "Reorder Frames"),
            Self::RepairAudio => write!(f, "Repair Audio"),
        }
    }
}

/// Risk level of a repair step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RiskLevel {
    /// No data loss expected.
    None,
    /// Slight chance of quality degradation.
    Low,
    /// Moderate chance of artifacts or minor data loss.
    Medium,
    /// High risk of further corruption if the step fails.
    High,
}

/// A single step in a repair plan.
#[derive(Debug, Clone)]
pub struct RepairStep {
    /// Unique step identifier.
    pub id: StepId,
    /// The repair action.
    pub action: RepairAction,
    /// Human-readable description of what this step does.
    pub description: String,
    /// Steps that must be completed before this one.
    pub depends_on: Vec<StepId>,
    /// Estimated duration in milliseconds.
    pub estimated_ms: u64,
    /// Risk level.
    pub risk: RiskLevel,
    /// Whether a rollback checkpoint should be created before this step.
    pub create_checkpoint: bool,
    /// Priority (lower = higher priority, executed first among peers).
    pub priority: u32,
}

impl RepairStep {
    /// Create a new repair step.
    pub fn new(id: StepId, action: RepairAction, description: impl Into<String>) -> Self {
        Self {
            id,
            action,
            description: description.into(),
            depends_on: Vec::new(),
            estimated_ms: 100,
            risk: RiskLevel::Low,
            create_checkpoint: false,
            priority: 100,
        }
    }

    /// Add a dependency on another step.
    pub fn depends_on(mut self, dep: StepId) -> Self {
        self.depends_on.push(dep);
        self
    }

    /// Set estimated duration.
    pub fn with_estimate_ms(mut self, ms: u64) -> Self {
        self.estimated_ms = ms;
        self
    }

    /// Set risk level.
    pub fn with_risk(mut self, risk: RiskLevel) -> Self {
        self.risk = risk;
        self
    }

    /// Mark this step as requiring a checkpoint.
    pub fn with_checkpoint(mut self) -> Self {
        self.create_checkpoint = true;
        self
    }

    /// Set priority.
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }
}

/// Status of an individual step during execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepStatus {
    /// Not yet started.
    Pending,
    /// Currently executing.
    Running,
    /// Completed successfully.
    Done,
    /// Failed with an error.
    Failed,
    /// Skipped because a dependency failed.
    Skipped,
}

/// Tracks execution progress across all steps.
#[derive(Debug, Clone)]
pub struct PlanProgress {
    /// Status of each step by id.
    pub step_status: HashMap<StepId, StepStatus>,
    /// Number of completed steps.
    pub completed: usize,
    /// Total steps.
    pub total: usize,
}

impl PlanProgress {
    /// Create a new progress tracker for the given step ids.
    pub fn new(step_ids: &[StepId]) -> Self {
        let mut status = HashMap::new();
        for &id in step_ids {
            status.insert(id, StepStatus::Pending);
        }
        Self {
            step_status: status,
            completed: 0,
            total: step_ids.len(),
        }
    }

    /// Mark a step with the given status.
    pub fn set_status(&mut self, id: StepId, status: StepStatus) {
        self.step_status.insert(id, status);
        self.completed = self
            .step_status
            .values()
            .filter(|s| matches!(s, StepStatus::Done | StepStatus::Failed | StepStatus::Skipped))
            .count();
    }

    /// Overall completion fraction from 0.0 to 1.0.
    #[allow(clippy::cast_precision_loss)]
    pub fn fraction(&self) -> f64 {
        if self.total == 0 {
            return 1.0;
        }
        self.completed as f64 / self.total as f64
    }

    /// Check whether all steps are finished (done, failed, or skipped).
    pub fn is_finished(&self) -> bool {
        self.completed == self.total
    }
}

/// An ordered repair plan containing steps and their dependencies.
#[derive(Debug, Clone)]
pub struct RecoveryPlan {
    /// All steps in the plan.
    pub steps: Vec<RepairStep>,
    /// Execution order (step ids in the order they should run).
    pub execution_order: Vec<StepId>,
    /// Total estimated time in milliseconds.
    pub total_estimated_ms: u64,
}

impl RecoveryPlan {
    /// Build a plan from a set of unordered steps, resolving dependency order.
    pub fn build(mut steps: Vec<RepairStep>) -> Self {
        // Sort by priority first, then topological order by dependencies
        steps.sort_by_key(|s| s.priority);

        let id_set: HashMap<StepId, usize> = steps
            .iter()
            .enumerate()
            .map(|(i, s)| (s.id, i))
            .collect();

        let execution_order = topological_sort(&steps, &id_set);
        let total_estimated_ms = steps.iter().map(|s| s.estimated_ms).sum();

        Self {
            steps,
            execution_order,
            total_estimated_ms,
        }
    }

    /// Return the number of steps in the plan.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Check whether the plan has no steps.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Return steps that have no unmet dependencies.
    pub fn ready_steps(&self, progress: &PlanProgress) -> Vec<StepId> {
        self.steps
            .iter()
            .filter(|s| {
                progress.step_status.get(&s.id) == Some(&StepStatus::Pending)
                    && s.depends_on.iter().all(|dep| {
                        progress.step_status.get(dep) == Some(&StepStatus::Done)
                    })
            })
            .map(|s| s.id)
            .collect()
    }

    /// Return the maximum risk level across all steps.
    pub fn max_risk(&self) -> RiskLevel {
        self.steps
            .iter()
            .map(|s| s.risk)
            .max()
            .unwrap_or(RiskLevel::None)
    }

    /// Create initial progress for this plan.
    pub fn create_progress(&self) -> PlanProgress {
        let ids: Vec<StepId> = self.steps.iter().map(|s| s.id).collect();
        PlanProgress::new(&ids)
    }
}

/// Topological sort using Kahn's algorithm.
fn topological_sort(steps: &[RepairStep], id_map: &HashMap<StepId, usize>) -> Vec<StepId> {
    let n = steps.len();
    let mut in_degree: HashMap<StepId, usize> = HashMap::new();
    let mut adj: HashMap<StepId, Vec<StepId>> = HashMap::new();

    for s in steps {
        in_degree.entry(s.id).or_insert(0);
        adj.entry(s.id).or_default();
        for &dep in &s.depends_on {
            if id_map.contains_key(&dep) {
                adj.entry(dep).or_default().push(s.id);
                *in_degree.entry(s.id).or_insert(0) += 1;
            }
        }
    }

    let mut queue: Vec<StepId> = in_degree
        .iter()
        .filter(|(_, &deg)| deg == 0)
        .map(|(&id, _)| id)
        .collect();
    queue.sort();

    let mut order = Vec::with_capacity(n);
    while let Some(id) = queue.pop() {
        order.push(id);
        if let Some(neighbors) = adj.get(&id) {
            for &next in neighbors {
                if let Some(deg) = in_degree.get_mut(&next) {
                    *deg = deg.saturating_sub(1);
                    if *deg == 0 {
                        queue.push(next);
                        queue.sort();
                    }
                }
            }
        }
    }

    order
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repair_action_display() {
        assert_eq!(format!("{}", RepairAction::FixHeader), "Fix Header");
        assert_eq!(format!("{}", RepairAction::RebuildIndex), "Rebuild Index");
    }

    #[test]
    fn test_risk_level_ordering() {
        assert!(RiskLevel::None < RiskLevel::Low);
        assert!(RiskLevel::Low < RiskLevel::Medium);
        assert!(RiskLevel::Medium < RiskLevel::High);
    }

    #[test]
    fn test_repair_step_creation() {
        let step = RepairStep::new(1, RepairAction::FixHeader, "Rebuild MP4 header");
        assert_eq!(step.id, 1);
        assert_eq!(step.action, RepairAction::FixHeader);
        assert!(step.depends_on.is_empty());
    }

    #[test]
    fn test_repair_step_builder_chain() {
        let step = RepairStep::new(2, RepairAction::RebuildIndex, "Rebuild index")
            .depends_on(1)
            .with_estimate_ms(500)
            .with_risk(RiskLevel::Medium)
            .with_checkpoint()
            .with_priority(10);

        assert_eq!(step.depends_on, vec![1]);
        assert_eq!(step.estimated_ms, 500);
        assert_eq!(step.risk, RiskLevel::Medium);
        assert!(step.create_checkpoint);
        assert_eq!(step.priority, 10);
    }

    #[test]
    fn test_plan_progress_initial() {
        let progress = PlanProgress::new(&[1, 2, 3]);
        assert_eq!(progress.total, 3);
        assert_eq!(progress.completed, 0);
        assert!(!progress.is_finished());
    }

    #[test]
    fn test_plan_progress_fraction() {
        let mut progress = PlanProgress::new(&[1, 2, 3, 4]);
        progress.set_status(1, StepStatus::Done);
        progress.set_status(2, StepStatus::Done);
        assert!((progress.fraction() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_plan_progress_finished() {
        let mut progress = PlanProgress::new(&[1, 2]);
        progress.set_status(1, StepStatus::Done);
        progress.set_status(2, StepStatus::Failed);
        assert!(progress.is_finished());
    }

    #[test]
    fn test_empty_plan() {
        let plan = RecoveryPlan::build(vec![]);
        assert!(plan.is_empty());
        assert_eq!(plan.len(), 0);
        assert_eq!(plan.total_estimated_ms, 0);
    }

    #[test]
    fn test_single_step_plan() {
        let steps = vec![RepairStep::new(1, RepairAction::FixHeader, "fix hdr")];
        let plan = RecoveryPlan::build(steps);
        assert_eq!(plan.len(), 1);
        assert_eq!(plan.execution_order, vec![1]);
    }

    #[test]
    fn test_dependency_order() {
        let steps = vec![
            RepairStep::new(2, RepairAction::RebuildIndex, "rebuild idx").depends_on(1),
            RepairStep::new(1, RepairAction::FixHeader, "fix hdr"),
        ];
        let plan = RecoveryPlan::build(steps);
        let hdr_pos = plan.execution_order.iter().position(|&id| id == 1).expect("unexpected None/Err");
        let idx_pos = plan.execution_order.iter().position(|&id| id == 2).expect("unexpected None/Err");
        assert!(hdr_pos < idx_pos, "Header must come before index rebuild");
    }

    #[test]
    fn test_ready_steps() {
        let steps = vec![
            RepairStep::new(1, RepairAction::FixHeader, "hdr"),
            RepairStep::new(2, RepairAction::RebuildIndex, "idx").depends_on(1),
            RepairStep::new(3, RepairAction::FixTimestamps, "ts"),
        ];
        let plan = RecoveryPlan::build(steps);
        let progress = plan.create_progress();

        let ready = plan.ready_steps(&progress);
        assert!(ready.contains(&1));
        assert!(ready.contains(&3));
        assert!(!ready.contains(&2));
    }

    #[test]
    fn test_ready_steps_after_completion() {
        let steps = vec![
            RepairStep::new(1, RepairAction::FixHeader, "hdr"),
            RepairStep::new(2, RepairAction::RebuildIndex, "idx").depends_on(1),
        ];
        let plan = RecoveryPlan::build(steps);
        let mut progress = plan.create_progress();
        progress.set_status(1, StepStatus::Done);

        let ready = plan.ready_steps(&progress);
        assert!(ready.contains(&2));
    }

    #[test]
    fn test_max_risk() {
        let steps = vec![
            RepairStep::new(1, RepairAction::FixHeader, "hdr").with_risk(RiskLevel::Low),
            RepairStep::new(2, RepairAction::ConcealFrames, "conceal").with_risk(RiskLevel::High),
        ];
        let plan = RecoveryPlan::build(steps);
        assert_eq!(plan.max_risk(), RiskLevel::High);
    }

    #[test]
    fn test_plan_total_estimate() {
        let steps = vec![
            RepairStep::new(1, RepairAction::FixHeader, "h").with_estimate_ms(200),
            RepairStep::new(2, RepairAction::RebuildIndex, "i").with_estimate_ms(300),
        ];
        let plan = RecoveryPlan::build(steps);
        assert_eq!(plan.total_estimated_ms, 500);
    }
}
