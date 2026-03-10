#![allow(dead_code)]

//! Migration planning for long-term digital preservation.
//!
//! This module provides tools to create, validate, and execute migration plans
//! for moving media assets between archival formats while preserving integrity.

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, SystemTime};

/// Priority level for a migration task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MigrationPriority {
    /// Critical — format is obsolete, immediate migration required.
    Critical,
    /// High — format approaching end-of-life.
    High,
    /// Medium — migration recommended within the planning window.
    Medium,
    /// Low — format is stable, migration optional.
    Low,
}

impl fmt::Display for MigrationPriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Critical => write!(f, "CRITICAL"),
            Self::High => write!(f, "HIGH"),
            Self::Medium => write!(f, "MEDIUM"),
            Self::Low => write!(f, "LOW"),
        }
    }
}

/// The current status of a migration task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MigrationStatus {
    /// Not yet started.
    Pending,
    /// Currently in progress.
    InProgress,
    /// Completed successfully.
    Completed,
    /// Failed with an error.
    Failed,
    /// Cancelled by the operator.
    Cancelled,
    /// Paused and can be resumed.
    Paused,
}

impl fmt::Display for MigrationStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pending => write!(f, "Pending"),
            Self::InProgress => write!(f, "InProgress"),
            Self::Completed => write!(f, "Completed"),
            Self::Failed => write!(f, "Failed"),
            Self::Cancelled => write!(f, "Cancelled"),
            Self::Paused => write!(f, "Paused"),
        }
    }
}

/// A single migration task within a migration plan.
#[derive(Debug, Clone)]
pub struct MigrationTask {
    /// Unique task identifier.
    pub id: u64,
    /// Human-readable name.
    pub name: String,
    /// Source format identifier (e.g. "video/x-msvideo").
    pub source_format: String,
    /// Target format identifier (e.g. "video/x-matroska").
    pub target_format: String,
    /// Number of assets affected.
    pub asset_count: usize,
    /// Total bytes to migrate.
    pub total_bytes: u64,
    /// Priority level.
    pub priority: MigrationPriority,
    /// Current status.
    pub status: MigrationStatus,
    /// Estimated duration for the migration.
    pub estimated_duration: Duration,
    /// Tags or labels for categorisation.
    pub tags: Vec<String>,
}

impl MigrationTask {
    /// Create a new migration task.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: u64,
        name: impl Into<String>,
        source_format: impl Into<String>,
        target_format: impl Into<String>,
        asset_count: usize,
        total_bytes: u64,
        priority: MigrationPriority,
        estimated_duration: Duration,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            source_format: source_format.into(),
            target_format: target_format.into(),
            asset_count,
            total_bytes,
            priority,
            status: MigrationStatus::Pending,
            estimated_duration,
            tags: Vec::new(),
        }
    }

    /// Add a tag to this task.
    pub fn add_tag(&mut self, tag: impl Into<String>) {
        self.tags.push(tag.into());
    }

    /// Check if this task is actionable (Pending or Paused).
    pub fn is_actionable(&self) -> bool {
        matches!(
            self.status,
            MigrationStatus::Pending | MigrationStatus::Paused
        )
    }

    /// Check if this task has finished (Completed, Failed, or Cancelled).
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.status,
            MigrationStatus::Completed | MigrationStatus::Failed | MigrationStatus::Cancelled
        )
    }

    /// Return the migration throughput estimate in bytes per second.
    #[allow(clippy::cast_precision_loss)]
    pub fn estimated_throughput_bps(&self) -> f64 {
        let secs = self.estimated_duration.as_secs_f64();
        if secs <= 0.0 {
            return 0.0;
        }
        self.total_bytes as f64 / secs
    }
}

/// A complete migration plan containing multiple tasks.
#[derive(Debug, Clone)]
pub struct MigrationPlan {
    /// Plan identifier.
    pub id: u64,
    /// Human-readable plan name.
    pub name: String,
    /// When the plan was created.
    pub created_at: SystemTime,
    /// Optional deadline for completing all tasks.
    pub deadline: Option<SystemTime>,
    /// Ordered list of tasks.
    tasks: Vec<MigrationTask>,
    /// Arbitrary key-value metadata.
    pub metadata: HashMap<String, String>,
}

impl MigrationPlan {
    /// Create a new empty migration plan.
    pub fn new(id: u64, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            created_at: SystemTime::now(),
            deadline: None,
            tasks: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Set a deadline for this plan.
    pub fn with_deadline(mut self, deadline: SystemTime) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Add a task to the plan.
    pub fn add_task(&mut self, task: MigrationTask) {
        self.tasks.push(task);
    }

    /// Get all tasks.
    pub fn tasks(&self) -> &[MigrationTask] {
        &self.tasks
    }

    /// Get mutable reference to all tasks.
    pub fn tasks_mut(&mut self) -> &mut [MigrationTask] {
        &mut self.tasks
    }

    /// Get tasks filtered by priority.
    pub fn tasks_by_priority(&self, priority: MigrationPriority) -> Vec<&MigrationTask> {
        self.tasks
            .iter()
            .filter(|t| t.priority == priority)
            .collect()
    }

    /// Get tasks filtered by status.
    pub fn tasks_by_status(&self, status: MigrationStatus) -> Vec<&MigrationTask> {
        self.tasks.iter().filter(|t| t.status == status).collect()
    }

    /// Return the total number of tasks.
    pub fn task_count(&self) -> usize {
        self.tasks.len()
    }

    /// Count tasks in a given status.
    pub fn count_by_status(&self, status: MigrationStatus) -> usize {
        self.tasks.iter().filter(|t| t.status == status).count()
    }

    /// Return the total bytes across all tasks.
    pub fn total_bytes(&self) -> u64 {
        self.tasks.iter().map(|t| t.total_bytes).sum()
    }

    /// Return the total asset count across all tasks.
    pub fn total_asset_count(&self) -> usize {
        self.tasks.iter().map(|t| t.asset_count).sum()
    }

    /// Completion ratio as a fraction in [0.0, 1.0].
    #[allow(clippy::cast_precision_loss)]
    pub fn completion_ratio(&self) -> f64 {
        if self.tasks.is_empty() {
            return 0.0;
        }
        let done = self.count_by_status(MigrationStatus::Completed) as f64;
        let total = self.tasks.len() as f64;
        done / total
    }

    /// True when all tasks have reached a terminal state.
    pub fn is_complete(&self) -> bool {
        !self.tasks.is_empty() && self.tasks.iter().all(MigrationTask::is_terminal)
    }

    /// Estimated total duration (sum of all task estimates).
    pub fn estimated_total_duration(&self) -> Duration {
        self.tasks.iter().map(|t| t.estimated_duration).sum()
    }

    /// Sort tasks by priority (Critical first).
    pub fn sort_by_priority(&mut self) {
        self.tasks.sort_by_key(|t| t.priority);
    }

    /// Validate the plan: every task must have non-empty formats and positive asset counts.
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        for task in &self.tasks {
            if task.source_format.is_empty() {
                errors.push(format!("Task {} has empty source format", task.id));
            }
            if task.target_format.is_empty() {
                errors.push(format!("Task {} has empty target format", task.id));
            }
            if task.asset_count == 0 {
                errors.push(format!("Task {} has zero assets", task.id));
            }
            if task.source_format == task.target_format {
                errors.push(format!(
                    "Task {} has identical source and target formats",
                    task.id
                ));
            }
        }
        errors
    }

    /// Add metadata key-value pair.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }
}

/// Summary statistics of a migration plan.
#[derive(Debug, Clone)]
pub struct PlanSummary {
    /// Total tasks.
    pub total_tasks: usize,
    /// Completed tasks.
    pub completed: usize,
    /// Failed tasks.
    pub failed: usize,
    /// Pending tasks.
    pub pending: usize,
    /// In-progress tasks.
    pub in_progress: usize,
    /// Overall completion percentage.
    pub completion_pct: f64,
    /// Total bytes to migrate.
    pub total_bytes: u64,
}

impl PlanSummary {
    /// Build a summary from a plan.
    #[allow(clippy::cast_precision_loss)]
    pub fn from_plan(plan: &MigrationPlan) -> Self {
        let total_tasks = plan.task_count();
        let completed = plan.count_by_status(MigrationStatus::Completed);
        let failed = plan.count_by_status(MigrationStatus::Failed);
        let pending = plan.count_by_status(MigrationStatus::Pending);
        let in_progress = plan.count_by_status(MigrationStatus::InProgress);
        let completion_pct = if total_tasks == 0 {
            0.0
        } else {
            completed as f64 / total_tasks as f64 * 100.0
        };
        Self {
            total_tasks,
            completed,
            failed,
            pending,
            in_progress,
            completion_pct,
            total_bytes: plan.total_bytes(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_task(id: u64, priority: MigrationPriority) -> MigrationTask {
        MigrationTask::new(
            id,
            format!("task-{id}"),
            "video/x-msvideo",
            "video/x-matroska",
            100,
            1_000_000,
            priority,
            Duration::from_secs(3600),
        )
    }

    #[test]
    fn test_task_creation() {
        let task = sample_task(1, MigrationPriority::High);
        assert_eq!(task.id, 1);
        assert_eq!(task.status, MigrationStatus::Pending);
        assert_eq!(task.asset_count, 100);
    }

    #[test]
    fn test_task_is_actionable() {
        let mut task = sample_task(1, MigrationPriority::Medium);
        assert!(task.is_actionable());
        task.status = MigrationStatus::InProgress;
        assert!(!task.is_actionable());
        task.status = MigrationStatus::Paused;
        assert!(task.is_actionable());
    }

    #[test]
    fn test_task_is_terminal() {
        let mut task = sample_task(1, MigrationPriority::Low);
        assert!(!task.is_terminal());
        task.status = MigrationStatus::Completed;
        assert!(task.is_terminal());
        task.status = MigrationStatus::Failed;
        assert!(task.is_terminal());
        task.status = MigrationStatus::Cancelled;
        assert!(task.is_terminal());
    }

    #[test]
    fn test_task_throughput() {
        let task = sample_task(1, MigrationPriority::High);
        let bps = task.estimated_throughput_bps();
        assert!(bps > 0.0);
        // 1_000_000 bytes / 3600 seconds ≈ 277.78 bps
        assert!((bps - 277.78).abs() < 1.0);
    }

    #[test]
    fn test_task_tags() {
        let mut task = sample_task(1, MigrationPriority::Low);
        assert!(task.tags.is_empty());
        task.add_tag("video");
        task.add_tag("legacy");
        assert_eq!(task.tags.len(), 2);
    }

    #[test]
    fn test_plan_creation() {
        let plan = MigrationPlan::new(1, "Test Plan");
        assert_eq!(plan.id, 1);
        assert_eq!(plan.task_count(), 0);
        assert!(plan.metadata.is_empty());
    }

    #[test]
    fn test_plan_add_tasks() {
        let mut plan = MigrationPlan::new(1, "Test Plan");
        plan.add_task(sample_task(1, MigrationPriority::High));
        plan.add_task(sample_task(2, MigrationPriority::Low));
        assert_eq!(plan.task_count(), 2);
        assert_eq!(plan.total_bytes(), 2_000_000);
        assert_eq!(plan.total_asset_count(), 200);
    }

    #[test]
    fn test_plan_completion_ratio() {
        let mut plan = MigrationPlan::new(1, "P");
        assert!((plan.completion_ratio() - 0.0).abs() < f64::EPSILON);
        plan.add_task(sample_task(1, MigrationPriority::High));
        plan.add_task(sample_task(2, MigrationPriority::Low));
        plan.tasks_mut()[0].status = MigrationStatus::Completed;
        assert!((plan.completion_ratio() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_plan_is_complete() {
        let mut plan = MigrationPlan::new(1, "P");
        assert!(!plan.is_complete()); // empty plan
        plan.add_task(sample_task(1, MigrationPriority::High));
        assert!(!plan.is_complete());
        plan.tasks_mut()[0].status = MigrationStatus::Completed;
        assert!(plan.is_complete());
    }

    #[test]
    fn test_plan_sort_by_priority() {
        let mut plan = MigrationPlan::new(1, "P");
        plan.add_task(sample_task(1, MigrationPriority::Low));
        plan.add_task(sample_task(2, MigrationPriority::Critical));
        plan.add_task(sample_task(3, MigrationPriority::Medium));
        plan.sort_by_priority();
        assert_eq!(plan.tasks()[0].priority, MigrationPriority::Critical);
        assert_eq!(plan.tasks()[2].priority, MigrationPriority::Low);
    }

    #[test]
    fn test_plan_validate_ok() {
        let mut plan = MigrationPlan::new(1, "P");
        plan.add_task(sample_task(1, MigrationPriority::High));
        let errors = plan.validate();
        assert!(errors.is_empty());
    }

    #[test]
    fn test_plan_validate_errors() {
        let mut plan = MigrationPlan::new(1, "P");
        let mut bad = sample_task(1, MigrationPriority::High);
        bad.source_format = String::new();
        bad.asset_count = 0;
        plan.add_task(bad);
        let errors = plan.validate();
        assert_eq!(errors.len(), 2);
    }

    #[test]
    fn test_plan_summary() {
        let mut plan = MigrationPlan::new(1, "P");
        plan.add_task(sample_task(1, MigrationPriority::High));
        plan.add_task(sample_task(2, MigrationPriority::Low));
        plan.tasks_mut()[0].status = MigrationStatus::Completed;
        let summary = PlanSummary::from_plan(&plan);
        assert_eq!(summary.total_tasks, 2);
        assert_eq!(summary.completed, 1);
        assert_eq!(summary.pending, 1);
        assert!((summary.completion_pct - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_plan_metadata() {
        let mut plan = MigrationPlan::new(1, "P");
        plan.set_metadata("author", "test");
        assert_eq!(
            plan.metadata
                .get("author")
                .expect("operation should succeed"),
            "test"
        );
    }

    #[test]
    fn test_priority_display() {
        assert_eq!(MigrationPriority::Critical.to_string(), "CRITICAL");
        assert_eq!(MigrationPriority::Low.to_string(), "LOW");
    }

    #[test]
    fn test_status_display() {
        assert_eq!(MigrationStatus::InProgress.to_string(), "InProgress");
        assert_eq!(MigrationStatus::Completed.to_string(), "Completed");
    }

    #[test]
    fn test_plan_filter_by_priority() {
        let mut plan = MigrationPlan::new(1, "P");
        plan.add_task(sample_task(1, MigrationPriority::High));
        plan.add_task(sample_task(2, MigrationPriority::Low));
        plan.add_task(sample_task(3, MigrationPriority::High));
        let high = plan.tasks_by_priority(MigrationPriority::High);
        assert_eq!(high.len(), 2);
    }

    #[test]
    fn test_plan_estimated_duration() {
        let mut plan = MigrationPlan::new(1, "P");
        plan.add_task(sample_task(1, MigrationPriority::High));
        plan.add_task(sample_task(2, MigrationPriority::Low));
        assert_eq!(plan.estimated_total_duration(), Duration::from_secs(7200));
    }

    #[test]
    fn test_plan_with_deadline() {
        let deadline = SystemTime::now() + Duration::from_secs(86400);
        let plan = MigrationPlan::new(1, "P").with_deadline(deadline);
        assert!(plan.deadline.is_some());
    }
}
