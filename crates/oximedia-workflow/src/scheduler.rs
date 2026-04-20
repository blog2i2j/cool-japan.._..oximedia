//! Workflow scheduling and triggers.

use crate::error::{Result, WorkflowError};
use crate::workflow::{Workflow, WorkflowId};
use chrono::{DateTime, Utc};
use cron::Schedule;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Trigger type for workflow execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Trigger {
    /// Cron-style schedule.
    Cron {
        /// Cron expression.
        expression: String,
        /// Timezone (defaults to UTC).
        #[serde(default)]
        timezone: String,
    },

    /// Watch folder for file changes.
    WatchFolder {
        /// Folder path to watch.
        path: PathBuf,
        /// File pattern to match (glob).
        #[serde(default)]
        pattern: String,
        /// Whether to watch recursively.
        #[serde(default)]
        recursive: bool,
    },

    /// Manual/API trigger.
    Manual,

    /// Time-based trigger (one-time).
    Time {
        /// Scheduled time.
        at: DateTime<Utc>,
    },

    /// Event-based trigger.
    Event {
        /// Event name/type.
        event_type: String,
        /// Event filter conditions.
        #[serde(default)]
        conditions: HashMap<String, String>,
    },

    /// Interval trigger.
    Interval {
        /// Interval duration in seconds.
        seconds: u64,
    },
}

/// Scheduled workflow.
#[derive(Debug, Clone)]
pub struct ScheduledWorkflow {
    /// Workflow to execute.
    pub workflow: Workflow,
    /// Trigger configuration.
    pub trigger: Trigger,
    /// Whether the schedule is enabled.
    pub enabled: bool,
    /// Last execution time.
    pub last_execution: Option<DateTime<Utc>>,
    /// Next scheduled execution time.
    pub next_execution: Option<DateTime<Utc>>,
}

impl ScheduledWorkflow {
    /// Create a new scheduled workflow.
    #[must_use]
    pub fn new(workflow: Workflow, trigger: Trigger) -> Self {
        let next_execution = Self::calculate_next_execution(&trigger, None);
        Self {
            workflow,
            trigger,
            enabled: true,
            last_execution: None,
            next_execution,
        }
    }

    /// Calculate next execution time.
    fn calculate_next_execution(
        trigger: &Trigger,
        after: Option<DateTime<Utc>>,
    ) -> Option<DateTime<Utc>> {
        let now = after.unwrap_or_else(Utc::now);

        match trigger {
            Trigger::Cron { expression, .. } => {
                if let Ok(schedule) = Schedule::from_str(expression) {
                    schedule.after(&now).next()
                } else {
                    None
                }
            }
            Trigger::Time { at } => {
                if *at > now {
                    Some(*at)
                } else {
                    None
                }
            }
            Trigger::Interval { seconds } => {
                Some(now + chrono::Duration::seconds(i64::try_from(*seconds).unwrap_or(60)))
            }
            Trigger::Manual | Trigger::WatchFolder { .. } | Trigger::Event { .. } => None,
        }
    }

    /// Update next execution time after execution.
    pub fn update_next_execution(&mut self) {
        self.last_execution = Some(Utc::now());
        self.next_execution = Self::calculate_next_execution(&self.trigger, self.last_execution);
    }

    /// Check if workflow should execute now.
    #[must_use]
    pub fn should_execute(&self) -> bool {
        if !self.enabled {
            return false;
        }

        match self.next_execution {
            Some(next) => Utc::now() >= next,
            None => false,
        }
    }
}

/// Workflow scheduler.
pub struct WorkflowScheduler {
    /// Scheduled workflows.
    schedules: Arc<RwLock<HashMap<WorkflowId, ScheduledWorkflow>>>,
    /// Whether scheduler is running.
    running: Arc<RwLock<bool>>,
}

impl WorkflowScheduler {
    /// Create a new scheduler.
    #[must_use]
    pub fn new() -> Self {
        Self {
            schedules: Arc::new(RwLock::new(HashMap::new())),
            running: Arc::new(RwLock::new(false)),
        }
    }

    /// Add a scheduled workflow.
    pub async fn add_schedule(&self, workflow: Workflow, trigger: Trigger) -> Result<WorkflowId> {
        let workflow_id = workflow.id;
        let scheduled = ScheduledWorkflow::new(workflow, trigger);

        let mut schedules = self.schedules.write().await;
        schedules.insert(workflow_id, scheduled);

        info!("Added scheduled workflow: {}", workflow_id);
        Ok(workflow_id)
    }

    /// Remove a scheduled workflow.
    pub async fn remove_schedule(&self, workflow_id: WorkflowId) -> Result<()> {
        let mut schedules = self.schedules.write().await;
        schedules.remove(&workflow_id);
        info!("Removed scheduled workflow: {}", workflow_id);
        Ok(())
    }

    /// Enable/disable a schedule.
    pub async fn set_schedule_enabled(&self, workflow_id: WorkflowId, enabled: bool) -> Result<()> {
        let mut schedules = self.schedules.write().await;
        if let Some(schedule) = schedules.get_mut(&workflow_id) {
            schedule.enabled = enabled;
            debug!("Schedule {} enabled: {}", workflow_id, enabled);
            Ok(())
        } else {
            Err(WorkflowError::WorkflowNotFound(workflow_id.to_string()))
        }
    }

    /// Get all scheduled workflows.
    pub async fn list_schedules(&self) -> Vec<(WorkflowId, ScheduledWorkflow)> {
        let schedules = self.schedules.read().await;
        schedules
            .iter()
            .map(|(id, sched)| (*id, sched.clone()))
            .collect()
    }

    /// Start the scheduler.
    pub async fn start(&self) -> Result<()> {
        let mut running = self.running.write().await;
        if *running {
            return Err(WorkflowError::AlreadyRunning("Scheduler".to_string()));
        }
        *running = true;
        drop(running);

        info!("Scheduler started");
        Ok(())
    }

    /// Stop the scheduler.
    pub async fn stop(&self) -> Result<()> {
        let mut running = self.running.write().await;
        if !*running {
            return Err(WorkflowError::NotRunning("Scheduler".to_string()));
        }
        *running = false;

        info!("Scheduler stopped");
        Ok(())
    }

    /// Check for workflows ready to execute.
    pub async fn check_schedules(&self) -> Vec<Workflow> {
        let running = self.running.read().await;
        if !*running {
            return Vec::new();
        }
        drop(running);

        let mut schedules = self.schedules.write().await;
        let mut ready_workflows = Vec::new();

        for (_, schedule) in schedules.iter_mut() {
            if schedule.should_execute() {
                ready_workflows.push(schedule.workflow.clone());
                schedule.update_next_execution();
                debug!(
                    "Workflow {} ready for execution. Next: {:?}",
                    schedule.workflow.id, schedule.next_execution
                );
            }
        }

        ready_workflows
    }

    /// Get next execution time for a workflow.
    pub async fn get_next_execution(&self, workflow_id: WorkflowId) -> Option<DateTime<Utc>> {
        let schedules = self.schedules.read().await;
        schedules.get(&workflow_id).and_then(|s| s.next_execution)
    }
}

impl Default for WorkflowScheduler {
    fn default() -> Self {
        Self::new()
    }
}

/// File watcher for watch folder triggers.
pub struct FileWatcher {
    /// Watched paths and their workflow IDs.
    watches: Arc<RwLock<HashMap<PathBuf, Vec<WorkflowId>>>>,
}

impl FileWatcher {
    /// Create a new file watcher.
    #[must_use]
    pub fn new() -> Self {
        Self {
            watches: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Add a watch for a path.
    pub async fn add_watch(&self, path: PathBuf, workflow_id: WorkflowId) -> Result<()> {
        let mut watches = self.watches.write().await;
        watches.entry(path.clone()).or_default().push(workflow_id);
        info!("Added file watch: {:?} -> {}", path, workflow_id);
        Ok(())
    }

    /// Remove a watch.
    pub async fn remove_watch(&self, path: &PathBuf, workflow_id: WorkflowId) -> Result<()> {
        let mut watches = self.watches.write().await;
        if let Some(workflows) = watches.get_mut(path) {
            workflows.retain(|id| *id != workflow_id);
            if workflows.is_empty() {
                watches.remove(path);
            }
        }
        info!("Removed file watch: {:?} -> {}", path, workflow_id);
        Ok(())
    }

    /// Get workflows for a path.
    pub async fn get_workflows_for_path(&self, path: &PathBuf) -> Vec<WorkflowId> {
        let watches = self.watches.read().await;
        watches.get(path).cloned().unwrap_or_default()
    }
}

impl Default for FileWatcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cron_trigger_parsing() {
        let trigger = Trigger::Cron {
            expression: "0 0 * * * *".to_string(), // Every hour
            timezone: "UTC".to_string(),
        };

        let next = ScheduledWorkflow::calculate_next_execution(&trigger, None);
        assert!(next.is_some());
    }

    #[test]
    fn test_interval_trigger() {
        let trigger = Trigger::Interval { seconds: 60 };
        let next = ScheduledWorkflow::calculate_next_execution(&trigger, None);
        assert!(next.is_some());
    }

    #[test]
    fn test_time_trigger_future() {
        let future_time = Utc::now() + chrono::Duration::hours(1);
        let trigger = Trigger::Time { at: future_time };
        let next = ScheduledWorkflow::calculate_next_execution(&trigger, None);
        assert_eq!(next, Some(future_time));
    }

    #[test]
    fn test_time_trigger_past() {
        let past_time = Utc::now() - chrono::Duration::hours(1);
        let trigger = Trigger::Time { at: past_time };
        let next = ScheduledWorkflow::calculate_next_execution(&trigger, None);
        assert!(next.is_none());
    }

    #[test]
    fn test_scheduled_workflow_creation() {
        let workflow = Workflow::new("test");
        let trigger = Trigger::Manual;
        let scheduled = ScheduledWorkflow::new(workflow, trigger);

        assert!(scheduled.enabled);
        assert!(scheduled.last_execution.is_none());
    }

    #[tokio::test]
    async fn test_scheduler_creation() {
        let scheduler = WorkflowScheduler::new();
        assert!(!*scheduler.running.read().await);
    }

    #[tokio::test]
    async fn test_add_schedule() {
        let scheduler = WorkflowScheduler::new();
        let workflow = Workflow::new("test-workflow");
        let trigger = Trigger::Manual;

        let workflow_id = scheduler
            .add_schedule(workflow, trigger)
            .await
            .expect("should succeed in test");
        let schedules = scheduler.list_schedules().await;

        assert_eq!(schedules.len(), 1);
        assert_eq!(schedules[0].0, workflow_id);
    }

    #[tokio::test]
    async fn test_remove_schedule() {
        let scheduler = WorkflowScheduler::new();
        let workflow = Workflow::new("test-workflow");
        let trigger = Trigger::Manual;

        let workflow_id = scheduler
            .add_schedule(workflow, trigger)
            .await
            .expect("should succeed in test");
        scheduler
            .remove_schedule(workflow_id)
            .await
            .expect("should succeed in test");

        let schedules = scheduler.list_schedules().await;
        assert_eq!(schedules.len(), 0);
    }

    #[tokio::test]
    async fn test_enable_disable_schedule() {
        let scheduler = WorkflowScheduler::new();
        let workflow = Workflow::new("test-workflow");
        let trigger = Trigger::Manual;

        let workflow_id = scheduler
            .add_schedule(workflow, trigger)
            .await
            .expect("should succeed in test");

        scheduler
            .set_schedule_enabled(workflow_id, false)
            .await
            .expect("should succeed in test");
        let schedules = scheduler.list_schedules().await;
        assert!(!schedules[0].1.enabled);

        scheduler
            .set_schedule_enabled(workflow_id, true)
            .await
            .expect("should succeed in test");
        let schedules = scheduler.list_schedules().await;
        assert!(schedules[0].1.enabled);
    }

    #[tokio::test]
    async fn test_scheduler_start_stop() {
        let scheduler = WorkflowScheduler::new();

        scheduler.start().await.expect("should succeed in test");
        assert!(*scheduler.running.read().await);

        scheduler.stop().await.expect("should succeed in test");
        assert!(!*scheduler.running.read().await);
    }

    #[tokio::test]
    async fn test_file_watcher() {
        let watcher = FileWatcher::new();
        let path = std::env::temp_dir().join("oximedia-workflow-scheduler-test");
        let workflow_id = WorkflowId::new();

        watcher
            .add_watch(path.clone(), workflow_id)
            .await
            .expect("should succeed in test");
        let workflows = watcher.get_workflows_for_path(&path).await;
        assert_eq!(workflows.len(), 1);
        assert_eq!(workflows[0], workflow_id);

        watcher
            .remove_watch(&path, workflow_id)
            .await
            .expect("should succeed in test");
        let workflows = watcher.get_workflows_for_path(&path).await;
        assert_eq!(workflows.len(), 0);
    }
}
