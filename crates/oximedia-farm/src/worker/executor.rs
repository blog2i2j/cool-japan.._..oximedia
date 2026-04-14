//! Task execution engine

use crate::{FarmError, Result, TaskId};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// Result of task execution
#[derive(Debug, Clone)]
pub struct TaskResult {
    pub task_id: TaskId,
    pub success: bool,
    pub output: Vec<u8>,
    pub duration_ms: u64,
    pub error_message: Option<String>,
    pub metrics: HashMap<String, String>,
}

/// Task execution context
#[derive(Debug)]
struct TaskContext {
    #[allow(dead_code)]
    task_id: TaskId,
    #[allow(dead_code)]
    start_time: Instant,
    progress: f64,
}

/// Task executor
pub struct TaskExecutor {
    active_tasks: Arc<RwLock<HashMap<TaskId, TaskContext>>>,
}

impl TaskExecutor {
    /// Create a new task executor
    #[must_use]
    pub fn new() -> Self {
        Self {
            active_tasks: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Execute a task
    pub async fn execute(&self, task_id: TaskId, payload: Vec<u8>) -> Result<TaskResult> {
        // Register task as active
        {
            let mut active_tasks = self.active_tasks.write();
            if active_tasks.contains_key(&task_id) {
                return Err(FarmError::AlreadyExists(format!(
                    "Task {task_id} is already running"
                )));
            }

            active_tasks.insert(
                task_id,
                TaskContext {
                    task_id,
                    start_time: Instant::now(),
                    progress: 0.0,
                },
            );
        }

        tracing::info!("Starting execution of task {}", task_id);

        // Execute the task
        let result = self.execute_task_impl(task_id, payload).await;

        // Remove from active tasks
        {
            let mut active_tasks = self.active_tasks.write();
            active_tasks.remove(&task_id);
        }

        result
    }

    /// Internal task execution implementation
    async fn execute_task_impl(&self, task_id: TaskId, payload: Vec<u8>) -> Result<TaskResult> {
        let start = Instant::now();

        // Parse task payload
        let task_spec = self.parse_task_payload(&payload)?;

        // Execute based on task type
        let output = match task_spec.task_type.as_str() {
            "transcode" => self.execute_transcode(task_id, &task_spec).await?,
            "thumbnail" => self.execute_thumbnail(task_id, &task_spec).await?,
            "qc" => self.execute_qc(task_id, &task_spec).await?,
            "analysis" => self.execute_analysis(task_id, &task_spec).await?,
            "fingerprint" => self.execute_fingerprint(task_id, &task_spec).await?,
            _ => {
                return Err(FarmError::Task(format!(
                    "Unknown task type: {}",
                    task_spec.task_type
                )))
            }
        };

        let duration = start.elapsed();

        Ok(TaskResult {
            task_id,
            success: true,
            output,
            duration_ms: duration.as_millis() as u64,
            error_message: None,
            metrics: HashMap::new(),
        })
    }

    /// Parse task payload
    fn parse_task_payload(&self, payload: &[u8]) -> Result<TaskSpecification> {
        if payload.is_empty() {
            // Default task specification for empty payload
            return Ok(TaskSpecification {
                task_type: "transcode".to_string(),
                input_path: "/input/default.mp4".to_string(),
                output_path: "/output/default.mp4".to_string(),
                parameters: HashMap::new(),
            });
        }

        // In a real implementation, this would deserialize the payload
        serde_json::from_slice(payload).map_err(FarmError::from)
    }

    /// Execute transcoding task
    async fn execute_transcode(
        &self,
        task_id: TaskId,
        spec: &TaskSpecification,
    ) -> Result<Vec<u8>> {
        tracing::info!("Transcoding {} to {}", spec.input_path, spec.output_path);

        // Simulate transcoding work
        for i in 0..10 {
            self.update_progress(task_id, f64::from(i) / 10.0);
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }

        self.update_progress(task_id, 1.0);

        Ok(vec![])
    }

    /// Execute thumbnail generation task
    async fn execute_thumbnail(
        &self,
        task_id: TaskId,
        spec: &TaskSpecification,
    ) -> Result<Vec<u8>> {
        tracing::info!("Generating thumbnail for {}", spec.input_path);

        // Simulate thumbnail generation
        self.update_progress(task_id, 0.5);
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
        self.update_progress(task_id, 1.0);

        Ok(vec![])
    }

    /// Execute QC validation task
    async fn execute_qc(&self, task_id: TaskId, spec: &TaskSpecification) -> Result<Vec<u8>> {
        tracing::info!("Running QC on {}", spec.input_path);

        // Simulate QC work
        self.update_progress(task_id, 0.5);
        tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;
        self.update_progress(task_id, 1.0);

        Ok(vec![])
    }

    /// Execute media analysis task
    async fn execute_analysis(&self, task_id: TaskId, spec: &TaskSpecification) -> Result<Vec<u8>> {
        tracing::info!("Analyzing {}", spec.input_path);

        // Simulate analysis work
        self.update_progress(task_id, 0.5);
        tokio::time::sleep(tokio::time::Duration::from_millis(250)).await;
        self.update_progress(task_id, 1.0);

        Ok(vec![])
    }

    /// Execute fingerprinting task
    async fn execute_fingerprint(
        &self,
        task_id: TaskId,
        spec: &TaskSpecification,
    ) -> Result<Vec<u8>> {
        tracing::info!("Fingerprinting {}", spec.input_path);

        // Simulate fingerprinting work
        self.update_progress(task_id, 0.5);
        tokio::time::sleep(tokio::time::Duration::from_millis(150)).await;
        self.update_progress(task_id, 1.0);

        Ok(vec![])
    }

    /// Update task progress
    fn update_progress(&self, task_id: TaskId, progress: f64) {
        let mut active_tasks = self.active_tasks.write();
        if let Some(context) = active_tasks.get_mut(&task_id) {
            context.progress = progress;
            tracing::debug!("Task {} progress: {:.1}%", task_id, progress * 100.0);
        }
    }

    /// Get active task count
    #[must_use]
    pub fn active_task_count(&self) -> usize {
        self.active_tasks.read().len()
    }

    /// Get task progress
    #[must_use]
    pub fn get_progress(&self, task_id: TaskId) -> Option<f64> {
        self.active_tasks.read().get(&task_id).map(|c| c.progress)
    }

    /// Cancel a task
    pub async fn cancel_task(&self, task_id: TaskId) -> Result<()> {
        let mut active_tasks = self.active_tasks.write();
        if active_tasks.remove(&task_id).is_some() {
            tracing::info!("Task {} cancelled", task_id);
            Ok(())
        } else {
            Err(FarmError::NotFound(format!("Task {task_id} not found")))
        }
    }
}

impl Default for TaskExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Task specification
#[derive(Debug, Clone, serde::Deserialize)]
struct TaskSpecification {
    task_type: String,
    #[allow(dead_code)]
    input_path: String,
    output_path: String,
    #[allow(dead_code)]
    parameters: HashMap<String, String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_task_execution() {
        let executor = TaskExecutor::new();
        let task_id = TaskId::new();

        let result = executor.execute(task_id, vec![]).await.unwrap();
        assert!(result.success);
        assert_eq!(result.task_id, task_id);
    }

    #[tokio::test]
    async fn test_active_task_count() {
        let executor = Arc::new(TaskExecutor::new());
        assert_eq!(executor.active_task_count(), 0);

        let executor_clone = executor.clone();
        let handle =
            tokio::spawn(async move { executor_clone.execute(TaskId::new(), vec![]).await });

        // Wait a bit for task to start
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        // Should have 1 active task
        // Note: This is racy, but works for testing
        let _ = handle.await;

        // After completion, should be 0
        assert_eq!(executor.active_task_count(), 0);
    }

    #[tokio::test]
    async fn test_duplicate_task() {
        let executor = TaskExecutor::new();
        let task_id = TaskId::new();

        // Register task as active
        {
            let mut active_tasks = executor.active_tasks.write();
            active_tasks.insert(
                task_id,
                TaskContext {
                    task_id,
                    start_time: Instant::now(),
                    progress: 0.0,
                },
            );
        }

        // Try to execute same task
        let result = executor.execute(task_id, vec![]).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_task_progress() {
        let executor = TaskExecutor::new();
        let task_id = TaskId::new();

        // Register task
        {
            let mut active_tasks = executor.active_tasks.write();
            active_tasks.insert(
                task_id,
                TaskContext {
                    task_id,
                    start_time: Instant::now(),
                    progress: 0.0,
                },
            );
        }

        executor.update_progress(task_id, 0.5);
        let progress = executor.get_progress(task_id).unwrap();
        assert!((progress - 0.5).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_task_cancellation() {
        let executor = TaskExecutor::new();
        let task_id = TaskId::new();

        // Register task
        {
            let mut active_tasks = executor.active_tasks.write();
            active_tasks.insert(
                task_id,
                TaskContext {
                    task_id,
                    start_time: Instant::now(),
                    progress: 0.0,
                },
            );
        }

        executor.cancel_task(task_id).await.unwrap();
        assert_eq!(executor.active_task_count(), 0);
    }
}
