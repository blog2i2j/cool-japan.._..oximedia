//! Workflow orchestration for complex automation sequences.

use crate::{AutomationError, Result};
use async_recursion::async_recursion;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, error, info};

/// Workflow step type.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WorkflowStep {
    /// Play content
    #[serde(rename = "play")]
    Play {
        /// Content ID
        content_id: String,
        /// Duration in seconds
        duration: f64,
    },

    /// Perform switcher cut
    #[serde(rename = "cut")]
    Cut {
        /// Target source
        source: usize,
    },

    /// Trigger device
    #[serde(rename = "trigger")]
    TriggerDevice {
        /// Device ID
        device_id: String,
        /// Command
        command: String,
    },

    /// Wait/delay
    #[serde(rename = "wait")]
    Wait {
        /// Duration in seconds
        duration: f64,
    },

    /// Insert graphics
    #[serde(rename = "graphics")]
    InsertGraphics {
        /// Layer ID
        layer_id: usize,
        /// Template
        template: String,
        /// Data
        data: HashMap<String, String>,
    },

    /// Conditional branch
    #[serde(rename = "conditional")]
    Conditional {
        /// Condition expression
        condition: String,
        /// Steps if true
        if_true: Vec<WorkflowStep>,
        /// Steps if false
        if_false: Vec<WorkflowStep>,
    },

    /// Loop
    #[serde(rename = "loop")]
    Loop {
        /// Number of iterations
        iterations: usize,
        /// Steps to repeat
        steps: Vec<WorkflowStep>,
    },

    /// Execute script
    #[serde(rename = "script")]
    ExecuteScript {
        /// Script content
        script: String,
    },

    /// Log message
    #[serde(rename = "log")]
    Log {
        /// Message
        message: String,
    },
}

/// Workflow definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workflow {
    /// Workflow ID
    pub id: String,
    /// Workflow name
    pub name: String,
    /// Description
    pub description: String,
    /// Steps
    pub steps: Vec<WorkflowStep>,
    /// Metadata
    pub metadata: HashMap<String, String>,
    /// Created time
    #[serde(default = "SystemTime::now")]
    pub created_at: SystemTime,
}

impl Workflow {
    /// Create a new workflow.
    pub fn new(id: String, name: String) -> Self {
        Self {
            id,
            name,
            description: String::new(),
            steps: Vec::new(),
            metadata: HashMap::new(),
            created_at: SystemTime::now(),
        }
    }

    /// Add a step to the workflow.
    pub fn add_step(&mut self, step: WorkflowStep) {
        self.steps.push(step);
    }

    /// Set description.
    pub fn with_description(mut self, description: String) -> Self {
        self.description = description;
        self
    }

    /// Add metadata.
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Validate workflow.
    pub fn validate(&self) -> Result<()> {
        if self.id.is_empty() {
            return Err(AutomationError::InvalidState(
                "Workflow ID cannot be empty".to_string(),
            ));
        }

        if self.steps.is_empty() {
            return Err(AutomationError::InvalidState(
                "Workflow has no steps".to_string(),
            ));
        }

        Ok(())
    }

    /// Estimate total duration.
    pub fn estimate_duration(&self) -> Duration {
        let mut total_secs = 0.0;

        for step in &self.steps {
            match step {
                WorkflowStep::Play { duration, .. } => total_secs += duration,
                WorkflowStep::Wait { duration } => total_secs += duration,
                WorkflowStep::Loop { iterations, steps } => {
                    let loop_duration = Self::estimate_steps_duration(steps);
                    total_secs += loop_duration.as_secs_f64() * (*iterations as f64);
                }
                _ => total_secs += 1.0, // Estimate 1 second for other operations
            }
        }

        Duration::from_secs_f64(total_secs)
    }

    /// Estimate duration of a list of steps.
    fn estimate_steps_duration(steps: &[WorkflowStep]) -> Duration {
        let mut total_secs = 0.0;

        for step in steps {
            match step {
                WorkflowStep::Play { duration, .. } => total_secs += duration,
                WorkflowStep::Wait { duration } => total_secs += duration,
                _ => total_secs += 1.0,
            }
        }

        Duration::from_secs_f64(total_secs)
    }
}

/// Workflow execution status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkflowStatus {
    /// Workflow is pending
    Pending,
    /// Workflow is running
    Running,
    /// Workflow is paused
    Paused,
    /// Workflow completed successfully
    Completed,
    /// Workflow failed
    Failed,
    /// Workflow was cancelled
    Cancelled,
}

/// Workflow execution context.
#[derive(Debug, Clone)]
pub struct WorkflowContext {
    /// Variables
    pub variables: HashMap<String, String>,
    /// Current step index
    pub current_step: usize,
    /// Start time
    pub start_time: SystemTime,
    /// Status
    pub status: WorkflowStatus,
}

impl Default for WorkflowContext {
    fn default() -> Self {
        Self {
            variables: HashMap::new(),
            current_step: 0,
            start_time: SystemTime::now(),
            status: WorkflowStatus::Pending,
        }
    }
}

impl WorkflowContext {
    /// Create a new workflow context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a variable.
    pub fn set_variable(&mut self, key: String, value: String) {
        self.variables.insert(key, value);
    }

    /// Get a variable.
    pub fn get_variable(&self, key: &str) -> Option<&String> {
        self.variables.get(key)
    }

    /// Get elapsed time.
    pub fn elapsed(&self) -> Duration {
        SystemTime::now()
            .duration_since(self.start_time)
            .unwrap_or(Duration::ZERO)
    }
}

/// Workflow executor.
pub struct WorkflowExecutor {
    workflow: Workflow,
    context: Arc<RwLock<WorkflowContext>>,
    event_tx: mpsc::UnboundedSender<WorkflowEvent>,
}

/// Workflow event.
#[derive(Debug, Clone)]
pub enum WorkflowEvent {
    /// Workflow started
    Started,
    /// Step started
    StepStarted {
        /// Step index
        index: usize,
    },
    /// Step completed
    StepCompleted {
        /// Step index
        index: usize,
    },
    /// Step failed
    StepFailed {
        /// Step index
        index: usize,
        /// Error message
        error: String,
    },
    /// Workflow completed
    Completed,
    /// Workflow failed
    Failed {
        /// Error message
        error: String,
    },
    /// Workflow paused
    Paused,
    /// Workflow resumed
    Resumed,
    /// Workflow cancelled
    Cancelled,
}

impl WorkflowExecutor {
    /// Create a new workflow executor.
    pub fn new(workflow: Workflow) -> (Self, mpsc::UnboundedReceiver<WorkflowEvent>) {
        let (tx, rx) = mpsc::unbounded_channel();

        let executor = Self {
            workflow,
            context: Arc::new(RwLock::new(WorkflowContext::new())),
            event_tx: tx,
        };

        (executor, rx)
    }

    /// Start workflow execution.
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting workflow: {}", self.workflow.name);

        self.workflow.validate()?;

        {
            let mut ctx = self.context.write().await;
            ctx.status = WorkflowStatus::Running;
            ctx.start_time = SystemTime::now();
        }

        let _ = self.event_tx.send(WorkflowEvent::Started);

        // Execute all steps
        for (index, step) in self.workflow.steps.iter().enumerate() {
            {
                let mut ctx = self.context.write().await;
                ctx.current_step = index;
            }

            let _ = self.event_tx.send(WorkflowEvent::StepStarted { index });

            match self.execute_step(step).await {
                Ok(()) => {
                    let _ = self.event_tx.send(WorkflowEvent::StepCompleted { index });
                }
                Err(e) => {
                    error!("Step {} failed: {}", index, e);
                    let _ = self.event_tx.send(WorkflowEvent::StepFailed {
                        index,
                        error: e.to_string(),
                    });

                    let mut ctx = self.context.write().await;
                    ctx.status = WorkflowStatus::Failed;

                    let _ = self.event_tx.send(WorkflowEvent::Failed {
                        error: e.to_string(),
                    });

                    return Err(e);
                }
            }
        }

        {
            let mut ctx = self.context.write().await;
            ctx.status = WorkflowStatus::Completed;
        }

        let _ = self.event_tx.send(WorkflowEvent::Completed);

        info!("Workflow completed: {}", self.workflow.name);

        Ok(())
    }

    /// Execute a single workflow step.
    #[async_recursion]
    async fn execute_step(&self, step: &WorkflowStep) -> Result<()> {
        debug!("Executing step: {:?}", step);

        match step {
            WorkflowStep::Play {
                content_id,
                duration,
            } => {
                info!("Playing content: {} for {}s", content_id, duration);
                tokio::time::sleep(Duration::from_secs_f64(*duration)).await;
            }

            WorkflowStep::Cut { source } => {
                info!("Cutting to source: {}", source);
                // In a real implementation, this would control the actual switcher
            }

            WorkflowStep::TriggerDevice { device_id, command } => {
                info!("Triggering device {}: {}", device_id, command);
                // In a real implementation, this would control the actual device
            }

            WorkflowStep::Wait { duration } => {
                info!("Waiting for {}s", duration);
                tokio::time::sleep(Duration::from_secs_f64(*duration)).await;
            }

            WorkflowStep::InsertGraphics {
                layer_id,
                template,
                data,
            } => {
                info!(
                    "Inserting graphics: layer={}, template={}",
                    layer_id, template
                );
                debug!("Graphics data: {:?}", data);
                // In a real implementation, this would control the graphics system
            }

            WorkflowStep::Conditional {
                condition,
                if_true,
                if_false,
            } => {
                info!("Evaluating condition: {}", condition);

                // Simple condition evaluation (in production, use a proper expression parser)
                let result = Self::evaluate_condition(condition, &self.context).await;

                let steps = if result { if_true } else { if_false };

                for substep in steps {
                    self.execute_step(substep).await?;
                }
            }

            WorkflowStep::Loop { iterations, steps } => {
                info!("Starting loop: {} iterations", iterations);

                for i in 0..*iterations {
                    debug!("Loop iteration: {}/{}", i + 1, iterations);

                    for substep in steps {
                        self.execute_step(substep).await?;
                    }
                }
            }

            WorkflowStep::ExecuteScript { script } => {
                info!("Executing script");
                debug!("Script: {}", script);
                // In a real implementation, this would use the script engine
            }

            WorkflowStep::Log { message } => {
                info!("Workflow log: {}", message);
            }
        }

        Ok(())
    }

    /// Evaluate a condition.
    async fn evaluate_condition(_condition: &str, _context: &Arc<RwLock<WorkflowContext>>) -> bool {
        // Simple implementation - in production, use a proper expression evaluator
        true
    }

    /// Pause workflow execution.
    pub async fn pause(&mut self) -> Result<()> {
        info!("Pausing workflow");

        let mut ctx = self.context.write().await;
        ctx.status = WorkflowStatus::Paused;

        let _ = self.event_tx.send(WorkflowEvent::Paused);

        Ok(())
    }

    /// Resume workflow execution.
    pub async fn resume(&mut self) -> Result<()> {
        info!("Resuming workflow");

        let mut ctx = self.context.write().await;
        ctx.status = WorkflowStatus::Running;

        let _ = self.event_tx.send(WorkflowEvent::Resumed);

        Ok(())
    }

    /// Cancel workflow execution.
    pub async fn cancel(&mut self) -> Result<()> {
        info!("Cancelling workflow");

        let mut ctx = self.context.write().await;
        ctx.status = WorkflowStatus::Cancelled;

        let _ = self.event_tx.send(WorkflowEvent::Cancelled);

        Ok(())
    }

    /// Get current status.
    pub async fn status(&self) -> WorkflowStatus {
        self.context.read().await.status
    }

    /// Get workflow progress (0.0 - 1.0).
    pub async fn progress(&self) -> f64 {
        let ctx = self.context.read().await;
        if self.workflow.steps.is_empty() {
            return 0.0;
        }

        ctx.current_step as f64 / self.workflow.steps.len() as f64
    }
}

/// Workflow builder for fluent API.
pub struct WorkflowBuilder {
    workflow: Workflow,
}

impl WorkflowBuilder {
    /// Create a new workflow builder.
    pub fn new(id: String, name: String) -> Self {
        Self {
            workflow: Workflow::new(id, name),
        }
    }

    /// Set description.
    pub fn description(mut self, description: String) -> Self {
        self.workflow.description = description;
        self
    }

    /// Add play step.
    pub fn play(mut self, content_id: String, duration: f64) -> Self {
        self.workflow.add_step(WorkflowStep::Play {
            content_id,
            duration,
        });
        self
    }

    /// Add cut step.
    pub fn cut(mut self, source: usize) -> Self {
        self.workflow.add_step(WorkflowStep::Cut { source });
        self
    }

    /// Add wait step.
    pub fn wait(mut self, duration: f64) -> Self {
        self.workflow.add_step(WorkflowStep::Wait { duration });
        self
    }

    /// Add log step.
    pub fn log(mut self, message: String) -> Self {
        self.workflow.add_step(WorkflowStep::Log { message });
        self
    }

    /// Add trigger device step.
    pub fn trigger_device(mut self, device_id: String, command: String) -> Self {
        self.workflow
            .add_step(WorkflowStep::TriggerDevice { device_id, command });
        self
    }

    /// Add graphics step.
    pub fn insert_graphics(
        mut self,
        layer_id: usize,
        template: String,
        data: HashMap<String, String>,
    ) -> Self {
        self.workflow.add_step(WorkflowStep::InsertGraphics {
            layer_id,
            template,
            data,
        });
        self
    }

    /// Add metadata.
    pub fn metadata(mut self, key: String, value: String) -> Self {
        self.workflow.metadata.insert(key, value);
        self
    }

    /// Build the workflow.
    pub fn build(self) -> Result<Workflow> {
        self.workflow.validate()?;
        Ok(self.workflow)
    }
}

/// Workflow templates for common scenarios.
pub struct WorkflowTemplates;

impl WorkflowTemplates {
    /// Create a news program workflow.
    pub fn news_program() -> Workflow {
        let mut workflow = Workflow::new("news_program".to_string(), "News Program".to_string());

        workflow.description = "Standard news program workflow".to_string();

        // Opening sequence
        workflow.add_step(WorkflowStep::Log {
            message: "Starting news program".to_string(),
        });

        workflow.add_step(WorkflowStep::InsertGraphics {
            layer_id: 1,
            template: "news_open.xml".to_string(),
            data: HashMap::new(),
        });

        workflow.add_step(WorkflowStep::Wait { duration: 5.0 });

        // Main content
        workflow.add_step(WorkflowStep::Play {
            content_id: "news_main.mxf".to_string(),
            duration: 3600.0,
        });

        // Closing sequence
        workflow.add_step(WorkflowStep::InsertGraphics {
            layer_id: 1,
            template: "news_close.xml".to_string(),
            data: HashMap::new(),
        });

        workflow.add_step(WorkflowStep::Wait { duration: 5.0 });

        workflow.add_step(WorkflowStep::Log {
            message: "News program complete".to_string(),
        });

        workflow
    }

    /// Create a commercial break workflow.
    pub fn commercial_break(num_spots: usize) -> Workflow {
        let mut workflow = Workflow::new(
            "commercial_break".to_string(),
            "Commercial Break".to_string(),
        );

        workflow.description = format!("Commercial break with {num_spots} spots");

        workflow.add_step(WorkflowStep::Log {
            message: "Starting commercial break".to_string(),
        });

        let mut spots = Vec::new();
        for i in 0..num_spots {
            spots.push(WorkflowStep::Play {
                content_id: format!("commercial_{}.mxf", i + 1),
                duration: 30.0,
            });
        }

        workflow.add_step(WorkflowStep::Loop {
            iterations: 1,
            steps: spots,
        });

        workflow.add_step(WorkflowStep::Log {
            message: "Commercial break complete".to_string(),
        });

        workflow
    }

    /// Create a live segment workflow.
    pub fn live_segment(duration: f64) -> Workflow {
        let mut workflow = Workflow::new("live_segment".to_string(), "Live Segment".to_string());

        workflow.description = "Live production segment".to_string();

        workflow.add_step(WorkflowStep::Log {
            message: "Preparing for live segment".to_string(),
        });

        // Pre-roll time
        workflow.add_step(WorkflowStep::Wait { duration: 5.0 });

        // Go live
        workflow.add_step(WorkflowStep::Cut { source: 1 });

        workflow.add_step(WorkflowStep::TriggerDevice {
            device_id: "camera_tally".to_string(),
            command: "on".to_string(),
        });

        workflow.add_step(WorkflowStep::Log {
            message: "Live segment in progress".to_string(),
        });

        // Live duration
        workflow.add_step(WorkflowStep::Wait { duration });

        // End live
        workflow.add_step(WorkflowStep::TriggerDevice {
            device_id: "camera_tally".to_string(),
            command: "off".to_string(),
        });

        workflow.add_step(WorkflowStep::Cut { source: 0 });

        workflow.add_step(WorkflowStep::Log {
            message: "Live segment complete".to_string(),
        });

        workflow
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workflow_creation() {
        let workflow = Workflow::new("test".to_string(), "Test Workflow".to_string());
        assert_eq!(workflow.id, "test");
        assert_eq!(workflow.name, "Test Workflow");
        assert!(workflow.steps.is_empty());
    }

    #[test]
    fn test_workflow_builder() {
        let workflow = WorkflowBuilder::new("test".to_string(), "Test".to_string())
            .description("Test workflow".to_string())
            .log("Starting".to_string())
            .play("content.mxf".to_string(), 60.0)
            .wait(5.0)
            .log("Complete".to_string())
            .build();

        assert!(workflow.is_ok());
        let workflow = workflow.expect("workflow should be valid");
        assert_eq!(workflow.steps.len(), 4);
    }

    #[test]
    fn test_workflow_validation() {
        let mut workflow = Workflow::new("test".to_string(), "Test".to_string());
        workflow.add_step(WorkflowStep::Log {
            message: "Test".to_string(),
        });

        assert!(workflow.validate().is_ok());

        let empty_workflow = Workflow::new("test".to_string(), "Test".to_string());
        assert!(empty_workflow.validate().is_err());
    }

    #[test]
    fn test_duration_estimation() {
        let workflow = WorkflowBuilder::new("test".to_string(), "Test".to_string())
            .play("content.mxf".to_string(), 60.0)
            .wait(10.0)
            .build()
            .expect("operation should succeed");

        let duration = workflow.estimate_duration();
        assert_eq!(duration.as_secs(), 70);
    }

    #[test]
    fn test_workflow_templates() {
        let news = WorkflowTemplates::news_program();
        assert!(!news.steps.is_empty());

        let commercials = WorkflowTemplates::commercial_break(3);
        assert!(!commercials.steps.is_empty());

        let live = WorkflowTemplates::live_segment(300.0);
        assert!(!live.steps.is_empty());
    }

    #[tokio::test]
    async fn test_workflow_execution() {
        let workflow = WorkflowBuilder::new("test".to_string(), "Test".to_string())
            .log("Starting test".to_string())
            .wait(0.1)
            .log("Test complete".to_string())
            .build()
            .expect("operation should succeed");

        let (mut executor, _rx) = WorkflowExecutor::new(workflow);

        assert!(executor.start().await.is_ok());
        assert_eq!(executor.status().await, WorkflowStatus::Completed);
    }
}
