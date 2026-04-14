#![allow(dead_code)]
//! Workflow DSL — define complex job workflows in a structured format.
//!
//! This module provides a YAML/JSON-compatible workflow definition language
//! with support for sequential stages, parallel fan-out, conditional branching,
//! and dependency declarations.
//!
//! # Example (JSON representation)
//! ```json
//! {
//!   "name": "ingest-pipeline",
//!   "steps": [
//!     { "name": "transcode", "job_type": "transcode", "params": { "preset": "hd" } },
//!     { "name": "thumbnail", "job_type": "thumbnail", "depends_on": ["transcode"] }
//!   ]
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Error type for workflow DSL operations.
#[derive(Debug, thiserror::Error)]
pub enum WorkflowDslError {
    /// Workflow definition parse error.
    #[error("Parse error: {0}")]
    Parse(String),
    /// A step references an unknown dependency.
    #[error("Unknown dependency '{dep}' in step '{step}'")]
    UnknownDependency { step: String, dep: String },
    /// The workflow dependency graph contains a cycle.
    #[error("Cycle detected involving step: {0}")]
    CyclicDependency(String),
    /// Duplicate step name in the same workflow.
    #[error("Duplicate step name: {0}")]
    DuplicateStep(String),
    /// Execution error for a step.
    #[error("Step '{step}' failed: {reason}")]
    StepFailed { step: String, reason: String },
}

/// The status of a workflow step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepStatus {
    /// Waiting for dependencies.
    Waiting,
    /// Ready to execute.
    Ready,
    /// Currently running.
    Running,
    /// Completed successfully.
    Completed,
    /// Failed.
    Failed,
    /// Skipped (due to a conditional or upstream failure).
    Skipped,
}

/// A single step in a workflow definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowStep {
    /// Unique name within this workflow.
    pub name: String,
    /// Job type identifier (maps to a `JobPayload` variant).
    pub job_type: String,
    /// Key/value parameters for the step.
    #[serde(default)]
    pub params: HashMap<String, String>,
    /// Names of steps that must complete before this one starts.
    #[serde(default)]
    pub depends_on: Vec<String>,
    /// Optional condition expression (simple string, e.g. "prev.status == completed").
    #[serde(default)]
    pub condition: Option<String>,
    /// Whether a failure here should abort the entire workflow.
    #[serde(default = "default_true")]
    pub required: bool,
    /// Maximum number of retries.
    #[serde(default)]
    pub max_retries: u32,
    /// Timeout in seconds (0 = no timeout).
    #[serde(default)]
    pub timeout_secs: u64,
}

fn default_true() -> bool {
    true
}

impl WorkflowStep {
    /// Create a minimal step with the given name and job type.
    #[must_use]
    pub fn new(name: impl Into<String>, job_type: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            job_type: job_type.into(),
            params: HashMap::new(),
            depends_on: Vec::new(),
            condition: None,
            required: true,
            max_retries: 0,
            timeout_secs: 0,
        }
    }

    /// Add a dependency.
    pub fn depends_on(mut self, dep: impl Into<String>) -> Self {
        self.depends_on.push(dep.into());
        self
    }

    /// Add a parameter.
    pub fn with_param(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.params.insert(key.into(), value.into());
        self
    }

    /// Set as optional (failure does not abort workflow).
    pub fn optional(mut self) -> Self {
        self.required = false;
        self
    }

    /// Set a condition expression.
    pub fn with_condition(mut self, cond: impl Into<String>) -> Self {
        self.condition = Some(cond.into());
        self
    }
}

/// A complete workflow definition.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WorkflowDefinition {
    /// Human-readable workflow name.
    pub name: String,
    /// Optional description.
    #[serde(default)]
    pub description: String,
    /// Version string.
    #[serde(default = "default_version")]
    pub version: String,
    /// Ordered list of steps (order matters for step lookup, not execution order).
    pub steps: Vec<WorkflowStep>,
    /// Global key/value variables available to all steps.
    #[serde(default)]
    pub variables: HashMap<String, String>,
}

fn default_version() -> String {
    "1.0".to_string()
}

impl WorkflowDefinition {
    /// Create a new workflow.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::default()
        }
    }

    /// Append a step.
    pub fn step(mut self, step: WorkflowStep) -> Self {
        self.steps.push(step);
        self
    }

    /// Set a global variable.
    pub fn var(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.variables.insert(key.into(), value.into());
        self
    }

    /// Serialise to JSON.
    pub fn to_json(&self) -> Result<String, WorkflowDslError> {
        serde_json::to_string_pretty(self).map_err(|e| WorkflowDslError::Parse(e.to_string()))
    }

    /// Deserialise from JSON.
    pub fn from_json(json: &str) -> Result<Self, WorkflowDslError> {
        serde_json::from_str(json).map_err(|e| WorkflowDslError::Parse(e.to_string()))
    }

    /// Validate the workflow: check for duplicate step names and unknown dependencies,
    /// and detect cycles via DFS.
    pub fn validate(&self) -> Result<(), WorkflowDslError> {
        // Check for duplicate step names
        let mut seen = HashSet::new();
        for step in &self.steps {
            if !seen.insert(step.name.clone()) {
                return Err(WorkflowDslError::DuplicateStep(step.name.clone()));
            }
        }
        let step_names: HashSet<&str> = self.steps.iter().map(|s| s.name.as_str()).collect();
        // Check for unknown dependencies
        for step in &self.steps {
            for dep in &step.depends_on {
                if !step_names.contains(dep.as_str()) {
                    return Err(WorkflowDslError::UnknownDependency {
                        step: step.name.clone(),
                        dep: dep.clone(),
                    });
                }
            }
        }
        // Cycle detection via DFS (Kahn-style)
        self.topological_order().map(|_| ())
    }

    /// Produce a topological order of steps respecting dependencies.
    ///
    /// Returns `Err(CyclicDependency)` if a cycle is detected.
    pub fn topological_order(&self) -> Result<Vec<&str>, WorkflowDslError> {
        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
        for step in &self.steps {
            in_degree.entry(step.name.as_str()).or_insert(0);
            for dep in &step.depends_on {
                // dep → step
                adj.entry(dep.as_str())
                    .or_default()
                    .push(step.name.as_str());
                *in_degree.entry(step.name.as_str()).or_insert(0) += 1;
            }
        }
        let mut queue: std::collections::VecDeque<&str> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&name, _)| name)
            .collect();
        let mut order = Vec::new();
        while let Some(node) = queue.pop_front() {
            order.push(node);
            if let Some(dependents) = adj.get(node) {
                for &dep in dependents {
                    let deg = in_degree.entry(dep).or_insert(0);
                    *deg = deg.saturating_sub(1);
                    if *deg == 0 {
                        queue.push_back(dep);
                    }
                }
            }
        }
        if order.len() != self.steps.len() {
            // Find a step still with in_degree > 0
            let cyclic = in_degree
                .iter()
                .find(|(_, &d)| d > 0)
                .map(|(&n, _)| n.to_string())
                .unwrap_or_else(|| "unknown".to_string());
            return Err(WorkflowDslError::CyclicDependency(cyclic));
        }
        Ok(order)
    }

    /// Return steps that are ready to execute given the set of already-completed step names.
    #[must_use]
    pub fn ready_steps<'a>(&'a self, completed: &HashSet<String>) -> Vec<&'a WorkflowStep> {
        self.steps
            .iter()
            .filter(|s| {
                !completed.contains(&s.name)
                    && s.depends_on.iter().all(|d| completed.contains(d))
            })
            .collect()
    }
}

/// Runtime state of a workflow execution.
#[derive(Debug, Default)]
pub struct WorkflowRuntime {
    /// Status of each step.
    pub step_status: HashMap<String, StepStatus>,
    /// Output values produced by each step.
    pub step_outputs: HashMap<String, HashMap<String, String>>,
}

impl WorkflowRuntime {
    /// Create a new runtime for the given workflow.
    pub fn new(workflow: &WorkflowDefinition) -> Self {
        let step_status = workflow
            .steps
            .iter()
            .map(|s| (s.name.clone(), StepStatus::Waiting))
            .collect();
        Self {
            step_status,
            step_outputs: HashMap::new(),
        }
    }

    /// Mark a step as completed with optional output.
    pub fn complete_step(&mut self, step_name: &str, output: HashMap<String, String>) {
        self.step_status
            .insert(step_name.to_string(), StepStatus::Completed);
        self.step_outputs.insert(step_name.to_string(), output);
    }

    /// Mark a step as failed.
    pub fn fail_step(&mut self, step_name: &str) {
        self.step_status
            .insert(step_name.to_string(), StepStatus::Failed);
    }

    /// Mark a step as skipped.
    pub fn skip_step(&mut self, step_name: &str) {
        self.step_status
            .insert(step_name.to_string(), StepStatus::Skipped);
    }

    /// Get step status.
    #[must_use]
    pub fn status(&self, step_name: &str) -> Option<StepStatus> {
        self.step_status.get(step_name).copied()
    }

    /// Completed step names.
    #[must_use]
    pub fn completed_steps(&self) -> HashSet<String> {
        self.step_status
            .iter()
            .filter(|(_, &s)| s == StepStatus::Completed)
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Number of completed steps.
    #[must_use]
    pub fn completed_count(&self) -> usize {
        self.step_status
            .values()
            .filter(|&&s| s == StepStatus::Completed)
            .count()
    }

    /// Number of failed steps.
    #[must_use]
    pub fn failed_count(&self) -> usize {
        self.step_status
            .values()
            .filter(|&&s| s == StepStatus::Failed)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_workflow() -> WorkflowDefinition {
        WorkflowDefinition::new("test-wf")
            .step(WorkflowStep::new("ingest", "ingest"))
            .step(WorkflowStep::new("transcode", "transcode").depends_on("ingest"))
            .step(WorkflowStep::new("thumbnail", "thumbnail").depends_on("transcode"))
    }

    #[test]
    fn test_workflow_validate_valid() {
        assert!(simple_workflow().validate().is_ok());
    }

    #[test]
    fn test_workflow_validate_unknown_dep() {
        let wf = WorkflowDefinition::new("wf")
            .step(WorkflowStep::new("a", "a").depends_on("nonexistent"));
        assert!(matches!(
            wf.validate(),
            Err(WorkflowDslError::UnknownDependency { .. })
        ));
    }

    #[test]
    fn test_workflow_validate_duplicate_step() {
        let wf = WorkflowDefinition::new("wf")
            .step(WorkflowStep::new("a", "a"))
            .step(WorkflowStep::new("a", "b")); // duplicate name
        assert!(matches!(
            wf.validate(),
            Err(WorkflowDslError::DuplicateStep(_))
        ));
    }

    #[test]
    fn test_topological_order_valid() {
        let wf = simple_workflow();
        let order = wf.topological_order().expect("topo order");
        assert_eq!(order.len(), 3);
        // ingest must come before transcode
        let pos_ingest = order.iter().position(|&n| n == "ingest").expect("ingest");
        let pos_transcode = order.iter().position(|&n| n == "transcode").expect("transcode");
        let pos_thumb = order.iter().position(|&n| n == "thumbnail").expect("thumbnail");
        assert!(pos_ingest < pos_transcode);
        assert!(pos_transcode < pos_thumb);
    }

    #[test]
    fn test_topological_order_cycle() {
        let wf = WorkflowDefinition::new("cyclic")
            .step(WorkflowStep::new("a", "a").depends_on("b"))
            .step(WorkflowStep::new("b", "b").depends_on("a"));
        assert!(matches!(
            wf.topological_order(),
            Err(WorkflowDslError::CyclicDependency(_))
        ));
    }

    #[test]
    fn test_ready_steps_initial() {
        let wf = simple_workflow();
        let completed = HashSet::new();
        let ready = wf.ready_steps(&completed);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].name, "ingest");
    }

    #[test]
    fn test_ready_steps_after_ingest() {
        let wf = simple_workflow();
        let mut completed = HashSet::new();
        completed.insert("ingest".to_string());
        let ready = wf.ready_steps(&completed);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].name, "transcode");
    }

    #[test]
    fn test_ready_steps_all_done_empty() {
        let wf = simple_workflow();
        let completed: HashSet<String> = ["ingest", "transcode", "thumbnail"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let ready = wf.ready_steps(&completed);
        assert!(ready.is_empty());
    }

    #[test]
    fn test_json_roundtrip() {
        let wf = simple_workflow();
        let json = wf.to_json().expect("to_json");
        let restored = WorkflowDefinition::from_json(&json).expect("from_json");
        assert_eq!(restored.name, wf.name);
        assert_eq!(restored.steps.len(), wf.steps.len());
    }

    #[test]
    fn test_runtime_complete_and_fail() {
        let wf = simple_workflow();
        let mut rt = WorkflowRuntime::new(&wf);
        rt.complete_step("ingest", HashMap::new());
        rt.fail_step("transcode");
        assert_eq!(rt.completed_count(), 1);
        assert_eq!(rt.failed_count(), 1);
    }

    #[test]
    fn test_workflow_variables() {
        let wf = WorkflowDefinition::new("wf")
            .var("bucket", "s3://my-bucket")
            .var("region", "eu-west-1");
        assert_eq!(wf.variables.get("bucket").map(String::as_str), Some("s3://my-bucket"));
        assert_eq!(wf.variables.len(), 2);
    }

    #[test]
    fn test_step_builder() {
        let step = WorkflowStep::new("encode", "h264")
            .depends_on("ingest")
            .with_param("preset", "fast")
            .optional()
            .with_condition("prev.status == completed");
        assert!(!step.required);
        assert_eq!(step.depends_on, vec!["ingest"]);
        assert_eq!(step.params.get("preset").map(String::as_str), Some("fast"));
        assert!(step.condition.is_some());
    }

    #[test]
    fn test_runtime_completed_steps_set() {
        let wf = simple_workflow();
        let mut rt = WorkflowRuntime::new(&wf);
        rt.complete_step("ingest", HashMap::new());
        rt.complete_step("transcode", HashMap::new());
        let done = rt.completed_steps();
        assert!(done.contains("ingest"));
        assert!(done.contains("transcode"));
        assert!(!done.contains("thumbnail"));
    }
}
