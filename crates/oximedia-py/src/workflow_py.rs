//! Python bindings for `oximedia-workflow` orchestration engine.
//!
//! Provides `PyWorkflow`, `PyWorkflowStep`, `PyWorkflowStatus`,
//! `PyWorkflowTemplate`, and standalone functions for workflow management.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// PyWorkflowStep
// ---------------------------------------------------------------------------

/// A single step (task) in a workflow.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyWorkflowStep {
    /// Step identifier.
    #[pyo3(get)]
    pub step_id: String,

    /// Task type: transcode, qc, transfer, analysis, wait, notification.
    #[pyo3(get)]
    pub task_type: String,

    /// Human-readable description.
    #[pyo3(get, set)]
    pub description: String,

    /// IDs of steps this step depends on.
    #[pyo3(get)]
    pub depends_on: Vec<String>,

    /// Task parameters as a JSON-compatible dict.
    #[pyo3(get)]
    pub params: HashMap<String, String>,
}

#[pymethods]
impl PyWorkflowStep {
    /// Create a new workflow step.
    #[new]
    #[pyo3(signature = (step_id, task_type, description=None, depends_on=None))]
    fn new(
        step_id: &str,
        task_type: &str,
        description: Option<&str>,
        depends_on: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let valid_types = [
            "transcode",
            "qc",
            "transfer",
            "analysis",
            "wait",
            "notification",
        ];
        if !valid_types.contains(&task_type) {
            return Err(PyValueError::new_err(format!(
                "Unknown task type '{}'. Valid: {}",
                task_type,
                valid_types.join(", ")
            )));
        }
        Ok(Self {
            step_id: step_id.to_string(),
            task_type: task_type.to_string(),
            description: description.unwrap_or("").to_string(),
            depends_on: depends_on.unwrap_or_default(),
            params: HashMap::new(),
        })
    }

    /// Set a parameter on this step.
    fn set_param(&mut self, key: &str, value: &str) {
        self.params.insert(key.to_string(), value.to_string());
    }

    /// Get a parameter value.
    fn get_param(&self, key: &str) -> Option<String> {
        self.params.get(key).cloned()
    }

    fn __repr__(&self) -> String {
        let deps = if self.depends_on.is_empty() {
            "none".to_string()
        } else {
            self.depends_on.join(", ")
        };
        format!(
            "PyWorkflowStep(id='{}', type='{}', deps=[{}])",
            self.step_id, self.task_type, deps,
        )
    }
}

// ---------------------------------------------------------------------------
// PyWorkflowStatus
// ---------------------------------------------------------------------------

/// Status of a workflow execution.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyWorkflowStatus {
    /// Workflow state: idle, running, completed, failed, cancelled.
    #[pyo3(get)]
    pub state: String,

    /// Progress as a fraction (0.0 - 1.0).
    #[pyo3(get)]
    pub progress: f64,

    /// Number of completed tasks.
    #[pyo3(get)]
    pub tasks_completed: usize,

    /// Total number of tasks.
    #[pyo3(get)]
    pub tasks_total: usize,

    /// Error message (if failed).
    #[pyo3(get)]
    pub error: Option<String>,
}

#[pymethods]
impl PyWorkflowStatus {
    fn __repr__(&self) -> String {
        format!(
            "PyWorkflowStatus(state='{}', progress={:.1}%, tasks={}/{})",
            self.state,
            self.progress * 100.0,
            self.tasks_completed,
            self.tasks_total,
        )
    }
}

// ---------------------------------------------------------------------------
// PyWorkflowTemplate
// ---------------------------------------------------------------------------

/// A reusable workflow template.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyWorkflowTemplate {
    /// Template name.
    #[pyo3(get)]
    pub name: String,

    /// Template description.
    #[pyo3(get)]
    pub description: String,

    /// Steps in the template.
    steps: Vec<PyWorkflowStep>,
}

#[pymethods]
impl PyWorkflowTemplate {
    /// Create a new template.
    #[new]
    fn new(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            steps: Vec::new(),
        }
    }

    /// Add a step to the template.
    fn add_step(&mut self, step: PyWorkflowStep) {
        self.steps.push(step);
    }

    /// Get all steps.
    fn steps(&self) -> Vec<PyWorkflowStep> {
        self.steps.clone()
    }

    /// Get step count.
    fn step_count(&self) -> usize {
        self.steps.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "PyWorkflowTemplate(name='{}', steps={})",
            self.name,
            self.steps.len(),
        )
    }
}

// ---------------------------------------------------------------------------
// PyWorkflow
// ---------------------------------------------------------------------------

/// A media processing workflow.
#[pyclass]
pub struct PyWorkflow {
    name: String,
    steps: Vec<PyWorkflowStep>,
    state: String,
}

#[pymethods]
impl PyWorkflow {
    /// Create a new workflow.
    #[new]
    #[pyo3(signature = (name=None))]
    fn new(name: Option<&str>) -> Self {
        Self {
            name: name.unwrap_or("Untitled Workflow").to_string(),
            steps: Vec::new(),
            state: "idle".to_string(),
        }
    }

    /// Get the workflow name.
    fn name(&self) -> String {
        self.name.clone()
    }

    /// Add a step to the workflow.
    fn add_step(&mut self, step: PyWorkflowStep) -> PyResult<()> {
        // Validate that dependencies exist
        let existing_ids: Vec<&str> = self.steps.iter().map(|s| s.step_id.as_str()).collect();
        for dep in &step.depends_on {
            if !existing_ids.contains(&dep.as_str()) {
                return Err(PyValueError::new_err(format!(
                    "Step '{}' depends on unknown step '{}'",
                    step.step_id, dep
                )));
            }
        }
        self.steps.push(step);
        Ok(())
    }

    /// Get all steps.
    fn steps(&self) -> Vec<PyWorkflowStep> {
        self.steps.clone()
    }

    /// Get step count.
    fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Get current workflow state.
    fn state(&self) -> String {
        self.state.clone()
    }

    /// Get the workflow status.
    fn status(&self) -> PyWorkflowStatus {
        PyWorkflowStatus {
            state: self.state.clone(),
            progress: 0.0,
            tasks_completed: 0,
            tasks_total: self.steps.len(),
            error: None,
        }
    }

    /// Validate the workflow DAG.
    fn validate(&self) -> PyResult<Vec<String>> {
        let mut issues = Vec::new();

        if self.steps.is_empty() {
            issues.push("Workflow has no steps".to_string());
            return Ok(issues);
        }

        let step_ids: Vec<&str> = self.steps.iter().map(|s| s.step_id.as_str()).collect();

        // Check for duplicate IDs
        let mut seen = std::collections::HashSet::new();
        for id in &step_ids {
            if !seen.insert(id) {
                issues.push(format!("Duplicate step ID: '{}'", id));
            }
        }

        // Check for missing dependencies
        for step in &self.steps {
            for dep in &step.depends_on {
                if !step_ids.contains(&dep.as_str()) {
                    issues.push(format!(
                        "Step '{}' depends on unknown step '{}'",
                        step.step_id, dep
                    ));
                }
            }
        }

        Ok(issues)
    }

    /// Serialize to JSON string.
    fn to_json(&self) -> PyResult<String> {
        let data = serde_json::json!({
            "name": self.name,
            "state": self.state,
            "steps": self.steps.iter().map(|s| {
                serde_json::json!({
                    "step_id": s.step_id,
                    "task_type": s.task_type,
                    "description": s.description,
                    "depends_on": s.depends_on,
                    "params": s.params,
                })
            }).collect::<Vec<_>>(),
        });
        serde_json::to_string_pretty(&data)
            .map_err(|e| PyRuntimeError::new_err(format!("JSON error: {e}")))
    }

    fn __repr__(&self) -> String {
        format!(
            "PyWorkflow(name='{}', steps={}, state='{}')",
            self.name,
            self.steps.len(),
            self.state,
        )
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Create a workflow from a list of steps.
#[pyfunction]
#[pyo3(signature = (steps, name=None))]
pub fn create_workflow(steps: Vec<PyWorkflowStep>, name: Option<&str>) -> PyResult<PyWorkflow> {
    let mut wf = PyWorkflow::new(name);
    for step in steps {
        wf.add_step(step)?;
    }
    Ok(wf)
}

/// List available built-in workflow templates.
#[pyfunction]
pub fn list_templates() -> Vec<PyWorkflowTemplate> {
    let mut templates = Vec::new();

    // Transcode template
    let mut t = PyWorkflowTemplate::new("transcode", "Validate -> Transcode -> Verify");
    if let Ok(s1) = PyWorkflowStep::new("validate", "qc", Some("Validate source"), None) {
        t.add_step(s1);
    }
    if let Ok(s2) = PyWorkflowStep::new(
        "transcode",
        "transcode",
        Some("Transcode"),
        Some(vec!["validate".to_string()]),
    ) {
        t.add_step(s2);
    }
    if let Ok(s3) = PyWorkflowStep::new(
        "verify",
        "qc",
        Some("Verify output"),
        Some(vec!["transcode".to_string()]),
    ) {
        t.add_step(s3);
    }
    templates.push(t);

    // Ingest template
    let mut t2 = PyWorkflowTemplate::new("ingest", "Copy -> Probe -> Generate proxy");
    if let Ok(s1) = PyWorkflowStep::new("copy", "transfer", Some("Copy to storage"), None) {
        t2.add_step(s1);
    }
    if let Ok(s2) = PyWorkflowStep::new(
        "probe",
        "analysis",
        Some("Probe format"),
        Some(vec!["copy".to_string()]),
    ) {
        t2.add_step(s2);
    }
    if let Ok(s3) = PyWorkflowStep::new(
        "proxy",
        "transcode",
        Some("Generate proxy"),
        Some(vec!["probe".to_string()]),
    ) {
        t2.add_step(s3);
    }
    templates.push(t2);

    // QC template
    let mut t3 = PyWorkflowTemplate::new("qc", "Format, quality, and loudness checks");
    if let Ok(s1) = PyWorkflowStep::new("format_check", "qc", Some("Format check"), None) {
        t3.add_step(s1);
    }
    if let Ok(s2) = PyWorkflowStep::new("quality_check", "qc", Some("Quality check"), None) {
        t3.add_step(s2);
    }
    if let Ok(s3) = PyWorkflowStep::new("loudness_check", "qc", Some("Loudness check"), None) {
        t3.add_step(s3);
    }
    templates.push(t3);

    templates
}

/// Run a workflow (synchronous, returns status).
#[pyfunction]
pub fn run_workflow(workflow: &PyWorkflow) -> PyResult<PyWorkflowStatus> {
    let issues = workflow.validate()?;
    if !issues.is_empty() {
        return Err(PyValueError::new_err(format!(
            "Workflow validation failed: {}",
            issues.join("; ")
        )));
    }

    // Placeholder: actual execution requires async runtime
    Ok(PyWorkflowStatus {
        state: "completed".to_string(),
        progress: 1.0,
        tasks_completed: workflow.step_count(),
        tasks_total: workflow.step_count(),
        error: None,
    })
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register workflow bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyWorkflow>()?;
    m.add_class::<PyWorkflowStep>()?;
    m.add_class::<PyWorkflowStatus>()?;
    m.add_class::<PyWorkflowTemplate>()?;
    m.add_function(wrap_pyfunction!(create_workflow, m)?)?;
    m.add_function(wrap_pyfunction!(list_templates, m)?)?;
    m.add_function(wrap_pyfunction!(run_workflow, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workflow_step_new() {
        let step = PyWorkflowStep::new("s1", "transcode", Some("Transcode"), None);
        assert!(step.is_ok());
        let step = step.expect("valid");
        assert_eq!(step.step_id, "s1");
        assert_eq!(step.task_type, "transcode");
    }

    #[test]
    fn test_workflow_step_invalid_type() {
        let step = PyWorkflowStep::new("s1", "unknown", None, None);
        assert!(step.is_err());
    }

    #[test]
    fn test_workflow_add_steps_and_validate() {
        let mut wf = PyWorkflow::new(Some("Test"));
        let s1 = PyWorkflowStep::new("s1", "qc", None, None).expect("valid");
        let s2 = PyWorkflowStep::new("s2", "transcode", None, Some(vec!["s1".to_string()]))
            .expect("valid");
        wf.add_step(s1).expect("valid");
        wf.add_step(s2).expect("valid");
        assert_eq!(wf.step_count(), 2);

        let issues = wf.validate().expect("validate should succeed");
        assert!(issues.is_empty());
    }

    #[test]
    fn test_workflow_validate_empty() {
        let wf = PyWorkflow::new(None);
        let issues = wf.validate().expect("validate should succeed");
        assert!(!issues.is_empty());
    }

    #[test]
    fn test_list_templates_fn() {
        let templates = list_templates();
        assert!(templates.len() >= 3);
        assert!(templates.iter().any(|t| t.name == "transcode"));
        assert!(templates.iter().any(|t| t.name == "ingest"));
    }
}
