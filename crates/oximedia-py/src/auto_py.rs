//! Python bindings for `oximedia-auto` automated video editing.
//!
//! Provides `PyAutomation`, `PyAutoTask`, `PyAutoSchedule` for automated
//! media processing from Python.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// PyAutoTask
// ---------------------------------------------------------------------------

/// An automated editing task.
#[pyclass]
#[derive(Clone)]
pub struct PyAutoTask {
    /// Unique task identifier.
    #[pyo3(get)]
    pub task_id: String,
    /// Task name.
    #[pyo3(get)]
    pub name: String,
    /// Use case: trailer, highlights, social, documentary, music_video.
    #[pyo3(get)]
    pub use_case: String,
    /// Task status: pending, running, completed, failed.
    #[pyo3(get)]
    pub status: String,
    /// Progress (0.0 to 1.0).
    #[pyo3(get)]
    pub progress: f64,
    /// Input path.
    #[pyo3(get)]
    pub input_path: String,
    /// Output path.
    #[pyo3(get)]
    pub output_path: String,
    /// Target duration in seconds.
    #[pyo3(get)]
    pub target_duration: Option<f64>,
    /// Pacing preset.
    #[pyo3(get)]
    pub pacing: String,
    /// Number of detected highlights.
    #[pyo3(get)]
    pub highlight_count: u32,
    /// Number of assembled clips.
    #[pyo3(get)]
    pub clip_count: u32,
}

#[pymethods]
impl PyAutoTask {
    fn __repr__(&self) -> String {
        format!(
            "PyAutoTask(id='{}', name='{}', use_case='{}', status='{}', progress={:.1}%)",
            self.task_id,
            self.name,
            self.use_case,
            self.status,
            self.progress * 100.0,
        )
    }

    /// Check if task is complete.
    fn is_complete(&self) -> bool {
        self.status == "completed"
    }

    /// Check if task failed.
    fn is_failed(&self) -> bool {
        self.status == "failed"
    }

    /// Convert to a Python dict.
    fn to_dict(&self) -> PyResult<HashMap<String, Py<PyAny>>> {
        Python::attach(|py| -> PyResult<HashMap<String, Py<PyAny>>> {
            let mut m: HashMap<String, Py<PyAny>> = HashMap::new();
            m.insert(
                "task_id".to_string(),
                self.task_id
                    .clone()
                    .into_pyobject(py)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .into(),
            );
            m.insert(
                "name".to_string(),
                self.name
                    .clone()
                    .into_pyobject(py)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .into(),
            );
            m.insert(
                "use_case".to_string(),
                self.use_case
                    .clone()
                    .into_pyobject(py)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .into(),
            );
            m.insert(
                "status".to_string(),
                self.status
                    .clone()
                    .into_pyobject(py)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .into(),
            );
            m.insert(
                "progress".to_string(),
                self.progress
                    .into_pyobject(py)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .into(),
            );
            m.insert(
                "highlight_count".to_string(),
                self.highlight_count
                    .into_pyobject(py)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .into(),
            );
            m.insert(
                "clip_count".to_string(),
                self.clip_count
                    .into_pyobject(py)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .into(),
            );
            Ok(m)
        })
    }
}

// ---------------------------------------------------------------------------
// PyAutoSchedule
// ---------------------------------------------------------------------------

/// An automated task schedule.
#[pyclass]
#[derive(Clone)]
pub struct PyAutoSchedule {
    /// Schedule identifier.
    #[pyo3(get)]
    pub schedule_id: String,
    /// Schedule name.
    #[pyo3(get)]
    pub name: String,
    /// Cron expression.
    #[pyo3(get)]
    pub cron: String,
    /// Input pattern or directory.
    #[pyo3(get)]
    pub input_pattern: String,
    /// Output directory.
    #[pyo3(get)]
    pub output_dir: String,
    /// Use case.
    #[pyo3(get)]
    pub use_case: String,
    /// Whether schedule is enabled.
    #[pyo3(get)]
    pub enabled: bool,
    /// Maximum concurrent tasks.
    #[pyo3(get)]
    pub max_concurrent: u32,
}

#[pymethods]
impl PyAutoSchedule {
    fn __repr__(&self) -> String {
        format!(
            "PyAutoSchedule(id='{}', name='{}', cron='{}', enabled={})",
            self.schedule_id, self.name, self.cron, self.enabled,
        )
    }

    /// Enable the schedule.
    fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable the schedule.
    fn disable(&mut self) {
        self.enabled = false;
    }
}

// ---------------------------------------------------------------------------
// PyAutomation
// ---------------------------------------------------------------------------

/// Automated video editing manager.
#[pyclass]
pub struct PyAutomation {
    tasks: Vec<PyAutoTask>,
    schedules: Vec<PyAutoSchedule>,
    next_task_idx: u64,
    next_schedule_idx: u64,
}

#[pymethods]
impl PyAutomation {
    /// Create a new automation manager.
    #[new]
    fn new() -> Self {
        Self {
            tasks: Vec::new(),
            schedules: Vec::new(),
            next_task_idx: 0,
            next_schedule_idx: 0,
        }
    }

    /// Create and run an automated editing task.
    ///
    /// Returns:
    ///     Task ID.
    #[pyo3(signature = (name, input_path, output_path, use_case=None, target_duration=None, pacing=None))]
    fn run_task(
        &mut self,
        name: &str,
        input_path: &str,
        output_path: &str,
        use_case: Option<&str>,
        target_duration: Option<f64>,
        pacing: Option<&str>,
    ) -> PyResult<String> {
        let uc = use_case.unwrap_or("highlights");
        validate_use_case(uc)?;
        let pac = pacing.unwrap_or("medium");
        validate_pacing(pac)?;

        // Validate the config through oximedia-auto
        let config = oximedia_auto::AutoEditorConfig::for_use_case(uc);
        let _editor = oximedia_auto::AutoEditor::new(config);

        self.next_task_idx += 1;
        let task_id = format!("auto-task-{}", self.next_task_idx);

        let task = PyAutoTask {
            task_id: task_id.clone(),
            name: name.to_string(),
            use_case: uc.to_string(),
            status: "pending".to_string(),
            progress: 0.0,
            input_path: input_path.to_string(),
            output_path: output_path.to_string(),
            target_duration,
            pacing: pac.to_string(),
            highlight_count: 0,
            clip_count: 0,
        };

        self.tasks.push(task);
        Ok(task_id)
    }

    /// Get task status.
    fn task_status(&self, task_id: &str) -> PyResult<PyAutoTask> {
        self.tasks
            .iter()
            .find(|t| t.task_id == task_id)
            .cloned()
            .ok_or_else(|| PyValueError::new_err(format!("Task '{}' not found", task_id)))
    }

    /// Cancel a task.
    fn cancel_task(&mut self, task_id: &str) -> PyResult<()> {
        let task = self
            .tasks
            .iter_mut()
            .find(|t| t.task_id == task_id)
            .ok_or_else(|| PyValueError::new_err(format!("Task '{}' not found", task_id)))?;
        if task.status == "completed" {
            return Err(PyValueError::new_err("Cannot cancel a completed task"));
        }
        task.status = "cancelled".to_string();
        Ok(())
    }

    /// Create a schedule for recurring automation.
    ///
    /// Returns:
    ///     Schedule ID.
    #[pyo3(signature = (name, input_pattern, output_dir, cron, use_case=None, max_concurrent=None))]
    fn create_schedule(
        &mut self,
        name: &str,
        input_pattern: &str,
        output_dir: &str,
        cron: &str,
        use_case: Option<&str>,
        max_concurrent: Option<u32>,
    ) -> PyResult<String> {
        let uc = use_case.unwrap_or("highlights");
        validate_use_case(uc)?;

        self.next_schedule_idx += 1;
        let schedule_id = format!("sched-{}", self.next_schedule_idx);

        let schedule = PyAutoSchedule {
            schedule_id: schedule_id.clone(),
            name: name.to_string(),
            cron: cron.to_string(),
            input_pattern: input_pattern.to_string(),
            output_dir: output_dir.to_string(),
            use_case: uc.to_string(),
            enabled: true,
            max_concurrent: max_concurrent.unwrap_or(1),
        };

        self.schedules.push(schedule);
        Ok(schedule_id)
    }

    /// Delete a schedule.
    fn delete_schedule(&mut self, schedule_id: &str) -> PyResult<()> {
        let initial_len = self.schedules.len();
        self.schedules.retain(|s| s.schedule_id != schedule_id);
        if self.schedules.len() == initial_len {
            return Err(PyValueError::new_err(format!(
                "Schedule '{}' not found",
                schedule_id
            )));
        }
        Ok(())
    }

    /// List all tasks, optionally filtered by status.
    #[pyo3(signature = (status=None))]
    fn list_tasks(&self, status: Option<&str>) -> Vec<PyAutoTask> {
        match status {
            Some(s) => self
                .tasks
                .iter()
                .filter(|t| t.status == s)
                .cloned()
                .collect(),
            None => self.tasks.clone(),
        }
    }

    /// List all schedules.
    fn list_schedules(&self) -> Vec<PyAutoSchedule> {
        self.schedules.clone()
    }

    /// Get the number of pending tasks.
    fn pending_count(&self) -> usize {
        self.tasks.iter().filter(|t| t.status == "pending").count()
    }

    /// Get the number of active schedules.
    fn active_schedule_count(&self) -> usize {
        self.schedules.iter().filter(|s| s.enabled).count()
    }

    fn __repr__(&self) -> String {
        format!(
            "PyAutomation(tasks={}, schedules={}, pending={})",
            self.tasks.len(),
            self.schedules.len(),
            self.pending_count(),
        )
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Create a new automation manager.
#[pyfunction]
pub fn create_automation() -> PyAutomation {
    PyAutomation::new()
}

/// List available use cases for auto editing.
#[pyfunction]
pub fn list_auto_use_cases() -> Vec<String> {
    vec![
        "trailer".to_string(),
        "highlights".to_string(),
        "social".to_string(),
        "documentary".to_string(),
        "music_video".to_string(),
    ]
}

/// List available pacing presets.
#[pyfunction]
pub fn list_auto_pacing_presets() -> Vec<String> {
    vec![
        "slow".to_string(),
        "medium".to_string(),
        "fast".to_string(),
        "dynamic".to_string(),
    ]
}

/// List available workflow templates.
#[pyfunction]
pub fn list_auto_templates() -> Vec<String> {
    vec![
        "highlight-reel".to_string(),
        "social-clips".to_string(),
        "trailer".to_string(),
        "batch-transcode".to_string(),
        "quality-check".to_string(),
    ]
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn validate_use_case(use_case: &str) -> PyResult<()> {
    match use_case {
        "trailer" | "highlights" | "social" | "documentary" | "music_video" => Ok(()),
        other => Err(PyValueError::new_err(format!(
            "Unknown use case '{}'. Expected: trailer, highlights, social, documentary, music_video",
            other
        ))),
    }
}

fn validate_pacing(pacing: &str) -> PyResult<()> {
    match pacing {
        "slow" | "medium" | "fast" | "dynamic" => Ok(()),
        other => Err(PyValueError::new_err(format!(
            "Unknown pacing '{}'. Expected: slow, medium, fast, dynamic",
            other
        ))),
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register all auto editing bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAutoTask>()?;
    m.add_class::<PyAutoSchedule>()?;
    m.add_class::<PyAutomation>()?;
    m.add_function(wrap_pyfunction!(create_automation, m)?)?;
    m.add_function(wrap_pyfunction!(list_auto_use_cases, m)?)?;
    m.add_function(wrap_pyfunction!(list_auto_pacing_presets, m)?)?;
    m.add_function(wrap_pyfunction!(list_auto_templates, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_and_cancel_task() {
        let mut auto = PyAutomation::new();
        let tmp = std::env::temp_dir();
        let in_path = tmp.join("oximedia-py-auto-in.mkv");
        let out_path = tmp.join("oximedia-py-auto-out.webm");
        let in_s = in_path.to_string_lossy();
        let out_s = out_path.to_string_lossy();
        let tid = auto
            .run_task("Test", &in_s, &out_s, Some("highlights"), Some(60.0), None)
            .expect("run_task should succeed");
        assert!(tid.starts_with("auto-task-"));
        assert_eq!(auto.pending_count(), 1);

        let status = auto.task_status(&tid).expect("should find task");
        assert_eq!(status.status, "pending");
        assert!(!status.is_complete());

        auto.cancel_task(&tid).expect("cancel should succeed");
        let status = auto.task_status(&tid).expect("should find task");
        assert_eq!(status.status, "cancelled");
    }

    #[test]
    fn test_invalid_use_case() {
        let mut auto = PyAutomation::new();
        let result = auto.run_task("T", "/in", "/out", Some("invalid"), None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_schedules() {
        let mut auto = PyAutomation::new();
        let sid = auto
            .create_schedule(
                "Nightly",
                "/media/*.mkv",
                "/out/",
                "0 2 * * *",
                None,
                Some(4),
            )
            .expect("create_schedule should succeed");
        assert!(sid.starts_with("sched-"));
        assert_eq!(auto.active_schedule_count(), 1);

        let schedules = auto.list_schedules();
        assert_eq!(schedules.len(), 1);
        assert_eq!(schedules[0].max_concurrent, 4);

        auto.delete_schedule(&sid).expect("delete should succeed");
        assert_eq!(auto.list_schedules().len(), 0);
    }

    #[test]
    fn test_standalone_functions() {
        let cases = list_auto_use_cases();
        assert!(cases.contains(&"highlights".to_string()));
        let pacings = list_auto_pacing_presets();
        assert!(pacings.contains(&"medium".to_string()));
        let templates = list_auto_templates();
        assert!(templates.contains(&"trailer".to_string()));
    }
}
