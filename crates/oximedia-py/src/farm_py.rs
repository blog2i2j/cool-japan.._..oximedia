//! Python bindings for `oximedia-farm` render farm management.
//!
//! Provides `PyFarmConfig`, `PyRenderNode`, `PyRenderJob`, and `PyRenderFarm`
//! for managing a render farm from Python.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// PyFarmConfig
// ---------------------------------------------------------------------------

/// Configuration for a render farm.
#[pyclass]
#[derive(Clone)]
pub struct PyFarmConfig {
    /// Farm coordinator address.
    #[pyo3(get)]
    pub address: String,
    /// Maximum number of concurrent jobs.
    #[pyo3(get)]
    pub max_jobs: u32,
    /// Data directory for persistent state.
    #[pyo3(get)]
    pub data_dir: Option<String>,
    /// Enable auto-scaling of render nodes.
    #[pyo3(get)]
    pub auto_scale: bool,
    /// Maximum retry attempts for failed tasks.
    #[pyo3(get)]
    pub max_retries: u32,
    /// Enable metrics collection.
    #[pyo3(get)]
    pub enable_metrics: bool,
}

#[pymethods]
impl PyFarmConfig {
    /// Create a new farm configuration.
    ///
    /// Args:
    ///     address: Farm coordinator address (default: "0.0.0.0:50051").
    #[new]
    #[pyo3(signature = (address=None))]
    fn new(address: Option<&str>) -> Self {
        Self {
            address: address.unwrap_or("0.0.0.0:50051").to_string(),
            max_jobs: 1000,
            data_dir: None,
            auto_scale: false,
            max_retries: 3,
            enable_metrics: true,
        }
    }

    /// Set the maximum number of concurrent jobs.
    fn with_max_jobs(&mut self, max_jobs: u32) -> PyResult<()> {
        if max_jobs == 0 {
            return Err(PyValueError::new_err("max_jobs must be greater than 0"));
        }
        self.max_jobs = max_jobs;
        Ok(())
    }

    /// Set the data directory.
    fn with_data_dir(&mut self, data_dir: &str) {
        self.data_dir = Some(data_dir.to_string());
    }

    /// Enable or disable auto-scaling.
    fn with_auto_scale(&mut self, enable: bool) {
        self.auto_scale = enable;
    }

    /// Set the maximum retry attempts.
    fn with_max_retries(&mut self, max_retries: u32) {
        self.max_retries = max_retries;
    }

    /// Enable or disable metrics.
    fn with_metrics(&mut self, enable: bool) {
        self.enable_metrics = enable;
    }

    fn __repr__(&self) -> String {
        format!(
            "PyFarmConfig(address='{}', max_jobs={}, auto_scale={}, metrics={})",
            self.address, self.max_jobs, self.auto_scale, self.enable_metrics,
        )
    }
}

// ---------------------------------------------------------------------------
// PyRenderNode
// ---------------------------------------------------------------------------

/// Information about a render node in the farm.
#[pyclass]
#[derive(Clone)]
pub struct PyRenderNode {
    /// Unique node identifier.
    #[pyo3(get)]
    pub id: String,
    /// Node hostname.
    #[pyo3(get)]
    pub hostname: String,
    /// Node status: idle, busy, overloaded, draining, offline.
    #[pyo3(get)]
    pub status: String,
    /// Number of CPU cores.
    #[pyo3(get)]
    pub cpu_cores: u32,
    /// Available memory in GB.
    #[pyo3(get)]
    pub memory_gb: f64,
    /// Whether GPU is available.
    #[pyo3(get)]
    pub gpu_available: bool,
    /// Currently assigned job ID, if any.
    #[pyo3(get)]
    pub current_job: Option<String>,
    /// Current CPU load percentage (0-100).
    #[pyo3(get)]
    pub load_percent: f64,
    /// Supported codecs.
    #[pyo3(get)]
    pub supported_codecs: Vec<String>,
}

#[pymethods]
impl PyRenderNode {
    fn __repr__(&self) -> String {
        format!(
            "PyRenderNode(id='{}', host='{}', status='{}', cpu={}, mem={:.1}GB, gpu={})",
            self.id, self.hostname, self.status, self.cpu_cores, self.memory_gb, self.gpu_available,
        )
    }

    /// Convert to a Python dict.
    fn to_dict(&self) -> HashMap<String, Py<PyAny>> {
        Python::attach(|py| {
            let mut m: HashMap<String, Py<PyAny>> = HashMap::new();
            m.insert(
                "id".to_string(),
                self.id.clone().into_pyobject(py).expect("str").into(),
            );
            m.insert(
                "hostname".to_string(),
                self.hostname.clone().into_pyobject(py).expect("str").into(),
            );
            m.insert(
                "status".to_string(),
                self.status.clone().into_pyobject(py).expect("str").into(),
            );
            m.insert(
                "cpu_cores".to_string(),
                self.cpu_cores.into_pyobject(py).expect("int").into(),
            );
            m.insert(
                "memory_gb".to_string(),
                self.memory_gb.into_pyobject(py).expect("float").into(),
            );
            m.insert(
                "gpu_available".to_string(),
                self.gpu_available
                    .into_pyobject(py)
                    .expect("bool")
                    .to_owned()
                    .into(),
            );
            m.insert(
                "load_percent".to_string(),
                self.load_percent.into_pyobject(py).expect("float").into(),
            );
            m.insert(
                "supported_codecs".to_string(),
                self.supported_codecs
                    .clone()
                    .into_pyobject(py)
                    .expect("list")
                    .into(),
            );
            m
        })
    }

    /// Check if the node is idle.
    fn is_idle(&self) -> bool {
        self.status == "idle"
    }

    /// Check if the node is busy.
    fn is_busy(&self) -> bool {
        self.status == "busy" || self.status == "overloaded"
    }

    /// Check if the node is offline.
    fn is_offline(&self) -> bool {
        self.status == "offline"
    }
}

// ---------------------------------------------------------------------------
// PyRenderJob
// ---------------------------------------------------------------------------

/// A render farm job.
#[pyclass]
#[derive(Clone)]
pub struct PyRenderJob {
    /// Unique job identifier.
    #[pyo3(get)]
    pub id: String,
    /// Job name.
    #[pyo3(get)]
    pub name: String,
    /// Job status: pending, queued, running, completed, failed, cancelled, paused.
    #[pyo3(get)]
    pub status: String,
    /// Job progress (0.0 to 1.0).
    #[pyo3(get)]
    pub progress: f64,
    /// Job priority: low, normal, high, critical.
    #[pyo3(get)]
    pub priority: String,
    /// Input file path.
    #[pyo3(get)]
    pub input_path: String,
    /// Output file path.
    #[pyo3(get)]
    pub output_path: String,
    /// Submission timestamp (ISO 8601).
    #[pyo3(get)]
    pub submitted_at: String,
    /// Start timestamp (ISO 8601), if started.
    #[pyo3(get)]
    pub started_at: Option<String>,
    /// Completion timestamp (ISO 8601), if completed.
    #[pyo3(get)]
    pub completed_at: Option<String>,
    /// Assigned render node ID, if any.
    #[pyo3(get)]
    pub node_id: Option<String>,
    /// Error message, if the job failed.
    #[pyo3(get)]
    pub error_message: Option<String>,
    /// Dependency job IDs that must complete first.
    #[pyo3(get)]
    pub dependencies: Vec<String>,
    /// Job type (e.g., "transcode", "thumbnail", "qc").
    #[pyo3(get)]
    pub job_type: String,
}

#[pymethods]
impl PyRenderJob {
    fn __repr__(&self) -> String {
        format!(
            "PyRenderJob(id='{}', name='{}', status='{}', progress={:.1}%, type='{}')",
            self.id,
            self.name,
            self.status,
            self.progress * 100.0,
            self.job_type,
        )
    }

    /// Convert to a Python dict.
    fn to_dict(&self) -> HashMap<String, Py<PyAny>> {
        Python::attach(|py| {
            let mut m: HashMap<String, Py<PyAny>> = HashMap::new();
            m.insert(
                "id".to_string(),
                self.id.clone().into_pyobject(py).expect("str").into(),
            );
            m.insert(
                "name".to_string(),
                self.name.clone().into_pyobject(py).expect("str").into(),
            );
            m.insert(
                "status".to_string(),
                self.status.clone().into_pyobject(py).expect("str").into(),
            );
            m.insert(
                "progress".to_string(),
                self.progress.into_pyobject(py).expect("float").into(),
            );
            m.insert(
                "priority".to_string(),
                self.priority.clone().into_pyobject(py).expect("str").into(),
            );
            m.insert(
                "input_path".to_string(),
                self.input_path
                    .clone()
                    .into_pyobject(py)
                    .expect("str")
                    .into(),
            );
            m.insert(
                "output_path".to_string(),
                self.output_path
                    .clone()
                    .into_pyobject(py)
                    .expect("str")
                    .into(),
            );
            m.insert(
                "submitted_at".to_string(),
                self.submitted_at
                    .clone()
                    .into_pyobject(py)
                    .expect("str")
                    .into(),
            );
            m.insert(
                "job_type".to_string(),
                self.job_type.clone().into_pyobject(py).expect("str").into(),
            );
            m.insert(
                "dependencies".to_string(),
                self.dependencies
                    .clone()
                    .into_pyobject(py)
                    .expect("list")
                    .into(),
            );
            m
        })
    }

    /// Check if the job is complete.
    fn is_complete(&self) -> bool {
        self.status == "completed"
    }

    /// Check if the job failed.
    fn is_failed(&self) -> bool {
        self.status == "failed"
    }

    /// Check if the job is still running.
    fn is_running(&self) -> bool {
        self.status == "running"
    }

    /// Check if the job is paused.
    fn is_paused(&self) -> bool {
        self.status == "paused"
    }

    /// Calculate elapsed seconds since submission (returns 0.0 if timestamps unavailable).
    fn elapsed_secs(&self) -> f64 {
        // Parse the submitted_at timestamp and compute elapsed
        // For now, return 0.0 since we use local timestamps
        0.0
    }
}

// ---------------------------------------------------------------------------
// PyRenderFarm
// ---------------------------------------------------------------------------

/// A render farm manager.
#[pyclass]
pub struct PyRenderFarm {
    config: PyFarmConfig,
    nodes: Vec<PyRenderNode>,
    jobs: Vec<PyRenderJob>,
    next_node_idx: u64,
    next_job_idx: u64,
}

#[pymethods]
impl PyRenderFarm {
    /// Create a new render farm with the given configuration.
    #[new]
    fn new(config: PyFarmConfig) -> Self {
        Self {
            config,
            nodes: Vec::new(),
            jobs: Vec::new(),
            next_node_idx: 0,
            next_job_idx: 0,
        }
    }

    /// Add a render node to the farm.
    ///
    /// Args:
    ///     hostname: Node hostname.
    ///     cpu_cores: Number of CPU cores.
    ///     memory_gb: Available memory in GB.
    ///     gpu: Whether GPU is available (default: false).
    ///     codecs: Supported codecs (default: ["av1", "vp9", "opus"]).
    ///
    /// Returns:
    ///     Node ID.
    #[pyo3(signature = (hostname, cpu_cores, memory_gb, gpu=None, codecs=None))]
    fn add_node(
        &mut self,
        hostname: &str,
        cpu_cores: u32,
        memory_gb: f64,
        gpu: Option<bool>,
        codecs: Option<Vec<String>>,
    ) -> PyResult<String> {
        if cpu_cores == 0 {
            return Err(PyValueError::new_err("cpu_cores must be greater than 0"));
        }
        if memory_gb <= 0.0 {
            return Err(PyValueError::new_err("memory_gb must be positive"));
        }

        self.next_node_idx += 1;
        let node_id = format!("node-{}", self.next_node_idx);

        let supported = codecs
            .unwrap_or_else(|| vec!["av1".to_string(), "vp9".to_string(), "opus".to_string()]);

        let node = PyRenderNode {
            id: node_id.clone(),
            hostname: hostname.to_string(),
            status: "idle".to_string(),
            cpu_cores,
            memory_gb,
            gpu_available: gpu.unwrap_or(false),
            current_job: None,
            load_percent: 0.0,
            supported_codecs: supported,
        };

        self.nodes.push(node);
        Ok(node_id)
    }

    /// Remove a render node by ID.
    fn remove_node(&mut self, node_id: &str) -> PyResult<()> {
        let initial_len = self.nodes.len();
        self.nodes.retain(|n| n.id != node_id);
        if self.nodes.len() == initial_len {
            return Err(PyValueError::new_err(format!(
                "Node '{}' not found",
                node_id
            )));
        }
        Ok(())
    }

    /// Submit a render job.
    ///
    /// Args:
    ///     name: Job name.
    ///     input_path: Input media file path.
    ///     output_path: Output file path.
    ///     preset: Encoding preset (default: "medium").
    ///     priority: Job priority (default: "normal").
    ///     dependencies: List of dependency job IDs (default: []).
    ///     job_type: Job type (default: "transcode").
    ///
    /// Returns:
    ///     Job ID.
    #[pyo3(signature = (name, input_path, output_path, preset=None, priority=None, dependencies=None, job_type=None))]
    fn submit_job(
        &mut self,
        name: &str,
        input_path: &str,
        output_path: &str,
        preset: Option<&str>,
        priority: Option<&str>,
        dependencies: Option<Vec<String>>,
        job_type: Option<&str>,
    ) -> PyResult<String> {
        let pri = priority.unwrap_or("normal");
        validate_farm_priority(pri)?;
        let jt = job_type.unwrap_or("transcode");
        validate_job_type(jt)?;
        let _preset_val = preset.unwrap_or("medium");

        self.next_job_idx += 1;
        let job_id = format!("farm-job-{}", self.next_job_idx);

        let now = chrono::Utc::now().to_rfc3339();

        let job = PyRenderJob {
            id: job_id.clone(),
            name: name.to_string(),
            status: "pending".to_string(),
            progress: 0.0,
            priority: pri.to_string(),
            input_path: input_path.to_string(),
            output_path: output_path.to_string(),
            submitted_at: now,
            started_at: None,
            completed_at: None,
            node_id: None,
            error_message: None,
            dependencies: dependencies.unwrap_or_default(),
            job_type: jt.to_string(),
        };

        self.jobs.push(job);
        Ok(job_id)
    }

    /// Cancel a job by ID.
    fn cancel_job(&mut self, job_id: &str) -> PyResult<()> {
        let job = self
            .jobs
            .iter_mut()
            .find(|j| j.id == job_id)
            .ok_or_else(|| PyValueError::new_err(format!("Job '{}' not found", job_id)))?;

        if job.status == "completed" || job.status == "cancelled" {
            return Err(PyRuntimeError::new_err(format!(
                "Cannot cancel job '{}' with status '{}'",
                job_id, job.status
            )));
        }

        job.status = "cancelled".to_string();
        Ok(())
    }

    /// Get the status of a specific job.
    fn job_status(&self, job_id: &str) -> PyResult<PyRenderJob> {
        self.jobs
            .iter()
            .find(|j| j.id == job_id)
            .cloned()
            .ok_or_else(|| PyValueError::new_err(format!("Job '{}' not found", job_id)))
    }

    /// List all render nodes.
    fn list_nodes(&self) -> Vec<PyRenderNode> {
        self.nodes.clone()
    }

    /// List jobs, optionally filtered by status.
    ///
    /// Args:
    ///     status_filter: Filter by status (e.g., "pending", "running"). None for all jobs.
    #[pyo3(signature = (status_filter=None))]
    fn list_jobs(&self, status_filter: Option<&str>) -> Vec<PyRenderJob> {
        match status_filter {
            Some(filter) => self
                .jobs
                .iter()
                .filter(|j| j.status == filter)
                .cloned()
                .collect(),
            None => self.jobs.clone(),
        }
    }

    /// Get the number of render nodes.
    fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of active (running) jobs.
    fn active_job_count(&self) -> usize {
        self.jobs.iter().filter(|j| j.status == "running").count()
    }

    /// Get the number of jobs waiting in the queue.
    fn queue_length(&self) -> usize {
        self.jobs
            .iter()
            .filter(|j| j.status == "pending" || j.status == "queued")
            .count()
    }

    /// Get the number of idle nodes.
    fn idle_node_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.status == "idle").count()
    }

    /// Get the farm configuration.
    fn get_config(&self) -> PyFarmConfig {
        self.config.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "PyRenderFarm(address='{}', nodes={}, jobs={}, queue={}, active={})",
            self.config.address,
            self.nodes.len(),
            self.jobs.len(),
            self.queue_length(),
            self.active_job_count(),
        )
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Create a new render farm with the given address.
///
/// Args:
///     address: Farm coordinator address (default: "0.0.0.0:50051").
///
/// Returns:
///     A new PyRenderFarm instance.
#[pyfunction]
#[pyo3(signature = (address=None))]
pub fn create_render_farm(address: Option<&str>) -> PyRenderFarm {
    let config = PyFarmConfig::new(address);
    PyRenderFarm::new(config)
}

/// List available farm job priorities.
#[pyfunction]
pub fn list_farm_priorities() -> Vec<String> {
    vec![
        "low".to_string(),
        "normal".to_string(),
        "high".to_string(),
        "critical".to_string(),
    ]
}

/// List available farm job types.
#[pyfunction]
pub fn list_farm_job_types() -> Vec<String> {
    vec![
        "transcode".to_string(),
        "audio".to_string(),
        "thumbnail".to_string(),
        "qc".to_string(),
        "analysis".to_string(),
        "fingerprint".to_string(),
        "multi".to_string(),
    ]
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn validate_farm_priority(priority: &str) -> PyResult<()> {
    match priority {
        "low" | "normal" | "high" | "critical" => Ok(()),
        other => Err(PyValueError::new_err(format!(
            "Unknown priority '{}'. Expected: low, normal, high, critical",
            other
        ))),
    }
}

fn validate_job_type(job_type: &str) -> PyResult<()> {
    match job_type {
        "transcode" | "audio" | "thumbnail" | "qc" | "analysis" | "fingerprint" | "multi" => {
            Ok(())
        }
        other => Err(PyValueError::new_err(format!(
            "Unknown job type '{}'. Expected: transcode, audio, thumbnail, qc, analysis, fingerprint, multi",
            other
        ))),
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register all farm bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFarmConfig>()?;
    m.add_class::<PyRenderNode>()?;
    m.add_class::<PyRenderJob>()?;
    m.add_class::<PyRenderFarm>()?;
    m.add_function(wrap_pyfunction!(create_render_farm, m)?)?;
    m.add_function(wrap_pyfunction!(list_farm_priorities, m)?)?;
    m.add_function(wrap_pyfunction!(list_farm_job_types, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_farm_config_default() {
        let config = PyFarmConfig::new(None);
        assert_eq!(config.address, "0.0.0.0:50051");
        assert_eq!(config.max_jobs, 1000);
        assert!(!config.auto_scale);
        assert!(config.enable_metrics);
    }

    #[test]
    fn test_farm_config_custom() {
        let mut config = PyFarmConfig::new(Some("10.0.0.1:9100"));
        assert_eq!(config.address, "10.0.0.1:9100");
        assert!(config.with_max_jobs(50).is_ok());
        assert_eq!(config.max_jobs, 50);
        assert!(config.with_max_jobs(0).is_err());
    }

    #[test]
    fn test_render_farm_add_remove_node() {
        let config = PyFarmConfig::new(None);
        let mut farm = PyRenderFarm::new(config);

        let id = farm
            .add_node("render-01", 16, 64.0, Some(true), None)
            .expect("add_node should succeed");
        assert_eq!(farm.node_count(), 1);

        let nodes = farm.list_nodes();
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].hostname, "render-01");
        assert_eq!(nodes[0].cpu_cores, 16);
        assert!(nodes[0].gpu_available);
        assert!(nodes[0].is_idle());

        farm.remove_node(&id).expect("remove_node should succeed");
        assert_eq!(farm.node_count(), 0);
    }

    #[test]
    fn test_render_farm_submit_and_cancel_job() {
        let config = PyFarmConfig::new(None);
        let mut farm = PyRenderFarm::new(config);

        let job_id = farm
            .submit_job(
                "Test Render",
                "/tmp/in.mkv",
                "/tmp/out.webm",
                None,
                Some("high"),
                None,
                None,
            )
            .expect("submit_job should succeed");
        assert_eq!(farm.queue_length(), 1);

        let status = farm.job_status(&job_id).expect("job_status should succeed");
        assert_eq!(status.status, "pending");
        assert_eq!(status.priority, "high");
        assert!(!status.is_complete());
        assert!(!status.is_failed());

        farm.cancel_job(&job_id).expect("cancel_job should succeed");
        let status = farm.job_status(&job_id).expect("job_status should succeed");
        assert_eq!(status.status, "cancelled");
        assert_eq!(farm.queue_length(), 0);
    }

    #[test]
    fn test_validate_farm_priority() {
        assert!(validate_farm_priority("low").is_ok());
        assert!(validate_farm_priority("normal").is_ok());
        assert!(validate_farm_priority("high").is_ok());
        assert!(validate_farm_priority("critical").is_ok());
        assert!(validate_farm_priority("invalid").is_err());
    }

    #[test]
    fn test_validate_job_type() {
        assert!(validate_job_type("transcode").is_ok());
        assert!(validate_job_type("audio").is_ok());
        assert!(validate_job_type("thumbnail").is_ok());
        assert!(validate_job_type("qc").is_ok());
        assert!(validate_job_type("analysis").is_ok());
        assert!(validate_job_type("invalid").is_err());
    }
}
