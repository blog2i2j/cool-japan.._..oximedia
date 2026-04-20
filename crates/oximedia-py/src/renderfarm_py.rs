//! Python bindings for `oximedia-renderfarm` cluster management.
//!
//! Provides `PyRenderFarmCluster`, `PyFarmNode`, `PyFarmJob` for managing
//! render farm clusters from Python.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// PyFarmNode
// ---------------------------------------------------------------------------

/// Information about a render farm cluster node.
#[pyclass]
#[derive(Clone)]
pub struct PyFarmNode {
    /// Unique node identifier.
    #[pyo3(get)]
    pub id: String,
    /// Node hostname.
    #[pyo3(get)]
    pub hostname: String,
    /// Node port.
    #[pyo3(get)]
    pub port: u16,
    /// Node status: idle, busy, draining, offline.
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
    /// Worker pool assignment.
    #[pyo3(get)]
    pub pool: Option<String>,
    /// Node tags.
    #[pyo3(get)]
    pub tags: Vec<String>,
    /// Current CPU load percentage (0-100).
    #[pyo3(get)]
    pub load_percent: f64,
}

#[pymethods]
impl PyFarmNode {
    fn __repr__(&self) -> String {
        format!(
            "PyFarmNode(id='{}', host='{}:{}', status='{}', cpu={}, mem={:.1}GB, gpu={})",
            self.id,
            self.hostname,
            self.port,
            self.status,
            self.cpu_cores,
            self.memory_gb,
            self.gpu_available,
        )
    }

    /// Check if the node is idle.
    fn is_idle(&self) -> bool {
        self.status == "idle"
    }

    /// Check if the node is busy.
    fn is_busy(&self) -> bool {
        self.status == "busy"
    }

    /// Check if the node is draining.
    fn is_draining(&self) -> bool {
        self.status == "draining"
    }

    /// Check if the node is offline.
    fn is_offline(&self) -> bool {
        self.status == "offline"
    }

    /// Convert to a Python dict.
    fn to_dict(&self) -> PyResult<HashMap<String, Py<PyAny>>> {
        Python::attach(|py| -> PyResult<HashMap<String, Py<PyAny>>> {
            let mut m: HashMap<String, Py<PyAny>> = HashMap::new();
            m.insert(
                "id".to_string(),
                self.id
                    .clone()
                    .into_pyobject(py)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .into(),
            );
            m.insert(
                "hostname".to_string(),
                self.hostname
                    .clone()
                    .into_pyobject(py)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .into(),
            );
            m.insert(
                "port".to_string(),
                self.port
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
                "cpu_cores".to_string(),
                self.cpu_cores
                    .into_pyobject(py)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .into(),
            );
            m.insert(
                "memory_gb".to_string(),
                self.memory_gb
                    .into_pyobject(py)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .into(),
            );
            m.insert(
                "gpu_available".to_string(),
                self.gpu_available
                    .into_pyobject(py)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .to_owned()
                    .into(),
            );
            m.insert(
                "load_percent".to_string(),
                self.load_percent
                    .into_pyobject(py)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .into(),
            );
            Ok(m)
        })
    }
}

// ---------------------------------------------------------------------------
// PyFarmJob
// ---------------------------------------------------------------------------

/// A render farm cluster job.
#[pyclass]
#[derive(Clone)]
pub struct PyFarmJob {
    /// Unique job identifier.
    #[pyo3(get)]
    pub id: String,
    /// Job name.
    #[pyo3(get)]
    pub name: String,
    /// Job status.
    #[pyo3(get)]
    pub status: String,
    /// Progress (0.0 to 1.0).
    #[pyo3(get)]
    pub progress: f64,
    /// Priority: low, normal, high, critical.
    #[pyo3(get)]
    pub priority: String,
    /// Project file path.
    #[pyo3(get)]
    pub project_path: String,
    /// Output directory.
    #[pyo3(get)]
    pub output_dir: String,
    /// Frame range string.
    #[pyo3(get)]
    pub frame_range: Option<String>,
    /// Target pool.
    #[pyo3(get)]
    pub pool: Option<String>,
    /// Submission timestamp (ISO 8601).
    #[pyo3(get)]
    pub submitted_at: String,
    /// Maximum retries per task.
    #[pyo3(get)]
    pub max_retries: u32,
}

#[pymethods]
impl PyFarmJob {
    fn __repr__(&self) -> String {
        format!(
            "PyFarmJob(id='{}', name='{}', status='{}', progress={:.1}%)",
            self.id,
            self.name,
            self.status,
            self.progress * 100.0,
        )
    }

    /// Check if the job is complete.
    fn is_complete(&self) -> bool {
        self.status == "completed"
    }

    /// Check if the job failed.
    fn is_failed(&self) -> bool {
        self.status == "failed"
    }

    /// Check if the job is running.
    fn is_running(&self) -> bool {
        self.status == "running"
    }

    /// Convert to a Python dict.
    fn to_dict(&self) -> PyResult<HashMap<String, Py<PyAny>>> {
        Python::attach(|py| -> PyResult<HashMap<String, Py<PyAny>>> {
            let mut m: HashMap<String, Py<PyAny>> = HashMap::new();
            m.insert(
                "id".to_string(),
                self.id
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
                "priority".to_string(),
                self.priority
                    .clone()
                    .into_pyobject(py)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .into(),
            );
            m.insert(
                "project_path".to_string(),
                self.project_path
                    .clone()
                    .into_pyobject(py)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .into(),
            );
            m.insert(
                "output_dir".to_string(),
                self.output_dir
                    .clone()
                    .into_pyobject(py)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .into(),
            );
            Ok(m)
        })
    }
}

// ---------------------------------------------------------------------------
// PyRenderFarmCluster
// ---------------------------------------------------------------------------

/// A render farm cluster manager.
#[pyclass]
pub struct PyRenderFarmCluster {
    name: String,
    nodes: Vec<PyFarmNode>,
    jobs: Vec<PyFarmJob>,
    next_node_idx: u64,
    next_job_idx: u64,
    scheduler: String,
}

#[pymethods]
impl PyRenderFarmCluster {
    /// Create a new render farm cluster.
    ///
    /// Args:
    ///     name: Cluster name.
    ///     scheduler: Scheduling algorithm (default: "least-loaded").
    #[new]
    #[pyo3(signature = (name, scheduler=None))]
    fn new(name: &str, scheduler: Option<&str>) -> Self {
        Self {
            name: name.to_string(),
            nodes: Vec::new(),
            jobs: Vec::new(),
            next_node_idx: 0,
            next_job_idx: 0,
            scheduler: scheduler.unwrap_or("least-loaded").to_string(),
        }
    }

    /// Add a node to the cluster.
    ///
    /// Returns:
    ///     Node ID.
    #[pyo3(signature = (hostname, port=None, cpu_cores=None, memory_gb=None, gpu=None, pool=None, tags=None))]
    fn add_node(
        &mut self,
        hostname: &str,
        port: Option<u16>,
        cpu_cores: Option<u32>,
        memory_gb: Option<f64>,
        gpu: Option<bool>,
        pool: Option<&str>,
        tags: Option<Vec<String>>,
    ) -> PyResult<String> {
        let cores = cpu_cores.unwrap_or(4);
        if cores == 0 {
            return Err(PyValueError::new_err("cpu_cores must be greater than 0"));
        }
        let mem = memory_gb.unwrap_or(16.0);
        if mem <= 0.0 {
            return Err(PyValueError::new_err("memory_gb must be positive"));
        }

        self.next_node_idx += 1;
        let node_id = format!("cluster-node-{}", self.next_node_idx);

        let node = PyFarmNode {
            id: node_id.clone(),
            hostname: hostname.to_string(),
            port: port.unwrap_or(9201),
            status: "idle".to_string(),
            cpu_cores: cores,
            memory_gb: mem,
            gpu_available: gpu.unwrap_or(false),
            pool: pool.map(|s| s.to_string()),
            tags: tags.unwrap_or_default(),
            load_percent: 0.0,
        };

        self.nodes.push(node);
        Ok(node_id)
    }

    /// Remove a node from the cluster.
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

    /// Submit a render job to the cluster.
    ///
    /// Returns:
    ///     Job ID.
    #[pyo3(signature = (name, project_path, output_dir, frames=None, priority=None, pool=None, max_retries=None))]
    fn submit_job(
        &mut self,
        name: &str,
        project_path: &str,
        output_dir: &str,
        frames: Option<&str>,
        priority: Option<&str>,
        pool: Option<&str>,
        max_retries: Option<u32>,
    ) -> PyResult<String> {
        let pri = priority.unwrap_or("normal");
        validate_cluster_priority(pri)?;

        self.next_job_idx += 1;
        let job_id = format!("cluster-job-{}", self.next_job_idx);
        let now = chrono::Utc::now().to_rfc3339();

        let job = PyFarmJob {
            id: job_id.clone(),
            name: name.to_string(),
            status: "pending".to_string(),
            progress: 0.0,
            priority: pri.to_string(),
            project_path: project_path.to_string(),
            output_dir: output_dir.to_string(),
            frame_range: frames.map(|f| f.to_string()),
            pool: pool.map(|p| p.to_string()),
            submitted_at: now,
            max_retries: max_retries.unwrap_or(3),
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
    fn job_status(&self, job_id: &str) -> PyResult<PyFarmJob> {
        self.jobs
            .iter()
            .find(|j| j.id == job_id)
            .cloned()
            .ok_or_else(|| PyValueError::new_err(format!("Job '{}' not found", job_id)))
    }

    /// List all nodes.
    fn list_nodes(&self) -> Vec<PyFarmNode> {
        self.nodes.clone()
    }

    /// List jobs, optionally filtered by status.
    #[pyo3(signature = (status_filter=None))]
    fn list_jobs(&self, status_filter: Option<&str>) -> Vec<PyFarmJob> {
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

    /// Get the cluster name.
    fn cluster_name(&self) -> &str {
        &self.name
    }

    /// Get the scheduler type.
    fn scheduler_type(&self) -> &str {
        &self.scheduler
    }

    /// Get the number of nodes.
    fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of idle nodes.
    fn idle_node_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.status == "idle").count()
    }

    /// Get the number of active jobs.
    fn active_job_count(&self) -> usize {
        self.jobs.iter().filter(|j| j.status == "running").count()
    }

    /// Get the queue length.
    fn queue_length(&self) -> usize {
        self.jobs.iter().filter(|j| j.status == "pending").count()
    }

    fn __repr__(&self) -> String {
        format!(
            "PyRenderFarmCluster(name='{}', scheduler='{}', nodes={}, jobs={}, queue={})",
            self.name,
            self.scheduler,
            self.nodes.len(),
            self.jobs.len(),
            self.queue_length(),
        )
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Create a new render farm cluster.
#[pyfunction]
#[pyo3(signature = (name, scheduler=None))]
pub fn init_farm(name: &str, scheduler: Option<&str>) -> PyRenderFarmCluster {
    PyRenderFarmCluster::new(name, scheduler)
}

/// List available scheduling algorithms.
#[pyfunction]
pub fn list_farm_schedulers() -> Vec<String> {
    vec![
        "round-robin".to_string(),
        "least-loaded".to_string(),
        "priority".to_string(),
        "affinity".to_string(),
    ]
}

/// List available node types/roles.
#[pyfunction]
pub fn list_farm_node_types() -> Vec<String> {
    vec![
        "cpu-general".to_string(),
        "cpu-high-mem".to_string(),
        "gpu-render".to_string(),
        "gpu-encode".to_string(),
        "storage".to_string(),
    ]
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn validate_cluster_priority(priority: &str) -> PyResult<()> {
    match priority {
        "low" | "normal" | "high" | "critical" => Ok(()),
        other => Err(PyValueError::new_err(format!(
            "Unknown priority '{}'. Expected: low, normal, high, critical",
            other
        ))),
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register all render farm cluster bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFarmNode>()?;
    m.add_class::<PyFarmJob>()?;
    m.add_class::<PyRenderFarmCluster>()?;
    m.add_function(wrap_pyfunction!(init_farm, m)?)?;
    m.add_function(wrap_pyfunction!(list_farm_schedulers, m)?)?;
    m.add_function(wrap_pyfunction!(list_farm_node_types, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cluster_creation() {
        let cluster = PyRenderFarmCluster::new("test-cluster", Some("priority"));
        assert_eq!(cluster.cluster_name(), "test-cluster");
        assert_eq!(cluster.scheduler_type(), "priority");
        assert_eq!(cluster.node_count(), 0);
    }

    #[test]
    fn test_add_remove_node() {
        let mut cluster = PyRenderFarmCluster::new("test", None);
        let id = cluster
            .add_node(
                "render-01",
                None,
                Some(16),
                Some(64.0),
                Some(true),
                None,
                None,
            )
            .expect("add_node should succeed");
        assert_eq!(cluster.node_count(), 1);

        let nodes = cluster.list_nodes();
        assert_eq!(nodes[0].hostname, "render-01");
        assert!(nodes[0].gpu_available);
        assert!(nodes[0].is_idle());

        cluster
            .remove_node(&id)
            .expect("remove_node should succeed");
        assert_eq!(cluster.node_count(), 0);
    }

    #[test]
    fn test_submit_and_cancel_job() {
        let mut cluster = PyRenderFarmCluster::new("test", None);
        let tmp = std::env::temp_dir();
        let proj = tmp.join("oximedia-py-renderfarm-project.blend");
        let out = tmp.join("oximedia-py-renderfarm-out");
        let proj_s = proj.to_string_lossy();
        let out_s = out.to_string_lossy();
        let job_id = cluster
            .submit_job(
                "Test Render",
                &proj_s,
                &out_s,
                Some("1-100"),
                Some("high"),
                None,
                None,
            )
            .expect("submit_job should succeed");

        assert_eq!(cluster.queue_length(), 1);
        let status = cluster.job_status(&job_id).expect("should find job");
        assert_eq!(status.priority, "high");
        assert!(!status.is_complete());

        cluster.cancel_job(&job_id).expect("cancel should succeed");
        let status = cluster.job_status(&job_id).expect("should find job");
        assert_eq!(status.status, "cancelled");
    }

    #[test]
    fn test_validate_priority() {
        assert!(validate_cluster_priority("low").is_ok());
        assert!(validate_cluster_priority("normal").is_ok());
        assert!(validate_cluster_priority("high").is_ok());
        assert!(validate_cluster_priority("critical").is_ok());
        assert!(validate_cluster_priority("invalid").is_err());
    }

    #[test]
    fn test_standalone_functions() {
        let schedulers = list_farm_schedulers();
        assert!(schedulers.contains(&"least-loaded".to_string()));

        let node_types = list_farm_node_types();
        assert!(node_types.contains(&"gpu-render".to_string()));
    }
}
