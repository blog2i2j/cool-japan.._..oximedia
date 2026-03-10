//! Python bindings for `oximedia-distributed` cluster management.
//!
//! Provides `PyClusterConfig`, `PyWorkerInfo`, `PyDistributedJob`, and `PyCluster`
//! for managing distributed encoding clusters from Python.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// PyClusterConfig
// ---------------------------------------------------------------------------

/// Configuration for a distributed encoding cluster.
#[pyclass]
#[derive(Clone)]
pub struct PyClusterConfig {
    /// Coordinator address (host:port).
    #[pyo3(get)]
    pub coordinator_address: String,
    /// Maximum number of workers allowed.
    #[pyo3(get)]
    pub max_workers: u32,
    /// Heartbeat interval in seconds.
    #[pyo3(get)]
    pub heartbeat_interval_secs: u32,
    /// Job timeout in seconds.
    #[pyo3(get)]
    pub job_timeout_secs: u32,
    /// Maximum concurrent jobs per worker.
    #[pyo3(get)]
    pub max_concurrent_per_worker: u32,
    /// Enable fault tolerance.
    #[pyo3(get)]
    pub fault_tolerance: bool,
}

#[pymethods]
impl PyClusterConfig {
    /// Create a new cluster configuration.
    ///
    /// Args:
    ///     coordinator_address: Address of the coordinator (default: "127.0.0.1:50051").
    #[new]
    #[pyo3(signature = (coordinator_address=None))]
    fn new(coordinator_address: Option<&str>) -> Self {
        Self {
            coordinator_address: coordinator_address.unwrap_or("127.0.0.1:50051").to_string(),
            max_workers: 64,
            heartbeat_interval_secs: 30,
            job_timeout_secs: 3600,
            max_concurrent_per_worker: 4,
            fault_tolerance: true,
        }
    }

    /// Set the maximum number of workers.
    fn with_max_workers(&mut self, max_workers: u32) -> PyResult<()> {
        if max_workers == 0 {
            return Err(PyValueError::new_err("max_workers must be greater than 0"));
        }
        self.max_workers = max_workers;
        Ok(())
    }

    /// Set the heartbeat interval in seconds.
    fn with_heartbeat(&mut self, interval_secs: u32) -> PyResult<()> {
        if interval_secs == 0 {
            return Err(PyValueError::new_err(
                "heartbeat_interval_secs must be greater than 0",
            ));
        }
        self.heartbeat_interval_secs = interval_secs;
        Ok(())
    }

    /// Set the job timeout in seconds.
    fn with_timeout(&mut self, timeout_secs: u32) -> PyResult<()> {
        if timeout_secs == 0 {
            return Err(PyValueError::new_err(
                "job_timeout_secs must be greater than 0",
            ));
        }
        self.job_timeout_secs = timeout_secs;
        Ok(())
    }

    /// Set the maximum concurrent jobs per worker.
    fn with_max_concurrent(&mut self, max_concurrent: u32) -> PyResult<()> {
        if max_concurrent == 0 {
            return Err(PyValueError::new_err(
                "max_concurrent must be greater than 0",
            ));
        }
        self.max_concurrent_per_worker = max_concurrent;
        Ok(())
    }

    /// Enable or disable fault tolerance.
    fn with_fault_tolerance(&mut self, enable: bool) {
        self.fault_tolerance = enable;
    }

    fn __repr__(&self) -> String {
        format!(
            "PyClusterConfig(coordinator='{}', max_workers={}, heartbeat={}s, timeout={}s)",
            self.coordinator_address,
            self.max_workers,
            self.heartbeat_interval_secs,
            self.job_timeout_secs,
        )
    }
}

// ---------------------------------------------------------------------------
// PyWorkerInfo
// ---------------------------------------------------------------------------

/// Information about a worker in the distributed cluster.
#[pyclass]
#[derive(Clone)]
pub struct PyWorkerInfo {
    /// Unique worker identifier.
    #[pyo3(get)]
    pub id: String,
    /// Human-readable worker name.
    #[pyo3(get)]
    pub name: String,
    /// Worker network address.
    #[pyo3(get)]
    pub address: String,
    /// Worker status: idle, busy, overloaded, draining, offline.
    #[pyo3(get)]
    pub status: String,
    /// List of codec/format capabilities.
    #[pyo3(get)]
    pub capabilities: Vec<String>,
    /// Currently assigned job ID, if any.
    #[pyo3(get)]
    pub current_job: Option<String>,
    /// Number of completed jobs.
    #[pyo3(get)]
    pub completed_jobs: u64,
    /// Maximum concurrent tasks this worker supports.
    #[pyo3(get)]
    pub max_concurrent: u32,
}

#[pymethods]
impl PyWorkerInfo {
    fn __repr__(&self) -> String {
        format!(
            "PyWorkerInfo(id='{}', name='{}', status='{}', caps={:?}, completed={})",
            self.id, self.name, self.status, self.capabilities, self.completed_jobs,
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
                "address".to_string(),
                self.address.clone().into_pyobject(py).expect("str").into(),
            );
            m.insert(
                "status".to_string(),
                self.status.clone().into_pyobject(py).expect("str").into(),
            );
            m.insert(
                "capabilities".to_string(),
                self.capabilities
                    .clone()
                    .into_pyobject(py)
                    .expect("list")
                    .into(),
            );
            m.insert(
                "completed_jobs".to_string(),
                self.completed_jobs.into_pyobject(py).expect("int").into(),
            );
            m.insert(
                "max_concurrent".to_string(),
                self.max_concurrent.into_pyobject(py).expect("int").into(),
            );
            m
        })
    }
}

// ---------------------------------------------------------------------------
// PyDistributedJob
// ---------------------------------------------------------------------------

/// A distributed encoding job.
#[pyclass]
#[derive(Clone)]
pub struct PyDistributedJob {
    /// Unique job identifier.
    #[pyo3(get)]
    pub id: String,
    /// Job status: pending, assigned, in_progress, completed, failed, cancelled.
    #[pyo3(get)]
    pub status: String,
    /// Job progress (0.0 to 1.0).
    #[pyo3(get)]
    pub progress: f64,
    /// Input file path.
    #[pyo3(get)]
    pub input_path: String,
    /// Output file path.
    #[pyo3(get)]
    pub output_path: String,
    /// Total number of chunks.
    #[pyo3(get)]
    pub chunks_total: u32,
    /// Number of completed chunks.
    #[pyo3(get)]
    pub chunks_completed: u32,
    /// Assigned worker ID, if any.
    #[pyo3(get)]
    pub worker_id: Option<String>,
    /// Error message, if the job failed.
    #[pyo3(get)]
    pub error_message: Option<String>,
    /// Target codec.
    #[pyo3(get)]
    pub codec: String,
    /// Job priority: low, normal, high, critical.
    #[pyo3(get)]
    pub priority: String,
}

#[pymethods]
impl PyDistributedJob {
    fn __repr__(&self) -> String {
        format!(
            "PyDistributedJob(id='{}', status='{}', progress={:.1}%, codec='{}', chunks={}/{})",
            self.id,
            self.status,
            self.progress * 100.0,
            self.codec,
            self.chunks_completed,
            self.chunks_total,
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
                "status".to_string(),
                self.status.clone().into_pyobject(py).expect("str").into(),
            );
            m.insert(
                "progress".to_string(),
                self.progress.into_pyobject(py).expect("float").into(),
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
                "chunks_total".to_string(),
                self.chunks_total.into_pyobject(py).expect("int").into(),
            );
            m.insert(
                "chunks_completed".to_string(),
                self.chunks_completed.into_pyobject(py).expect("int").into(),
            );
            m.insert(
                "codec".to_string(),
                self.codec.clone().into_pyobject(py).expect("str").into(),
            );
            m.insert(
                "priority".to_string(),
                self.priority.clone().into_pyobject(py).expect("str").into(),
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
        self.status == "in_progress" || self.status == "assigned"
    }
}

// ---------------------------------------------------------------------------
// PyCluster
// ---------------------------------------------------------------------------

/// A distributed encoding cluster manager.
#[pyclass]
pub struct PyCluster {
    config: PyClusterConfig,
    workers: Vec<PyWorkerInfo>,
    jobs: Vec<PyDistributedJob>,
    next_worker_idx: u64,
    next_job_idx: u64,
}

#[pymethods]
impl PyCluster {
    /// Create a new cluster with the given configuration.
    #[new]
    fn new(config: PyClusterConfig) -> Self {
        Self {
            config,
            workers: Vec::new(),
            jobs: Vec::new(),
            next_worker_idx: 0,
            next_job_idx: 0,
        }
    }

    /// Add a worker to the cluster.
    ///
    /// Args:
    ///     name: Worker name.
    ///     address: Worker network address.
    ///     capabilities: List of codec/format capabilities.
    ///     max_concurrent: Maximum concurrent tasks (default: 4).
    ///
    /// Returns:
    ///     Worker ID.
    #[pyo3(signature = (name, address, capabilities=None, max_concurrent=None))]
    fn add_worker(
        &mut self,
        name: &str,
        address: &str,
        capabilities: Option<Vec<String>>,
        max_concurrent: Option<u32>,
    ) -> PyResult<String> {
        let total = self.workers.len() as u32;
        if total >= self.config.max_workers {
            return Err(PyRuntimeError::new_err(format!(
                "Maximum workers ({}) reached",
                self.config.max_workers
            )));
        }

        self.next_worker_idx += 1;
        let worker_id = format!("worker-{}", self.next_worker_idx);

        let caps = capabilities
            .unwrap_or_else(|| vec!["av1".to_string(), "vp9".to_string(), "opus".to_string()]);

        let worker = PyWorkerInfo {
            id: worker_id.clone(),
            name: name.to_string(),
            address: address.to_string(),
            status: "idle".to_string(),
            capabilities: caps,
            current_job: None,
            completed_jobs: 0,
            max_concurrent: max_concurrent.unwrap_or(4),
        };

        self.workers.push(worker);
        Ok(worker_id)
    }

    /// Remove a worker from the cluster by ID.
    fn remove_worker(&mut self, worker_id: &str) -> PyResult<()> {
        let initial_len = self.workers.len();
        self.workers.retain(|w| w.id != worker_id);
        if self.workers.len() == initial_len {
            return Err(PyValueError::new_err(format!(
                "Worker '{}' not found",
                worker_id
            )));
        }
        Ok(())
    }

    /// Submit a job to the cluster.
    ///
    /// Args:
    ///     input_path: Input media file path.
    ///     output_path: Output file path.
    ///     codec: Target codec (default: "av1").
    ///     preset: Encoding preset (default: "medium").
    ///     priority: Job priority: low, normal, high, critical (default: "normal").
    ///     chunks: Number of chunks to split into (default: 1).
    ///
    /// Returns:
    ///     Job ID.
    #[pyo3(signature = (input_path, output_path, codec=None, preset=None, priority=None, chunks=None))]
    fn submit_job(
        &mut self,
        input_path: &str,
        output_path: &str,
        codec: Option<&str>,
        preset: Option<&str>,
        priority: Option<&str>,
        chunks: Option<u32>,
    ) -> PyResult<String> {
        let pri = priority.unwrap_or("normal");
        validate_priority(pri)?;
        let target_codec = codec.unwrap_or("av1");
        let _preset_val = preset.unwrap_or("medium");
        let chunk_count = chunks.unwrap_or(1);

        self.next_job_idx += 1;
        let job_id = format!("job-{}", self.next_job_idx);

        let job = PyDistributedJob {
            id: job_id.clone(),
            status: "pending".to_string(),
            progress: 0.0,
            input_path: input_path.to_string(),
            output_path: output_path.to_string(),
            chunks_total: chunk_count,
            chunks_completed: 0,
            worker_id: None,
            error_message: None,
            codec: target_codec.to_string(),
            priority: pri.to_string(),
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
    fn job_status(&self, job_id: &str) -> PyResult<PyDistributedJob> {
        self.jobs
            .iter()
            .find(|j| j.id == job_id)
            .cloned()
            .ok_or_else(|| PyValueError::new_err(format!("Job '{}' not found", job_id)))
    }

    /// List all workers.
    fn list_workers(&self) -> Vec<PyWorkerInfo> {
        self.workers.clone()
    }

    /// List all jobs.
    fn list_jobs(&self) -> Vec<PyDistributedJob> {
        self.jobs.clone()
    }

    /// Get the number of active (non-completed, non-cancelled) jobs.
    fn active_job_count(&self) -> usize {
        self.jobs
            .iter()
            .filter(|j| j.status != "completed" && j.status != "cancelled" && j.status != "failed")
            .count()
    }

    /// Get the number of workers.
    fn worker_count(&self) -> usize {
        self.workers.len()
    }

    /// Get the cluster configuration.
    fn get_config(&self) -> PyClusterConfig {
        self.config.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "PyCluster(coordinator='{}', workers={}, jobs={}, active={})",
            self.config.coordinator_address,
            self.workers.len(),
            self.jobs.len(),
            self.active_job_count(),
        )
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Create a new cluster with the given coordinator address.
///
/// Args:
///     coordinator_address: Address of the coordinator (default: "127.0.0.1:50051").
///
/// Returns:
///     A new PyCluster instance.
#[pyfunction]
#[pyo3(signature = (coordinator_address=None))]
pub fn create_cluster(coordinator_address: Option<&str>) -> PyCluster {
    let config = PyClusterConfig::new(coordinator_address);
    PyCluster::new(config)
}

/// List available job priorities.
#[pyfunction]
pub fn list_distributed_priorities() -> Vec<String> {
    vec![
        "low".to_string(),
        "normal".to_string(),
        "high".to_string(),
        "critical".to_string(),
    ]
}

/// List available split strategies.
#[pyfunction]
pub fn list_split_strategies() -> Vec<String> {
    vec!["segment".to_string(), "tile".to_string(), "gop".to_string()]
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn validate_priority(priority: &str) -> PyResult<()> {
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

/// Register all distributed bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyClusterConfig>()?;
    m.add_class::<PyWorkerInfo>()?;
    m.add_class::<PyDistributedJob>()?;
    m.add_class::<PyCluster>()?;
    m.add_function(wrap_pyfunction!(create_cluster, m)?)?;
    m.add_function(wrap_pyfunction!(list_distributed_priorities, m)?)?;
    m.add_function(wrap_pyfunction!(list_split_strategies, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cluster_config_default() {
        let config = PyClusterConfig::new(None);
        assert_eq!(config.coordinator_address, "127.0.0.1:50051");
        assert_eq!(config.max_workers, 64);
        assert_eq!(config.heartbeat_interval_secs, 30);
        assert_eq!(config.job_timeout_secs, 3600);
        assert!(config.fault_tolerance);
    }

    #[test]
    fn test_cluster_config_custom() {
        let mut config = PyClusterConfig::new(Some("10.0.0.1:9000"));
        assert_eq!(config.coordinator_address, "10.0.0.1:9000");
        assert!(config.with_max_workers(16).is_ok());
        assert_eq!(config.max_workers, 16);
        assert!(config.with_max_workers(0).is_err());
    }

    #[test]
    fn test_cluster_add_remove_worker() {
        let config = PyClusterConfig::new(None);
        let mut cluster = PyCluster::new(config);

        let id = cluster
            .add_worker("w1", "10.0.0.2:8080", None, None)
            .expect("add_worker should succeed");
        assert_eq!(cluster.worker_count(), 1);

        cluster
            .remove_worker(&id)
            .expect("remove_worker should succeed");
        assert_eq!(cluster.worker_count(), 0);
    }

    #[test]
    fn test_cluster_submit_and_cancel_job() {
        let config = PyClusterConfig::new(None);
        let mut cluster = PyCluster::new(config);

        let job_id = cluster
            .submit_job("/tmp/in.mkv", "/tmp/out.webm", None, None, None, None)
            .expect("submit_job should succeed");
        assert_eq!(cluster.active_job_count(), 1);

        let status = cluster
            .job_status(&job_id)
            .expect("job_status should succeed");
        assert_eq!(status.status, "pending");
        assert!(!status.is_complete());
        assert!(!status.is_failed());

        cluster
            .cancel_job(&job_id)
            .expect("cancel_job should succeed");
        let status = cluster
            .job_status(&job_id)
            .expect("job_status should succeed");
        assert_eq!(status.status, "cancelled");
        assert_eq!(cluster.active_job_count(), 0);
    }

    #[test]
    fn test_validate_priority() {
        assert!(validate_priority("low").is_ok());
        assert!(validate_priority("normal").is_ok());
        assert!(validate_priority("high").is_ok());
        assert!(validate_priority("critical").is_ok());
        assert!(validate_priority("invalid").is_err());
    }

    #[test]
    fn test_distributed_job_methods() {
        let job = PyDistributedJob {
            id: "job-1".to_string(),
            status: "completed".to_string(),
            progress: 1.0,
            input_path: "/tmp/in.mkv".to_string(),
            output_path: "/tmp/out.webm".to_string(),
            chunks_total: 4,
            chunks_completed: 4,
            worker_id: Some("worker-1".to_string()),
            error_message: None,
            codec: "av1".to_string(),
            priority: "normal".to_string(),
        };
        assert!(job.is_complete());
        assert!(!job.is_failed());
        assert!(!job.is_running());
    }
}
