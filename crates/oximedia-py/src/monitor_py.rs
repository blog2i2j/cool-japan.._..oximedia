//! Python bindings for `oximedia-monitor` system monitoring.
//!
//! Provides `PyStreamMonitor`, `PyMonitorConfig`, `PyMonitorAlert`,
//! `PyMonitorStatus`, and convenience functions for health checks.

use pyo3::prelude::*;
use std::collections::HashMap;

use oximedia_monitor::{
    CodecMetrics, Comparison, NotificationAction, SimpleAlertManager, SimpleAlertRule,
    SimpleMetricsCollector, SimpleMetricsSnapshot,
};

// ---------------------------------------------------------------------------
// PyMonitorConfig
// ---------------------------------------------------------------------------

/// Configuration for stream/system monitoring.
#[pyclass]
#[derive(Clone)]
pub struct PyMonitorConfig {
    /// Enable system metrics (CPU, memory, disk).
    #[pyo3(get, set)]
    pub enable_system_metrics: bool,
    /// Enable quality metrics (PSNR, SSIM, bitrate).
    #[pyo3(get, set)]
    pub enable_quality_metrics: bool,
    /// CPU alert threshold (percentage).
    #[pyo3(get, set)]
    pub cpu_threshold: f64,
    /// Memory alert threshold (percentage).
    #[pyo3(get, set)]
    pub memory_threshold: f64,
    /// Quality score alert threshold (0-100).
    #[pyo3(get, set)]
    pub quality_threshold: f64,
    /// Collection interval in milliseconds.
    #[pyo3(get, set)]
    pub interval_ms: u64,
}

#[pymethods]
impl PyMonitorConfig {
    /// Create a new monitor configuration with default values.
    #[new]
    fn new() -> Self {
        Self {
            enable_system_metrics: true,
            enable_quality_metrics: true,
            cpu_threshold: 90.0,
            memory_threshold: 90.0,
            quality_threshold: 50.0,
            interval_ms: 1000,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PyMonitorConfig(cpu_thresh={:.1}, mem_thresh={:.1}, quality_thresh={:.1}, interval={}ms)",
            self.cpu_threshold, self.memory_threshold, self.quality_threshold, self.interval_ms,
        )
    }
}

// ---------------------------------------------------------------------------
// PyMonitorAlert
// ---------------------------------------------------------------------------

/// A monitoring alert.
#[pyclass]
#[derive(Clone)]
pub struct PyMonitorAlert {
    /// Alert timestamp (ISO 8601 string).
    #[pyo3(get)]
    pub timestamp: String,
    /// Severity level (info, warning, error, critical).
    #[pyo3(get)]
    pub severity: String,
    /// Alert message.
    #[pyo3(get)]
    pub message: String,
    /// Name of the rule that triggered the alert.
    #[pyo3(get)]
    pub rule_name: String,
}

#[pymethods]
impl PyMonitorAlert {
    fn __repr__(&self) -> String {
        format!(
            "PyMonitorAlert(severity='{}', message='{}', rule='{}')",
            self.severity, self.message, self.rule_name,
        )
    }

    /// Convert to a Python dict.
    fn to_dict(&self, py: Python<'_>) -> PyResult<HashMap<String, Py<PyAny>>> {
        let mut m = HashMap::new();
        m.insert(
            "timestamp".to_string(),
            self.timestamp
                .clone()
                .into_pyobject(py)?
                .into_any()
                .unbind(),
        );
        m.insert(
            "severity".to_string(),
            self.severity.clone().into_pyobject(py)?.into_any().unbind(),
        );
        m.insert(
            "message".to_string(),
            self.message.clone().into_pyobject(py)?.into_any().unbind(),
        );
        m.insert(
            "rule_name".to_string(),
            self.rule_name
                .clone()
                .into_pyobject(py)?
                .into_any()
                .unbind(),
        );
        Ok(m)
    }
}

// ---------------------------------------------------------------------------
// PyMonitorStatus
// ---------------------------------------------------------------------------

/// A point-in-time monitoring status snapshot.
#[pyclass]
#[derive(Clone)]
pub struct PyMonitorStatus {
    /// CPU usage percentage.
    #[pyo3(get)]
    pub cpu_percent: f64,
    /// Memory usage in megabytes.
    #[pyo3(get)]
    pub memory_mb: f64,
    /// Disk I/O throughput in MB/s.
    #[pyo3(get)]
    pub disk_io_mbps: f64,
    /// Number of active codec sessions.
    #[pyo3(get)]
    pub codec_count: usize,
    /// Timestamp (ISO 8601 string).
    #[pyo3(get)]
    pub timestamp: String,
}

#[pymethods]
impl PyMonitorStatus {
    fn __repr__(&self) -> String {
        format!(
            "PyMonitorStatus(cpu={:.1}%, mem={:.1}MB, disk={:.2}MB/s, codecs={})",
            self.cpu_percent, self.memory_mb, self.disk_io_mbps, self.codec_count,
        )
    }

    /// Check if all metrics are within healthy ranges.
    fn is_healthy(&self, cpu_thresh: f64, memory_thresh_mb: f64) -> bool {
        self.cpu_percent < cpu_thresh && self.memory_mb < memory_thresh_mb
    }
}

fn snapshot_to_status(snap: &SimpleMetricsSnapshot) -> PyMonitorStatus {
    PyMonitorStatus {
        cpu_percent: snap.cpu_percent,
        memory_mb: snap.memory_mb,
        disk_io_mbps: snap.disk_io_mbps,
        codec_count: snap.codecs.len(),
        timestamp: snap.timestamp.to_rfc3339(),
    }
}

// ---------------------------------------------------------------------------
// PyStreamMonitor
// ---------------------------------------------------------------------------

/// Stream and system monitor with alerting.
#[pyclass]
pub struct PyStreamMonitor {
    collector: SimpleMetricsCollector,
    alert_manager: SimpleAlertManager,
    config: PyMonitorConfig,
}

#[pymethods]
impl PyStreamMonitor {
    /// Create a new stream monitor.
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<&PyMonitorConfig>) -> PyResult<Self> {
        let cfg = config.cloned().unwrap_or_else(PyMonitorConfig::new);

        let collector = SimpleMetricsCollector::new();

        let alert_manager = SimpleAlertManager::new();
        alert_manager.add_rule(SimpleAlertRule::new(
            "cpu_high",
            "cpu",
            Comparison::GreaterThan,
            cfg.cpu_threshold,
            NotificationAction::Log("CPU usage high".to_string()),
        ));
        alert_manager.add_rule(SimpleAlertRule::new(
            "memory_high",
            "memory",
            Comparison::GreaterThan,
            cfg.memory_threshold,
            NotificationAction::Log("Memory usage high".to_string()),
        ));
        alert_manager.add_rule(SimpleAlertRule::new(
            "quality_low",
            "quality",
            Comparison::LessThan,
            cfg.quality_threshold,
            NotificationAction::Log("Quality below threshold".to_string()),
        ));

        Ok(Self {
            collector,
            alert_manager,
            config: cfg,
        })
    }

    /// Take a metrics snapshot and return the current status.
    fn poll(&self) -> PyResult<PyMonitorStatus> {
        let snap = self.collector.snapshot();
        Ok(snapshot_to_status(&snap))
    }

    /// Get current status without polling new data.
    fn status(&self) -> PyResult<PyMonitorStatus> {
        let snap = self.collector.snapshot();
        Ok(snapshot_to_status(&snap))
    }

    /// Get recent alerts.
    fn alerts(&self, count: usize) -> Vec<PyMonitorAlert> {
        let history = self.alert_manager.history();
        history
            .iter()
            .rev()
            .take(count)
            .map(|a| PyMonitorAlert {
                timestamp: a.timestamp.to_rfc3339(),
                severity: "warning".to_string(),
                message: format!("Rule '{}' fired (value={:.2})", a.rule_name, a.metric_value),
                rule_name: a.rule_name.clone(),
            })
            .collect()
    }

    /// Record codec metrics for a given codec name.
    fn record_codec(
        &self,
        name: &str,
        fps: f64,
        bitrate_kbps: f64,
        quality_score: f64,
    ) -> PyResult<()> {
        let mut metrics = CodecMetrics::new(name);
        metrics.fps = fps;
        metrics.bitrate_kbps = bitrate_kbps;
        metrics.quality_score = quality_score;
        self.collector.poll_codec(metrics);
        Ok(())
    }

    /// Get the current configuration.
    fn get_config(&self) -> PyMonitorConfig {
        self.config.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "PyStreamMonitor(cpu_thresh={:.1}, mem_thresh={:.1})",
            self.config.cpu_threshold, self.config.memory_threshold,
        )
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Create a new stream monitor with the given configuration.
///
/// Args:
///     config: Optional monitoring configuration.
///
/// Returns:
///     A configured PyStreamMonitor instance.
#[pyfunction]
#[pyo3(signature = (config=None))]
pub fn create_monitor(config: Option<&PyMonitorConfig>) -> PyResult<PyStreamMonitor> {
    PyStreamMonitor::new(config)
}

/// Quick health check: returns a status snapshot.
///
/// Returns:
///     PyMonitorStatus with current system metrics.
#[pyfunction]
pub fn check_stream_health() -> PyResult<PyMonitorStatus> {
    let collector = SimpleMetricsCollector::new();
    let snap = collector.snapshot();
    Ok(snapshot_to_status(&snap))
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register all monitor bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMonitorConfig>()?;
    m.add_class::<PyMonitorAlert>()?;
    m.add_class::<PyMonitorStatus>()?;
    m.add_class::<PyStreamMonitor>()?;
    m.add_function(wrap_pyfunction!(create_monitor, m)?)?;
    m.add_function(wrap_pyfunction!(check_stream_health, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let cfg = PyMonitorConfig::new();
        assert!(cfg.enable_system_metrics);
        assert!(cfg.enable_quality_metrics);
        assert!((cfg.cpu_threshold - 90.0).abs() < f64::EPSILON);
        assert!((cfg.memory_threshold - 90.0).abs() < f64::EPSILON);
        assert!((cfg.quality_threshold - 50.0).abs() < f64::EPSILON);
        assert_eq!(cfg.interval_ms, 1000);
    }

    #[test]
    fn test_config_repr() {
        let cfg = PyMonitorConfig::new();
        let repr = cfg.__repr__();
        assert!(repr.contains("90.0"));
        assert!(repr.contains("1000ms"));
    }

    #[test]
    fn test_alert_repr() {
        let alert = PyMonitorAlert {
            timestamp: "2026-01-01T00:00:00Z".to_string(),
            severity: "warning".to_string(),
            message: "CPU high".to_string(),
            rule_name: "cpu_threshold".to_string(),
        };
        let repr = alert.__repr__();
        assert!(repr.contains("warning"));
        assert!(repr.contains("CPU high"));
    }

    #[test]
    fn test_status_healthy() {
        let status = PyMonitorStatus {
            cpu_percent: 50.0,
            memory_mb: 1024.0,
            disk_io_mbps: 100.0,
            codec_count: 0,
            timestamp: "2026-01-01T00:00:00Z".to_string(),
        };
        assert!(status.is_healthy(90.0, 8192.0));
        assert!(!status.is_healthy(40.0, 8192.0));
    }

    #[test]
    fn test_status_repr() {
        let status = PyMonitorStatus {
            cpu_percent: 45.5,
            memory_mb: 2048.0,
            disk_io_mbps: 50.0,
            codec_count: 2,
            timestamp: "2026-01-01T00:00:00Z".to_string(),
        };
        let repr = status.__repr__();
        assert!(repr.contains("45.5%"));
        assert!(repr.contains("codecs=2"));
    }
}
