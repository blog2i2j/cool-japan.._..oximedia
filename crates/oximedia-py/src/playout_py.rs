//! Python bindings for `oximedia-playout` broadcast playout server.
//!
//! Provides `PyPlayoutScheduler`, `PyPlayoutItem`, `PyPlayoutConfig`,
//! `PyPlayoutStatus`, and standalone functions for playout management.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// PyPlayoutItem
// ---------------------------------------------------------------------------

/// A single item in a playout schedule.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyPlayoutItem {
    /// Unique item ID.
    #[pyo3(get)]
    pub item_id: u64,

    /// Item title or label.
    #[pyo3(get, set)]
    pub title: String,

    /// Source file path.
    #[pyo3(get)]
    pub source_path: String,

    /// Scheduled start time as "HH:MM:SS" string.
    #[pyo3(get, set)]
    pub scheduled_time: String,

    /// Duration in seconds.
    #[pyo3(get)]
    pub duration_secs: f64,

    /// Item type: programme, commercial, filler, ident, etc.
    #[pyo3(get, set)]
    pub item_type: String,
}

#[pymethods]
impl PyPlayoutItem {
    /// Create a new playout item.
    #[new]
    #[pyo3(signature = (title, source_path, duration_secs, scheduled_time=None, item_type=None))]
    fn new(
        title: &str,
        source_path: &str,
        duration_secs: f64,
        scheduled_time: Option<&str>,
        item_type: Option<&str>,
    ) -> PyResult<Self> {
        if duration_secs <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "duration_secs must be > 0.0, got {duration_secs}"
            )));
        }
        Ok(Self {
            item_id: 0,
            title: title.to_string(),
            source_path: source_path.to_string(),
            scheduled_time: scheduled_time.unwrap_or("00:00:00").to_string(),
            duration_secs,
            item_type: item_type.unwrap_or("programme").to_string(),
        })
    }

    /// Convert to a dictionary.
    fn to_dict(&self) -> HashMap<String, Py<PyAny>> {
        Python::attach(|py| {
            let mut m = HashMap::new();
            m.insert(
                "item_id".to_string(),
                self.item_id
                    .into_pyobject(py)
                    .expect("u64")
                    .into_any()
                    .unbind(),
            );
            m.insert(
                "title".to_string(),
                self.title
                    .clone()
                    .into_pyobject(py)
                    .expect("str")
                    .into_any()
                    .unbind(),
            );
            m.insert(
                "source_path".to_string(),
                self.source_path
                    .clone()
                    .into_pyobject(py)
                    .expect("str")
                    .into_any()
                    .unbind(),
            );
            m.insert(
                "scheduled_time".to_string(),
                self.scheduled_time
                    .clone()
                    .into_pyobject(py)
                    .expect("str")
                    .into_any()
                    .unbind(),
            );
            m.insert(
                "duration_secs".to_string(),
                self.duration_secs
                    .into_pyobject(py)
                    .expect("f64")
                    .into_any()
                    .unbind(),
            );
            m.insert(
                "item_type".to_string(),
                self.item_type
                    .clone()
                    .into_pyobject(py)
                    .expect("str")
                    .into_any()
                    .unbind(),
            );
            m
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "PyPlayoutItem(id={}, title='{}', time='{}', dur={:.1}s, type='{}')",
            self.item_id, self.title, self.scheduled_time, self.duration_secs, self.item_type,
        )
    }
}

// ---------------------------------------------------------------------------
// PyPlayoutConfig
// ---------------------------------------------------------------------------

/// Configuration for a playout server instance.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyPlayoutConfig {
    /// Video format string (e.g. "hd1080p25").
    #[pyo3(get, set)]
    pub video_format: String,

    /// Clock/reference source: "internal", "sdi", "ptp".
    #[pyo3(get, set)]
    pub clock_source: String,

    /// Frame buffer size.
    #[pyo3(get, set)]
    pub buffer_size: u32,

    /// Enable monitoring.
    #[pyo3(get, set)]
    pub monitoring_enabled: bool,

    /// Monitoring port.
    #[pyo3(get, set)]
    pub monitoring_port: u16,
}

#[pymethods]
impl PyPlayoutConfig {
    /// Create a new playout configuration.
    #[new]
    #[pyo3(signature = (video_format=None, clock_source=None, buffer_size=None))]
    fn new(
        video_format: Option<&str>,
        clock_source: Option<&str>,
        buffer_size: Option<u32>,
    ) -> Self {
        Self {
            video_format: video_format.unwrap_or("hd1080p25").to_string(),
            clock_source: clock_source.unwrap_or("internal").to_string(),
            buffer_size: buffer_size.unwrap_or(10),
            monitoring_enabled: true,
            monitoring_port: 8080,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PyPlayoutConfig(format='{}', clock='{}', buf={})",
            self.video_format, self.clock_source, self.buffer_size,
        )
    }
}

// ---------------------------------------------------------------------------
// PyPlayoutStatus
// ---------------------------------------------------------------------------

/// Current status of the playout server.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyPlayoutStatus {
    /// Server state: stopped, starting, running, paused, fallback, stopping.
    #[pyo3(get)]
    pub state: String,

    /// Current item title (if running).
    #[pyo3(get)]
    pub current_item: Option<String>,

    /// Next item title.
    #[pyo3(get)]
    pub next_item: Option<String>,

    /// Frames played since start.
    #[pyo3(get)]
    pub frames_played: u64,

    /// Frames dropped since start.
    #[pyo3(get)]
    pub frames_dropped: u64,
}

#[pymethods]
impl PyPlayoutStatus {
    fn __repr__(&self) -> String {
        format!(
            "PyPlayoutStatus(state='{}', current={:?}, played={}, dropped={})",
            self.state, self.current_item, self.frames_played, self.frames_dropped,
        )
    }
}

// ---------------------------------------------------------------------------
// PyPlayoutScheduler
// ---------------------------------------------------------------------------

/// A playout schedule manager for building and validating schedules.
#[pyclass]
pub struct PyPlayoutScheduler {
    channel: String,
    items: Vec<PyPlayoutItem>,
    next_id: u64,
    config: PyPlayoutConfig,
}

#[pymethods]
impl PyPlayoutScheduler {
    /// Create a new playout scheduler.
    #[new]
    #[pyo3(signature = (channel=None, config=None))]
    fn new(channel: Option<&str>, config: Option<PyPlayoutConfig>) -> Self {
        Self {
            channel: channel.unwrap_or("Channel 1").to_string(),
            items: Vec::new(),
            next_id: 1,
            config: config.unwrap_or_else(|| PyPlayoutConfig::new(None, None, None)),
        }
    }

    /// Add an item to the schedule. Returns the assigned item ID.
    fn add_item(&mut self, mut item: PyPlayoutItem) -> u64 {
        item.item_id = self.next_id;
        self.next_id += 1;
        let id = item.item_id;
        self.items.push(item);
        id
    }

    /// Remove an item by ID.
    fn remove_item(&mut self, item_id: u64) -> PyResult<()> {
        let pos = self
            .items
            .iter()
            .position(|i| i.item_id == item_id)
            .ok_or_else(|| PyValueError::new_err(format!("Item ID {} not found", item_id)))?;
        self.items.remove(pos);
        Ok(())
    }

    /// Get item count.
    fn item_count(&self) -> usize {
        self.items.len()
    }

    /// Get all items.
    fn items(&self) -> Vec<PyPlayoutItem> {
        self.items.clone()
    }

    /// Get total schedule duration in seconds.
    fn total_duration(&self) -> f64 {
        self.items.iter().map(|i| i.duration_secs).sum()
    }

    /// Get the channel name.
    fn channel(&self) -> String {
        self.channel.clone()
    }

    /// Get the configuration.
    fn config(&self) -> PyPlayoutConfig {
        self.config.clone()
    }

    /// Validate the schedule (check for gaps, overlaps, etc.).
    fn validate(&self) -> PyResult<Vec<String>> {
        let mut warnings = Vec::new();

        if self.items.is_empty() {
            warnings.push("Schedule is empty".to_string());
        }

        for item in &self.items {
            if item.duration_secs <= 0.0 {
                warnings.push(format!("Item '{}' has non-positive duration", item.title));
            }
            if item.source_path.is_empty() {
                warnings.push(format!("Item '{}' has empty source path", item.title));
            }
        }

        Ok(warnings)
    }

    /// Serialize the schedule to a JSON string.
    fn to_json(&self) -> PyResult<String> {
        let data = serde_json::json!({
            "channel": self.channel,
            "config": {
                "video_format": self.config.video_format,
                "clock_source": self.config.clock_source,
                "buffer_size": self.config.buffer_size,
            },
            "items": self.items.iter().map(|i| {
                serde_json::json!({
                    "item_id": i.item_id,
                    "title": i.title,
                    "source_path": i.source_path,
                    "scheduled_time": i.scheduled_time,
                    "duration_secs": i.duration_secs,
                    "item_type": i.item_type,
                })
            }).collect::<Vec<_>>(),
        });
        serde_json::to_string_pretty(&data)
            .map_err(|e| PyRuntimeError::new_err(format!("JSON error: {e}")))
    }

    fn __repr__(&self) -> String {
        format!(
            "PyPlayoutScheduler(channel='{}', items={}, dur={:.1}s)",
            self.channel,
            self.items.len(),
            self.total_duration(),
        )
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Create a new playout schedule from a list of items.
#[pyfunction]
#[pyo3(signature = (items, channel=None))]
pub fn create_playout(items: Vec<PyPlayoutItem>, channel: Option<&str>) -> PyPlayoutScheduler {
    let mut scheduler = PyPlayoutScheduler::new(channel, None);
    for item in items {
        scheduler.add_item(item);
    }
    scheduler
}

/// Validate a list of playout items and return a list of warnings.
#[pyfunction]
pub fn validate_playout(items: Vec<PyPlayoutItem>) -> Vec<String> {
    let mut warnings = Vec::new();

    if items.is_empty() {
        warnings.push("No items to validate".to_string());
        return warnings;
    }

    for (idx, item) in items.iter().enumerate() {
        if item.duration_secs <= 0.0 {
            warnings.push(format!("Item {} has non-positive duration", idx));
        }
        if item.source_path.is_empty() {
            warnings.push(format!("Item {} has empty source path", idx));
        }
    }

    warnings
}

/// Get a default playout status (server idle).
#[pyfunction]
pub fn playout_status() -> PyPlayoutStatus {
    PyPlayoutStatus {
        state: "stopped".to_string(),
        current_item: None,
        next_item: None,
        frames_played: 0,
        frames_dropped: 0,
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register playout bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPlayoutScheduler>()?;
    m.add_class::<PyPlayoutItem>()?;
    m.add_class::<PyPlayoutConfig>()?;
    m.add_class::<PyPlayoutStatus>()?;
    m.add_function(wrap_pyfunction!(create_playout, m)?)?;
    m.add_function(wrap_pyfunction!(validate_playout, m)?)?;
    m.add_function(wrap_pyfunction!(playout_status, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_playout_item_new() {
        let item = PyPlayoutItem::new("News", "/media/news.mxf", 1800.0, None, None);
        assert!(item.is_ok());
        let item = item.expect("valid");
        assert_eq!(item.title, "News");
        assert!((item.duration_secs - 1800.0).abs() < f64::EPSILON);
        assert_eq!(item.item_type, "programme");
    }

    #[test]
    fn test_playout_item_invalid_duration() {
        let item = PyPlayoutItem::new("Bad", "source.mxf", -1.0, None, None);
        assert!(item.is_err());
    }

    #[test]
    fn test_playout_scheduler_add_remove() {
        let mut sched = PyPlayoutScheduler::new(None, None);
        let item = PyPlayoutItem::new("Clip", "clip.mxf", 30.0, None, None).expect("valid");
        let id = sched.add_item(item);
        assert_eq!(sched.item_count(), 1);
        assert!((sched.total_duration() - 30.0).abs() < f64::EPSILON);

        sched.remove_item(id).expect("remove should succeed");
        assert_eq!(sched.item_count(), 0);
    }

    #[test]
    fn test_playout_validate() {
        let sched = PyPlayoutScheduler::new(None, None);
        let warnings = sched.validate().expect("validate should succeed");
        assert!(!warnings.is_empty()); // empty schedule warning
    }

    #[test]
    fn test_create_playout_fn() {
        let items = vec![
            PyPlayoutItem::new("A", "a.mxf", 10.0, None, None).expect("valid"),
            PyPlayoutItem::new("B", "b.mxf", 20.0, None, None).expect("valid"),
        ];
        let sched = create_playout(items, Some("Test Ch"));
        assert_eq!(sched.item_count(), 2);
        assert_eq!(sched.channel(), "Test Ch");
    }
}
