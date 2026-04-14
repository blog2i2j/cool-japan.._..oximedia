#![allow(dead_code)]
//! Callback-based progress reporting for long-running operations.
//!
//! Provides structured progress events with phase tracking, ETA estimation,
//! and cancellation support for encode, transcode, and analysis operations.
//!
//! # Example (Python)
//!
//! ```python
//! import oximedia
//!
//! def on_event(event):
//!     print(f"Phase: {event.phase}, Progress: {event.percentage:.1f}%")
//!     if event.eta_seconds is not None:
//!         print(f"  ETA: {event.eta_seconds:.0f}s")
//!
//! reporter = oximedia.ProgressReporter(total_steps=1000, callback=on_event)
//! reporter.begin("encoding")
//! for i in range(1000):
//!     reporter.advance(1)
//! reporter.finish()
//! ```

use pyo3::prelude::*;

use std::time::Instant;

// ---------------------------------------------------------------------------
// ProgressPhase
// ---------------------------------------------------------------------------

/// Named phase of a multi-phase operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProgressPhase {
    /// Operation has not started.
    Idle,
    /// Initialisation / setup.
    Init,
    /// Analysis / probing.
    Analyzing,
    /// Encoding frames.
    Encoding,
    /// Decoding frames.
    Decoding,
    /// Transcoding (decode + encode).
    Transcoding,
    /// Muxing output.
    Muxing,
    /// Finalising / flushing.
    Finalizing,
    /// Operation complete.
    Complete,
    /// A user-defined phase name.
    Custom(String),
}

impl ProgressPhase {
    /// Convert to a string label.
    pub fn label(&self) -> &str {
        match self {
            Self::Idle => "idle",
            Self::Init => "init",
            Self::Analyzing => "analyzing",
            Self::Encoding => "encoding",
            Self::Decoding => "decoding",
            Self::Transcoding => "transcoding",
            Self::Muxing => "muxing",
            Self::Finalizing => "finalizing",
            Self::Complete => "complete",
            Self::Custom(s) => s.as_str(),
        }
    }

    /// Parse a phase from a string label.
    pub fn from_label(label: &str) -> Self {
        match label {
            "idle" => Self::Idle,
            "init" => Self::Init,
            "analyzing" => Self::Analyzing,
            "encoding" => Self::Encoding,
            "decoding" => Self::Decoding,
            "transcoding" => Self::Transcoding,
            "muxing" => Self::Muxing,
            "finalizing" => Self::Finalizing,
            "complete" => Self::Complete,
            other => Self::Custom(other.to_string()),
        }
    }
}

// ---------------------------------------------------------------------------
// PyProgressEvent
// ---------------------------------------------------------------------------

/// A progress event emitted by [`ProgressReporter`].
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyProgressEvent {
    /// Current phase label.
    #[pyo3(get)]
    pub phase: String,
    /// Steps completed so far.
    #[pyo3(get)]
    pub steps_done: u64,
    /// Total steps expected.
    #[pyo3(get)]
    pub total_steps: u64,
    /// Percentage complete (0.0 .. 100.0).
    #[pyo3(get)]
    pub percentage: f64,
    /// Estimated time remaining in seconds, if available.
    #[pyo3(get)]
    pub eta_seconds: Option<f64>,
    /// Elapsed time in seconds since operation start.
    #[pyo3(get)]
    pub elapsed_seconds: f64,
    /// Steps per second throughput.
    #[pyo3(get)]
    pub throughput: f64,
    /// Whether cancellation has been requested.
    #[pyo3(get)]
    pub cancelled: bool,
}

#[pymethods]
impl PyProgressEvent {
    fn __repr__(&self) -> String {
        format!(
            "PyProgressEvent(phase={:?}, {:.1}%, {}/{}, eta={:?}, throughput={:.1}/s)",
            self.phase, self.percentage, self.steps_done, self.total_steps,
            self.eta_seconds, self.throughput
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Whether the operation is complete.
    pub fn is_complete(&self) -> bool {
        self.steps_done >= self.total_steps
    }

    /// Remaining steps.
    pub fn steps_remaining(&self) -> u64 {
        self.total_steps.saturating_sub(self.steps_done)
    }
}

// ---------------------------------------------------------------------------
// ProgressReporter
// ---------------------------------------------------------------------------

/// Callback-based progress reporter for long-running operations.
///
/// Tracks steps, computes ETA, and invokes a Python callback with
/// [`PyProgressEvent`] objects at each update.
#[pyclass]
#[derive(Debug)]
pub struct ProgressReporter {
    /// Total expected steps.
    total_steps: u64,
    /// Steps completed so far.
    steps_done: u64,
    /// Current phase.
    phase: ProgressPhase,
    /// Time when the operation started.
    start_time: Option<Instant>,
    /// Python callback object (optional).
    callback: Option<Py<PyAny>>,
    /// Minimum interval between callback invocations (in steps).
    report_interval: u64,
    /// Steps at last callback invocation.
    last_reported_at: u64,
    /// Whether cancellation has been requested.
    cancelled: bool,
    /// Event history for post-hoc analysis.
    history: Vec<ProgressSnapshot>,
}

/// Internal snapshot for history tracking.
#[derive(Debug, Clone)]
struct ProgressSnapshot {
    steps_done: u64,
    elapsed_secs: f64,
    phase: String,
}

#[pymethods]
impl ProgressReporter {
    /// Create a new progress reporter.
    ///
    /// Parameters
    /// ----------
    /// total_steps : int
    ///     Total expected steps.
    /// callback : callable, optional
    ///     Python callable that receives a `PyProgressEvent` each update.
    /// report_interval : int
    ///     Minimum steps between callback invocations. Defaults to 1.
    #[new]
    #[pyo3(signature = (total_steps=100, callback=None, report_interval=1))]
    pub fn new(
        total_steps: u64,
        callback: Option<Py<PyAny>>,
        report_interval: u64,
    ) -> Self {
        Self {
            total_steps: total_steps.max(1),
            steps_done: 0,
            phase: ProgressPhase::Idle,
            start_time: None,
            callback,
            report_interval: report_interval.max(1),
            last_reported_at: 0,
            cancelled: false,
            history: Vec::new(),
        }
    }

    /// Begin the operation with a named phase.
    pub fn begin(&mut self, py: Python<'_>, phase: &str) -> PyResult<()> {
        self.phase = ProgressPhase::from_label(phase);
        self.start_time = Some(Instant::now());
        self.steps_done = 0;
        self.cancelled = false;
        self.emit_event(py)?;
        Ok(())
    }

    /// Advance by `n` steps and optionally fire callback.
    pub fn advance(&mut self, py: Python<'_>, n: u64) -> PyResult<()> {
        self.steps_done = self.steps_done.saturating_add(n).min(self.total_steps);
        let since_last = self.steps_done.saturating_sub(self.last_reported_at);
        if since_last >= self.report_interval || self.steps_done == self.total_steps {
            self.emit_event(py)?;
            self.last_reported_at = self.steps_done;
        }
        Ok(())
    }

    /// Set the phase label mid-operation.
    pub fn set_phase(&mut self, py: Python<'_>, phase: &str) -> PyResult<()> {
        self.phase = ProgressPhase::from_label(phase);
        self.emit_event(py)?;
        Ok(())
    }

    /// Request cancellation. The next progress event will have `cancelled=True`.
    pub fn cancel(&mut self) {
        self.cancelled = true;
    }

    /// Mark the operation as finished.
    pub fn finish(&mut self, py: Python<'_>) -> PyResult<()> {
        self.steps_done = self.total_steps;
        self.phase = ProgressPhase::Complete;
        self.emit_event(py)?;
        Ok(())
    }

    /// Current percentage (0.0 .. 100.0).
    #[getter]
    pub fn percentage(&self) -> f64 {
        (self.steps_done as f64 / self.total_steps as f64) * 100.0
    }

    /// Current phase label.
    #[getter]
    pub fn phase(&self) -> String {
        self.phase.label().to_string()
    }

    /// Whether the operation has been cancelled.
    #[getter]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled
    }

    /// Steps completed.
    #[getter]
    pub fn steps_done(&self) -> u64 {
        self.steps_done
    }

    /// Total steps.
    #[getter]
    pub fn total_steps(&self) -> u64 {
        self.total_steps
    }

    /// Elapsed seconds since `begin()`.
    #[getter]
    pub fn elapsed_seconds(&self) -> f64 {
        self.start_time.map_or(0.0, |t| t.elapsed().as_secs_f64())
    }

    /// Number of snapshots in the history.
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// Reset the reporter for reuse.
    pub fn reset(&mut self) {
        self.steps_done = 0;
        self.phase = ProgressPhase::Idle;
        self.start_time = None;
        self.last_reported_at = 0;
        self.cancelled = false;
        self.history.clear();
    }

    fn __repr__(&self) -> String {
        format!(
            "ProgressReporter(phase={:?}, {}/{} steps, {:.1}%)",
            self.phase.label(),
            self.steps_done,
            self.total_steps,
            self.percentage()
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

impl ProgressReporter {
    /// Build and emit a progress event.
    fn emit_event(&mut self, py: Python<'_>) -> PyResult<()> {
        let elapsed = self.start_time.map_or(0.0, |t| t.elapsed().as_secs_f64());
        let throughput = if elapsed > 0.0 {
            self.steps_done as f64 / elapsed
        } else {
            0.0
        };
        let eta = if throughput > 0.0 && self.steps_done < self.total_steps {
            let remaining = self.total_steps - self.steps_done;
            Some(remaining as f64 / throughput)
        } else {
            None
        };

        let snapshot = ProgressSnapshot {
            steps_done: self.steps_done,
            elapsed_secs: elapsed,
            phase: self.phase.label().to_string(),
        };
        self.history.push(snapshot);

        let event = PyProgressEvent {
            phase: self.phase.label().to_string(),
            steps_done: self.steps_done,
            total_steps: self.total_steps,
            percentage: self.percentage(),
            eta_seconds: eta,
            elapsed_seconds: elapsed,
            throughput,
            cancelled: self.cancelled,
        };

        if let Some(ref cb) = self.callback {
            let event_obj = Py::new(py, event)?;
            cb.call1(py, (event_obj,))?;
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Register progress reporting types into the parent module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyProgressEvent>()?;
    m.add_class::<ProgressReporter>()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progress_phase_labels() {
        assert_eq!(ProgressPhase::Idle.label(), "idle");
        assert_eq!(ProgressPhase::Encoding.label(), "encoding");
        assert_eq!(ProgressPhase::Complete.label(), "complete");
        assert_eq!(
            ProgressPhase::Custom("my_phase".into()).label(),
            "my_phase"
        );
    }

    #[test]
    fn test_progress_phase_roundtrip() {
        let labels = [
            "idle", "init", "analyzing", "encoding", "decoding",
            "transcoding", "muxing", "finalizing", "complete",
        ];
        for label in labels {
            let phase = ProgressPhase::from_label(label);
            assert_eq!(phase.label(), label);
        }
    }

    #[test]
    fn test_progress_phase_custom_roundtrip() {
        let phase = ProgressPhase::from_label("my_custom_phase");
        assert_eq!(phase.label(), "my_custom_phase");
        assert!(matches!(phase, ProgressPhase::Custom(_)));
    }

    #[test]
    fn test_progress_event_is_complete() {
        let event = PyProgressEvent {
            phase: "encoding".to_string(),
            steps_done: 100,
            total_steps: 100,
            percentage: 100.0,
            eta_seconds: None,
            elapsed_seconds: 5.0,
            throughput: 20.0,
            cancelled: false,
        };
        assert!(event.is_complete());
        assert_eq!(event.steps_remaining(), 0);
    }

    #[test]
    fn test_progress_event_not_complete() {
        let event = PyProgressEvent {
            phase: "encoding".to_string(),
            steps_done: 50,
            total_steps: 100,
            percentage: 50.0,
            eta_seconds: Some(2.5),
            elapsed_seconds: 2.5,
            throughput: 20.0,
            cancelled: false,
        };
        assert!(!event.is_complete());
        assert_eq!(event.steps_remaining(), 50);
    }

    #[test]
    fn test_progress_event_repr() {
        let event = PyProgressEvent {
            phase: "encoding".to_string(),
            steps_done: 50,
            total_steps: 100,
            percentage: 50.0,
            eta_seconds: Some(2.5),
            elapsed_seconds: 2.5,
            throughput: 20.0,
            cancelled: false,
        };
        let repr = event.__repr__();
        assert!(repr.contains("encoding"));
        assert!(repr.contains("50.0%"));
        assert!(repr.contains("50/100"));
    }

    #[test]
    fn test_reporter_new_defaults() {
        let r = ProgressReporter::new(1000, None, 1);
        assert_eq!(r.total_steps(), 1000);
        assert_eq!(r.steps_done(), 0);
        assert!((r.percentage()).abs() < f64::EPSILON);
        assert!(!r.is_cancelled());
        assert_eq!(r.phase(), "idle");
    }

    #[test]
    fn test_reporter_zero_total_clamps() {
        let r = ProgressReporter::new(0, None, 1);
        assert_eq!(r.total_steps(), 1); // clamped to 1
    }

    #[test]
    fn test_reporter_cancel() {
        let mut r = ProgressReporter::new(100, None, 1);
        assert!(!r.is_cancelled());
        r.cancel();
        assert!(r.is_cancelled());
    }

    #[test]
    fn test_reporter_reset() {
        let mut r = ProgressReporter::new(100, None, 1);
        r.steps_done = 50;
        r.phase = ProgressPhase::Encoding;
        r.cancelled = true;
        r.history.push(ProgressSnapshot {
            steps_done: 50,
            elapsed_secs: 1.0,
            phase: "encoding".to_string(),
        });
        r.reset();
        assert_eq!(r.steps_done(), 0);
        assert_eq!(r.phase(), "idle");
        assert!(!r.is_cancelled());
        assert_eq!(r.history_len(), 0);
    }

    #[test]
    fn test_reporter_repr() {
        let r = ProgressReporter::new(100, None, 1);
        let repr = r.__repr__();
        assert!(repr.contains("idle"));
        assert!(repr.contains("0/100"));
    }

    #[test]
    fn test_reporter_percentage_calculation() {
        let mut r = ProgressReporter::new(200, None, 1);
        r.steps_done = 100;
        assert!((r.percentage() - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_reporter_elapsed_no_start() {
        let r = ProgressReporter::new(100, None, 1);
        assert!((r.elapsed_seconds()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_reporter_history_tracking() {
        let mut r = ProgressReporter::new(100, None, 1);
        r.history.push(ProgressSnapshot {
            steps_done: 0,
            elapsed_secs: 0.0,
            phase: "init".to_string(),
        });
        r.history.push(ProgressSnapshot {
            steps_done: 50,
            elapsed_secs: 1.0,
            phase: "encoding".to_string(),
        });
        assert_eq!(r.history_len(), 2);
    }
}
