//! Python bindings for EDL parsing and timeline manipulation via `oximedia-edl`.
//!
//! Exposes CMX 3600 parsing, event iteration and XML export to Python through
//! the `PyEdl` and `PyEdlEvent` classes.

use pyo3::prelude::*;

use oximedia_edl::{parse_edl, Edl};

// ---------------------------------------------------------------------------
// Python-visible types
// ---------------------------------------------------------------------------

/// A single EDL event exposed to Python.
///
/// Wraps the underlying `EdlEvent` and exposes its most commonly used fields
/// as readable Python attributes.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyEdlEvent {
    /// Sequential event number (1-based).
    #[pyo3(get)]
    pub number: u32,
    /// Reel / source name (e.g. ``"A001"`` or ``"AX"``).
    #[pyo3(get)]
    pub reel: String,
    /// Track type string: ``"Video"``, ``"Audio"``, or ``"Both"``.
    #[pyo3(get)]
    pub track: String,
    /// Edit type string: ``"Cut"``, ``"Dissolve"``, ``"Wipe"``, or ``"Key"``.
    #[pyo3(get)]
    pub edit_type: String,
    /// Source in timecode string (``"HH:MM:SS:FF"``).
    #[pyo3(get)]
    pub source_in: String,
    /// Source out timecode string.
    #[pyo3(get)]
    pub source_out: String,
    /// Record in timecode string.
    #[pyo3(get)]
    pub record_in: String,
    /// Record out timecode string.
    #[pyo3(get)]
    pub record_out: String,
    /// Optional clip name (from ``* FROM CLIP NAME:`` comment lines).
    #[pyo3(get)]
    pub clip_name: Option<String>,
}

#[pymethods]
impl PyEdlEvent {
    fn __repr__(&self) -> String {
        format!(
            "PyEdlEvent(number={}, reel='{}', in={}, out={})",
            self.number, self.reel, self.record_in, self.record_out
        )
    }
}

/// Python representation of a parsed EDL.
///
/// Use `PyEdl.parse_cmx3600(text)` to parse a CMX 3600-format EDL string,
/// then iterate over `.events` and call `.export_xml()` to serialise back.
#[pyclass]
pub struct PyEdl {
    inner: Edl,
}

#[pymethods]
impl PyEdl {
    /// Parse a CMX 3600 EDL from a string.
    ///
    /// Parameters
    /// ----------
    /// text : str
    ///     Raw CMX 3600 EDL text.
    ///
    /// Returns
    /// -------
    /// PyEdl
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If the text cannot be parsed as a valid EDL.
    #[staticmethod]
    #[pyo3(signature = (text))]
    pub fn parse_cmx3600(text: &str) -> PyResult<Self> {
        let edl = parse_edl(text)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: edl })
    }

    /// EDL title string, or ``None`` if not set.
    #[getter]
    pub fn title(&self) -> Option<String> {
        self.inner.title.clone()
    }

    /// Number of events in the EDL.
    #[getter]
    pub fn event_count(&self) -> usize {
        self.inner.event_count()
    }

    /// Total timeline duration in seconds.
    #[getter]
    pub fn duration_seconds(&self) -> f64 {
        self.inner.total_duration_seconds()
    }

    /// List of all events as `PyEdlEvent` objects.
    #[getter]
    pub fn events(&self) -> Vec<PyEdlEvent> {
        self.inner
            .events
            .iter()
            .map(|ev| {
                let track_str = format!("{:?}", ev.track);
                let edit_str = format!("{:?}", ev.edit_type);
                PyEdlEvent {
                    number: ev.number,
                    reel: ev.reel.clone(),
                    track: track_str,
                    edit_type: edit_str,
                    source_in: ev.source_in.to_string(),
                    source_out: ev.source_out.to_string(),
                    record_in: ev.record_in.to_string(),
                    record_out: ev.record_out.to_string(),
                    clip_name: ev.clip_name.clone(),
                }
            })
            .collect()
    }

    /// Export the EDL as a CMX 3600 text string.
    ///
    /// Returns
    /// -------
    /// str
    pub fn export_cmx3600(&self) -> PyResult<String> {
        self.inner
            .to_string_format()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Export the EDL as a simple XML string.
    ///
    /// The schema is intentionally minimal — one ``<edl>`` root, one
    /// ``<event>`` per event, with attributes matching the Python API.
    ///
    /// Returns
    /// -------
    /// str
    pub fn export_xml(&self) -> PyResult<String> {
        let mut xml = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<edl");
        if let Some(ref title) = self.inner.title {
            xml.push_str(&format!(" title=\"{}\"", xml_escape(title)));
        }
        xml.push_str(&format!(
            " duration_seconds=\"{:.3}\">\n",
            self.inner.total_duration_seconds()
        ));

        for ev in &self.inner.events {
            let track_str = format!("{:?}", ev.track);
            let edit_str = format!("{:?}", ev.edit_type);
            xml.push_str(&format!(
                "  <event number=\"{}\" reel=\"{}\" track=\"{}\" edit_type=\"{}\" \
                 source_in=\"{}\" source_out=\"{}\" record_in=\"{}\" record_out=\"{}\"",
                ev.number,
                xml_escape(&ev.reel),
                xml_escape(&track_str),
                xml_escape(&edit_str),
                xml_escape(&ev.source_in.to_string()),
                xml_escape(&ev.source_out.to_string()),
                xml_escape(&ev.record_in.to_string()),
                xml_escape(&ev.record_out.to_string()),
            ));
            if let Some(ref clip) = ev.clip_name {
                xml.push_str(&format!(" clip_name=\"{}\"", xml_escape(clip)));
            }
            xml.push_str(" />\n");
        }

        xml.push_str("</edl>\n");
        Ok(xml)
    }

    /// Sort events in-place by record-in timecode.
    pub fn sort_events(&mut self) {
        self.inner.sort_events();
    }

    /// Renumber events sequentially starting from 1.
    pub fn renumber_events(&mut self) {
        self.inner.renumber_events();
    }

    fn __repr__(&self) -> String {
        format!(
            "PyEdl(title={:?}, events={})",
            self.inner.title,
            self.inner.event_count()
        )
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Minimal XML character escaping.
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

// ---------------------------------------------------------------------------
// Module registration helper
// ---------------------------------------------------------------------------

/// Register all EDL classes into the given Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEdlEvent>()?;
    m.add_class::<PyEdl>()?;
    Ok(())
}
