//! Python bindings for `oximedia-captions` caption processing.
//!
//! Provides `PyCaptionTrack`, `PyCaptionEntry`, `PyCaptionConverter`,
//! and standalone functions for caption parsing, conversion, and validation.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;

use oximedia_captions::{
    export::Exporter, import::Importer, validation, Caption, CaptionFormat, CaptionTrack,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse_format(s: &str) -> PyResult<CaptionFormat> {
    match s.to_lowercase().as_str() {
        "srt" => Ok(CaptionFormat::Srt),
        "vtt" | "webvtt" => Ok(CaptionFormat::WebVtt),
        "ass" => Ok(CaptionFormat::Ass),
        "ssa" => Ok(CaptionFormat::Ssa),
        "ttml" => Ok(CaptionFormat::Ttml),
        "dfxp" => Ok(CaptionFormat::Dfxp),
        "scc" => Ok(CaptionFormat::Scc),
        "stl" | "ebu-stl" => Ok(CaptionFormat::EbuStl),
        "itt" => Ok(CaptionFormat::ITt),
        "cea608" | "cea-608" => Ok(CaptionFormat::Cea608),
        "cea708" | "cea-708" => Ok(CaptionFormat::Cea708),
        other => Err(PyValueError::new_err(format!(
            "Unknown caption format: {other}"
        ))),
    }
}

fn caption_to_py(cap: &Caption) -> PyCaptionEntry {
    let (sh, sm, ss, sms) = cap.start.as_hmsm();
    let (eh, em, es, ems) = cap.end.as_hmsm();
    PyCaptionEntry {
        id: cap.id.to_string(),
        start_ms: cap.start.as_millis(),
        end_ms: cap.end.as_millis(),
        text: cap.text.clone(),
        start_timecode: format!("{sh:02}:{sm:02}:{ss:02}.{sms:03}"),
        end_timecode: format!("{eh:02}:{em:02}:{es:02}.{ems:03}"),
        speaker: cap.speaker.clone(),
    }
}

// ---------------------------------------------------------------------------
// PyCaptionEntry
// ---------------------------------------------------------------------------

/// A single caption entry.
#[pyclass]
#[derive(Clone)]
pub struct PyCaptionEntry {
    /// Caption ID.
    #[pyo3(get)]
    pub id: String,
    /// Start time in milliseconds.
    #[pyo3(get)]
    pub start_ms: i64,
    /// End time in milliseconds.
    #[pyo3(get)]
    pub end_ms: i64,
    /// Caption text.
    #[pyo3(get)]
    pub text: String,
    /// Start timecode (HH:MM:SS.mmm).
    #[pyo3(get)]
    pub start_timecode: String,
    /// End timecode (HH:MM:SS.mmm).
    #[pyo3(get)]
    pub end_timecode: String,
    /// Speaker identification (optional).
    #[pyo3(get)]
    pub speaker: Option<String>,
}

#[pymethods]
impl PyCaptionEntry {
    /// Duration in milliseconds.
    fn duration_ms(&self) -> i64 {
        self.end_ms - self.start_ms
    }

    fn __repr__(&self) -> String {
        let text_preview = if self.text.len() > 40 {
            format!("{}...", &self.text[..40])
        } else {
            self.text.clone()
        };
        format!(
            "PyCaptionEntry(start={}, end={}, text='{}')",
            self.start_timecode, self.end_timecode, text_preview,
        )
    }

    /// Convert to a Python dict.
    fn to_dict(&self, py: Python<'_>) -> PyResult<HashMap<String, Py<PyAny>>> {
        let mut m = HashMap::new();
        m.insert(
            "id".to_string(),
            self.id.clone().into_pyobject(py)?.into_any().unbind(),
        );
        m.insert(
            "start_ms".to_string(),
            self.start_ms.into_pyobject(py)?.into_any().unbind(),
        );
        m.insert(
            "end_ms".to_string(),
            self.end_ms.into_pyobject(py)?.into_any().unbind(),
        );
        m.insert(
            "text".to_string(),
            self.text.clone().into_pyobject(py)?.into_any().unbind(),
        );
        m.insert(
            "start_timecode".to_string(),
            self.start_timecode
                .clone()
                .into_pyobject(py)?
                .into_any()
                .unbind(),
        );
        m.insert(
            "end_timecode".to_string(),
            self.end_timecode
                .clone()
                .into_pyobject(py)?
                .into_any()
                .unbind(),
        );
        Ok(m)
    }
}

// ---------------------------------------------------------------------------
// PyCaptionTrack
// ---------------------------------------------------------------------------

/// A track of captions with language metadata.
#[pyclass]
#[derive(Clone)]
pub struct PyCaptionTrack {
    /// Language code.
    #[pyo3(get)]
    pub language: String,
    /// Number of captions in the track.
    #[pyo3(get)]
    pub count: usize,
    entries: Vec<PyCaptionEntry>,
}

#[pymethods]
impl PyCaptionTrack {
    /// Get all caption entries.
    fn entries(&self) -> Vec<PyCaptionEntry> {
        self.entries.clone()
    }

    /// Get a caption by index.
    fn get(&self, index: usize) -> PyResult<PyCaptionEntry> {
        self.entries
            .get(index)
            .cloned()
            .ok_or_else(|| PyValueError::new_err(format!("Index {index} out of range")))
    }

    /// Search captions containing the given text.
    fn search(&self, query: &str) -> Vec<PyCaptionEntry> {
        let q = query.to_lowercase();
        self.entries
            .iter()
            .filter(|e| e.text.to_lowercase().contains(&q))
            .cloned()
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "PyCaptionTrack(language='{}', captions={})",
            self.language, self.count,
        )
    }

    fn __len__(&self) -> usize {
        self.count
    }
}

fn track_to_py(track: &CaptionTrack) -> PyCaptionTrack {
    let entries: Vec<PyCaptionEntry> = track.captions.iter().map(caption_to_py).collect();
    PyCaptionTrack {
        language: track.language.code.clone(),
        count: track.captions.len(),
        entries,
    }
}

// ---------------------------------------------------------------------------
// PyCaptionConverter
// ---------------------------------------------------------------------------

/// Caption format converter.
#[pyclass]
pub struct PyCaptionConverter;

#[pymethods]
impl PyCaptionConverter {
    /// Create a new converter.
    #[new]
    fn new() -> Self {
        Self
    }

    /// Convert caption data from one format to another.
    ///
    /// Args:
    ///     data: Caption file content as bytes.
    ///     from_format: Source format string (or "auto").
    ///     to_format: Target format string.
    ///
    /// Returns:
    ///     Converted caption content as bytes.
    #[pyo3(signature = (data, from_format="auto", to_format="srt"))]
    fn convert(&self, data: Vec<u8>, from_format: &str, to_format: &str) -> PyResult<Vec<u8>> {
        let track = if from_format == "auto" {
            Importer::import_auto(&data)
                .map_err(|e| PyRuntimeError::new_err(format!("Import failed: {e}")))?
        } else {
            let fmt = parse_format(from_format)?;
            Importer::import(&data, fmt)
                .map_err(|e| PyRuntimeError::new_err(format!("Import failed: {e}")))?
        };

        let target = parse_format(to_format)?;
        Exporter::export(&track, target)
            .map_err(|e| PyRuntimeError::new_err(format!("Export failed: {e}")))
    }

    /// List supported formats.
    fn supported_formats(&self) -> Vec<String> {
        vec![
            "srt".to_string(),
            "vtt".to_string(),
            "ass".to_string(),
            "ssa".to_string(),
            "ttml".to_string(),
            "dfxp".to_string(),
            "scc".to_string(),
            "ebu-stl".to_string(),
            "itt".to_string(),
        ]
    }

    fn __repr__(&self) -> String {
        "PyCaptionConverter()".to_string()
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Parse caption data and return a caption track.
///
/// Args:
///     data: Caption file content as bytes.
///     format: Caption format string (or "auto" for auto-detection).
///
/// Returns:
///     PyCaptionTrack with parsed captions.
#[pyfunction]
#[pyo3(signature = (data, format="auto"))]
pub fn parse_captions(data: Vec<u8>, format: &str) -> PyResult<PyCaptionTrack> {
    let track = if format == "auto" {
        Importer::import_auto(&data)
            .map_err(|e| PyRuntimeError::new_err(format!("Parse failed: {e}")))?
    } else {
        let fmt = parse_format(format)?;
        Importer::import(&data, fmt)
            .map_err(|e| PyRuntimeError::new_err(format!("Parse failed: {e}")))?
    };
    Ok(track_to_py(&track))
}

/// Convert caption data between formats.
///
/// Args:
///     data: Caption file content as bytes.
///     from_format: Source format string (or "auto").
///     to_format: Target format string.
///
/// Returns:
///     Converted caption content as bytes.
#[pyfunction]
#[pyo3(signature = (data, from_format="auto", to_format="srt"))]
pub fn convert_captions(data: Vec<u8>, from_format: &str, to_format: &str) -> PyResult<Vec<u8>> {
    let track = if from_format == "auto" {
        Importer::import_auto(&data)
            .map_err(|e| PyRuntimeError::new_err(format!("Import failed: {e}")))?
    } else {
        let fmt = parse_format(from_format)?;
        Importer::import(&data, fmt)
            .map_err(|e| PyRuntimeError::new_err(format!("Import failed: {e}")))?
    };

    let target = parse_format(to_format)?;
    Exporter::export(&track, target)
        .map_err(|e| PyRuntimeError::new_err(format!("Export failed: {e}")))
}

/// Validate captions against a standard.
///
/// Args:
///     data: Caption file content as bytes.
///     standard: Standard name (fcc, wcag, cea608, cea708).
///
/// Returns:
///     Dict with validation results.
#[pyfunction]
#[pyo3(signature = (data, standard="fcc"))]
pub fn validate_captions(
    data: Vec<u8>,
    standard: &str,
    py: Python<'_>,
) -> PyResult<HashMap<String, Py<PyAny>>> {
    let track = Importer::import_auto(&data)
        .map_err(|e| PyRuntimeError::new_err(format!("Parse failed: {e}")))?;

    let validator = validation::Validator::new();
    let report = validator
        .validate(&track)
        .map_err(|e| PyRuntimeError::new_err(format!("Validation failed: {e}")))?;

    let mut result = HashMap::new();
    result.insert(
        "passed".to_string(),
        report
            .passed()
            .into_pyobject(py)?
            .to_owned()
            .into_any()
            .unbind(),
    );
    result.insert(
        "standard".to_string(),
        standard.into_pyobject(py)?.into_any().unbind(),
    );
    result.insert(
        "total_captions".to_string(),
        report
            .statistics
            .total_captions
            .into_pyobject(py)?
            .into_any()
            .unbind(),
    );
    result.insert(
        "error_count".to_string(),
        report
            .statistics
            .error_count
            .into_pyobject(py)?
            .into_any()
            .unbind(),
    );
    result.insert(
        "warning_count".to_string(),
        report
            .statistics
            .warning_count
            .into_pyobject(py)?
            .into_any()
            .unbind(),
    );
    result.insert(
        "avg_reading_speed".to_string(),
        report
            .statistics
            .avg_reading_speed
            .into_pyobject(py)?
            .into_any()
            .unbind(),
    );

    let issues: Vec<String> = report
        .issues
        .iter()
        .map(|i| format!("[{:?}] {} ({})", i.severity, i.message, i.rule))
        .collect();
    result.insert(
        "issues".to_string(),
        issues.into_pyobject(py)?.into_any().unbind(),
    );

    Ok(result)
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register all captions bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCaptionEntry>()?;
    m.add_class::<PyCaptionTrack>()?;
    m.add_class::<PyCaptionConverter>()?;
    m.add_function(wrap_pyfunction!(parse_captions, m)?)?;
    m.add_function(wrap_pyfunction!(convert_captions, m)?)?;
    m.add_function(wrap_pyfunction!(validate_captions, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_format_srt() {
        let fmt = parse_format("srt");
        assert!(fmt.is_ok());
    }

    #[test]
    fn test_parse_format_unknown() {
        let fmt = parse_format("xyz");
        assert!(fmt.is_err());
    }

    #[test]
    fn test_caption_entry_duration() {
        let entry = PyCaptionEntry {
            id: "1".to_string(),
            start_ms: 1000,
            end_ms: 3000,
            text: "Hello world".to_string(),
            start_timecode: "00:00:01.000".to_string(),
            end_timecode: "00:00:03.000".to_string(),
            speaker: None,
        };
        assert_eq!(entry.duration_ms(), 2000);
    }

    #[test]
    fn test_caption_entry_repr() {
        let entry = PyCaptionEntry {
            id: "1".to_string(),
            start_ms: 0,
            end_ms: 1000,
            text: "Short text".to_string(),
            start_timecode: "00:00:00.000".to_string(),
            end_timecode: "00:00:01.000".to_string(),
            speaker: None,
        };
        let repr = entry.__repr__();
        assert!(repr.contains("Short text"));
    }

    #[test]
    fn test_caption_track_search() {
        let entries = vec![
            PyCaptionEntry {
                id: "1".to_string(),
                start_ms: 0,
                end_ms: 1000,
                text: "Hello world".to_string(),
                start_timecode: "00:00:00.000".to_string(),
                end_timecode: "00:00:01.000".to_string(),
                speaker: None,
            },
            PyCaptionEntry {
                id: "2".to_string(),
                start_ms: 1000,
                end_ms: 2000,
                text: "Goodbye moon".to_string(),
                start_timecode: "00:00:01.000".to_string(),
                end_timecode: "00:00:02.000".to_string(),
                speaker: None,
            },
        ];
        let track = PyCaptionTrack {
            language: "en".to_string(),
            count: 2,
            entries,
        };
        let results = track.search("hello");
        assert_eq!(results.len(), 1);
        assert!(results[0].text.contains("Hello"));
    }
}
