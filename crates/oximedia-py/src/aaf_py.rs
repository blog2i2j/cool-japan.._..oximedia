//! Python bindings for `oximedia-aaf` AAF file handling.
//!
//! Provides `PyAafReader`, `PyAafWriter`, `PyAafTrack`,
//! and standalone convenience functions for AAF operations from Python.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// PyAafTrack
// ---------------------------------------------------------------------------

/// Represents an AAF track with name and type info.
#[pyclass]
#[derive(Clone)]
pub struct PyAafTrack {
    /// Track index.
    #[pyo3(get)]
    pub index: usize,
    /// Track name.
    #[pyo3(get)]
    pub name: String,
    /// Track type (e.g., "Video", "Audio", "Timecode").
    #[pyo3(get)]
    pub track_type: String,
}

#[pymethods]
impl PyAafTrack {
    fn __repr__(&self) -> String {
        format!(
            "PyAafTrack(index={}, name='{}', type='{}')",
            self.index, self.name, self.track_type,
        )
    }

    /// Convert to dict.
    fn to_dict(&self, py: Python<'_>) -> PyResult<HashMap<String, Py<PyAny>>> {
        let mut m = HashMap::new();
        m.insert(
            "index".to_string(),
            self.index.into_pyobject(py)?.into_any().unbind(),
        );
        m.insert(
            "name".to_string(),
            self.name.clone().into_pyobject(py)?.into_any().unbind(),
        );
        m.insert(
            "track_type".to_string(),
            self.track_type
                .clone()
                .into_pyobject(py)?
                .into_any()
                .unbind(),
        );
        Ok(m)
    }
}

// ---------------------------------------------------------------------------
// PyAafReader
// ---------------------------------------------------------------------------

/// AAF file reader.
#[pyclass]
pub struct PyAafReader {
    aaf: oximedia_aaf::AafFile,
    path: String,
}

#[pymethods]
impl PyAafReader {
    /// Open and read an AAF file.
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let mut reader = oximedia_aaf::AafReader::open(path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to open AAF: {e}")))?;
        let aaf = reader
            .read()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to read AAF: {e}")))?;
        Ok(Self {
            aaf,
            path: path.to_string(),
        })
    }

    /// Get file info as a dict.
    fn info(&self, py: Python<'_>) -> PyResult<HashMap<String, Py<PyAny>>> {
        let mut m = HashMap::new();
        m.insert(
            "file".to_string(),
            self.path.clone().into_pyobject(py)?.into_any().unbind(),
        );
        m.insert(
            "composition_mobs".to_string(),
            self.aaf
                .composition_mobs()
                .len()
                .into_pyobject(py)?
                .into_any()
                .unbind(),
        );
        m.insert(
            "master_mobs".to_string(),
            self.aaf
                .master_mobs()
                .len()
                .into_pyobject(py)?
                .into_any()
                .unbind(),
        );
        m.insert(
            "source_mobs".to_string(),
            self.aaf
                .source_mobs()
                .len()
                .into_pyobject(py)?
                .into_any()
                .unbind(),
        );

        if let Some(rate) = self.aaf.edit_rate() {
            let rate_str = format!("{}/{}", rate.numerator, rate.denominator);
            m.insert(
                "edit_rate".to_string(),
                rate_str.into_pyobject(py)?.into_any().unbind(),
            );
        }

        if let Some(dur) = self.aaf.duration() {
            m.insert(
                "duration".to_string(),
                dur.into_pyobject(py)?.into_any().unbind(),
            );
        }

        Ok(m)
    }

    /// Extract embedded media data.
    fn extract_media(&self) -> Vec<(String, Vec<u8>)> {
        self.aaf
            .essence_data()
            .iter()
            .map(|e| (e.mob_id().to_string(), e.data().to_vec()))
            .collect()
    }

    /// Get the number of tracks across all compositions.
    fn track_count(&self) -> usize {
        self.aaf
            .composition_mobs()
            .iter()
            .map(|c| c.tracks().len())
            .sum()
    }

    /// Get all tracks.
    fn tracks(&self) -> Vec<PyAafTrack> {
        let mut result = Vec::new();
        let mut idx = 0usize;
        for comp in self.aaf.composition_mobs() {
            for track in comp.tracks() {
                result.push(PyAafTrack {
                    index: idx,
                    name: track.name.clone(),
                    track_type: format!("{:?}", track.track_type),
                });
                idx += 1;
            }
        }
        result
    }

    /// Get file metadata.
    fn metadata(&self, py: Python<'_>) -> PyResult<HashMap<String, Py<PyAny>>> {
        let mut m = HashMap::new();
        let header = self.aaf.header();
        m.insert(
            "version_major".to_string(),
            header.major_version.into_pyobject(py)?.into_any().unbind(),
        );
        m.insert(
            "version_minor".to_string(),
            header.minor_version.into_pyobject(py)?.into_any().unbind(),
        );
        m.insert(
            "version".to_string(),
            header
                .version_string()
                .into_pyobject(py)?
                .into_any()
                .unbind(),
        );
        Ok(m)
    }

    fn __repr__(&self) -> String {
        format!(
            "PyAafReader(path='{}', compositions={})",
            self.path,
            self.aaf.composition_mobs().len(),
        )
    }
}

// ---------------------------------------------------------------------------
// PyAafWriter
// ---------------------------------------------------------------------------

/// AAF file writer.
#[pyclass]
pub struct PyAafWriter {
    _placeholder: bool,
}

#[pymethods]
impl PyAafWriter {
    /// Create a new AAF writer.
    #[new]
    fn new() -> Self {
        Self { _placeholder: true }
    }

    /// Write a new empty AAF file to disk.
    fn write(&self, path: &str) -> PyResult<()> {
        let mut writer = oximedia_aaf::AafWriter::create(path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create AAF: {e}")))?;
        writer
            .write()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to write AAF: {e}")))?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        "PyAafWriter()".to_string()
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Read an AAF file and return a reader object.
#[pyfunction]
pub fn read_aaf(path: &str) -> PyResult<PyAafReader> {
    PyAafReader::new(path)
}

/// Convert an AAF file to EDL format.
#[pyfunction]
pub fn convert_aaf_to_edl(aaf_path: &str) -> PyResult<String> {
    let mut reader = oximedia_aaf::AafReader::open(aaf_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to open AAF: {e}")))?;
    let aaf = reader
        .read()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to read AAF: {e}")))?;

    let edit_rate = aaf.edit_rate().unwrap_or(oximedia_aaf::EditRate {
        numerator: 24,
        denominator: 1,
    });

    let exporter = oximedia_aaf::EdlExporter::new("OxiMedia Export", edit_rate);
    let edl = exporter
        .export(&aaf)
        .map_err(|e| PyRuntimeError::new_err(format!("EDL export failed: {e}")))?;

    Ok(edl)
}

/// Validate an AAF file structure.
#[pyfunction]
pub fn validate_aaf(path: &str) -> PyResult<(bool, Vec<String>)> {
    let mut reader = oximedia_aaf::AafReader::open(path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to open AAF: {e}")))?;
    let aaf = reader
        .read()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to read AAF: {e}")))?;

    let mut issues = Vec::new();

    if aaf.composition_mobs().is_empty() {
        issues.push("No composition mobs found".to_string());
    }

    if aaf.edit_rate().is_none() {
        issues.push("No edit rate defined".to_string());
    }

    if aaf.duration().is_none() {
        issues.push("No duration available".to_string());
    }

    for comp in aaf.composition_mobs() {
        if comp.tracks().is_empty() {
            issues.push(format!("Composition '{}' has no tracks", comp.name()));
        }
    }

    let has_errors = issues
        .iter()
        .any(|i| !i.starts_with("No duration") && !i.starts_with("No edit rate"));

    Ok((!has_errors, issues))
}

// ---------------------------------------------------------------------------
// Registration helper
// ---------------------------------------------------------------------------

/// Register all AAF bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAafReader>()?;
    m.add_class::<PyAafWriter>()?;
    m.add_class::<PyAafTrack>()?;
    m.add_function(wrap_pyfunction!(read_aaf, m)?)?;
    m.add_function(wrap_pyfunction!(convert_aaf_to_edl, m)?)?;
    m.add_function(wrap_pyfunction!(validate_aaf, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aaf_track_repr() {
        let track = PyAafTrack {
            index: 0,
            name: "Video 1".to_string(),
            track_type: "Video".to_string(),
        };
        let repr = track.__repr__();
        assert!(repr.contains("Video 1"));
    }

    #[test]
    fn test_aaf_writer_creation() {
        let writer = PyAafWriter::new();
        let repr = writer.__repr__();
        assert!(repr.contains("PyAafWriter"));
    }

    #[test]
    fn test_read_nonexistent_aaf() {
        let result = PyAafReader::new("/tmp/nonexistent_test_aaf_12345.aaf");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_nonexistent_aaf() {
        let result = validate_aaf("/tmp/nonexistent_test_aaf_12345.aaf");
        assert!(result.is_err());
    }

    #[test]
    fn test_convert_nonexistent_aaf() {
        let result = convert_aaf_to_edl("/tmp/nonexistent_test_aaf_12345.aaf");
        assert!(result.is_err());
    }
}
