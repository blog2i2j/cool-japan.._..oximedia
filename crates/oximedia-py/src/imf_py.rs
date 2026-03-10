//! Python bindings for `oximedia-imf` IMF package handling.
//!
//! Provides `PyImfPackage`, `PyImfComposition`, `PyImfTrack`,
//! and standalone convenience functions for IMF operations from Python.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// PyImfTrack
// ---------------------------------------------------------------------------

/// Represents an IMF track (sequence) with type and resource info.
#[pyclass]
#[derive(Clone)]
pub struct PyImfTrack {
    /// Track index.
    #[pyo3(get)]
    pub index: usize,
    /// Track type (e.g., "MainImage", "MainAudio").
    #[pyo3(get)]
    pub track_type: String,
    /// Number of resources in this track.
    #[pyo3(get)]
    pub resource_count: usize,
}

#[pymethods]
impl PyImfTrack {
    fn __repr__(&self) -> String {
        format!(
            "PyImfTrack(index={}, type='{}', resources={})",
            self.index, self.track_type, self.resource_count,
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
            "track_type".to_string(),
            self.track_type
                .clone()
                .into_pyobject(py)?
                .into_any()
                .unbind(),
        );
        m.insert(
            "resource_count".to_string(),
            self.resource_count.into_pyobject(py)?.into_any().unbind(),
        );
        Ok(m)
    }
}

// ---------------------------------------------------------------------------
// PyImfComposition
// ---------------------------------------------------------------------------

/// Represents an IMF composition (CPL).
#[pyclass]
#[derive(Clone)]
pub struct PyImfComposition {
    /// Composition title.
    #[pyo3(get)]
    pub title: String,
    /// Total duration in frames.
    #[pyo3(get)]
    pub duration_frames: u64,
    /// Edit rate as "N/D" string.
    #[pyo3(get)]
    pub edit_rate: String,
    tracks: Vec<PyImfTrack>,
}

#[pymethods]
impl PyImfComposition {
    /// Get all tracks.
    fn tracks(&self) -> Vec<PyImfTrack> {
        self.tracks.clone()
    }

    /// Get the number of tracks.
    fn track_count(&self) -> usize {
        self.tracks.len()
    }

    /// Get duration in seconds (approximate).
    fn duration_seconds(&self) -> f64 {
        let parts: Vec<&str> = self.edit_rate.split('/').collect();
        if parts.len() == 2 {
            let num: f64 = parts[0].parse().unwrap_or(24.0);
            let den: f64 = parts[1].parse().unwrap_or(1.0);
            if num > 0.0 {
                return self.duration_frames as f64 * den / num;
            }
        }
        0.0
    }

    fn __repr__(&self) -> String {
        format!(
            "PyImfComposition(title='{}', duration={} frames, rate={}, tracks={})",
            self.title,
            self.duration_frames,
            self.edit_rate,
            self.tracks.len(),
        )
    }
}

// ---------------------------------------------------------------------------
// PyImfPackage
// ---------------------------------------------------------------------------

/// IMF package reader and validator.
#[pyclass]
pub struct PyImfPackage {
    path: String,
}

#[pymethods]
impl PyImfPackage {
    /// Open an IMF package from a directory path.
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        if !std::path::Path::new(path).exists() {
            return Err(PyValueError::new_err(format!(
                "IMF package directory not found: {path}"
            )));
        }
        Ok(Self {
            path: path.to_string(),
        })
    }

    /// Validate the IMF package.
    ///
    /// Returns:
    ///     Tuple of (is_valid, error_messages, warning_messages).
    fn validate(&self) -> PyResult<(bool, Vec<String>, Vec<String>)> {
        let package = oximedia_imf::ImfPackage::open(&self.path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to open IMF package: {e}")))?;

        let validator = oximedia_imf::Validator::new()
            .with_conformance_level(oximedia_imf::ConformanceLevel::ImfCore);

        let report = validator
            .validate(&package)
            .map_err(|e| PyRuntimeError::new_err(format!("IMF validation failed: {e}")))?;

        // Collect all issues; split by severity using format string heuristic
        let all_issues = report.errors();
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        for issue in all_issues {
            let sev_str = format!("{}", issue.severity());
            if sev_str.contains("Error") {
                errors.push(issue.message().to_string());
            } else {
                warnings.push(issue.message().to_string());
            }
        }
        Ok((report.is_valid(), errors, warnings))
    }

    /// Get package information as a composition object.
    fn info(&self) -> PyResult<PyImfComposition> {
        let package = oximedia_imf::ImfPackage::open(&self.path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to open IMF package: {e}")))?;

        let cpl = package
            .primary_cpl()
            .ok_or_else(|| PyRuntimeError::new_err("No composition playlists found"))?;

        let tracks: Vec<PyImfTrack> = cpl
            .sequences()
            .iter()
            .enumerate()
            .map(|(i, s)| PyImfTrack {
                index: i,
                track_type: format!("{:?}", s.sequence_type()),
                resource_count: s.resources().len(),
            })
            .collect();

        Ok(PyImfComposition {
            title: cpl.content_title().to_string(),
            duration_frames: cpl.total_duration(),
            edit_rate: format!("{}", cpl.edit_rate()),
            tracks,
        })
    }

    /// Extract a track by index.
    fn extract_track(&self, track_index: usize) -> PyResult<PyImfTrack> {
        let package = oximedia_imf::ImfPackage::open(&self.path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to open IMF package: {e}")))?;

        let cpl = package
            .primary_cpl()
            .ok_or_else(|| PyRuntimeError::new_err("No composition playlists found"))?;

        let sequences = cpl.sequences();
        if track_index >= sequences.len() {
            return Err(PyValueError::new_err(format!(
                "Track index {track_index} out of range (0..{})",
                sequences.len()
            )));
        }

        let seq = sequences[track_index];
        Ok(PyImfTrack {
            index: track_index,
            track_type: format!("{:?}", seq.sequence_type()),
            resource_count: seq.resources().len(),
        })
    }

    /// Get number of tracks.
    fn track_count(&self) -> PyResult<usize> {
        let package = oximedia_imf::ImfPackage::open(&self.path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to open IMF package: {e}")))?;

        match package.primary_cpl() {
            Some(cpl) => Ok(cpl.sequences().len()),
            None => Ok(0),
        }
    }

    /// Get duration in frames.
    fn duration(&self) -> PyResult<u64> {
        let package = oximedia_imf::ImfPackage::open(&self.path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to open IMF package: {e}")))?;

        match package.primary_cpl() {
            Some(cpl) => Ok(cpl.total_duration()),
            None => Ok(0),
        }
    }

    fn __repr__(&self) -> String {
        format!("PyImfPackage(path='{}')", self.path)
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Validate an IMF package directory.
#[pyfunction]
pub fn validate_imf(path: &str) -> PyResult<(bool, Vec<String>, Vec<String>)> {
    let pkg = PyImfPackage::new(path)?;
    pkg.validate()
}

/// Create an IMF package.
#[pyfunction]
#[pyo3(signature = (output, title="Untitled", edit_rate="24/1"))]
pub fn create_imf_package(output: &str, title: &str, edit_rate: &str) -> PyResult<String> {
    let parts: Vec<&str> = edit_rate.split('/').collect();
    if parts.len() != 2 {
        return Err(PyValueError::new_err(format!(
            "Invalid edit rate: {edit_rate}. Expected N/D."
        )));
    }
    let num: u32 = parts[0]
        .trim()
        .parse()
        .map_err(|_| PyValueError::new_err("Invalid edit rate numerator"))?;
    let den: u32 = parts[1]
        .trim()
        .parse()
        .map_err(|_| PyValueError::new_err("Invalid edit rate denominator"))?;

    let rate = oximedia_imf::EditRate::new(num, den);
    let builder = oximedia_imf::ImfPackageBuilder::new(output)
        .with_title(title.to_string())
        .with_creator("OxiMedia Python".to_string())
        .with_edit_rate(rate);

    let _package = builder
        .build()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create IMF package: {e}")))?;

    Ok(output.to_string())
}

/// Get IMF package info.
#[pyfunction]
pub fn imf_info(path: &str) -> PyResult<PyImfComposition> {
    let pkg = PyImfPackage::new(path)?;
    pkg.info()
}

// ---------------------------------------------------------------------------
// Registration helper
// ---------------------------------------------------------------------------

/// Register all IMF bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyImfPackage>()?;
    m.add_class::<PyImfComposition>()?;
    m.add_class::<PyImfTrack>()?;
    m.add_function(wrap_pyfunction!(validate_imf, m)?)?;
    m.add_function(wrap_pyfunction!(create_imf_package, m)?)?;
    m.add_function(wrap_pyfunction!(imf_info, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_imf_track_repr() {
        let track = PyImfTrack {
            index: 0,
            track_type: "MainImage".to_string(),
            resource_count: 3,
        };
        let repr = track.__repr__();
        assert!(repr.contains("MainImage"));
        assert!(repr.contains("resources=3"));
    }

    #[test]
    fn test_imf_composition_duration() {
        let comp = PyImfComposition {
            title: "Test".to_string(),
            duration_frames: 240,
            edit_rate: "24/1".to_string(),
            tracks: Vec::new(),
        };
        let secs = comp.duration_seconds();
        assert!((secs - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_imf_composition_track_count() {
        let comp = PyImfComposition {
            title: "Test".to_string(),
            duration_frames: 100,
            edit_rate: "25/1".to_string(),
            tracks: vec![
                PyImfTrack {
                    index: 0,
                    track_type: "MainImage".to_string(),
                    resource_count: 1,
                },
                PyImfTrack {
                    index: 1,
                    track_type: "MainAudio".to_string(),
                    resource_count: 2,
                },
            ],
        };
        assert_eq!(comp.track_count(), 2);
    }

    #[test]
    fn test_package_nonexistent_path() {
        let result = PyImfPackage::new("/tmp/nonexistent_imf_test_dir_12345");
        assert!(result.is_err());
    }
}
