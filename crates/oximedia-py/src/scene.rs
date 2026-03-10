//! Python bindings for scene and shot detection from `oximedia-shots`.
//!
//! Converts raw RGB frame bytes into `FrameBuffer` for the underlying
//! shot detector, returning lightweight `PyShot` / `PyScene` results.

use pyo3::prelude::*;

use oximedia_shots::frame_buffer::FrameBuffer;
use oximedia_shots::{ShotDetector, ShotDetectorConfig};

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// A detected shot with timing and classification.
#[pyclass]
#[derive(Clone)]
pub struct PyShot {
    /// Shot index (sequential within a detection run).
    #[pyo3(get)]
    pub index: u64,
    /// Start frame number.
    #[pyo3(get)]
    pub start_frame: usize,
    /// End frame number.
    #[pyo3(get)]
    pub end_frame: usize,
    /// Human-readable shot type (e.g. "CloseUp", "LongShot").
    #[pyo3(get)]
    pub shot_type: String,
    /// Detection confidence (0.0 to 1.0).
    #[pyo3(get)]
    pub confidence: f32,
}

#[pymethods]
impl PyShot {
    fn __repr__(&self) -> String {
        format!(
            "PyShot(index={}, frames={}..{}, type='{}', confidence={:.2})",
            self.index, self.start_frame, self.end_frame, self.shot_type, self.confidence
        )
    }
}

/// A detected scene (group of related shots).
#[pyclass]
#[derive(Clone)]
pub struct PyScene {
    /// Scene index.
    #[pyo3(get)]
    pub index: usize,
    /// Start frame number.
    #[pyo3(get)]
    pub start_frame: usize,
    /// End frame number.
    #[pyo3(get)]
    pub end_frame: usize,
    /// Number of shots in this scene.
    #[pyo3(get)]
    pub shot_count: usize,
}

#[pymethods]
impl PyScene {
    fn __repr__(&self) -> String {
        format!(
            "PyScene(index={}, frames={}..{}, shots={})",
            self.index, self.start_frame, self.end_frame, self.shot_count
        )
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn bytes_to_frame(data: &[u8], width: usize, height: usize) -> PyResult<FrameBuffer> {
    let expected = height * width * 3;
    if data.len() < expected {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Frame data too small: need {expected} bytes ({}x{}x3), got {}",
            width,
            height,
            data.len()
        )));
    }
    FrameBuffer::from_vec(height, width, 3, data[..expected].to_vec()).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "Failed to create FrameBuffer from {}x{}x3 data",
            width, height
        ))
    })
}

fn frames_to_buffers(
    frames: &[Vec<u8>],
    width: usize,
    height: usize,
) -> PyResult<Vec<FrameBuffer>> {
    frames
        .iter()
        .map(|f| bytes_to_frame(f, width, height))
        .collect()
}

/// Convert a `Shot` timestamp (PTS-based) to a frame number approximation.
///
/// Shots use `Timestamp { pts, timebase, .. }`. Without knowing the actual
/// timebase numerator/denominator we fall back to treating pts as a frame
/// index (which is correct when the detector is fed frame-sequential data).
fn shot_start_frame(shot: &oximedia_shots::Shot) -> usize {
    shot.start.pts.max(0) as usize
}

fn shot_end_frame(shot: &oximedia_shots::Shot) -> usize {
    shot.end.pts.max(0) as usize
}

// ---------------------------------------------------------------------------
// Module functions
// ---------------------------------------------------------------------------

/// Detect shots from a list of video frames.
///
/// Each frame is provided as a flat bytes buffer of RGB pixels (H x W x 3).
///
/// Parameters
/// ----------
/// frames : list[bytes]
///     List of raw RGB frame data.
/// width : int
///     Frame width in pixels.
/// height : int
///     Frame height in pixels.
/// threshold : float, default 0.3
///     Cut detection threshold.
#[pyfunction]
#[pyo3(signature = (frames, width, height, threshold=0.3))]
pub fn detect_scenes(
    frames: Vec<Vec<u8>>,
    width: usize,
    height: usize,
    threshold: f32,
) -> PyResult<Vec<PyShot>> {
    let arrays = frames_to_buffers(&frames, width, height)?;
    let config = ShotDetectorConfig {
        cut_threshold: threshold,
        ..ShotDetectorConfig::default()
    };
    let detector = ShotDetector::new(config);
    let shots = detector
        .detect_shots(&arrays)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;

    Ok(shots
        .iter()
        .map(|s| PyShot {
            index: s.id,
            start_frame: shot_start_frame(s),
            end_frame: shot_end_frame(s),
            shot_type: format!("{:?}", s.shot_type),
            confidence: s.confidence,
        })
        .collect())
}

/// Classify shots by type (close-up, medium, long, etc.).
///
/// Runs full shot detection with classification enabled and returns the
/// classified shots.
#[pyfunction]
pub fn classify_shots(frames: Vec<Vec<u8>>, width: usize, height: usize) -> PyResult<Vec<PyShot>> {
    let arrays = frames_to_buffers(&frames, width, height)?;
    let config = ShotDetectorConfig {
        enable_classification: true,
        ..ShotDetectorConfig::default()
    };
    let detector = ShotDetector::new(config);
    let shots = detector
        .detect_shots(&arrays)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;

    Ok(shots
        .iter()
        .map(|s| PyShot {
            index: s.id,
            start_frame: shot_start_frame(s),
            end_frame: shot_end_frame(s),
            shot_type: format!("{:?}", s.shot_type),
            confidence: s.confidence,
        })
        .collect())
}
