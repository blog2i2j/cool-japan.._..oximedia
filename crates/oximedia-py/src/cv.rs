//! Python bindings for computer vision operations via `oximedia-cv`.
//!
//! Exposes face detection, motion detection, histogram computation and
//! edge detection as free functions on the Python `oximedia` module.

use pyo3::prelude::*;

use oximedia_cv::detect::{FaceDetector, HaarCascade, MotionDetector};
use oximedia_cv::image::edge::{CannyEdge, EdgeDetector};
use oximedia_cv::image::histogram::Histogram;

// ---------------------------------------------------------------------------
// Python-visible result types
// ---------------------------------------------------------------------------

/// A detected face bounding box returned by `detect_faces`.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyFaceRegion {
    /// X coordinate of top-left corner (pixels).
    #[pyo3(get)]
    pub x: u32,
    /// Y coordinate of top-left corner (pixels).
    #[pyo3(get)]
    pub y: u32,
    /// Width of the face bounding box.
    #[pyo3(get)]
    pub width: u32,
    /// Height of the face bounding box.
    #[pyo3(get)]
    pub height: u32,
    /// Detection confidence in [0.0, 1.0].
    #[pyo3(get)]
    pub confidence: f64,
}

#[pymethods]
impl PyFaceRegion {
    fn __repr__(&self) -> String {
        format!(
            "PyFaceRegion(x={}, y={}, w={}, h={}, confidence={:.3})",
            self.x, self.y, self.width, self.height, self.confidence
        )
    }
}

/// A detected motion region returned by `detect_motion`.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyMotionRegion {
    /// X coordinate of top-left corner.
    #[pyo3(get)]
    pub x: u32,
    /// Y coordinate of top-left corner.
    #[pyo3(get)]
    pub y: u32,
    /// Width of the motion bounding box.
    #[pyo3(get)]
    pub width: u32,
    /// Height of the motion bounding box.
    #[pyo3(get)]
    pub height: u32,
    /// Motion intensity in [0.0, 1.0].
    #[pyo3(get)]
    pub intensity: f64,
    /// Centroid X of the motion region.
    #[pyo3(get)]
    pub centroid_x: f64,
    /// Centroid Y of the motion region.
    #[pyo3(get)]
    pub centroid_y: f64,
}

#[pymethods]
impl PyMotionRegion {
    fn __repr__(&self) -> String {
        format!(
            "PyMotionRegion(x={}, y={}, w={}, h={}, intensity={:.3})",
            self.x, self.y, self.width, self.height, self.intensity
        )
    }
}

/// Per-channel RGB histogram returned by `compute_histogram`.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyHistogram {
    /// Red channel frequency bins (256 entries, index = intensity 0-255).
    #[pyo3(get)]
    pub red: Vec<u32>,
    /// Green channel frequency bins.
    #[pyo3(get)]
    pub green: Vec<u32>,
    /// Blue channel frequency bins.
    #[pyo3(get)]
    pub blue: Vec<u32>,
    /// Total pixel count.
    #[pyo3(get)]
    pub total_pixels: u64,
}

#[pymethods]
impl PyHistogram {
    fn __repr__(&self) -> String {
        format!("PyHistogram(total_pixels={})", self.total_pixels)
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Detect faces in an RGB image using a Haar cascade classifier.
///
/// Parameters
/// ----------
/// rgb_bytes : bytes
///     Raw RGB pixel data (width * height * 3 bytes, row-major).
/// width : int
///     Image width in pixels.
/// height : int
///     Image height in pixels.
///
/// Returns
/// -------
/// list[PyFaceRegion]
///     Detected face bounding boxes, sorted by confidence descending.
#[pyfunction]
#[pyo3(signature = (rgb_bytes, width, height))]
pub fn detect_faces(rgb_bytes: &[u8], width: u32, height: u32) -> PyResult<Vec<PyFaceRegion>> {
    let expected = (width as usize) * (height as usize) * 3;
    if rgb_bytes.len() < expected {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "rgb_bytes too small: expected >= {expected} bytes ({}x{}x3), got {}",
            width,
            height,
            rgb_bytes.len()
        )));
    }

    // Convert RGB to grayscale (BT.601 luminance coefficients)
    let gray: Vec<u8> = rgb_bytes
        .chunks_exact(3)
        .map(|px| {
            let r = px[0] as u32;
            let g = px[1] as u32;
            let b = px[2] as u32;
            ((r * 299 + g * 587 + b * 114) / 1000) as u8
        })
        .collect();

    // Use a standard 24x24 detection window (frontal face cascade size)
    let cascade = HaarCascade::new(24, 24);
    let faces = cascade
        .detect(&gray, width, height)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let mut results: Vec<PyFaceRegion> = faces
        .into_iter()
        .map(|r| PyFaceRegion {
            x: r.bbox.x,
            y: r.bbox.y,
            width: r.bbox.width,
            height: r.bbox.height,
            confidence: r.bbox.confidence,
        })
        .collect();

    // Sort by confidence descending
    results.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(results)
}

/// Detect motion regions between two consecutive grayscale frames.
///
/// Parameters
/// ----------
/// frame1_bytes : bytes
///     First frame (grayscale, width * height bytes).
/// frame2_bytes : bytes
///     Second frame (grayscale, width * height bytes).
/// width : int
///     Frame width.
/// height : int
///     Frame height.
///
/// Returns
/// -------
/// list[PyMotionRegion]
///     Detected motion regions, sorted by intensity descending.
#[pyfunction]
#[pyo3(signature = (frame1_bytes, frame2_bytes, width, height))]
pub fn detect_motion(
    frame1_bytes: &[u8],
    frame2_bytes: &[u8],
    width: u32,
    height: u32,
) -> PyResult<Vec<PyMotionRegion>> {
    let expected = (width as usize) * (height as usize);
    if frame1_bytes.len() < expected {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "frame1_bytes too small: need {expected}, got {}",
            frame1_bytes.len()
        )));
    }
    if frame2_bytes.len() < expected {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "frame2_bytes too small: need {expected}, got {}",
            frame2_bytes.len()
        )));
    }

    let mut detector = MotionDetector::new(width, height);

    // Feed first frame — establishes background model
    detector
        .process(&frame1_bytes[..expected])
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    // Feed second frame — produces motion regions
    let (_mask, regions) = detector
        .process(&frame2_bytes[..expected])
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let mut results: Vec<PyMotionRegion> = regions
        .into_iter()
        .map(|r| PyMotionRegion {
            x: r.x,
            y: r.y,
            width: r.width,
            height: r.height,
            intensity: r.intensity,
            centroid_x: r.centroid.0,
            centroid_y: r.centroid.1,
        })
        .collect();

    results.sort_by(|a, b| {
        b.intensity
            .partial_cmp(&a.intensity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(results)
}

/// Compute a per-channel RGB histogram for an image.
///
/// Parameters
/// ----------
/// rgb_bytes : bytes
///     Raw RGB pixel data (width * height * 3 bytes).
/// width : int
///     Image width.
/// height : int
///     Image height.
///
/// Returns
/// -------
/// PyHistogram
///     Object with `.red`, `.green`, `.blue` lists of 256 frequency values each.
#[pyfunction]
#[pyo3(signature = (rgb_bytes, width, height))]
pub fn compute_histogram(rgb_bytes: &[u8], width: u32, height: u32) -> PyResult<PyHistogram> {
    let expected = (width as usize) * (height as usize) * 3;
    if rgb_bytes.len() < expected {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "rgb_bytes too small: need {expected}, got {}",
            rgb_bytes.len()
        )));
    }

    let r_chan: Vec<u8> = rgb_bytes.chunks_exact(3).map(|p| p[0]).collect();
    let g_chan: Vec<u8> = rgb_bytes.chunks_exact(3).map(|p| p[1]).collect();
    let b_chan: Vec<u8> = rgb_bytes.chunks_exact(3).map(|p| p[2]).collect();

    let hist_r = Histogram::compute(&r_chan);
    let hist_g = Histogram::compute(&g_chan);
    let hist_b = Histogram::compute(&b_chan);

    let total_pixels = (width as u64) * (height as u64);

    Ok(PyHistogram {
        red: hist_r.bins().to_vec(),
        green: hist_g.bins().to_vec(),
        blue: hist_b.bins().to_vec(),
        total_pixels,
    })
}

/// Run Canny edge detection on a grayscale image.
///
/// Parameters
/// ----------
/// gray_bytes : bytes
///     Grayscale pixel data (width * height bytes, row-major).
/// width : int
///     Image width in pixels.
/// height : int
///     Image height in pixels.
/// threshold : float, optional
///     Low threshold for Canny hysteresis (0 – 255).  High threshold is
///     automatically set to ``threshold * 3``.  Default: 50.0.
///
/// Returns
/// -------
/// bytes
///     Edge magnitude image with the same dimensions (grayscale, 1 byte per
///     pixel), where non-zero pixels are detected edges.
#[pyfunction]
#[pyo3(signature = (gray_bytes, width, height, threshold = 50.0))]
pub fn detect_edges(
    gray_bytes: &[u8],
    width: u32,
    height: u32,
    threshold: f64,
) -> PyResult<Vec<u8>> {
    let expected = (width as usize) * (height as usize);
    if gray_bytes.len() < expected {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "gray_bytes too small: need {expected}, got {}",
            gray_bytes.len()
        )));
    }

    let low = threshold.clamp(0.0, 255.0);
    let high = (threshold * 3.0).clamp(0.0, 255.0);

    let canny = CannyEdge::new(low, high, 1.4);
    let edges = canny
        .detect(&gray_bytes[..expected], width, height)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok(edges)
}

// ---------------------------------------------------------------------------
// Module registration helper (called from lib.rs)
// ---------------------------------------------------------------------------

/// Register all CV classes and free functions into the given Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFaceRegion>()?;
    m.add_class::<PyMotionRegion>()?;
    m.add_class::<PyHistogram>()?;
    m.add_function(wrap_pyfunction!(detect_faces, m)?)?;
    m.add_function(wrap_pyfunction!(detect_motion, m)?)?;
    m.add_function(wrap_pyfunction!(compute_histogram, m)?)?;
    m.add_function(wrap_pyfunction!(detect_edges, m)?)?;
    Ok(())
}
