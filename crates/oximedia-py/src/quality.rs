//! Python bindings for quality metrics from `oximedia-quality`.
//!
//! Provides full-reference metrics (PSNR, SSIM) and no-reference metrics
//! (BRISQUE, NIQE, Blockiness, Blur, Noise) via `PyQualityAssessor` and
//! standalone functions.

use pyo3::prelude::*;
use std::collections::HashMap;

use oximedia_core::PixelFormat;
use oximedia_quality::{Frame, MetricType, QualityAssessor};

/// Quality score result accessible from Python.
#[pyclass]
#[derive(Clone)]
pub struct PyQualityScore {
    /// Name of the metric (e.g. "PSNR", "SSIM").
    #[pyo3(get)]
    pub metric: String,
    /// Overall score value.
    #[pyo3(get)]
    pub score: f64,
    /// Per-component breakdown (e.g. Y / Cb / Cr channels).
    #[pyo3(get)]
    pub components: HashMap<String, f64>,
}

#[pymethods]
impl PyQualityScore {
    fn __repr__(&self) -> String {
        format!(
            "PyQualityScore(metric='{}', score={:.4})",
            self.metric, self.score
        )
    }
}

/// Quality assessor wrapping `oximedia-quality`.
#[pyclass]
pub struct PyQualityAssessor {
    inner: QualityAssessor,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn make_frame(data: &[u8], width: usize, height: usize) -> PyResult<Frame> {
    let mut frame = Frame::new(width, height, PixelFormat::Gray8)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
    let expected = width * height;
    if data.len() < expected {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Data too small: need {expected}, got {}",
            data.len()
        )));
    }
    frame.planes[0] = data[..expected].to_vec();
    Ok(frame)
}

fn metric_name(mt: MetricType) -> String {
    match mt {
        MetricType::Psnr => "PSNR".to_string(),
        MetricType::Ssim => "SSIM".to_string(),
        MetricType::MsSsim => "MS-SSIM".to_string(),
        MetricType::Vmaf => "VMAF".to_string(),
        MetricType::Vif => "VIF".to_string(),
        MetricType::Fsim => "FSIM".to_string(),
        MetricType::Niqe => "NIQE".to_string(),
        MetricType::Brisque => "BRISQUE".to_string(),
        MetricType::Blockiness => "Blockiness".to_string(),
        MetricType::Blur => "Blur".to_string(),
        MetricType::Noise => "Noise".to_string(),
        _ => "Unknown".to_string(),
    }
}

fn to_py_score(qs: &oximedia_quality::QualityScore) -> PyQualityScore {
    PyQualityScore {
        metric: metric_name(qs.metric),
        score: qs.score,
        components: qs.components.clone(),
    }
}

fn assess_full_ref(
    assessor: &QualityAssessor,
    ref_data: &[u8],
    dist_data: &[u8],
    width: usize,
    height: usize,
    mt: MetricType,
) -> PyResult<PyQualityScore> {
    let reference = make_frame(ref_data, width, height)?;
    let distorted = make_frame(dist_data, width, height)?;
    let qs = assessor
        .assess(&reference, &distorted, mt)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
    Ok(to_py_score(&qs))
}

fn assess_no_ref(
    assessor: &QualityAssessor,
    data: &[u8],
    width: usize,
    height: usize,
    mt: MetricType,
) -> PyResult<PyQualityScore> {
    let frame = make_frame(data, width, height)?;
    let qs = assessor
        .assess_no_reference(&frame, mt)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
    Ok(to_py_score(&qs))
}

// ---------------------------------------------------------------------------
// PyMethods
// ---------------------------------------------------------------------------

#[pymethods]
impl PyQualityAssessor {
    #[new]
    fn new() -> Self {
        Self {
            inner: QualityAssessor::new(),
        }
    }

    /// Compute PSNR between reference and distorted frames.
    fn compute_psnr(
        &self,
        ref_data: &[u8],
        dist_data: &[u8],
        width: usize,
        height: usize,
    ) -> PyResult<PyQualityScore> {
        assess_full_ref(
            &self.inner,
            ref_data,
            dist_data,
            width,
            height,
            MetricType::Psnr,
        )
    }

    /// Compute SSIM between reference and distorted frames.
    fn compute_ssim(
        &self,
        ref_data: &[u8],
        dist_data: &[u8],
        width: usize,
        height: usize,
    ) -> PyResult<PyQualityScore> {
        assess_full_ref(
            &self.inner,
            ref_data,
            dist_data,
            width,
            height,
            MetricType::Ssim,
        )
    }

    /// Compute BRISQUE no-reference quality score.
    fn compute_brisque(
        &self,
        data: &[u8],
        width: usize,
        height: usize,
    ) -> PyResult<PyQualityScore> {
        assess_no_ref(&self.inner, data, width, height, MetricType::Brisque)
    }

    /// Compute NIQE no-reference quality score.
    fn compute_niqe(&self, data: &[u8], width: usize, height: usize) -> PyResult<PyQualityScore> {
        assess_no_ref(&self.inner, data, width, height, MetricType::Niqe)
    }

    /// Generate a comprehensive quality report with all no-reference metrics.
    fn quality_report(
        &self,
        data: &[u8],
        width: usize,
        height: usize,
    ) -> PyResult<Vec<PyQualityScore>> {
        let no_ref_metrics = [
            MetricType::Brisque,
            MetricType::Niqe,
            MetricType::Blockiness,
            MetricType::Blur,
            MetricType::Noise,
        ];
        let mut results = Vec::with_capacity(no_ref_metrics.len());
        for mt in &no_ref_metrics {
            results.push(assess_no_ref(&self.inner, data, width, height, *mt)?);
        }
        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Compute PSNR between two grayscale images. Returns the score as a float.
#[pyfunction]
pub fn compute_psnr(
    ref_data: &[u8],
    dist_data: &[u8],
    width: usize,
    height: usize,
) -> PyResult<f64> {
    let assessor = QualityAssessor::new();
    let reference = make_frame(ref_data, width, height)?;
    let distorted = make_frame(dist_data, width, height)?;
    let qs = assessor
        .assess(&reference, &distorted, MetricType::Psnr)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
    Ok(qs.score)
}

/// Compute SSIM between two grayscale images. Returns the score as a float.
#[pyfunction]
pub fn compute_ssim(
    ref_data: &[u8],
    dist_data: &[u8],
    width: usize,
    height: usize,
) -> PyResult<f64> {
    let assessor = QualityAssessor::new();
    let reference = make_frame(ref_data, width, height)?;
    let distorted = make_frame(dist_data, width, height)?;
    let qs = assessor
        .assess(&reference, &distorted, MetricType::Ssim)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
    Ok(qs.score)
}

/// Generate a quality report with all no-reference metrics for a grayscale image.
#[pyfunction]
pub fn quality_report(data: &[u8], width: usize, height: usize) -> PyResult<Vec<PyQualityScore>> {
    let assessor = QualityAssessor::new();
    let no_ref_metrics = [
        MetricType::Brisque,
        MetricType::Niqe,
        MetricType::Blockiness,
        MetricType::Blur,
        MetricType::Noise,
    ];
    let mut results = Vec::with_capacity(no_ref_metrics.len());
    for mt in &no_ref_metrics {
        results.push(assess_no_ref(&assessor, data, width, height, *mt)?);
    }
    Ok(results)
}
