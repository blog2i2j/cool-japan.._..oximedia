//! Frame-level video quality metrics exposed to the browser via WASM.

use oximedia_core::PixelFormat;
use oximedia_quality::{Frame, MetricType, QualityAssessor};
use wasm_bindgen::prelude::*;

// ─── Frame construction helper ────────────────────────────────────────────────

/// Build a grayscale [`Frame`] from a raw `u8` slice.
///
/// # Errors
///
/// Returns an error if `data` is shorter than `width * height` bytes or if
/// frame allocation fails.
fn make_quality_frame(data: &[u8], width: u32, height: u32) -> Result<Frame, JsValue> {
    let expected = (width * height) as usize;
    if data.len() < expected {
        return Err(crate::utils::js_err(&format!(
            "Data too small: need {expected} bytes for {width}x{height} Gray8, got {}",
            data.len()
        )));
    }
    let mut frame = Frame::new(width as usize, height as usize, PixelFormat::Gray8)
        .map_err(|e| crate::utils::js_err(&format!("Frame allocation failed: {e}")))?;
    frame.planes[0] = data[..expected].to_vec();
    Ok(frame)
}

// ─── Full-reference metrics ───────────────────────────────────────────────────

/// Compute PSNR between a reference and a distorted grayscale frame.
///
/// Both slices must contain at least `width * height` bytes of Gray8 data.
///
/// Returns PSNR in dB.
///
/// # Errors
///
/// Returns an error if the frames cannot be created or the metric fails.
#[wasm_bindgen]
pub fn wasm_compute_psnr(
    reference: &[u8],
    distorted: &[u8],
    width: u32,
    height: u32,
) -> Result<f64, JsValue> {
    let ref_frame = make_quality_frame(reference, width, height)?;
    let dist_frame = make_quality_frame(distorted, width, height)?;
    let assessor = QualityAssessor::new();
    let score = assessor
        .assess(&ref_frame, &dist_frame, MetricType::Psnr)
        .map_err(|e| crate::utils::js_err(&format!("PSNR error: {e}")))?;
    Ok(score.score)
}

/// Compute SSIM between a reference and a distorted grayscale frame.
///
/// Both slices must contain at least `width * height` bytes of Gray8 data.
///
/// Returns SSIM score in the range [0.0, 1.0].
///
/// # Errors
///
/// Returns an error if the frames cannot be created or the metric fails.
#[wasm_bindgen]
pub fn wasm_compute_ssim(
    reference: &[u8],
    distorted: &[u8],
    width: u32,
    height: u32,
) -> Result<f64, JsValue> {
    let ref_frame = make_quality_frame(reference, width, height)?;
    let dist_frame = make_quality_frame(distorted, width, height)?;
    let assessor = QualityAssessor::new();
    let score = assessor
        .assess(&ref_frame, &dist_frame, MetricType::Ssim)
        .map_err(|e| crate::utils::js_err(&format!("SSIM error: {e}")))?;
    Ok(score.score)
}

// ─── No-reference metrics ─────────────────────────────────────────────────────

/// Compute no-reference quality metrics on a single grayscale frame.
///
/// `data` must contain at least `width * height` bytes of Gray8 data.
///
/// Returns a JSON object with scores for BRISQUE, NIQE, blockiness, blur,
/// and noise:
/// ```json
/// {
///   "brisque": 25.3,
///   "niqe": 3.1,
///   "blockiness": 0.04,
///   "blur": 0.87,
///   "noise": 0.11
/// }
/// ```
///
/// # Errors
///
/// Returns an error if the frame cannot be created or a metric fails.
#[wasm_bindgen]
pub fn wasm_frame_quality(data: &[u8], width: u32, height: u32) -> Result<String, JsValue> {
    let frame = make_quality_frame(data, width, height)?;
    let assessor = QualityAssessor::new();

    let brisque = assessor
        .assess_no_reference(&frame, MetricType::Brisque)
        .map_err(|e| crate::utils::js_err(&format!("BRISQUE error: {e}")))?
        .score;

    let niqe = assessor
        .assess_no_reference(&frame, MetricType::Niqe)
        .map_err(|e| crate::utils::js_err(&format!("NIQE error: {e}")))?
        .score;

    let blockiness = assessor
        .assess_no_reference(&frame, MetricType::Blockiness)
        .map_err(|e| crate::utils::js_err(&format!("Blockiness error: {e}")))?
        .score;

    let blur = assessor
        .assess_no_reference(&frame, MetricType::Blur)
        .map_err(|e| crate::utils::js_err(&format!("Blur error: {e}")))?
        .score;

    let noise = assessor
        .assess_no_reference(&frame, MetricType::Noise)
        .map_err(|e| crate::utils::js_err(&format!("Noise error: {e}")))?
        .score;

    Ok(format!(
        "{{\"brisque\":{brisque},\"niqe\":{niqe},\"blockiness\":{blockiness},\"blur\":{blur},\"noise\":{noise}}}"
    ))
}
