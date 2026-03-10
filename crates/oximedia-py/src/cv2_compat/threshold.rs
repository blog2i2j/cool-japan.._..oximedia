//! OpenCV-compatible thresholding functions.
//!
//! Provides global threshold (with Otsu and Triangle methods) and adaptive threshold.

use super::image_io::{extract_img, make_image_output};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

// ── Otsu's method ─────────────────────────────────────────────────────────────

pub(crate) fn otsu_threshold(data: &[u8]) -> u8 {
    let mut hist = [0u32; 256];
    for &p in data {
        hist[p as usize] += 1;
    }
    let total = data.len() as f64;
    let mut sum_total = 0.0f64;
    for i in 0..256usize {
        sum_total += i as f64 * hist[i] as f64;
    }
    let mut w_b = 0.0f64;
    let mut sum_b = 0.0f64;
    let mut max_var = 0.0f64;
    let mut best_t = 0u8;
    for t in 0..256usize {
        w_b += hist[t] as f64;
        if w_b == 0.0 {
            continue;
        }
        let w_f = total - w_b;
        if w_f == 0.0 {
            break;
        }
        sum_b += t as f64 * hist[t] as f64;
        let m_b = sum_b / w_b;
        let m_f = (sum_total - sum_b) / w_f;
        let var = w_b * w_f * (m_b - m_f).powi(2);
        if var > max_var {
            max_var = var;
            best_t = t as u8;
        }
    }
    best_t
}

// ── Triangle method ───────────────────────────────────────────────────────────

pub(crate) fn triangle_threshold(data: &[u8]) -> u8 {
    let mut hist = [0u32; 256];
    for &p in data {
        hist[p as usize] += 1;
    }

    // Find occupied range
    let mut min_idx = 0usize;
    let mut max_idx = 255usize;
    for i in 0..256 {
        if hist[i] > 0 {
            min_idx = i;
            break;
        }
    }
    for i in (0..256).rev() {
        if hist[i] > 0 {
            max_idx = i;
            break;
        }
    }
    if min_idx >= max_idx {
        return min_idx as u8;
    }

    // Find histogram peak
    let peak_idx = (min_idx..=max_idx).max_by_key(|&i| hist[i]).unwrap_or(128);

    // Choose the side with the longest run from peak to edge
    let (line_start, line_end) = if peak_idx - min_idx > max_idx - peak_idx {
        (min_idx, peak_idx)
    } else {
        (peak_idx, max_idx)
    };

    let dx = (line_end - line_start) as f64;
    let dy = hist[line_start] as f64 - hist[line_end] as f64;
    let line_len = (dx * dx + dy * dy).sqrt();
    if line_len < 1e-10 {
        return line_start as u8;
    }

    let mut best_t = line_start;
    let mut max_dist = 0.0f64;

    for i in line_start..=line_end {
        let px = (i - line_start) as f64;
        let py = hist[i] as f64 - hist[line_end] as f64;
        let dist = (dy * px - dx * py).abs() / line_len;
        if dist > max_dist {
            max_dist = dist;
            best_t = i;
        }
    }

    best_t as u8
}

// ── Threshold type application ────────────────────────────────────────────────

#[inline]
pub(crate) fn apply_thresh_type(pixel: u8, thresh: u8, maxval: u8, thresh_type_base: i32) -> u8 {
    let v = pixel;
    let t = thresh;
    match thresh_type_base {
        0 => {
            if v > t {
                maxval
            } else {
                0
            }
        } // THRESH_BINARY
        1 => {
            if v > t {
                0
            } else {
                maxval
            }
        } // THRESH_BINARY_INV
        2 => {
            if v > t {
                t
            } else {
                v
            }
        } // THRESH_TRUNC
        3 => {
            if v > t {
                v
            } else {
                0
            }
        } // THRESH_TOZERO
        4 => {
            if v > t {
                0
            } else {
                v
            }
        } // THRESH_TOZERO_INV
        _ => v,
    }
}

// ── Public functions ───────────────────────────────────────────────────────────

/// Apply a fixed-level threshold to each array element.
///
/// Mirrors `cv2.threshold(src, thresh, maxval, type)`.
/// Returns `(retval, dst)` where `retval` is the threshold value used
/// (useful when Otsu or Triangle method is selected).
///
/// # threshold_type flags
/// - 0  = THRESH_BINARY
/// - 1  = THRESH_BINARY_INV
/// - 2  = THRESH_TRUNC
/// - 3  = THRESH_TOZERO
/// - 4  = THRESH_TOZERO_INV
/// - 8  = THRESH_OTSU  (OR-able with 0 or 1)
/// - 16 = THRESH_TRIANGLE (OR-able with 0 or 1)
#[pyfunction]
#[pyo3(name = "threshold")]
pub fn threshold(
    py: Python<'_>,
    src: Py<PyAny>,
    thresh: f64,
    maxval: f64,
    threshold_type: i32,
) -> PyResult<Py<PyAny>> {
    let (data, h, w, ch) = extract_img(py, &src)?;

    let use_otsu = (threshold_type & 8) != 0;
    let use_triangle = (threshold_type & 16) != 0;
    let base_type = threshold_type & 0x07;

    let effective_thresh: u8 = if use_otsu {
        otsu_threshold(&data)
    } else if use_triangle {
        triangle_threshold(&data)
    } else {
        thresh.clamp(0.0, 255.0) as u8
    };

    let maxval_u8 = maxval.clamp(0.0, 255.0) as u8;

    let out: Vec<u8> = data
        .iter()
        .map(|&p| apply_thresh_type(p, effective_thresh, maxval_u8, base_type))
        .collect();

    let dst = make_image_output(py, out, h, w, ch)?;
    let retval = effective_thresh as f64;
    let result = PyTuple::new(py, [retval.into_pyobject(py)?.into_any().unbind(), dst])?;
    Ok(result.into())
}

// ── Adaptive threshold ────────────────────────────────────────────────────────

/// Gaussian neighbourhood weight (unnormalised).
#[inline]
fn gaussian_weight(dx: i64, dy: i64, sigma: f64) -> f64 {
    let r2 = (dx * dx + dy * dy) as f64;
    (-r2 / (2.0 * sigma * sigma)).exp()
}

/// Apply adaptive thresholding to a single-channel image.
///
/// Mirrors `cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C)`.
///
/// # Parameters
/// - `adaptive_method`: 0 = ADAPTIVE_THRESH_MEAN_C, 1 = ADAPTIVE_THRESH_GAUSSIAN_C
/// - `threshold_type`: 0 = THRESH_BINARY, 1 = THRESH_BINARY_INV
/// - `block_size`: neighbourhood size (must be odd and ≥ 3)
/// - `c`: constant subtracted from the computed local mean
#[pyfunction]
#[pyo3(name = "adaptiveThreshold")]
pub fn adaptive_threshold(
    py: Python<'_>,
    src: Py<PyAny>,
    maxval: f64,
    adaptive_method: i32,
    threshold_type: i32,
    block_size: i32,
    c: f64,
) -> PyResult<Py<PyAny>> {
    let (data, h, w, ch) = extract_img(py, &src)?;

    if ch != 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "adaptiveThreshold requires a single-channel (grayscale) image",
        ));
    }
    if block_size < 3 || block_size % 2 == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "adaptiveThreshold: block_size must be odd and >= 3",
        ));
    }

    let half = (block_size / 2) as i64;
    let sigma = half as f64 / 2.0;
    let maxval_u8 = maxval.clamp(0.0, 255.0) as u8;

    // Pre-compute Gaussian weights (and normalise) if needed
    let gauss_weights: Option<Vec<f64>> = if adaptive_method == 1 {
        let mut wts = Vec::with_capacity(((2 * half + 1) * (2 * half + 1)) as usize);
        let mut wsum = 0.0f64;
        for dr in -half..=half {
            for dc in -half..=half {
                let wt = gaussian_weight(dr, dc, sigma);
                wts.push(wt);
                wsum += wt;
            }
        }
        if wsum > 0.0 {
            for w in &mut wts {
                *w /= wsum;
            }
        }
        Some(wts)
    } else {
        None
    };

    let total_in_block = ((2 * half + 1) * (2 * half + 1)) as f64;

    let mut out = vec![0u8; h * w];

    for row in 0..h {
        for col in 0..w {
            let local_mean = if let Some(ref gwts) = gauss_weights {
                let mut acc = 0.0f64;
                let mut kidx = 0usize;
                for dr in -half..=half {
                    for dc in -half..=half {
                        let r = (row as i64 + dr).clamp(0, h as i64 - 1) as usize;
                        let cc = (col as i64 + dc).clamp(0, w as i64 - 1) as usize;
                        acc += data[r * w + cc] as f64 * gwts[kidx];
                        kidx += 1;
                    }
                }
                acc
            } else {
                // Mean
                let mut acc = 0.0f64;
                for dr in -half..=half {
                    for dc in -half..=half {
                        let r = (row as i64 + dr).clamp(0, h as i64 - 1) as usize;
                        let cc = (col as i64 + dc).clamp(0, w as i64 - 1) as usize;
                        acc += data[r * w + cc] as f64;
                    }
                }
                acc / total_in_block
            };

            let local_thresh = (local_mean - c).clamp(0.0, 255.0) as u8;
            let pixel = data[row * w + col];
            out[row * w + col] = match threshold_type {
                1 => {
                    if pixel > local_thresh {
                        0
                    } else {
                        maxval_u8
                    }
                }
                _ => {
                    if pixel > local_thresh {
                        maxval_u8
                    } else {
                        0
                    }
                }
            };
        }
    }

    make_image_output(py, out, h, w, 1)
}

/// Register all threshold functions with a Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(threshold, m)?)?;
    m.add_function(wrap_pyfunction!(adaptive_threshold, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── apply_thresh_type ─────────────────────────────────────────────────────

    #[test]
    fn test_thresh_binary_above_threshold() {
        // pixel > thresh → maxval
        assert_eq!(
            apply_thresh_type(200, 128, 255, 0),
            255,
            "200 > 128 should map to maxval"
        );
    }

    #[test]
    fn test_thresh_binary_below_threshold() {
        // pixel <= thresh → 0
        assert_eq!(
            apply_thresh_type(100, 128, 255, 0),
            0,
            "100 <= 128 should map to 0"
        );
    }

    #[test]
    fn test_thresh_binary_batch() {
        let data = [100u8, 200u8, 50u8, 150u8];
        let out: Vec<u8> = data
            .iter()
            .map(|&p| apply_thresh_type(p, 128, 255, 0))
            .collect();
        assert_eq!(out[0], 0, "100 < 128 → 0");
        assert_eq!(out[1], 255, "200 > 128 → 255");
        assert_eq!(out[2], 0, "50 < 128 → 0");
        assert_eq!(out[3], 255, "150 > 128 → 255");
    }

    #[test]
    fn test_thresh_binary_inv() {
        // THRESH_BINARY_INV: pixel > thresh → 0, else maxval
        assert_eq!(
            apply_thresh_type(100, 128, 255, 1),
            255,
            "100 < 128 → 255 (inv)"
        );
        assert_eq!(
            apply_thresh_type(200, 128, 255, 1),
            0,
            "200 > 128 → 0 (inv)"
        );
    }

    #[test]
    fn test_thresh_trunc() {
        // THRESH_TRUNC: pixel > thresh → thresh, else pixel unchanged
        assert_eq!(
            apply_thresh_type(100, 128, 255, 2),
            100,
            "100 < 128 unchanged"
        );
        assert_eq!(
            apply_thresh_type(200, 128, 255, 2),
            128,
            "200 > 128 truncated to 128"
        );
    }

    #[test]
    fn test_thresh_tozero() {
        // THRESH_TOZERO: pixel > thresh → pixel, else 0
        assert_eq!(apply_thresh_type(100, 128, 255, 3), 0, "100 < 128 → 0");
        assert_eq!(
            apply_thresh_type(200, 128, 255, 3),
            200,
            "200 > 128 unchanged"
        );
    }

    #[test]
    fn test_thresh_tozero_inv() {
        // THRESH_TOZERO_INV: pixel > thresh → 0, else pixel
        assert_eq!(
            apply_thresh_type(100, 128, 255, 4),
            100,
            "100 < 128 unchanged"
        );
        assert_eq!(apply_thresh_type(200, 128, 255, 4), 0, "200 > 128 → 0");
    }

    // ── otsu_threshold ────────────────────────────────────────────────────────

    #[test]
    fn test_otsu_uniform_image_gives_valid_threshold() {
        // All same value → Otsu should return a valid u8
        let data = vec![128u8; 100];
        let t = otsu_threshold(&data);
        // Any valid u8 is acceptable; just ensure it doesn't panic
        let _ = t;
    }

    #[test]
    fn test_otsu_bimodal_separates_peaks() {
        // 50% pixels at 50 + 50% pixels at 200.
        // Otsu maximises inter-class variance; threshold is at the lower peak value
        // (inclusive) — pixels *above* threshold are foreground.
        let mut data = vec![50u8; 50];
        data.extend(vec![200u8; 50]);
        let t = otsu_threshold(&data);
        // The optimal threshold must lie between the two clusters (inclusive of peak values)
        assert!(
            t >= 50 && t < 200,
            "Otsu threshold should separate the two peaks (>= 50 and < 200), got {}",
            t
        );
    }

    #[test]
    fn test_otsu_all_same_value_returns_that_value_or_zero() {
        // Single value: all dark
        let data = vec![0u8; 200];
        let t = otsu_threshold(&data);
        // Between iterations, variance never exceeds 0 → best_t stays at 0
        assert_eq!(t, 0, "all-dark image Otsu threshold should be 0");
    }

    // ── triangle_threshold ────────────────────────────────────────────────────

    #[test]
    fn test_triangle_valid_range() {
        // Ramp from 0 to 255 — triangle method should pick something in the middle
        let data: Vec<u8> = (0u8..=255).collect();
        let t = triangle_threshold(&data);
        let _ = t; // just verify it doesn't panic and returns a u8
    }

    #[test]
    fn test_triangle_single_value() {
        // All same value — trivially degenerates; result should be that value
        let data = vec![100u8; 100];
        let t = triangle_threshold(&data);
        assert_eq!(t, 100, "single-value image should return that value");
    }
}
