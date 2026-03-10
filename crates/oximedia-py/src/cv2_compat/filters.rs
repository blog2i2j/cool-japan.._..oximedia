//! OpenCV-compatible image filtering functions.
//!
//! Provides Gaussian blur, median blur, bilateral filter, 2D convolution, and box filter.

use super::image_io::{extract_img, make_image_output};
use pyo3::prelude::*;
use pyo3::types::PyList;

// ── Gaussian kernel helpers ────────────────────────────────────────────────────

fn gaussian_kernel_1d(ksize: usize, sigma: f64) -> Vec<f64> {
    let half = (ksize / 2) as i64;
    let mut kernel: Vec<f64> = (0..ksize as i64)
        .map(|i| {
            let x = (i - half) as f64;
            (-x * x / (2.0 * sigma * sigma)).exp()
        })
        .collect();
    let sum: f64 = kernel.iter().sum();
    for v in &mut kernel {
        *v /= sum;
    }
    kernel
}

fn apply_1d_horizontal(src: &[u8], dst: &mut [u8], h: usize, w: usize, ch: usize, kernel: &[f64]) {
    let half = kernel.len() / 2;
    for row in 0..h {
        for col in 0..w {
            for c in 0..ch {
                let mut acc = 0.0f64;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let src_col =
                        (col as i64 + ki as i64 - half as i64).clamp(0, w as i64 - 1) as usize;
                    acc += src[(row * w + src_col) * ch + c] as f64 * kv;
                }
                dst[(row * w + col) * ch + c] = acc.clamp(0.0, 255.0) as u8;
            }
        }
    }
}

fn apply_1d_vertical(src: &[u8], dst: &mut [u8], h: usize, w: usize, ch: usize, kernel: &[f64]) {
    let half = kernel.len() / 2;
    for row in 0..h {
        for col in 0..w {
            for c in 0..ch {
                let mut acc = 0.0f64;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let src_row =
                        (row as i64 + ki as i64 - half as i64).clamp(0, h as i64 - 1) as usize;
                    acc += src[(src_row * w + col) * ch + c] as f64 * kv;
                }
                dst[(row * w + col) * ch + c] = acc.clamp(0.0, 255.0) as u8;
            }
        }
    }
}

// ── Public functions ───────────────────────────────────────────────────────────

/// Apply Gaussian blur to an image.
///
/// Mirrors `cv2.GaussianBlur(src, ksize, sigmaX, sigmaY=0)`.
/// `ksize` is a `(width, height)` tuple — both must be positive odd integers.
/// If `sigma_x == 0`, it is computed from `ksize` per the OpenCV formula.
#[pyfunction]
#[pyo3(name = "GaussianBlur", signature = (src, ksize, sigma_x, sigma_y = 0.0))]
pub fn gaussian_blur(
    py: Python<'_>,
    src: Py<PyAny>,
    ksize: (usize, usize),
    sigma_x: f64,
    sigma_y: f64,
) -> PyResult<Py<PyAny>> {
    let (data, h, w, ch) = extract_img(py, &src)?;

    let (kw, kh) = ksize;
    if kw == 0 || kh == 0 || kw % 2 == 0 || kh % 2 == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "GaussianBlur: ksize width and height must be positive odd integers",
        ));
    }

    let sx = if sigma_x == 0.0 {
        0.3 * ((kw as f64 / 2.0) - 1.0) + 0.8
    } else {
        sigma_x
    };
    let sy = if sigma_y == 0.0 { sx } else { sigma_y };

    let kx = gaussian_kernel_1d(kw, sx);
    let ky = gaussian_kernel_1d(kh, sy);

    let mut tmp = vec![0u8; h * w * ch];
    apply_1d_horizontal(&data, &mut tmp, h, w, ch, &kx);
    let mut out = vec![0u8; h * w * ch];
    apply_1d_vertical(&tmp, &mut out, h, w, ch, &ky);

    make_image_output(py, out, h, w, ch)
}

/// Apply median blur to an image.
///
/// Mirrors `cv2.medianBlur(src, ksize)`.
/// `ksize` must be a positive odd integer (e.g., 3, 5, 7).
#[pyfunction]
#[pyo3(name = "medianBlur")]
pub fn median_blur(py: Python<'_>, src: Py<PyAny>, ksize: usize) -> PyResult<Py<PyAny>> {
    let (data, h, w, ch) = extract_img(py, &src)?;

    if ksize == 0 || ksize % 2 == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "medianBlur: ksize must be a positive odd integer",
        ));
    }

    let half = (ksize / 2) as i64;
    let mut out = vec![0u8; h * w * ch];

    for row in 0..h {
        for col in 0..w {
            for c in 0..ch {
                let mut neighbors: Vec<u8> = Vec::with_capacity(ksize * ksize);
                for dr in -half..=half {
                    for dc in -half..=half {
                        let r = (row as i64 + dr).clamp(0, h as i64 - 1) as usize;
                        let cc = (col as i64 + dc).clamp(0, w as i64 - 1) as usize;
                        neighbors.push(data[(r * w + cc) * ch + c]);
                    }
                }
                neighbors.sort_unstable();
                out[(row * w + col) * ch + c] = neighbors[neighbors.len() / 2];
            }
        }
    }

    make_image_output(py, out, h, w, ch)
}

/// Apply bilateral filter to an image.
///
/// Mirrors `cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)`.
/// Preserves edges by weighting neighbours by both spatial proximity and colour similarity.
///
/// - `d`: diameter of the pixel neighbourhood; ≤0 means compute from `sigma_spatial`
/// - `sigma_color`: filter sigma in colour space
/// - `sigma_spatial`: filter sigma in coordinate space
#[pyfunction]
#[pyo3(name = "bilateralFilter")]
pub fn bilateral_filter(
    py: Python<'_>,
    src: Py<PyAny>,
    d: i32,
    sigma_color: f64,
    sigma_spatial: f64,
) -> PyResult<Py<PyAny>> {
    let (data, h, w, ch) = extract_img(py, &src)?;

    let radius = if d <= 0 {
        (sigma_spatial * 2.0).ceil() as i64
    } else {
        d as i64 / 2
    };

    let two_sc2 = 2.0 * sigma_color * sigma_color;
    let two_ss2 = 2.0 * sigma_spatial * sigma_spatial;

    let mut out = vec![0u8; h * w * ch];

    for row in 0..h {
        for col in 0..w {
            let center_off = (row * w + col) * ch;
            let mut weighted_sum = vec![0.0f64; ch];
            let mut weight_total = 0.0f64;

            for dr in -radius..=radius {
                for dc in -radius..=radius {
                    let r = (row as i64 + dr).clamp(0, h as i64 - 1) as usize;
                    let cc = (col as i64 + dc).clamp(0, w as i64 - 1) as usize;
                    let nb_off = (r * w + cc) * ch;

                    let dist2 = (dr * dr + dc * dc) as f64;
                    let spatial_w = (-dist2 / two_ss2).exp();

                    let color_diff2: f64 = (0..ch)
                        .map(|c| {
                            let diff = data[center_off + c] as f64 - data[nb_off + c] as f64;
                            diff * diff
                        })
                        .sum();
                    let color_w = (-color_diff2 / two_sc2).exp();

                    let w_combined = spatial_w * color_w;
                    weight_total += w_combined;

                    for c in 0..ch {
                        weighted_sum[c] += data[nb_off + c] as f64 * w_combined;
                    }
                }
            }

            let out_off = (row * w + col) * ch;
            for c in 0..ch {
                let v = if weight_total > 0.0 {
                    weighted_sum[c] / weight_total
                } else {
                    data[center_off + c] as f64
                };
                out[out_off + c] = v.clamp(0.0, 255.0) as u8;
            }
        }
    }

    make_image_output(py, out, h, w, ch)
}

/// Apply a custom 2D convolution kernel to an image.
///
/// Mirrors `cv2.filter2D(src, ddepth, kernel)`.
/// `kernel` is a Python list of lists of floats.
/// `ddepth` is ignored (output is always uint8); pass -1 for same-depth semantics.
#[pyfunction]
#[pyo3(name = "filter2D")]
pub fn filter_2d(
    py: Python<'_>,
    src: Py<PyAny>,
    ddepth: i32,
    kernel: Py<PyAny>,
) -> PyResult<Py<PyAny>> {
    let _ = ddepth;
    let (data, h, w, ch) = extract_img(py, &src)?;

    // Parse 2-D kernel from Python list of lists
    let kernel_bound = kernel.bind(py);
    let kernel_list = kernel_bound.cast::<PyList>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err("filter2D: kernel must be a list of lists")
    })?;
    let kh_size = kernel_list.len();
    if kh_size == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "filter2D: kernel is empty",
        ));
    }
    let mut flat_kernel: Vec<f64> = Vec::new();
    let mut kw_size = 0usize;
    for row_obj in kernel_list.iter() {
        let row = row_obj.cast::<PyList>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err("filter2D: kernel rows must be lists of floats")
        })?;
        if kw_size == 0 {
            kw_size = row.len();
        } else if row.len() != kw_size {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "filter2D: kernel rows must all have the same length",
            ));
        }
        for v in row.iter() {
            flat_kernel.push(v.extract::<f64>()?);
        }
    }

    let half_h = (kh_size / 2) as i64;
    let half_w = (kw_size / 2) as i64;
    let mut out = vec![0u8; h * w * ch];

    for row in 0..h {
        for col in 0..w {
            for c in 0..ch {
                let mut acc = 0.0f64;
                for (ki, row_kernels) in flat_kernel.chunks(kw_size).enumerate() {
                    for (kj, &kv) in row_kernels.iter().enumerate() {
                        let src_r =
                            (row as i64 + ki as i64 - half_h).clamp(0, h as i64 - 1) as usize;
                        let src_c =
                            (col as i64 + kj as i64 - half_w).clamp(0, w as i64 - 1) as usize;
                        acc += data[(src_r * w + src_c) * ch + c] as f64 * kv;
                    }
                }
                out[(row * w + col) * ch + c] = acc.clamp(0.0, 255.0) as u8;
            }
        }
    }

    make_image_output(py, out, h, w, ch)
}

/// Apply a uniform box (average) filter to an image.
///
/// Mirrors `cv2.boxFilter(src, ddepth, ksize, normalize=True)`.
/// `ksize` is a `(kw, kh)` tuple.
/// When `normalize` is true the result is divided by kw×kh; otherwise the raw sum is used.
#[pyfunction]
#[pyo3(name = "boxFilter", signature = (src, ddepth, ksize, normalize = true))]
pub fn box_filter(
    py: Python<'_>,
    src: Py<PyAny>,
    ddepth: i32,
    ksize: (usize, usize),
    normalize: bool,
) -> PyResult<Py<PyAny>> {
    let _ = ddepth;
    let (data, h, w, ch) = extract_img(py, &src)?;
    let (kw, kh) = ksize;

    if kw == 0 || kh == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "boxFilter: ksize width and height must be positive",
        ));
    }

    let half_h = (kh / 2) as i64;
    let half_w = (kw / 2) as i64;
    let divisor = if normalize { (kw * kh) as f64 } else { 1.0 };

    let mut out = vec![0u8; h * w * ch];

    for row in 0..h {
        for col in 0..w {
            for c in 0..ch {
                let mut acc = 0.0f64;
                for dr in -half_h..=half_h {
                    for dc in -half_w..=half_w {
                        let r = (row as i64 + dr).clamp(0, h as i64 - 1) as usize;
                        let cc = (col as i64 + dc).clamp(0, w as i64 - 1) as usize;
                        acc += data[(r * w + cc) * ch + c] as f64;
                    }
                }
                out[(row * w + col) * ch + c] = (acc / divisor).clamp(0.0, 255.0) as u8;
            }
        }
    }

    make_image_output(py, out, h, w, ch)
}

/// Register all filter functions with a Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gaussian_blur, m)?)?;
    m.add_function(wrap_pyfunction!(median_blur, m)?)?;
    m.add_function(wrap_pyfunction!(bilateral_filter, m)?)?;
    m.add_function(wrap_pyfunction!(filter_2d, m)?)?;
    m.add_function(wrap_pyfunction!(box_filter, m)?)?;
    Ok(())
}
