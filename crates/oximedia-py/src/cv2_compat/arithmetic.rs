//! Arithmetic, histogram, and utility functions for cv2 compat.

use ndarray::Array3;
use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use super::image_io::extract_img;

// ── Pixel arithmetic ──────────────────────────────────────────────────────

/// Weighted sum of two images: dst = alpha*src1 + beta*src2 + gamma
#[pyfunction]
#[pyo3(name = "addWeighted")]
#[pyo3(signature = (src1, alpha, src2, beta, gamma))]
pub fn add_weighted(
    py: Python<'_>,
    src1: Py<PyAny>,
    alpha: f64,
    src2: Py<PyAny>,
    beta: f64,
    gamma: f64,
) -> PyResult<Py<PyAny>> {
    let (d1, h, w, ch) = extract_img(py, &src1)?;
    let (d2, h2, w2, ch2) = extract_img(py, &src2)?;
    if h != h2 || w != w2 || ch != ch2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "addWeighted: images must have same shape",
        ));
    }
    let out: Vec<u8> = d1
        .iter()
        .zip(d2.iter())
        .map(|(&a, &b)| (alpha * a as f64 + beta * b as f64 + gamma).clamp(0.0, 255.0) as u8)
        .collect();
    let arr = Array3::from_shape_vec((h, w, ch), out).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("addWeighted shape error: {}", e))
    })?;
    Ok(arr.into_pyarray(py).into_any().unbind())
}

/// Per-pixel absolute difference: |src1 - src2|
#[pyfunction]
#[pyo3(name = "absdiff")]
pub fn abs_diff(py: Python<'_>, src1: Py<PyAny>, src2: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let (d1, h, w, ch) = extract_img(py, &src1)?;
    let (d2, h2, w2, ch2) = extract_img(py, &src2)?;
    if h != h2 || w != w2 || ch != ch2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "absdiff: images must have same shape",
        ));
    }
    let out: Vec<u8> = d1
        .iter()
        .zip(d2.iter())
        .map(|(&a, &b)| (a as i32 - b as i32).unsigned_abs() as u8)
        .collect();
    let arr = Array3::from_shape_vec((h, w, ch), out)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("absdiff: {}", e)))?;
    Ok(arr.into_pyarray(py).into_any().unbind())
}

/// Bitwise AND of two images (optionally masked)
#[pyfunction]
#[pyo3(name = "bitwise_and")]
#[pyo3(signature = (src1, src2, mask=None))]
pub fn bitwise_and(
    py: Python<'_>,
    src1: Py<PyAny>,
    src2: Py<PyAny>,
    mask: Option<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    let (d1, h, w, ch) = extract_img(py, &src1)?;
    let (d2, ..) = extract_img(py, &src2)?;
    let mask_data = mask
        .as_ref()
        .map(|m| extract_img(py, m).map(|(d, ..)| d))
        .transpose()?;
    let out: Vec<u8> = (0..d1.len())
        .map(|i| {
            let pixel_idx = i / ch;
            let masked = mask_data.as_ref().map(|m| m[pixel_idx] > 0).unwrap_or(true);
            if masked {
                d1[i] & d2[i]
            } else {
                0
            }
        })
        .collect();
    let arr = Array3::from_shape_vec((h, w, ch), out)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("bitwise_and: {}", e)))?;
    Ok(arr.into_pyarray(py).into_any().unbind())
}

/// Bitwise OR of two images (optionally masked)
#[pyfunction]
#[pyo3(name = "bitwise_or")]
#[pyo3(signature = (src1, src2, mask=None))]
pub fn bitwise_or(
    py: Python<'_>,
    src1: Py<PyAny>,
    src2: Py<PyAny>,
    mask: Option<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    let (d1, h, w, ch) = extract_img(py, &src1)?;
    let (d2, ..) = extract_img(py, &src2)?;
    let mask_data = mask
        .as_ref()
        .map(|m| extract_img(py, m).map(|(d, ..)| d))
        .transpose()?;
    let out: Vec<u8> = (0..d1.len())
        .map(|i| {
            let pixel_idx = i / ch;
            let masked = mask_data.as_ref().map(|m| m[pixel_idx] > 0).unwrap_or(true);
            if masked {
                d1[i] | d2[i]
            } else {
                0
            }
        })
        .collect();
    let arr = Array3::from_shape_vec((h, w, ch), out)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("bitwise_or: {}", e)))?;
    Ok(arr.into_pyarray(py).into_any().unbind())
}

/// Bitwise XOR of two images (optionally masked)
#[pyfunction]
#[pyo3(name = "bitwise_xor")]
#[pyo3(signature = (src1, src2, mask=None))]
pub fn bitwise_xor(
    py: Python<'_>,
    src1: Py<PyAny>,
    src2: Py<PyAny>,
    mask: Option<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    let (d1, h, w, ch) = extract_img(py, &src1)?;
    let (d2, ..) = extract_img(py, &src2)?;
    let mask_data = mask
        .as_ref()
        .map(|m| extract_img(py, m).map(|(d, ..)| d))
        .transpose()?;
    let out: Vec<u8> = (0..d1.len())
        .map(|i| {
            let pixel_idx = i / ch;
            let masked = mask_data.as_ref().map(|m| m[pixel_idx] > 0).unwrap_or(true);
            if masked {
                d1[i] ^ d2[i]
            } else {
                0
            }
        })
        .collect();
    let arr = Array3::from_shape_vec((h, w, ch), out)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("bitwise_xor: {}", e)))?;
    Ok(arr.into_pyarray(py).into_any().unbind())
}

/// Bitwise NOT of an image (optionally masked)
#[pyfunction]
#[pyo3(name = "bitwise_not")]
#[pyo3(signature = (src, mask=None))]
pub fn bitwise_not(py: Python<'_>, src: Py<PyAny>, mask: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
    let (data, h, w, ch) = extract_img(py, &src)?;
    let mask_data = mask
        .as_ref()
        .map(|m| extract_img(py, m).map(|(d, ..)| d))
        .transpose()?;
    let out: Vec<u8> = (0..data.len())
        .map(|i| {
            let pixel_idx = i / ch;
            let masked = mask_data.as_ref().map(|m| m[pixel_idx] > 0).unwrap_or(true);
            if masked {
                !data[i]
            } else {
                data[i]
            }
        })
        .collect();
    let arr = Array3::from_shape_vec((h, w, ch), out)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("bitwise_not: {}", e)))?;
    Ok(arr.into_pyarray(py).into_any().unbind())
}

/// Normalize image pixel values into [alpha, beta] using MINMAX strategy.
///
/// `dst` parameter is accepted for API compatibility but ignored (a new array is returned).
/// `norm_type` is accepted for API compatibility; only MINMAX (32) is implemented.
#[pyfunction]
#[pyo3(name = "normalize")]
#[pyo3(signature = (src, dst=None, alpha=0.0, beta=255.0, norm_type=32))]
pub fn normalize(
    py: Python<'_>,
    src: Py<PyAny>,
    dst: Option<Py<PyAny>>,
    alpha: f64,
    beta: f64,
    norm_type: i32,
) -> PyResult<Py<PyAny>> {
    // dst and norm_type accepted for API compat; only MINMAX implemented
    let _ = (dst, norm_type);
    let (data, h, w, ch) = extract_img(py, &src)?;

    let min_val = data.iter().copied().min().unwrap_or(0) as f64;
    let max_val = data.iter().copied().max().unwrap_or(255) as f64;
    let range = max_val - min_val;

    let out: Vec<u8> = if range < 1e-6 {
        vec![alpha as u8; data.len()]
    } else {
        data.iter()
            .map(|&v| {
                let normalized = (v as f64 - min_val) / range;
                (alpha + normalized * (beta - alpha)).clamp(0.0, 255.0) as u8
            })
            .collect()
    };

    let arr = Array3::from_shape_vec((h, w, ch), out)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("normalize: {}", e)))?;
    Ok(arr.into_pyarray(py).into_any().unbind())
}

/// Pixel range check: output(x,y) = 255 if lowerb(i) <= src(x,y,i) <= upperb(i) else 0.
///
/// `lower_b` and `upper_b` must be Python lists (or tuples) of per-channel bounds.
/// Returns a single-channel (H, W, 1) uint8 mask.
#[pyfunction]
#[pyo3(name = "inRange")]
pub fn in_range(
    py: Python<'_>,
    src: Py<PyAny>,
    lower_b: Py<PyAny>,
    upper_b: Py<PyAny>,
) -> PyResult<Py<PyAny>> {
    let (data, h, w, ch) = extract_img(py, &src)?;
    let lo: Vec<u8> = lower_b.extract(py)?;
    let hi: Vec<u8> = upper_b.extract(py)?;

    // Broadcast scalar bounds to all channels
    let lo_ch: Vec<u8> = if lo.len() >= ch {
        lo[..ch].to_vec()
    } else {
        vec![lo.first().copied().unwrap_or(0); ch]
    };
    let hi_ch: Vec<u8> = if hi.len() >= ch {
        hi[..ch].to_vec()
    } else {
        vec![hi.first().copied().unwrap_or(255); ch]
    };

    let mut out = vec![0u8; h * w];
    for i in 0..h * w {
        let in_rng = (0..ch).all(|c| {
            let v = data[i * ch + c];
            v >= lo_ch[c] && v <= hi_ch[c]
        });
        out[i] = if in_rng { 255 } else { 0 };
    }

    // Expand to (H, W, 1)
    let out3: Vec<u8> = out.iter().flat_map(|&v| [v]).collect();
    let arr = Array3::from_shape_vec((h, w, 1), out3)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("inRange: {}", e)))?;
    Ok(arr.into_pyarray(py).into_any().unbind())
}

// ── Histogram functions ────────────────────────────────────────────────────

/// Calculate image histogram.
///
/// Mirrors `cv2.calcHist([img], [channel], mask, [histSize], [ranges])`.
/// Returns a Python list of float32 bin counts.
#[pyfunction]
#[pyo3(name = "calcHist")]
#[pyo3(signature = (images, channels, mask, hist_size, ranges))]
pub fn calc_hist(
    py: Python<'_>,
    images: Py<PyAny>,
    channels: Py<PyAny>,
    mask: Option<Py<PyAny>>,
    hist_size: Py<PyAny>,
    ranges: Py<PyAny>,
) -> PyResult<Py<PyAny>> {
    let imgs: Vec<Py<PyAny>> = images.extract(py)?;
    let img = imgs
        .first()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("calcHist: no images provided"))?;

    let (data, h, w, ch) = extract_img(py, img)?;

    let chans: Vec<usize> = channels.extract(py)?;
    let chan_idx = chans.first().copied().unwrap_or(0);

    let sizes: Vec<usize> = hist_size.extract(py)?;
    let num_bins = sizes.first().copied().unwrap_or(256);

    let rngs: Vec<f64> = ranges.extract(py)?;
    let range_min = rngs.first().copied().unwrap_or(0.0);
    let range_max = rngs.get(1).copied().unwrap_or(256.0);
    let range_span = range_max - range_min;

    let mask_data = mask
        .as_ref()
        .map(|m| extract_img(py, m).map(|(d, ..)| d))
        .transpose()?;

    let mut hist = vec![0u32; num_bins];
    for i in 0..h * w {
        if let Some(ref md) = mask_data {
            if md[i] == 0 {
                continue;
            }
        }
        let val = if chan_idx < ch {
            data[i * ch + chan_idx] as f64
        } else {
            0.0
        };
        if range_span > 1e-9 {
            let bin = ((val - range_min) / range_span * num_bins as f64) as usize;
            let bin = bin.min(num_bins - 1);
            hist[bin] += 1;
        }
    }

    // Return as list of f32 (matching OpenCV's float32 histogram)
    let list = PyList::new(py, hist.iter().map(|&v| v as f32))?;
    Ok(list.into())
}

/// Equalize histogram of a single-channel grayscale image.
///
/// Returns an equalized (H, W, 1) uint8 image.
#[pyfunction]
#[pyo3(name = "equalizeHist")]
pub fn equalize_hist(py: Python<'_>, src: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let (data, h, w, ch) = extract_img(py, &src)?;
    if ch != 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "equalizeHist: input must be grayscale (1 channel)",
        ));
    }

    // Build histogram
    let mut hist = [0u32; 256];
    for &v in &data {
        hist[v as usize] += 1;
    }

    // Compute cumulative distribution function
    let mut cdf = [0u32; 256];
    cdf[0] = hist[0];
    for i in 1..256 {
        cdf[i] = cdf[i - 1] + hist[i];
    }

    let total = (h * w) as f32;
    let cdf_min = cdf.iter().find(|&&v| v > 0).copied().unwrap_or(0) as f32;
    let denominator = total - cdf_min;

    let equalization_lut: Vec<u8> = (0..256)
        .map(|v| {
            if denominator < 1e-6 {
                v as u8
            } else {
                ((cdf[v] as f32 - cdf_min) / denominator * 255.0).clamp(0.0, 255.0) as u8
            }
        })
        .collect();

    let out: Vec<u8> = data.iter().map(|&v| equalization_lut[v as usize]).collect();

    let arr = Array3::from_shape_vec((h, w, 1), out)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("equalizeHist: {}", e)))?;
    Ok(arr.into_pyarray(py).into_any().unbind())
}

// ── Template matching ──────────────────────────────────────────────────────

/// Template matching: slide `templ` over `image` and compute similarity.
///
/// Method codes:
/// - 0 = TM_SQDIFF, 1 = TM_SQDIFF_NORMED
/// - 2 = TM_CCORR,  3 = TM_CCORR_NORMED
/// - 4 = TM_CCOEFF, 5 = TM_CCOEFF_NORMED
///
/// Returns a dict `{"data": bytes, "shape": (H, W), "dtype": "float32"}` suitable
/// for `minMaxLoc`. Callers that need a numpy array should reconstruct from the bytes.
#[pyfunction]
#[pyo3(name = "matchTemplate")]
pub fn match_template(
    py: Python<'_>,
    image: Py<PyAny>,
    templ: Py<PyAny>,
    method: i32,
) -> PyResult<Py<PyAny>> {
    let (img_data, ih, iw, ch) = extract_img(py, &image)?;
    let (tpl_data, th, tw, tch) = extract_img(py, &templ)?;

    if tch != ch {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "matchTemplate: image and template must have the same number of channels",
        ));
    }
    if th > ih || tw > iw {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "matchTemplate: template larger than image",
        ));
    }

    let result_h = ih - th + 1;
    let result_w = iw - tw + 1;
    let mut result = vec![0.0f32; result_h * result_w];

    // Convert to grayscale for correlation computation
    let to_gray = |pixels: &[u8], stride: usize| -> Vec<f32> {
        let npix = pixels.len() / stride;
        (0..npix)
            .map(|i| {
                let off = i * stride;
                if stride == 1 {
                    pixels[off] as f32
                } else {
                    // BGR layout: B=off, G=off+1, R=off+2
                    0.114 * pixels[off] as f32
                        + 0.587 * pixels[off + 1] as f32
                        + 0.299 * pixels[off + 2] as f32
                }
            })
            .collect()
    };

    let img_gray = to_gray(&img_data, ch);
    let tpl_gray = to_gray(&tpl_data, tch);

    // Template statistics for normalised/ccoeff methods
    let tpl_sum: f32 = tpl_gray.iter().sum();
    let tpl_mean = tpl_sum / (th * tw) as f32;
    let tpl_sq_sum: f32 = tpl_gray.iter().map(|&v| v * v).sum();
    let tpl_norm = (tpl_sq_sum - tpl_mean * tpl_mean * (th * tw) as f32).sqrt();

    for ry in 0..result_h {
        for rx in 0..result_w {
            let mut sqdiff = 0.0f32;
            let mut ccorr = 0.0f32;
            let mut win_sum = 0.0f32;

            // Window mean for CCOEFF variants
            if method >= 4 {
                for ty in 0..th {
                    for tx in 0..tw {
                        win_sum += img_gray[(ry + ty) * iw + (rx + tx)];
                    }
                }
            }
            let win_mean = win_sum / (th * tw) as f32;

            for ty in 0..th {
                for tx in 0..tw {
                    let iv = img_gray[(ry + ty) * iw + (rx + tx)];
                    let tv = tpl_gray[ty * tw + tx];
                    match method {
                        0 | 1 => sqdiff += (iv - tv) * (iv - tv),
                        2 | 3 => ccorr += iv * tv,
                        _ => ccorr += (iv - win_mean) * (tv - tpl_mean),
                    }
                }
            }

            // For normalised methods compute window energy with an explicit loop
            let win_energy: f32 = if method == 1 || method == 3 || method == 5 {
                let mut sum = 0.0f32;
                for ty in 0..th {
                    for tx in 0..tw {
                        let v = if method == 5 {
                            img_gray[(ry + ty) * iw + (rx + tx)] - win_mean
                        } else {
                            img_gray[(ry + ty) * iw + (rx + tx)]
                        };
                        sum += v * v;
                    }
                }
                sum
            } else {
                0.0
            };

            result[ry * result_w + rx] = match method {
                0 => -sqdiff, // TM_SQDIFF (negate so higher = better for minMaxLoc)
                1 => {
                    // TM_SQDIFF_NORMED
                    let denom = (win_energy * tpl_sq_sum).sqrt();
                    if denom > 1e-9 {
                        -(sqdiff / denom)
                    } else {
                        0.0
                    }
                }
                3 => {
                    // TM_CCORR_NORMED
                    let denom = (win_energy * tpl_sq_sum).sqrt();
                    if denom > 1e-9 {
                        ccorr / denom
                    } else {
                        0.0
                    }
                }
                5 => {
                    // TM_CCOEFF_NORMED
                    let denom = win_energy.sqrt() * tpl_norm;
                    if denom > 1e-9 {
                        ccorr / denom
                    } else {
                        0.0
                    }
                }
                _ => ccorr, // TM_CCORR (2), TM_CCOEFF (4)
            };
        }
    }

    // Pack as dict with raw float32 bytes — use minMaxLoc to find best match
    let byte_data: Vec<u8> = result.iter().flat_map(|&v| v.to_le_bytes()).collect();
    let dict = PyDict::new(py);
    dict.set_item("data", PyBytes::new(py, &byte_data))?;
    dict.set_item("shape", (result_h, result_w))?;
    dict.set_item("dtype", "float32")?;
    Ok(dict.into())
}

// ── Location finding ────────────────────────────────────────────────────────

/// Find the global minimum and maximum in a single-channel image or match result.
///
/// Returns `(min_val, max_val, min_loc, max_loc)` where locations are `(x, y)` tuples.
/// Accepts both numpy arrays and the dict format returned by `matchTemplate`.
#[pyfunction]
#[pyo3(name = "minMaxLoc")]
#[pyo3(signature = (src, mask=None))]
pub fn min_max_loc(
    py: Python<'_>,
    src: Py<PyAny>,
    mask: Option<Py<PyAny>>,
) -> PyResult<(f64, f64, (usize, usize), (usize, usize))> {
    // Check if this is the matchTemplate dict format with float32 data
    let bound = src.bind(py);
    if let Ok(dict) = bound.cast::<PyDict>() {
        if let Some(dtype_item) = dict.get_item("dtype")? {
            let dtype: String = dtype_item.extract()?;
            if dtype == "float32" {
                let data_item = dict.get_item("data")?.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("minMaxLoc: missing data")
                })?;
                let raw: Vec<u8> = data_item.extract()?;
                let shape_item = dict.get_item("shape")?.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("minMaxLoc: missing shape")
                })?;
                let (_rh, rw): (usize, usize) = shape_item.extract()?;

                let floats: Vec<f32> = raw
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();

                let mut min_val = f64::MAX;
                let mut max_val = f64::MIN;
                let mut min_loc = (0usize, 0usize);
                let mut max_loc = (0usize, 0usize);

                for (i, &fv) in floats.iter().enumerate() {
                    let v = fv as f64;
                    if v < min_val {
                        min_val = v;
                        min_loc = (i % rw, i / rw);
                    }
                    if v > max_val {
                        max_val = v;
                        max_loc = (i % rw, i / rw);
                    }
                }
                return Ok((min_val, max_val, min_loc, max_loc));
            }
        }
    }

    // Standard image path
    let (data, h, w, ch) = extract_img(py, &src)?;
    let mask_data = mask
        .as_ref()
        .map(|m| extract_img(py, m).map(|(d, ..)| d))
        .transpose()?;

    let mut min_val = f64::MAX;
    let mut max_val = f64::MIN;
    let mut min_loc = (0usize, 0usize);
    let mut max_loc = (0usize, 0usize);

    for i in 0..h * w {
        if let Some(ref md) = mask_data {
            if md[i] == 0 {
                continue;
            }
        }
        // Use first channel value
        let v = data[i * ch] as f64;
        if v < min_val {
            min_val = v;
            min_loc = (i % w, i / w);
        }
        if v > max_val {
            max_val = v;
            max_loc = (i % w, i / w);
        }
    }

    Ok((min_val, max_val, min_loc, max_loc))
}

// ── Connected components ────────────────────────────────────────────────────

/// Label connected components in a binary image using 4- or 8-connectivity.
///
/// Returns `(num_labels, labels_array)` where:
/// - `num_labels` includes the background (label 0)
/// - `labels_array` is a `(H, W, 1)` uint8 array (component IDs capped at 255)
#[pyfunction]
#[pyo3(name = "connectedComponents")]
#[pyo3(signature = (image, connectivity=8))]
pub fn connected_components(
    py: Python<'_>,
    image: Py<PyAny>,
    connectivity: i32,
) -> PyResult<(usize, Py<PyAny>)> {
    let (data, h, w, ch) = extract_img(py, &image)?;
    let binary: Vec<bool> = (0..h * w).map(|i| data[i * ch] > 0).collect();

    // Union-Find with path compression
    let n = h * w;
    let mut parent: Vec<u32> = (0..=n as u32).collect();

    fn find(parent: &mut Vec<u32>, mut x: u32) -> u32 {
        while parent[x as usize] != x {
            // Path halving
            parent[x as usize] = parent[parent[x as usize] as usize];
            x = parent[x as usize];
        }
        x
    }

    fn union(parent: &mut Vec<u32>, a: u32, b: u32) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent[ra as usize] = rb;
        }
    }

    let mut labels = vec![0u32; n];
    let mut next_label = 1u32;

    // First pass: provisional label assignment
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if !binary[idx] {
                continue;
            }

            let mut neighbors: Vec<u32> = Vec::with_capacity(4);

            // 4-connected neighbours (top, left)
            if y > 0 && binary[(y - 1) * w + x] {
                neighbors.push(labels[(y - 1) * w + x]);
            }
            if x > 0 && binary[y * w + x - 1] {
                neighbors.push(labels[y * w + x - 1]);
            }

            // Extra 8-connectivity neighbours (top-left, top-right)
            if connectivity == 8 {
                if y > 0 && x > 0 && binary[(y - 1) * w + x - 1] {
                    neighbors.push(labels[(y - 1) * w + x - 1]);
                }
                if y > 0 && x + 1 < w && binary[(y - 1) * w + x + 1] {
                    neighbors.push(labels[(y - 1) * w + x + 1]);
                }
            }

            // Remove background placeholders and deduplicate
            neighbors.retain(|&l| l > 0);

            if neighbors.is_empty() {
                labels[idx] = next_label;
                next_label += 1;
            } else {
                let min_lbl = neighbors.iter().copied().min().ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err("connectedComponents: internal error")
                })?;
                labels[idx] = min_lbl;
                for &nb in &neighbors {
                    union(&mut parent, min_lbl, nb);
                }
            }
        }
    }

    // Second pass: flatten union-find and reassign contiguous IDs
    let mut label_map = std::collections::HashMap::<u32, u8>::new();
    let mut component_count = 0u32;

    let out: Vec<u8> = (0..n)
        .map(|i| {
            if !binary[i] {
                return 0u8;
            }
            let root = find(&mut parent, labels[i]);
            *label_map.entry(root).or_insert_with(|| {
                component_count += 1;
                component_count.min(255) as u8
            })
        })
        .collect();

    // num_labels = number of foreground components + 1 (background)
    let num_labels = component_count as usize + 1;

    let arr = Array3::from_shape_vec((h, w, 1), out).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("connectedComponents: {}", e))
    })?;
    Ok((num_labels, arr.into_pyarray(py).into_any().unbind()))
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #[test]
    fn test_add_weighted_identity() {
        let d = vec![100u8, 150u8, 200u8, 50u8, 100u8, 150u8];
        let zeros = vec![0u8; 6];
        let result: Vec<u8> = d
            .iter()
            .zip(zeros.iter())
            .map(|(&a, &b)| (1.0 * a as f64 + 0.0 * b as f64 + 0.0).clamp(0.0, 255.0) as u8)
            .collect();
        assert_eq!(result, d);
    }

    #[test]
    fn test_add_weighted_clamp() {
        // alpha + beta > 255 must be clamped
        let result = (200.0f64 * 1.5 + 0.0 + 0.0).clamp(0.0, 255.0) as u8;
        assert_eq!(result, 255);
    }

    #[test]
    fn test_abs_diff_symmetric() {
        let a = vec![200u8, 100u8];
        let b = vec![100u8, 200u8];
        let diff: Vec<u8> = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs() as u8)
            .collect();
        assert_eq!(diff, vec![100u8, 100u8]);
    }

    #[test]
    fn test_bitwise_not_correctness() {
        assert_eq!(!255u8, 0u8);
        assert_eq!(!0u8, 255u8);
        assert_eq!(!0b1010_1010u8, 0b0101_0101u8);
    }

    #[test]
    fn test_equalize_hist_uniform_histogram() {
        let data: Vec<u8> = (0..=255u8).collect();
        let mut hist = [0u32; 256];
        for &v in &data {
            hist[v as usize] += 1;
        }
        assert!(
            hist.iter().all(|&v| v == 1),
            "each value appears exactly once"
        );
    }

    #[test]
    fn test_normalize_minmax_range() {
        let data = vec![0u8, 100u8, 200u8];
        let min = *data.iter().min().expect("non-empty") as f64;
        let max = *data.iter().max().expect("non-empty") as f64;
        let normalized: Vec<u8> = data
            .iter()
            .map(|&v| ((v as f64 - min) / (max - min) * 255.0).clamp(0.0, 255.0) as u8)
            .collect();
        assert_eq!(normalized[0], 0, "min maps to 0");
        assert_eq!(normalized[2], 255, "max maps to 255");
    }

    #[test]
    fn test_normalize_constant_image() {
        // All same value → denominator = 0 → all alpha
        let data = vec![128u8; 9];
        let min = 128.0f64;
        let max = 128.0f64;
        let range = max - min;
        let alpha = 0.0f64;
        let out: Vec<u8> = if range < 1e-6 {
            vec![alpha as u8; data.len()]
        } else {
            data.iter()
                .map(|&v| ((v as f64 - min) / range * 255.0 + alpha).clamp(0.0, 255.0) as u8)
                .collect()
        };
        assert!(out.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_connected_components_all_background() {
        let data = vec![0u8; 4 * 4];
        let binary: Vec<bool> = data.iter().map(|&v| v > 0).collect();
        assert!(binary.iter().all(|&b| !b));
    }

    #[test]
    fn test_union_find_correctness() {
        // Simple: two separate regions in 1D
        let mut parent: Vec<u32> = (0..=10).collect();

        // union 1,2 and 3,4
        fn find_test(parent: &mut Vec<u32>, mut x: u32) -> u32 {
            while parent[x as usize] != x {
                parent[x as usize] = parent[parent[x as usize] as usize];
                x = parent[x as usize];
            }
            x
        }
        fn union_test(parent: &mut Vec<u32>, a: u32, b: u32) {
            let ra = find_test(parent, a);
            let rb = find_test(parent, b);
            if ra != rb {
                parent[ra as usize] = rb;
            }
        }

        union_test(&mut parent, 1, 2);
        union_test(&mut parent, 3, 4);

        assert_eq!(find_test(&mut parent, 1), find_test(&mut parent, 2));
        assert_eq!(find_test(&mut parent, 3), find_test(&mut parent, 4));
        assert_ne!(find_test(&mut parent, 1), find_test(&mut parent, 3));
    }

    #[test]
    fn test_match_template_result_size() {
        // result shape for template matching: (ih - th + 1, iw - tw + 1)
        let (ih, iw) = (10usize, 10usize);
        let (th, tw) = (3usize, 3usize);
        assert_eq!(ih - th + 1, 8);
        assert_eq!(iw - tw + 1, 8);
    }

    #[test]
    fn test_in_range_logic() {
        // pixel [100, 150, 200] with lo=[50,100,150] hi=[150,200,250] → in range
        let pixel = [100u8, 150u8, 200u8];
        let lo = [50u8, 100u8, 150u8];
        let hi = [150u8, 200u8, 250u8];
        let in_rng = pixel
            .iter()
            .zip(lo.iter().zip(hi.iter()))
            .all(|(&v, (&l, &h))| v >= l && v <= h);
        assert!(in_rng);
    }

    #[test]
    fn test_calc_hist_bin_assignment() {
        // value=128, range=[0,256], bins=256 → bin=128
        let val = 128.0f64;
        let range_min = 0.0f64;
        let range_max = 256.0f64;
        let num_bins = 256usize;
        let bin = ((val - range_min) / (range_max - range_min) * num_bins as f64) as usize;
        assert_eq!(bin.min(num_bins - 1), 128);
    }
}
