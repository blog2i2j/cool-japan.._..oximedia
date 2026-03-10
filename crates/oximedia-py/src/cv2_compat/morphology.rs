//! Morphological operations: erode, dilate, morphologyEx, getStructuringElement.

use super::image_io::{extract_img, make_image_output};
use pyo3::prelude::*;

/// Create a structuring element for morphological operations.
///
/// Mirrors `cv2.getStructuringElement(shape, ksize)`.
/// Returns a Python list of lists (0 or 1 values).
/// - `shape` 0 = MORPH_RECT, 1 = MORPH_CROSS, 2 = MORPH_ELLIPSE
#[pyfunction]
#[pyo3(name = "getStructuringElement")]
pub fn get_structuring_element(
    py: Python<'_>,
    shape: i32,
    ksize: (usize, usize),
) -> PyResult<Py<PyAny>> {
    let (kw, kh) = ksize;
    if kw == 0 || kh == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "ksize must be positive",
        ));
    }

    let mut kernel = vec![vec![0u8; kw]; kh];
    let cx = kw as f64 / 2.0 - 0.5;
    let cy = kh as f64 / 2.0 - 0.5;
    let rx = (kw as f64 / 2.0).max(1.0);
    let ry = (kh as f64 / 2.0).max(1.0);

    for y in 0..kh {
        for x in 0..kw {
            kernel[y][x] = match shape {
                0 => 1, // MORPH_RECT
                1 => {
                    // MORPH_CROSS
                    if x == kw / 2 || y == kh / 2 {
                        1
                    } else {
                        0
                    }
                }
                2 => {
                    // MORPH_ELLIPSE
                    let dx = (x as f64 - cx) / rx;
                    let dy = (y as f64 - cy) / ry;
                    if dx * dx + dy * dy <= 1.0 {
                        1
                    } else {
                        0
                    }
                }
                _ => 1,
            };
        }
    }

    // Return as Python list of lists
    let rows: Vec<Py<PyAny>> = kernel
        .into_iter()
        .map(|row| pyo3::types::PyList::new(py, row.iter().map(|&v| v as i32)).map(|l| l.into()))
        .collect::<PyResult<_>>()?;
    let result = pyo3::types::PyList::new(py, rows)?;
    Ok(result.into())
}

/// Erode an image using a structuring element.
///
/// Mirrors `cv2.erode(src, kernel, iterations=1)`.
#[pyfunction]
#[pyo3(name = "erode", signature = (src, kernel, iterations=1))]
pub fn erode(
    py: Python<'_>,
    src: Py<PyAny>,
    kernel: Py<PyAny>,
    iterations: i32,
) -> PyResult<Py<PyAny>> {
    let (mut data, h, w, ch) = extract_img(py, &src)?;
    let kern = extract_kernel_binary(py, &kernel)?;

    for _ in 0..iterations.max(1) {
        data = apply_erosion(&data, w, h, ch, &kern);
    }

    make_image_output(py, data, h, w, ch)
}

/// Dilate an image using a structuring element.
///
/// Mirrors `cv2.dilate(src, kernel, iterations=1)`.
#[pyfunction]
#[pyo3(name = "dilate", signature = (src, kernel, iterations=1))]
pub fn dilate(
    py: Python<'_>,
    src: Py<PyAny>,
    kernel: Py<PyAny>,
    iterations: i32,
) -> PyResult<Py<PyAny>> {
    let (mut data, h, w, ch) = extract_img(py, &src)?;
    let kern = extract_kernel_binary(py, &kernel)?;

    for _ in 0..iterations.max(1) {
        data = apply_dilation(&data, w, h, ch, &kern);
    }

    make_image_output(py, data, h, w, ch)
}

/// Apply a morphological operation to an image.
///
/// Mirrors `cv2.morphologyEx(src, op, kernel, iterations=1)`.
/// Supported ops: MORPH_OPEN=2, MORPH_CLOSE=3, MORPH_GRADIENT=4,
///                MORPH_TOPHAT=5, MORPH_BLACKHAT=6, MORPH_ERODE=0, MORPH_DILATE=1.
#[pyfunction]
#[pyo3(name = "morphologyEx", signature = (src, op, kernel, iterations=1))]
pub fn morphology_ex(
    py: Python<'_>,
    src: Py<PyAny>,
    op: i32,
    kernel: Py<PyAny>,
    iterations: i32,
) -> PyResult<Py<PyAny>> {
    let (data, h, w, ch) = extract_img(py, &src)?;
    let kern = extract_kernel_binary(py, &kernel)?;
    let iters = iterations.max(1);

    let out = match op {
        0 => {
            // MORPH_ERODE
            let mut d = data.clone();
            for _ in 0..iters {
                d = apply_erosion(&d, w, h, ch, &kern);
            }
            d
        }
        1 => {
            // MORPH_DILATE
            let mut d = data.clone();
            for _ in 0..iters {
                d = apply_dilation(&d, w, h, ch, &kern);
            }
            d
        }
        2 => {
            // MORPH_OPEN: erode then dilate
            let mut d = data.clone();
            for _ in 0..iters {
                d = apply_erosion(&d, w, h, ch, &kern);
            }
            for _ in 0..iters {
                d = apply_dilation(&d, w, h, ch, &kern);
            }
            d
        }
        3 => {
            // MORPH_CLOSE: dilate then erode
            let mut d = data.clone();
            for _ in 0..iters {
                d = apply_dilation(&d, w, h, ch, &kern);
            }
            for _ in 0..iters {
                d = apply_erosion(&d, w, h, ch, &kern);
            }
            d
        }
        4 => {
            // MORPH_GRADIENT: dilate - erode
            let eroded = apply_erosion(&data, w, h, ch, &kern);
            let dilated = apply_dilation(&data, w, h, ch, &kern);
            dilated
                .iter()
                .zip(eroded.iter())
                .map(|(&d, &e)| d.saturating_sub(e))
                .collect()
        }
        5 => {
            // MORPH_TOPHAT: src - open(src)
            let mut opened = data.clone();
            for _ in 0..iters {
                opened = apply_erosion(&opened, w, h, ch, &kern);
            }
            for _ in 0..iters {
                opened = apply_dilation(&opened, w, h, ch, &kern);
            }
            data.iter()
                .zip(opened.iter())
                .map(|(&s, &o)| s.saturating_sub(o))
                .collect()
        }
        6 => {
            // MORPH_BLACKHAT: close(src) - src
            let mut closed = data.clone();
            for _ in 0..iters {
                closed = apply_dilation(&closed, w, h, ch, &kern);
            }
            for _ in 0..iters {
                closed = apply_erosion(&closed, w, h, ch, &kern);
            }
            closed
                .iter()
                .zip(data.iter())
                .map(|(&c, &s)| c.saturating_sub(s))
                .collect()
        }
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "morphologyEx: unsupported op {}",
                op
            )));
        }
    };

    make_image_output(py, out, h, w, ch)
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Extract binary kernel from Python list of lists.
/// Returns (kernel_data, kh, kw) where kernel_data[row*kw+col] is bool.
fn extract_kernel_binary(py: Python<'_>, obj: &Py<PyAny>) -> PyResult<(Vec<bool>, usize, usize)> {
    let bound = obj.bind(py);

    // Handle numpy array-like: has shape and tobytes
    if bound.hasattr("shape")? {
        let shape_any = bound.getattr("shape")?;
        let (kh, kw): (usize, usize) = shape_any.extract()?;
        let flat: Vec<u8> = bound.call_method0("tobytes")?.extract()?;
        let kern: Vec<bool> = flat.iter().map(|&v| v != 0).collect();
        return Ok((kern, kh, kw));
    }

    // Handle Python list of lists
    let rows: Vec<Py<PyAny>> = bound.extract()?;
    let kh = rows.len();
    if kh == 0 {
        return Ok((vec![true], 1, 1));
    }
    let row0: Vec<i32> = rows[0].bind(py).extract()?;
    let kw = row0.len();
    let mut kern = Vec::with_capacity(kh * kw);
    for v in &row0 {
        kern.push(*v != 0);
    }
    for row_obj in rows.iter().skip(1) {
        let row: Vec<i32> = row_obj.bind(py).extract()?;
        for v in &row {
            kern.push(*v != 0);
        }
    }
    Ok((kern, kh, kw))
}

pub(crate) fn apply_erosion(
    data: &[u8],
    w: usize,
    h: usize,
    ch: usize,
    kern: &(Vec<bool>, usize, usize),
) -> Vec<u8> {
    let (ref k, kh, kw) = *kern;
    let half_h = kh / 2;
    let half_w = kw / 2;
    let mut out = vec![0u8; h * w * ch];

    for y in 0..h {
        for x in 0..w {
            let dst_off = (y * w + x) * ch;
            for c in 0..ch {
                let mut min_val = 255u8;
                for ky in 0..kh {
                    let sy_i = y as isize + ky as isize - half_h as isize;
                    let sy = sy_i.clamp(0, h as isize - 1) as usize;
                    for kx in 0..kw {
                        if !k[ky * kw + kx] {
                            continue;
                        }
                        let sx_i = x as isize + kx as isize - half_w as isize;
                        let sx = sx_i.clamp(0, w as isize - 1) as usize;
                        let v = data[(sy * w + sx) * ch + c];
                        if v < min_val {
                            min_val = v;
                        }
                    }
                }
                out[dst_off + c] = min_val;
            }
        }
    }
    out
}

pub(crate) fn apply_dilation(
    data: &[u8],
    w: usize,
    h: usize,
    ch: usize,
    kern: &(Vec<bool>, usize, usize),
) -> Vec<u8> {
    let (ref k, kh, kw) = *kern;
    let half_h = kh / 2;
    let half_w = kw / 2;
    let mut out = vec![0u8; h * w * ch];

    for y in 0..h {
        for x in 0..w {
            let dst_off = (y * w + x) * ch;
            for c in 0..ch {
                let mut max_val = 0u8;
                for ky in 0..kh {
                    let sy_i = y as isize + ky as isize - half_h as isize;
                    let sy = sy_i.clamp(0, h as isize - 1) as usize;
                    for kx in 0..kw {
                        if !k[ky * kw + kx] {
                            continue;
                        }
                        let sx_i = x as isize + kx as isize - half_w as isize;
                        let sx = sx_i.clamp(0, w as isize - 1) as usize;
                        let v = data[(sy * w + sx) * ch + c];
                        if v > max_val {
                            max_val = v;
                        }
                    }
                }
                out[dst_off + c] = max_val;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a 3x3 all-ones rectangular kernel (1-channel).
    fn rect_3x3() -> (Vec<bool>, usize, usize) {
        (vec![true; 9], 3, 3)
    }

    /// Build a 1x1 identity kernel.
    fn identity_1x1() -> (Vec<bool>, usize, usize) {
        (vec![true], 1, 1)
    }

    // ── apply_erosion ──────────────────────────────────────────────────────────

    #[test]
    fn test_erode_removes_isolated_bright_pixel() {
        // 5x5 image, single bright pixel in centre surrounded by dark
        let mut data = vec![0u8; 5 * 5];
        data[12] = 255; // centre (row=2, col=2)
        let kern = rect_3x3();
        let result = apply_erosion(&data, 5, 5, 1, &kern);
        assert_eq!(result[12], 0, "isolated bright pixel should be eroded away");
    }

    #[test]
    fn test_erode_identity_kernel_preserves_image() {
        // A 1x1 kernel should leave the image unchanged
        let data: Vec<u8> = (0u8..25).collect();
        let kern = identity_1x1();
        let result = apply_erosion(&data, 5, 5, 1, &kern);
        assert_eq!(result, data, "identity kernel should not change the image");
    }

    #[test]
    fn test_erode_all_bright_image_stays_bright() {
        // Eroding a fully bright image with any kernel gives a fully bright image
        let data = vec![255u8; 5 * 5];
        let kern = rect_3x3();
        let result = apply_erosion(&data, 5, 5, 1, &kern);
        assert!(
            result.iter().all(|&v| v == 255),
            "eroding all-255 image should give all-255"
        );
    }

    #[test]
    fn test_erode_all_dark_image_stays_dark() {
        let data = vec![0u8; 5 * 5];
        let kern = rect_3x3();
        let result = apply_erosion(&data, 5, 5, 1, &kern);
        assert!(
            result.iter().all(|&v| v == 0),
            "eroding all-0 image should give all-0"
        );
    }

    // ── apply_dilation ────────────────────────────────────────────────────────

    #[test]
    fn test_dilate_expands_isolated_bright_pixel() {
        // 5x5 image with single bright centre pixel — dilation should spread to neighbours
        let mut data = vec![0u8; 5 * 5];
        data[12] = 255; // centre (row=2, col=2)
        let kern = rect_3x3();
        let result = apply_dilation(&data, 5, 5, 1, &kern);
        assert_eq!(
            result[12], 255,
            "centre pixel should stay 255 after dilation"
        );
        assert_eq!(result[11], 255, "left neighbour should be dilated to 255");
        assert_eq!(result[13], 255, "right neighbour should be dilated to 255");
        assert_eq!(result[7], 255, "top neighbour should be dilated to 255");
        assert_eq!(result[17], 255, "bottom neighbour should be dilated to 255");
    }

    #[test]
    fn test_dilate_identity_kernel_preserves_image() {
        let data: Vec<u8> = (0u8..25).collect();
        let kern = identity_1x1();
        let result = apply_dilation(&data, 5, 5, 1, &kern);
        assert_eq!(result, data, "identity kernel should not change the image");
    }

    #[test]
    fn test_dilate_all_dark_stays_dark() {
        let data = vec![0u8; 5 * 5];
        let kern = rect_3x3();
        let result = apply_dilation(&data, 5, 5, 1, &kern);
        assert!(
            result.iter().all(|&v| v == 0),
            "dilating all-0 image should give all-0"
        );
    }

    // ── open = erode then dilate ───────────────────────────────────────────────

    #[test]
    fn test_open_removes_isolated_pixel_restores_large_region() {
        // 10x10 image with a large bright block in the centre
        let mut data = vec![0u8; 10 * 10];
        for y in 2..8usize {
            for x in 2..8usize {
                data[y * 10 + x] = 255;
            }
        }
        let kern = rect_3x3();
        let eroded = apply_erosion(&data, 10, 10, 1, &kern);
        let opened = apply_dilation(&eroded, 10, 10, 1, &kern);
        // The centre of the large region should survive opening
        assert_eq!(
            opened[5 * 10 + 5],
            255,
            "centre of large region should survive open"
        );
    }

    #[test]
    fn test_open_removes_isolated_pixel() {
        // 7x7 image with a single isolated pixel — opening should remove it
        let mut data = vec![0u8; 7 * 7];
        data[3 * 7 + 3] = 255; // centre pixel only
        let kern = rect_3x3();
        let eroded = apply_erosion(&data, 7, 7, 1, &kern);
        let opened = apply_dilation(&eroded, 7, 7, 1, &kern);
        assert_eq!(
            opened[3 * 7 + 3],
            0,
            "isolated pixel should be removed by opening"
        );
    }

    // ── multi-channel ─────────────────────────────────────────────────────────

    #[test]
    fn test_erode_multi_channel() {
        // 3x3 3-channel image, only one pixel bright in channel 0
        let mut data = vec![0u8; 3 * 3 * 3];
        data[4 * 3] = 255; // centre pixel, channel 0
        let kern = rect_3x3();
        let result = apply_erosion(&data, 3, 3, 3, &kern);
        assert_eq!(
            result[4 * 3],
            0,
            "isolated bright channel 0 pixel should be eroded"
        );
    }

    #[test]
    fn test_dilate_multi_channel() {
        let mut data = vec![0u8; 3 * 3 * 3];
        data[4 * 3 + 1] = 255; // centre pixel, channel 1
        let kern = rect_3x3();
        let result = apply_dilation(&data, 3, 3, 3, &kern);
        assert_eq!(
            result[4 * 3 + 1],
            255,
            "centre channel 1 should be 255 after dilation"
        );
        // A neighbour's channel 1 should also be dilated
        assert_eq!(
            result[3 * 3 + 1],
            255,
            "left neighbour channel 1 should be dilated"
        );
    }
}
