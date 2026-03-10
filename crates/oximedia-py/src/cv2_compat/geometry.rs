//! Geometric transformations: resize, flip, rotate, warpAffine, getRotationMatrix2D.

use super::image_io::{extract_img, make_image_output};
use super::numpy_bridge::lerp_u8;
use pyo3::prelude::*;

/// Resize an image to the given size.
///
/// Mirrors `cv2.resize(src, dsize, interpolation=INTER_LINEAR)`.
/// `dsize` is `(width, height)` — same convention as OpenCV.
#[pyfunction]
#[pyo3(name = "resize", signature = (src, dsize, interpolation=1))]
pub fn resize(
    py: Python<'_>,
    src: Py<PyAny>,
    dsize: (usize, usize),
    interpolation: i32,
) -> PyResult<Py<PyAny>> {
    let (data, h, w, ch) = extract_img(py, &src)?;
    let (new_w, new_h) = dsize;

    if new_w == 0 || new_h == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "resize: dsize must be positive",
        ));
    }

    let out = match interpolation {
        0 => resize_nearest(&data, w, h, ch, new_w, new_h),
        _ => resize_bilinear(&data, w, h, ch, new_w, new_h),
    };

    make_image_output(py, out, new_h, new_w, ch)
}

/// Flip an image around an axis.
///
/// Mirrors `cv2.flip(src, flipCode)`:
/// - `0` → vertical (flip around x-axis)
/// - `1` → horizontal (flip around y-axis)
/// - `-1` → both axes
#[pyfunction]
#[pyo3(name = "flip")]
pub fn flip(py: Python<'_>, src: Py<PyAny>, flip_code: i32) -> PyResult<Py<PyAny>> {
    let (data, h, w, ch) = extract_img(py, &src)?;
    let mut out = vec![0u8; h * w * ch];

    for y in 0..h {
        for x in 0..w {
            let src_y = if flip_code == 0 || flip_code == -1 {
                h - 1 - y
            } else {
                y
            };
            let src_x = if flip_code == 1 || flip_code == -1 {
                w - 1 - x
            } else {
                x
            };
            let src_off = (src_y * w + src_x) * ch;
            let dst_off = (y * w + x) * ch;
            out[dst_off..dst_off + ch].copy_from_slice(&data[src_off..src_off + ch]);
        }
    }

    make_image_output(py, out, h, w, ch)
}

/// Rotate an image by a multiple of 90 degrees.
///
/// Mirrors `cv2.rotate(src, rotateCode)`:
/// - `0` → 90° clockwise
/// - `1` → 180°
/// - `2` → 90° counter-clockwise
#[pyfunction]
#[pyo3(name = "rotate")]
pub fn rotate(py: Python<'_>, src: Py<PyAny>, rotate_code: i32) -> PyResult<Py<PyAny>> {
    let (data, h, w, ch) = extract_img(py, &src)?;

    let (out, out_h, out_w) = match rotate_code {
        0 => {
            // 90° clockwise: new dims (h=w_orig, w=h_orig)
            let mut out = vec![0u8; h * w * ch];
            for y in 0..h {
                for x in 0..w {
                    let src_off = (y * w + x) * ch;
                    // dst: row=x, col=(h-1-y), dst_w=h, dst_h=w
                    let dst_off = (x * h + (h - 1 - y)) * ch;
                    out[dst_off..dst_off + ch].copy_from_slice(&data[src_off..src_off + ch]);
                }
            }
            (out, w, h) // output shape: (new_h=w, new_w=h)
        }
        1 => {
            // 180°: dims unchanged
            let mut out = vec![0u8; h * w * ch];
            for y in 0..h {
                for x in 0..w {
                    let src_off = (y * w + x) * ch;
                    let dst_off = ((h - 1 - y) * w + (w - 1 - x)) * ch;
                    out[dst_off..dst_off + ch].copy_from_slice(&data[src_off..src_off + ch]);
                }
            }
            (out, h, w)
        }
        2 => {
            // 90° counter-clockwise: new dims (h=w_orig, w=h_orig)
            let mut out = vec![0u8; h * w * ch];
            for y in 0..h {
                for x in 0..w {
                    let src_off = (y * w + x) * ch;
                    // dst: row=(w-1-x), col=y, dst_w=h, dst_h=w
                    let dst_off = ((w - 1 - x) * h + y) * ch;
                    out[dst_off..dst_off + ch].copy_from_slice(&data[src_off..src_off + ch]);
                }
            }
            (out, w, h)
        }
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "rotate: invalid rotateCode {}; must be 0, 1, or 2",
                rotate_code
            )));
        }
    };

    make_image_output(py, out, out_h, out_w, ch)
}

/// Apply a 2x3 affine transformation matrix to an image.
///
/// Mirrors `cv2.warpAffine(src, M, dsize, flags=INTER_LINEAR)`.
/// `m` is a Python list `[[a, b, c], [d, e, f]]`.
/// `dsize` is `(width, height)`.
#[pyfunction]
#[pyo3(name = "warpAffine", signature = (src, m, dsize, flags=1))]
pub fn warp_affine(
    py: Python<'_>,
    src: Py<PyAny>,
    m: Py<PyAny>,
    dsize: (usize, usize),
    flags: i32,
) -> PyResult<Py<PyAny>> {
    let (data, h, w, ch) = extract_img(py, &src)?;
    let (out_w, out_h) = dsize;

    // Extract 2x3 matrix from Python list [[a,b,c],[d,e,f]]
    let mat = extract_affine_matrix(py, &m)?;

    // Compute inverse of the 2x2 part for inverse mapping
    // M = [[a,b,c],[d,e,f]] so the 2x2 rotation part is [[a,b],[d,e]]
    let (a, b, c) = (mat[0][0], mat[0][1], mat[0][2]);
    let (d, e, f) = (mat[1][0], mat[1][1], mat[1][2]);

    // Inverse of 2x2: [[a,b],[d,e]]^-1 = 1/det * [[e,-b],[-d,a]]
    let det = a * e - b * d;

    let mut out = vec![0u8; out_h * out_w * ch];

    if det.abs() < 1e-10 {
        // Degenerate matrix — return blank image
        return make_image_output(py, out, out_h, out_w, ch);
    }

    let inv_det = 1.0 / det;
    let ia = e * inv_det;
    let ib = -b * inv_det;
    let id = -d * inv_det;
    let ie = a * inv_det;

    for y in 0..out_h {
        for x in 0..out_w {
            let xf = x as f64;
            let yf = y as f64;
            // Inverse map: src_pt = M^-1 * (dst_pt - t)
            let sx = ia * (xf - c) + ib * (yf - f);
            let sy = id * (xf - c) + ie * (yf - f);

            let out_off = (y * out_w + x) * ch;

            match flags {
                0 => {
                    // INTER_NEAREST
                    let sx_i = sx.round() as i64;
                    let sy_i = sy.round() as i64;
                    if sx_i >= 0 && sx_i < w as i64 && sy_i >= 0 && sy_i < h as i64 {
                        let src_off = (sy_i as usize * w + sx_i as usize) * ch;
                        out[out_off..out_off + ch].copy_from_slice(&data[src_off..src_off + ch]);
                    }
                }
                _ => {
                    // INTER_LINEAR (bilinear)
                    if sx >= 0.0 && sx < (w - 1) as f64 && sy >= 0.0 && sy < (h - 1) as f64 {
                        let sx0 = sx.floor() as usize;
                        let sy0 = sy.floor() as usize;
                        let fx = (sx - sx.floor()) as f32;
                        let fy = (sy - sy.floor()) as f32;
                        for c_idx in 0..ch {
                            let p00 = data[(sy0 * w + sx0) * ch + c_idx];
                            let p10 = data[(sy0 * w + sx0 + 1) * ch + c_idx];
                            let p01 = data[((sy0 + 1) * w + sx0) * ch + c_idx];
                            let p11 = data[((sy0 + 1) * w + sx0 + 1) * ch + c_idx];
                            let top = lerp_u8(p00, p10, fx);
                            let bot = lerp_u8(p01, p11, fx);
                            out[out_off + c_idx] = lerp_u8(top, bot, fy);
                        }
                    } else if sx >= 0.0 && sx < w as f64 && sy >= 0.0 && sy < h as f64 {
                        // Edge — nearest fallback
                        let sx_i = sx as usize;
                        let sy_i = sy as usize;
                        let src_off = (sy_i * w + sx_i) * ch;
                        out[out_off..out_off + ch].copy_from_slice(&data[src_off..src_off + ch]);
                    }
                }
            }
        }
    }

    make_image_output(py, out, out_h, out_w, ch)
}

/// Compute the 2x3 affine rotation matrix.
///
/// Mirrors `cv2.getRotationMatrix2D(center, angle, scale)`.
/// Returns a `[[cos*scale, sin*scale, tx], [-sin*scale, cos*scale, ty]]` list.
#[pyfunction]
#[pyo3(name = "getRotationMatrix2D")]
pub fn get_rotation_matrix_2d(
    py: Python<'_>,
    center: (f64, f64),
    angle: f64,
    scale: f64,
) -> PyResult<Py<PyAny>> {
    let rad = angle * std::f64::consts::PI / 180.0;
    let cos_a = rad.cos() * scale;
    let sin_a = rad.sin() * scale;
    let (cx, cy) = center;

    // tx = (1 - cos)*cx - sin*cy
    // ty = sin*cx + (1 - cos)*cy
    let tx = (1.0 - cos_a) * cx - sin_a * cy;
    let ty = sin_a * cx + (1.0 - cos_a) * cy;

    let row0 = pyo3::types::PyList::new(py, [cos_a, sin_a, tx])?;
    let row1 = pyo3::types::PyList::new(py, [-sin_a, cos_a, ty])?;
    let mat = pyo3::types::PyList::new(py, [row0, row1])?;
    Ok(mat.into())
}

// ── Private helpers ───────────────────────────────────────────────────────────

pub(crate) fn resize_nearest(
    data: &[u8],
    w: usize,
    h: usize,
    ch: usize,
    new_w: usize,
    new_h: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; new_h * new_w * ch];
    for y in 0..new_h {
        for x in 0..new_w {
            let src_x = (x * w / new_w).min(w - 1);
            let src_y = (y * h / new_h).min(h - 1);
            let src_off = (src_y * w + src_x) * ch;
            let dst_off = (y * new_w + x) * ch;
            out[dst_off..dst_off + ch].copy_from_slice(&data[src_off..src_off + ch]);
        }
    }
    out
}

pub(crate) fn resize_bilinear(
    data: &[u8],
    w: usize,
    h: usize,
    ch: usize,
    new_w: usize,
    new_h: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; new_h * new_w * ch];
    let x_scale = w as f32 / new_w as f32;
    let y_scale = h as f32 / new_h as f32;

    for y in 0..new_h {
        for x in 0..new_w {
            // Map destination pixel to source space
            let sx = (x as f32 + 0.5) * x_scale - 0.5;
            let sy = (y as f32 + 0.5) * y_scale - 0.5;

            let sx0 = (sx.floor() as isize).clamp(0, w as isize - 1) as usize;
            let sy0 = (sy.floor() as isize).clamp(0, h as isize - 1) as usize;
            let sx1 = (sx0 + 1).min(w - 1);
            let sy1 = (sy0 + 1).min(h - 1);

            let fx = (sx - sx.floor()).clamp(0.0, 1.0);
            let fy = (sy - sy.floor()).clamp(0.0, 1.0);

            let dst_off = (y * new_w + x) * ch;
            for c_idx in 0..ch {
                let p00 = data[(sy0 * w + sx0) * ch + c_idx];
                let p10 = data[(sy0 * w + sx1) * ch + c_idx];
                let p01 = data[(sy1 * w + sx0) * ch + c_idx];
                let p11 = data[(sy1 * w + sx1) * ch + c_idx];
                let top = lerp_u8(p00, p10, fx);
                let bot = lerp_u8(p01, p11, fx);
                out[dst_off + c_idx] = lerp_u8(top, bot, fy);
            }
        }
    }
    out
}

/// Pure-Rust flip helper exposed for testing.
/// `flip_code`: 0 = vertical, 1 = horizontal, -1 = both.
#[allow(dead_code)]
pub(crate) fn flip_image(data: &[u8], h: usize, w: usize, ch: usize, flip_code: i32) -> Vec<u8> {
    let mut out = vec![0u8; h * w * ch];
    for y in 0..h {
        for x in 0..w {
            let src_y = if flip_code == 0 || flip_code == -1 {
                h - 1 - y
            } else {
                y
            };
            let src_x = if flip_code == 1 || flip_code == -1 {
                w - 1 - x
            } else {
                x
            };
            let src_off = (src_y * w + src_x) * ch;
            let dst_off = (y * w + x) * ch;
            out[dst_off..dst_off + ch].copy_from_slice(&data[src_off..src_off + ch]);
        }
    }
    out
}

/// Pure-Rust rotate helper exposed for testing.
/// `rotate_code`: 0 = 90° CW, 1 = 180°, 2 = 90° CCW.
/// Returns (pixel_data, out_h, out_w).
#[allow(dead_code)]
pub(crate) fn rotate_image(
    data: &[u8],
    h: usize,
    w: usize,
    ch: usize,
    rotate_code: i32,
) -> (Vec<u8>, usize, usize) {
    match rotate_code {
        0 => {
            let mut out = vec![0u8; h * w * ch];
            for y in 0..h {
                for x in 0..w {
                    let src_off = (y * w + x) * ch;
                    let dst_off = (x * h + (h - 1 - y)) * ch;
                    out[dst_off..dst_off + ch].copy_from_slice(&data[src_off..src_off + ch]);
                }
            }
            (out, w, h)
        }
        1 => {
            let mut out = vec![0u8; h * w * ch];
            for y in 0..h {
                for x in 0..w {
                    let src_off = (y * w + x) * ch;
                    let dst_off = ((h - 1 - y) * w + (w - 1 - x)) * ch;
                    out[dst_off..dst_off + ch].copy_from_slice(&data[src_off..src_off + ch]);
                }
            }
            (out, h, w)
        }
        _ => {
            // 90° CCW
            let mut out = vec![0u8; h * w * ch];
            for y in 0..h {
                for x in 0..w {
                    let src_off = (y * w + x) * ch;
                    let dst_off = ((w - 1 - x) * h + y) * ch;
                    out[dst_off..dst_off + ch].copy_from_slice(&data[src_off..src_off + ch]);
                }
            }
            (out, w, h)
        }
    }
}

fn extract_affine_matrix(py: Python<'_>, obj: &Py<PyAny>) -> PyResult<[[f64; 3]; 2]> {
    let bound = obj.bind(py);
    let row0 = bound.get_item(0)?;
    let row1 = bound.get_item(1)?;

    let r0: Vec<f64> = row0.extract()?;
    let r1: Vec<f64> = row1.extract()?;

    if r0.len() < 3 || r1.len() < 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "warpAffine: M must be a 2x3 matrix [[a,b,c],[d,e,f]]",
        ));
    }

    Ok([[r0[0], r0[1], r0[2]], [r1[0], r1[1], r1[2]]])
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── flip_image ─────────────────────────────────────────────────────────────

    #[test]
    fn test_flip_horizontal_swaps_columns() {
        // 2x2 BGR image
        // Row 0: [10,20,30] [40,50,60]
        // Row 1: [70,80,90] [100,110,120]
        // After horizontal flip (flip_code=1):
        // Row 0: [40,50,60] [10,20,30]
        let data = vec![10u8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
        let result = flip_image(&data, 2, 2, 3, 1);
        assert_eq!(
            &result[0..3],
            &[40, 50, 60],
            "row 0 col 0 should be old col 1"
        );
        assert_eq!(
            &result[3..6],
            &[10, 20, 30],
            "row 0 col 1 should be old col 0"
        );
        assert_eq!(
            &result[6..9],
            &[100, 110, 120],
            "row 1 col 0 should be old col 1"
        );
        assert_eq!(
            &result[9..12],
            &[70, 80, 90],
            "row 1 col 1 should be old col 0"
        );
    }

    #[test]
    fn test_flip_vertical_swaps_rows() {
        let data = vec![10u8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
        let result = flip_image(&data, 2, 2, 3, 0);
        // Row 0 should become old row 1
        assert_eq!(
            &result[0..3],
            &[70, 80, 90],
            "row 0 col 0 should be old row 1 col 0"
        );
        assert_eq!(
            &result[3..6],
            &[100, 110, 120],
            "row 0 col 1 should be old row 1 col 1"
        );
        assert_eq!(
            &result[6..9],
            &[10, 20, 30],
            "row 1 col 0 should be old row 0 col 0"
        );
        assert_eq!(
            &result[9..12],
            &[40, 50, 60],
            "row 1 col 1 should be old row 0 col 1"
        );
    }

    #[test]
    fn test_flip_both_axes() {
        // flip_code=-1: both horizontal and vertical
        let data = vec![1u8, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0];
        let result = flip_image(&data, 2, 2, 3, -1);
        // pixel [0,0] maps to original [1,1]
        assert_eq!(&result[0..3], &[4, 0, 0]);
        // pixel [1,1] maps to original [0,0]
        assert_eq!(&result[9..12], &[1, 0, 0]);
    }

    #[test]
    fn test_flip_single_pixel_unchanged() {
        let data = vec![42u8, 13, 200];
        let result = flip_image(&data, 1, 1, 3, 1);
        assert_eq!(
            &result[0..3],
            &[42, 13, 200],
            "1x1 image should be unchanged by any flip"
        );
    }

    // ── rotate_image ──────────────────────────────────────────────────────────

    #[test]
    fn test_rotate_180_reverses_order() {
        // 2x2 image with unique BGR pixels
        let data = vec![1u8, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0];
        let (result, out_h, out_w) = rotate_image(&data, 2, 2, 3, 1);
        assert_eq!(out_h, 2, "180° rotation keeps same height");
        assert_eq!(out_w, 2, "180° rotation keeps same width");
        // Last pixel of input becomes first pixel of output
        assert_eq!(&result[0..3], &[4, 0, 0], "first pixel after 180°");
        assert_eq!(&result[9..12], &[1, 0, 0], "last pixel after 180°");
    }

    #[test]
    fn test_rotate_90cw_swaps_dimensions() {
        // 3x2 image (h=3, w=2) rotated 90° CW → h=2, w=3
        let data: Vec<u8> = (1u8..=18).collect(); // 3*2*3 bytes
        let (result, out_h, out_w) = rotate_image(&data, 3, 2, 3, 0);
        assert_eq!(out_h, 2, "90° CW rotation: new height == old width");
        assert_eq!(out_w, 3, "90° CW rotation: new width == old height");
        assert_eq!(result.len(), 2 * 3 * 3, "total pixel count unchanged");
    }

    #[test]
    fn test_rotate_90ccw_swaps_dimensions() {
        let data: Vec<u8> = (1u8..=18).collect();
        let (result, out_h, out_w) = rotate_image(&data, 3, 2, 3, 2);
        assert_eq!(out_h, 2);
        assert_eq!(out_w, 3);
        assert_eq!(result.len(), 2 * 3 * 3);
    }

    #[test]
    fn test_rotate_360_is_identity() {
        // Two 180° rotations == identity
        let data = vec![10u8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
        let (half, _, _) = rotate_image(&data, 2, 2, 3, 1);
        let (full, _, _) = rotate_image(&half, 2, 2, 3, 1);
        assert_eq!(full, data, "two 180° rotations should restore original");
    }

    // ── resize_nearest ────────────────────────────────────────────────────────

    #[test]
    fn test_resize_nearest_same_size_is_identity() {
        let data = vec![100u8; 4 * 4 * 3];
        let result = resize_nearest(&data, 4, 4, 3, 4, 4);
        assert_eq!(result, data, "resizing to same size should preserve data");
    }

    #[test]
    fn test_resize_nearest_2x_upscale() {
        // 1x2 image → 1x4 (nearest neighbour)
        let data = vec![255u8, 0, 0, 128, 0, 0]; // 1x2 BGR
        let result = resize_nearest(&data, 2, 1, 3, 4, 1);
        assert_eq!(result.len(), 1 * 4 * 3, "output should have 4 pixels");
    }

    #[test]
    fn test_resize_nearest_solid_colour_preserved() {
        // Any solid-colour image resized to any target should remain solid
        let data = vec![42u8; 8 * 8 * 3];
        let result = resize_nearest(&data, 8, 8, 3, 5, 5);
        assert!(
            result.iter().all(|&v| v == 42),
            "solid colour should survive nearest-neighbour resize"
        );
    }

    // ── resize_bilinear ───────────────────────────────────────────────────────

    #[test]
    fn test_resize_bilinear_solid_colour_preserved() {
        let data = vec![200u8; 6 * 6 * 3];
        let result = resize_bilinear(&data, 6, 6, 3, 6, 6);
        assert!(
            result.iter().all(|&v| v == 200),
            "solid colour should survive bilinear resize"
        );
    }

    #[test]
    fn test_resize_bilinear_output_dimensions() {
        let data = vec![0u8; 10 * 10 * 1];
        let result = resize_bilinear(&data, 10, 10, 1, 7, 5);
        assert_eq!(
            result.len(),
            7 * 5 * 1,
            "output size matches requested dimensions"
        );
    }
}
