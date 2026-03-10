//! Optical flow: calcOpticalFlowPyrLK (Lucas-Kanade sparse optical flow).

use super::image_io::extract_img;
use pyo3::prelude::*;

/// Calculate sparse optical flow using Lucas-Kanade method with pyramids.
///
/// Mirrors `cv2.calcOpticalFlowPyrLK(prevImg, nextImg, prevPts, nextPts, winSize, maxLevel, ...)`.
///
/// Returns `(nextPts, status, err)`:
/// - `nextPts`: list of (x, y) float tuples with tracked positions
/// - `status`: list of ints (1 = tracked, 0 = lost)
/// - `err`: list of f32 residuals (or None)
#[pyfunction]
#[pyo3(name = "calcOpticalFlowPyrLK", signature = (
    prev_img,
    next_img,
    prev_pts,
    next_pts,
    win_size=(21, 21),
    max_level=3,
    criteria=None,
    flags=0,
    min_eig_threshold=1e-4,
))]
#[allow(clippy::too_many_arguments)]
pub fn calc_optical_flow_pyr_lk(
    py: Python<'_>,
    prev_img: Py<PyAny>,
    next_img: Py<PyAny>,
    prev_pts: Py<PyAny>,
    next_pts: Option<Py<PyAny>>,
    win_size: (usize, usize),
    max_level: usize,
    criteria: Option<Py<PyAny>>,
    flags: i32,
    min_eig_threshold: f64,
) -> PyResult<Py<PyAny>> {
    let _ = (next_pts, criteria, flags);
    let (prev_data, h, w, ch) = extract_img(py, &prev_img)?;
    let (next_data, _, _, _) = extract_img(py, &next_img)?;

    let prev_gray = to_gray_f32(&prev_data, w, h, ch);
    let next_gray = to_gray_f32(&next_data, w, h, ch);

    let input_pts = extract_points(py, &prev_pts)?;
    let (half_w, half_h) = (win_size.0 / 2, win_size.1 / 2);

    let mut out_pts: Vec<(f32, f32)> = Vec::with_capacity(input_pts.len());
    let mut status: Vec<i32> = Vec::with_capacity(input_pts.len());
    let mut err: Vec<f32> = Vec::with_capacity(input_pts.len());

    // Build image pyramid
    let max_level = max_level.min(4);
    let prev_pyr = build_pyramid(&prev_gray, w, h, max_level);
    let next_pyr = build_pyramid(&next_gray, w, h, max_level);

    for &(px, py_coord) in &input_pts {
        // Coarse-to-fine Lucas-Kanade
        let mut disp_x = 0.0f32;
        let mut disp_y = 0.0f32;
        let mut tracked = true;

        for level in (0..=max_level).rev() {
            let scale = (1usize << level) as f32;
            let lw = (w >> level).max(1);
            let lh = (h >> level).max(1);
            let lprev = &prev_pyr[level];
            let lnext = &next_pyr[level];

            let cx = px / scale;
            let cy = py_coord / scale;
            disp_x /= if level < max_level { 2.0 } else { 1.0 };
            disp_y /= if level < max_level { 2.0 } else { 1.0 };

            // Iterative Lucas-Kanade (up to 20 iterations)
            for _iter in 0..20 {
                let nx = cx + disp_x;
                let ny = cy + disp_y;

                // Compute image gradients and compute H matrix in window
                let mut h11 = 0.0f32;
                let mut h12 = 0.0f32;
                let mut h22 = 0.0f32;
                let mut b1 = 0.0f32;
                let mut b2 = 0.0f32;

                for wy in 0..win_size.1 {
                    for wx in 0..win_size.0 {
                        let ix = (cx as isize + wx as isize - half_w as isize)
                            .clamp(1, lw as isize - 2) as usize;
                        let iy = (cy as isize + wy as isize - half_h as isize)
                            .clamp(1, lh as isize - 2) as usize;
                        let jx = (nx + wx as f32 - half_w as f32).clamp(1.0, lw as f32 - 2.0);
                        let jy = (ny + wy as f32 - half_h as f32).clamp(1.0, lh as f32 - 2.0);

                        let it = interpolate(lprev, lw, ix as f32, iy as f32);
                        let jt = interpolate(lnext, lw, jx, jy);
                        let diff = it - jt;

                        // Spatial gradient of I (prev)
                        let gx = (lprev[iy * lw + (ix + 1).min(lw - 1)]
                            - lprev[iy * lw + ix.saturating_sub(1)])
                            / 2.0;
                        let gy = (lprev[(iy + 1).min(lh - 1) * lw + ix]
                            - lprev[iy.saturating_sub(1) * lw + ix])
                            / 2.0;

                        h11 += gx * gx;
                        h12 += gx * gy;
                        h22 += gy * gy;
                        b1 += gx * diff;
                        b2 += gy * diff;
                    }
                }

                let det = h11 * h22 - h12 * h12;
                if det.abs() < min_eig_threshold as f32 {
                    tracked = false;
                    break;
                }

                let inv_det = 1.0 / det;
                let delta_x = (h22 * b1 - h12 * b2) * inv_det;
                let delta_y = (h11 * b2 - h12 * b1) * inv_det;
                disp_x += delta_x;
                disp_y += delta_y;

                if delta_x * delta_x + delta_y * delta_y < 0.001 {
                    break;
                }
            }

            if level == 0 {
                // Scale up to full resolution
                disp_x *= scale;
                disp_y *= scale;
            } else {
                disp_x *= 2.0;
                disp_y *= 2.0;
            }
        }

        let final_x = px + disp_x;
        let final_y = py_coord + disp_y;

        // Check bounds
        if !tracked || final_x < 0.0 || final_y < 0.0 || final_x >= w as f32 || final_y >= h as f32
        {
            out_pts.push((px, py_coord));
            status.push(0);
            err.push(0.0);
        } else {
            // Compute residual error
            let residual = compute_patch_error(
                &prev_gray, &next_gray, w, h, px, py_coord, final_x, final_y, half_w, half_h,
            );
            out_pts.push((final_x, final_y));
            status.push(1);
            err.push(residual);
        }
    }

    // Build Python return value
    let py_pts = pyo3::types::PyList::empty(py);
    for (x, y) in &out_pts {
        py_pts.append(pyo3::types::PyTuple::new(py, [*x, *y])?)?;
    }
    let py_status = pyo3::types::PyList::new(py, &status)?;
    let py_err = pyo3::types::PyList::new(py, &err)?;

    let result =
        pyo3::types::PyTuple::new(py, [py_pts.as_any(), py_status.as_any(), py_err.as_any()])?;
    Ok(result.into())
}

// ── Private helpers ───────────────────────────────────────────────────────────

fn to_gray_f32(data: &[u8], w: usize, h: usize, ch: usize) -> Vec<f32> {
    let mut gray = vec![0.0f32; w * h];
    if ch == 1 {
        for (i, &v) in data.iter().enumerate() {
            gray[i] = v as f32;
        }
    } else {
        for i in 0..w * h {
            let off = i * ch;
            gray[i] = 0.114 * data[off] as f32
                + 0.587 * data[off + 1] as f32
                + 0.299 * data[off + 2] as f32;
        }
    }
    gray
}

fn build_pyramid(img: &[f32], w: usize, h: usize, levels: usize) -> Vec<Vec<f32>> {
    let mut pyr = vec![img.to_vec()];
    let mut cw = w;
    let mut ch = h;
    for _ in 0..levels {
        let nw = (cw / 2).max(1);
        let nh = (ch / 2).max(1);
        let prev = pyr
            .last()
            .expect("pyr is non-empty: it is initialised with one element before this loop");
        let mut down = vec![0.0f32; nw * nh];
        for y in 0..nh {
            for x in 0..nw {
                let sy = (y * 2).min(ch - 1);
                let sx = (x * 2).min(cw - 1);
                down[y * nw + x] = prev[sy * cw + sx];
            }
        }
        pyr.push(down);
        cw = nw;
        ch = nh;
    }
    pyr
}

fn interpolate(img: &[f32], w: usize, x: f32, y: f32) -> f32 {
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let fx = x - x.floor();
    let fy = y - y.floor();
    let h_approx = img.len() / w;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h_approx - 1);
    let p00 = img[y0 * w + x0];
    let p10 = img[y0 * w + x1];
    let p01 = img[y1 * w + x0];
    let p11 = img[y1 * w + x1];
    let top = p00 + (p10 - p00) * fx;
    let bot = p01 + (p11 - p01) * fx;
    top + (bot - top) * fy
}

fn compute_patch_error(
    prev: &[f32],
    next: &[f32],
    w: usize,
    h: usize,
    px: f32,
    py: f32,
    nx: f32,
    ny: f32,
    half_w: usize,
    half_h: usize,
) -> f32 {
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for wy in 0..half_h * 2 + 1 {
        for wx in 0..half_w * 2 + 1 {
            let ix = (px + wx as f32 - half_w as f32).clamp(0.0, w as f32 - 1.0);
            let iy = (py + wy as f32 - half_h as f32).clamp(0.0, h as f32 - 1.0);
            let jx = (nx + wx as f32 - half_w as f32).clamp(0.0, w as f32 - 1.0);
            let jy = (ny + wy as f32 - half_h as f32).clamp(0.0, h as f32 - 1.0);
            let diff = interpolate(prev, w, ix, iy) - interpolate(next, w, jx, jy);
            sum += diff * diff;
            count += 1;
        }
    }
    if count > 0 {
        (sum / count as f32).sqrt()
    } else {
        0.0
    }
}

fn extract_points(py: Python<'_>, obj: &Py<PyAny>) -> PyResult<Vec<(f32, f32)>> {
    let bound = obj.bind(py);
    let pts: Vec<Py<PyAny>> = bound.extract()?;
    let mut result = Vec::with_capacity(pts.len());
    for pt in pts {
        let (x, y): (f32, f32) = pt.bind(py).extract()?;
        result.push((x, y));
    }
    Ok(result)
}
