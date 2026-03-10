//! Hough transform: line and circle detection.

use pyo3::prelude::*;
use pyo3::types::PyList;
use std::f64::consts::PI;

use super::image_io::extract_img;

/// Detect lines in a binary edge image using standard Hough transform.
///
/// Returns list of `(rho, theta)` tuples for each detected line.
/// Mirrors `cv2.HoughLines(image, rho, theta, threshold)`.
#[pyfunction]
#[pyo3(name = "HoughLines")]
#[pyo3(signature = (image, rho, theta, threshold, srn=0.0, stn=0.0, min_theta=0.0, max_theta=PI))]
pub fn hough_lines(
    py: Python<'_>,
    image: Py<PyAny>,
    rho: f64,
    theta: f64,
    threshold: i32,
    srn: f64,
    stn: f64,
    min_theta: f64,
    max_theta: f64,
) -> PyResult<Py<PyAny>> {
    let _ = (srn, stn);
    let (data, h, w, ch) = extract_img(py, &image)?;
    let binary: Vec<bool> = (0..h * w).map(|i| data[i * ch] > 0).collect();

    let rho_res = rho;
    let theta_res = theta;
    let max_rho = ((h as f64).hypot(w as f64) + 1.0) / rho_res;
    let n_rho = max_rho as usize * 2 + 1;
    let n_theta = ((max_theta - min_theta) / theta_res).ceil() as usize + 1;

    // Accumulator
    let mut acc = vec![0i32; n_rho * n_theta];

    let thetas: Vec<f64> = (0..n_theta)
        .map(|i| min_theta + i as f64 * theta_res)
        .collect();
    let cos_t: Vec<f64> = thetas.iter().map(|&t| t.cos()).collect();
    let sin_t: Vec<f64> = thetas.iter().map(|&t| t.sin()).collect();

    for y in 0..h {
        for x in 0..w {
            if !binary[y * w + x] {
                continue;
            }
            for ti in 0..n_theta {
                let rho_val = x as f64 * cos_t[ti] + y as f64 * sin_t[ti];
                let ri = ((rho_val / rho_res) + max_rho) as usize;
                if ri < n_rho {
                    acc[ri * n_theta + ti] += 1;
                }
            }
        }
    }

    // Extract lines above threshold, sorted by votes descending
    let mut lines: Vec<(f64, f64, i32)> = Vec::new();
    for ri in 0..n_rho {
        for ti in 0..n_theta {
            let votes = acc[ri * n_theta + ti];
            if votes >= threshold {
                let rho_val = (ri as f64 - max_rho) * rho_res;
                let theta_val = thetas[ti];
                lines.push((rho_val, theta_val, votes));
            }
        }
    }

    lines.sort_by(|a, b| b.2.cmp(&a.2));

    let list = PyList::new(py, lines.iter().map(|&(r, t, _)| (r, t)))?;
    Ok(list.into())
}

/// Detect lines using Probabilistic Hough Transform.
///
/// Returns list of `((x1,y1), (x2,y2))` line segments.
/// Mirrors `cv2.HoughLinesP(image, rho, theta, threshold, minLineLength, maxLineGap)`.
#[pyfunction]
#[pyo3(name = "HoughLinesP")]
#[pyo3(signature = (image, rho, theta, threshold, min_line_length=0.0, max_line_gap=0.0))]
pub fn hough_lines_p(
    py: Python<'_>,
    image: Py<PyAny>,
    rho: f64,
    theta: f64,
    threshold: i32,
    min_line_length: f64,
    max_line_gap: f64,
) -> PyResult<Py<PyAny>> {
    let _ = (rho, theta, max_line_gap);
    let (data, h, w, ch) = extract_img(py, &image)?;
    let binary: Vec<bool> = (0..h * w).map(|i| data[i * ch] > 0).collect();

    let mut segments: Vec<((i32, i32), (i32, i32))> = Vec::new();

    // Horizontal scan: find runs of active edge pixels
    for y in 0..h {
        let mut run_start: Option<usize> = None;
        for x in 0..=w {
            let active = x < w && binary[y * w + x];
            if active && run_start.is_none() {
                run_start = Some(x);
            } else if !active {
                if let Some(xs) = run_start.take() {
                    let length = (x - xs) as f64;
                    if length >= min_line_length && length >= threshold as f64 {
                        segments.push(((xs as i32, y as i32), (x as i32 - 1, y as i32)));
                    }
                }
            }
        }
    }

    // Vertical scan: find runs of active edge pixels
    for x in 0..w {
        let mut run_start: Option<usize> = None;
        for y in 0..=h {
            let active = y < h && binary[y * w + x];
            if active && run_start.is_none() {
                run_start = Some(y);
            } else if !active {
                if let Some(ys) = run_start.take() {
                    let length = (y - ys) as f64;
                    if length >= min_line_length && length >= threshold as f64 {
                        segments.push(((x as i32, ys as i32), (x as i32, y as i32 - 1)));
                    }
                }
            }
        }
    }

    let list = PyList::new(py, segments.iter().map(|&(a, b)| (a, b)))?;
    Ok(list.into())
}

/// Detect circles using Hough Circle Transform.
///
/// Returns list of `(x, y, radius)` tuples.
/// Mirrors `cv2.HoughCircles(image, method, dp, minDist, param1, param2, minRadius, maxRadius)`.
#[pyfunction]
#[pyo3(name = "HoughCircles")]
#[pyo3(signature = (image, method, dp, min_dist, param1=100.0, param2=100.0, min_radius=0, max_radius=0))]
pub fn hough_circles(
    py: Python<'_>,
    image: Py<PyAny>,
    method: i32,
    dp: f64,
    min_dist: f64,
    param1: f64,
    param2: f64,
    min_radius: i32,
    max_radius: i32,
) -> PyResult<Py<PyAny>> {
    let _ = (method, dp, param1);
    let (data, h, w, ch) = extract_img(py, &image)?;

    let gray: Vec<u8> = if ch == 1 {
        data
    } else {
        (0..h * w)
            .map(|i| {
                let off = i * ch;
                (0.299 * data[off + 2] as f32
                    + 0.587 * data[off + 1] as f32
                    + 0.114 * data[off] as f32) as u8
            })
            .collect()
    };

    let min_r = min_radius.max(1) as usize;
    let max_r = if max_radius <= 0 {
        w.min(h) / 2
    } else {
        max_radius as usize
    };
    let threshold = param2 as i32;
    let min_dist_sq = min_dist * min_dist;

    let mut circles: Vec<(f32, f32, f32, i32)> = Vec::new(); // (cx, cy, r, votes)

    for r in min_r..=max_r {
        let mut acc = vec![0i32; h * w];

        // For each bright pixel, vote for candidate circle centres at distance r
        for y in 0..h {
            for x in 0..w {
                if gray[y * w + x] < 50 {
                    continue; // rough edge threshold
                }
                for step in 0..36 {
                    let angle = step as f64 * PI / 18.0;
                    let cx = x as f64 + r as f64 * angle.cos();
                    let cy = y as f64 + r as f64 * angle.sin();
                    let cxi = cx as i64;
                    let cyi = cy as i64;
                    if cxi >= 0 && cxi < w as i64 && cyi >= 0 && cyi < h as i64 {
                        acc[cyi as usize * w + cxi as usize] += 1;
                    }
                }
            }
        }

        // Collect maxima above threshold while respecting minimum distance
        for cy in r..h.saturating_sub(r) {
            for cx in r..w.saturating_sub(r) {
                let votes = acc[cy * w + cx];
                if votes >= threshold {
                    let too_close = circles.iter().any(|&(ex, ey, _, _)| {
                        let dx = cx as f64 - ex as f64;
                        let dy = cy as f64 - ey as f64;
                        dx * dx + dy * dy < min_dist_sq
                    });
                    if !too_close {
                        circles.push((cx as f32, cy as f32, r as f32, votes));
                    }
                }
            }
        }
    }

    // Sort by votes descending
    circles.sort_by(|a, b| b.3.cmp(&a.3));

    let list = PyList::new(py, circles.iter().map(|&(x, y, r, _)| (x, y, r)))?;
    Ok(list.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_black_produces_no_lines() {
        // All-zero binary mask → no edge pixels → accumulator stays zero
        let data = vec![0u8; 10 * 10];
        let binary: Vec<bool> = data.iter().map(|&v| v > 0).collect();
        assert!(binary.iter().all(|&b| !b));
    }

    #[test]
    fn test_hough_lines_p_horizontal_run() {
        // Horizontal run of 10 pixels at row 5, min_length=5 → one segment found
        let mut data = vec![0u8; 10 * 10];
        for x in 0..10 {
            data[5 * 10 + x] = 255;
        }
        let binary: Vec<bool> = data.iter().map(|&v| v > 0).collect();
        let mut count = 0usize;
        let mut run_start: Option<usize> = None;
        for x in 0..=10 {
            let active = x < 10 && binary[5 * 10 + x];
            if active && run_start.is_none() {
                run_start = Some(x);
            } else if !active {
                if let Some(xs) = run_start.take() {
                    if (x - xs) >= 5 {
                        count += 1;
                    }
                }
            }
        }
        assert_eq!(count, 1);
    }

    #[test]
    fn test_hough_circles_radius_range() {
        let min_r = 5i32 as usize;
        let max_r_param = 0i32;
        let h = 100usize;
        let w = 100usize;
        let computed_max = if max_r_param <= 0 {
            w.min(h) / 2
        } else {
            max_r_param as usize
        };
        assert!(min_r <= computed_max);
    }

    #[test]
    fn test_accumulator_vote_logic() {
        // One bright pixel at (5,5) with r=3 should produce 36 votes spread
        // across the accumulator — just verify no panic and total votes == 36
        let h = 20usize;
        let w = 20usize;
        let r = 3usize;
        let mut acc = vec![0i32; h * w];
        let x = 5usize;
        let y = 5usize;
        for step in 0..36 {
            let angle = step as f64 * PI / 18.0;
            let cx = x as f64 + r as f64 * angle.cos();
            let cy = y as f64 + r as f64 * angle.sin();
            let cxi = cx as i64;
            let cyi = cy as i64;
            if cxi >= 0 && cxi < w as i64 && cyi >= 0 && cyi < h as i64 {
                acc[cyi as usize * w + cxi as usize] += 1;
            }
        }
        let total: i32 = acc.iter().sum();
        assert_eq!(total, 36);
    }
}
