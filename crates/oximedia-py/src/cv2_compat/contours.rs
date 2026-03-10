//! Contour finding and analysis: findContours, drawContours, contourArea, boundingRect,
//! arcLength, approxPolyDP.

use super::image_io::{extract_img, make_image_output};
use pyo3::prelude::*;

// A contour is a list of (x, y) integer points.
type Contour = Vec<(i32, i32)>;

/// Find contours in a binary image.
///
/// Mirrors `cv2.findContours(image, mode, method)`.
/// Returns `(contours, hierarchy)` tuple.
/// `contours` is a list of lists of (x, y) tuples.
/// `hierarchy` is None (simplified).
#[pyfunction]
#[pyo3(name = "findContours", signature = (image, mode, method))]
pub fn find_contours(
    py: Python<'_>,
    image: Py<PyAny>,
    mode: i32,
    method: i32,
) -> PyResult<Py<PyAny>> {
    let _ = (mode, method);
    let (data, h, w, ch) = extract_img(py, &image)?;

    // Convert to binary if needed (threshold at 127)
    let gray: Vec<bool> = if ch == 1 {
        data.iter().map(|&v| v > 0).collect()
    } else {
        data.chunks(ch)
            .map(|px| px[0] > 0 || px[1] > 0 || px[2] > 0)
            .collect()
    };

    let contours = trace_contours(&gray, w, h);

    // Convert contours to Python list of lists of (x, y) tuples
    let py_contours = pyo3::types::PyList::empty(py);
    for contour in &contours {
        let py_pts = pyo3::types::PyList::empty(py);
        for &(x, y) in contour {
            let pt = pyo3::types::PyTuple::new(py, [x, y])?;
            py_pts.append(pt)?;
        }
        py_contours.append(py_pts)?;
    }

    let hierarchy = py.None();
    let result = pyo3::types::PyTuple::new(py, [py_contours.as_any(), hierarchy.bind(py)])?;
    Ok(result.into())
}

/// Draw contours on an image.
///
/// Mirrors `cv2.drawContours(image, contours, contourIdx, color, thickness=1)`.
/// Modifies image in-place (returns copy).
#[pyfunction]
#[pyo3(name = "drawContours", signature = (image, contours, contour_idx, color, thickness=1))]
pub fn draw_contours(
    py: Python<'_>,
    image: Py<PyAny>,
    contours: Py<PyAny>,
    contour_idx: i32,
    color: (i32, i32, i32),
    thickness: i32,
) -> PyResult<Py<PyAny>> {
    let (mut data, h, w, ch) = extract_img(py, &image)?;

    let contour_list = extract_contours_from_py(py, &contours)?;

    let indices: Vec<usize> = if contour_idx < 0 {
        (0..contour_list.len()).collect()
    } else if (contour_idx as usize) < contour_list.len() {
        vec![contour_idx as usize]
    } else {
        vec![]
    };

    let (r, g, b) = (
        color.0.clamp(0, 255) as u8,
        color.1.clamp(0, 255) as u8,
        color.2.clamp(0, 255) as u8,
    );
    let pixel = if ch == 1 { vec![r] } else { vec![b, g, r] };

    for idx in indices {
        let contour = &contour_list[idx];
        if thickness < 0 {
            // Fill contour — simple scanline fill
            fill_contour(&mut data, w, h, ch, contour, &pixel);
        } else {
            // Draw contour outline
            let pts = contour.len();
            for i in 0..pts {
                let (x0, y0) = contour[i];
                let (x1, y1) = contour[(i + 1) % pts];
                draw_line_segment(
                    &mut data,
                    w,
                    h,
                    ch,
                    x0,
                    y0,
                    x1,
                    y1,
                    &pixel,
                    thickness.max(1) as usize,
                );
            }
        }
    }

    make_image_output(py, data, h, w, ch)
}

/// Compute the area of a contour.
///
/// Mirrors `cv2.contourArea(contour, oriented=False)`.
#[pyfunction]
#[pyo3(name = "contourArea", signature = (contour, oriented=false))]
pub fn contour_area(py: Python<'_>, contour: Py<PyAny>, oriented: bool) -> PyResult<f64> {
    let pts = extract_contour_from_py(py, &contour)?;
    let area = shoelace_area(&pts);
    if oriented {
        Ok(area)
    } else {
        Ok(area.abs())
    }
}

/// Compute the upright bounding rectangle of a contour.
///
/// Mirrors `cv2.boundingRect(contour)`.
/// Returns `(x, y, w, h)`.
#[pyfunction]
#[pyo3(name = "boundingRect")]
pub fn bounding_rect(py: Python<'_>, contour: Py<PyAny>) -> PyResult<(i32, i32, i32, i32)> {
    let pts = extract_contour_from_py(py, &contour)?;
    if pts.is_empty() {
        return Ok((0, 0, 0, 0));
    }
    let x_min = pts.iter().map(|&(x, _)| x).min().unwrap_or(0);
    let x_max = pts.iter().map(|&(x, _)| x).max().unwrap_or(0);
    let y_min = pts.iter().map(|&(_, y)| y).min().unwrap_or(0);
    let y_max = pts.iter().map(|&(_, y)| y).max().unwrap_or(0);
    Ok((x_min, y_min, x_max - x_min + 1, y_max - y_min + 1))
}

/// Compute the perimeter of a contour.
///
/// Mirrors `cv2.arcLength(contour, closed)`.
#[pyfunction]
#[pyo3(name = "arcLength")]
pub fn arc_length(py: Python<'_>, contour: Py<PyAny>, closed: bool) -> PyResult<f64> {
    let pts = extract_contour_from_py(py, &contour)?;
    if pts.len() < 2 {
        return Ok(0.0);
    }
    let mut length = 0.0f64;
    let n = pts.len();
    let pairs = if closed { n } else { n - 1 };
    for i in 0..pairs {
        let (x0, y0) = pts[i];
        let (x1, y1) = pts[(i + 1) % n];
        let dx = (x1 - x0) as f64;
        let dy = (y1 - y0) as f64;
        length += (dx * dx + dy * dy).sqrt();
    }
    Ok(length)
}

/// Approximate a contour with fewer vertices (Douglas-Peucker algorithm).
///
/// Mirrors `cv2.approxPolyDP(contour, epsilon, closed)`.
/// Returns a list of (x, y) tuples.
#[pyfunction]
#[pyo3(name = "approxPolyDP")]
pub fn approx_poly_dp(
    py: Python<'_>,
    contour: Py<PyAny>,
    epsilon: f64,
    closed: bool,
) -> PyResult<Py<PyAny>> {
    let pts = extract_contour_from_py(py, &contour)?;
    let simplified = douglas_peucker(&pts, epsilon, closed);

    let py_pts = pyo3::types::PyList::empty(py);
    for (x, y) in simplified {
        py_pts.append(pyo3::types::PyTuple::new(py, [x, y])?)?;
    }
    Ok(py_pts.into())
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Trace contours using a simplified border-following algorithm.
fn trace_contours(binary: &[bool], w: usize, h: usize) -> Vec<Contour> {
    let mut visited = vec![false; h * w];
    let mut contours = Vec::new();

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if binary[idx] && !visited[idx] {
                // Start tracing from this pixel
                let contour = trace_single_contour(binary, &mut visited, w, h, x, y);
                if !contour.is_empty() {
                    contours.push(contour);
                }
            }
        }
    }
    contours
}

fn trace_single_contour(
    binary: &[bool],
    visited: &mut Vec<bool>,
    w: usize,
    h: usize,
    start_x: usize,
    start_y: usize,
) -> Contour {
    // BFS boundary tracing
    let mut contour = Vec::new();
    let mut queue = std::collections::VecDeque::new();
    queue.push_back((start_x as i32, start_y as i32));

    while let Some((x, y)) = queue.pop_front() {
        if x < 0 || y < 0 || x >= w as i32 || y >= h as i32 {
            continue;
        }
        let idx = y as usize * w + x as usize;
        if visited[idx] || !binary[idx] {
            continue;
        }
        visited[idx] = true;

        // Check if it's a boundary pixel (has at least one non-binary neighbor)
        let is_boundary = [(-1, 0), (1, 0), (0, -1), (0, 1)].iter().any(|&(dx, dy)| {
            let nx = x + dx;
            let ny = y + dy;
            if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                true
            } else {
                !binary[ny as usize * w + nx as usize]
            }
        });

        if is_boundary {
            contour.push((x, y));
        }

        for (dx, dy) in [(-1, 0i32), (1, 0), (0, -1), (0, 1)] {
            queue.push_back((x + dx, y + dy));
        }
    }
    contour
}

fn shoelace_area(pts: &[(i32, i32)]) -> f64 {
    let n = pts.len();
    if n < 3 {
        return 0.0;
    }
    let mut sum = 0i64;
    for i in 0..n {
        let (x0, y0) = pts[i];
        let (x1, y1) = pts[(i + 1) % n];
        sum += x0 as i64 * y1 as i64 - x1 as i64 * y0 as i64;
    }
    sum as f64 / 2.0
}

fn extract_contours_from_py(py: Python<'_>, obj: &Py<PyAny>) -> PyResult<Vec<Contour>> {
    let bound = obj.bind(py);
    let outer: Vec<Py<PyAny>> = bound.extract()?;
    let mut result = Vec::with_capacity(outer.len());
    for item in outer {
        result.push(extract_contour_from_py(py, &item)?);
    }
    Ok(result)
}

fn extract_contour_from_py(py: Python<'_>, obj: &Py<PyAny>) -> PyResult<Contour> {
    let bound = obj.bind(py);
    let pts: Vec<Py<PyAny>> = bound.extract()?;
    let mut result = Vec::with_capacity(pts.len());
    for pt in pts {
        let (x, y): (i32, i32) = pt.bind(py).extract()?;
        result.push((x, y));
    }
    Ok(result)
}

fn draw_line_segment(
    data: &mut Vec<u8>,
    w: usize,
    h: usize,
    ch: usize,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    pixel: &[u8],
    thickness: usize,
) {
    // Bresenham's line algorithm
    let mut x = x0;
    let mut y = y0;
    let dx = (x1 - x0).abs();
    let dy = (y1 - y0).abs();
    let sx: i32 = if x0 < x1 { 1 } else { -1 };
    let sy: i32 = if y0 < y1 { 1 } else { -1 };
    let mut err = dx - dy;
    let half = (thickness / 2) as i32;

    loop {
        // Draw a square of size thickness centered on (x, y)
        for ty in -half..=(half) {
            for tx in -half..=(half) {
                let px = x + tx;
                let py = y + ty;
                if px >= 0 && py >= 0 && px < w as i32 && py < h as i32 {
                    let off = (py as usize * w + px as usize) * ch;
                    data[off..off + ch].copy_from_slice(pixel);
                }
            }
        }
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 > -dy {
            err -= dy;
            x += sx;
        }
        if e2 < dx {
            err += dx;
            y += sy;
        }
    }
}

fn fill_contour(
    data: &mut Vec<u8>,
    w: usize,
    h: usize,
    ch: usize,
    contour: &Contour,
    pixel: &[u8],
) {
    if contour.is_empty() {
        return;
    }

    let y_min = contour
        .iter()
        .map(|&(_, y)| y)
        .min()
        .unwrap_or(0)
        .clamp(0, h as i32 - 1) as usize;
    let y_max = contour
        .iter()
        .map(|&(_, y)| y)
        .max()
        .unwrap_or(0)
        .clamp(0, h as i32 - 1) as usize;

    for y in y_min..=y_max {
        // Find x intersections
        let mut intersections: Vec<i32> = Vec::new();
        let n = contour.len();
        for i in 0..n {
            let (x0, y0) = contour[i];
            let (x1, y1) = contour[(i + 1) % n];
            let yf = y as i32;
            if (y0 <= yf && y1 > yf) || (y1 <= yf && y0 > yf) {
                let xi = x0 + (yf - y0) * (x1 - x0) / (y1 - y0);
                intersections.push(xi);
            }
        }
        intersections.sort_unstable();
        let mut i = 0;
        while i + 1 < intersections.len() {
            let x_start = intersections[i].clamp(0, w as i32 - 1) as usize;
            let x_end = intersections[i + 1].clamp(0, w as i32 - 1) as usize;
            for x in x_start..=x_end {
                let off = (y * w + x) * ch;
                data[off..off + ch].copy_from_slice(pixel);
            }
            i += 2;
        }
    }
}

fn douglas_peucker(pts: &[(i32, i32)], epsilon: f64, _closed: bool) -> Vec<(i32, i32)> {
    if pts.len() < 3 {
        return pts.to_vec();
    }

    // Find the point with max distance from the line start-end
    let (start, end) = (pts[0], pts[pts.len() - 1]);
    let mut max_dist = 0.0f64;
    let mut max_idx = 0;

    for i in 1..pts.len() - 1 {
        let d = point_line_distance(pts[i], start, end);
        if d > max_dist {
            max_dist = d;
            max_idx = i;
        }
    }

    if max_dist > epsilon {
        let mut left = douglas_peucker(&pts[..=max_idx], epsilon, false);
        let right = douglas_peucker(&pts[max_idx..], epsilon, false);
        left.pop(); // remove duplicate at junction
        left.extend(right);
        left
    } else {
        vec![start, end]
    }
}

fn point_line_distance(p: (i32, i32), a: (i32, i32), b: (i32, i32)) -> f64 {
    let (px, py) = (p.0 as f64, p.1 as f64);
    let (ax, ay) = (a.0 as f64, a.1 as f64);
    let (bx, by) = (b.0 as f64, b.1 as f64);
    let num = ((by - ay) * px - (bx - ax) * py + bx * ay - by * ax).abs();
    let den = ((by - ay).powi(2) + (bx - ax).powi(2)).sqrt();
    if den < 1e-10 {
        0.0
    } else {
        num / den
    }
}
