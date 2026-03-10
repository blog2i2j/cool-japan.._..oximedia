//! Drawing functions: rectangle, circle, line, putText, polylines, fillPoly, ellipse.

use super::image_io::{extract_img, make_image_output};
use pyo3::prelude::*;

/// Draw a rectangle on an image.
///
/// Mirrors `cv2.rectangle(img, pt1, pt2, color, thickness=1)`.
#[pyfunction]
#[pyo3(name = "rectangle", signature = (img, pt1, pt2, color, thickness=1))]
pub fn rectangle(
    py: Python<'_>,
    img: Py<PyAny>,
    pt1: (i32, i32),
    pt2: (i32, i32),
    color: Py<PyAny>,
    thickness: i32,
) -> PyResult<Py<PyAny>> {
    let (mut data, h, w, ch) = extract_img(py, &img)?;
    let pixel = extract_color(py, &color, ch)?;

    let (x1, y1) = (pt1.0.min(pt2.0), pt1.1.min(pt2.1));
    let (x2, y2) = (pt1.0.max(pt2.0), pt1.1.max(pt2.1));

    if thickness < 0 {
        // Filled
        for y in y1.max(0)..(y2 + 1).min(h as i32) {
            for x in x1.max(0)..(x2 + 1).min(w as i32) {
                put_pixel(&mut data, w, ch, x as usize, y as usize, &pixel);
            }
        }
    } else {
        let t = thickness.max(1) as usize;
        // Top and bottom edges
        for y in [y1, y2] {
            for x in x1.max(0)..(x2 + 1).min(w as i32) {
                for ty in 0..t as i32 {
                    let py_coord = y + ty;
                    if py_coord >= 0 && py_coord < h as i32 {
                        put_pixel(&mut data, w, ch, x as usize, py_coord as usize, &pixel);
                    }
                }
            }
        }
        // Left and right edges
        for x in [x1, x2] {
            for y in y1.max(0)..(y2 + 1).min(h as i32) {
                for tx in 0..t as i32 {
                    let px_coord = x + tx;
                    if px_coord >= 0 && px_coord < w as i32 {
                        put_pixel(&mut data, w, ch, px_coord as usize, y as usize, &pixel);
                    }
                }
            }
        }
    }

    make_image_output(py, data, h, w, ch)
}

/// Draw a circle on an image.
///
/// Mirrors `cv2.circle(img, center, radius, color, thickness=1)`.
#[pyfunction]
#[pyo3(name = "circle", signature = (img, center, radius, color, thickness=1))]
pub fn circle(
    py: Python<'_>,
    img: Py<PyAny>,
    center: (i32, i32),
    radius: i32,
    color: Py<PyAny>,
    thickness: i32,
) -> PyResult<Py<PyAny>> {
    let (mut data, h, w, ch) = extract_img(py, &img)?;
    let pixel = extract_color(py, &color, ch)?;
    let (cx, cy) = (center.0, center.1);
    let r = radius.max(0);

    if thickness < 0 {
        // Filled circle
        for y in (cy - r).max(0)..(cy + r + 1).min(h as i32) {
            let dy = y - cy;
            let dx = ((r * r - dy * dy) as f64).sqrt() as i32;
            for x in (cx - dx).max(0)..(cx + dx + 1).min(w as i32) {
                put_pixel(&mut data, w, ch, x as usize, y as usize, &pixel);
            }
        }
    } else {
        // Outline using midpoint circle algorithm
        let t = (thickness.max(1) - 1) / 2;
        let mut x = 0i32;
        let mut y = r;
        let mut d = 3 - 2 * r;
        while x <= y {
            for &(px, py_coord) in &[
                (cx + x, cy + y),
                (cx - x, cy + y),
                (cx + x, cy - y),
                (cx - x, cy - y),
                (cx + y, cy + x),
                (cx - y, cy + x),
                (cx + y, cy - x),
                (cx - y, cy - x),
            ] {
                for tx in -t..=t {
                    for ty in -t..=t {
                        let fx = px + tx;
                        let fy = py_coord + ty;
                        if fx >= 0 && fy >= 0 && fx < w as i32 && fy < h as i32 {
                            put_pixel(&mut data, w, ch, fx as usize, fy as usize, &pixel);
                        }
                    }
                }
            }
            if d < 0 {
                d += 4 * x + 6;
            } else {
                d += 4 * (x - y) + 10;
                y -= 1;
            }
            x += 1;
        }
    }

    make_image_output(py, data, h, w, ch)
}

/// Draw a line on an image.
///
/// Mirrors `cv2.line(img, pt1, pt2, color, thickness=1)`.
#[pyfunction]
#[pyo3(name = "line", signature = (img, pt1, pt2, color, thickness=1))]
pub fn line(
    py: Python<'_>,
    img: Py<PyAny>,
    pt1: (i32, i32),
    pt2: (i32, i32),
    color: Py<PyAny>,
    thickness: i32,
) -> PyResult<Py<PyAny>> {
    let (mut data, h, w, ch) = extract_img(py, &img)?;
    let pixel = extract_color(py, &color, ch)?;
    let t = thickness.max(1) as usize;

    draw_bresenham_line(&mut data, w, h, ch, pt1.0, pt1.1, pt2.0, pt2.1, &pixel, t);
    make_image_output(py, data, h, w, ch)
}

/// Render ASCII text on an image using a minimal bitmap font.
///
/// Mirrors `cv2.putText(img, text, org, fontFace, fontScale, color, thickness=1)`.
#[pyfunction]
#[pyo3(name = "putText", signature = (img, text, org, font_face, font_scale, color, thickness=1, line_type=8, bottom_left_origin=false))]
#[allow(clippy::too_many_arguments)]
pub fn put_text(
    py: Python<'_>,
    img: Py<PyAny>,
    text: &str,
    org: (i32, i32),
    font_face: i32,
    font_scale: f64,
    color: Py<PyAny>,
    thickness: i32,
    line_type: i32,
    bottom_left_origin: bool,
) -> PyResult<Py<PyAny>> {
    let _ = (font_face, line_type, bottom_left_origin);
    let (mut data, h, w, ch) = extract_img(py, &img)?;
    let pixel = extract_color(py, &color, ch)?;

    let char_w = (6.0 * font_scale).round() as i32;
    let char_h = (10.0 * font_scale).round() as i32;
    let stroke = thickness.max(1);

    let (mut cursor_x, cursor_y) = org;

    for c in text.chars() {
        let bitmap = char_bitmap(c);
        for (by, row) in bitmap.iter().enumerate() {
            for bx in 0..5 {
                if (row >> (4 - bx)) & 1 == 1 {
                    let px = cursor_x + (bx as f64 * font_scale) as i32;
                    let py_coord = cursor_y - char_h + (by as f64 * font_scale) as i32;
                    for sy in 0..stroke {
                        for sx in 0..stroke {
                            let fx = px + sx;
                            let fy = py_coord + sy;
                            if fx >= 0 && fy >= 0 && fx < w as i32 && fy < h as i32 {
                                put_pixel(&mut data, w, ch, fx as usize, fy as usize, &pixel);
                            }
                        }
                    }
                }
            }
        }
        cursor_x += char_w + stroke;
    }

    make_image_output(py, data, h, w, ch)
}

/// Draw polylines (open or closed polygon outline).
///
/// Mirrors `cv2.polylines(img, pts, isClosed, color, thickness=1)`.
#[pyfunction]
#[pyo3(name = "polylines", signature = (img, pts, is_closed, color, thickness=1))]
pub fn polylines(
    py: Python<'_>,
    img: Py<PyAny>,
    pts: Py<PyAny>,
    is_closed: bool,
    color: Py<PyAny>,
    thickness: i32,
) -> PyResult<Py<PyAny>> {
    let (mut data, h, w, ch) = extract_img(py, &img)?;
    let pixel = extract_color(py, &color, ch)?;
    let t = thickness.max(1) as usize;

    let all_contours = extract_point_sets(py, &pts)?;
    for contour in &all_contours {
        let n = contour.len();
        for i in 0..n {
            let (x0, y0) = contour[i];
            let (x1, y1) = if i + 1 < n {
                contour[i + 1]
            } else if is_closed {
                contour[0]
            } else {
                break;
            };
            draw_bresenham_line(&mut data, w, h, ch, x0, y0, x1, y1, &pixel, t);
        }
    }

    make_image_output(py, data, h, w, ch)
}

/// Fill a polygon with a color.
///
/// Mirrors `cv2.fillPoly(img, pts, color)`.
#[pyfunction]
#[pyo3(name = "fillPoly", signature = (img, pts, color))]
pub fn fill_poly(
    py: Python<'_>,
    img: Py<PyAny>,
    pts: Py<PyAny>,
    color: Py<PyAny>,
) -> PyResult<Py<PyAny>> {
    let (mut data, h, w, ch) = extract_img(py, &img)?;
    let pixel = extract_color(py, &color, ch)?;

    let all_contours = extract_point_sets(py, &pts)?;
    for contour in &all_contours {
        fill_polygon(&mut data, w, h, ch, contour, &pixel);
    }

    make_image_output(py, data, h, w, ch)
}

/// Draw an ellipse on an image.
///
/// Mirrors `cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness=1)`.
#[pyfunction]
#[pyo3(name = "ellipse", signature = (img, center, axes, angle, start_angle, end_angle, color, thickness=1))]
#[allow(clippy::too_many_arguments)]
pub fn ellipse(
    py: Python<'_>,
    img: Py<PyAny>,
    center: (i32, i32),
    axes: (i32, i32),
    angle: f64,
    start_angle: f64,
    end_angle: f64,
    color: Py<PyAny>,
    thickness: i32,
) -> PyResult<Py<PyAny>> {
    let (mut data, h, w, ch) = extract_img(py, &img)?;
    let pixel = extract_color(py, &color, ch)?;

    let (cx, cy) = (center.0 as f64, center.1 as f64);
    let (ax, ay) = (axes.0 as f64, axes.1 as f64);
    let rot_rad = angle * std::f64::consts::PI / 180.0;
    let cos_r = rot_rad.cos();
    let sin_r = rot_rad.sin();

    let start_rad = start_angle * std::f64::consts::PI / 180.0;
    let end_rad = end_angle * std::f64::consts::PI / 180.0;

    let steps = ((ax.max(ay) * 2.0 * std::f64::consts::PI) as usize).max(360);
    let delta = (end_rad - start_rad) / steps as f64;

    let t = thickness.max(1) as usize;
    let filled = thickness < 0;

    let mut prev: Option<(i32, i32)> = None;
    for step in 0..=steps {
        let theta = start_rad + step as f64 * delta;
        let ex = ax * theta.cos();
        let ey = ay * theta.sin();
        let rx = (ex * cos_r - ey * sin_r + cx).round() as i32;
        let ry = (ex * sin_r + ey * cos_r + cy).round() as i32;

        if let Some((px, py_coord)) = prev {
            if filled {
                // Draw line from center to edge for filled ellipse
                draw_bresenham_line(&mut data, w, h, ch, center.0, center.1, rx, ry, &pixel, 1);
            } else {
                draw_bresenham_line(&mut data, w, h, ch, px, py_coord, rx, ry, &pixel, t);
            }
        }
        prev = Some((rx, ry));
    }

    make_image_output(py, data, h, w, ch)
}

// ── Private helpers ───────────────────────────────────────────────────────────

fn put_pixel(data: &mut [u8], w: usize, ch: usize, x: usize, y: usize, pixel: &[u8]) {
    let off = (y * w + x) * ch;
    if off + ch <= data.len() {
        data[off..off + ch].copy_from_slice(pixel);
    }
}

fn extract_color(py: Python<'_>, color: &Py<PyAny>, ch: usize) -> PyResult<Vec<u8>> {
    let bound = color.bind(py);
    let vals: Vec<i32> = bound.extract()?;
    let mut pixel = vec![0u8; ch];
    for i in 0..ch.min(vals.len()) {
        pixel[i] = vals[i].clamp(0, 255) as u8;
    }
    Ok(pixel)
}

fn draw_bresenham_line(
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
    let mut x = x0;
    let mut y = y0;
    let dx = (x1 - x0).abs();
    let dy = (y1 - y0).abs();
    let sx: i32 = if x0 < x1 { 1 } else { -1 };
    let sy: i32 = if y0 < y1 { 1 } else { -1 };
    let mut err = dx - dy;
    let half = (thickness / 2) as i32;

    loop {
        for ty in -half..=half {
            for tx in -half..=half {
                let px = x + tx;
                let py_coord = y + ty;
                if px >= 0 && py_coord >= 0 && px < w as i32 && py_coord < h as i32 {
                    put_pixel(data, w, ch, px as usize, py_coord as usize, pixel);
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

fn extract_point_sets(py: Python<'_>, obj: &Py<PyAny>) -> PyResult<Vec<Vec<(i32, i32)>>> {
    let bound = obj.bind(py);
    let outer: Vec<Py<PyAny>> = bound.extract()?;
    let mut result = Vec::with_capacity(outer.len());
    for item in outer {
        let inner: Vec<Py<PyAny>> = item.bind(py).extract()?;
        let pts: Vec<(i32, i32)> = inner
            .iter()
            .map(|pt| pt.bind(py).extract::<(i32, i32)>())
            .collect::<PyResult<_>>()?;
        result.push(pts);
    }
    Ok(result)
}

fn fill_polygon(
    data: &mut Vec<u8>,
    w: usize,
    h: usize,
    ch: usize,
    pts: &[(i32, i32)],
    pixel: &[u8],
) {
    if pts.len() < 3 {
        return;
    }
    let y_min = pts.iter().map(|&(_, y)| y).min().unwrap_or(0).max(0) as usize;
    let y_max = pts
        .iter()
        .map(|&(_, y)| y)
        .max()
        .unwrap_or(0)
        .min(h as i32 - 1) as usize;
    let n = pts.len();

    for y in y_min..=y_max {
        let mut intersections: Vec<i32> = Vec::new();
        let yf = y as i32;
        for i in 0..n {
            let (x0, y0) = pts[i];
            let (x1, y1) = pts[(i + 1) % n];
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
                put_pixel(data, w, ch, x, y, pixel);
            }
            i += 2;
        }
    }
}

/// Minimal 5x8 bitmap font for ASCII chars 32-126.
fn char_bitmap(c: char) -> [u8; 8] {
    let code = c as usize;
    if code < 32 || code > 126 {
        return [0; 8];
    }
    // Very compact 5x8 font — each byte encodes one row, MSB = leftmost pixel
    // This covers digits, uppercase, lowercase, and common punctuation.
    match c {
        '0' => [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E, 0x00],
        '1' => [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E, 0x00],
        '2' => [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F, 0x00],
        '3' => [0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E, 0x00],
        '4' => [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02, 0x00],
        '5' => [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E, 0x00],
        '6' => [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E, 0x00],
        '7' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08, 0x00],
        '8' => [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E, 0x00],
        '9' => [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C, 0x00],
        'A' => [0x04, 0x0A, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x00],
        'B' => [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E, 0x00],
        'C' => [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E, 0x00],
        'D' => [0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C, 0x00],
        'E' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F, 0x00],
        'F' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10, 0x00],
        'G' => [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F, 0x00],
        'H' => [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11, 0x00],
        'I' => [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E, 0x00],
        'J' => [0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C, 0x00],
        'K' => [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11, 0x00],
        'L' => [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F, 0x00],
        'M' => [0x11, 0x1B, 0x15, 0x11, 0x11, 0x11, 0x11, 0x00],
        'N' => [0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11, 0x00],
        'O' => [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E, 0x00],
        'P' => [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10, 0x00],
        'Q' => [0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D, 0x00],
        'R' => [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11, 0x00],
        'S' => [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E, 0x00],
        'T' => [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x00],
        'U' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E, 0x00],
        'V' => [0x11, 0x11, 0x11, 0x0A, 0x0A, 0x04, 0x04, 0x00],
        'W' => [0x11, 0x11, 0x15, 0x15, 0x0A, 0x0A, 0x11, 0x00],
        'X' => [0x11, 0x0A, 0x04, 0x04, 0x04, 0x0A, 0x11, 0x00],
        'Y' => [0x11, 0x0A, 0x04, 0x04, 0x04, 0x04, 0x04, 0x00],
        'Z' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F, 0x00],
        ' ' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
        '.' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x00],
        ':' => [0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00, 0x00],
        '-' => [0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00, 0x00],
        '_' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F, 0x00],
        '/' => [0x01, 0x02, 0x02, 0x04, 0x08, 0x08, 0x10, 0x00],
        _ => [0x15, 0x0A, 0x15, 0x0A, 0x15, 0x0A, 0x15, 0x00], // checkered for unknown
    }
}
