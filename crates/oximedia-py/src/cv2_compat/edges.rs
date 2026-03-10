//! OpenCV-compatible edge detection functions.
//!
//! Provides Canny edge detection, Sobel gradients, and Laplacian operator.

use super::image_io::{extract_img, make_image_output};
use pyo3::prelude::*;
use std::collections::VecDeque;

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Convert multi-channel image to single-channel grayscale using BT.601.
pub(crate) fn to_grayscale(data: &[u8], h: usize, w: usize, ch: usize) -> Vec<u8> {
    let n = h * w;
    match ch {
        1 => data[..n].to_vec(),
        2 => (0..n).map(|i| data[i * 2]).collect(),
        3 => (0..n)
            .map(|i| {
                let b = data[i * 3] as f32;
                let g = data[i * 3 + 1] as f32;
                let r = data[i * 3 + 2] as f32;
                (0.114 * b + 0.587 * g + 0.299 * r).clamp(0.0, 255.0) as u8
            })
            .collect(),
        _ => (0..n)
            .map(|i| {
                let b = data[i * ch] as f32;
                let g = data[i * ch + 1] as f32;
                let r = data[i * ch + 2] as f32;
                (0.114 * b + 0.587 * g + 0.299 * r).clamp(0.0, 255.0) as u8
            })
            .collect(),
    }
}

/// Apply a 2D convolution kernel (flat, kh×kw) to a single-channel f32 image.
pub(crate) fn convolve_f32(
    src: &[f32],
    h: usize,
    w: usize,
    kernel: &[f32],
    kh: usize,
    kw: usize,
) -> Vec<f32> {
    let half_h = (kh / 2) as i64;
    let half_w = (kw / 2) as i64;
    let mut out = vec![0.0f32; h * w];
    for row in 0..h {
        for col in 0..w {
            let mut acc = 0.0f32;
            for ki in 0..kh {
                for kj in 0..kw {
                    let r = (row as i64 + ki as i64 - half_h).clamp(0, h as i64 - 1) as usize;
                    let c = (col as i64 + kj as i64 - half_w).clamp(0, w as i64 - 1) as usize;
                    acc += src[r * w + c] * kernel[ki * kw + kj];
                }
            }
            out[row * w + col] = acc;
        }
    }
    out
}

// ── Canny constants ───────────────────────────────────────────────────────────

/// 5×5 Gaussian kernel (sigma ≈ 1.4) for Canny pre-smoothing.
#[rustfmt::skip]
const GAUSS5X5: [f32; 25] = [
    2.0/159.0,  4.0/159.0,  5.0/159.0,  4.0/159.0,  2.0/159.0,
    4.0/159.0,  9.0/159.0, 12.0/159.0,  9.0/159.0,  4.0/159.0,
    5.0/159.0, 12.0/159.0, 15.0/159.0, 12.0/159.0,  5.0/159.0,
    4.0/159.0,  9.0/159.0, 12.0/159.0,  9.0/159.0,  4.0/159.0,
    2.0/159.0,  4.0/159.0,  5.0/159.0,  4.0/159.0,  2.0/159.0,
];

#[rustfmt::skip]
const SOBEL_X: [f32; 9] = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
#[rustfmt::skip]
const SOBEL_Y: [f32; 9] = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];
#[rustfmt::skip]
const LAPLACIAN_3X3: [f32; 9] = [0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0];

// ── Public functions ───────────────────────────────────────────────────────────

/// Canny edge detector.
///
/// Mirrors `cv2.Canny(src, threshold1, threshold2, apertureSize=3, L2gradient=False)`.
///
/// Steps:
/// 1. Convert to grayscale (BT.601) if multi-channel.
/// 2. 5×5 Gaussian smoothing (σ ≈ 1.4).
/// 3. 3×3 Sobel gradient magnitude and direction.
/// 4. Non-maximum suppression (4 quantised directions).
/// 5. Double threshold + hysteresis edge tracking (BFS).
///
/// Returns a single-channel binary edge image (0 or 255).
#[pyfunction]
#[pyo3(name = "Canny", signature = (src, threshold1, threshold2, aperture_size = 3, l2_gradient = false))]
pub fn canny(
    py: Python<'_>,
    src: Py<PyAny>,
    threshold1: f64,
    threshold2: f64,
    aperture_size: i32,
    l2_gradient: bool,
) -> PyResult<Py<PyAny>> {
    let _ = aperture_size; // only 3×3 Sobel is implemented
    let (data, h, w, ch) = extract_img(py, &src)?;

    // Step 1: grayscale
    let gray: Vec<f32> = to_grayscale(&data, h, w, ch)
        .into_iter()
        .map(|v| v as f32)
        .collect();

    // Step 2: Gaussian smoothing
    let smoothed = convolve_f32(&gray, h, w, &GAUSS5X5, 5, 5);

    // Step 3: Sobel gradients
    let gx = convolve_f32(&smoothed, h, w, &SOBEL_X, 3, 3);
    let gy = convolve_f32(&smoothed, h, w, &SOBEL_Y, 3, 3);

    let n = h * w;
    let mut magnitude = vec![0.0f32; n];
    let mut angle = vec![0.0f32; n]; // quantised: 0 / 45 / 90 / 135 degrees
    for i in 0..n {
        let mx = gx[i];
        let my = gy[i];
        magnitude[i] = if l2_gradient {
            (mx * mx + my * my).sqrt()
        } else {
            mx.abs() + my.abs()
        };
        // Angle in [0, 180) degrees
        let deg = my.atan2(mx).to_degrees();
        let deg = if deg < 0.0 { deg + 180.0 } else { deg };
        angle[i] = if deg < 22.5 || deg >= 157.5 {
            0.0
        } else if deg < 67.5 {
            45.0
        } else if deg < 112.5 {
            90.0
        } else {
            135.0
        };
    }

    // Step 4: Non-maximum suppression
    let low = threshold1 as f32;
    let high = threshold2 as f32;
    // 0=suppressed, 1=weak, 2=strong
    let mut nms = vec![0u8; n];
    for row in 1..h.saturating_sub(1) {
        for col in 1..w.saturating_sub(1) {
            let idx = row * w + col;
            let mag = magnitude[idx];
            let (n1, n2) = match angle[idx] as u32 {
                0 => (idx.wrapping_sub(1), idx + 1),
                45 => (
                    row.wrapping_sub(1) * w + (col + 1),
                    (row + 1) * w + col.wrapping_sub(1),
                ),
                90 => (row.wrapping_sub(1) * w + col, (row + 1) * w + col),
                _ => (
                    row.wrapping_sub(1) * w + col.wrapping_sub(1),
                    (row + 1) * w + (col + 1),
                ),
            };
            let m1 = if n1 < n { magnitude[n1] } else { 0.0 };
            let m2 = if n2 < n { magnitude[n2] } else { 0.0 };
            if mag >= m1 && mag >= m2 {
                nms[idx] = if mag >= high {
                    2
                } else if mag >= low {
                    1
                } else {
                    0
                };
            }
        }
    }

    // Step 5: Hysteresis — BFS flood from strong edges
    let mut edges = vec![0u8; n];
    let mut queue: VecDeque<usize> = VecDeque::new();
    for i in 0..n {
        if nms[i] == 2 {
            edges[i] = 255;
            queue.push_back(i);
        }
    }
    while let Some(idx) = queue.pop_front() {
        let row = idx / w;
        let col = idx % w;
        for dr in [-1i64, 0, 1] {
            for dc in [-1i64, 0, 1] {
                if dr == 0 && dc == 0 {
                    continue;
                }
                let nr = row as i64 + dr;
                let nc = col as i64 + dc;
                if nr < 0 || nr >= h as i64 || nc < 0 || nc >= w as i64 {
                    continue;
                }
                let ni = nr as usize * w + nc as usize;
                if nms[ni] == 1 && edges[ni] == 0 {
                    edges[ni] = 255;
                    queue.push_back(ni);
                }
            }
        }
    }

    make_image_output(py, edges, h, w, 1)
}

/// Sobel derivative operator.
///
/// Mirrors `cv2.Sobel(src, ddepth, dx, dy, ksize=3, scale=1, delta=0)`.
/// - `dx=1, dy=0` → horizontal gradient
/// - `dx=0, dy=1` → vertical gradient
/// - Both non-zero → L2 magnitude of both gradients
///
/// Only 3×3 kernels are implemented. Output depth is always uint8.
#[pyfunction]
#[pyo3(name = "Sobel", signature = (src, ddepth, dx, dy, ksize = 3, scale = 1.0, delta = 0.0))]
pub fn sobel(
    py: Python<'_>,
    src: Py<PyAny>,
    ddepth: i32,
    dx: i32,
    dy: i32,
    ksize: i32,
    scale: f64,
    delta: f64,
) -> PyResult<Py<PyAny>> {
    let _ = (ksize, ddepth); // 3×3 only
    let (data, h, w, ch) = extract_img(py, &src)?;

    if dx == 1 && dy == 0 {
        let mut out = vec![0u8; h * w * ch];
        for c in 0..ch {
            let plane: Vec<f32> = (0..h * w).map(|i| data[i * ch + c] as f32).collect();
            let conv = convolve_f32(&plane, h, w, &SOBEL_X, 3, 3);
            for i in 0..h * w {
                out[i * ch + c] =
                    ((conv[i] * scale as f32 + delta as f32).abs()).clamp(0.0, 255.0) as u8;
            }
        }
        return make_image_output(py, out, h, w, ch);
    }

    if dx == 0 && dy == 1 {
        let mut out = vec![0u8; h * w * ch];
        for c in 0..ch {
            let plane: Vec<f32> = (0..h * w).map(|i| data[i * ch + c] as f32).collect();
            let conv = convolve_f32(&plane, h, w, &SOBEL_Y, 3, 3);
            for i in 0..h * w {
                out[i * ch + c] =
                    ((conv[i] * scale as f32 + delta as f32).abs()).clamp(0.0, 255.0) as u8;
            }
        }
        return make_image_output(py, out, h, w, ch);
    }

    // Both derivatives: L2 magnitude
    let mut out = vec![0u8; h * w * ch];
    for c in 0..ch {
        let plane: Vec<f32> = (0..h * w).map(|i| data[i * ch + c] as f32).collect();
        let gx_conv = convolve_f32(&plane, h, w, &SOBEL_X, 3, 3);
        let gy_conv = convolve_f32(&plane, h, w, &SOBEL_Y, 3, 3);
        for i in 0..h * w {
            let mag = (gx_conv[i] * gx_conv[i] + gy_conv[i] * gy_conv[i]).sqrt();
            out[i * ch + c] = (mag * scale as f32 + delta as f32).clamp(0.0, 255.0) as u8;
        }
    }
    make_image_output(py, out, h, w, ch)
}

/// Laplacian second-derivative operator.
///
/// Mirrors `cv2.Laplacian(src, ddepth, ksize=1, scale=1, delta=0)`.
/// Uses the 3×3 discrete Laplacian kernel `[[0,1,0],[1,-4,1],[0,1,0]]`.
/// The absolute value of each result is taken before scaling.
#[pyfunction]
#[pyo3(name = "Laplacian", signature = (src, ddepth, ksize = 1, scale = 1.0, delta = 0.0))]
pub fn laplacian(
    py: Python<'_>,
    src: Py<PyAny>,
    ddepth: i32,
    ksize: i32,
    scale: f64,
    delta: f64,
) -> PyResult<Py<PyAny>> {
    let _ = (ksize, ddepth); // only 3×3 is supported
    let (data, h, w, ch) = extract_img(py, &src)?;

    let mut out = vec![0u8; h * w * ch];
    for c in 0..ch {
        let plane: Vec<f32> = (0..h * w).map(|i| data[i * ch + c] as f32).collect();
        let conv = convolve_f32(&plane, h, w, &LAPLACIAN_3X3, 3, 3);
        for i in 0..h * w {
            out[i * ch + c] = (conv[i].abs() * scale as f32 + delta as f32).clamp(0.0, 255.0) as u8;
        }
    }

    make_image_output(py, out, h, w, ch)
}

/// Register all edge-detection functions with a Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(canny, m)?)?;
    m.add_function(wrap_pyfunction!(sobel, m)?)?;
    m.add_function(wrap_pyfunction!(laplacian, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── to_grayscale ──────────────────────────────────────────────────────────

    #[test]
    fn test_to_grayscale_single_channel_passthrough() {
        let data = vec![10u8, 50, 200, 100];
        let gray = to_grayscale(&data, 2, 2, 1);
        assert_eq!(
            gray, data,
            "single-channel image should pass through unchanged"
        );
    }

    #[test]
    fn test_to_grayscale_white_bgr_gives_255() {
        // White BGR pixel: all channels 255 → gray should be 255
        let data = vec![255u8, 255, 255];
        let gray = to_grayscale(&data, 1, 1, 3);
        assert_eq!(gray[0], 255, "white BGR pixel should give gray = 255");
    }

    #[test]
    fn test_to_grayscale_black_bgr_gives_0() {
        let data = vec![0u8, 0, 0];
        let gray = to_grayscale(&data, 1, 1, 3);
        assert_eq!(gray[0], 0, "black BGR pixel should give gray = 0");
    }

    #[test]
    fn test_to_grayscale_pure_blue_bgr() {
        // Pure blue in BGR: [255, 0, 0] → BT.601: 0.114 * 255 ≈ 29
        let data = vec![255u8, 0, 0];
        let gray = to_grayscale(&data, 1, 1, 3);
        let expected = (0.114f32 * 255.0) as u8;
        assert!(
            (gray[0] as i32 - expected as i32).abs() <= 2,
            "pure blue BGR: expected ~{} got {}",
            expected,
            gray[0]
        );
    }

    #[test]
    fn test_to_grayscale_output_length_matches_pixel_count() {
        // 4x6 3-channel image → 4*6 = 24 grayscale pixels
        let data = vec![100u8; 4 * 6 * 3];
        let gray = to_grayscale(&data, 4, 6, 3);
        assert_eq!(gray.len(), 4 * 6, "grayscale output length should be h*w");
    }

    // ── convolve_f32 ──────────────────────────────────────────────────────────

    #[test]
    fn test_convolve_identity_kernel() {
        // 3x3 identity kernel (1 in centre, zeros elsewhere)
        let kernel = [0.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let src: Vec<f32> = (0..25).map(|i| i as f32).collect();
        let result = convolve_f32(&src, 5, 5, &kernel, 3, 3);
        // Interior pixels should be unchanged; border pixels will use clamped padding
        assert!(
            (result[12] - src[12]).abs() < 0.01,
            "centre pixel should be unchanged after identity convolution"
        );
    }

    #[test]
    fn test_convolve_box_blur_constant_image() {
        // Box blur (all ones / 9) on a constant image should give the same constant
        let kernel = [1.0f32 / 9.0; 9];
        let src = vec![90.0f32; 5 * 5];
        let result = convolve_f32(&src, 5, 5, &kernel, 3, 3);
        for v in &result {
            assert!(
                (v - 90.0).abs() < 0.5,
                "box blur on constant image should preserve value, got {}",
                v
            );
        }
    }

    #[test]
    fn test_convolve_output_size_matches_input() {
        let src = vec![0.0f32; 8 * 6];
        let kernel = [1.0f32; 9];
        let result = convolve_f32(&src, 8, 6, &kernel, 3, 3);
        assert_eq!(result.len(), 8 * 6, "output should be same size as input");
    }

    // ── Sobel gradient via convolve_f32 ───────────────────────────────────────

    #[test]
    fn test_sobel_y_detects_horizontal_edge() {
        // Top half dark (0), bottom half bright (255) — SOBEL_Y should give large response
        // at the boundary row.
        let mut src = vec![0.0f32; 10 * 10];
        for idx in (5 * 10)..(10 * 10) {
            src[idx] = 255.0;
        }
        let gy = convolve_f32(&src, 10, 10, &SOBEL_Y, 3, 3);
        // At the boundary (row 4 or 5) the gradient should be significant
        let boundary_val = gy[4 * 10 + 5].abs();
        assert!(
            boundary_val > 50.0,
            "horizontal edge should produce Sobel Y gradient > 50, got {}",
            boundary_val
        );
    }

    #[test]
    fn test_sobel_x_detects_vertical_edge() {
        // Left half dark (0), right half bright (255)
        let mut src = vec![0.0f32; 10 * 10];
        for y in 0..10usize {
            for x in 5..10usize {
                src[y * 10 + x] = 255.0;
            }
        }
        let gx = convolve_f32(&src, 10, 10, &SOBEL_X, 3, 3);
        let boundary_val = gx[5 * 10 + 4].abs();
        assert!(
            boundary_val > 50.0,
            "vertical edge should produce Sobel X gradient > 50, got {}",
            boundary_val
        );
    }

    #[test]
    fn test_sobel_constant_image_gives_zero_gradient() {
        // Constant image has no edges — both Sobel gradients should be zero everywhere
        let src = vec![128.0f32; 10 * 10];
        let gx = convolve_f32(&src, 10, 10, &SOBEL_X, 3, 3);
        let gy = convolve_f32(&src, 10, 10, &SOBEL_Y, 3, 3);
        for v in gx.iter().chain(gy.iter()) {
            assert!(
                v.abs() < 0.01,
                "constant image should have zero gradient, got {}",
                v
            );
        }
    }

    // ── Laplacian via convolve_f32 ────────────────────────────────────────────

    #[test]
    fn test_laplacian_constant_image_is_zero() {
        // Laplacian of a constant image should be zero everywhere
        let src = vec![128.0f32; 10 * 10];
        let lap = convolve_f32(&src, 10, 10, &LAPLACIAN_3X3, 3, 3);
        // Interior pixels should be exactly zero
        for row in 1..9usize {
            for col in 1..9usize {
                let v = lap[row * 10 + col];
                assert!(
                    v.abs() < 0.01,
                    "Laplacian of constant image should be ~0, got {} at ({},{})",
                    v,
                    row,
                    col
                );
            }
        }
    }

    #[test]
    fn test_laplacian_isolated_bright_pixel_gives_negative_response() {
        // A single bright pixel surrounded by zeros — Laplacian centre is -4 * pixel
        // (kernel centre weight is -4, neighbours are 0 since they are zeros)
        let mut src = vec![0.0f32; 7 * 7];
        src[3 * 7 + 3] = 255.0; // centre pixel
        let lap = convolve_f32(&src, 7, 7, &LAPLACIAN_3X3, 3, 3);
        let centre_lap = lap[3 * 7 + 3];
        assert!(
            centre_lap < -100.0,
            "Laplacian at isolated bright pixel should be strongly negative, got {}",
            centre_lap
        );
    }
}
