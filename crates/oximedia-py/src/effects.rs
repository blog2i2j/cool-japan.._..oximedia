//! Python bindings for video and color effects.
//!
//! Provides per-pixel image processing operations including color grading,
//! chroma key removal, Gaussian-style blur, and vignette effects.
//! All operations work on raw RGB24 byte buffers.

use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// Color grading
// ---------------------------------------------------------------------------

/// Apply color grade to raw RGB24 frame data.
///
/// Adjusts contrast, brightness, and saturation per pixel.
///
/// # Arguments
/// * `data` - RGB24 bytes (`width * height * 3`)
/// * `width` - Frame width in pixels
/// * `height` - Frame height in pixels
/// * `contrast` - Contrast adjustment [-1.0, 1.0] (0 = no change)
/// * `brightness` - Brightness adjustment [-1.0, 1.0] (0 = no change)
/// * `saturation` - Saturation scale [0.0, 2.0] (1.0 = no change)
#[pyfunction]
#[pyo3(signature = (data, width, height, contrast=0.0, brightness=0.0, saturation=1.0))]
pub fn apply_color_grade(
    data: Vec<u8>,
    width: usize,
    height: usize,
    contrast: f32,
    brightness: f32,
    saturation: f32,
) -> PyResult<Vec<u8>> {
    let expected = width * height * 3;
    if data.len() != expected {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Expected {} bytes for {}x{} RGB24, got {}",
            expected,
            width,
            height,
            data.len()
        )));
    }

    // Contrast multiplier: maps [-1,1] → slope around 0.5
    // contrast = 0 → slope 1.0; contrast = 1 → slope ~2.0; contrast = -1 → slope ~0.0
    let contrast_factor = (contrast + 1.0).max(0.0);
    let brightness_offset = brightness * 255.0;

    let mut out = data;

    for chunk in out.chunks_exact_mut(3) {
        let r = chunk[0] as f32;
        let g = chunk[1] as f32;
        let b = chunk[2] as f32;

        // Step 1: Contrast (pivot at mid-grey = 127.5)
        let r = (r - 127.5) * contrast_factor + 127.5;
        let g = (g - 127.5) * contrast_factor + 127.5;
        let b = (b - 127.5) * contrast_factor + 127.5;

        // Step 2: Brightness
        let r = r + brightness_offset;
        let g = g + brightness_offset;
        let b = b + brightness_offset;

        // Step 3: Saturation — convert to luma then lerp
        // BT.601 luma coefficients
        let luma = 0.299 * r + 0.587 * g + 0.114 * b;
        let r = luma + saturation * (r - luma);
        let g = luma + saturation * (g - luma);
        let b = luma + saturation * (b - luma);

        chunk[0] = r.clamp(0.0, 255.0) as u8;
        chunk[1] = g.clamp(0.0, 255.0) as u8;
        chunk[2] = b.clamp(0.0, 255.0) as u8;
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Chroma key
// ---------------------------------------------------------------------------

/// Apply chroma key (green screen) removal to RGB24 frame data.
///
/// Pixels whose color is within `tolerance` of the key color are made
/// transparent (alpha = 0). The output format is RGBA32.
///
/// # Arguments
/// * `data` - RGB24 bytes (`width * height * 3`)
/// * `width` - Frame width in pixels
/// * `height` - Frame height in pixels
/// * `key_r`, `key_g`, `key_b` - Key color components (default: pure green 0,255,0)
/// * `tolerance` - Maximum Euclidean distance in RGB space (0–255)
#[pyfunction]
#[pyo3(signature = (data, width, height, key_r=0, key_g=255, key_b=0, tolerance=50))]
pub fn apply_chromakey(
    data: Vec<u8>,
    width: usize,
    height: usize,
    key_r: u8,
    key_g: u8,
    key_b: u8,
    tolerance: u8,
) -> PyResult<Vec<u8>> {
    let expected = width * height * 3;
    if data.len() != expected {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Expected {} bytes for {}x{} RGB24, got {}",
            expected,
            width,
            height,
            data.len()
        )));
    }

    let tolerance_sq = (tolerance as f32) * (tolerance as f32);
    let kr = key_r as f32;
    let kg = key_g as f32;
    let kb = key_b as f32;

    let pixel_count = width * height;
    let mut out = vec![0u8; pixel_count * 4];

    for i in 0..pixel_count {
        let r = data[i * 3] as f32;
        let g = data[i * 3 + 1] as f32;
        let b = data[i * 3 + 2] as f32;

        let dist_sq = (r - kr) * (r - kr) + (g - kg) * (g - kg) + (b - kb) * (b - kb);

        let alpha = if dist_sq <= tolerance_sq { 0u8 } else { 255u8 };

        out[i * 4] = data[i * 3];
        out[i * 4 + 1] = data[i * 3 + 1];
        out[i * 4 + 2] = data[i * 3 + 2];
        out[i * 4 + 3] = alpha;
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Blur
// ---------------------------------------------------------------------------

/// Apply box blur to RGB24 frame data.
///
/// Uses a two-pass (horizontal + vertical) box blur for efficiency.
/// The effective kernel size is `2 * radius + 1`.
///
/// # Arguments
/// * `data` - RGB24 bytes (`width * height * 3`)
/// * `width` - Frame width in pixels
/// * `height` - Frame height in pixels
/// * `radius` - Blur radius in pixels (default: 2)
#[pyfunction]
#[pyo3(signature = (data, width, height, radius=2))]
pub fn apply_blur(data: Vec<u8>, width: usize, height: usize, radius: u32) -> PyResult<Vec<u8>> {
    let expected = width * height * 3;
    if data.len() != expected {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Expected {} bytes for {}x{} RGB24, got {}",
            expected,
            width,
            height,
            data.len()
        )));
    }

    let r = radius as usize;
    if r == 0 {
        return Ok(data);
    }

    // Horizontal pass
    let h_pass = box_blur_horizontal(&data, width, height, r);
    // Vertical pass
    let v_pass = box_blur_vertical(&h_pass, width, height, r);

    Ok(v_pass)
}

/// Horizontal box blur pass on RGB24 data.
fn box_blur_horizontal(src: &[u8], width: usize, height: usize, radius: usize) -> Vec<u8> {
    let mut dst = vec![0u8; width * height * 3];
    let kernel_size = 2 * radius + 1;

    for y in 0..height {
        // Build initial window sum for each channel
        let mut sum = [0i32; 3];
        // Initialize with the leftmost kernel_size pixels (clamped)
        for kx in 0..kernel_size {
            let sx = (kx as i32 - radius as i32).clamp(0, width as i32 - 1) as usize;
            for c in 0..3 {
                sum[c] += src[(y * width + sx) * 3 + c] as i32;
            }
        }

        for x in 0..width {
            for c in 0..3 {
                dst[(y * width + x) * 3 + c] = (sum[c] / kernel_size as i32) as u8;
            }

            // Slide window: remove left edge, add right edge
            let remove_x = (x as i32 - radius as i32).clamp(0, width as i32 - 1) as usize;
            let add_x = (x as i32 + radius as i32 + 1).clamp(0, width as i32 - 1) as usize;
            for c in 0..3 {
                sum[c] -= src[(y * width + remove_x) * 3 + c] as i32;
                sum[c] += src[(y * width + add_x) * 3 + c] as i32;
            }
        }
    }

    dst
}

/// Vertical box blur pass on RGB24 data.
fn box_blur_vertical(src: &[u8], width: usize, height: usize, radius: usize) -> Vec<u8> {
    let mut dst = vec![0u8; width * height * 3];
    let kernel_size = 2 * radius + 1;

    for x in 0..width {
        let mut sum = [0i32; 3];
        // Initialize window
        for ky in 0..kernel_size {
            let sy = (ky as i32 - radius as i32).clamp(0, height as i32 - 1) as usize;
            for c in 0..3 {
                sum[c] += src[(sy * width + x) * 3 + c] as i32;
            }
        }

        for y in 0..height {
            for c in 0..3 {
                dst[(y * width + x) * 3 + c] = (sum[c] / kernel_size as i32) as u8;
            }

            let remove_y = (y as i32 - radius as i32).clamp(0, height as i32 - 1) as usize;
            let add_y = (y as i32 + radius as i32 + 1).clamp(0, height as i32 - 1) as usize;
            for c in 0..3 {
                sum[c] -= src[(remove_y * width + x) * 3 + c] as i32;
                sum[c] += src[(add_y * width + x) * 3 + c] as i32;
            }
        }
    }

    dst
}

// ---------------------------------------------------------------------------
// Vignette
// ---------------------------------------------------------------------------

/// Apply a vignette effect to RGB24 frame data.
///
/// Darkens the edges of the frame based on distance from the center.
/// The falloff is computed as `1 - strength * (distance / max_distance)^2`.
///
/// # Arguments
/// * `data` - RGB24 bytes (`width * height * 3`)
/// * `width` - Frame width in pixels
/// * `height` - Frame height in pixels
/// * `strength` - Vignette strength [0.0, 1.0] (0 = no vignette, 1 = full black edges)
#[pyfunction]
#[pyo3(signature = (data, width, height, strength=0.5))]
pub fn apply_vignette(
    data: Vec<u8>,
    width: usize,
    height: usize,
    strength: f32,
) -> PyResult<Vec<u8>> {
    let expected = width * height * 3;
    if data.len() != expected {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Expected {} bytes for {}x{} RGB24, got {}",
            expected,
            width,
            height,
            data.len()
        )));
    }

    if width == 0 || height == 0 {
        return Ok(data);
    }

    let cx = (width as f32 - 1.0) / 2.0;
    let cy = (height as f32 - 1.0) / 2.0;
    // Normalise against half-diagonal so corners = max distance
    let max_dist = (cx * cx + cy * cy).sqrt().max(1.0);
    let strength_clamped = strength.clamp(0.0, 1.0);

    let mut out = data;

    for y in 0..height {
        for x in 0..width {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            let t = dist / max_dist; // 0 at center, ~1 at corners
                                     // Quadratic falloff
            let factor = 1.0 - strength_clamped * t * t;
            let factor = factor.clamp(0.0, 1.0);

            let idx = (y * width + x) * 3;
            out[idx] = (out[idx] as f32 * factor) as u8;
            out[idx + 1] = (out[idx + 1] as f32 * factor) as u8;
            out[idx + 2] = (out[idx + 2] as f32 * factor) as u8;
        }
    }

    Ok(out)
}
