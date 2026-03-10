//! WebAssembly bindings for visual effects and compositing.
//!
//! Provides standalone functions for applying effects, chroma keying,
//! dissolve transitions, and test pattern generation in the browser.

use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Effect application
// ---------------------------------------------------------------------------

/// Apply a visual effect to an RGB frame.
///
/// `data` must contain `w * h * 3` bytes of RGB data.
/// `effect` is one of: blur, sharpen, sepia, negative, posterize, pixelate, vintage.
/// `strength` is 0.0-1.0.
///
/// Returns the processed RGB frame.
#[wasm_bindgen]
pub fn wasm_apply_effect(
    data: &[u8],
    w: u32,
    h: u32,
    effect: &str,
    strength: f64,
) -> Result<Vec<u8>, JsValue> {
    let expected = (w as usize) * (h as usize) * 3;
    if data.len() < expected {
        return Err(crate::utils::js_err(&format!(
            "Frame data too small: need {} bytes, got {}",
            expected,
            data.len()
        )));
    }
    if !(0.0..=1.0).contains(&strength) {
        return Err(crate::utils::js_err("strength must be between 0.0 and 1.0"));
    }

    let mut output = data[..expected].to_vec();

    match effect {
        "sepia" => apply_sepia(&mut output, strength),
        "negative" => apply_negative(&mut output),
        "posterize" => {
            let levels = (4.0 + strength * 12.0) as u8;
            apply_posterize(&mut output, levels);
        }
        "pixelate" => {
            let bs = (2.0 + strength * 30.0) as usize;
            apply_pixelate(&mut output, w as usize, h as usize, bs);
        }
        "blur" => {
            let radius = (1.0 + strength * 10.0) as usize;
            output = apply_box_blur(&output, w as usize, h as usize, radius);
        }
        "sharpen" => {
            output = apply_sharpen(&output, w as usize, h as usize, strength);
        }
        "vintage" => apply_vintage(&mut output, strength),
        "brightness" => {
            let adj = strength * 2.0 - 1.0;
            apply_brightness(&mut output, adj);
        }
        "contrast" => {
            apply_contrast(&mut output, strength * 2.0);
        }
        _ => {
            return Err(crate::utils::js_err(&format!(
                "Unknown effect '{}'. Available: blur, sharpen, sepia, negative, posterize, pixelate, vintage, brightness, contrast",
                effect
            )));
        }
    }

    Ok(output)
}

/// Apply chroma key compositing.
///
/// `data` (foreground) and `bg` (background) must each be `w * h * 3` RGB bytes.
/// `key_r`, `key_g`, `key_b` define the key color.
/// `tolerance` is the color distance threshold (0.0-1.0).
///
/// Returns the composited RGB frame.
#[wasm_bindgen]
pub fn wasm_chroma_key(
    data: &[u8],
    w: u32,
    h: u32,
    key_r: u8,
    key_g: u8,
    key_b: u8,
    tolerance: f64,
    bg: &[u8],
) -> Result<Vec<u8>, JsValue> {
    let expected = (w as usize) * (h as usize) * 3;
    if data.len() < expected {
        return Err(crate::utils::js_err(&format!(
            "Foreground too small: need {} bytes, got {}",
            expected,
            data.len()
        )));
    }
    if bg.len() < expected {
        return Err(crate::utils::js_err(&format!(
            "Background too small: need {} bytes, got {}",
            expected,
            bg.len()
        )));
    }
    if !(0.0..=1.0).contains(&tolerance) {
        return Err(crate::utils::js_err(
            "tolerance must be between 0.0 and 1.0",
        ));
    }

    let tol = tolerance * 441.67; // sqrt(255^2 * 3)
    let softness = 0.1 * 441.67;

    let mut output = vec![0u8; expected];

    for i in (0..expected).step_by(3) {
        let fr = data[i] as f64;
        let fg = data[i + 1] as f64;
        let fb = data[i + 2] as f64;

        let dr = fr - key_r as f64;
        let dg = fg - key_g as f64;
        let db = fb - key_b as f64;
        let dist = (dr * dr + dg * dg + db * db).sqrt();

        let alpha = if dist < tol {
            0.0
        } else if dist < tol + softness {
            (dist - tol) / softness.max(0.001)
        } else {
            1.0
        };

        output[i] = (fr * alpha + bg[i] as f64 * (1.0 - alpha))
            .round()
            .min(255.0) as u8;
        output[i + 1] = (fg * alpha + bg[i + 1] as f64 * (1.0 - alpha))
            .round()
            .min(255.0) as u8;
        output[i + 2] = (fb * alpha + bg[i + 2] as f64 * (1.0 - alpha))
            .round()
            .min(255.0) as u8;
    }

    Ok(output)
}

/// Cross-dissolve between two RGB frames.
///
/// Both `frame1` and `frame2` must be `w * h * 3` bytes.
/// `progress` is 0.0 (all frame1) to 1.0 (all frame2).
#[wasm_bindgen]
pub fn wasm_dissolve(
    frame1: &[u8],
    frame2: &[u8],
    w: u32,
    h: u32,
    progress: f64,
) -> Result<Vec<u8>, JsValue> {
    let expected = (w as usize) * (h as usize) * 3;
    if frame1.len() < expected {
        return Err(crate::utils::js_err(&format!(
            "frame1 too small: need {} bytes, got {}",
            expected,
            frame1.len()
        )));
    }
    if frame2.len() < expected {
        return Err(crate::utils::js_err(&format!(
            "frame2 too small: need {} bytes, got {}",
            expected,
            frame2.len()
        )));
    }

    let p = progress.clamp(0.0, 1.0);
    let inv = 1.0 - p;

    let mut output = vec![0u8; expected];
    for i in 0..expected {
        output[i] = (frame1[i] as f64 * inv + frame2[i] as f64 * p).round() as u8;
    }

    Ok(output)
}

/// Generate a test pattern as RGB data.
///
/// `pattern` is one of: color_bars, gradient, noise, checkerboard, solid.
///
/// Returns `w * h * 3` bytes of RGB data.
#[wasm_bindgen]
pub fn wasm_generate_pattern(pattern: &str, w: u32, h: u32) -> Result<Vec<u8>, JsValue> {
    if w == 0 || h == 0 {
        return Err(crate::utils::js_err("width and height must be > 0"));
    }

    let width = w as usize;
    let height = h as usize;
    let mut data = vec![0u8; width * height * 3];

    match pattern {
        "color_bars" => generate_color_bars(&mut data, width, height),
        "gradient" => generate_gradient(&mut data, width, height),
        "noise" => generate_noise(&mut data, width, height),
        "checkerboard" => generate_checkerboard(&mut data, width, height, 64),
        "solid" => {
            for pixel in data.chunks_exact_mut(3) {
                pixel[0] = 128;
                pixel[1] = 128;
                pixel[2] = 128;
            }
        }
        _ => {
            return Err(crate::utils::js_err(&format!(
                "Unknown pattern '{}'. Available: color_bars, gradient, noise, checkerboard, solid",
                pattern
            )));
        }
    }

    Ok(data)
}

/// List available effects as a JSON array.
#[wasm_bindgen]
pub fn wasm_list_effects() -> String {
    "[\"blur\",\"sharpen\",\"sepia\",\"negative\",\"posterize\",\"pixelate\",\"vintage\",\"brightness\",\"contrast\"]".to_string()
}

/// List available transitions as a JSON array.
#[wasm_bindgen]
pub fn wasm_list_transitions() -> String {
    "[\"dissolve\",\"wipe_left\",\"wipe_right\",\"wipe_up\",\"wipe_down\",\"fade\"]".to_string()
}

/// List available test patterns as a JSON array.
#[wasm_bindgen]
pub fn wasm_list_patterns() -> String {
    "[\"color_bars\",\"gradient\",\"noise\",\"checkerboard\",\"solid\"]".to_string()
}

// ---------------------------------------------------------------------------
// Internal effect implementations
// ---------------------------------------------------------------------------

fn apply_sepia(data: &mut [u8], strength: f64) {
    let s = strength.clamp(0.0, 1.0);
    for pixel in data.chunks_exact_mut(3) {
        let r = pixel[0] as f64;
        let g = pixel[1] as f64;
        let b = pixel[2] as f64;

        let sr = (0.393 * r + 0.769 * g + 0.189 * b).min(255.0);
        let sg = (0.349 * r + 0.686 * g + 0.168 * b).min(255.0);
        let sb = (0.272 * r + 0.534 * g + 0.131 * b).min(255.0);

        pixel[0] = (r * (1.0 - s) + sr * s).round() as u8;
        pixel[1] = (g * (1.0 - s) + sg * s).round() as u8;
        pixel[2] = (b * (1.0 - s) + sb * s).round() as u8;
    }
}

fn apply_negative(data: &mut [u8]) {
    for val in data.iter_mut() {
        *val = 255 - *val;
    }
}

fn apply_posterize(data: &mut [u8], levels: u8) {
    let levels = levels.max(2);
    let divisor = 256.0 / levels as f64;
    for val in data.iter_mut() {
        *val = ((*val as f64 / divisor).floor() * divisor).round() as u8;
    }
}

fn apply_pixelate(data: &mut [u8], width: usize, height: usize, block_size: usize) {
    let bs = block_size.max(1);
    for by in (0..height).step_by(bs) {
        for bx in (0..width).step_by(bs) {
            let bw = bs.min(width - bx);
            let bh = bs.min(height - by);

            let mut sr: u64 = 0;
            let mut sg: u64 = 0;
            let mut sb: u64 = 0;
            let count = (bw * bh) as u64;

            for yy in 0..bh {
                for xx in 0..bw {
                    let idx = ((by + yy) * width + (bx + xx)) * 3;
                    if idx + 2 < data.len() {
                        sr += data[idx] as u64;
                        sg += data[idx + 1] as u64;
                        sb += data[idx + 2] as u64;
                    }
                }
            }

            let ar = (sr / count.max(1)) as u8;
            let ag = (sg / count.max(1)) as u8;
            let ab = (sb / count.max(1)) as u8;

            for yy in 0..bh {
                for xx in 0..bw {
                    let idx = ((by + yy) * width + (bx + xx)) * 3;
                    if idx + 2 < data.len() {
                        data[idx] = ar;
                        data[idx + 1] = ag;
                        data[idx + 2] = ab;
                    }
                }
            }
        }
    }
}

fn apply_box_blur(data: &[u8], width: usize, height: usize, radius: usize) -> Vec<u8> {
    let r = radius.max(1).min(width / 2);
    let mut temp = data.to_vec();
    let mut output = data.to_vec();

    // Horizontal pass
    for y in 0..height {
        for x in 0..width {
            let mut sr: u64 = 0;
            let mut sg: u64 = 0;
            let mut sb: u64 = 0;
            let mut count: u64 = 0;

            let x_start = x.saturating_sub(r);
            let x_end = (x + r + 1).min(width);

            for xx in x_start..x_end {
                let idx = (y * width + xx) * 3;
                if idx + 2 < data.len() {
                    sr += data[idx] as u64;
                    sg += data[idx + 1] as u64;
                    sb += data[idx + 2] as u64;
                    count += 1;
                }
            }

            let idx = (y * width + x) * 3;
            if idx + 2 < temp.len() && count > 0 {
                temp[idx] = (sr / count) as u8;
                temp[idx + 1] = (sg / count) as u8;
                temp[idx + 2] = (sb / count) as u8;
            }
        }
    }

    // Vertical pass
    for y in 0..height {
        for x in 0..width {
            let mut sr: u64 = 0;
            let mut sg: u64 = 0;
            let mut sb: u64 = 0;
            let mut count: u64 = 0;

            let y_start = y.saturating_sub(r);
            let y_end = (y + r + 1).min(height);

            for yy in y_start..y_end {
                let idx = (yy * width + x) * 3;
                if idx + 2 < temp.len() {
                    sr += temp[idx] as u64;
                    sg += temp[idx + 1] as u64;
                    sb += temp[idx + 2] as u64;
                    count += 1;
                }
            }

            let idx = (y * width + x) * 3;
            if idx + 2 < output.len() && count > 0 {
                output[idx] = (sr / count) as u8;
                output[idx + 1] = (sg / count) as u8;
                output[idx + 2] = (sb / count) as u8;
            }
        }
    }

    output
}

fn apply_sharpen(data: &[u8], width: usize, height: usize, amount: f64) -> Vec<u8> {
    let blurred = apply_box_blur(data, width, height, 1);
    let mut output = data.to_vec();
    for i in 0..output.len() {
        let original = data[i] as f64;
        let blur_val = blurred[i] as f64;
        output[i] = (original + amount * (original - blur_val))
            .round()
            .clamp(0.0, 255.0) as u8;
    }
    output
}

fn apply_vintage(data: &mut [u8], strength: f64) {
    let s = strength.clamp(0.0, 1.0);
    for pixel in data.chunks_exact_mut(3) {
        let r = pixel[0] as f64;
        let g = pixel[1] as f64;
        let b = pixel[2] as f64;
        let avg = (r + g + b) / 3.0;
        let vr = (avg * 0.3 + r * 0.7 + 20.0).min(255.0);
        let vg = (avg * 0.3 + g * 0.7 + 5.0).min(255.0);
        let vb = (avg * 0.3 + b * 0.7 - 10.0).max(0.0).min(255.0);
        pixel[0] = (r * (1.0 - s) + vr * s).round() as u8;
        pixel[1] = (g * (1.0 - s) + vg * s).round() as u8;
        pixel[2] = (b * (1.0 - s) + vb * s).round() as u8;
    }
}

fn apply_brightness(data: &mut [u8], adjustment: f64) {
    let adj = (adjustment * 255.0).round() as i16;
    for val in data.iter_mut() {
        *val = (*val as i16 + adj).clamp(0, 255) as u8;
    }
}

fn apply_contrast(data: &mut [u8], factor: f64) {
    let f = factor.max(0.0);
    for val in data.iter_mut() {
        *val = ((*val as f64 - 128.0) * f + 128.0)
            .round()
            .clamp(0.0, 255.0) as u8;
    }
}

fn generate_color_bars(data: &mut [u8], width: usize, height: usize) {
    let bars: [(u8, u8, u8); 7] = [
        (235, 235, 235),
        (235, 235, 16),
        (16, 235, 235),
        (16, 235, 16),
        (235, 16, 235),
        (235, 16, 16),
        (16, 16, 235),
    ];
    let bar_width = width / 7;
    for y in 0..height {
        for x in 0..width {
            let bi = (x / bar_width.max(1)).min(6);
            let (r, g, b) = bars[bi];
            let idx = (y * width + x) * 3;
            if idx + 2 < data.len() {
                data[idx] = r;
                data[idx + 1] = g;
                data[idx + 2] = b;
            }
        }
    }
}

fn generate_gradient(data: &mut [u8], width: usize, height: usize) {
    for y in 0..height {
        for x in 0..width {
            let t = if width > 1 {
                x as f64 / (width - 1) as f64
            } else {
                0.0
            };
            let v = (t * 255.0).round() as u8;
            let idx = (y * width + x) * 3;
            if idx + 2 < data.len() {
                data[idx] = v;
                data[idx + 1] = v;
                data[idx + 2] = v;
            }
        }
    }
}

fn generate_noise(data: &mut [u8], width: usize, height: usize) {
    let mut state: u64 = 0x5DEE_CE66_D_u64;
    for y in 0..height {
        for x in 0..width {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let val = ((state >> 33) & 0xFF) as u8;
            let idx = (y * width + x) * 3;
            if idx + 2 < data.len() {
                data[idx] = val;
                data[idx + 1] = val;
                data[idx + 2] = val;
            }
        }
    }
}

fn generate_checkerboard(data: &mut [u8], width: usize, height: usize, block_size: usize) {
    let bs = block_size.max(1);
    for y in 0..height {
        for x in 0..width {
            let v = if ((x / bs) + (y / bs)) % 2 == 0 {
                255u8
            } else {
                0u8
            };
            let idx = (y * width + x) * 3;
            if idx + 2 < data.len() {
                data[idx] = v;
                data[idx + 1] = v;
                data[idx + 2] = v;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(w: u32, h: u32, val: u8) -> Vec<u8> {
        vec![val; (w as usize) * (h as usize) * 3]
    }

    #[test]
    fn test_apply_sepia() {
        let result = wasm_apply_effect(&make_frame(4, 4, 128), 4, 4, "sepia", 1.0);
        assert!(result.is_ok());
        let out = result.expect("should succeed");
        assert_eq!(out.len(), 4 * 4 * 3);
    }

    #[test]
    fn test_apply_negative() {
        let result = wasm_apply_effect(&make_frame(4, 4, 100), 4, 4, "negative", 1.0);
        assert!(result.is_ok());
        let out = result.expect("should succeed");
        assert_eq!(out[0], 155);
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_apply_unknown_effect() {
        let result = wasm_apply_effect(&make_frame(4, 4, 100), 4, 4, "unknown", 0.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_chroma_key() {
        let fg = vec![0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0]; // green
        let bg = vec![255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0]; // red
        let result = wasm_chroma_key(&fg, 2, 2, 0, 255, 0, 0.5, &bg);
        assert!(result.is_ok());
        let out = result.expect("should succeed");
        assert!(out[0] > 200); // should be mostly red
    }

    #[test]
    fn test_dissolve() {
        let f1 = make_frame(4, 4, 0);
        let f2 = make_frame(4, 4, 200);
        let result = wasm_dissolve(&f1, &f2, 4, 4, 0.5);
        assert!(result.is_ok());
        let out = result.expect("should succeed");
        assert_eq!(out[0], 100);
    }

    #[test]
    fn test_dissolve_boundaries() {
        let f1 = make_frame(2, 2, 100);
        let f2 = make_frame(2, 2, 200);

        let r0 = wasm_dissolve(&f1, &f2, 2, 2, 0.0).expect("valid");
        assert_eq!(r0[0], 100);

        let r1 = wasm_dissolve(&f1, &f2, 2, 2, 1.0).expect("valid");
        assert_eq!(r1[0], 200);
    }

    #[test]
    fn test_generate_pattern_color_bars() {
        let result = wasm_generate_pattern("color_bars", 70, 10);
        assert!(result.is_ok());
        let data = result.expect("should succeed");
        assert_eq!(data.len(), 70 * 10 * 3);
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_generate_pattern_unknown() {
        let result = wasm_generate_pattern("unknown", 10, 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_list_effects() {
        let effects = wasm_list_effects();
        assert!(effects.contains("blur"));
        assert!(effects.contains("sepia"));
    }

    #[test]
    fn test_list_transitions() {
        let transitions = wasm_list_transitions();
        assert!(transitions.contains("dissolve"));
        assert!(transitions.contains("wipe_left"));
    }

    #[test]
    fn test_list_patterns() {
        let patterns = wasm_list_patterns();
        assert!(patterns.contains("color_bars"));
        assert!(patterns.contains("noise"));
    }
}
