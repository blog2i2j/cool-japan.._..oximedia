//! WebAssembly bindings for video/image scaling utilities.
//!
//! Provides functions for upscaling, downscaling, and comparing
//! scaling algorithms in the browser.

use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Upscale an RGB frame using a simple bilinear algorithm.
///
/// # Arguments
/// * `frame` - Input RGB frame data.
/// * `src_width` - Source width.
/// * `src_height` - Source height.
/// * `dst_width` - Target width.
/// * `dst_height` - Target height.
///
/// # Returns
/// Upscaled RGB frame data.
#[wasm_bindgen]
pub fn wasm_upscale_frame(
    frame: &[u8],
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
) -> Result<Vec<u8>, JsValue> {
    validate_dimensions(src_width, src_height, dst_width, dst_height)?;

    let sw = src_width as usize;
    let sh = src_height as usize;
    let expected = sw * sh * 3;

    if frame.len() < expected {
        return Err(crate::utils::js_err(&format!(
            "Frame too small: need {} bytes, got {}",
            expected,
            frame.len()
        )));
    }

    let dw = dst_width as usize;
    let dh = dst_height as usize;
    let mut output = vec![0u8; dw * dh * 3];

    // Bilinear interpolation
    for dy in 0..dh {
        let sy_f = dy as f64 * (sh as f64 - 1.0) / (dh as f64 - 1.0).max(1.0);
        let sy0 = sy_f.floor() as usize;
        let sy1 = (sy0 + 1).min(sh - 1);
        let fy = sy_f - sy0 as f64;

        for dx in 0..dw {
            let sx_f = dx as f64 * (sw as f64 - 1.0) / (dw as f64 - 1.0).max(1.0);
            let sx0 = sx_f.floor() as usize;
            let sx1 = (sx0 + 1).min(sw - 1);
            let fx = sx_f - sx0 as f64;

            for c in 0..3 {
                let p00 = frame[(sy0 * sw + sx0) * 3 + c] as f64;
                let p10 = frame[(sy0 * sw + sx1) * 3 + c] as f64;
                let p01 = frame[(sy1 * sw + sx0) * 3 + c] as f64;
                let p11 = frame[(sy1 * sw + sx1) * 3 + c] as f64;

                let val = p00 * (1.0 - fx) * (1.0 - fy)
                    + p10 * fx * (1.0 - fy)
                    + p01 * (1.0 - fx) * fy
                    + p11 * fx * fy;

                output[(dy * dw + dx) * 3 + c] = val.round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    Ok(output)
}

/// Downscale an RGB frame using area averaging.
///
/// # Arguments
/// * `frame` - Input RGB frame data.
/// * `src_width` - Source width.
/// * `src_height` - Source height.
/// * `dst_width` - Target width.
/// * `dst_height` - Target height.
///
/// # Returns
/// Downscaled RGB frame data.
#[wasm_bindgen]
pub fn wasm_downscale_frame(
    frame: &[u8],
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
) -> Result<Vec<u8>, JsValue> {
    validate_dimensions(src_width, src_height, dst_width, dst_height)?;

    let sw = src_width as usize;
    let sh = src_height as usize;
    let expected = sw * sh * 3;

    if frame.len() < expected {
        return Err(crate::utils::js_err(&format!(
            "Frame too small: need {} bytes, got {}",
            expected,
            frame.len()
        )));
    }

    let dw = dst_width as usize;
    let dh = dst_height as usize;
    let mut output = vec![0u8; dw * dh * 3];

    let x_ratio = sw as f64 / dw as f64;
    let y_ratio = sh as f64 / dh as f64;

    for dy in 0..dh {
        let sy_start = (dy as f64 * y_ratio).floor() as usize;
        let sy_end = (((dy + 1) as f64 * y_ratio).ceil() as usize).min(sh);

        for dx in 0..dw {
            let sx_start = (dx as f64 * x_ratio).floor() as usize;
            let sx_end = (((dx + 1) as f64 * x_ratio).ceil() as usize).min(sw);

            let mut sum = [0.0_f64; 3];
            let mut count = 0.0_f64;

            for sy in sy_start..sy_end {
                for sx in sx_start..sx_end {
                    for c in 0..3 {
                        sum[c] += frame[(sy * sw + sx) * 3 + c] as f64;
                    }
                    count += 1.0;
                }
            }

            if count > 0.0 {
                for c in 0..3 {
                    output[(dy * dw + dx) * 3 + c] =
                        (sum[c] / count).round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }

    Ok(output)
}

/// Get available scaling algorithms as JSON.
#[wasm_bindgen]
pub fn wasm_scaling_algorithms() -> String {
    "[\"bilinear\",\"bicubic\",\"lanczos\"]".to_string()
}

/// Calculate output dimensions with aspect ratio preservation.
///
/// # Arguments
/// * `src_width` - Source width.
/// * `src_height` - Source height.
/// * `dst_width` - Target width.
/// * `dst_height` - Target height.
/// * `mode` - Aspect mode: stretch, letterbox, crop.
///
/// # Returns
/// JSON with calculated dimensions.
#[wasm_bindgen]
pub fn wasm_calculate_dimensions(
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    mode: &str,
) -> Result<String, JsValue> {
    validate_dimensions(src_width, src_height, dst_width, dst_height)?;

    let aspect_mode = match mode {
        "stretch" => oximedia_scaling::AspectRatioMode::Stretch,
        "letterbox" => oximedia_scaling::AspectRatioMode::Letterbox,
        "crop" => oximedia_scaling::AspectRatioMode::Crop,
        other => {
            return Err(crate::utils::js_err(&format!(
                "Unknown mode '{}'. Supported: stretch, letterbox, crop",
                other
            )));
        }
    };

    let params =
        oximedia_scaling::ScalingParams::new(dst_width, dst_height).with_aspect_ratio(aspect_mode);
    let scaler = oximedia_scaling::VideoScaler::new(params);
    let (out_w, out_h) = scaler.calculate_dimensions(src_width, src_height);

    Ok(format!(
        "{{\"output_width\":{out_w},\"output_height\":{out_h},\"mode\":\"{mode}\"}}"
    ))
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn validate_dimensions(
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
) -> Result<(), JsValue> {
    if src_width == 0 || src_height == 0 || dst_width == 0 || dst_height == 0 {
        return Err(crate::utils::js_err("All dimensions must be > 0"));
    }
    if src_width > 7680 || src_height > 4320 || dst_width > 7680 || dst_height > 4320 {
        return Err(crate::utils::js_err("Dimensions exceed 7680x4320"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_upscale_frame() {
        let frame = vec![128u8; 4 * 4 * 3];
        let result = wasm_upscale_frame(&frame, 4, 4, 8, 8);
        assert!(result.is_ok());
        let output = result.expect("should succeed");
        assert_eq!(output.len(), 8 * 8 * 3);
    }

    #[test]
    fn test_downscale_frame() {
        let frame = vec![200u8; 8 * 8 * 3];
        let result = wasm_downscale_frame(&frame, 8, 8, 4, 4);
        assert!(result.is_ok());
        let output = result.expect("should succeed");
        assert_eq!(output.len(), 4 * 4 * 3);
        // All pixels were 200, so downscaled should be ~200
        assert!((output[0] as i32 - 200).abs() < 2);
    }

    #[test]
    fn test_scaling_algorithms() {
        let json = wasm_scaling_algorithms();
        assert!(json.contains("lanczos"));
    }

    #[test]
    fn test_calculate_dimensions_stretch() {
        let result = wasm_calculate_dimensions(3840, 2160, 1920, 1080, "stretch");
        assert!(result.is_ok());
        let json = result.expect("should succeed");
        assert!(json.contains("1920"));
        assert!(json.contains("1080"));
    }

    #[test]
    fn test_invalid_dimensions() {
        let result = wasm_upscale_frame(&[0u8; 12], 0, 0, 10, 10);
        assert!(result.is_err());
    }
}
