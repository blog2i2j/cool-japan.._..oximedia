//! WebAssembly bindings for video/audio denoising from `oximedia-denoise`.
//!
//! Provides standalone functions for single-frame denoising, audio denoising,
//! noise estimation, and querying available modes/presets.

use oximedia_codec::{Plane, VideoFrame};
use oximedia_core::PixelFormat;
use oximedia_denoise::{DenoiseConfig, DenoiseMode, Denoiser};
use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Video frame denoising
// ---------------------------------------------------------------------------

/// Denoise a single video frame.
///
/// `data` is raw pixel data: `width * height * channels` bytes.
/// `channels` must be 1 (grayscale) or 3 (RGB24).
/// `mode` is one of: `"fast"`, `"balanced"`, `"quality"`, `"grain_aware"`.
/// `strength` is 0.0 (none) to 1.0 (maximum).
///
/// Returns denoised pixel data in the same format as the input.
///
/// # Errors
///
/// Returns an error if the mode is unknown, data is too small, or processing
/// fails.
#[wasm_bindgen]
pub fn wasm_denoise_frame(
    data: &[u8],
    width: u32,
    height: u32,
    channels: u32,
    mode: &str,
    strength: f64,
) -> Result<Vec<u8>, JsValue> {
    let mode_enum = parse_mode(mode)?;
    let strength_val = (strength as f32).clamp(0.0, 1.0);

    let expected = (width * height * channels) as usize;
    if data.len() < expected {
        return Err(crate::utils::js_err(&format!(
            "Data too small: need {expected} bytes for {width}x{height}x{channels}, got {}",
            data.len()
        )));
    }

    if channels != 1 && channels != 3 {
        return Err(crate::utils::js_err(&format!(
            "Unsupported channel count {channels}. Use 1 (gray) or 3 (RGB)"
        )));
    }

    let config = DenoiseConfig {
        mode: mode_enum,
        strength: strength_val,
        temporal_window: 5,
        preserve_edges: true,
        preserve_grain: mode_enum == DenoiseMode::GrainAware,
    };
    config
        .validate()
        .map_err(|e| crate::utils::js_err(&format!("Invalid config: {e}")))?;

    let mut denoiser = Denoiser::new(config);

    // Build VideoFrame in YUV420P
    let mut frame = VideoFrame::new(PixelFormat::Yuv420p, width, height);
    frame.allocate();

    if channels == 3 {
        rgb24_to_yuv420p(data, width, height, &mut frame);
    } else {
        fill_grayscale_yuv(&data[..expected], width, height, &mut frame);
    }

    let denoised = denoiser
        .process(&frame)
        .map_err(|e| crate::utils::js_err(&format!("Denoise failed: {e}")))?;

    if channels == 3 {
        Ok(yuv420p_to_rgb24(&denoised, width, height))
    } else {
        if denoised.planes.is_empty() {
            return Err(crate::utils::js_err("Denoised frame has no plane data"));
        }
        Ok(denoised.planes[0].data.clone())
    }
}

// ---------------------------------------------------------------------------
// Audio denoising
// ---------------------------------------------------------------------------

/// Denoise audio samples using noise gating and spectral subtraction.
///
/// `samples` is a flat `f32` array of PCM audio.
/// `strength` is 0.0 (none) to 1.0 (maximum).
///
/// Returns denoised samples of the same length.
///
/// # Errors
///
/// Returns an error if processing fails.
#[wasm_bindgen]
pub fn wasm_denoise_audio(
    samples: &[f32],
    sample_rate: u32,
    strength: f64,
) -> Result<Vec<f32>, JsValue> {
    use oximedia_denoise::audio_denoise::AudioDenoiseFilter;

    if samples.is_empty() {
        return Ok(Vec::new());
    }

    let strength_val = (strength as f32).clamp(0.0, 1.0);
    let gate_threshold = (1.0 - strength_val) * 0.05;
    let hold_samples = (f64::from(sample_rate) * 0.05) as usize;

    let mut filter = AudioDenoiseFilter::new(gate_threshold, hold_samples);
    let block_size = 1024;
    let mut output = Vec::with_capacity(samples.len());

    for chunk in samples.chunks(block_size) {
        let denoised_block = filter.process(chunk);
        output.extend_from_slice(&denoised_block);
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Noise estimation
// ---------------------------------------------------------------------------

/// Estimate the noise level in an image.
///
/// `data` is raw pixel data (grayscale `width*height` bytes, or RGB24
/// `width*height*3` bytes).
///
/// Returns the estimated noise standard deviation (0-255 scale).
///
/// # Errors
///
/// Returns an error if data is too small.
#[wasm_bindgen]
pub fn wasm_estimate_noise(data: &[u8], width: u32, height: u32) -> Result<f64, JsValue> {
    use oximedia_denoise::noise_estimate::{NoiseEstimateMethod, NoiseEstimator};

    let expected_rgb = (width * height * 3) as usize;
    let expected_gray = (width * height) as usize;

    let luma_data = if data.len() >= expected_rgb {
        // RGB: extract luma
        let mut luma = Vec::with_capacity(expected_gray);
        for i in 0..expected_gray {
            let idx = i * 3;
            let r = f64::from(data[idx]);
            let g = f64::from(data[idx + 1]);
            let b = f64::from(data[idx + 2]);
            let y = (0.2126 * r + 0.7152 * g + 0.0722 * b).clamp(0.0, 255.0);
            luma.push(y as u8);
        }
        luma
    } else if data.len() >= expected_gray {
        data[..expected_gray].to_vec()
    } else {
        return Err(crate::utils::js_err(&format!(
            "Data too small: need at least {expected_gray} bytes, got {}",
            data.len()
        )));
    };

    let mut estimator = NoiseEstimator::new(NoiseEstimateMethod::LocalVariance);
    let estimate = estimator.estimate_from_frame(&luma_data, width as usize, height as usize);

    Ok(f64::from(estimate.sigma))
}

// ---------------------------------------------------------------------------
// Mode / preset queries
// ---------------------------------------------------------------------------

/// List available denoise modes as a JSON array with descriptions.
///
/// # Errors
///
/// Returns an error if JSON serialisation fails.
#[wasm_bindgen]
pub fn wasm_denoise_modes() -> Result<String, JsValue> {
    let modes = serde_json::json!([
        { "id": "fast", "name": "Fast", "description": "Bilateral filter, real-time capable" },
        { "id": "balanced", "name": "Balanced", "description": "Motion-compensated temporal + spatial" },
        { "id": "quality", "name": "Quality", "description": "Non-Local Means, highest quality" },
        { "id": "grain_aware", "name": "Grain-Aware", "description": "Preserves film grain texture" },
    ]);
    serde_json::to_string(&modes)
        .map_err(|e| crate::utils::js_err(&format!("JSON serialisation failed: {e}")))
}

/// List denoise presets as a JSON array with configurations.
///
/// # Errors
///
/// Returns an error if JSON serialisation fails.
#[wasm_bindgen]
pub fn wasm_denoise_presets() -> Result<String, JsValue> {
    let presets = serde_json::json!([
        {
            "name": "light",
            "mode": "fast",
            "strength": 0.3,
            "temporal_window": 5,
            "preserve_edges": true,
            "preserve_grain": false,
        },
        {
            "name": "medium",
            "mode": "balanced",
            "strength": 0.5,
            "temporal_window": 5,
            "preserve_edges": true,
            "preserve_grain": false,
        },
        {
            "name": "strong",
            "mode": "quality",
            "strength": 0.8,
            "temporal_window": 7,
            "preserve_edges": true,
            "preserve_grain": false,
        },
        {
            "name": "grain_aware",
            "mode": "grain_aware",
            "strength": 0.5,
            "temporal_window": 5,
            "preserve_edges": true,
            "preserve_grain": true,
        },
    ]);
    serde_json::to_string(&presets)
        .map_err(|e| crate::utils::js_err(&format!("JSON serialisation failed: {e}")))
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn resolve_mode(mode: &str) -> Result<DenoiseMode, String> {
    match mode {
        "fast" => Ok(DenoiseMode::Fast),
        "balanced" => Ok(DenoiseMode::Balanced),
        "quality" => Ok(DenoiseMode::Quality),
        "grain_aware" | "grain-aware" => Ok(DenoiseMode::GrainAware),
        other => Err(format!(
            "Unknown denoise mode '{other}'. Use: fast, balanced, quality, grain_aware"
        )),
    }
}

fn parse_mode(mode: &str) -> Result<DenoiseMode, JsValue> {
    resolve_mode(mode).map_err(|e| crate::utils::js_err(&e))
}

/// Convert RGB24 to YUV420P in a VideoFrame.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn rgb24_to_yuv420p(rgb: &[u8], width: u32, height: u32, frame: &mut VideoFrame) {
    let w = width as usize;
    let h = height as usize;
    let chroma_w = (w + 1) / 2;
    let chroma_h = (h + 1) / 2;

    let mut y_plane = vec![0u8; w * h];
    let mut u_plane = vec![0u8; chroma_w * chroma_h];
    let mut v_plane = vec![0u8; chroma_w * chroma_h];

    for row in 0..h {
        for col in 0..w {
            let idx = (row * w + col) * 3;
            let r = f32::from(rgb[idx]);
            let g = f32::from(rgb[idx + 1]);
            let b = f32::from(rgb[idx + 2]);
            let y = (0.2126 * r + 0.7152 * g + 0.0722 * b).clamp(0.0, 255.0);
            y_plane[row * w + col] = y as u8;
        }
    }

    for row in 0..chroma_h {
        for col in 0..chroma_w {
            let sr = (row * 2).min(h - 1);
            let sc = (col * 2).min(w - 1);
            let idx = (sr * w + sc) * 3;
            let r = f32::from(rgb[idx]);
            let g = f32::from(rgb[idx + 1]);
            let b = f32::from(rgb[idx + 2]);
            let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            let cb = ((b - y) / 1.8556 + 128.0).clamp(0.0, 255.0);
            let cr = ((r - y) / 1.5748 + 128.0).clamp(0.0, 255.0);
            u_plane[row * chroma_w + col] = cb as u8;
            v_plane[row * chroma_w + col] = cr as u8;
        }
    }

    if frame.planes.len() > 2 {
        frame.planes[0] = Plane::with_dimensions(y_plane, w, width, height);
        frame.planes[1] =
            Plane::with_dimensions(u_plane, chroma_w, chroma_w as u32, chroma_h as u32);
        frame.planes[2] =
            Plane::with_dimensions(v_plane, chroma_w, chroma_w as u32, chroma_h as u32);
    }
}

/// Fill a YUV420P frame from grayscale data (Y only, chroma neutral).
fn fill_grayscale_yuv(gray: &[u8], width: u32, height: u32, frame: &mut VideoFrame) {
    let w = width as usize;
    let h = height as usize;
    let chroma_w = (w + 1) / 2;
    let chroma_h = (h + 1) / 2;

    if frame.planes.len() > 2 {
        frame.planes[0] = Plane::with_dimensions(gray[..w * h].to_vec(), w, width, height);
        frame.planes[1] = Plane::with_dimensions(
            vec![128u8; chroma_w * chroma_h],
            chroma_w,
            chroma_w as u32,
            chroma_h as u32,
        );
        frame.planes[2] = Plane::with_dimensions(
            vec![128u8; chroma_w * chroma_h],
            chroma_w,
            chroma_w as u32,
            chroma_h as u32,
        );
    }
}

/// Convert YUV420P VideoFrame to RGB24.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn yuv420p_to_rgb24(frame: &VideoFrame, width: u32, height: u32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let chroma_w = (w + 1) / 2;

    let y_data = if !frame.planes.is_empty() {
        &frame.planes[0].data
    } else {
        return vec![0u8; w * h * 3];
    };
    let u_data = if frame.planes.len() > 1 {
        &frame.planes[1].data
    } else {
        return vec![0u8; w * h * 3];
    };
    let v_data = if frame.planes.len() > 2 {
        &frame.planes[2].data
    } else {
        return vec![0u8; w * h * 3];
    };

    let mut rgb = vec![0u8; w * h * 3];

    for row in 0..h {
        for col in 0..w {
            let y_idx = row * w + col;
            let c_idx = (row / 2) * chroma_w + (col / 2);

            let y_f = f32::from(y_data.get(y_idx).copied().unwrap_or(128));
            let cb_f = f32::from(u_data.get(c_idx).copied().unwrap_or(128)) - 128.0;
            let cr_f = f32::from(v_data.get(c_idx).copied().unwrap_or(128)) - 128.0;

            let r = (y_f + 1.5748 * cr_f).clamp(0.0, 255.0);
            let g = (y_f - 0.1873 * cb_f - 0.4681 * cr_f).clamp(0.0, 255.0);
            let b = (y_f + 1.8556 * cb_f).clamp(0.0, 255.0);

            let out = (row * w + col) * 3;
            rgb[out] = r as u8;
            rgb[out + 1] = g as u8;
            rgb[out + 2] = b as u8;
        }
    }

    rgb
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn gray_frame(width: u32, height: u32, value: u8) -> Vec<u8> {
        vec![value; (width * height) as usize]
    }

    fn rgb_frame(width: u32, height: u32) -> Vec<u8> {
        vec![128u8; (width * height * 3) as usize]
    }

    // --- Internal helper tests (run on any target) ---

    #[test]
    fn test_resolve_mode_valid() {
        assert!(resolve_mode("fast").is_ok());
        assert!(resolve_mode("balanced").is_ok());
        assert!(resolve_mode("quality").is_ok());
        assert!(resolve_mode("grain_aware").is_ok());
        assert!(resolve_mode("grain-aware").is_ok());
    }

    #[test]
    fn test_resolve_mode_invalid() {
        assert!(resolve_mode("bad").is_err());
        assert!(resolve_mode("").is_err());
    }

    #[test]
    fn test_rgb_yuv_roundtrip() {
        let data = rgb_frame(16, 16);
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 16, 16);
        frame.allocate();
        rgb24_to_yuv420p(&data, 16, 16, &mut frame);
        let rgb_back = yuv420p_to_rgb24(&frame, 16, 16);

        assert_eq!(rgb_back.len(), (16 * 16 * 3) as usize);
        // Mid-gray should survive roundtrip within tolerance
        for i in (0..rgb_back.len()).step_by(3) {
            assert!((rgb_back[i] as i16 - 128).abs() < 5);
        }
    }

    #[test]
    fn test_fill_grayscale_yuv() {
        let gray = gray_frame(16, 16, 200);
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 16, 16);
        frame.allocate();
        fill_grayscale_yuv(&gray, 16, 16, &mut frame);

        assert!(!frame.planes.is_empty());
        // Y plane should contain the same luma values
        assert_eq!(frame.planes[0].data.len(), 16 * 16);
        assert_eq!(frame.planes[0].data[0], 200);
        // U and V planes should be neutral 128
        assert_eq!(frame.planes[1].data[0], 128);
        assert_eq!(frame.planes[2].data[0], 128);
    }

    #[test]
    fn test_yuv420p_to_rgb24_empty_frame() {
        let frame = VideoFrame::new(PixelFormat::Yuv420p, 4, 4);
        // No allocate => empty planes
        let rgb = yuv420p_to_rgb24(&frame, 4, 4);
        assert_eq!(rgb.len(), 4 * 4 * 3);
        // Should return zeros since planes are empty
        assert!(rgb.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_denoise_via_internal_api() {
        let gray = gray_frame(64, 64, 128);
        let config = DenoiseConfig {
            mode: DenoiseMode::Fast,
            strength: 0.3,
            temporal_window: 5,
            preserve_edges: true,
            preserve_grain: false,
        };
        assert!(config.validate().is_ok());

        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        frame.allocate();
        fill_grayscale_yuv(&gray, 64, 64, &mut frame);

        let mut denoiser = Denoiser::new(config);
        let result = denoiser.process(&frame);
        assert!(result.is_ok());
        let denoised = result.expect("denoise should succeed");
        assert!(!denoised.planes.is_empty());
        assert_eq!(denoised.planes[0].data.len(), 64 * 64);
    }

    #[test]
    fn test_denoise_rgb_via_internal_api() {
        let data = rgb_frame(32, 32);
        let config = DenoiseConfig {
            mode: DenoiseMode::Balanced,
            strength: 0.5,
            temporal_window: 5,
            preserve_edges: true,
            preserve_grain: false,
        };

        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 32, 32);
        frame.allocate();
        rgb24_to_yuv420p(&data, 32, 32, &mut frame);

        let mut denoiser = Denoiser::new(config);
        let result = denoiser.process(&frame);
        assert!(result.is_ok());
        let denoised = result.expect("denoise should succeed");
        let rgb_back = yuv420p_to_rgb24(&denoised, 32, 32);
        assert_eq!(rgb_back.len(), 32 * 32 * 3);
    }

    #[test]
    fn test_audio_denoise_via_internal_api() {
        use oximedia_denoise::audio_denoise::AudioDenoiseFilter;

        let samples: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
        let mut filter = AudioDenoiseFilter::new(0.025, 2205);
        let mut output = Vec::with_capacity(samples.len());
        for chunk in samples.chunks(1024) {
            let denoised = filter.process(chunk);
            output.extend_from_slice(&denoised);
        }
        assert_eq!(output.len(), samples.len());
    }

    #[test]
    fn test_audio_denoise_empty() {
        use oximedia_denoise::audio_denoise::AudioDenoiseFilter;

        let mut filter = AudioDenoiseFilter::new(0.025, 2205);
        let result = filter.process(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_noise_estimate_via_internal_api() {
        use oximedia_denoise::noise_estimate::{NoiseEstimateMethod, NoiseEstimator};

        let gray = gray_frame(64, 64, 128);
        let mut estimator = NoiseEstimator::new(NoiseEstimateMethod::LocalVariance);
        let estimate = estimator.estimate_from_frame(&gray, 64, 64);
        assert!(estimate.sigma < 5.0);
    }

    #[test]
    fn test_denoise_config_presets() {
        // Test all preset configurations are valid
        for mode in &[
            DenoiseMode::Fast,
            DenoiseMode::Balanced,
            DenoiseMode::Quality,
            DenoiseMode::GrainAware,
        ] {
            let config = DenoiseConfig {
                mode: *mode,
                strength: 0.5,
                temporal_window: 5,
                preserve_edges: true,
                preserve_grain: *mode == DenoiseMode::GrainAware,
            };
            assert!(config.validate().is_ok(), "Failed for mode: {mode:?}");
        }
    }
}
