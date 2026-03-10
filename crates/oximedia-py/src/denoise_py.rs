//! Python bindings for video/audio denoising from `oximedia-denoise`.
//!
//! Provides `PyDenoiseConfig`, `PyDenoiser`, plus standalone functions for
//! single-frame denoising, audio denoising, and noise estimation.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use oximedia_codec::{Plane, VideoFrame};
use oximedia_core::PixelFormat;
use oximedia_denoise::{DenoiseConfig, DenoiseMode, Denoiser};

// ---------------------------------------------------------------------------
// PyDenoiseConfig
// ---------------------------------------------------------------------------

/// Configuration for the denoiser.
#[pyclass]
#[derive(Clone)]
pub struct PyDenoiseConfig {
    /// Mode name: fast, balanced, quality, grain_aware.
    #[pyo3(get)]
    pub mode: String,

    /// Denoising strength (0.0 = none, 1.0 = maximum).
    #[pyo3(get)]
    pub strength: f64,

    /// Temporal window in frames (must be odd, 3-15).
    #[pyo3(get)]
    pub temporal_window: u32,

    /// Whether to preserve edges.
    #[pyo3(get)]
    pub preserve_edges: bool,

    /// Whether to preserve film grain.
    #[pyo3(get)]
    pub preserve_grain: bool,
}

#[pymethods]
impl PyDenoiseConfig {
    /// Create a new denoise configuration.
    ///
    /// Defaults: `mode="balanced"`, `strength=0.5`.
    #[new]
    #[pyo3(signature = (mode=None, strength=None))]
    fn new(mode: Option<&str>, strength: Option<f64>) -> PyResult<Self> {
        let mode_str = mode.unwrap_or("balanced");
        let _ = parse_mode(mode_str)?;
        Ok(Self {
            mode: mode_str.to_string(),
            strength: strength.unwrap_or(0.5),
            temporal_window: 5,
            preserve_edges: true,
            preserve_grain: false,
        })
    }

    /// Light denoising preset (fast mode, strength 0.3).
    #[classmethod]
    #[allow(unused_variables)]
    fn light(cls: &Bound<'_, pyo3::types::PyType>) -> Self {
        Self {
            mode: "fast".to_string(),
            strength: 0.3,
            temporal_window: 5,
            preserve_edges: true,
            preserve_grain: false,
        }
    }

    /// Medium denoising preset (balanced mode, strength 0.5).
    #[classmethod]
    #[allow(unused_variables)]
    fn medium(cls: &Bound<'_, pyo3::types::PyType>) -> Self {
        Self {
            mode: "balanced".to_string(),
            strength: 0.5,
            temporal_window: 5,
            preserve_edges: true,
            preserve_grain: false,
        }
    }

    /// Strong denoising preset (quality mode, strength 0.8).
    #[classmethod]
    #[allow(unused_variables)]
    fn strong(cls: &Bound<'_, pyo3::types::PyType>) -> Self {
        Self {
            mode: "quality".to_string(),
            strength: 0.8,
            temporal_window: 7,
            preserve_edges: true,
            preserve_grain: false,
        }
    }

    /// Grain-aware denoising preset that preserves film grain.
    #[classmethod]
    #[pyo3(signature = (strength=None))]
    #[allow(unused_variables)]
    fn grain_aware(cls: &Bound<'_, pyo3::types::PyType>, strength: Option<f64>) -> Self {
        Self {
            mode: "grain_aware".to_string(),
            strength: strength.unwrap_or(0.5),
            temporal_window: 5,
            preserve_edges: true,
            preserve_grain: true,
        }
    }

    /// Set the temporal processing window (must be odd, 3-15).
    fn with_temporal_window(&mut self, frames: u32) -> PyResult<()> {
        if frames < 3 || frames > 15 {
            return Err(PyValueError::new_err(
                "Temporal window must be between 3 and 15",
            ));
        }
        if frames % 2 == 0 {
            return Err(PyValueError::new_err("Temporal window must be odd"));
        }
        self.temporal_window = frames;
        Ok(())
    }

    /// Enable or disable edge preservation.
    fn with_edge_preservation(&mut self, enable: bool) {
        self.preserve_edges = enable;
    }

    /// Enable or disable grain preservation.
    fn with_grain_preservation(&mut self, enable: bool) {
        self.preserve_grain = enable;
    }

    fn __repr__(&self) -> String {
        format!(
            "PyDenoiseConfig(mode='{}', strength={:.2}, temporal_window={}, edges={}, grain={})",
            self.mode,
            self.strength,
            self.temporal_window,
            self.preserve_edges,
            self.preserve_grain
        )
    }
}

impl PyDenoiseConfig {
    /// Convert to internal `DenoiseConfig`.
    fn to_internal(&self) -> Result<DenoiseConfig, PyErr> {
        let mode = parse_mode(&self.mode)?;
        let config = DenoiseConfig {
            mode,
            strength: self.strength as f32,
            temporal_window: self.temporal_window as usize,
            preserve_edges: self.preserve_edges,
            preserve_grain: self.preserve_grain,
        };
        config
            .validate()
            .map_err(|e| PyValueError::new_err(format!("Invalid config: {e}")))?;
        Ok(config)
    }
}

// ---------------------------------------------------------------------------
// PyDenoiser
// ---------------------------------------------------------------------------

/// Stateful video denoiser that processes frames sequentially.
///
/// Wraps `oximedia_denoise::Denoiser` and can be fed individual video frames
/// as raw pixel data.
#[pyclass]
pub struct PyDenoiser {
    inner: Denoiser,
    config_snapshot: PyDenoiseConfig,
}

#[pymethods]
impl PyDenoiser {
    /// Create a new denoiser from a configuration.
    #[new]
    fn new(config: &PyDenoiseConfig) -> PyResult<Self> {
        let internal = config.to_internal()?;
        Ok(Self {
            inner: Denoiser::new(internal),
            config_snapshot: config.clone(),
        })
    }

    /// Process a single video frame.
    ///
    /// `frame_data` is raw pixel data (RGB24, length = `width * height * channels`).
    /// `channels` must be 3 (RGB) or 1 (grayscale).
    ///
    /// Returns the denoised pixel data with the same dimensions and layout.
    fn process_frame(
        &mut self,
        frame_data: Vec<u8>,
        width: u32,
        height: u32,
        channels: u32,
    ) -> PyResult<Vec<u8>> {
        // Build a VideoFrame from raw data
        let pixel_format = match channels {
            1 => PixelFormat::Gray8,
            3 => PixelFormat::Rgb24,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported channel count {channels}. Use 1 (gray) or 3 (RGB)"
                )))
            }
        };

        let expected = (width * height * channels) as usize;
        if frame_data.len() < expected {
            return Err(PyValueError::new_err(format!(
                "Frame data too small: need {expected} bytes, got {}",
                frame_data.len()
            )));
        }

        // Create an internal VideoFrame in YUV420P (Denoiser operates on YUV)
        let mut video_frame = VideoFrame::new(PixelFormat::Yuv420p, width, height);
        video_frame.allocate();

        // Convert RGB24 to YUV420P for processing
        if pixel_format == PixelFormat::Rgb24 {
            rgb24_to_yuv420p_frame(&frame_data, width, height, &mut video_frame);
        } else {
            // Grayscale: fill Y plane, set chroma to neutral
            let y_size = (width * height) as usize;
            if video_frame.planes.len() > 2 {
                let chroma_w = ((width + 1) / 2) as usize;
                let chroma_h = ((height + 1) / 2) as usize;
                video_frame.planes[0] = Plane::with_dimensions(
                    frame_data[..y_size].to_vec(),
                    width as usize,
                    width,
                    height,
                );
                video_frame.planes[1] = Plane::with_dimensions(
                    vec![128u8; chroma_w * chroma_h],
                    chroma_w,
                    chroma_w as u32,
                    chroma_h as u32,
                );
                video_frame.planes[2] = Plane::with_dimensions(
                    vec![128u8; chroma_w * chroma_h],
                    chroma_w,
                    chroma_w as u32,
                    chroma_h as u32,
                );
            }
        }

        // Process
        let denoised = self
            .inner
            .process(&video_frame)
            .map_err(|e| PyRuntimeError::new_err(format!("Denoise failed: {e}")))?;

        // Convert back to output format
        if pixel_format == PixelFormat::Rgb24 {
            Ok(yuv420p_frame_to_rgb24(&denoised, width, height))
        } else {
            // Return Y plane
            if denoised.planes.is_empty() {
                return Err(PyRuntimeError::new_err("Denoised frame has no plane data"));
            }
            Ok(denoised.planes[0].data.clone())
        }
    }

    /// Get the estimated noise level (if available after processing at least one frame).
    fn noise_level(&self) -> Option<f32> {
        self.inner.noise_level()
    }

    /// Reset the denoiser state (clears frame buffer and noise estimates).
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Get the current configuration.
    fn config(&self) -> PyDenoiseConfig {
        self.config_snapshot.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "PyDenoiser(mode='{}', strength={:.2})",
            self.config_snapshot.mode, self.config_snapshot.strength
        )
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Denoise a single image frame (one-shot, no temporal context).
///
/// `data` is raw pixel data, `channels` is 1 or 3.
/// `mode` defaults to `"balanced"`, `strength` defaults to `0.5`.
#[pyfunction]
#[pyo3(signature = (data, width, height, channels, mode=None, strength=None))]
pub fn denoise_image(
    data: Vec<u8>,
    width: u32,
    height: u32,
    channels: u32,
    mode: Option<&str>,
    strength: Option<f64>,
) -> PyResult<Vec<u8>> {
    let mode_enum = parse_mode(mode.unwrap_or("balanced"))?;
    let strength_val = strength.unwrap_or(0.5) as f32;

    let config = DenoiseConfig {
        mode: mode_enum,
        strength: strength_val.clamp(0.0, 1.0),
        temporal_window: 5,
        preserve_edges: true,
        preserve_grain: mode_enum == DenoiseMode::GrainAware,
    };
    config
        .validate()
        .map_err(|e| PyValueError::new_err(format!("Invalid config: {e}")))?;

    let py_config = PyDenoiseConfig {
        mode: mode.unwrap_or("balanced").to_string(),
        strength: strength.unwrap_or(0.5),
        temporal_window: 5,
        preserve_edges: true,
        preserve_grain: mode_enum == DenoiseMode::GrainAware,
    };

    let mut denoiser = PyDenoiser::new(&py_config)?;
    denoiser.process_frame(data, width, height, channels)
}

/// Denoise audio samples using spectral subtraction and noise gating.
///
/// `samples` is a flat `f32` buffer of PCM audio.
/// `strength` controls aggressiveness (0.0 to 1.0, default 0.5).
#[pyfunction]
#[pyo3(signature = (samples, sample_rate, strength=None))]
pub fn denoise_audio_samples(
    samples: Vec<f32>,
    sample_rate: u32,
    strength: Option<f64>,
) -> PyResult<Vec<f32>> {
    use oximedia_denoise::audio_denoise::AudioDenoiseFilter;

    if samples.is_empty() {
        return Ok(Vec::new());
    }

    let strength_val = strength.unwrap_or(0.5) as f32;

    // Gate threshold: lower strength = higher threshold (less aggressive)
    let gate_threshold = (1.0 - strength_val) * 0.05;
    // Hold time: ~50ms worth of samples
    let hold_samples = (f64::from(sample_rate) * 0.05) as usize;

    let mut filter = AudioDenoiseFilter::new(gate_threshold, hold_samples);

    // Process in blocks (e.g., 1024 samples at a time)
    let block_size = 1024;
    let mut output = Vec::with_capacity(samples.len());

    for chunk in samples.chunks(block_size) {
        let denoised_block = filter.process(chunk);
        output.extend_from_slice(&denoised_block);
    }

    Ok(output)
}

/// Estimate the noise level in an image frame.
///
/// `data` is raw pixel data (grayscale or RGB24, will use luma if RGB).
/// Returns estimated noise standard deviation (0-255 scale).
#[pyfunction]
pub fn estimate_noise(data: Vec<u8>, width: u32, height: u32) -> PyResult<f64> {
    use oximedia_denoise::noise_estimate::{NoiseEstimateMethod, NoiseEstimator};

    let expected_rgb = (width * height * 3) as usize;
    let expected_gray = (width * height) as usize;

    let luma_data = if data.len() >= expected_rgb {
        // RGB data: extract luma
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
        // Grayscale
        data[..expected_gray].to_vec()
    } else {
        return Err(PyValueError::new_err(format!(
            "Data too small: need at least {expected_gray} bytes for {width}x{height}, got {}",
            data.len()
        )));
    };

    let mut estimator = NoiseEstimator::new(NoiseEstimateMethod::LocalVariance);
    let estimate = estimator.estimate_from_frame(&luma_data, width as usize, height as usize);

    Ok(f64::from(estimate.sigma))
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register all denoise bindings on the given Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDenoiseConfig>()?;
    m.add_class::<PyDenoiser>()?;
    m.add_function(wrap_pyfunction!(denoise_image, m)?)?;
    m.add_function(wrap_pyfunction!(denoise_audio_samples, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_noise, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn parse_mode(mode: &str) -> Result<DenoiseMode, PyErr> {
    match mode.to_lowercase().replace('-', "_").as_str() {
        "fast" => Ok(DenoiseMode::Fast),
        "balanced" => Ok(DenoiseMode::Balanced),
        "quality" => Ok(DenoiseMode::Quality),
        "grain_aware" | "grain-aware" => Ok(DenoiseMode::GrainAware),
        other => Err(PyValueError::new_err(format!(
            "Unknown denoise mode '{other}'. Use: fast, balanced, quality, grain_aware"
        ))),
    }
}

/// Convert RGB24 raw bytes into a YUV420P `VideoFrame`.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn rgb24_to_yuv420p_frame(rgb: &[u8], width: u32, height: u32, frame: &mut VideoFrame) {
    let w = width as usize;
    let h = height as usize;
    let chroma_w = (w + 1) / 2;
    let chroma_h = (h + 1) / 2;

    let mut y_plane = vec![0u8; w * h];
    let mut u_plane = vec![0u8; chroma_w * chroma_h];
    let mut v_plane = vec![0u8; chroma_w * chroma_h];

    // Fill Y plane
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

    // Fill U/V planes (subsampled 2x2)
    for row in 0..chroma_h {
        for col in 0..chroma_w {
            let src_row = (row * 2).min(h - 1);
            let src_col = (col * 2).min(w - 1);
            let idx = (src_row * w + src_col) * 3;
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
            Plane::with_dimensions(u_plane, chroma_w, (chroma_w) as u32, (chroma_h) as u32);
        frame.planes[2] =
            Plane::with_dimensions(v_plane, chroma_w, (chroma_w) as u32, (chroma_h) as u32);
    }
}

/// Convert a YUV420P `VideoFrame` back to RGB24.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn yuv420p_frame_to_rgb24(frame: &VideoFrame, width: u32, height: u32) -> Vec<u8> {
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

            let y_val = y_data.get(y_idx).copied().unwrap_or(128);
            let cb_val = u_data.get(c_idx).copied().unwrap_or(128);
            let cr_val = v_data.get(c_idx).copied().unwrap_or(128);

            let y_f = f32::from(y_val);
            let cb_f = f32::from(cb_val) - 128.0;
            let cr_f = f32::from(cr_val) - 128.0;

            let r = (y_f + 1.5748 * cr_f).clamp(0.0, 255.0);
            let g = (y_f - 0.1873 * cb_f - 0.4681 * cr_f).clamp(0.0, 255.0);
            let b = (y_f + 1.8556 * cb_f).clamp(0.0, 255.0);

            let out_idx = (row * w + col) * 3;
            rgb[out_idx] = r as u8;
            rgb[out_idx + 1] = g as u8;
            rgb[out_idx + 2] = b as u8;
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
        let mut data = vec![0u8; (width * height * 3) as usize];
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                data[idx] = 128;
                data[idx + 1] = 128;
                data[idx + 2] = 128;
            }
        }
        data
    }

    #[test]
    fn test_config_default() {
        let cfg = PyDenoiseConfig::new(None, None);
        assert!(cfg.is_ok());
        let cfg = cfg.expect("should succeed");
        assert_eq!(cfg.mode, "balanced");
        assert!((cfg.strength - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_config_custom() {
        let cfg = PyDenoiseConfig::new(Some("quality"), Some(0.8));
        assert!(cfg.is_ok());
        let cfg = cfg.expect("should succeed");
        assert_eq!(cfg.mode, "quality");
    }

    #[test]
    fn test_config_bad_mode() {
        let cfg = PyDenoiseConfig::new(Some("invalid_mode"), None);
        assert!(cfg.is_err());
    }

    #[test]
    fn test_config_temporal_window() {
        let mut cfg = PyDenoiseConfig::new(None, None).expect("should succeed");
        assert!(cfg.with_temporal_window(7).is_ok());
        assert_eq!(cfg.temporal_window, 7);
        assert!(cfg.with_temporal_window(2).is_err());
        assert!(cfg.with_temporal_window(4).is_err());
    }

    #[test]
    fn test_config_to_internal() {
        let cfg = PyDenoiseConfig::new(Some("fast"), Some(0.3)).expect("should succeed");
        let internal = cfg.to_internal();
        assert!(internal.is_ok());
        let internal = internal.expect("should succeed");
        assert_eq!(internal.mode, DenoiseMode::Fast);
    }

    #[test]
    fn test_parse_mode() {
        assert_eq!(parse_mode("fast").expect("ok"), DenoiseMode::Fast);
        assert_eq!(parse_mode("balanced").expect("ok"), DenoiseMode::Balanced);
        assert_eq!(parse_mode("quality").expect("ok"), DenoiseMode::Quality);
        assert_eq!(
            parse_mode("grain_aware").expect("ok"),
            DenoiseMode::GrainAware
        );
        assert!(parse_mode("bad").is_err());
    }

    #[test]
    fn test_estimate_noise_gray() {
        let data = gray_frame(64, 64, 128);
        let result = estimate_noise(data, 64, 64);
        assert!(result.is_ok());
        let sigma = result.expect("should succeed");
        // Flat frame should have low noise
        assert!(sigma < 5.0);
    }

    #[test]
    fn test_estimate_noise_rgb() {
        let data = rgb_frame(64, 64);
        let result = estimate_noise(data, 64, 64);
        assert!(result.is_ok());
    }

    #[test]
    fn test_estimate_noise_bad_size() {
        let result = estimate_noise(vec![0u8; 10], 100, 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_denoise_audio_empty() {
        let result = denoise_audio_samples(vec![], 44100, None);
        assert!(result.is_ok());
        assert!(result.expect("should succeed").is_empty());
    }

    #[test]
    fn test_denoise_audio_basic() {
        let samples: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
        let result = denoise_audio_samples(samples.clone(), 44100, Some(0.3));
        assert!(result.is_ok());
        let output = result.expect("should succeed");
        assert_eq!(output.len(), samples.len());
    }

    #[test]
    fn test_rgb_to_yuv_roundtrip() {
        let w = 16u32;
        let h = 16u32;
        let rgb = rgb_frame(w, h);
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, w, h);
        frame.allocate();
        rgb24_to_yuv420p_frame(&rgb, w, h, &mut frame);

        let rgb_back = yuv420p_frame_to_rgb24(&frame, w, h);
        assert_eq!(rgb_back.len(), (w * h * 3) as usize);

        // Check mid-gray survives roundtrip within tolerance
        for i in (0..rgb_back.len()).step_by(3) {
            let r_diff = (rgb_back[i] as i16 - 128).abs();
            let g_diff = (rgb_back[i + 1] as i16 - 128).abs();
            let b_diff = (rgb_back[i + 2] as i16 - 128).abs();
            assert!(r_diff < 5, "R diff too large: {r_diff}");
            assert!(g_diff < 5, "G diff too large: {g_diff}");
            assert!(b_diff < 5, "B diff too large: {b_diff}");
        }
    }

    #[test]
    fn test_config_repr() {
        let cfg = PyDenoiseConfig::new(Some("balanced"), Some(0.5)).expect("should succeed");
        let repr = cfg.__repr__();
        assert!(repr.contains("balanced"));
        assert!(repr.contains("0.50"));
    }
}
