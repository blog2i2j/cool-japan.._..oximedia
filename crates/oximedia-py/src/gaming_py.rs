//! Python bindings for `oximedia-gaming` game streaming and highlight detection.
//!
//! Provides `PyCaptureConfig`, `PyHighlightDetector`, `PyHighlight`, `PyGameCapture`,
//! and standalone convenience functions for gaming workflows.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// PyCaptureConfig
// ---------------------------------------------------------------------------

/// Configuration for game capture.
#[pyclass]
#[derive(Clone)]
pub struct PyCaptureConfig {
    /// Capture width in pixels.
    #[pyo3(get)]
    pub width: u32,
    /// Capture height in pixels.
    #[pyo3(get)]
    pub height: u32,
    /// Target framerate.
    #[pyo3(get)]
    pub fps: u32,
    /// Video codec (av1, vp9, vp8).
    #[pyo3(get)]
    pub codec: String,
    /// Bitrate in kbps (None for auto).
    #[pyo3(get)]
    pub bitrate: Option<u32>,
    /// Container format (webm, mkv).
    #[pyo3(get)]
    pub format: String,
}

#[pymethods]
impl PyCaptureConfig {
    /// Create a new capture configuration.
    ///
    /// Args:
    ///     width: Capture width in pixels (default: 1920).
    ///     height: Capture height in pixels (default: 1080).
    ///     fps: Target framerate (default: 60).
    #[new]
    #[pyo3(signature = (width=None, height=None, fps=None))]
    fn new(width: Option<u32>, height: Option<u32>, fps: Option<u32>) -> PyResult<Self> {
        let w = width.unwrap_or(1920);
        let h = height.unwrap_or(1080);
        let f = fps.unwrap_or(60);

        if w == 0 || h == 0 {
            return Err(PyValueError::new_err("Width and height must be > 0"));
        }
        if f == 0 || f > 240 {
            return Err(PyValueError::new_err("FPS must be between 1 and 240"));
        }

        Ok(Self {
            width: w,
            height: h,
            fps: f,
            codec: "av1".to_string(),
            bitrate: None,
            format: "webm".to_string(),
        })
    }

    /// Set the video codec (av1, vp9, vp8).
    fn with_codec(&mut self, codec: &str) -> PyResult<()> {
        match codec {
            "av1" | "vp9" | "vp8" => {
                self.codec = codec.to_string();
                Ok(())
            }
            other => Err(PyValueError::new_err(format!(
                "Unsupported codec '{}'. Supported: av1, vp9, vp8",
                other
            ))),
        }
    }

    /// Set the bitrate in kbps.
    fn with_bitrate(&mut self, bitrate: u32) -> PyResult<()> {
        if bitrate < 100 {
            return Err(PyValueError::new_err("Bitrate must be at least 100 kbps"));
        }
        self.bitrate = Some(bitrate);
        Ok(())
    }

    /// Set the container format (webm, mkv).
    fn with_format(&mut self, format: &str) -> PyResult<()> {
        match format {
            "webm" | "mkv" => {
                self.format = format.to_string();
                Ok(())
            }
            other => Err(PyValueError::new_err(format!(
                "Unsupported format '{}'. Supported: webm, mkv",
                other
            ))),
        }
    }

    /// Create a config optimized for live streaming (low latency).
    #[staticmethod]
    fn for_streaming() -> PyResult<Self> {
        Ok(Self {
            width: 1920,
            height: 1080,
            fps: 60,
            codec: "vp9".to_string(),
            bitrate: Some(6000),
            format: "webm".to_string(),
        })
    }

    /// Create a config optimized for local recording (high quality).
    #[staticmethod]
    fn for_recording() -> PyResult<Self> {
        Ok(Self {
            width: 1920,
            height: 1080,
            fps: 60,
            codec: "av1".to_string(),
            bitrate: Some(12000),
            format: "mkv".to_string(),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "PyCaptureConfig({}x{} @ {}fps, codec='{}', bitrate={}, format='{}')",
            self.width,
            self.height,
            self.fps,
            self.codec,
            self.bitrate
                .map_or("auto".to_string(), |b| format!("{}kbps", b)),
            self.format,
        )
    }
}

// ---------------------------------------------------------------------------
// PyHighlight
// ---------------------------------------------------------------------------

/// A detected highlight moment.
#[pyclass]
#[derive(Clone)]
pub struct PyHighlight {
    /// Start time in seconds.
    #[pyo3(get)]
    pub start_time: f64,
    /// End time in seconds.
    #[pyo3(get)]
    pub end_time: f64,
    /// Highlight score (0.0 to 1.0).
    #[pyo3(get)]
    pub score: f64,
    /// Type of highlight (motion, audio, combined).
    #[pyo3(get)]
    pub highlight_type: String,
}

#[pymethods]
impl PyHighlight {
    fn __repr__(&self) -> String {
        format!(
            "PyHighlight(start={:.3}s, end={:.3}s, score={:.3}, type='{}')",
            self.start_time, self.end_time, self.score, self.highlight_type,
        )
    }

    /// Duration of the highlight in seconds.
    fn duration(&self) -> f64 {
        self.end_time - self.start_time
    }

    /// Convert to a dict with numeric fields.
    fn to_dict(&self) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        m.insert("start_time".to_string(), self.start_time);
        m.insert("end_time".to_string(), self.end_time);
        m.insert("score".to_string(), self.score);
        m.insert("duration".to_string(), self.duration());
        m
    }

    /// Get the highlight type string.
    fn get_type(&self) -> String {
        self.highlight_type.clone()
    }
}

// ---------------------------------------------------------------------------
// PyHighlightDetector
// ---------------------------------------------------------------------------

/// Detects gaming highlights from video frames or audio samples.
#[pyclass]
pub struct PyHighlightDetector {
    threshold: f64,
    min_duration: f64,
    context_before: f64,
    context_after: f64,
}

#[pymethods]
impl PyHighlightDetector {
    /// Create a new highlight detector.
    ///
    /// Args:
    ///     threshold: Minimum score to count as a highlight (0.0-1.0, default: 0.5).
    #[new]
    #[pyo3(signature = (threshold=None))]
    fn new(threshold: Option<f64>) -> PyResult<Self> {
        let t = threshold.unwrap_or(0.5);
        if !(0.0..=1.0).contains(&t) {
            return Err(PyValueError::new_err(format!(
                "Threshold must be between 0.0 and 1.0, got {t}"
            )));
        }
        Ok(Self {
            threshold: t,
            min_duration: 2.0,
            context_before: 3.0,
            context_after: 2.0,
        })
    }

    /// Set minimum highlight duration in seconds.
    fn with_min_duration(&mut self, seconds: f64) -> PyResult<()> {
        if seconds <= 0.0 {
            return Err(PyValueError::new_err("Minimum duration must be positive"));
        }
        self.min_duration = seconds;
        Ok(())
    }

    /// Set context padding around highlights.
    ///
    /// Args:
    ///     before: Seconds of context before the highlight.
    ///     after: Seconds of context after the highlight.
    fn with_context(&mut self, before: f64, after: f64) -> PyResult<()> {
        if before < 0.0 || after < 0.0 {
            return Err(PyValueError::new_err(
                "Context durations must be non-negative",
            ));
        }
        self.context_before = before;
        self.context_after = after;
        Ok(())
    }

    /// Detect highlights from video frames using motion analysis.
    ///
    /// Args:
    ///     frames: List of RGB frame byte arrays (each width * height * 3).
    ///     width: Frame width.
    ///     height: Frame height.
    ///     fps: Frames per second.
    ///
    /// Returns:
    ///     List of detected highlights.
    fn detect_highlights(
        &self,
        frames: Vec<Vec<u8>>,
        width: u32,
        height: u32,
        fps: f64,
    ) -> PyResult<Vec<PyHighlight>> {
        if frames.is_empty() {
            return Ok(Vec::new());
        }
        if fps <= 0.0 {
            return Err(PyValueError::new_err("FPS must be positive"));
        }
        if width == 0 || height == 0 {
            return Err(PyValueError::new_err("Width and height must be > 0"));
        }

        let expected_size = (width as usize) * (height as usize) * 3;
        for (i, frame) in frames.iter().enumerate() {
            if frame.len() < expected_size {
                return Err(PyValueError::new_err(format!(
                    "Frame {} too small: need {} bytes, got {}",
                    i,
                    expected_size,
                    frame.len()
                )));
            }
        }

        // Compute per-frame motion intensity scores
        let mut scores = Vec::with_capacity(frames.len());
        scores.push(0.0_f64); // First frame has no motion reference

        for i in 1..frames.len() {
            let intensity = compute_motion_intensity(
                &frames[i - 1],
                &frames[i],
                width as usize,
                height as usize,
            );
            scores.push(intensity);
        }

        // Find highlight regions from scores
        Ok(extract_highlights(
            &scores,
            fps,
            self.threshold,
            self.min_duration,
            self.context_before,
            self.context_after,
            "motion",
        ))
    }

    /// Detect highlights from audio samples using energy analysis.
    ///
    /// Args:
    ///     samples: Audio samples as float array (interleaved if stereo).
    ///     sample_rate: Sample rate in Hz.
    ///
    /// Returns:
    ///     List of detected audio highlights.
    fn detect_audio_highlights(
        &self,
        samples: Vec<f32>,
        sample_rate: u32,
    ) -> PyResult<Vec<PyHighlight>> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }
        if sample_rate == 0 {
            return Err(PyValueError::new_err("Sample rate must be > 0"));
        }

        // Compute RMS energy in sliding windows
        let window_samples = (sample_rate as usize) / 4; // 250ms windows
        let hop = window_samples / 2; // 50% overlap
        let fps_equiv = sample_rate as f64 / hop as f64;

        let mut scores = Vec::new();
        let mut offset = 0;

        while offset + window_samples <= samples.len() {
            let window = &samples[offset..offset + window_samples];
            let rms = compute_rms(window);
            scores.push(rms as f64);
            offset += hop;
        }

        // Normalize scores to 0.0-1.0
        let max_score = scores.iter().cloned().fold(0.0_f64, f64::max);
        if max_score > 0.0 {
            for s in &mut scores {
                *s /= max_score;
            }
        }

        Ok(extract_highlights(
            &scores,
            fps_equiv,
            self.threshold,
            self.min_duration,
            self.context_before,
            self.context_after,
            "audio",
        ))
    }

    fn __repr__(&self) -> String {
        format!(
            "PyHighlightDetector(threshold={:.2}, min_duration={:.1}s, context=[{:.1}s, {:.1}s])",
            self.threshold, self.min_duration, self.context_before, self.context_after,
        )
    }
}

// ---------------------------------------------------------------------------
// PyGameCapture
// ---------------------------------------------------------------------------

/// Simulates a game capture session that accumulates frames.
#[pyclass]
pub struct PyGameCapture {
    config: PyCaptureConfig,
    frame_count: u64,
    recording: bool,
    output_path: Option<String>,
}

#[pymethods]
impl PyGameCapture {
    /// Create a new game capture session.
    #[new]
    fn new(config: PyCaptureConfig) -> Self {
        Self {
            config,
            frame_count: 0,
            recording: false,
            output_path: None,
        }
    }

    /// Start recording to the given output path.
    fn start_recording(&mut self, output: &str) -> PyResult<()> {
        if self.recording {
            return Err(PyRuntimeError::new_err("Already recording"));
        }
        self.recording = true;
        self.output_path = Some(output.to_string());
        self.frame_count = 0;
        Ok(())
    }

    /// Stop recording.
    fn stop_recording(&mut self) -> PyResult<()> {
        if !self.recording {
            return Err(PyRuntimeError::new_err("Not recording"));
        }
        self.recording = false;
        Ok(())
    }

    /// Add a frame to the recording.
    ///
    /// Args:
    ///     data: Raw RGB bytes (width * height * 3).
    ///     width: Frame width.
    ///     height: Frame height.
    fn add_frame(&mut self, data: Vec<u8>, width: u32, height: u32) -> PyResult<()> {
        if !self.recording {
            return Err(PyRuntimeError::new_err(
                "Not recording. Call start_recording() first",
            ));
        }

        let expected = (width as usize) * (height as usize) * 3;
        if data.len() < expected {
            return Err(PyValueError::new_err(format!(
                "Frame data too small: need {} bytes for {}x{} RGB, got {}",
                expected,
                width,
                height,
                data.len()
            )));
        }

        self.frame_count += 1;
        Ok(())
    }

    /// Get the total number of frames captured.
    fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Check if currently recording.
    fn is_recording(&self) -> bool {
        self.recording
    }

    /// Get capture session statistics as a JSON string.
    fn stats(&self) -> String {
        let duration = if self.config.fps > 0 {
            self.frame_count as f64 / self.config.fps as f64
        } else {
            0.0
        };
        let output_str = self.output_path.as_deref().unwrap_or("");
        format!(
            "{{\"frame_count\":{},\"recording\":{},\"width\":{},\"height\":{},\
             \"fps\":{},\"codec\":\"{}\",\"output\":\"{}\",\"duration_seconds\":{:.4}}}",
            self.frame_count,
            self.recording,
            self.config.width,
            self.config.height,
            self.config.fps,
            self.config.codec,
            output_str,
            duration,
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "PyGameCapture(recording={}, frames={}, config={})",
            self.recording,
            self.frame_count,
            self.config.__repr__(),
        )
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Detect highlights from video frames using motion intensity.
///
/// Args:
///     frames: List of RGB byte arrays (each width * height * 3).
///     width: Frame width.
///     height: Frame height.
///     fps: Frames per second.
///     threshold: Minimum score to count as highlight (0.0-1.0, default: 0.5).
///
/// Returns:
///     List of detected highlights.
#[pyfunction]
#[pyo3(signature = (frames, width, height, fps, threshold=None))]
pub fn detect_highlights(
    frames: Vec<Vec<u8>>,
    width: u32,
    height: u32,
    fps: f64,
    threshold: Option<f64>,
) -> PyResult<Vec<PyHighlight>> {
    let detector = PyHighlightDetector::new(threshold)?;
    detector.detect_highlights(frames, width, height, fps)
}

/// Create a clip configuration summary (validates parameters).
///
/// Args:
///     input: Input file path.
///     output: Output file path.
///     start: Start time in seconds.
///     end: End time in seconds.
///     slow_motion: Optional slow-motion factor (0.0-1.0).
///
/// Returns:
///     JSON string with clip configuration.
#[pyfunction]
#[pyo3(signature = (input, output, start, end, slow_motion=None))]
pub fn create_clip(
    input: &str,
    output: &str,
    start: f64,
    end: f64,
    slow_motion: Option<f64>,
) -> PyResult<String> {
    if start >= end {
        return Err(PyValueError::new_err(format!(
            "Start time ({start:.3}s) must be before end time ({end:.3}s)"
        )));
    }
    if start < 0.0 {
        return Err(PyValueError::new_err(format!(
            "Start time must be non-negative, got {start:.3}s"
        )));
    }
    if let Some(sm) = slow_motion {
        if sm <= 0.0 || sm > 1.0 {
            return Err(PyValueError::new_err(format!(
                "Slow motion factor must be between 0.0 (exclusive) and 1.0, got {sm:.3}"
            )));
        }
    }

    let clip_duration = end - start;
    let effective_duration = slow_motion.map_or(clip_duration, |sm| clip_duration / sm);
    let sm_str = slow_motion.map_or("null".to_string(), |sm| format!("{sm:.4}"));

    Ok(format!(
        "{{\"input\":\"{input}\",\"output\":\"{output}\",\"start\":{start:.4},\
         \"end\":{end:.4},\"clip_duration\":{clip_duration:.4},\
         \"effective_duration\":{effective_duration:.4},\"slow_motion\":{sm_str},\
         \"status\":\"configured\"}}"
    ))
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute motion intensity between two RGB frames as a value in [0.0, 1.0].
fn compute_motion_intensity(prev: &[u8], curr: &[u8], width: usize, height: usize) -> f64 {
    let n = width * height * 3;
    let len = prev.len().min(curr.len()).min(n);
    if len == 0 {
        return 0.0;
    }

    let mut sum_diff: u64 = 0;
    for i in 0..len {
        let diff = (prev[i] as i32 - curr[i] as i32).unsigned_abs();
        sum_diff += diff as u64;
    }

    let avg_diff = sum_diff as f64 / len as f64;
    // Normalize: typical max SAD per pixel is ~128 for dramatic changes
    (avg_diff / 128.0).min(1.0)
}

/// Compute RMS energy of audio samples.
fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = samples.iter().map(|&s| (s as f64) * (s as f64)).sum();
    ((sum_sq / samples.len() as f64).sqrt()) as f32
}

/// Extract highlight regions from a score series.
fn extract_highlights(
    scores: &[f64],
    fps: f64,
    threshold: f64,
    min_duration: f64,
    context_before: f64,
    context_after: f64,
    highlight_type: &str,
) -> Vec<PyHighlight> {
    if scores.is_empty() || fps <= 0.0 {
        return Vec::new();
    }

    let total_duration = scores.len() as f64 / fps;
    let mut highlights = Vec::new();
    let mut region_start: Option<usize> = None;
    let mut region_max_score: f64 = 0.0;

    for (i, &score) in scores.iter().enumerate() {
        if score >= threshold {
            if region_start.is_none() {
                region_start = Some(i);
                region_max_score = score;
            } else if score > region_max_score {
                region_max_score = score;
            }
        } else if let Some(start) = region_start {
            let start_time = start as f64 / fps;
            let end_time = i as f64 / fps;
            let duration = end_time - start_time;

            if duration >= min_duration {
                let padded_start = (start_time - context_before).max(0.0);
                let padded_end = (end_time + context_after).min(total_duration);
                highlights.push(PyHighlight {
                    start_time: padded_start,
                    end_time: padded_end,
                    score: region_max_score,
                    highlight_type: highlight_type.to_string(),
                });
            }
            region_start = None;
            region_max_score = 0.0;
        }
    }

    // Handle region extending to end
    if let Some(start) = region_start {
        let start_time = start as f64 / fps;
        let end_time = scores.len() as f64 / fps;
        let duration = end_time - start_time;

        if duration >= min_duration {
            let padded_start = (start_time - context_before).max(0.0);
            let padded_end = (end_time + context_after).min(total_duration);
            highlights.push(PyHighlight {
                start_time: padded_start,
                end_time: padded_end,
                score: region_max_score,
                highlight_type: highlight_type.to_string(),
            });
        }
    }

    highlights
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register all gaming bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCaptureConfig>()?;
    m.add_class::<PyHighlight>()?;
    m.add_class::<PyHighlightDetector>()?;
    m.add_class::<PyGameCapture>()?;
    m.add_function(wrap_pyfunction!(detect_highlights, m)?)?;
    m.add_function(wrap_pyfunction!(create_clip, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capture_config_defaults() {
        let cfg = PyCaptureConfig::new(None, None, None);
        assert!(cfg.is_ok());
        let cfg = cfg.expect("config should be valid");
        assert_eq!(cfg.width, 1920);
        assert_eq!(cfg.height, 1080);
        assert_eq!(cfg.fps, 60);
        assert_eq!(cfg.codec, "av1");
    }

    #[test]
    fn test_capture_config_invalid_dimensions() {
        let result = PyCaptureConfig::new(Some(0), Some(1080), None);
        assert!(result.is_err());
    }

    #[test]
    fn test_highlight_detector_invalid_threshold() {
        let result = PyHighlightDetector::new(Some(1.5));
        assert!(result.is_err());
    }

    #[test]
    fn test_motion_intensity_identical_frames() {
        let frame = vec![128u8; 32 * 32 * 3];
        let intensity = compute_motion_intensity(&frame, &frame, 32, 32);
        assert!(intensity < 0.01);
    }

    #[test]
    fn test_compute_rms() {
        let samples = vec![0.5_f32; 100];
        let rms = compute_rms(&samples);
        assert!((rms - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_extract_highlights_empty() {
        let highlights = extract_highlights(&[], 30.0, 0.5, 2.0, 1.0, 1.0, "motion");
        assert!(highlights.is_empty());
    }
}
