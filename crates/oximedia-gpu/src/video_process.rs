//! GPU-accelerated video frame processing.
//!
//! This module provides CPU-fallback simulations of GPU compute operations
//! for video frame processing. In production, these would be replaced by
//! actual WGPU compute shader kernels.

use crate::{GpuError, Result};
use std::time::{SystemTime, UNIX_EPOCH};

/// Configuration for GPU-based frame operations.
#[derive(Debug, Clone)]
pub struct FrameProcessConfig {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Number of channels: 1=Y (grayscale), 3=RGB, 4=RGBA.
    pub channels: u8,
}

/// Result of a GPU frame processing operation.
#[derive(Debug, Clone)]
pub struct FrameProcessResult {
    /// Processed pixel data.
    pub data: Vec<u8>,
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// Processing time in microseconds.
    pub processing_time_us: u64,
}

/// GPU-accelerated video frame processor.
///
/// Provides CPU-fallback implementations of common video frame operations
/// that would execute on the GPU in production environments.
pub struct VideoFrameProcessor {
    config: FrameProcessConfig,
}

impl VideoFrameProcessor {
    /// Create a new `VideoFrameProcessor` with the given configuration.
    #[must_use]
    pub fn new(config: FrameProcessConfig) -> Self {
        Self { config }
    }

    /// Get the current timestamp in microseconds (for timing).
    fn timestamp_us() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_micros()
            .into()
    }

    /// Validate that a frame buffer has the expected size.
    fn validate_frame(&self, frame: &[u8]) -> Result<()> {
        let expected = self.config.width as usize
            * self.config.height as usize
            * self.config.channels as usize;
        if frame.len() != expected {
            return Err(GpuError::InvalidBufferSize {
                expected,
                actual: frame.len(),
            });
        }
        Ok(())
    }

    /// Simulate GPU-accelerated frame histogram computation.
    ///
    /// For each channel, counts pixel value occurrences (0-255).
    /// Returns a `Vec` of `256 * channels` counts (interleaved per channel).
    ///
    /// # Errors
    ///
    /// Returns an error if the frame buffer size does not match the configured dimensions.
    pub fn compute_histogram(&self, frame: &[u8]) -> Result<Vec<u32>> {
        self.validate_frame(frame)?;

        let channels = self.config.channels as usize;
        let mut histogram = vec![0u32; 256 * channels];

        for (i, &pixel) in frame.iter().enumerate() {
            let ch = i % channels;
            histogram[ch * 256 + pixel as usize] += 1;
        }

        Ok(histogram)
    }

    /// Simulate GPU-accelerated frame brightness adjustment.
    ///
    /// Adds `delta` to each pixel value, clamping the result to `[0, 255]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the frame buffer size does not match the configured dimensions.
    pub fn adjust_brightness(&self, frame: &[u8], delta: i16) -> Result<Vec<u8>> {
        self.validate_frame(frame)?;

        let result = frame
            .iter()
            .map(|&p| (i16::from(p) + delta).clamp(0, 255) as u8)
            .collect();

        Ok(result)
    }

    /// Simulate GPU-accelerated contrast adjustment.
    ///
    /// For each pixel: `clamp((pixel - 128) * factor + 128, 0, 255)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the frame buffer size does not match the configured dimensions.
    pub fn adjust_contrast(&self, frame: &[u8], factor: f32) -> Result<Vec<u8>> {
        self.validate_frame(frame)?;

        let result = frame
            .iter()
            .map(|&p| {
                let adjusted = (f32::from(p) - 128.0) * factor + 128.0;
                adjusted.clamp(0.0, 255.0) as u8
            })
            .collect();

        Ok(result)
    }

    /// Simulate GPU-accelerated saturation adjustment for RGB frames (3 channels).
    ///
    /// Converts each RGB pixel to HSL, multiplies the S component by `factor`,
    /// then converts back to RGB. For non-RGB frames this is a no-op copy.
    ///
    /// # Errors
    ///
    /// Returns an error if the frame buffer size does not match the configured dimensions.
    pub fn adjust_saturation(&self, frame: &[u8], factor: f32) -> Result<Vec<u8>> {
        self.validate_frame(frame)?;

        if self.config.channels != 3 {
            // For non-RGB frames, return as-is (saturation is RGB concept)
            return Ok(frame.to_vec());
        }

        let mut result = Vec::with_capacity(frame.len());
        for chunk in frame.chunks(3) {
            let (r, g, b) = (
                f32::from(chunk[0]) / 255.0,
                f32::from(chunk[1]) / 255.0,
                f32::from(chunk[2]) / 255.0,
            );

            let (h, s, l) = rgb_to_hsl(r, g, b);
            let new_s = (s * factor).clamp(0.0, 1.0);
            let (nr, ng, nb) = hsl_to_rgb(h, new_s, l);

            result.push((nr * 255.0).clamp(0.0, 255.0) as u8);
            result.push((ng * 255.0).clamp(0.0, 255.0) as u8);
            result.push((nb * 255.0).clamp(0.0, 255.0) as u8);
        }

        Ok(result)
    }

    /// Compute frame difference (absolute difference per pixel).
    ///
    /// # Errors
    ///
    /// Returns an error if either frame buffer size does not match the configured dimensions.
    pub fn frame_difference(&self, frame_a: &[u8], frame_b: &[u8]) -> Result<Vec<u8>> {
        self.validate_frame(frame_a)?;
        self.validate_frame(frame_b)?;

        let result = frame_a
            .iter()
            .zip(frame_b.iter())
            .map(|(&a, &b)| a.abs_diff(b))
            .collect();

        Ok(result)
    }

    /// Compute mean absolute error between two frames.
    ///
    /// # Errors
    ///
    /// Returns an error if either frame buffer size does not match the configured dimensions.
    pub fn mean_absolute_error(&self, frame_a: &[u8], frame_b: &[u8]) -> Result<f64> {
        self.validate_frame(frame_a)?;
        self.validate_frame(frame_b)?;

        if frame_a.is_empty() {
            return Ok(0.0);
        }

        let sum: u64 = frame_a
            .iter()
            .zip(frame_b.iter())
            .map(|(&a, &b)| u64::from(a.abs_diff(b)))
            .sum();

        Ok(sum as f64 / frame_a.len() as f64)
    }

    /// Get the configuration.
    #[must_use]
    pub fn config(&self) -> &FrameProcessConfig {
        &self.config
    }

    /// Process a frame and return a `FrameProcessResult` with timing information.
    ///
    /// This is a convenience wrapper that applies brightness adjustment and
    /// records the simulated GPU processing time.
    ///
    /// # Errors
    ///
    /// Returns an error if the frame buffer size does not match the configured dimensions.
    pub fn process_frame(&self, frame: &[u8], brightness_delta: i16) -> Result<FrameProcessResult> {
        let start = Self::timestamp_us();
        let data = self.adjust_brightness(frame, brightness_delta)?;
        let end = Self::timestamp_us();

        Ok(FrameProcessResult {
            data,
            width: self.config.width,
            height: self.config.height,
            processing_time_us: end.saturating_sub(start),
        })
    }
}

// ---------------------------------------------------------------------------
// HSL / RGB conversion helpers
// ---------------------------------------------------------------------------

/// Convert RGB (0.0–1.0 each) to HSL.
fn rgb_to_hsl(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;
    let l = (max + min) / 2.0;

    if delta < f32::EPSILON {
        return (0.0, 0.0, l);
    }

    let s = if l < 0.5 {
        delta / (max + min)
    } else {
        delta / (2.0 - max - min)
    };

    let h = if (max - r).abs() < f32::EPSILON {
        ((g - b) / delta).rem_euclid(6.0) / 6.0
    } else if (max - g).abs() < f32::EPSILON {
        ((b - r) / delta + 2.0) / 6.0
    } else {
        ((r - g) / delta + 4.0) / 6.0
    };

    (h, s, l)
}

/// Helper for HSL-to-RGB conversion.
fn hsl_hue_to_rgb(p: f32, q: f32, mut t: f32) -> f32 {
    if t < 0.0 {
        t += 1.0;
    }
    if t > 1.0 {
        t -= 1.0;
    }
    if t < 1.0 / 6.0 {
        return p + (q - p) * 6.0 * t;
    }
    if t < 1.0 / 2.0 {
        return q;
    }
    if t < 2.0 / 3.0 {
        return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
    }
    p
}

/// Convert HSL to RGB (0.0–1.0 each).
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
    if s < f32::EPSILON {
        return (l, l, l);
    }

    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;

    let r = hsl_hue_to_rgb(p, q, h + 1.0 / 3.0);
    let g = hsl_hue_to_rgb(p, q, h);
    let b = hsl_hue_to_rgb(p, q, h - 1.0 / 3.0);

    (r, g, b)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_processor(w: u32, h: u32, ch: u8) -> VideoFrameProcessor {
        VideoFrameProcessor::new(FrameProcessConfig {
            width: w,
            height: h,
            channels: ch,
        })
    }

    #[test]
    fn test_histogram_uniform_frame() {
        // 4x4 single-channel frame, all pixels = 128
        let proc = make_processor(4, 4, 1);
        let frame = vec![128u8; 16];
        let hist = proc
            .compute_histogram(&frame)
            .expect("histogram computation should succeed");

        assert_eq!(hist.len(), 256);
        assert_eq!(hist[128], 16, "All 16 pixels should be at bin 128");
        for i in 0..256 {
            if i != 128 {
                assert_eq!(hist[i], 0);
            }
        }
    }

    #[test]
    fn test_histogram_rgb_frame() {
        // 2x2 RGB frame: all red=255, green=0, blue=128
        let proc = make_processor(2, 2, 3);
        let frame: Vec<u8> = (0..4).flat_map(|_| vec![255u8, 0u8, 128u8]).collect();
        let hist = proc
            .compute_histogram(&frame)
            .expect("histogram computation should succeed");

        assert_eq!(hist.len(), 768); // 3 * 256
                                     // Channel 0 (red): all 4 pixels at 255
        assert_eq!(hist[0 * 256 + 255], 4);
        // Channel 1 (green): all 4 pixels at 0
        assert_eq!(hist[1 * 256 + 0], 4);
        // Channel 2 (blue): all 4 pixels at 128
        assert_eq!(hist[2 * 256 + 128], 4);
    }

    #[test]
    fn test_adjust_brightness_clamp_up() {
        let proc = make_processor(2, 2, 1);
        let frame = vec![200u8, 100u8, 50u8, 10u8];
        let result = proc
            .adjust_brightness(&frame, 100)
            .expect("brightness adjustment should succeed");
        assert_eq!(result, vec![255, 200, 150, 110]);
    }

    #[test]
    fn test_adjust_brightness_clamp_down() {
        let proc = make_processor(2, 2, 1);
        let frame = vec![200u8, 100u8, 50u8, 10u8];
        let result = proc
            .adjust_brightness(&frame, -100)
            .expect("brightness adjustment should succeed");
        assert_eq!(result, vec![100, 0, 0, 0]);
    }

    #[test]
    fn test_adjust_contrast() {
        let proc = make_processor(1, 1, 1);
        // pixel=128, factor=1.0 → should stay at 128
        let frame = vec![128u8];
        let result = proc
            .adjust_contrast(&frame, 1.0)
            .expect("contrast adjustment should succeed");
        assert_eq!(result[0], 128);
    }

    #[test]
    fn test_adjust_contrast_increase() {
        let proc = make_processor(1, 1, 1);
        // pixel=200, factor=2.0 → (200-128)*2+128 = 272 → clamped to 255
        let frame = vec![200u8];
        let result = proc
            .adjust_contrast(&frame, 2.0)
            .expect("contrast adjustment should succeed");
        assert_eq!(result[0], 255);
    }

    #[test]
    fn test_adjust_saturation_no_change_at_one() {
        let proc = make_processor(1, 1, 3);
        let frame = vec![255u8, 0u8, 0u8]; // pure red
        let result = proc
            .adjust_saturation(&frame, 1.0)
            .expect("saturation adjustment should succeed");
        // With factor=1.0, saturation should be unchanged, red should stay red
        assert_eq!(result[0], 255);
        assert_eq!(result[1], 0);
        assert_eq!(result[2], 0);
    }

    #[test]
    fn test_adjust_saturation_zero_desaturates() {
        let proc = make_processor(1, 1, 3);
        let frame = vec![255u8, 0u8, 0u8]; // pure red
        let result = proc
            .adjust_saturation(&frame, 0.0)
            .expect("saturation adjustment should succeed");
        // With factor=0.0, becomes grayscale: all channels equal
        assert_eq!(result[0], result[1]);
        assert_eq!(result[1], result[2]);
    }

    #[test]
    fn test_frame_difference() {
        let proc = make_processor(2, 2, 1);
        let a = vec![100u8, 200u8, 50u8, 0u8];
        let b = vec![80u8, 210u8, 50u8, 255u8];
        let diff = proc
            .frame_difference(&a, &b)
            .expect("frame difference should succeed");
        assert_eq!(diff, vec![20, 10, 0, 255]);
    }

    #[test]
    fn test_mean_absolute_error() {
        let proc = make_processor(2, 2, 1);
        let a = vec![100u8, 100u8, 100u8, 100u8];
        let b = vec![110u8, 90u8, 100u8, 120u8];
        // diffs: 10, 10, 0, 20 → mean = 10.0
        let mae = proc
            .mean_absolute_error(&a, &b)
            .expect("MAE computation should succeed");
        assert!((mae - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_invalid_frame_size() {
        let proc = make_processor(4, 4, 1);
        let frame = vec![0u8; 10]; // wrong size
        assert!(proc.compute_histogram(&frame).is_err());
        assert!(proc.adjust_brightness(&frame, 0).is_err());
        assert!(proc.adjust_contrast(&frame, 1.0).is_err());
    }

    #[test]
    fn test_config_accessor() {
        let config = FrameProcessConfig {
            width: 1920,
            height: 1080,
            channels: 4,
        };
        let proc = VideoFrameProcessor::new(config.clone());
        assert_eq!(proc.config().width, 1920);
        assert_eq!(proc.config().height, 1080);
        assert_eq!(proc.config().channels, 4);
    }
}
