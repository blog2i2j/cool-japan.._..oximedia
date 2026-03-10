//! Black frame and silence detection.
//!
//! This module detects black or near-black video frames and silent audio segments.
//! Black frames are common in:
//! - Content start/end markers
//! - Commercial breaks
//! - Transmission errors
//! - Encoding artifacts
//!
//! # Detection Algorithm
//!
//! A frame is considered black if:
//! 1. Average luminance is below threshold
//! 2. Most pixels are below threshold
//! 3. Duration exceeds minimum length
//!
//! # Example
//!
//! ```
//! use oximedia_analysis::black::BlackFrameDetector;
//!
//! let mut detector = BlackFrameDetector::new(16, 10);
//!
//! // Process frames
//! # let black_frame = vec![0u8; 1920 * 1080];
//! detector.process_frame(&black_frame, 1920, 1080, 0)?;
//!
//! let segments = detector.finalize();
//! println!("Found {} black segments", segments.len());
//! ```

use crate::{AnalysisError, AnalysisResult};
use serde::{Deserialize, Serialize};

/// Black frame or silence segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlackSegment {
    /// Starting frame number
    pub start_frame: usize,
    /// Ending frame number (exclusive)
    pub end_frame: usize,
    /// Average luminance level (0-255)
    pub avg_luminance: f64,
    /// Percentage of pixels below threshold
    pub black_pixel_ratio: f64,
}

/// Black frame detector.
pub struct BlackFrameDetector {
    threshold: u8,
    min_duration: usize,
    segments: Vec<BlackSegment>,
    current_segment: Option<(usize, Vec<f64>, Vec<f64>)>,
}

impl BlackFrameDetector {
    /// Create a new black frame detector.
    ///
    /// # Parameters
    ///
    /// - `threshold`: Pixel value threshold (0-255). Pixels below this are considered black.
    /// - `min_duration`: Minimum number of consecutive black frames to report.
    #[must_use]
    pub fn new(threshold: u8, min_duration: usize) -> Self {
        Self {
            threshold,
            min_duration,
            segments: Vec::new(),
            current_segment: None,
        }
    }

    /// Process a video frame.
    pub fn process_frame(
        &mut self,
        y_plane: &[u8],
        width: usize,
        height: usize,
        frame_number: usize,
    ) -> AnalysisResult<()> {
        if y_plane.len() != width * height {
            return Err(AnalysisError::InvalidInput(
                "Y plane size mismatch".to_string(),
            ));
        }

        // Compute average luminance
        let total: usize = y_plane.iter().map(|&p| p as usize).sum();
        let avg_luminance = total as f64 / y_plane.len() as f64;

        // Count black pixels
        let black_pixels = y_plane.iter().filter(|&&p| p < self.threshold).count();
        let black_ratio = black_pixels as f64 / y_plane.len() as f64;

        // Determine if frame is black
        let is_black = avg_luminance < f64::from(self.threshold) || black_ratio > 0.98;

        if is_black {
            // Start or continue a black segment
            if let Some((_start, ref mut lums, ref mut ratios)) = self.current_segment {
                lums.push(avg_luminance);
                ratios.push(black_ratio);
            } else {
                self.current_segment = Some((frame_number, vec![avg_luminance], vec![black_ratio]));
            }
        } else {
            // End current black segment if it exists
            if let Some((start, lums, ratios)) = self.current_segment.take() {
                let duration = frame_number - start;
                if duration >= self.min_duration {
                    let avg_lum = lums.iter().sum::<f64>() / lums.len() as f64;
                    let avg_ratio = ratios.iter().sum::<f64>() / ratios.len() as f64;
                    self.segments.push(BlackSegment {
                        start_frame: start,
                        end_frame: frame_number,
                        avg_luminance: avg_lum,
                        black_pixel_ratio: avg_ratio,
                    });
                }
            }
        }

        Ok(())
    }

    /// Finalize and return detected black segments.
    pub fn finalize(mut self) -> Vec<BlackSegment> {
        // Close any open segment
        if let Some((start, lums, ratios)) = self.current_segment.take() {
            let duration = lums.len();
            if duration >= self.min_duration {
                let avg_lum = lums.iter().sum::<f64>() / lums.len() as f64;
                let avg_ratio = ratios.iter().sum::<f64>() / ratios.len() as f64;
                self.segments.push(BlackSegment {
                    start_frame: start,
                    end_frame: start + duration,
                    avg_luminance: avg_lum,
                    black_pixel_ratio: avg_ratio,
                });
            }
        }
        self.segments
    }
}

/// Silence detector for audio.
pub struct SilenceDetector {
    threshold_db: f64,
    min_duration_samples: usize,
    #[allow(dead_code)]
    sample_rate: u32,
    segments: Vec<SilenceSegment>,
    current_segment: Option<(usize, Vec<f64>)>,
    sample_count: usize,
}

/// Silence segment in audio.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SilenceSegment {
    /// Starting sample index
    pub start_sample: usize,
    /// Ending sample index (exclusive)
    pub end_sample: usize,
    /// Average RMS level (dB)
    pub avg_level_db: f64,
}

impl SilenceDetector {
    /// Create a new silence detector.
    ///
    /// # Parameters
    ///
    /// - `threshold_db`: RMS threshold in dB (e.g., -60.0)
    /// - `min_duration_samples`: Minimum silence duration in samples
    /// - `sample_rate`: Audio sample rate
    #[must_use]
    pub fn new(threshold_db: f64, min_duration_samples: usize, sample_rate: u32) -> Self {
        Self {
            threshold_db,
            min_duration_samples,
            sample_rate,
            segments: Vec::new(),
            current_segment: None,
            sample_count: 0,
        }
    }

    /// Process audio samples (mono or interleaved).
    pub fn process_samples(&mut self, samples: &[f32]) -> AnalysisResult<()> {
        // Compute RMS in blocks
        const BLOCK_SIZE: usize = 1024;

        for chunk in samples.chunks(BLOCK_SIZE) {
            let rms = compute_rms(chunk);
            let db = if rms > 0.0 {
                20.0 * rms.log10()
            } else {
                -100.0
            };

            let is_silent = db < self.threshold_db;

            if is_silent {
                if let Some((_start, ref mut levels)) = self.current_segment {
                    levels.push(db);
                } else {
                    self.current_segment = Some((self.sample_count, vec![db]));
                }
            } else if let Some((start, levels)) = self.current_segment.take() {
                let duration = self.sample_count - start;
                if duration >= self.min_duration_samples {
                    let avg_db = levels.iter().sum::<f64>() / levels.len() as f64;
                    self.segments.push(SilenceSegment {
                        start_sample: start,
                        end_sample: self.sample_count,
                        avg_level_db: avg_db,
                    });
                }
            }

            self.sample_count += chunk.len();
        }

        Ok(())
    }

    /// Finalize and return detected silence segments.
    pub fn finalize(mut self) -> Vec<SilenceSegment> {
        if let Some((start, levels)) = self.current_segment.take() {
            let duration = self.sample_count - start;
            if duration >= self.min_duration_samples {
                let avg_db = levels.iter().sum::<f64>() / levels.len() as f64;
                self.segments.push(SilenceSegment {
                    start_sample: start,
                    end_sample: self.sample_count,
                    avg_level_db: avg_db,
                });
            }
        }
        self.segments
    }
}

/// Compute RMS (Root Mean Square) of audio samples.
fn compute_rms(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f64 = samples.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
    (sum / samples.len() as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_black_frame_detection() {
        let mut detector = BlackFrameDetector::new(16, 2);

        // Process black frames
        let black_frame = vec![0u8; 64 * 64];
        for i in 0..5 {
            detector
                .process_frame(&black_frame, 64, 64, i)
                .expect("frame processing should succeed");
        }

        // Process non-black frames
        let normal_frame = vec![128u8; 64 * 64];
        for i in 5..10 {
            detector
                .process_frame(&normal_frame, 64, 64, i)
                .expect("frame processing should succeed");
        }

        let segments = detector.finalize();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].start_frame, 0);
        assert_eq!(segments[0].end_frame, 5);
    }

    #[test]
    fn test_black_frame_min_duration() {
        let mut detector = BlackFrameDetector::new(16, 10);

        // Process only 5 black frames (below minimum)
        let black_frame = vec![0u8; 64 * 64];
        for i in 0..5 {
            detector
                .process_frame(&black_frame, 64, 64, i)
                .expect("frame processing should succeed");
        }

        let segments = detector.finalize();
        assert_eq!(segments.len(), 0); // Should not report short segments
    }

    #[test]
    fn test_rms_calculation() {
        let samples = vec![0.5, -0.5, 0.5, -0.5];
        let rms = compute_rms(&samples);
        assert!((rms - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_silence_detection() {
        let mut detector = SilenceDetector::new(-60.0, 1000, 48000);

        // Silent audio
        let silent = vec![0.0f32; 5000];
        detector
            .process_samples(&silent)
            .expect("sample processing should succeed");

        // Loud audio
        let loud = vec![0.5f32; 5000];
        detector
            .process_samples(&loud)
            .expect("sample processing should succeed");

        let segments = detector.finalize();
        assert!(!segments.is_empty());
    }

    #[test]
    fn test_empty_audio() {
        let detector = SilenceDetector::new(-60.0, 1000, 48000);
        let segments = detector.finalize();
        assert!(segments.is_empty());
    }
}
