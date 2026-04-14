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
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut detector = BlackFrameDetector::new(16, 10);
//!
//! // Process frames
//! # let black_frame = vec![0u8; 1920 * 1080];
//! detector.process_frame(&black_frame, 1920, 1080, 0)?;
//!
//! let segments = detector.finalize();
//! println!("Found {} black segments", segments.len());
//! # Ok(())
//! # }
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

// ---------------------------------------------------------------------------
// Letterbox / Pillarbox Detection
// ---------------------------------------------------------------------------

/// Type of black bar pattern detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlackBarPattern {
    /// Black bars at top and bottom (widescreen content in 4:3 container).
    Letterbox,
    /// Black bars on left and right (4:3 content in 16:9 container).
    Pillarbox,
    /// Black bars on all four sides (window-boxed).
    Windowbox,
    /// No black bar pattern detected.
    None,
}

impl std::fmt::Display for BlackBarPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Letterbox => write!(f, "Letterbox"),
            Self::Pillarbox => write!(f, "Pillarbox"),
            Self::Windowbox => write!(f, "Windowbox"),
            Self::None => write!(f, "None"),
        }
    }
}

/// Result of letterbox/pillarbox detection for a single frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlackBarDetection {
    /// Frame number.
    pub frame_number: usize,
    /// Detected pattern type.
    pub pattern: BlackBarPattern,
    /// Number of black rows at the top.
    pub top_rows: usize,
    /// Number of black rows at the bottom.
    pub bottom_rows: usize,
    /// Number of black columns on the left.
    pub left_cols: usize,
    /// Number of black columns on the right.
    pub right_cols: usize,
    /// Active picture area as fraction of total frame (0.0-1.0).
    pub active_ratio: f64,
}

/// Configuration for letterbox/pillarbox detection.
#[derive(Debug, Clone)]
pub struct BlackBarConfig {
    /// Pixel luminance threshold for "black" (0-255).
    pub threshold: u8,
    /// Fraction of pixels in a row/column that must be black to count
    /// the entire row/column as a black bar (0.0-1.0).
    pub line_black_ratio: f64,
    /// Minimum bar thickness in pixels to report.
    pub min_bar_pixels: usize,
    /// Minimum fraction of analysed frames that must agree on a pattern
    /// for it to be considered dominant (0.0-1.0).
    pub consensus_ratio: f64,
}

impl Default for BlackBarConfig {
    fn default() -> Self {
        Self {
            threshold: 20,
            line_black_ratio: 0.95,
            min_bar_pixels: 4,
            consensus_ratio: 0.75,
        }
    }
}

/// Stateful letterbox/pillarbox detector.
///
/// Analyses per-frame black border regions and aggregates them over time
/// to determine the dominant pattern.
pub struct BlackBarDetector {
    config: BlackBarConfig,
    detections: Vec<BlackBarDetection>,
}

impl BlackBarDetector {
    /// Create a new detector with the given configuration.
    #[must_use]
    pub fn new(config: BlackBarConfig) -> Self {
        Self {
            config,
            detections: Vec::new(),
        }
    }

    /// Analyse a single Y-plane frame for black bars.
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
        if width == 0 || height == 0 {
            return Err(AnalysisError::InvalidInput(
                "Frame dimensions must be non-zero".to_string(),
            ));
        }

        let threshold = self.config.threshold;
        let line_ratio = self.config.line_black_ratio;

        // --- count black rows from top ---
        let mut top_rows = 0usize;
        for y in 0..height {
            let row_start = y * width;
            let black_count = y_plane[row_start..row_start + width]
                .iter()
                .filter(|&&p| p <= threshold)
                .count();
            if (black_count as f64 / width as f64) >= line_ratio {
                top_rows += 1;
            } else {
                break;
            }
        }

        // --- count black rows from bottom ---
        let mut bottom_rows = 0usize;
        for y in (0..height).rev() {
            // Do not double-count if the entire frame is black
            if y < top_rows {
                break;
            }
            let row_start = y * width;
            let black_count = y_plane[row_start..row_start + width]
                .iter()
                .filter(|&&p| p <= threshold)
                .count();
            if (black_count as f64 / width as f64) >= line_ratio {
                bottom_rows += 1;
            } else {
                break;
            }
        }

        // --- count black columns from left ---
        let mut left_cols = 0usize;
        for x in 0..width {
            let black_count = (0..height)
                .filter(|&y| y_plane[y * width + x] <= threshold)
                .count();
            if (black_count as f64 / height as f64) >= line_ratio {
                left_cols += 1;
            } else {
                break;
            }
        }

        // --- count black columns from right ---
        let mut right_cols = 0usize;
        for x in (0..width).rev() {
            if x < left_cols {
                break;
            }
            let black_count = (0..height)
                .filter(|&y| y_plane[y * width + x] <= threshold)
                .count();
            if (black_count as f64 / height as f64) >= line_ratio {
                right_cols += 1;
            } else {
                break;
            }
        }

        let has_top_bottom =
            top_rows >= self.config.min_bar_pixels && bottom_rows >= self.config.min_bar_pixels;
        let has_left_right =
            left_cols >= self.config.min_bar_pixels && right_cols >= self.config.min_bar_pixels;

        let pattern = match (has_top_bottom, has_left_right) {
            (true, true) => BlackBarPattern::Windowbox,
            (true, false) => BlackBarPattern::Letterbox,
            (false, true) => BlackBarPattern::Pillarbox,
            (false, false) => BlackBarPattern::None,
        };

        let active_height = height.saturating_sub(top_rows + bottom_rows);
        let active_width = width.saturating_sub(left_cols + right_cols);
        let total_pixels = width * height;
        let active_ratio = if total_pixels > 0 {
            (active_width * active_height) as f64 / total_pixels as f64
        } else {
            1.0
        };

        self.detections.push(BlackBarDetection {
            frame_number,
            pattern,
            top_rows,
            bottom_rows,
            left_cols,
            right_cols,
            active_ratio,
        });

        Ok(())
    }

    /// Return all per-frame detections.
    #[must_use]
    pub fn detections(&self) -> &[BlackBarDetection] {
        &self.detections
    }

    /// Finalize and return an aggregate result.
    #[must_use]
    pub fn finalize(self) -> BlackBarAnalysis {
        if self.detections.is_empty() {
            return BlackBarAnalysis {
                dominant_pattern: BlackBarPattern::None,
                confidence: 0.0,
                avg_top_rows: 0,
                avg_bottom_rows: 0,
                avg_left_cols: 0,
                avg_right_cols: 0,
                avg_active_ratio: 1.0,
                frame_count: 0,
            };
        }

        let n = self.detections.len() as f64;

        // Count pattern occurrences
        let mut letterbox_count = 0usize;
        let mut pillarbox_count = 0usize;
        let mut windowbox_count = 0usize;
        let mut none_count = 0usize;

        let mut sum_top = 0usize;
        let mut sum_bottom = 0usize;
        let mut sum_left = 0usize;
        let mut sum_right = 0usize;
        let mut sum_active = 0.0f64;

        for d in &self.detections {
            match d.pattern {
                BlackBarPattern::Letterbox => letterbox_count += 1,
                BlackBarPattern::Pillarbox => pillarbox_count += 1,
                BlackBarPattern::Windowbox => windowbox_count += 1,
                BlackBarPattern::None => none_count += 1,
            }
            sum_top += d.top_rows;
            sum_bottom += d.bottom_rows;
            sum_left += d.left_cols;
            sum_right += d.right_cols;
            sum_active += d.active_ratio;
        }

        let counts = [
            (BlackBarPattern::Letterbox, letterbox_count),
            (BlackBarPattern::Pillarbox, pillarbox_count),
            (BlackBarPattern::Windowbox, windowbox_count),
            (BlackBarPattern::None, none_count),
        ];

        let (dominant_pattern, dominant_count) = counts
            .iter()
            .max_by_key(|(_, c)| *c)
            .copied()
            .unwrap_or((BlackBarPattern::None, 0));

        let confidence = dominant_count as f64 / n;
        let frame_count = self.detections.len();

        BlackBarAnalysis {
            dominant_pattern,
            confidence,
            avg_top_rows: (sum_top as f64 / n).round() as usize,
            avg_bottom_rows: (sum_bottom as f64 / n).round() as usize,
            avg_left_cols: (sum_left as f64 / n).round() as usize,
            avg_right_cols: (sum_right as f64 / n).round() as usize,
            avg_active_ratio: sum_active / n,
            frame_count,
        }
    }
}

/// Aggregated black bar analysis result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlackBarAnalysis {
    /// Dominant pattern across all frames.
    pub dominant_pattern: BlackBarPattern,
    /// Confidence in the dominant pattern (0.0-1.0).
    pub confidence: f64,
    /// Average number of black rows at top.
    pub avg_top_rows: usize,
    /// Average number of black rows at bottom.
    pub avg_bottom_rows: usize,
    /// Average number of black columns on left.
    pub avg_left_cols: usize,
    /// Average number of black columns on right.
    pub avg_right_cols: usize,
    /// Average active picture area ratio.
    pub avg_active_ratio: f64,
    /// Number of frames analysed.
    pub frame_count: usize,
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

    // -----------------------------------------------------------------------
    // Letterbox / Pillarbox tests
    // -----------------------------------------------------------------------

    /// Helper: create a frame with black bars at top and bottom.
    fn make_letterbox_frame(width: usize, height: usize, bar_height: usize) -> Vec<u8> {
        let mut frame = vec![128u8; width * height];
        // Top bar
        for y in 0..bar_height {
            for x in 0..width {
                frame[y * width + x] = 0;
            }
        }
        // Bottom bar
        for y in (height - bar_height)..height {
            for x in 0..width {
                frame[y * width + x] = 0;
            }
        }
        frame
    }

    /// Helper: create a frame with black bars on left and right.
    fn make_pillarbox_frame(width: usize, height: usize, bar_width: usize) -> Vec<u8> {
        let mut frame = vec![128u8; width * height];
        for y in 0..height {
            for x in 0..bar_width {
                frame[y * width + x] = 0;
            }
            for x in (width - bar_width)..width {
                frame[y * width + x] = 0;
            }
        }
        frame
    }

    #[test]
    fn test_letterbox_detection() {
        let config = BlackBarConfig {
            threshold: 20,
            line_black_ratio: 0.95,
            min_bar_pixels: 4,
            consensus_ratio: 0.75,
        };
        let mut detector = BlackBarDetector::new(config);

        let frame = make_letterbox_frame(160, 120, 15);
        for i in 0..10 {
            detector
                .process_frame(&frame, 160, 120, i)
                .expect("should process");
        }

        let analysis = detector.finalize();
        assert_eq!(analysis.dominant_pattern, BlackBarPattern::Letterbox);
        assert!(analysis.confidence > 0.9);
        assert_eq!(analysis.avg_top_rows, 15);
        assert_eq!(analysis.avg_bottom_rows, 15);
        assert!(analysis.avg_active_ratio < 1.0);
    }

    #[test]
    fn test_pillarbox_detection() {
        let config = BlackBarConfig::default();
        let mut detector = BlackBarDetector::new(config);

        let frame = make_pillarbox_frame(160, 120, 20);
        for i in 0..10 {
            detector
                .process_frame(&frame, 160, 120, i)
                .expect("should process");
        }

        let analysis = detector.finalize();
        assert_eq!(analysis.dominant_pattern, BlackBarPattern::Pillarbox);
        assert!(analysis.confidence > 0.9);
        assert_eq!(analysis.avg_left_cols, 20);
        assert_eq!(analysis.avg_right_cols, 20);
    }

    #[test]
    fn test_windowbox_detection() {
        let config = BlackBarConfig::default();
        let mut detector = BlackBarDetector::new(config);

        // Create a frame with black bars on all sides
        let width = 160;
        let height = 120;
        let bar_h = 10;
        let bar_w = 15;
        let mut frame = vec![128u8; width * height];
        // Top/bottom bars
        for y in 0..bar_h {
            for x in 0..width {
                frame[y * width + x] = 0;
            }
        }
        for y in (height - bar_h)..height {
            for x in 0..width {
                frame[y * width + x] = 0;
            }
        }
        // Left/right bars
        for y in 0..height {
            for x in 0..bar_w {
                frame[y * width + x] = 0;
            }
            for x in (width - bar_w)..width {
                frame[y * width + x] = 0;
            }
        }

        for i in 0..5 {
            detector
                .process_frame(&frame, width, height, i)
                .expect("should process");
        }

        let analysis = detector.finalize();
        assert_eq!(analysis.dominant_pattern, BlackBarPattern::Windowbox);
    }

    #[test]
    fn test_no_black_bars() {
        let config = BlackBarConfig::default();
        let mut detector = BlackBarDetector::new(config);

        let frame = vec![128u8; 160 * 120];
        for i in 0..5 {
            detector
                .process_frame(&frame, 160, 120, i)
                .expect("should process");
        }

        let analysis = detector.finalize();
        assert_eq!(analysis.dominant_pattern, BlackBarPattern::None);
        assert!((analysis.avg_active_ratio - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_black_bar_empty_finalize() {
        let config = BlackBarConfig::default();
        let detector = BlackBarDetector::new(config);
        let analysis = detector.finalize();
        assert_eq!(analysis.dominant_pattern, BlackBarPattern::None);
        assert_eq!(analysis.frame_count, 0);
    }

    #[test]
    fn test_black_bar_pattern_display() {
        assert_eq!(BlackBarPattern::Letterbox.to_string(), "Letterbox");
        assert_eq!(BlackBarPattern::Pillarbox.to_string(), "Pillarbox");
        assert_eq!(BlackBarPattern::Windowbox.to_string(), "Windowbox");
        assert_eq!(BlackBarPattern::None.to_string(), "None");
    }

    #[test]
    fn test_black_bar_invalid_input() {
        let config = BlackBarConfig::default();
        let mut detector = BlackBarDetector::new(config);
        let frame = vec![0u8; 100]; // wrong size
        let result = detector.process_frame(&frame, 160, 120, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_letterbox_active_ratio() {
        let config = BlackBarConfig::default();
        let mut detector = BlackBarDetector::new(config);

        // 120 height, 10 top + 10 bottom = 100 active rows => 100/120 ≈ 0.833
        let frame = make_letterbox_frame(160, 120, 10);
        detector
            .process_frame(&frame, 160, 120, 0)
            .expect("should process");

        let analysis = detector.finalize();
        let expected = (160.0 * 100.0) / (160.0 * 120.0);
        assert!((analysis.avg_active_ratio - expected).abs() < 0.01);
    }

    #[test]
    fn test_thin_bars_below_min_ignored() {
        let config = BlackBarConfig {
            min_bar_pixels: 10,
            ..Default::default()
        };
        let mut detector = BlackBarDetector::new(config);

        // Only 5 rows of black at top/bottom (below min_bar_pixels=10)
        let frame = make_letterbox_frame(160, 120, 5);
        detector
            .process_frame(&frame, 160, 120, 0)
            .expect("should process");

        let analysis = detector.finalize();
        assert_eq!(analysis.dominant_pattern, BlackBarPattern::None);
    }
}
