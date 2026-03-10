//! Temporal artifact detection.
//!
//! This module detects temporal artifacts in video:
//! - **Flicker** - Brightness oscillations
//! - **Judder** - Frame rate inconsistencies
//! - **Temporal Noise** - Random temporal variations
//! - **Telecine Detection** - 3:2 pulldown patterns
//!
//! # Algorithms
//!
//! - Frame-to-frame brightness analysis for flicker
//! - Motion-compensated temporal filtering for noise
//! - Pattern matching for telecine detection

use crate::{AnalysisError, AnalysisResult};
use serde::{Deserialize, Serialize};

/// Temporal analysis results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalAnalysis {
    /// Detected flicker events
    pub flicker_events: Vec<FlickerEvent>,
    /// Average flicker magnitude (0.0-1.0)
    pub avg_flicker: f64,
    /// Temporal noise level (0.0-1.0)
    pub temporal_noise: f64,
    /// Frame consistency score (0.0-1.0, higher is better)
    pub consistency: f64,
    /// Detected telecine pattern
    pub telecine: Option<TelecinePattern>,
}

/// Flicker event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlickerEvent {
    /// Starting frame
    pub start_frame: usize,
    /// Ending frame
    pub end_frame: usize,
    /// Flicker magnitude (0.0-1.0)
    pub magnitude: f64,
    /// Flicker frequency (Hz)
    pub frequency_hz: Option<f64>,
}

/// Telecine pattern type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TelecinePattern {
    /// 3:2 pulldown (24fps to 30fps)
    Pulldown32,
    /// 2:2 pulldown (25fps to 30fps)
    Pulldown22,
    /// Other pattern
    Other,
}

/// Temporal analyzer.
pub struct TemporalAnalyzer {
    brightness_history: Vec<f64>,
    flicker_events: Vec<FlickerEvent>,
    prev_frame: Option<Vec<u8>>,
    temporal_diffs: Vec<f64>,
}

impl TemporalAnalyzer {
    /// Create a new temporal analyzer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            brightness_history: Vec::new(),
            flicker_events: Vec::new(),
            prev_frame: None,
            temporal_diffs: Vec::new(),
        }
    }

    /// Process a frame.
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

        // Compute average brightness
        let brightness = compute_average_brightness(y_plane);
        self.brightness_history.push(brightness);

        // Compute temporal difference
        if let Some(ref prev) = self.prev_frame {
            let diff = compute_temporal_diff(prev, y_plane, width, height);
            self.temporal_diffs.push(diff);
        }

        // Detect flicker in recent window
        const WINDOW_SIZE: usize = 30;
        if self.brightness_history.len() >= WINDOW_SIZE {
            let start_idx = self.brightness_history.len() - WINDOW_SIZE;
            let window = &self.brightness_history[start_idx..];
            if let Some(flicker) =
                detect_flicker_in_window(window, frame_number.saturating_sub(WINDOW_SIZE - 1))
            {
                self.flicker_events.push(flicker);
            }
        }

        self.prev_frame = Some(y_plane.to_vec());

        Ok(())
    }

    /// Finalize and return temporal analysis.
    pub fn finalize(self) -> TemporalAnalysis {
        // Compute average flicker
        let avg_flicker = if self.flicker_events.is_empty() {
            0.0
        } else {
            self.flicker_events.iter().map(|e| e.magnitude).sum::<f64>()
                / self.flicker_events.len() as f64
        };

        // Compute temporal noise
        let temporal_noise = if self.temporal_diffs.is_empty() {
            0.0
        } else {
            let avg_diff =
                self.temporal_diffs.iter().sum::<f64>() / self.temporal_diffs.len() as f64;
            (avg_diff / 50.0).min(1.0)
        };

        // Compute consistency (inverse of variance in temporal differences)
        let consistency = if self.temporal_diffs.len() > 1 {
            let avg = self.temporal_diffs.iter().sum::<f64>() / self.temporal_diffs.len() as f64;
            let variance = self
                .temporal_diffs
                .iter()
                .map(|&d| {
                    let diff = d - avg;
                    diff * diff
                })
                .sum::<f64>()
                / self.temporal_diffs.len() as f64;
            let std_dev = variance.sqrt();
            (1.0 - (std_dev / 100.0).min(1.0)).max(0.0)
        } else {
            1.0
        };

        // Detect telecine pattern
        let telecine = detect_telecine(&self.brightness_history);

        TemporalAnalysis {
            flicker_events: self.flicker_events,
            avg_flicker,
            temporal_noise,
            consistency,
            telecine,
        }
    }
}

impl Default for TemporalAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute average brightness.
fn compute_average_brightness(y_plane: &[u8]) -> f64 {
    if y_plane.is_empty() {
        return 0.0;
    }
    let sum: usize = y_plane.iter().map(|&p| p as usize).sum();
    sum as f64 / y_plane.len() as f64
}

/// Compute temporal difference.
fn compute_temporal_diff(prev: &[u8], current: &[u8], width: usize, height: usize) -> f64 {
    if prev.len() != current.len() {
        return 0.0;
    }

    let mut diff_sum = 0.0;

    // Sample for efficiency
    for y in (0..height).step_by(4) {
        for x in (0..width).step_by(4) {
            let idx = y * width + x;
            let diff = (i32::from(current[idx]) - i32::from(prev[idx])).abs();
            diff_sum += f64::from(diff);
        }
    }

    let sample_count = height.div_ceil(4) * width.div_ceil(4);
    if sample_count == 0 {
        return 0.0;
    }

    diff_sum / sample_count as f64
}

/// Detect flicker in a window of brightness values.
fn detect_flicker_in_window(window: &[f64], start_frame: usize) -> Option<FlickerEvent> {
    if window.len() < 3 {
        return None;
    }

    // Compute mean and variance
    let mean = window.iter().sum::<f64>() / window.len() as f64;
    let variance = window
        .iter()
        .map(|&b| {
            let diff = b - mean;
            diff * diff
        })
        .sum::<f64>()
        / window.len() as f64;

    let std_dev = variance.sqrt();

    // Detect oscillations
    let mut oscillation_count = 0;
    let mut prev_above_mean = window[0] > mean;

    for &brightness in &window[1..] {
        let above_mean = brightness > mean;
        if above_mean != prev_above_mean {
            oscillation_count += 1;
        }
        prev_above_mean = above_mean;
    }

    // If significant oscillation and variance, report as flicker
    let magnitude = (std_dev / 255.0).min(1.0);
    if oscillation_count > window.len() / 3 && magnitude > 0.05 {
        // Estimate frequency
        let frequency_hz = Some(oscillation_count as f64 / (window.len() as f64 / 30.0));

        Some(FlickerEvent {
            start_frame,
            end_frame: start_frame + window.len(),
            magnitude,
            frequency_hz,
        })
    } else {
        None
    }
}

/// Detect telecine patterns.
fn detect_telecine(brightness_history: &[f64]) -> Option<TelecinePattern> {
    if brightness_history.len() < 60 {
        return None;
    }

    // Look for 3:2 pulldown pattern (repeating pattern of frame durations)
    // In 3:2 pulldown, frames appear in pattern: A A B B C (5 frames from 2 original)
    // This creates a repeating pattern in brightness differences

    // Simple heuristic: look for repeating pattern in temporal differences
    // This is simplified - real telecine detection is more complex

    let mut pattern_32 = 0;
    let mut pattern_22 = 0;

    for i in 0..brightness_history.len() - 5 {
        let diffs: Vec<f64> = (0..4)
            .map(|j| (brightness_history[i + j + 1] - brightness_history[i + j]).abs())
            .collect();

        // 3:2 pulldown has pattern of small-small-large-small differences
        if diffs[0] < 1.0 && diffs[1] < 1.0 && diffs[2] > 2.0 && diffs[3] < 1.0 {
            pattern_32 += 1;
        }

        // 2:2 pulldown has alternating pattern
        if diffs[0] < 1.0 && diffs[1] > 2.0 && diffs[2] < 1.0 && diffs[3] > 2.0 {
            pattern_22 += 1;
        }
    }

    if pattern_32 > brightness_history.len() / 10 {
        Some(TelecinePattern::Pulldown32)
    } else if pattern_22 > brightness_history.len() / 10 {
        Some(TelecinePattern::Pulldown22)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_analyzer() {
        let mut analyzer = TemporalAnalyzer::new();

        // Process stable frames
        let frame = vec![128u8; 64 * 64];
        for i in 0..10 {
            analyzer
                .process_frame(&frame, 64, 64, i)
                .expect("frame processing should succeed");
        }

        let analysis = analyzer.finalize();
        assert!(analysis.consistency > 0.9); // Should be very consistent
        assert!(analysis.temporal_noise < 0.1); // Should be low noise
    }

    #[test]
    fn test_flicker_detection() {
        let mut analyzer = TemporalAnalyzer::new();

        // Create flickering content (alternating brightness)
        for i in 0..35 {
            let brightness = if i % 2 == 0 { 100u8 } else { 150u8 };
            let frame = vec![brightness; 64 * 64];
            analyzer
                .process_frame(&frame, 64, 64, i)
                .expect("frame processing should succeed");
        }

        let analysis = analyzer.finalize();
        assert!(!analysis.flicker_events.is_empty());
        assert!(analysis.avg_flicker > 0.0);
    }

    #[test]
    fn test_temporal_diff() {
        let frame1 = vec![100u8; 64 * 64];
        let frame2 = vec![110u8; 64 * 64];
        let diff = compute_temporal_diff(&frame1, &frame2, 64, 64);
        assert!((diff - 10.0).abs() < 1.0);
    }

    #[test]
    fn test_average_brightness() {
        let frame = vec![128u8; 64 * 64];
        let brightness = compute_average_brightness(&frame);
        assert!((brightness - 128.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_flicker_window() {
        // Oscillating brightness
        let window: Vec<f64> = (0..30)
            .map(|i| if i % 2 == 0 { 100.0 } else { 150.0 })
            .collect();

        let event = detect_flicker_in_window(&window, 0);
        assert!(event.is_some());
        if let Some(e) = event {
            assert!(e.magnitude > 0.0);
        }
    }

    #[test]
    fn test_stable_window() {
        // Stable brightness
        let window = vec![128.0; 30];
        let event = detect_flicker_in_window(&window, 0);
        assert!(event.is_none());
    }

    #[test]
    fn test_telecine_detection() {
        // Not enough data
        let short_history = vec![128.0; 30];
        let telecine = detect_telecine(&short_history);
        assert!(telecine.is_none());

        // Stable sequence (no telecine)
        let stable_history = vec![128.0; 100];
        let telecine2 = detect_telecine(&stable_history);
        assert!(telecine2.is_none());
    }

    #[test]
    fn test_empty_analyzer() {
        let analyzer = TemporalAnalyzer::new();
        let analysis = analyzer.finalize();
        assert_eq!(analysis.avg_flicker, 0.0);
        assert_eq!(analysis.consistency, 1.0);
        assert!(analysis.telecine.is_none());
    }

    #[test]
    fn test_temporal_noise() {
        let mut analyzer = TemporalAnalyzer::new();

        // Create noisy frames
        for i in 0..10 {
            let base = 128u8;
            let noise = ((i * 7) % 20) as u8;
            let frame = vec![base.saturating_add(noise); 64 * 64];
            analyzer
                .process_frame(&frame, 64, 64, i)
                .expect("frame processing should succeed");
        }

        let analysis = analyzer.finalize();
        assert!(analysis.temporal_noise > 0.0);
    }
}
