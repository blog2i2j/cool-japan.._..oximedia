#![allow(dead_code)]
//! Temporal statistics analysis for video content.
//!
//! This module computes per-segment and whole-clip temporal statistics
//! such as average frame luminance over time, flicker rate, frame-to-frame
//! differences, and temporal complexity metrics useful for adaptive encoding
//! and quality-of-experience estimation.

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for temporal statistics computation.
#[derive(Debug, Clone)]
pub struct TemporalStatsConfig {
    /// Window size (in frames) for rolling statistics.
    pub window_size: usize,
    /// Threshold for considering a frame-to-frame change significant (0.0..1.0).
    pub change_threshold: f64,
    /// Minimum number of frames required before statistics are valid.
    pub min_frames: usize,
    /// Enable flicker detection pass.
    pub detect_flicker: bool,
}

impl Default for TemporalStatsConfig {
    fn default() -> Self {
        Self {
            window_size: 30,
            change_threshold: 0.05,
            min_frames: 2,
            detect_flicker: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Frame-level record
// ---------------------------------------------------------------------------

/// Statistics for a single frame.
#[derive(Debug, Clone, Copy)]
pub struct FrameRecord {
    /// Frame index (0-based).
    pub index: usize,
    /// Mean luminance (0.0..255.0).
    pub mean_luma: f64,
    /// Standard deviation of luminance.
    pub std_luma: f64,
    /// Absolute difference from previous frame mean luminance.
    pub delta_luma: f64,
}

// ---------------------------------------------------------------------------
// Segment summary
// ---------------------------------------------------------------------------

/// Summary statistics over a window / segment of frames.
#[derive(Debug, Clone)]
pub struct SegmentStats {
    /// Start frame index (inclusive).
    pub start_frame: usize,
    /// End frame index (inclusive).
    pub end_frame: usize,
    /// Mean of mean-luminance values in the segment.
    pub avg_luma: f64,
    /// Max delta-luma in the segment.
    pub max_delta: f64,
    /// Min delta-luma in the segment.
    pub min_delta: f64,
    /// Average delta-luma (temporal activity).
    pub avg_delta: f64,
    /// Variance of mean-luma across frames in the segment.
    pub luma_variance: f64,
    /// Number of frames where delta exceeds the change threshold.
    pub significant_changes: usize,
}

// ---------------------------------------------------------------------------
// Flicker event
// ---------------------------------------------------------------------------

/// Describes a detected flicker event — rapid luminance oscillation.
#[derive(Debug, Clone, Copy)]
pub struct FlickerEvent {
    /// Frame index where flicker was first detected.
    pub start_frame: usize,
    /// Frame index where flicker ended.
    pub end_frame: usize,
    /// Peak-to-peak luminance swing during the event.
    pub amplitude: f64,
    /// Estimated frequency in oscillations-per-frame.
    pub frequency: f64,
}

// ---------------------------------------------------------------------------
// Overall result
// ---------------------------------------------------------------------------

/// Complete temporal statistics result for an analyzed clip.
#[derive(Debug, Clone)]
pub struct TemporalStatsResult {
    /// Total frames processed.
    pub total_frames: usize,
    /// Per-frame records.
    pub frame_records: Vec<FrameRecord>,
    /// Per-segment (rolling window) summaries.
    pub segments: Vec<SegmentStats>,
    /// Detected flicker events (empty if detection was disabled).
    pub flicker_events: Vec<FlickerEvent>,
    /// Global average luminance across all frames.
    pub global_avg_luma: f64,
    /// Global temporal complexity (average of all deltas).
    pub global_temporal_complexity: f64,
    /// Ratio of frames with significant change vs total frames.
    pub change_ratio: f64,
}

// ---------------------------------------------------------------------------
// Analyzer
// ---------------------------------------------------------------------------

/// Stateful temporal statistics analyzer fed one frame at a time.
#[derive(Debug)]
pub struct TemporalStatsAnalyzer {
    /// Active configuration.
    config: TemporalStatsConfig,
    /// Accumulated frame records.
    records: Vec<FrameRecord>,
    /// Previous frame mean luminance for delta computation.
    prev_mean: Option<f64>,
    /// Rolling window buffer of mean-luma values.
    window: VecDeque<f64>,
    /// Rolling window buffer of delta values.
    delta_window: VecDeque<f64>,
}

impl TemporalStatsAnalyzer {
    /// Create a new analyzer with the given configuration.
    pub fn new(config: TemporalStatsConfig) -> Self {
        Self {
            window: VecDeque::with_capacity(config.window_size),
            delta_window: VecDeque::with_capacity(config.window_size),
            config,
            records: Vec::new(),
            prev_mean: None,
        }
    }

    /// Feed a single Y-plane frame into the analyzer.
    ///
    /// `y_plane` must contain `width * height` bytes.
    pub fn push_frame(&mut self, y_plane: &[u8], width: usize, height: usize) {
        let n = width * height;
        if n == 0 {
            return;
        }

        // Mean luminance
        #[allow(clippy::cast_precision_loss)]
        let mean = y_plane.iter().map(|&v| f64::from(v)).sum::<f64>() / n as f64;

        // Std-dev of luminance
        #[allow(clippy::cast_precision_loss)]
        let variance = y_plane
            .iter()
            .map(|&v| {
                let d = f64::from(v) - mean;
                d * d
            })
            .sum::<f64>()
            / n as f64;
        let std_dev = variance.sqrt();

        let delta = self.prev_mean.map_or(0.0, |pm| (mean - pm).abs());
        self.prev_mean = Some(mean);

        let index = self.records.len();
        self.records.push(FrameRecord {
            index,
            mean_luma: mean,
            std_luma: std_dev,
            delta_luma: delta,
        });

        // Maintain rolling window
        if self.window.len() == self.config.window_size {
            self.window.pop_front();
            self.delta_window.pop_front();
        }
        self.window.push_back(mean);
        self.delta_window.push_back(delta);
    }

    /// Compute the current rolling-window segment statistics.
    pub fn current_segment_stats(&self) -> Option<SegmentStats> {
        if self.window.len() < self.config.min_frames {
            return None;
        }
        let start = self.records.len().saturating_sub(self.window.len());
        let end = self.records.len().saturating_sub(1);
        Some(compute_segment(
            &self.window,
            &self.delta_window,
            start,
            end,
            self.config.change_threshold,
        ))
    }

    /// Finalize analysis and return the complete result.
    pub fn finalize(self) -> TemporalStatsResult {
        let total = self.records.len();
        if total == 0 {
            return TemporalStatsResult {
                total_frames: 0,
                frame_records: Vec::new(),
                segments: Vec::new(),
                flicker_events: Vec::new(),
                global_avg_luma: 0.0,
                global_temporal_complexity: 0.0,
                change_ratio: 0.0,
            };
        }

        // Global stats
        let global_avg_luma = self.records.iter().map(|r| r.mean_luma).sum::<f64>() / total as f64;
        let global_temporal_complexity = if total > 1 {
            self.records.iter().map(|r| r.delta_luma).sum::<f64>() / (total - 1) as f64
        } else {
            0.0
        };
        let sig_count = self
            .records
            .iter()
            .filter(|r| r.delta_luma / 255.0 > self.config.change_threshold)
            .count();
        #[allow(clippy::cast_precision_loss)]
        let change_ratio = sig_count as f64 / total as f64;

        // Build segments
        let segments = build_segments(
            &self.records,
            self.config.window_size,
            self.config.change_threshold,
        );

        // Flicker detection
        let flicker_events = if self.config.detect_flicker {
            detect_flicker_events(&self.records, self.config.change_threshold)
        } else {
            Vec::new()
        };

        TemporalStatsResult {
            total_frames: total,
            frame_records: self.records,
            segments,
            flicker_events,
            global_avg_luma,
            global_temporal_complexity,
            change_ratio,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute segment stats from window slices.
fn compute_segment(
    luma_window: &VecDeque<f64>,
    delta_window: &VecDeque<f64>,
    start: usize,
    end: usize,
    change_threshold: f64,
) -> SegmentStats {
    let n = luma_window.len();
    #[allow(clippy::cast_precision_loss)]
    let avg_luma = luma_window.iter().sum::<f64>() / n as f64;
    let luma_variance = luma_window
        .iter()
        .map(|v| {
            let d = v - avg_luma;
            d * d
        })
        .sum::<f64>()
        / n as f64;

    let max_delta = delta_window.iter().copied().fold(0.0_f64, f64::max);
    let min_delta = delta_window.iter().copied().fold(f64::MAX, f64::min);
    #[allow(clippy::cast_precision_loss)]
    let avg_delta = delta_window.iter().sum::<f64>() / n as f64;
    let sig = delta_window
        .iter()
        .filter(|&&d| d / 255.0 > change_threshold)
        .count();

    SegmentStats {
        start_frame: start,
        end_frame: end,
        avg_luma,
        max_delta,
        min_delta,
        avg_delta,
        luma_variance,
        significant_changes: sig,
    }
}

/// Build non-overlapping segment summaries.
fn build_segments(
    records: &[FrameRecord],
    window_size: usize,
    change_threshold: f64,
) -> Vec<SegmentStats> {
    let mut segments = Vec::new();
    let mut i = 0;
    while i < records.len() {
        let end = (i + window_size).min(records.len());
        let slice = &records[i..end];
        let n = slice.len();
        if n == 0 {
            break;
        }
        #[allow(clippy::cast_precision_loss)]
        let avg_luma = slice.iter().map(|r| r.mean_luma).sum::<f64>() / n as f64;
        let luma_variance = slice
            .iter()
            .map(|r| {
                let d = r.mean_luma - avg_luma;
                d * d
            })
            .sum::<f64>()
            / n as f64;
        let max_delta = slice.iter().map(|r| r.delta_luma).fold(0.0_f64, f64::max);
        let min_delta = slice.iter().map(|r| r.delta_luma).fold(f64::MAX, f64::min);
        #[allow(clippy::cast_precision_loss)]
        let avg_delta = slice.iter().map(|r| r.delta_luma).sum::<f64>() / n as f64;
        let sig = slice
            .iter()
            .filter(|r| r.delta_luma / 255.0 > change_threshold)
            .count();

        segments.push(SegmentStats {
            start_frame: i,
            end_frame: end - 1,
            avg_luma,
            max_delta,
            min_delta,
            avg_delta,
            luma_variance,
            significant_changes: sig,
        });
        i = end;
    }
    segments
}

/// Simple flicker detector: looks for rapid sign-alternating delta sequences.
fn detect_flicker_events(records: &[FrameRecord], _change_threshold: f64) -> Vec<FlickerEvent> {
    if records.len() < 4 {
        return Vec::new();
    }
    let mut events = Vec::new();
    let mut run_start: Option<usize> = None;
    let mut prev_sign: Option<bool> = None;
    let mut run_len = 0usize;

    for i in 1..records.len() {
        let diff = records[i].mean_luma - records[i - 1].mean_luma;
        let positive = diff >= 0.0;
        let alternates = prev_sign.is_some_and(|ps| ps != positive);

        if alternates && diff.abs() > 2.0 {
            if run_start.is_none() {
                run_start = Some(i - 1);
            }
            run_len += 1;
        } else {
            if run_len >= 3 {
                if let Some(start) = run_start {
                    let end = i - 1;
                    let amp = records[start..=end]
                        .iter()
                        .map(|r| r.mean_luma)
                        .fold(f64::MIN, f64::max)
                        - records[start..=end]
                            .iter()
                            .map(|r| r.mean_luma)
                            .fold(f64::MAX, f64::min);
                    let span = (end - start).max(1);
                    #[allow(clippy::cast_precision_loss)]
                    let freq = run_len as f64 / (2.0 * span as f64);
                    events.push(FlickerEvent {
                        start_frame: start,
                        end_frame: end,
                        amplitude: amp,
                        frequency: freq,
                    });
                }
            }
            run_start = None;
            run_len = 0;
        }
        prev_sign = Some(positive);
    }

    // Close trailing run
    if run_len >= 3 {
        if let Some(start) = run_start {
            let end = records.len() - 1;
            let amp = records[start..=end]
                .iter()
                .map(|r| r.mean_luma)
                .fold(f64::MIN, f64::max)
                - records[start..=end]
                    .iter()
                    .map(|r| r.mean_luma)
                    .fold(f64::MAX, f64::min);
            let span = (end - start).max(1);
            #[allow(clippy::cast_precision_loss)]
            let freq = run_len as f64 / (2.0 * span as f64);
            events.push(FlickerEvent {
                start_frame: start,
                end_frame: end,
                amplitude: amp,
                frequency: freq,
            });
        }
    }

    events
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(width: usize, height: usize, value: u8) -> Vec<u8> {
        vec![value; width * height]
    }

    #[test]
    fn test_config_defaults() {
        let cfg = TemporalStatsConfig::default();
        assert_eq!(cfg.window_size, 30);
        assert!(cfg.detect_flicker);
        assert_eq!(cfg.min_frames, 2);
    }

    #[test]
    fn test_empty_analyzer() {
        let a = TemporalStatsAnalyzer::new(TemporalStatsConfig::default());
        let result = a.finalize();
        assert_eq!(result.total_frames, 0);
        assert!(result.segments.is_empty());
        assert!(result.flicker_events.is_empty());
    }

    #[test]
    fn test_single_frame() {
        let mut a = TemporalStatsAnalyzer::new(TemporalStatsConfig::default());
        a.push_frame(&make_frame(8, 8, 128), 8, 8);
        let result = a.finalize();
        assert_eq!(result.total_frames, 1);
        assert!((result.global_avg_luma - 128.0).abs() < 0.01);
    }

    #[test]
    fn test_constant_luminance_no_change() {
        let mut a = TemporalStatsAnalyzer::new(TemporalStatsConfig::default());
        for _ in 0..10 {
            a.push_frame(&make_frame(4, 4, 100), 4, 4);
        }
        let result = a.finalize();
        assert_eq!(result.total_frames, 10);
        assert!(result.global_temporal_complexity < 0.01);
        assert!((result.change_ratio).abs() < 0.01);
    }

    #[test]
    fn test_step_change_detected() {
        let cfg = TemporalStatsConfig {
            change_threshold: 0.01,
            ..Default::default()
        };
        let mut a = TemporalStatsAnalyzer::new(cfg);
        for _ in 0..5 {
            a.push_frame(&make_frame(4, 4, 50), 4, 4);
        }
        for _ in 0..5 {
            a.push_frame(&make_frame(4, 4, 200), 4, 4);
        }
        let result = a.finalize();
        assert!(result.global_temporal_complexity > 0.0);
        // The single big step should produce at least one significant change
        assert!(result.change_ratio > 0.0);
    }

    #[test]
    fn test_segment_construction() {
        let cfg = TemporalStatsConfig {
            window_size: 5,
            ..Default::default()
        };
        let mut a = TemporalStatsAnalyzer::new(cfg);
        for i in 0..12 {
            let val = if i < 6 { 80 } else { 160 };
            a.push_frame(&make_frame(4, 4, val), 4, 4);
        }
        let result = a.finalize();
        assert!(!result.segments.is_empty());
        // 12 frames / window 5 => at least 2 full segments
        assert!(result.segments.len() >= 2);
    }

    #[test]
    fn test_current_segment_stats() {
        let cfg = TemporalStatsConfig {
            window_size: 4,
            min_frames: 2,
            ..Default::default()
        };
        let mut a = TemporalStatsAnalyzer::new(cfg);
        assert!(a.current_segment_stats().is_none());
        a.push_frame(&make_frame(4, 4, 100), 4, 4);
        assert!(a.current_segment_stats().is_none()); // only 1 frame
        a.push_frame(&make_frame(4, 4, 110), 4, 4);
        let seg = a.current_segment_stats();
        assert!(seg.is_some());
        let seg = seg.expect("expected seg to be Some/Ok");
        assert!((seg.avg_luma - 105.0).abs() < 0.01);
    }

    #[test]
    fn test_flicker_detection() {
        let cfg = TemporalStatsConfig {
            detect_flicker: true,
            ..Default::default()
        };
        let mut a = TemporalStatsAnalyzer::new(cfg);
        // Alternate bright/dark to simulate flicker
        for i in 0..20 {
            let val = if i % 2 == 0 { 40 } else { 200 };
            a.push_frame(&make_frame(4, 4, val), 4, 4);
        }
        let result = a.finalize();
        assert!(!result.flicker_events.is_empty());
        assert!(result.flicker_events[0].amplitude > 100.0);
    }

    #[test]
    fn test_flicker_disabled() {
        let cfg = TemporalStatsConfig {
            detect_flicker: false,
            ..Default::default()
        };
        let mut a = TemporalStatsAnalyzer::new(cfg);
        for i in 0..20 {
            let val = if i % 2 == 0 { 40 } else { 200 };
            a.push_frame(&make_frame(4, 4, val), 4, 4);
        }
        let result = a.finalize();
        assert!(result.flicker_events.is_empty());
    }

    #[test]
    fn test_zero_dimension_frame_ignored() {
        let mut a = TemporalStatsAnalyzer::new(TemporalStatsConfig::default());
        a.push_frame(&[], 0, 0);
        let result = a.finalize();
        assert_eq!(result.total_frames, 0);
    }

    #[test]
    fn test_frame_record_fields() {
        let mut a = TemporalStatsAnalyzer::new(TemporalStatsConfig::default());
        a.push_frame(&make_frame(2, 2, 120), 2, 2);
        a.push_frame(&make_frame(2, 2, 130), 2, 2);
        let result = a.finalize();
        assert_eq!(result.frame_records.len(), 2);
        assert_eq!(result.frame_records[0].index, 0);
        assert!((result.frame_records[0].delta_luma).abs() < 0.01); // first frame has 0 delta
        assert!((result.frame_records[1].delta_luma - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_std_dev_uniform_frame() {
        let mut a = TemporalStatsAnalyzer::new(TemporalStatsConfig::default());
        a.push_frame(&make_frame(4, 4, 100), 4, 4);
        let result = a.finalize();
        // Uniform frame should have zero std-dev
        assert!(result.frame_records[0].std_luma < 0.01);
    }

    #[test]
    fn test_std_dev_varied_frame() {
        let mut a = TemporalStatsAnalyzer::new(TemporalStatsConfig::default());
        // Half 0, half 255
        let mut frame = vec![0u8; 8];
        for px in frame.iter_mut().skip(4) {
            *px = 254;
        }
        a.push_frame(&frame, 4, 2);
        let result = a.finalize();
        assert!(result.frame_records[0].std_luma > 100.0);
    }
}
