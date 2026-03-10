#![allow(dead_code)]
//! GPU timing and profiling utilities.
//!
//! This module provides high-resolution timing infrastructure for measuring
//! GPU operation latencies, frame times, and pipeline stage durations.
//! It maintains a rolling history for statistical analysis.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// A named timing region for GPU profiling.
#[derive(Debug, Clone)]
pub struct TimerRegion {
    /// Human-readable label for this region.
    pub label: String,
    /// Start timestamp.
    pub start: Instant,
    /// End timestamp (set when the region is stopped).
    pub end: Option<Instant>,
}

impl TimerRegion {
    /// Create a new timer region with the given label. Starts immediately.
    #[must_use]
    pub fn start(label: &str) -> Self {
        Self {
            label: label.to_string(),
            start: Instant::now(),
            end: None,
        }
    }

    /// Stop the timer region.
    pub fn stop(&mut self) {
        self.end = Some(Instant::now());
    }

    /// Return the elapsed duration, or duration since start if still running.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        match self.end {
            Some(end) => end.duration_since(self.start),
            None => self.start.elapsed(),
        }
    }

    /// Check if the region has been stopped.
    #[must_use]
    pub fn is_stopped(&self) -> bool {
        self.end.is_some()
    }
}

/// A single timing sample with label and duration.
#[derive(Debug, Clone)]
pub struct TimingSample {
    /// Label identifying what was timed.
    pub label: String,
    /// Measured duration.
    pub duration: Duration,
    /// Frame number when this sample was taken.
    pub frame_number: u64,
}

/// Configuration for the GPU timer.
#[derive(Debug, Clone)]
pub struct GpuTimerConfig {
    /// Maximum number of samples to keep in the rolling history.
    pub max_history: usize,
    /// Whether to enable timing collection.
    pub enabled: bool,
    /// Target frame time for performance budgeting.
    pub target_frame_time: Duration,
}

impl Default for GpuTimerConfig {
    fn default() -> Self {
        Self {
            max_history: 300,
            enabled: true,
            target_frame_time: Duration::from_micros(16_667), // ~60 FPS
        }
    }
}

/// Statistical summary of timing data.
#[derive(Debug, Clone)]
pub struct TimingStats {
    /// Minimum duration in the sample window.
    pub min: Duration,
    /// Maximum duration in the sample window.
    pub max: Duration,
    /// Mean (average) duration.
    pub mean: Duration,
    /// Median duration.
    pub median: Duration,
    /// 95th percentile duration.
    pub p95: Duration,
    /// 99th percentile duration.
    pub p99: Duration,
    /// Standard deviation in microseconds.
    pub std_dev_us: f64,
    /// Number of samples.
    pub sample_count: usize,
}

impl TimingStats {
    /// Compute timing statistics from a slice of durations.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn from_durations(durations: &[Duration]) -> Option<Self> {
        if durations.is_empty() {
            return None;
        }

        let mut sorted: Vec<Duration> = durations.to_vec();
        sorted.sort();

        let count = sorted.len();
        let min = sorted[0];
        let max = sorted[count - 1];
        let median = sorted[count / 2];

        let sum_us: f64 = sorted.iter().map(|d| d.as_micros() as f64).sum();
        let mean_us = sum_us / count as f64;
        let mean = Duration::from_micros(mean_us as u64);

        let p95_idx = ((count as f64) * 0.95).ceil() as usize;
        let p95 = sorted[p95_idx.min(count - 1)];

        let p99_idx = ((count as f64) * 0.99).ceil() as usize;
        let p99 = sorted[p99_idx.min(count - 1)];

        let variance: f64 = sorted
            .iter()
            .map(|d| {
                let diff = d.as_micros() as f64 - mean_us;
                diff * diff
            })
            .sum::<f64>()
            / count as f64;
        let std_dev_us = variance.sqrt();

        Some(Self {
            min,
            max,
            mean,
            median,
            p95,
            p99,
            std_dev_us,
            sample_count: count,
        })
    }

    /// Return mean as frames per second equivalent.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn mean_fps(&self) -> f64 {
        let mean_secs = self.mean.as_secs_f64();
        if mean_secs > 0.0 {
            1.0 / mean_secs
        } else {
            0.0
        }
    }
}

/// Frame time tracker that measures per-frame GPU durations.
#[derive(Debug, Clone)]
pub struct FrameTimer {
    /// Rolling history of frame durations.
    history: VecDeque<Duration>,
    /// Maximum history size.
    max_history: usize,
    /// Current frame start time.
    frame_start: Option<Instant>,
    /// Total frames measured.
    total_frames: u64,
}

impl FrameTimer {
    /// Create a new frame timer with the given history capacity.
    #[must_use]
    pub fn new(max_history: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(max_history),
            max_history,
            frame_start: None,
            total_frames: 0,
        }
    }

    /// Mark the start of a frame.
    pub fn begin_frame(&mut self) {
        self.frame_start = Some(Instant::now());
    }

    /// Mark the end of a frame, recording the duration.
    pub fn end_frame(&mut self) -> Option<Duration> {
        let start = self.frame_start.take()?;
        let duration = start.elapsed();
        if self.history.len() >= self.max_history {
            self.history.pop_front();
        }
        self.history.push_back(duration);
        self.total_frames += 1;
        Some(duration)
    }

    /// Return the latest frame duration.
    #[must_use]
    pub fn last_frame_time(&self) -> Option<Duration> {
        self.history.back().copied()
    }

    /// Return the average frame time over the history window.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn average_frame_time(&self) -> Option<Duration> {
        if self.history.is_empty() {
            return None;
        }
        let sum: Duration = self.history.iter().sum();
        Some(sum / self.history.len() as u32)
    }

    /// Return the current FPS based on average frame time.
    #[must_use]
    pub fn current_fps(&self) -> Option<f64> {
        self.average_frame_time().map(|avg| 1.0 / avg.as_secs_f64())
    }

    /// Return the total number of frames measured.
    #[must_use]
    pub fn total_frames(&self) -> u64 {
        self.total_frames
    }

    /// Return statistics over the history window.
    #[must_use]
    pub fn stats(&self) -> Option<TimingStats> {
        let durations: Vec<Duration> = self.history.iter().copied().collect();
        TimingStats::from_durations(&durations)
    }

    /// Clear the frame history.
    pub fn clear(&mut self) {
        self.history.clear();
        self.frame_start = None;
    }

    /// Return the number of samples in the history.
    #[must_use]
    pub fn history_len(&self) -> usize {
        self.history.len()
    }
}

/// High-level GPU timer that manages multiple named timing regions.
pub struct GpuTimer {
    /// Active timing regions.
    active_regions: Vec<TimerRegion>,
    /// History of timing samples organized by label.
    samples: VecDeque<TimingSample>,
    /// Frame timer for per-frame tracking.
    frame_timer: FrameTimer,
    /// Configuration.
    config: GpuTimerConfig,
    /// Current frame number.
    current_frame: u64,
}

impl GpuTimer {
    /// Create a new GPU timer with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(GpuTimerConfig::default())
    }

    /// Create a new GPU timer with the given configuration.
    #[must_use]
    pub fn with_config(config: GpuTimerConfig) -> Self {
        let max_history = config.max_history;
        Self {
            active_regions: Vec::new(),
            samples: VecDeque::with_capacity(max_history),
            frame_timer: FrameTimer::new(max_history),
            config,
            current_frame: 0,
        }
    }

    /// Begin a named timing region. Returns the index for stopping it later.
    pub fn begin_region(&mut self, label: &str) -> usize {
        if !self.config.enabled {
            return 0;
        }
        let region = TimerRegion::start(label);
        self.active_regions.push(region);
        self.active_regions.len() - 1
    }

    /// End a timing region by index, recording the sample.
    pub fn end_region(&mut self, index: usize) -> Option<Duration> {
        if !self.config.enabled || index >= self.active_regions.len() {
            return None;
        }
        self.active_regions[index].stop();
        let region = &self.active_regions[index];
        let duration = region.elapsed();
        let sample = TimingSample {
            label: region.label.clone(),
            duration,
            frame_number: self.current_frame,
        };
        if self.samples.len() >= self.config.max_history {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
        Some(duration)
    }

    /// Begin a new frame for the frame timer.
    pub fn begin_frame(&mut self) {
        self.current_frame += 1;
        self.frame_timer.begin_frame();
        self.active_regions.clear();
    }

    /// End the current frame.
    pub fn end_frame(&mut self) -> Option<Duration> {
        self.frame_timer.end_frame()
    }

    /// Get timing statistics for a specific label.
    #[must_use]
    pub fn stats_for_label(&self, label: &str) -> Option<TimingStats> {
        let durations: Vec<Duration> = self
            .samples
            .iter()
            .filter(|s| s.label == label)
            .map(|s| s.duration)
            .collect();
        TimingStats::from_durations(&durations)
    }

    /// Get the frame timer statistics.
    #[must_use]
    pub fn frame_stats(&self) -> Option<TimingStats> {
        self.frame_timer.stats()
    }

    /// Get the current FPS.
    #[must_use]
    pub fn current_fps(&self) -> Option<f64> {
        self.frame_timer.current_fps()
    }

    /// Check if the average frame time exceeds the target.
    #[must_use]
    pub fn is_over_budget(&self) -> bool {
        self.frame_timer
            .average_frame_time()
            .is_some_and(|avg| avg > self.config.target_frame_time)
    }

    /// Return all unique labels that have been recorded.
    #[must_use]
    pub fn labels(&self) -> Vec<String> {
        let mut labels: Vec<String> = self
            .samples
            .iter()
            .map(|s| s.label.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        labels.sort();
        labels
    }

    /// Return the total number of samples recorded.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Return the current frame number.
    #[must_use]
    pub fn current_frame_number(&self) -> u64 {
        self.current_frame
    }

    /// Check if timing collection is enabled.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Enable or disable timing collection.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.config.enabled = enabled;
    }

    /// Clear all samples and reset the timer.
    pub fn reset(&mut self) {
        self.active_regions.clear();
        self.samples.clear();
        self.frame_timer.clear();
        self.current_frame = 0;
    }
}

impl Default for GpuTimer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timer_region_start_stop() {
        let mut region = TimerRegion::start("test");
        assert!(!region.is_stopped());
        region.stop();
        assert!(region.is_stopped());
        assert!(region.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn test_timer_region_label() {
        let region = TimerRegion::start("my_region");
        assert_eq!(region.label, "my_region");
    }

    #[test]
    fn test_timing_stats_basic() {
        let durations = vec![
            Duration::from_micros(100),
            Duration::from_micros(200),
            Duration::from_micros(300),
            Duration::from_micros(400),
            Duration::from_micros(500),
        ];
        let stats = TimingStats::from_durations(&durations)
            .expect("from_durations should succeed with valid durations");
        assert_eq!(stats.min, Duration::from_micros(100));
        assert_eq!(stats.max, Duration::from_micros(500));
        assert_eq!(stats.sample_count, 5);
        assert_eq!(stats.median, Duration::from_micros(300));
    }

    #[test]
    fn test_timing_stats_empty() {
        let result = TimingStats::from_durations(&[]);
        assert!(result.is_none());
    }

    #[test]
    fn test_timing_stats_single() {
        let durations = vec![Duration::from_micros(1000)];
        let stats = TimingStats::from_durations(&durations)
            .expect("from_durations should succeed with valid durations");
        assert_eq!(stats.min, stats.max);
        assert_eq!(stats.sample_count, 1);
        assert!((stats.std_dev_us - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_timing_stats_mean_fps() {
        let durations = vec![Duration::from_millis(16), Duration::from_millis(17)];
        let stats = TimingStats::from_durations(&durations)
            .expect("from_durations should succeed with valid durations");
        let fps = stats.mean_fps();
        assert!(fps > 50.0 && fps < 70.0);
    }

    #[test]
    fn test_frame_timer_basic() {
        let mut timer = FrameTimer::new(100);
        timer.begin_frame();
        let dur = timer.end_frame();
        assert!(dur.is_some());
        assert_eq!(timer.total_frames(), 1);
    }

    #[test]
    fn test_frame_timer_history_limit() {
        let mut timer = FrameTimer::new(3);
        for _ in 0..5 {
            timer.begin_frame();
            timer.end_frame();
        }
        assert_eq!(timer.history_len(), 3);
        assert_eq!(timer.total_frames(), 5);
    }

    #[test]
    fn test_frame_timer_clear() {
        let mut timer = FrameTimer::new(100);
        timer.begin_frame();
        timer.end_frame();
        timer.clear();
        assert_eq!(timer.history_len(), 0);
        assert!(timer.last_frame_time().is_none());
    }

    #[test]
    fn test_frame_timer_no_begin() {
        let mut timer = FrameTimer::new(100);
        let dur = timer.end_frame();
        assert!(dur.is_none());
    }

    #[test]
    fn test_gpu_timer_create() {
        let timer = GpuTimer::new();
        assert!(timer.is_enabled());
        assert_eq!(timer.sample_count(), 0);
    }

    #[test]
    fn test_gpu_timer_region() {
        let mut timer = GpuTimer::new();
        let idx = timer.begin_region("vertex_shader");
        let dur = timer.end_region(idx);
        assert!(dur.is_some());
        assert_eq!(timer.sample_count(), 1);
    }

    #[test]
    fn test_gpu_timer_frame_cycle() {
        let mut timer = GpuTimer::new();
        timer.begin_frame();
        let _idx = timer.begin_region("pass1");
        timer.end_region(0);
        let frame_dur = timer.end_frame();
        assert!(frame_dur.is_some());
        assert_eq!(timer.current_frame_number(), 1);
    }

    #[test]
    fn test_gpu_timer_labels() {
        let mut timer = GpuTimer::new();
        let i1 = timer.begin_region("alpha");
        timer.end_region(i1);
        let i2 = timer.begin_region("beta");
        timer.end_region(i2);
        let labels = timer.labels();
        assert_eq!(labels.len(), 2);
        assert!(labels.contains(&"alpha".to_string()));
        assert!(labels.contains(&"beta".to_string()));
    }

    #[test]
    fn test_gpu_timer_disabled() {
        let config = GpuTimerConfig {
            enabled: false,
            ..Default::default()
        };
        let mut timer = GpuTimer::with_config(config);
        assert!(!timer.is_enabled());
        let idx = timer.begin_region("test");
        assert_eq!(idx, 0);
        let dur = timer.end_region(idx);
        assert!(dur.is_none());
    }

    #[test]
    fn test_gpu_timer_reset() {
        let mut timer = GpuTimer::new();
        timer.begin_frame();
        let idx = timer.begin_region("test");
        timer.end_region(idx);
        timer.end_frame();
        timer.reset();
        assert_eq!(timer.sample_count(), 0);
        assert_eq!(timer.current_frame_number(), 0);
    }

    #[test]
    fn test_gpu_timer_set_enabled() {
        let mut timer = GpuTimer::new();
        assert!(timer.is_enabled());
        timer.set_enabled(false);
        assert!(!timer.is_enabled());
    }

    #[test]
    fn test_gpu_timer_stats_for_label() {
        let mut timer = GpuTimer::new();
        for _ in 0..5 {
            let idx = timer.begin_region("compute");
            timer.end_region(idx);
        }
        let stats = timer.stats_for_label("compute");
        assert!(stats.is_some());
        assert_eq!(stats.expect("stats should be available").sample_count, 5);
    }

    #[test]
    fn test_gpu_timer_over_budget() {
        let config = GpuTimerConfig {
            target_frame_time: Duration::from_nanos(1), // impossibly small
            ..Default::default()
        };
        let mut timer = GpuTimer::with_config(config);
        timer.begin_frame();
        // Spin briefly
        let _x: u64 = (0..1000).sum();
        timer.end_frame();
        assert!(timer.is_over_budget());
    }
}
