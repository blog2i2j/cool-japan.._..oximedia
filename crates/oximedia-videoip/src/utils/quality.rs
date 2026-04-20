//! Stream quality monitoring and assessment.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Quality metrics for video streams.
#[derive(Debug, Clone, Default)]
pub struct QualityMetrics {
    /// Video bitrate in bits per second.
    pub video_bitrate: u64,
    /// Audio bitrate in bits per second.
    pub audio_bitrate: u64,
    /// Actual frame rate (frames per second).
    pub actual_fps: f64,
    /// Expected frame rate.
    pub expected_fps: f64,
    /// Frame drop rate (0.0-1.0).
    pub frame_drop_rate: f64,
    /// Average frame encoding time in microseconds.
    pub avg_encode_time_us: u64,
    /// Peak-to-peak jitter in microseconds.
    pub jitter_us: u64,
    /// Keyframe interval (frames).
    pub keyframe_interval: u32,
    /// Overall quality score (0.0-1.0).
    pub quality_score: f64,
}

impl QualityMetrics {
    /// Returns true if the quality is acceptable.
    #[must_use]
    pub fn is_acceptable(&self) -> bool {
        self.quality_score >= 0.7
            && self.frame_drop_rate < 0.05
            && (self.actual_fps - self.expected_fps).abs() < 1.0
    }

    /// Returns a quality grade.
    #[must_use]
    pub fn quality_grade(&self) -> QualityGrade {
        if self.quality_score >= 0.9 {
            QualityGrade::Excellent
        } else if self.quality_score >= 0.7 {
            QualityGrade::Good
        } else if self.quality_score >= 0.5 {
            QualityGrade::Fair
        } else {
            QualityGrade::Poor
        }
    }
}

/// Quality grade classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityGrade {
    /// Excellent quality (score >= 0.9).
    Excellent,
    /// Good quality (score >= 0.7).
    Good,
    /// Fair quality (score >= 0.5).
    Fair,
    /// Poor quality (score < 0.5).
    Poor,
}

/// Quality monitor for tracking stream quality over time.
pub struct QualityMonitor {
    /// Frame timestamps for FPS calculation.
    frame_times: VecDeque<Instant>,
    /// Frame drops.
    frames_dropped: u64,
    /// Frames received.
    frames_received: u64,
    /// Expected frame rate.
    expected_fps: f64,
    /// Window size for calculations.
    window_size: Duration,
    /// Encoding times for averaging.
    encode_times: VecDeque<u64>,
    /// Last keyframe number.
    last_keyframe: u64,
    /// Current frame number.
    current_frame: u64,
    /// Jitter samples.
    jitter_samples: VecDeque<u64>,
}

impl QualityMonitor {
    /// Creates a new quality monitor.
    #[must_use]
    pub fn new(expected_fps: f64) -> Self {
        Self {
            frame_times: VecDeque::new(),
            frames_dropped: 0,
            frames_received: 0,
            expected_fps,
            window_size: Duration::from_secs(1),
            encode_times: VecDeque::new(),
            last_keyframe: 0,
            current_frame: 0,
            jitter_samples: VecDeque::new(),
        }
    }

    /// Records a received frame.
    pub fn record_frame(&mut self, is_keyframe: bool, encode_time_us: u64) {
        let now = Instant::now();
        self.frame_times.push_back(now);
        self.frames_received += 1;
        self.current_frame += 1;

        if is_keyframe {
            self.last_keyframe = self.current_frame;
        }

        self.encode_times.push_back(encode_time_us);

        // Remove old samples outside window
        let cutoff = now.checked_sub(self.window_size).unwrap_or(now);
        while let Some(&time) = self.frame_times.front() {
            if time < cutoff {
                self.frame_times.pop_front();
            } else {
                break;
            }
        }

        // Limit encode time samples
        while self.encode_times.len() > 100 {
            self.encode_times.pop_front();
        }
    }

    /// Records a dropped frame.
    pub fn record_drop(&mut self) {
        self.frames_dropped += 1;
    }

    /// Records jitter measurement.
    pub fn record_jitter(&mut self, jitter_us: u64) {
        self.jitter_samples.push_back(jitter_us);

        while self.jitter_samples.len() > 100 {
            self.jitter_samples.pop_front();
        }
    }

    /// Calculates current quality metrics.
    #[must_use]
    pub fn calculate_metrics(&self) -> QualityMetrics {
        let actual_fps = if self.frame_times.len() >= 2 {
            let (Some(first), Some(last)) = (self.frame_times.front(), self.frame_times.back())
            else {
                return QualityMetrics::default();
            };
            let duration = last.duration_since(*first).as_secs_f64();

            if duration > 0.0 {
                (self.frame_times.len() - 1) as f64 / duration
            } else {
                0.0
            }
        } else {
            0.0
        };

        let frame_drop_rate = if self.frames_received + self.frames_dropped > 0 {
            self.frames_dropped as f64 / (self.frames_received + self.frames_dropped) as f64
        } else {
            0.0
        };

        let avg_encode_time_us = if self.encode_times.is_empty() {
            0
        } else {
            self.encode_times.iter().sum::<u64>() / self.encode_times.len() as u64
        };

        let jitter_us = if self.jitter_samples.is_empty() {
            0
        } else {
            let max_jitter = *self.jitter_samples.iter().max().unwrap_or(&0);
            let min_jitter = *self.jitter_samples.iter().min().unwrap_or(&0);
            max_jitter - min_jitter
        };

        let keyframe_interval = (self.current_frame - self.last_keyframe) as u32;

        // Calculate quality score
        let fps_score = 1.0 - ((actual_fps - self.expected_fps).abs() / self.expected_fps).min(1.0);
        let drop_score = 1.0 - frame_drop_rate;
        let jitter_score = (1.0 - (jitter_us as f64 / 50_000.0)).max(0.0);

        let quality_score = (fps_score + drop_score + jitter_score) / 3.0;

        QualityMetrics {
            video_bitrate: 0, // Set externally
            audio_bitrate: 0, // Set externally
            actual_fps,
            expected_fps: self.expected_fps,
            frame_drop_rate,
            avg_encode_time_us,
            jitter_us,
            keyframe_interval,
            quality_score,
        }
    }

    /// Resets all counters.
    pub fn reset(&mut self) {
        self.frame_times.clear();
        self.frames_dropped = 0;
        self.frames_received = 0;
        self.encode_times.clear();
        self.last_keyframe = 0;
        self.current_frame = 0;
        self.jitter_samples.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_metrics_default() {
        let metrics = QualityMetrics::default();
        assert_eq!(metrics.video_bitrate, 0);
        assert_eq!(metrics.quality_score, 0.0);
    }

    #[test]
    fn test_quality_grade() {
        let mut metrics = QualityMetrics::default();

        metrics.quality_score = 0.95;
        assert_eq!(metrics.quality_grade(), QualityGrade::Excellent);

        metrics.quality_score = 0.75;
        assert_eq!(metrics.quality_grade(), QualityGrade::Good);

        metrics.quality_score = 0.55;
        assert_eq!(metrics.quality_grade(), QualityGrade::Fair);

        metrics.quality_score = 0.3;
        assert_eq!(metrics.quality_grade(), QualityGrade::Poor);
    }

    #[test]
    fn test_quality_monitor() {
        let mut monitor = QualityMonitor::new(30.0);

        // Record some frames
        for i in 0..30 {
            monitor.record_frame(i % 30 == 0, 1000);
            std::thread::sleep(std::time::Duration::from_millis(33));
        }

        let metrics = monitor.calculate_metrics();

        assert!(metrics.actual_fps > 0.0);
        assert_eq!(metrics.expected_fps, 30.0);
        assert!(metrics.frame_drop_rate >= 0.0);
    }

    #[test]
    fn test_frame_drop_rate() {
        let mut monitor = QualityMonitor::new(30.0);

        monitor.record_frame(true, 1000);
        monitor.record_frame(false, 1000);
        monitor.record_drop();

        let metrics = monitor.calculate_metrics();

        assert!((metrics.frame_drop_rate - (1.0 / 3.0)).abs() < 0.01);
    }

    #[test]
    fn test_quality_acceptable() {
        let mut metrics = QualityMetrics::default();

        metrics.quality_score = 0.8;
        metrics.frame_drop_rate = 0.02;
        metrics.actual_fps = 29.5;
        metrics.expected_fps = 30.0;

        assert!(metrics.is_acceptable());

        metrics.frame_drop_rate = 0.1;
        assert!(!metrics.is_acceptable());
    }

    #[test]
    fn test_jitter_calculation() {
        let mut monitor = QualityMonitor::new(30.0);

        monitor.record_jitter(1000);
        monitor.record_jitter(5000);
        monitor.record_jitter(3000);

        let metrics = monitor.calculate_metrics();

        assert_eq!(metrics.jitter_us, 4000); // max - min = 5000 - 1000
    }

    #[test]
    fn test_monitor_reset() {
        let mut monitor = QualityMonitor::new(30.0);

        monitor.record_frame(true, 1000);
        monitor.record_drop();

        monitor.reset();

        assert_eq!(monitor.frames_received, 0);
        assert_eq!(monitor.frames_dropped, 0);
    }
}
