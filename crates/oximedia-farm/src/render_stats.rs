#![allow(dead_code)]
//! Render statistics collection and aggregation for the encoding farm.
//!
//! Tracks per-job and per-worker encoding metrics such as throughput
//! (frames/second), bitrate, quality scores, and wall-clock durations.
//! These statistics feed into the coordinator dashboard and are used by
//! the scheduler for capacity-planning decisions.

use std::collections::HashMap;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Per-task snapshot
// ---------------------------------------------------------------------------

/// Statistics for a single render task.
#[derive(Debug, Clone)]
pub struct TaskStats {
    /// Unique task identifier.
    pub task_id: String,
    /// Total frames rendered.
    pub frames_rendered: u64,
    /// Total bytes written to output.
    pub bytes_written: u64,
    /// Wall-clock elapsed time.
    pub wall_time: Duration,
    /// Average encoding speed in frames per second.
    pub fps: f64,
    /// Average output bitrate in kbps.
    pub bitrate_kbps: f64,
    /// Optional quality metric (e.g. VMAF).
    pub quality_score: Option<f64>,
}

/// Aggregated statistics across multiple tasks.
#[derive(Debug, Clone, Default)]
pub struct AggregateStats {
    /// Number of tasks included.
    pub task_count: u64,
    /// Total frames across all tasks.
    pub total_frames: u64,
    /// Total bytes across all tasks.
    pub total_bytes: u64,
    /// Total wall-clock time.
    pub total_wall_time: Duration,
    /// Mean fps.
    pub mean_fps: f64,
    /// Mean bitrate kbps.
    pub mean_bitrate_kbps: f64,
    /// Mean quality score (only over tasks that reported one).
    pub mean_quality: Option<f64>,
}

// ---------------------------------------------------------------------------
// Live tracker
// ---------------------------------------------------------------------------

/// A live tracker that accumulates task statistics during a render session.
#[derive(Debug)]
pub struct RenderStatsTracker {
    /// When the tracker was created.
    start: Instant,
    /// Per-task records.
    records: Vec<TaskStats>,
    /// Per-worker accumulated frame counts.
    worker_frames: HashMap<String, u64>,
}

impl RenderStatsTracker {
    /// Create a new tracker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
            records: Vec::new(),
            worker_frames: HashMap::new(),
        }
    }

    /// Record a completed task.
    pub fn record_task(&mut self, stats: TaskStats) {
        self.records.push(stats);
    }

    /// Record a completed task and attribute its frames to a worker.
    pub fn record_task_for_worker(&mut self, worker_id: &str, stats: TaskStats) {
        *self.worker_frames.entry(worker_id.to_string()).or_insert(0) += stats.frames_rendered;
        self.records.push(stats);
    }

    /// Total tasks recorded so far.
    #[must_use]
    pub fn task_count(&self) -> usize {
        self.records.len()
    }

    /// Elapsed time since the tracker was created.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Total frames across all recorded tasks.
    #[must_use]
    pub fn total_frames(&self) -> u64 {
        self.records.iter().map(|r| r.frames_rendered).sum()
    }

    /// Total bytes written across all recorded tasks.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.records.iter().map(|r| r.bytes_written).sum()
    }

    /// Compute aggregate statistics.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn aggregate(&self) -> AggregateStats {
        if self.records.is_empty() {
            return AggregateStats::default();
        }
        let n = self.records.len() as f64;
        let total_frames: u64 = self.records.iter().map(|r| r.frames_rendered).sum();
        let total_bytes: u64 = self.records.iter().map(|r| r.bytes_written).sum();
        let total_wall: Duration = self.records.iter().map(|r| r.wall_time).sum();
        let mean_fps: f64 = self.records.iter().map(|r| r.fps).sum::<f64>() / n;
        let mean_br: f64 = self.records.iter().map(|r| r.bitrate_kbps).sum::<f64>() / n;
        let quality_records: Vec<f64> = self
            .records
            .iter()
            .filter_map(|r| r.quality_score)
            .collect();
        let mean_quality = if quality_records.is_empty() {
            None
        } else {
            Some(quality_records.iter().sum::<f64>() / quality_records.len() as f64)
        };

        AggregateStats {
            task_count: self.records.len() as u64,
            total_frames,
            total_bytes,
            total_wall_time: total_wall,
            mean_fps,
            mean_bitrate_kbps: mean_br,
            mean_quality,
        }
    }

    /// Return per-worker frame counts.
    #[must_use]
    pub fn worker_frame_counts(&self) -> &HashMap<String, u64> {
        &self.worker_frames
    }

    /// Clear all records.
    pub fn reset(&mut self) {
        self.records.clear();
        self.worker_frames.clear();
        self.start = Instant::now();
    }
}

impl Default for RenderStatsTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Throughput calculator
// ---------------------------------------------------------------------------

/// Calculate encoding throughput in frames per second.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn compute_fps(frames: u64, duration: Duration) -> f64 {
    let secs = duration.as_secs_f64();
    if secs <= 0.0 {
        return 0.0;
    }
    frames as f64 / secs
}

/// Calculate average bitrate in kbps.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn compute_bitrate_kbps(bytes: u64, duration: Duration) -> f64 {
    let secs = duration.as_secs_f64();
    if secs <= 0.0 {
        return 0.0;
    }
    (bytes as f64 * 8.0) / (secs * 1000.0)
}

/// Estimate remaining time given current progress.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn estimate_remaining(
    completed_frames: u64,
    total_frames: u64,
    elapsed: Duration,
) -> Option<Duration> {
    if completed_frames == 0 || total_frames == 0 {
        return None;
    }
    if completed_frames >= total_frames {
        return Some(Duration::ZERO);
    }
    let secs = elapsed.as_secs_f64();
    let fps = completed_frames as f64 / secs;
    if fps <= 0.0 {
        return None;
    }
    let remaining_frames = total_frames - completed_frames;
    let remaining_secs = remaining_frames as f64 / fps;
    Some(Duration::from_secs_f64(remaining_secs))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_stats(id: &str, frames: u64, bytes: u64, secs: u64) -> TaskStats {
        let wall = Duration::from_secs(secs);
        #[allow(clippy::cast_precision_loss)]
        let fps = if secs > 0 {
            frames as f64 / secs as f64
        } else {
            0.0
        };
        #[allow(clippy::cast_precision_loss)]
        let br = if secs > 0 {
            (bytes as f64 * 8.0) / (secs as f64 * 1000.0)
        } else {
            0.0
        };
        TaskStats {
            task_id: id.to_string(),
            frames_rendered: frames,
            bytes_written: bytes,
            wall_time: wall,
            fps,
            bitrate_kbps: br,
            quality_score: None,
        }
    }

    #[test]
    fn test_tracker_new() {
        let t = RenderStatsTracker::new();
        assert_eq!(t.task_count(), 0);
        assert_eq!(t.total_frames(), 0);
    }

    #[test]
    fn test_record_task() {
        let mut t = RenderStatsTracker::new();
        t.record_task(make_stats("t1", 100, 5000, 10));
        assert_eq!(t.task_count(), 1);
        assert_eq!(t.total_frames(), 100);
    }

    #[test]
    fn test_record_for_worker() {
        let mut t = RenderStatsTracker::new();
        t.record_task_for_worker("w1", make_stats("t1", 200, 8000, 20));
        t.record_task_for_worker("w1", make_stats("t2", 300, 12000, 30));
        t.record_task_for_worker("w2", make_stats("t3", 100, 4000, 10));
        assert_eq!(*t.worker_frame_counts().get("w1").unwrap(), 500);
        assert_eq!(*t.worker_frame_counts().get("w2").unwrap(), 100);
    }

    #[test]
    fn test_aggregate_empty() {
        let t = RenderStatsTracker::new();
        let agg = t.aggregate();
        assert_eq!(agg.task_count, 0);
        assert_eq!(agg.total_frames, 0);
    }

    #[test]
    fn test_aggregate_multiple() {
        let mut t = RenderStatsTracker::new();
        t.record_task(make_stats("t1", 100, 5000, 10));
        t.record_task(make_stats("t2", 200, 10000, 20));
        let agg = t.aggregate();
        assert_eq!(agg.task_count, 2);
        assert_eq!(agg.total_frames, 300);
        assert_eq!(agg.total_bytes, 15000);
    }

    #[test]
    fn test_aggregate_quality() {
        let mut t = RenderStatsTracker::new();
        let mut s1 = make_stats("t1", 100, 5000, 10);
        s1.quality_score = Some(90.0);
        let mut s2 = make_stats("t2", 100, 5000, 10);
        s2.quality_score = Some(80.0);
        t.record_task(s1);
        t.record_task(s2);
        let agg = t.aggregate();
        assert!((agg.mean_quality.unwrap() - 85.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_reset() {
        let mut t = RenderStatsTracker::new();
        t.record_task(make_stats("t1", 100, 5000, 10));
        t.reset();
        assert_eq!(t.task_count(), 0);
        assert!(t.worker_frame_counts().is_empty());
    }

    #[test]
    fn test_compute_fps() {
        let fps = compute_fps(300, Duration::from_secs(10));
        assert!((fps - 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compute_fps_zero_duration() {
        let fps = compute_fps(100, Duration::ZERO);
        assert!((fps - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compute_bitrate_kbps() {
        // 125_000 bytes in 1 second = 1_000_000 bits/s = 1000 kbps
        let br = compute_bitrate_kbps(125_000, Duration::from_secs(1));
        assert!((br - 1000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_estimate_remaining() {
        let est = estimate_remaining(50, 100, Duration::from_secs(10));
        let remaining = est.unwrap();
        // 50 frames in 10s => 5fps => 50 remaining => 10s
        assert!((remaining.as_secs_f64() - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_estimate_remaining_complete() {
        let est = estimate_remaining(100, 100, Duration::from_secs(10));
        assert_eq!(est.unwrap(), Duration::ZERO);
    }

    #[test]
    fn test_estimate_remaining_zero() {
        assert!(estimate_remaining(0, 100, Duration::from_secs(10)).is_none());
    }
}
