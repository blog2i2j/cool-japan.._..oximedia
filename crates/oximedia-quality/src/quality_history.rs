//! Quality metric history with rolling windows, trend detection, anomaly flagging,
//! and per-scene statistics.
//!
//! # Overview
//!
//! [`QualityHistory`] maintains a bounded ring-buffer of (frame_index, score)
//! pairs for a single quality metric.  From this buffer it can:
//!
//! * Compute **rolling statistics** (mean, std-dev, min, max) over the most
//!   recent `N` frames.
//! * Detect **trends** (improving / degrading / flat) using a least-squares
//!   linear regression slope over the window.
//! * Flag **anomalies** — frames whose scores deviate from the rolling mean by
//!   more than a configurable `z_threshold` standard deviations.
//! * Accumulate **per-scene statistics** when the caller notifies the history
//!   of a scene cut.
//!
//! All public methods are allocation-light; the ring-buffer is pre-allocated
//! at construction time.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

// ── Types ─────────────────────────────────────────────────────────────────────

/// A single recorded quality observation.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct QualityObservation {
    /// Frame index (monotonically increasing; caller-managed).
    pub frame_idx: u64,
    /// Quality score value.
    pub score: f64,
}

/// Direction of a measured quality trend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendDirection {
    /// Score is increasing (quality improving).
    Improving,
    /// Score is stable within the flat-band threshold.
    Stable,
    /// Score is decreasing (quality degrading).
    Degrading,
}

/// Result of a trend analysis.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrendAnalysis {
    /// Linear regression slope (score units per frame).
    pub slope: f64,
    /// Intercept of the regression line at `frame_idx = 0`.
    pub intercept: f64,
    /// Coefficient of determination R².
    pub r_squared: f64,
    /// Human interpretation of the slope.
    pub direction: TrendDirection,
}

/// A detected anomaly — a frame whose score deviates significantly from the
/// rolling mean.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QualityAnomaly {
    /// Frame index of the anomalous observation.
    pub frame_idx: u64,
    /// Recorded score.
    pub score: f64,
    /// Rolling mean at the time of detection.
    pub mean: f64,
    /// Z-score of the observation.
    pub z_score: f64,
    /// `true` if the score is *below* the mean (quality drop).
    pub is_quality_drop: bool,
}

/// Rolling statistics over the current window.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RollingStats {
    /// Number of frames in the window.
    pub count: usize,
    /// Arithmetic mean.
    pub mean: f64,
    /// Standard deviation (population).
    pub std_dev: f64,
    /// Minimum score in the window.
    pub min: f64,
    /// Maximum score in the window.
    pub max: f64,
    /// Median (≈ 50th percentile).
    pub median: f64,
}

/// Statistics accumulated for a single scene.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SceneStats {
    /// Scene identifier (0-based, incremented on each `mark_scene_cut`).
    pub scene_id: u32,
    /// Index of the first frame in this scene.
    pub start_frame: u64,
    /// Index of the last frame recorded for this scene.
    pub end_frame: u64,
    /// Number of frames in this scene.
    pub frame_count: u32,
    /// Mean quality score for this scene.
    pub mean: f64,
    /// Minimum quality score for this scene.
    pub min: f64,
    /// Maximum quality score for this scene.
    pub max: f64,
    /// Standard deviation.
    pub std_dev: f64,
}

// ── History ───────────────────────────────────────────────────────────────────

/// Configuration for the quality history tracker.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QualityHistoryConfig {
    /// Maximum number of frames retained in the rolling window.
    pub window_size: usize,
    /// Z-score threshold above which a frame is flagged as an anomaly.
    pub z_threshold: f64,
    /// Absolute slope (score/frame) below which the trend is considered stable.
    pub flat_band: f64,
    /// Maximum number of anomalies stored (oldest are dropped first).
    pub max_anomalies: usize,
    /// Maximum number of completed scene records stored.
    pub max_scenes: usize,
}

impl Default for QualityHistoryConfig {
    fn default() -> Self {
        Self {
            window_size: 300,
            z_threshold: 3.0,
            flat_band: 0.05,
            max_anomalies: 100,
            max_scenes: 1000,
        }
    }
}

/// Rolling quality metric history tracker.
///
/// Maintains a fixed-capacity ring-buffer of observations and provides
/// real-time statistics, trend detection, and anomaly detection.
pub struct QualityHistory {
    config: QualityHistoryConfig,
    /// Ring buffer of recent observations (at most `window_size` entries).
    window: VecDeque<QualityObservation>,
    /// All detected anomalies (bounded by `max_anomalies`).
    anomalies: VecDeque<QualityAnomaly>,
    /// Completed scene statistics.
    scenes: VecDeque<SceneStats>,
    /// Accumulator for the currently open scene.
    current_scene: CurrentScene,
    /// Total frames ever pushed (regardless of window).
    total_frames: u64,
}

/// Internal accumulator for the currently open scene.
#[derive(Clone, Debug, Default)]
struct CurrentScene {
    id: u32,
    start_frame: u64,
    frame_count: u32,
    sum: f64,
    sum_sq: f64,
    min: f64,
    max: f64,
}

impl CurrentScene {
    fn new(id: u32, start_frame: u64) -> Self {
        Self {
            id,
            start_frame,
            frame_count: 0,
            sum: 0.0,
            sum_sq: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    fn push(&mut self, frame_idx: u64, score: f64) {
        let _ = frame_idx; // used only for start_frame tracking
        self.frame_count += 1;
        self.sum += score;
        self.sum_sq += score * score;
        if score < self.min {
            self.min = score;
        }
        if score > self.max {
            self.max = score;
        }
    }

    fn is_empty(&self) -> bool {
        self.frame_count == 0
    }

    fn finalise(&self, end_frame: u64) -> Option<SceneStats> {
        if self.frame_count == 0 {
            return None;
        }
        let n = self.frame_count as f64;
        let mean = self.sum / n;
        let variance = (self.sum_sq / n) - mean * mean;
        let std_dev = variance.max(0.0).sqrt();
        Some(SceneStats {
            scene_id: self.id,
            start_frame: self.start_frame,
            end_frame,
            frame_count: self.frame_count,
            mean,
            min: self.min,
            max: self.max,
            std_dev,
        })
    }
}

impl QualityHistory {
    /// Creates a history tracker with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(QualityHistoryConfig::default())
    }

    /// Creates a history tracker with a custom configuration.
    #[must_use]
    pub fn with_config(config: QualityHistoryConfig) -> Self {
        let cap = config.window_size;
        Self {
            window: VecDeque::with_capacity(cap),
            anomalies: VecDeque::with_capacity(config.max_anomalies.min(1000)),
            scenes: VecDeque::with_capacity(config.max_scenes.min(10_000)),
            current_scene: CurrentScene::new(0, 0),
            total_frames: 0,
            config,
        }
    }

    // ── Ingestion ──────────────────────────────────────────────────────────────

    /// Records a new quality observation.
    ///
    /// If the window is full the oldest observation is evicted.
    /// Anomaly detection runs after the observation is added.
    pub fn push(&mut self, frame_idx: u64, score: f64) {
        // Maintain rolling window.
        if self.window.len() >= self.config.window_size {
            self.window.pop_front();
        }
        self.window
            .push_back(QualityObservation { frame_idx, score });
        self.total_frames += 1;

        // Feed the current scene accumulator.
        self.current_scene.push(frame_idx, score);

        // Check for anomaly using current rolling stats.
        self.check_anomaly(frame_idx, score);
    }

    /// Signals a scene cut: finalises the current scene's statistics and begins
    /// a new one starting at `next_frame_idx`.
    pub fn mark_scene_cut(&mut self, next_frame_idx: u64) {
        if !self.current_scene.is_empty() {
            let last_frame = if self.current_scene.frame_count > 0 {
                next_frame_idx.saturating_sub(1)
            } else {
                next_frame_idx
            };
            if let Some(stats) = self.current_scene.finalise(last_frame) {
                if self.scenes.len() >= self.config.max_scenes {
                    self.scenes.pop_front();
                }
                self.scenes.push_back(stats);
            }
        }

        let next_id = self.current_scene.id + 1;
        self.current_scene = CurrentScene::new(next_id, next_frame_idx);
    }

    // ── Queries ────────────────────────────────────────────────────────────────

    /// Computes rolling statistics over the current window.
    #[must_use]
    pub fn rolling_stats(&self) -> RollingStats {
        let n = self.window.len();
        if n == 0 {
            return RollingStats::default();
        }

        let mut sum = 0.0_f64;
        let mut sum_sq = 0.0_f64;
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;

        for obs in &self.window {
            sum += obs.score;
            sum_sq += obs.score * obs.score;
            if obs.score < min {
                min = obs.score;
            }
            if obs.score > max {
                max = obs.score;
            }
        }

        let n_f = n as f64;
        let mean = sum / n_f;
        let variance = (sum_sq / n_f) - mean * mean;
        let std_dev = variance.max(0.0).sqrt();

        // Median (sort a copy).
        let mut sorted: Vec<f64> = self.window.iter().map(|o| o.score).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        };

        RollingStats {
            count: n,
            mean,
            std_dev,
            min,
            max,
            median,
        }
    }

    /// Analyses the trend of scores within the rolling window using
    /// ordinary least-squares linear regression.
    ///
    /// Returns `None` if fewer than two observations are in the window.
    #[must_use]
    pub fn trend(&self) -> Option<TrendAnalysis> {
        let n = self.window.len();
        if n < 2 {
            return None;
        }

        // Use sequential index (0, 1, 2, …) rather than raw frame_idx
        // to avoid large-number precision issues.
        let n_f = n as f64;
        let mut sum_x = 0.0_f64;
        let mut sum_y = 0.0_f64;
        let mut sum_xx = 0.0_f64;
        let mut sum_xy = 0.0_f64;
        let mut sum_yy = 0.0_f64;

        for (i, obs) in self.window.iter().enumerate() {
            let x = i as f64;
            let y = obs.score;
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_xy += x * y;
            sum_yy += y * y;
        }

        let denom = n_f * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-12 {
            return None;
        }

        let slope = (n_f * sum_xy - sum_x * sum_y) / denom;
        let intercept = (sum_y - slope * sum_x) / n_f;

        // R² = 1 - SS_res / SS_tot
        let ss_tot = sum_yy - sum_y * sum_y / n_f;
        let r_squared = if ss_tot.abs() < 1e-12 {
            1.0 // all y values equal → perfect "fit"
        } else {
            let ss_res: f64 = self
                .window
                .iter()
                .enumerate()
                .map(|(i, obs)| {
                    let predicted = slope * i as f64 + intercept;
                    let residual = obs.score - predicted;
                    residual * residual
                })
                .sum();
            (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)
        };

        let direction = if slope > self.config.flat_band {
            TrendDirection::Improving
        } else if slope < -self.config.flat_band {
            TrendDirection::Degrading
        } else {
            TrendDirection::Stable
        };

        Some(TrendAnalysis {
            slope,
            intercept,
            r_squared,
            direction,
        })
    }

    /// Returns a slice of all detected anomalies (oldest first).
    #[must_use]
    pub fn anomalies(&self) -> &VecDeque<QualityAnomaly> {
        &self.anomalies
    }

    /// Returns a slice of completed scene statistics (oldest first).
    #[must_use]
    pub fn scenes(&self) -> &VecDeque<SceneStats> {
        &self.scenes
    }

    /// Returns the number of frames ever pushed (not just the window).
    #[must_use]
    pub fn total_frames(&self) -> u64 {
        self.total_frames
    }

    /// Returns the most recent observation in the window, if any.
    #[must_use]
    pub fn latest(&self) -> Option<QualityObservation> {
        self.window.back().copied()
    }

    /// Clears all stored observations, anomalies and scene history.
    pub fn reset(&mut self) {
        self.window.clear();
        self.anomalies.clear();
        self.scenes.clear();
        self.current_scene = CurrentScene::new(0, 0);
        self.total_frames = 0;
    }

    // ── Private helpers ────────────────────────────────────────────────────────

    /// Checks whether the latest observation is an anomaly and stores it.
    fn check_anomaly(&mut self, frame_idx: u64, score: f64) {
        let n = self.window.len();
        if n < 2 {
            return;
        }

        // Compute mean and std-dev over the window (including the just-added obs).
        let sum: f64 = self.window.iter().map(|o| o.score).sum();
        let sum_sq: f64 = self.window.iter().map(|o| o.score * o.score).sum();
        let n_f = n as f64;
        let mean = sum / n_f;
        let variance = (sum_sq / n_f) - mean * mean;
        let std_dev = variance.max(0.0).sqrt();

        if std_dev < 1e-9 {
            // All values identical — nothing to flag.
            return;
        }

        let z = (score - mean) / std_dev;
        if z.abs() > self.config.z_threshold {
            if self.anomalies.len() >= self.config.max_anomalies {
                self.anomalies.pop_front();
            }
            self.anomalies.push_back(QualityAnomaly {
                frame_idx,
                score,
                mean,
                z_score: z,
                is_quality_drop: z < 0.0,
            });
        }
    }
}

impl Default for QualityHistory {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_history_returns_defaults() {
        let hist = QualityHistory::new();
        let stats = hist.rolling_stats();
        assert_eq!(stats.count, 0);
        assert_eq!(stats.mean, 0.0);
        assert!(hist.trend().is_none());
        assert!(hist.latest().is_none());
        assert_eq!(hist.total_frames(), 0);
    }

    #[test]
    fn test_rolling_stats_single_observation() {
        let mut hist = QualityHistory::new();
        hist.push(0, 42.0);
        let stats = hist.rolling_stats();
        assert_eq!(stats.count, 1);
        assert!((stats.mean - 42.0).abs() < 1e-9);
        assert!((stats.min - 42.0).abs() < 1e-9);
        assert!((stats.max - 42.0).abs() < 1e-9);
        assert!((stats.std_dev).abs() < 1e-9);
    }

    #[test]
    fn test_window_eviction() {
        let config = QualityHistoryConfig {
            window_size: 5,
            ..Default::default()
        };
        let mut hist = QualityHistory::with_config(config);
        for i in 0..10_u64 {
            hist.push(i, i as f64);
        }
        let stats = hist.rolling_stats();
        // Window should hold only the last 5 frames (5..=9 → values 5,6,7,8,9).
        assert_eq!(stats.count, 5);
        assert!((stats.mean - 7.0).abs() < 1e-9); // (5+6+7+8+9)/5 = 7
        assert_eq!(hist.total_frames(), 10);
    }

    #[test]
    fn test_trend_improving() {
        let mut hist = QualityHistory::new();
        for i in 0..20_u64 {
            hist.push(i, i as f64 * 0.5); // strictly increasing
        }
        let trend = hist
            .trend()
            .expect("trend should be Some with >=2 observations");
        assert_eq!(trend.direction, TrendDirection::Improving);
        assert!(trend.slope > 0.0);
    }

    #[test]
    fn test_trend_degrading() {
        let mut hist = QualityHistory::new();
        for i in 0..20_u64 {
            hist.push(i, 100.0 - i as f64 * 2.0); // strictly decreasing
        }
        let trend = hist.trend().expect("trend should be Some");
        assert_eq!(trend.direction, TrendDirection::Degrading);
        assert!(trend.slope < 0.0);
    }

    #[test]
    fn test_trend_stable() {
        let mut hist = QualityHistory::new();
        for i in 0..20_u64 {
            hist.push(i, 50.0); // constant
        }
        let trend = hist.trend().expect("trend should be Some");
        assert_eq!(trend.direction, TrendDirection::Stable);
        assert!(trend.slope.abs() < 1e-9);
    }

    #[test]
    fn test_anomaly_detection() {
        let config = QualityHistoryConfig {
            z_threshold: 2.5,
            ..Default::default()
        };
        let mut hist = QualityHistory::with_config(config);

        // Push a stable series of scores around 50.
        for i in 0..50_u64 {
            hist.push(i, 50.0);
        }
        // Push a very low score to trigger anomaly.
        hist.push(50, 0.0);

        let anomalies = hist.anomalies();
        assert!(
            !anomalies.is_empty(),
            "Expected at least one anomaly after score=0 in a stable-50 series"
        );
        let last = anomalies.back().expect("anomalies non-empty");
        assert!(last.is_quality_drop, "Expected quality drop");
        assert!((last.score - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_scene_cut_finalises_stats() {
        let mut hist = QualityHistory::new();

        // Scene 0: frames 0..5 with score 80.0
        for i in 0..5_u64 {
            hist.push(i, 80.0);
        }
        hist.mark_scene_cut(5);

        // Scene 1: frames 5..10 with score 60.0
        for i in 5..10_u64 {
            hist.push(i, 60.0);
        }
        hist.mark_scene_cut(10);

        let scenes = hist.scenes();
        assert_eq!(scenes.len(), 2, "Two scenes should be finalised");
        assert!((scenes[0].mean - 80.0).abs() < 1e-6, "Scene 0 mean");
        assert!((scenes[1].mean - 60.0).abs() < 1e-6, "Scene 1 mean");
    }

    #[test]
    fn test_reset_clears_state() {
        let mut hist = QualityHistory::new();
        for i in 0..20_u64 {
            hist.push(i, 50.0 + i as f64);
        }
        hist.mark_scene_cut(20);
        hist.reset();

        assert_eq!(hist.total_frames(), 0);
        assert_eq!(hist.rolling_stats().count, 0);
        assert!(hist.anomalies().is_empty());
        assert!(hist.scenes().is_empty());
    }

    #[test]
    fn test_latest_observation() {
        let mut hist = QualityHistory::new();
        hist.push(10, 77.5);
        hist.push(11, 82.0);
        let latest = hist.latest().expect("latest should be Some");
        assert_eq!(latest.frame_idx, 11);
        assert!((latest.score - 82.0).abs() < 1e-9);
    }

    #[test]
    fn test_r_squared_perfect_fit() {
        let mut hist = QualityHistory::new();
        for i in 0..10_u64 {
            hist.push(i, i as f64 * 3.0 + 5.0); // exact linear
        }
        let trend = hist.trend().expect("trend should be Some");
        assert!(
            trend.r_squared > 0.999,
            "R² for exact linear data should be ~1.0, got {}",
            trend.r_squared
        );
    }
}
