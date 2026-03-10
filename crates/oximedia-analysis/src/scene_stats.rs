//! Scene duration statistics and reporting.
//!
//! Tracks per-scene durations (in milliseconds) and produces
//! summary reports with longest, shortest, and average durations.

#![allow(dead_code)]

/// Duration metadata for a single scene.
#[derive(Debug, Clone, Copy)]
pub struct SceneDuration {
    /// Scene index (0-based).
    pub index: usize,
    /// Duration of the scene in milliseconds.
    pub duration_ms: u64,
}

impl SceneDuration {
    /// Create a new `SceneDuration`.
    #[must_use]
    pub fn new(index: usize, duration_ms: u64) -> Self {
        Self { index, duration_ms }
    }

    /// Returns `true` if the scene is longer than `threshold_ms`.
    #[must_use]
    pub fn is_long(&self, threshold_ms: u64) -> bool {
        self.duration_ms > threshold_ms
    }

    /// Returns `true` if the scene is shorter than `threshold_ms`.
    #[must_use]
    pub fn is_short(&self, threshold_ms: u64) -> bool {
        self.duration_ms < threshold_ms
    }
}

/// Accumulator for scene duration statistics.
#[derive(Debug, Clone, Default)]
pub struct SceneStatistics {
    scenes: Vec<SceneDuration>,
}

impl SceneStatistics {
    /// Create an empty `SceneStatistics`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a scene duration record.
    pub fn add_scene(&mut self, scene: SceneDuration) {
        self.scenes.push(scene);
    }

    /// Return the total number of scenes recorded.
    #[must_use]
    pub fn scene_count(&self) -> usize {
        self.scenes.len()
    }

    /// Compute the average duration in milliseconds.
    ///
    /// Returns `None` if no scenes have been added.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_duration_ms(&self) -> Option<f64> {
        if self.scenes.is_empty() {
            return None;
        }
        let total: u64 = self.scenes.iter().map(|s| s.duration_ms).sum();
        Some(total as f64 / self.scenes.len() as f64)
    }

    /// Return the longest scene, or `None` if empty.
    #[must_use]
    pub fn longest(&self) -> Option<&SceneDuration> {
        self.scenes.iter().max_by_key(|s| s.duration_ms)
    }

    /// Return the shortest scene, or `None` if empty.
    #[must_use]
    pub fn shortest(&self) -> Option<&SceneDuration> {
        self.scenes.iter().min_by_key(|s| s.duration_ms)
    }

    /// Return all recorded scenes as a slice.
    #[must_use]
    pub fn scenes(&self) -> &[SceneDuration] {
        &self.scenes
    }

    /// Build a `SceneStatsReport` from the current state.
    #[must_use]
    pub fn build_report(&self) -> SceneStatsReport {
        SceneStatsReport {
            scene_count: self.scene_count(),
            avg_duration_ms: self.avg_duration_ms().unwrap_or(0.0),
            longest_ms: self.longest().map_or(0, |s| s.duration_ms),
            shortest_ms: self.shortest().map_or(0, |s| s.duration_ms),
        }
    }
}

/// A summary report of scene statistics.
#[derive(Debug, Clone, Copy)]
pub struct SceneStatsReport {
    /// Total number of scenes analysed.
    pub scene_count: usize,
    /// Average scene duration in milliseconds.
    pub avg_duration_ms: f64,
    /// Duration of the longest scene in milliseconds.
    pub longest_ms: u64,
    /// Duration of the shortest scene in milliseconds.
    pub shortest_ms: u64,
}

impl SceneStatsReport {
    /// Return the total number of scenes in this report.
    #[must_use]
    pub fn scene_count(&self) -> usize {
        self.scene_count
    }

    /// Return `true` if the average duration exceeds `threshold_ms`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_exceeds(&self, threshold_ms: u64) -> bool {
        self.avg_duration_ms > threshold_ms as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_stats(durations: &[u64]) -> SceneStatistics {
        let mut stats = SceneStatistics::new();
        for (i, &d) in durations.iter().enumerate() {
            stats.add_scene(SceneDuration::new(i, d));
        }
        stats
    }

    #[test]
    fn test_scene_duration_is_long() {
        let s = SceneDuration::new(0, 5000);
        assert!(s.is_long(4000));
        assert!(!s.is_long(6000));
    }

    #[test]
    fn test_scene_duration_is_short() {
        let s = SceneDuration::new(0, 500);
        assert!(s.is_short(1000));
        assert!(!s.is_short(400));
    }

    #[test]
    fn test_scene_stats_empty() {
        let stats = SceneStatistics::new();
        assert_eq!(stats.scene_count(), 0);
        assert!(stats.avg_duration_ms().is_none());
        assert!(stats.longest().is_none());
        assert!(stats.shortest().is_none());
    }

    #[test]
    fn test_scene_stats_single() {
        let stats = make_stats(&[3000]);
        assert_eq!(stats.scene_count(), 1);
        assert!((stats.avg_duration_ms().expect("unexpected None/Err") - 3000.0).abs() < 1e-9);
        assert_eq!(
            stats.longest().expect("unexpected None/Err").duration_ms,
            3000
        );
        assert_eq!(
            stats.shortest().expect("unexpected None/Err").duration_ms,
            3000
        );
    }

    #[test]
    fn test_scene_stats_avg() {
        let stats = make_stats(&[1000, 2000, 3000]);
        assert!((stats.avg_duration_ms().expect("unexpected None/Err") - 2000.0).abs() < 1e-9);
    }

    #[test]
    fn test_scene_stats_longest() {
        let stats = make_stats(&[500, 9000, 3000]);
        assert_eq!(
            stats.longest().expect("unexpected None/Err").duration_ms,
            9000
        );
    }

    #[test]
    fn test_scene_stats_shortest() {
        let stats = make_stats(&[500, 9000, 3000]);
        assert_eq!(
            stats.shortest().expect("unexpected None/Err").duration_ms,
            500
        );
    }

    #[test]
    fn test_scene_stats_scenes_slice() {
        let stats = make_stats(&[100, 200]);
        assert_eq!(stats.scenes().len(), 2);
    }

    #[test]
    fn test_build_report_scene_count() {
        let stats = make_stats(&[1000, 2000]);
        let report = stats.build_report();
        assert_eq!(report.scene_count(), 2);
    }

    #[test]
    fn test_build_report_avg() {
        let stats = make_stats(&[1000, 3000]);
        let report = stats.build_report();
        assert!((report.avg_duration_ms - 2000.0).abs() < 1e-9);
    }

    #[test]
    fn test_build_report_longest_shortest() {
        let stats = make_stats(&[100, 500, 250]);
        let report = stats.build_report();
        assert_eq!(report.longest_ms, 500);
        assert_eq!(report.shortest_ms, 100);
    }

    #[test]
    fn test_report_avg_exceeds() {
        let stats = make_stats(&[5000, 7000]);
        let report = stats.build_report();
        assert!(report.avg_exceeds(5000));
        assert!(!report.avg_exceeds(7000));
    }

    #[test]
    fn test_build_report_empty() {
        let stats = SceneStatistics::new();
        let report = stats.build_report();
        assert_eq!(report.scene_count(), 0);
        assert!(report.avg_duration_ms.abs() < f64::EPSILON);
    }
}
