//! RPU sequence statistics — per-scene L1 luminance stats, histogram of MaxPQ
//! values, and percentile computation.
//!
//! This module aggregates frame-level L1 metadata across an RPU sequence to
//! produce useful statistical summaries for content analysis, QC, and display
//! management diagnostics.
//!
//! # Examples
//!
//! ```rust
//! use oximedia_dolbyvision::rpu_statistics::{RpuStatistics, SceneStats, PqHistogram};
//! use oximedia_dolbyvision::dv_xml_export::{DvShotEntry, DvL2Entry};
//!
//! let shots = vec![
//!     DvShotEntry {
//!         frame_start: 0,
//!         frame_end: 23,
//!         l1_min: 0.001,
//!         l1_mid: 0.10,
//!         l1_max: 0.45,
//!         l2_entries: vec![DvL2Entry::identity(2081)],
//!     },
//!     DvShotEntry {
//!         frame_start: 24,
//!         frame_end: 47,
//!         l1_min: 0.0,
//!         l1_mid: 0.20,
//!         l1_max: 0.70,
//!         l2_entries: vec![DvL2Entry::identity(2081)],
//!     },
//! ];
//!
//! let stats = RpuStatistics::from_shots(&shots);
//! assert_eq!(stats.scene_stats.len(), 2);
//! assert!((stats.overall.avg_max_pq_f32 - 0.575).abs() < 0.001);
//! ```

use crate::dv_xml_export::DvShotEntry;

// ── PQ Histogram ──────────────────────────────────────────────────────────────

/// Fine-grained histogram over normalised PQ values in the range [0.0, 1.0].
///
/// The histogram uses `BUCKET_COUNT` uniform bins spanning [0, 1].
#[derive(Debug, Clone)]
pub struct PqHistogram {
    /// Bucket counts; index `i` covers PQ range `[i/N, (i+1)/N)`.
    pub buckets: Vec<u64>,
    /// Total number of samples added to this histogram.
    pub total_samples: u64,
}

/// Number of buckets in a [`PqHistogram`].
pub const BUCKET_COUNT: usize = 1024;

impl PqHistogram {
    /// Create a new empty histogram with [`BUCKET_COUNT`] buckets.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buckets: vec![0u64; BUCKET_COUNT],
            total_samples: 0,
        }
    }

    /// Add a normalised PQ sample (`0.0..=1.0`) to the histogram.
    ///
    /// Values outside [0, 1] are clamped to the nearest bucket edge.
    pub fn add_sample(&mut self, pq: f32) {
        let clamped = pq.clamp(0.0, 1.0);
        let idx = ((clamped * BUCKET_COUNT as f32) as usize).min(BUCKET_COUNT - 1);
        self.buckets[idx] = self.buckets[idx].saturating_add(1);
        self.total_samples = self.total_samples.saturating_add(1);
    }

    /// Compute the PQ value at the given percentile (0.0–100.0).
    ///
    /// Returns `0.0` if the histogram is empty.
    #[must_use]
    pub fn percentile(&self, p: f32) -> f32 {
        if self.total_samples == 0 {
            return 0.0;
        }
        let target = ((f64::from(p) / 100.0) * self.total_samples as f64).ceil() as u64;
        let mut cumulative: u64 = 0;
        for (i, &count) in self.buckets.iter().enumerate() {
            cumulative += count;
            if cumulative >= target {
                // Return the midpoint of bucket i
                return (i as f32 + 0.5) / BUCKET_COUNT as f32;
            }
        }
        1.0
    }

    /// Merge another histogram into this one.
    pub fn merge(&mut self, other: &PqHistogram) {
        debug_assert_eq!(self.buckets.len(), other.buckets.len());
        for (a, b) in self.buckets.iter_mut().zip(other.buckets.iter()) {
            *a = a.saturating_add(*b);
        }
        self.total_samples = self.total_samples.saturating_add(other.total_samples);
    }

    /// Return the bucket index with the highest count (mode bin).
    ///
    /// Returns `None` if the histogram is empty.
    #[must_use]
    pub fn mode_bucket(&self) -> Option<usize> {
        if self.total_samples == 0 {
            return None;
        }
        self.buckets
            .iter()
            .enumerate()
            .max_by_key(|&(_, &count)| count)
            .map(|(i, _)| i)
    }

    /// Return the normalised PQ value at the centre of the mode bucket.
    #[must_use]
    pub fn mode_pq(&self) -> Option<f32> {
        self.mode_bucket()
            .map(|i| (i as f32 + 0.5) / BUCKET_COUNT as f32)
    }
}

impl Default for PqHistogram {
    fn default() -> Self {
        Self::new()
    }
}

// ── Per-scene statistics ──────────────────────────────────────────────────────

/// Per-scene (per-shot) L1 luminance statistics.
#[derive(Debug, Clone)]
pub struct SceneStats {
    /// Absolute first frame index of this scene in the sequence.
    pub frame_start: u64,
    /// Absolute last frame index (inclusive) of this scene.
    pub frame_end: u64,
    /// Minimum L1 min across all frames in the scene.
    pub l1_min_min: f32,
    /// Maximum L1 max across all frames in the scene.
    pub l1_max_max: f32,
    /// Average of L1 max values across frames, weighted by frame count.
    pub avg_max_pq: f32,
    /// Average of L1 mid values across frames, weighted by frame count.
    pub avg_mid_pq: f32,
    /// Average of L1 min values across frames, weighted by frame count.
    pub avg_min_pq: f32,
    /// Percentile 10 of L1 max distribution within the scene.
    pub p10_max_pq: f32,
    /// Percentile 90 of L1 max distribution within the scene.
    pub p90_max_pq: f32,
    /// Percentile 99 of L1 max distribution within the scene.
    pub p99_max_pq: f32,
    /// Frame count for this scene.
    pub frame_count: u64,
}

impl SceneStats {
    /// Compute per-scene stats from a [`DvShotEntry`].
    ///
    /// The shot's L1 values are treated as representative for the entire shot
    /// duration (each frame within the shot shares the same metadata).
    #[must_use]
    pub fn from_shot(shot: &DvShotEntry) -> Self {
        let frame_count = shot.duration();

        // Build a local histogram by repeating the shot's L1 max for each frame
        let mut hist = PqHistogram::new();
        for _ in 0..frame_count {
            hist.add_sample(shot.l1_max);
        }

        Self {
            frame_start: shot.frame_start,
            frame_end: shot.frame_end,
            l1_min_min: shot.l1_min,
            l1_max_max: shot.l1_max,
            avg_max_pq: shot.l1_max,
            avg_mid_pq: shot.l1_mid,
            avg_min_pq: shot.l1_min,
            p10_max_pq: hist.percentile(10.0),
            p90_max_pq: hist.percentile(90.0),
            p99_max_pq: hist.percentile(99.0),
            frame_count,
        }
    }

    /// Check whether this scene qualifies as high-brightness (L1 max > 0.5).
    #[must_use]
    pub fn is_high_brightness(&self) -> bool {
        self.l1_max_max > 0.5
    }

    /// Check whether this scene qualifies as low-brightness (L1 max < 0.1).
    #[must_use]
    pub fn is_dark(&self) -> bool {
        self.l1_max_max < 0.1
    }

    /// Dynamic range of the scene: L1 max_max − L1 min_min.
    #[must_use]
    pub fn dynamic_range(&self) -> f32 {
        (self.l1_max_max - self.l1_min_min).max(0.0)
    }
}

// ── Sequence-level summary ────────────────────────────────────────────────────

/// Aggregate luminance statistics across the full RPU sequence.
#[derive(Debug, Clone)]
pub struct SequenceSummary {
    /// Minimum of all L1 min values across the entire sequence.
    pub global_min_pq: f32,
    /// Maximum of all L1 max values across the entire sequence.
    pub global_max_pq: f32,
    /// Weighted average of per-scene L1 max values (weight = frame count).
    pub avg_max_pq_f32: f32,
    /// Percentile 1 of global L1 max histogram.
    pub p01_max_pq: f32,
    /// Percentile 10 of global L1 max histogram.
    pub p10_max_pq: f32,
    /// Percentile 50 (median) of global L1 max histogram.
    pub p50_max_pq: f32,
    /// Percentile 90 of global L1 max histogram.
    pub p90_max_pq: f32,
    /// Percentile 99 of global L1 max histogram.
    pub p99_max_pq: f32,
    /// Total number of frames across all scenes.
    pub total_frames: u64,
    /// Total number of scenes (shots).
    pub total_scenes: usize,
    /// Number of scenes classified as high-brightness (L1 max > 0.5).
    pub high_brightness_scenes: usize,
    /// Number of scenes classified as dark (L1 max < 0.1).
    pub dark_scenes: usize,
}

// ── Main statistics aggregator ────────────────────────────────────────────────

/// Aggregates RPU L1 luminance statistics over a full shot sequence.
///
/// Use [`RpuStatistics::from_shots`] to build the stats from a slice of
/// [`DvShotEntry`] values.
#[derive(Debug, Clone)]
pub struct RpuStatistics {
    /// Per-scene statistics, in presentation order.
    pub scene_stats: Vec<SceneStats>,
    /// Sequence-level summary across all scenes.
    pub overall: SequenceSummary,
    /// Global L1 max histogram across all frames.
    pub global_histogram: PqHistogram,
}

impl RpuStatistics {
    /// Build statistics from a slice of [`DvShotEntry`] values.
    ///
    /// If `shots` is empty, the resulting statistics will reflect zero frames.
    #[must_use]
    pub fn from_shots(shots: &[DvShotEntry]) -> Self {
        let mut scene_stats = Vec::with_capacity(shots.len());
        let mut global_hist = PqHistogram::new();
        let mut global_min: f32 = f32::MAX;
        let mut global_max: f32 = f32::MIN;
        let mut weighted_sum: f64 = 0.0;
        let mut total_frames: u64 = 0;
        let mut high_brightness_scenes = 0usize;
        let mut dark_scenes = 0usize;

        for shot in shots {
            let ss = SceneStats::from_shot(shot);

            // Accumulate global stats
            if ss.l1_min_min < global_min {
                global_min = ss.l1_min_min;
            }
            if ss.l1_max_max > global_max {
                global_max = ss.l1_max_max;
            }
            weighted_sum += f64::from(ss.avg_max_pq) * ss.frame_count as f64;
            total_frames += ss.frame_count;

            // Populate global histogram
            for _ in 0..ss.frame_count {
                global_hist.add_sample(ss.avg_max_pq);
            }

            if ss.is_high_brightness() {
                high_brightness_scenes += 1;
            }
            if ss.is_dark() {
                dark_scenes += 1;
            }

            scene_stats.push(ss);
        }

        // Handle empty input
        if total_frames == 0 {
            global_min = 0.0;
            global_max = 0.0;
        }

        let avg_max_pq_f32 = if total_frames == 0 {
            0.0
        } else {
            (weighted_sum / total_frames as f64) as f32
        };

        let overall = SequenceSummary {
            global_min_pq: global_min,
            global_max_pq: global_max,
            avg_max_pq_f32,
            p01_max_pq: global_hist.percentile(1.0),
            p10_max_pq: global_hist.percentile(10.0),
            p50_max_pq: global_hist.percentile(50.0),
            p90_max_pq: global_hist.percentile(90.0),
            p99_max_pq: global_hist.percentile(99.0),
            total_frames,
            total_scenes: scene_stats.len(),
            high_brightness_scenes,
            dark_scenes,
        };

        Self {
            scene_stats,
            overall,
            global_histogram: global_hist,
        }
    }

    /// Return all scenes whose L1 max exceeds `threshold` (normalised PQ).
    #[must_use]
    pub fn scenes_above_threshold(&self, threshold: f32) -> Vec<&SceneStats> {
        self.scene_stats
            .iter()
            .filter(|s| s.l1_max_max > threshold)
            .collect()
    }

    /// Return all scenes whose L1 max falls below `threshold` (normalised PQ).
    #[must_use]
    pub fn scenes_below_threshold(&self, threshold: f32) -> Vec<&SceneStats> {
        self.scene_stats
            .iter()
            .filter(|s| s.l1_max_max < threshold)
            .collect()
    }

    /// Find the scene with the highest peak luminance.
    #[must_use]
    pub fn brightest_scene(&self) -> Option<&SceneStats> {
        self.scene_stats
            .iter()
            .max_by(|a, b| a.l1_max_max.partial_cmp(&b.l1_max_max).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Find the scene with the lowest peak luminance.
    #[must_use]
    pub fn darkest_scene(&self) -> Option<&SceneStats> {
        self.scene_stats
            .iter()
            .min_by(|a, b| a.l1_max_max.partial_cmp(&b.l1_max_max).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Compute the average dynamic range across all scenes.
    ///
    /// Returns `0.0` if there are no scenes.
    #[must_use]
    pub fn average_dynamic_range(&self) -> f32 {
        if self.scene_stats.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.scene_stats.iter().map(|s| s.dynamic_range()).sum();
        sum / self.scene_stats.len() as f32
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dv_xml_export::{DvL2Entry, DvShotEntry};

    fn make_shot(start: u64, end: u64, l1_min: f32, l1_mid: f32, l1_max: f32) -> DvShotEntry {
        DvShotEntry {
            frame_start: start,
            frame_end: end,
            l1_min,
            l1_mid,
            l1_max,
            l2_entries: vec![DvL2Entry::identity(2081)],
        }
    }

    #[test]
    fn test_empty_sequence() {
        let stats = RpuStatistics::from_shots(&[]);
        assert_eq!(stats.scene_stats.len(), 0);
        assert_eq!(stats.overall.total_frames, 0);
        assert_eq!(stats.overall.total_scenes, 0);
        assert!((stats.overall.avg_max_pq_f32).abs() < 1e-6);
    }

    #[test]
    fn test_single_shot_stats() {
        let shot = make_shot(0, 23, 0.001, 0.10, 0.45);
        let stats = RpuStatistics::from_shots(&[shot]);
        assert_eq!(stats.scene_stats.len(), 1);
        assert_eq!(stats.overall.total_frames, 24);
        assert!((stats.overall.avg_max_pq_f32 - 0.45).abs() < 0.01);
        assert!((stats.overall.global_max_pq - 0.45).abs() < 1e-5);
    }

    #[test]
    fn test_two_shots_average() {
        let shots = vec![
            make_shot(0, 23, 0.001, 0.10, 0.45),
            make_shot(24, 47, 0.0, 0.20, 0.70),
        ];
        let stats = RpuStatistics::from_shots(&shots);
        assert_eq!(stats.scene_stats.len(), 2);
        // Both shots have 24 frames, so weighted average = (0.45 + 0.70) / 2
        assert!((stats.overall.avg_max_pq_f32 - 0.575).abs() < 0.01);
    }

    #[test]
    fn test_global_min_max() {
        let shots = vec![
            make_shot(0, 23, 0.002, 0.10, 0.30),
            make_shot(24, 47, 0.001, 0.15, 0.80),
            make_shot(48, 71, 0.005, 0.08, 0.50),
        ];
        let stats = RpuStatistics::from_shots(&shots);
        assert!((stats.overall.global_min_pq - 0.001).abs() < 1e-5);
        assert!((stats.overall.global_max_pq - 0.80).abs() < 1e-5);
    }

    #[test]
    fn test_brightness_classification() {
        let shots = vec![
            make_shot(0, 23, 0.0, 0.05, 0.05),   // dark
            make_shot(24, 47, 0.01, 0.30, 0.60),  // high brightness
            make_shot(48, 71, 0.0, 0.05, 0.08),   // dark
        ];
        let stats = RpuStatistics::from_shots(&shots);
        assert_eq!(stats.overall.dark_scenes, 2);
        assert_eq!(stats.overall.high_brightness_scenes, 1);
    }

    #[test]
    fn test_scenes_above_threshold() {
        let shots = vec![
            make_shot(0, 23, 0.0, 0.10, 0.30),
            make_shot(24, 47, 0.0, 0.20, 0.60),
            make_shot(48, 71, 0.0, 0.05, 0.90),
        ];
        let stats = RpuStatistics::from_shots(&shots);
        let above = stats.scenes_above_threshold(0.5);
        assert_eq!(above.len(), 2);
    }

    #[test]
    fn test_brightest_and_darkest_scene() {
        let shots = vec![
            make_shot(0, 23, 0.0, 0.10, 0.30),
            make_shot(24, 47, 0.0, 0.20, 0.90),
            make_shot(48, 71, 0.0, 0.05, 0.10),
        ];
        let stats = RpuStatistics::from_shots(&shots);

        let brightest = stats.brightest_scene().expect("should have brightest scene");
        assert!((brightest.l1_max_max - 0.90).abs() < 1e-5);

        let darkest = stats.darkest_scene().expect("should have darkest scene");
        assert!((darkest.l1_max_max - 0.10).abs() < 1e-5);
    }

    #[test]
    fn test_pq_histogram_percentiles() {
        let mut hist = PqHistogram::new();
        // Add 100 samples at 0.5
        for _ in 0..100 {
            hist.add_sample(0.5);
        }
        assert_eq!(hist.total_samples, 100);
        // Percentile 50 should be near 0.5
        let p50 = hist.percentile(50.0);
        assert!((p50 - 0.5).abs() < 0.01, "p50={p50}");
    }

    #[test]
    fn test_pq_histogram_mode() {
        let mut hist = PqHistogram::new();
        for _ in 0..10 {
            hist.add_sample(0.2);
        }
        for _ in 0..50 {
            hist.add_sample(0.7);
        }
        let mode = hist.mode_pq().expect("should have mode");
        // Mode should be near 0.7
        assert!((mode - 0.7).abs() < 0.02, "mode={mode}");
    }

    #[test]
    fn test_histogram_merge() {
        let mut h1 = PqHistogram::new();
        let mut h2 = PqHistogram::new();
        h1.add_sample(0.3);
        h2.add_sample(0.7);
        h1.merge(&h2);
        assert_eq!(h1.total_samples, 2);
    }

    #[test]
    fn test_average_dynamic_range() {
        let shots = vec![
            make_shot(0, 23, 0.0, 0.10, 0.5),  // range = 0.5
            make_shot(24, 47, 0.1, 0.30, 0.9), // range = 0.8
        ];
        let stats = RpuStatistics::from_shots(&shots);
        let avg_dr = stats.average_dynamic_range();
        assert!((avg_dr - 0.65).abs() < 0.01, "avg_dr={avg_dr}");
    }
}
