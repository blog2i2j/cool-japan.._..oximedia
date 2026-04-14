//! Metric downsampling with multi-tier retention policies.
//!
//! Provides:
//!
//! - [`RetentionPolicy`] — resolution + retention duration pair.
//! - [`DownsampledPoint`] — aggregated (mean/min/max/count/sum) for one window.
//! - [`downsample_points`] — group raw `(timestamp_ms, value)` pairs into windows.
//! - [`MetricTierStorage`] — multi-tier in-memory storage with automatic pruning.

#![allow(dead_code)]

use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// RetentionPolicy
// ---------------------------------------------------------------------------

/// How long to keep data at a specific resolution.
#[derive(Debug, Clone)]
pub struct RetentionPolicy {
    /// Sample interval at this resolution (milliseconds).
    pub resolution_ms: u64,
    /// How long to retain data at this resolution (milliseconds).
    pub retention_ms: u64,
}

impl RetentionPolicy {
    /// 1-second resolution retained for 1 hour.
    #[must_use]
    pub fn one_second_for_one_hour() -> Self {
        Self {
            resolution_ms: 1_000,
            retention_ms: 3_600 * 1_000,
        }
    }

    /// 1-minute resolution retained for 24 hours.
    #[must_use]
    pub fn one_minute_for_one_day() -> Self {
        Self {
            resolution_ms: 60_000,
            retention_ms: 86_400 * 1_000,
        }
    }

    /// 5-minute resolution retained for 30 days.
    #[must_use]
    pub fn five_minutes_for_thirty_days() -> Self {
        Self {
            resolution_ms: 300_000,
            retention_ms: 30 * 86_400 * 1_000,
        }
    }
}

// ---------------------------------------------------------------------------
// DownsampledPoint
// ---------------------------------------------------------------------------

/// A single aggregated data point (one downsampled window).
#[derive(Debug, Clone)]
pub struct DownsampledPoint {
    /// Aligned start timestamp of this window (milliseconds since epoch).
    pub timestamp_ms: u64,
    /// Mean value across all samples in the window.
    pub mean: f32,
    /// Minimum value in the window.
    pub min: f32,
    /// Maximum value in the window.
    pub max: f32,
    /// Number of samples aggregated.
    pub count: u32,
    /// Sum of all sample values.
    pub sum: f64,
}

impl DownsampledPoint {
    /// Aggregate `samples` into one point aligned to `timestamp_ms`.
    ///
    /// Returns `None` if `samples` is empty.
    #[must_use]
    pub fn from_samples(timestamp_ms: u64, samples: &[f32]) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }
        let mut min = samples[0];
        let mut max = samples[0];
        let mut sum = 0.0f64;
        for &v in samples {
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
            sum += v as f64;
        }
        let count = samples.len() as u32;
        let mean = (sum / count as f64) as f32;
        Some(Self {
            timestamp_ms,
            mean,
            min,
            max,
            count,
            sum,
        })
    }
}

// ---------------------------------------------------------------------------
// downsample_points
// ---------------------------------------------------------------------------

/// Downsample a slice of `(timestamp_ms, value)` pairs into windows of
/// `resolution_ms` milliseconds.
///
/// Points are grouped by `floor(ts / resolution_ms) * resolution_ms`.
/// Empty input produces an empty output.
#[must_use]
pub fn downsample_points(
    points: &[(u64, f32)],
    resolution_ms: u64,
) -> Vec<DownsampledPoint> {
    if points.is_empty() || resolution_ms == 0 {
        return Vec::new();
    }

    // Collect samples per aligned window using BTreeMap to preserve order.
    let mut windows: BTreeMap<u64, Vec<f32>> = BTreeMap::new();
    for &(ts, val) in points {
        let window_start = (ts / resolution_ms) * resolution_ms;
        windows.entry(window_start).or_default().push(val);
    }

    windows
        .into_iter()
        .filter_map(|(ts, samples)| DownsampledPoint::from_samples(ts, &samples))
        .collect()
}

// ---------------------------------------------------------------------------
// MetricTierStorage
// ---------------------------------------------------------------------------

/// Multi-tier downsampled metric storage.
///
/// Each tier has its own [`RetentionPolicy`] that controls window size and
/// how long data is kept.
pub struct MetricTierStorage {
    tiers: Vec<(RetentionPolicy, Vec<DownsampledPoint>)>,
}

impl MetricTierStorage {
    /// Create storage with the given retention policies (one entry per tier).
    #[must_use]
    pub fn new(policies: Vec<RetentionPolicy>) -> Self {
        let tiers = policies.into_iter().map(|p| (p, Vec::new())).collect();
        Self { tiers }
    }

    /// Ingest raw `(timestamp_ms, value)` points, downsampling to each tier
    /// and pruning data that has expired according to `now_ms`.
    pub fn ingest(&mut self, points: &[(u64, f32)], now_ms: u64) {
        for (policy, stored) in &mut self.tiers {
            let new_points = downsample_points(points, policy.resolution_ms);

            // Merge new windows into stored, combining samples that fall into
            // an already-existing window.
            for new_pt in new_points {
                match stored
                    .iter_mut()
                    .find(|p| p.timestamp_ms == new_pt.timestamp_ms)
                {
                    Some(existing) => {
                        // Merge by recomputing aggregates.
                        let total_count = existing.count + new_pt.count;
                        let total_sum = existing.sum + new_pt.sum;
                        existing.min = existing.min.min(new_pt.min);
                        existing.max = existing.max.max(new_pt.max);
                        existing.count = total_count;
                        existing.sum = total_sum;
                        existing.mean = (total_sum / total_count as f64) as f32;
                    }
                    None => stored.push(new_pt),
                }
            }

            // Prune expired entries.
            let cutoff = now_ms.saturating_sub(policy.retention_ms);
            stored.retain(|p| p.timestamp_ms >= cutoff);
        }
    }

    /// Query a specific tier for data within `[from_ms, to_ms]`.
    #[must_use]
    pub fn query_tier(&self, tier_index: usize, from_ms: u64, to_ms: u64) -> Vec<&DownsampledPoint> {
        match self.tiers.get(tier_index) {
            None => Vec::new(),
            Some((_, stored)) => stored
                .iter()
                .filter(|p| p.timestamp_ms >= from_ms && p.timestamp_ms <= to_ms)
                .collect(),
        }
    }

    /// Prune all tiers of data older than `now_ms - retention_ms`.
    ///
    /// Returns the total number of points removed across all tiers.
    pub fn prune_expired(&mut self, now_ms: u64) -> usize {
        let mut removed = 0;
        for (policy, stored) in &mut self.tiers {
            let cutoff = now_ms.saturating_sub(policy.retention_ms);
            let before = stored.len();
            stored.retain(|p| p.timestamp_ms >= cutoff);
            removed += before - stored.len();
        }
        removed
    }

    /// Number of tiers.
    #[must_use]
    pub fn tier_count(&self) -> usize {
        self.tiers.len()
    }

    /// Total number of stored points across all tiers.
    #[must_use]
    pub fn total_points(&self) -> usize {
        self.tiers.iter().map(|(_, pts)| pts.len()).sum()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- RetentionPolicy ----

    #[test]
    fn test_retention_policy_one_second() {
        let p = RetentionPolicy::one_second_for_one_hour();
        assert_eq!(p.resolution_ms, 1_000);
        assert_eq!(p.retention_ms, 3_600_000);
    }

    #[test]
    fn test_retention_policy_one_minute() {
        let p = RetentionPolicy::one_minute_for_one_day();
        assert_eq!(p.resolution_ms, 60_000);
        assert_eq!(p.retention_ms, 86_400_000);
    }

    #[test]
    fn test_retention_policy_five_minutes() {
        let p = RetentionPolicy::five_minutes_for_thirty_days();
        assert_eq!(p.resolution_ms, 300_000);
        assert_eq!(p.retention_ms, 30 * 86_400_000);
    }

    // ---- DownsampledPoint ----

    #[test]
    fn test_downsampled_point_from_samples_stats() {
        let samples = [10.0f32, 20.0, 30.0];
        let pt = DownsampledPoint::from_samples(1000, &samples).expect("should produce point");
        assert!((pt.mean - 20.0).abs() < 0.001, "mean should be 20");
        assert!((pt.min - 10.0).abs() < 0.001, "min should be 10");
        assert!((pt.max - 30.0).abs() < 0.001, "max should be 30");
        assert_eq!(pt.count, 3);
        assert!((pt.sum - 60.0).abs() < 0.001, "sum should be 60");
    }

    #[test]
    fn test_downsampled_point_from_samples_empty_returns_none() {
        assert!(DownsampledPoint::from_samples(0, &[]).is_none());
    }

    #[test]
    fn test_downsampled_point_single_sample() {
        let pt = DownsampledPoint::from_samples(500, &[42.0]).expect("ok");
        assert!((pt.mean - 42.0).abs() < 0.001);
        assert_eq!(pt.min, pt.max);
    }

    // ---- downsample_points ----

    #[test]
    fn test_downsample_points_empty_input() {
        let result = downsample_points(&[], 1_000);
        assert!(result.is_empty());
    }

    #[test]
    fn test_downsample_points_groups_correctly() {
        // Three points in the first 1-second window, one in the next.
        let points = vec![
            (0u64, 10.0f32),
            (200, 20.0),
            (800, 30.0),
            (1_000, 40.0),
        ];
        let result = downsample_points(&points, 1_000);
        assert_eq!(result.len(), 2, "should produce 2 windows");
        let w0 = &result[0];
        assert_eq!(w0.timestamp_ms, 0);
        assert_eq!(w0.count, 3);
        assert!((w0.mean - 20.0).abs() < 0.001);
        let w1 = &result[1];
        assert_eq!(w1.timestamp_ms, 1_000);
        assert_eq!(w1.count, 1);
    }

    #[test]
    fn test_downsample_points_aligns_to_window() {
        let points = vec![(1_500u64, 5.0f32), (1_750, 7.0), (2_100, 9.0)];
        let result = downsample_points(&points, 1_000);
        // 1_500 → window 1_000; 1_750 → window 1_000; 2_100 → window 2_000
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].timestamp_ms, 1_000);
        assert_eq!(result[0].count, 2);
        assert_eq!(result[1].timestamp_ms, 2_000);
        assert_eq!(result[1].count, 1);
    }

    // ---- MetricTierStorage ----

    #[test]
    fn test_tier_storage_tier_count() {
        let storage = MetricTierStorage::new(vec![
            RetentionPolicy::one_second_for_one_hour(),
            RetentionPolicy::one_minute_for_one_day(),
        ]);
        assert_eq!(storage.tier_count(), 2);
    }

    #[test]
    fn test_tier_storage_ingest_then_query() {
        let mut storage = MetricTierStorage::new(vec![
            RetentionPolicy::one_second_for_one_hour(),
        ]);
        let now_ms = 10_000u64;
        let points = vec![(now_ms, 50.0f32), (now_ms + 200, 60.0), (now_ms + 700, 70.0)];
        storage.ingest(&points, now_ms + 1_000);

        let result = storage.query_tier(0, now_ms, now_ms + 1_000);
        assert!(!result.is_empty(), "should have at least one point after ingest");
    }

    #[test]
    fn test_tier_storage_prune_removes_old_data() {
        let mut storage = MetricTierStorage::new(vec![
            RetentionPolicy {
                resolution_ms: 1_000,
                retention_ms: 5_000, // keep only 5 seconds
            },
        ]);
        // Ingest old data.
        let old_ts = 0u64;
        storage.ingest(&[(old_ts, 1.0)], old_ts + 100);

        // Now prune at t=10_000 (old data is 10s old, beyond 5s retention).
        let removed = storage.prune_expired(10_000);
        assert!(removed > 0, "should have pruned old data");
        assert_eq!(storage.total_points(), 0);
    }

    #[test]
    fn test_tier_storage_multi_tier_different_resolutions() {
        let mut storage = MetricTierStorage::new(vec![
            RetentionPolicy::one_second_for_one_hour(),
            RetentionPolicy::one_minute_for_one_day(),
        ]);

        let base = 0u64;
        // Spread points across 2 seconds and into minute windows.
        let points: Vec<(u64, f32)> = (0..120).map(|i| (base + i * 1_000, i as f32)).collect();
        storage.ingest(&points, base + 200_000);

        // 1-second tier should have 120 windows.
        let tier0 = storage.query_tier(0, base, base + 200_000);
        // 1-minute tier should have 2 windows (0..59 → minute 0, 60..119 → minute 1).
        let tier1 = storage.query_tier(1, base, base + 200_000);

        assert!(
            tier0.len() >= tier1.len(),
            "finer tier should have >= points than coarser tier"
        );
        assert!(tier1.len() >= 1, "coarser tier should have at least one point");
    }

    #[test]
    fn test_tier_storage_query_out_of_bounds_tier() {
        let storage = MetricTierStorage::new(vec![
            RetentionPolicy::one_second_for_one_hour(),
        ]);
        let result = storage.query_tier(99, 0, u64::MAX);
        assert!(result.is_empty());
    }
}
