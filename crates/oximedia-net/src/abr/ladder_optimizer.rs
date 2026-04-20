//! Per-title encoding ladder optimizer.
//!
//! A *per-title* encoding ladder avoids the "one-size-fits-all" bitrate ladder
//! used by most ABR deployments.  Instead, the optimal quality-bitrate trade-off
//! is determined per content title (movie, live segment sequence, etc.) by
//! analyzing its *complexity* characteristics.
//!
//! ## Algorithm overview
//!
//! 1. **Complexity sampling** — a set of probe encodes (or quality metrics from
//!    previously encoded segments) is accumulated for the title.
//! 2. **Convex-hull selection** — a subset of (bitrate, quality) operating points
//!    that lie on the Pareto-optimal frontier is identified.  Points inside the
//!    convex hull are discarded because a better quality-bitrate trade-off can
//!    always be achieved using an adjacent hull point.
//! 3. **Ladder generation** — the hull points are mapped to a configurable
//!    number of rungs, with bitrates snapped to a set of allowed values and
//!    resolutions chosen based on the complexity-per-pixel profile.
//! 4. **BBA integration** — the generated ladder is compatible with the Buffer-
//!    Based Adaptation algorithm ([`super::bba`]) so that the ABR controller
//!    can perform quality selection using the optimized rungs.
//!
//! ## Key types
//!
//! - [`ComplexitySample`] — a (bitrate, quality_score) measurement for one probe.
//! - [`LadderPoint`] — an optimized (bitrate, resolution, quality) rung.
//! - [`LadderOptimizerConfig`] — tunable parameters.
//! - [`LadderOptimizer`] — accumulates samples and generates optimized ladders.
//! - [`OptimizedLadder`] — the output ladder ready for use by an ABR controller.

#![allow(dead_code)]

use std::time::{Duration, Instant};

use crate::error::{NetError, NetResult};

// ─── Complexity Sample ────────────────────────────────────────────────────────

/// A single quality measurement at a given encoding bitrate.
///
/// The quality score should be on a scale where higher is better.  Common
/// choices are VMAF (0–100), SSIM (0–1), or PSNR (dB, typically 30–50).
#[derive(Debug, Clone)]
pub struct ComplexitySample {
    /// Encoded bitrate in bits per second.
    pub bitrate_bps: u64,
    /// Quality score (higher is better; scale depends on metric).
    pub quality_score: f64,
    /// Spatial complexity measure (0.0–1.0; 0 = static, 1 = very complex).
    pub spatial_complexity: f64,
    /// Temporal complexity measure (0.0–1.0; 0 = no motion, 1 = very active).
    pub temporal_complexity: f64,
    /// Video resolution at which the measurement was taken.
    pub resolution: (u32, u32),
    /// When this sample was collected.
    pub sampled_at: Instant,
}

impl ComplexitySample {
    /// Creates a new complexity sample.
    #[must_use]
    pub fn new(
        bitrate_bps: u64,
        quality_score: f64,
        spatial_complexity: f64,
        temporal_complexity: f64,
        resolution: (u32, u32),
    ) -> Self {
        Self {
            bitrate_bps,
            quality_score,
            spatial_complexity,
            temporal_complexity,
            resolution,
            sampled_at: Instant::now(),
        }
    }

    /// Returns the combined complexity score (simple average).
    #[must_use]
    pub fn combined_complexity(&self) -> f64 {
        (self.spatial_complexity + self.temporal_complexity) * 0.5
    }
}

// ─── Ladder Point ─────────────────────────────────────────────────────────────

/// One rung in the optimized encoding ladder.
#[derive(Debug, Clone, PartialEq)]
pub struct LadderPoint {
    /// Target bitrate for this rung in bits per second.
    pub bitrate_bps: u64,
    /// Recommended encoding resolution.
    pub resolution: (u32, u32),
    /// Expected quality score at this rung.
    pub expected_quality: f64,
    /// Whether this point lies exactly on the Pareto-optimal convex hull.
    pub on_hull: bool,
}

impl LadderPoint {
    /// Returns a human-readable label such as `"1080p @ 4.2 Mbps"`.
    #[must_use]
    pub fn label(&self) -> String {
        let (w, h) = self.resolution;
        let mbps = self.bitrate_bps as f64 / 1_000_000.0;
        format!("{h}p @ {mbps:.1} Mbps ({w}x{h})")
    }
}

// ─── Allowed Resolutions ──────────────────────────────────────────────────────

/// Standard resolutions available for ladder rungs.
pub const STANDARD_RESOLUTIONS: &[(u32, u32)] = &[
    (416, 234),
    (640, 360),
    (854, 480),
    (1280, 720),
    (1920, 1080),
    (2560, 1440),
    (3840, 2160),
];

/// Allowed bitrate values (bps) that rungs are snapped to.
pub const ALLOWED_BITRATES_BPS: &[u64] = &[
    150_000,
    300_000,
    500_000,
    800_000,
    1_200_000,
    2_000_000,
    3_000_000,
    4_500_000,
    6_000_000,
    9_000_000,
    15_000_000,
    25_000_000,
];

// ─── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for [`LadderOptimizer`].
#[derive(Debug, Clone)]
pub struct LadderOptimizerConfig {
    /// Target quality score for the highest rung (e.g., VMAF 95).
    pub target_top_quality: f64,
    /// Minimum quality score acceptable for the lowest rung (e.g., VMAF 50).
    pub min_bottom_quality: f64,
    /// Desired number of rungs in the output ladder.
    pub target_rung_count: usize,
    /// Minimum number of samples required before optimization is attempted.
    pub min_samples: usize,
    /// Maximum age of a sample before it is discarded as stale.
    pub sample_ttl: Duration,
    /// Quality metric scale: true = VMAF (0–100), false = SSIM-like (0–1).
    pub vmaf_scale: bool,
    /// Safety margin: scale down bitrates by this factor to avoid rebuffering.
    /// Typical values: 0.85–0.95.
    pub bitrate_safety: f64,
}

impl Default for LadderOptimizerConfig {
    fn default() -> Self {
        Self {
            target_top_quality: 95.0,
            min_bottom_quality: 45.0,
            target_rung_count: 6,
            min_samples: 3,
            sample_ttl: Duration::from_secs(3600), // 1 hour
            vmaf_scale: true,
            bitrate_safety: 0.90,
        }
    }
}

impl LadderOptimizerConfig {
    /// Creates a configuration optimised for live streaming (tighter latency budget).
    #[must_use]
    pub fn live() -> Self {
        Self {
            target_top_quality: 88.0,
            min_bottom_quality: 40.0,
            target_rung_count: 4,
            min_samples: 2,
            sample_ttl: Duration::from_secs(300), // 5 min live window
            vmaf_scale: true,
            bitrate_safety: 0.85,
        }
    }

    /// Creates a configuration for high-quality VOD publishing.
    #[must_use]
    pub fn vod_high_quality() -> Self {
        Self {
            target_top_quality: 97.0,
            min_bottom_quality: 55.0,
            target_rung_count: 8,
            min_samples: 5,
            sample_ttl: Duration::from_secs(86400), // 24 h
            vmaf_scale: true,
            bitrate_safety: 0.92,
        }
    }
}

// ─── Optimized Ladder ─────────────────────────────────────────────────────────

/// The output of the per-title optimization: an ordered list of ladder rungs.
///
/// Rungs are sorted in ascending order of bitrate (index 0 = lowest quality).
#[derive(Debug, Clone)]
pub struct OptimizedLadder {
    /// The optimized ladder rungs.
    pub rungs: Vec<LadderPoint>,
    /// Average complexity of the title content (0.0–1.0).
    pub avg_complexity: f64,
    /// Estimated bitrate savings vs. a default reference ladder (0.0–1.0).
    /// E.g., 0.15 means "15% lower bitrate for equivalent quality".
    pub bitrate_saving_fraction: f64,
    /// Number of samples used to generate this ladder.
    pub sample_count: usize,
    /// When the ladder was generated.
    pub generated_at: Instant,
}

impl OptimizedLadder {
    /// Returns the rung with the highest bitrate.
    #[must_use]
    pub fn top_rung(&self) -> Option<&LadderPoint> {
        self.rungs.last()
    }

    /// Returns the rung with the lowest bitrate.
    #[must_use]
    pub fn bottom_rung(&self) -> Option<&LadderPoint> {
        self.rungs.first()
    }

    /// Returns the rung whose bitrate is closest to `target_bps` from below.
    ///
    /// Useful for BBA integration: select the highest rung whose bitrate does
    /// not exceed the available bandwidth estimate.
    #[must_use]
    pub fn rung_at_or_below(&self, target_bps: u64) -> Option<&LadderPoint> {
        self.rungs
            .iter()
            .filter(|r| r.bitrate_bps <= target_bps)
            .last()
    }

    /// Returns the index (0-based) of the rung matching `bitrate_bps`, or
    /// `None` if no exact match.
    #[must_use]
    pub fn rung_index(&self, bitrate_bps: u64) -> Option<usize> {
        self.rungs.iter().position(|r| r.bitrate_bps == bitrate_bps)
    }
}

// ─── Optimizer ────────────────────────────────────────────────────────────────

/// Accumulates complexity samples for a title and generates an optimized
/// per-title encoding ladder.
///
/// # Example
///
/// ```
/// use oximedia_net::abr::ladder_optimizer::{
///     ComplexitySample, LadderOptimizerConfig, LadderOptimizer,
/// };
///
/// let config = LadderOptimizerConfig::default();
/// let mut optimizer = LadderOptimizer::new(config);
///
/// // Feed in probe-encode measurements.
/// optimizer.add_sample(ComplexitySample::new(
///     2_000_000, 82.0, 0.4, 0.3, (1280, 720),
/// ));
/// optimizer.add_sample(ComplexitySample::new(
///     4_500_000, 93.0, 0.4, 0.3, (1920, 1080),
/// ));
/// optimizer.add_sample(ComplexitySample::new(
///     800_000, 65.0, 0.4, 0.3, (854, 480),
/// ));
///
/// let ladder = optimizer.optimize().expect("optimizer has sufficient samples");
/// assert!(!ladder.rungs.is_empty());
/// ```
pub struct LadderOptimizer {
    config: LadderOptimizerConfig,
    samples: Vec<ComplexitySample>,
}

impl LadderOptimizer {
    /// Creates a new optimizer with the given configuration.
    #[must_use]
    pub fn new(config: LadderOptimizerConfig) -> Self {
        Self {
            config,
            samples: Vec::new(),
        }
    }

    /// Adds a complexity sample.
    ///
    /// Stale samples (older than `config.sample_ttl`) are pruned automatically.
    pub fn add_sample(&mut self, sample: ComplexitySample) {
        let ttl = self.config.sample_ttl;
        self.samples
            .retain(|s| s.sampled_at.elapsed() < ttl);
        self.samples.push(sample);
    }

    /// Returns the number of valid (non-stale) samples.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        let ttl = self.config.sample_ttl;
        self.samples
            .iter()
            .filter(|s| s.sampled_at.elapsed() < ttl)
            .count()
    }

    /// Generates an optimized ladder from the accumulated samples.
    ///
    /// # Errors
    ///
    /// Returns [`NetError::InvalidState`] if fewer than `min_samples` valid
    /// samples are available.
    pub fn optimize(&self) -> NetResult<OptimizedLadder> {
        let ttl = self.config.sample_ttl;
        let valid: Vec<&ComplexitySample> = self
            .samples
            .iter()
            .filter(|s| s.sampled_at.elapsed() < ttl)
            .collect();

        if valid.len() < self.config.min_samples {
            return Err(NetError::invalid_state(format!(
                "insufficient samples: {} < {} required",
                valid.len(),
                self.config.min_samples
            )));
        }

        // ── 1. Sort by bitrate ascending ──────────────────────────────────
        let mut sorted: Vec<&ComplexitySample> = valid.clone();
        sorted.sort_by_key(|s| s.bitrate_bps);

        // ── 2. Compute average complexity ────────────────────────────────
        let avg_complexity = sorted
            .iter()
            .map(|s| s.combined_complexity())
            .sum::<f64>()
            / sorted.len() as f64;

        // ── 3. Build convex hull in (bitrate, quality) space ─────────────
        let hull = self.upper_convex_hull(&sorted);

        // ── 4. Select target_rung_count points from the hull ─────────────
        let rungs = self.select_rungs(&hull, avg_complexity);

        // ── 5. Estimate bitrate savings vs. a default ladder ─────────────
        let saving = self.estimate_savings(&rungs);

        Ok(OptimizedLadder {
            rungs,
            avg_complexity,
            bitrate_saving_fraction: saving,
            sample_count: valid.len(),
            generated_at: Instant::now(),
        })
    }

    /// Clears all accumulated samples.
    pub fn reset(&mut self) {
        self.samples.clear();
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Computes the upper convex hull of the (bitrate, quality) points.
    ///
    /// Uses Graham scan on points sorted by bitrate ascending.
    /// Returns points on the upper hull (monotonically increasing quality).
    fn upper_convex_hull<'a>(&self, sorted: &[&'a ComplexitySample]) -> Vec<&'a ComplexitySample> {
        if sorted.is_empty() {
            return vec![];
        }

        let mut hull: Vec<&ComplexitySample> = Vec::new();

        for &s in sorted {
            // Cross-product test: remove hull points that are below the line
            // from the last-but-one to `s`.
            while hull.len() >= 2 {
                let n = hull.len();
                let p1 = hull[n - 2];
                let p2 = hull[n - 1];

                // Cross product of (p2 - p1) × (s - p1)
                let dx1 = p2.bitrate_bps as f64 - p1.bitrate_bps as f64;
                let dy1 = p2.quality_score - p1.quality_score;
                let dx2 = s.bitrate_bps as f64 - p1.bitrate_bps as f64;
                let dy2 = s.quality_score - p1.quality_score;

                let cross = dx1 * dy2 - dy1 * dx2;

                if cross <= 0.0 {
                    hull.pop(); // p2 is below or on the line → discard
                } else {
                    break;
                }
            }
            hull.push(s);
        }

        hull
    }

    /// Selects up to `target_rung_count` ladder points from the hull.
    fn select_rungs(&self, hull: &[&ComplexitySample], avg_complexity: f64) -> Vec<LadderPoint> {
        if hull.is_empty() {
            return vec![];
        }

        // Filter hull points within the target quality range.
        let filtered: Vec<&&ComplexitySample> = hull
            .iter()
            .filter(|s| s.quality_score >= self.config.min_bottom_quality)
            .collect();

        if filtered.is_empty() {
            // Fall back to all hull points if none pass the quality gate.
            return self.hull_to_ladder_points(hull, avg_complexity);
        }

        // Down-sample to target_rung_count evenly spaced hull points.
        let n = filtered.len();
        let count = self.config.target_rung_count.min(n);

        let indices: Vec<usize> = if count == 1 {
            vec![0]
        } else {
            (0..count)
                .map(|i| i * (n - 1) / (count - 1))
                .collect()
        };

        indices
            .into_iter()
            .map(|i| self.sample_to_ladder_point(filtered[i], avg_complexity, true))
            .collect()
    }

    fn hull_to_ladder_points(
        &self,
        hull: &[&ComplexitySample],
        avg_complexity: f64,
    ) -> Vec<LadderPoint> {
        hull.iter()
            .map(|s| self.sample_to_ladder_point(s, avg_complexity, true))
            .collect()
    }

    fn sample_to_ladder_point(
        &self,
        sample: &ComplexitySample,
        _avg_complexity: f64,
        on_hull: bool,
    ) -> LadderPoint {
        let raw_bitrate = (sample.bitrate_bps as f64 * self.config.bitrate_safety) as u64;
        let snapped_bitrate = Self::snap_bitrate(raw_bitrate);
        let resolution = Self::choose_resolution(sample.resolution, sample.combined_complexity());

        LadderPoint {
            bitrate_bps: snapped_bitrate,
            resolution,
            expected_quality: sample.quality_score,
            on_hull,
        }
    }

    /// Snaps a bitrate to the nearest allowed value.
    fn snap_bitrate(bps: u64) -> u64 {
        ALLOWED_BITRATES_BPS
            .iter()
            .copied()
            .min_by_key(|&allowed| {
                let diff = if allowed > bps { allowed - bps } else { bps - allowed };
                diff
            })
            .unwrap_or(bps)
    }

    /// Chooses the most appropriate resolution based on original resolution
    /// and complexity.
    ///
    /// Complex content benefits from higher resolutions; simple content can
    /// be down-scaled to save bitrate.
    fn choose_resolution(original: (u32, u32), complexity: f64) -> (u32, u32) {
        let (ow, oh) = original;

        // Find the closest standard resolution at or below original height.
        let target_height = if complexity < 0.25 {
            // Simple content: allow one step down.
            (oh as f64 * 0.75) as u32
        } else {
            oh
        };

        STANDARD_RESOLUTIONS
            .iter()
            .copied()
            .filter(|&(_, h)| h <= oh)
            .min_by_key(|&(_, h)| {
                let diff = if h > target_height {
                    h - target_height
                } else {
                    target_height - h
                };
                diff
            })
            .unwrap_or((ow, oh))
    }

    /// Estimates bitrate savings vs. a naive linear ladder spanning the same
    /// quality range.
    fn estimate_savings(&self, rungs: &[LadderPoint]) -> f64 {
        if rungs.is_empty() {
            return 0.0;
        }

        let max_bitrate = rungs
            .iter()
            .map(|r| r.bitrate_bps)
            .max()
            .unwrap_or(0);

        // Reference ladder: equally-spaced bitrates from 150 kbps to max_bitrate.
        let n = rungs.len();
        let ref_total: u64 = (0..n)
            .map(|i| {
                let frac = if n == 1 {
                    1.0
                } else {
                    i as f64 / (n - 1) as f64
                };
                (150_000.0 + frac * (max_bitrate as f64 - 150_000.0)) as u64
            })
            .sum();

        let opt_total: u64 = rungs.iter().map(|r| r.bitrate_bps).sum();

        if ref_total == 0 {
            return 0.0;
        }

        let saving = 1.0 - (opt_total as f64 / ref_total as f64);
        saving.clamp(0.0, 1.0)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sample(bps: u64, quality: f64) -> ComplexitySample {
        ComplexitySample::new(bps, quality, 0.3, 0.3, (1280, 720))
    }

    fn make_optimizer_with_samples(n: usize) -> LadderOptimizer {
        let config = LadderOptimizerConfig {
            min_samples: 3,
            target_rung_count: 4,
            ..LadderOptimizerConfig::default()
        };
        let mut opt = LadderOptimizer::new(config);
        let bitrates = [300_000, 800_000, 2_000_000, 4_500_000, 6_000_000, 9_000_000];
        let qualities = [40.0, 60.0, 78.0, 90.0, 94.0, 97.0];
        for i in 0..n {
            opt.add_sample(make_sample(bitrates[i % 6], qualities[i % 6]));
        }
        opt
    }

    #[test]
    fn test_optimizer_requires_min_samples() {
        let opt = make_optimizer_with_samples(1);
        let err = opt
            .optimize()
            .expect_err("fewer than min_samples should fail");
        assert!(matches!(err, NetError::InvalidState(_)));
    }

    #[test]
    fn test_optimizer_generates_ladder() {
        let opt = make_optimizer_with_samples(5);
        let ladder = opt.optimize().expect("optimizer has sufficient samples");
        assert!(!ladder.rungs.is_empty());
    }

    #[test]
    fn test_ladder_rungs_sorted_ascending() {
        let opt = make_optimizer_with_samples(5);
        let ladder = opt.optimize().expect("optimizer has sufficient samples");
        let bitrates: Vec<u64> = ladder.rungs.iter().map(|r| r.bitrate_bps).collect();
        let mut sorted = bitrates.clone();
        sorted.sort_unstable();
        assert_eq!(bitrates, sorted, "ladder rungs must be sorted ascending");
    }

    #[test]
    fn test_rung_at_or_below() {
        let opt = make_optimizer_with_samples(5);
        let ladder = opt.optimize().expect("optimizer has sufficient samples");
        // The optimized ladder may have one or more rungs; query at a very high
        // bitrate to guarantee at least one rung is at or below it.
        let top = ladder.top_rung().expect("ladder has at least one rung");
        let target = top.bitrate_bps + 1_000_000;
        let rung = ladder.rung_at_or_below(target);
        assert!(rung.is_some(), "should find a rung below {target}bps");
        assert!(
            rung.expect("rung is Some, checked above").bitrate_bps <= target
        );
    }

    #[test]
    fn test_top_and_bottom_rungs() {
        let opt = make_optimizer_with_samples(5);
        let ladder = opt.optimize().expect("optimizer has sufficient samples");
        let top = ladder.top_rung().expect("ladder has at least one rung");
        let bottom = ladder.bottom_rung().expect("ladder has at least one rung");
        assert!(top.bitrate_bps >= bottom.bitrate_bps);
    }

    #[test]
    fn test_ladder_point_label() {
        let pt = LadderPoint {
            bitrate_bps: 4_200_000,
            resolution: (1920, 1080),
            expected_quality: 93.0,
            on_hull: true,
        };
        let label = pt.label();
        assert!(label.contains("1080p"), "label should contain height: {label}");
        assert!(label.contains("4.2 Mbps"), "label should contain bitrate: {label}");
    }

    #[test]
    fn test_sample_count_after_add() {
        let config = LadderOptimizerConfig::default();
        let mut opt = LadderOptimizer::new(config);
        opt.add_sample(make_sample(2_000_000, 80.0));
        opt.add_sample(make_sample(4_000_000, 90.0));
        assert_eq!(opt.sample_count(), 2);
    }

    #[test]
    fn test_reset_clears_samples() {
        let mut opt = make_optimizer_with_samples(4);
        opt.reset();
        assert_eq!(opt.sample_count(), 0);
    }

    #[test]
    fn test_snap_bitrate_chooses_nearest() {
        let snapped = LadderOptimizer::snap_bitrate(2_200_000);
        // Nearest in ALLOWED_BITRATES_BPS: 2_000_000 or 3_000_000.
        // Diff to 2M = 200k, diff to 3M = 800k → should snap to 2M.
        assert_eq!(snapped, 2_000_000);
    }

    #[test]
    fn test_snap_bitrate_exact_match() {
        let snapped = LadderOptimizer::snap_bitrate(800_000);
        assert_eq!(snapped, 800_000);
    }

    #[test]
    fn test_bitrate_saving_fraction_in_range() {
        let opt = make_optimizer_with_samples(6);
        let ladder = opt.optimize().expect("optimizer has sufficient samples");
        assert!(
            (0.0..=1.0).contains(&ladder.bitrate_saving_fraction),
            "saving fraction out of range: {}",
            ladder.bitrate_saving_fraction
        );
    }

    #[test]
    fn test_avg_complexity_bounded() {
        let opt = make_optimizer_with_samples(5);
        let ladder = opt.optimize().expect("optimizer has sufficient samples");
        assert!(
            (0.0..=1.0).contains(&ladder.avg_complexity),
            "avg_complexity out of [0,1]: {}",
            ladder.avg_complexity
        );
    }

    #[test]
    fn test_live_config_fewer_rungs() {
        let config = LadderOptimizerConfig::live();
        assert!(config.target_rung_count <= 4);
    }

    #[test]
    fn test_vod_config_more_rungs() {
        let config = LadderOptimizerConfig::vod_high_quality();
        assert!(config.target_rung_count >= 6);
    }

    #[test]
    fn test_complexity_sample_combined() {
        let s = ComplexitySample::new(1_000_000, 75.0, 0.4, 0.6, (1280, 720));
        let combined = s.combined_complexity();
        assert!((combined - 0.5).abs() < 1e-9);
    }
}
