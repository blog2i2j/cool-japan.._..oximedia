//! Scene-aware adaptive Group-of-Pictures (GOP) size optimizer.
//!
//! Conventional encoders use a fixed GOP length (e.g. every 2 seconds at 30 fps
//! = 60 frames).  This module improves visual quality and compression efficiency
//! by:
//!
//! 1. Aligning I-frames with detected scene cuts so reference frames are never
//!    wasted across a hard scene change.
//! 2. Capping the maximum GOP length to limit random-access latency.
//! 3. Enforcing a minimum GOP length to prevent excessive I-frame overhead.
//! 4. Optionally inserting periodic "safe" I-frames within a long scene to
//!    satisfy streaming segment boundaries.

use crate::scene_cut::SceneCut;

/// Policy for placing I-frames within the video stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GopPlacementPolicy {
    /// Insert I-frames only at scene cuts (plus the very first frame).
    CutsOnly,
    /// Insert I-frames at scene cuts AND at regular periodic intervals.
    CutsPlusFixed,
    /// Fixed period only; scene cuts are ignored.
    FixedOnly,
}

/// Configuration for the GOP optimizer.
#[derive(Debug, Clone)]
pub struct GopConfig {
    /// Minimum allowed GOP length in frames (prevents excessive I-frame overhead).
    pub min_gop: u32,
    /// Maximum allowed GOP length in frames (limits random-access latency).
    pub max_gop: u32,
    /// Fixed periodic I-frame interval when `policy` includes periodic insertion.
    pub fixed_period: u32,
    /// Policy governing when I-frames are placed.
    pub policy: GopPlacementPolicy,
    /// Confidence threshold: scene cuts below this value are ignored.
    pub cut_confidence_threshold: f32,
}

impl Default for GopConfig {
    fn default() -> Self {
        Self {
            min_gop: 12,
            max_gop: 300,
            fixed_period: 120, // 4 s at 30 fps
            policy: GopPlacementPolicy::CutsPlusFixed,
            cut_confidence_threshold: 0.4,
        }
    }
}

impl GopConfig {
    /// Creates a new config with explicit min/max and fixed period.
    #[must_use]
    pub fn new(min_gop: u32, max_gop: u32, fixed_period: u32) -> Self {
        Self {
            min_gop,
            max_gop,
            fixed_period,
            ..Self::default()
        }
    }

    /// Builder: sets the placement policy.
    #[must_use]
    pub fn with_policy(mut self, policy: GopPlacementPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Builder: sets the minimum confidence required for a scene cut to force an I-frame.
    #[must_use]
    pub fn with_cut_threshold(mut self, threshold: f32) -> Self {
        self.cut_confidence_threshold = threshold;
        self
    }
}

/// A single I-frame placement decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IFramePoint {
    /// Zero-based frame index where the I-frame should be placed.
    pub frame: u64,
    /// Why this I-frame was inserted.
    pub reason: IFrameReason,
}

/// Reason an I-frame was inserted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IFrameReason {
    /// First frame of the stream.
    StreamStart,
    /// Coincides with a detected scene cut.
    SceneCut,
    /// Regular periodic insertion (does not align with a scene cut).
    Periodic,
}

impl IFramePoint {
    /// Creates an `IFramePoint`.
    #[must_use]
    pub fn new(frame: u64, reason: IFrameReason) -> Self {
        Self { frame, reason }
    }
}

/// Computes I-frame placement given detected scene cuts and a configuration.
///
/// The algorithm:
/// 1. Always place an I-frame at frame 0.
/// 2. Collect scene-cut frames whose confidence exceeds the threshold.
/// 3. Depending on policy, also add periodic I-frames.
/// 4. Merge and de-duplicate, then enforce min/max GOP constraints:
///    - Remove any I-frame that would create a GOP shorter than `min_gop`
///      (except at stream start and forced cuts).
///    - Insert additional I-frames wherever a gap exceeds `max_gop`.
/// 5. Return the final sorted list.
#[must_use]
pub fn compute_i_frame_placements(
    total_frames: u64,
    cuts: &[SceneCut],
    config: &GopConfig,
) -> Vec<IFramePoint> {
    if total_frames == 0 {
        return Vec::new();
    }

    let mut candidates: Vec<IFramePoint> = Vec::new();

    // 1. Stream start
    candidates.push(IFramePoint::new(0, IFrameReason::StreamStart));

    // 2. Scene cuts (if policy includes them)
    if config.policy != GopPlacementPolicy::FixedOnly {
        for cut in cuts {
            if cut.confidence >= config.cut_confidence_threshold && cut.frame < total_frames {
                candidates.push(IFramePoint::new(cut.frame, IFrameReason::SceneCut));
            }
        }
    }

    // 3. Periodic I-frames
    if config.policy != GopPlacementPolicy::CutsOnly && config.fixed_period > 0 {
        let period = u64::from(config.fixed_period);
        let mut f = period;
        while f < total_frames {
            // Only add if not already covered by a scene cut at the same frame
            let already = candidates.iter().any(|p| p.frame == f);
            if !already {
                candidates.push(IFramePoint::new(f, IFrameReason::Periodic));
            }
            f += period;
        }
    }

    // 4. Sort by frame number
    candidates.sort_by_key(|p| p.frame);
    candidates.dedup_by_key(|p| p.frame);

    // 5. Enforce min_gop: remove cuts that are too close to the previous I-frame
    //    (we never remove the stream-start I-frame or forced hard cuts).
    let min_gop = u64::from(config.min_gop);
    let mut filtered: Vec<IFramePoint> = Vec::with_capacity(candidates.len());
    let mut last_iframe: u64 = 0;
    for point in candidates {
        let gap = point.frame.saturating_sub(last_iframe);
        let is_first = point.frame == 0;
        if is_first || gap >= min_gop {
            last_iframe = point.frame;
            filtered.push(point);
        }
        // else: skip — too close to previous I-frame
    }

    // 6. Enforce max_gop: insert synthetic periodic I-frames in gaps that are too long
    let max_gop = u64::from(config.max_gop);
    let mut result: Vec<IFramePoint> = Vec::with_capacity(filtered.len() * 2);
    for i in 0..filtered.len() {
        result.push(filtered[i].clone());
        let next_iframe = if i + 1 < filtered.len() {
            filtered[i + 1].frame
        } else {
            total_frames
        };
        let gap = next_iframe.saturating_sub(filtered[i].frame);
        if gap > max_gop {
            // Fill with periodic synthetic I-frames
            let mut fill = filtered[i].frame + max_gop;
            while fill < next_iframe {
                result.push(IFramePoint::new(fill, IFrameReason::Periodic));
                fill += max_gop;
            }
        }
    }

    result
}

/// Analyses an existing `IFramePoint` schedule and returns summary statistics.
#[derive(Debug, Clone)]
pub struct GopStats {
    /// Total number of I-frames.
    pub i_frame_count: u32,
    /// Mean GOP length in frames.
    pub mean_gop: f64,
    /// Shortest GOP in frames.
    pub min_gop: u64,
    /// Longest GOP in frames.
    pub max_gop: u64,
    /// Number of scene-cut I-frames.
    pub scene_cut_count: u32,
    /// Number of periodic I-frames.
    pub periodic_count: u32,
}

/// Computes statistics over a set of I-frame placements.
///
/// `total_frames` is the total number of frames in the video.
#[must_use]
pub fn gop_stats(placements: &[IFramePoint], total_frames: u64) -> GopStats {
    if placements.is_empty() {
        return GopStats {
            i_frame_count: 0,
            mean_gop: 0.0,
            min_gop: 0,
            max_gop: 0,
            scene_cut_count: 0,
            periodic_count: 0,
        };
    }

    let mut gop_lengths: Vec<u64> = Vec::with_capacity(placements.len());
    for i in 0..placements.len() {
        let end = if i + 1 < placements.len() {
            placements[i + 1].frame
        } else {
            total_frames
        };
        gop_lengths.push(end.saturating_sub(placements[i].frame));
    }

    let min_gop = gop_lengths.iter().copied().min().unwrap_or(0);
    let max_gop = gop_lengths.iter().copied().max().unwrap_or(0);
    let sum: u64 = gop_lengths.iter().sum();
    let mean_gop = if gop_lengths.is_empty() {
        0.0
    } else {
        sum as f64 / gop_lengths.len() as f64
    };

    let scene_cut_count = placements
        .iter()
        .filter(|p| p.reason == IFrameReason::SceneCut)
        .count() as u32;
    let periodic_count = placements
        .iter()
        .filter(|p| p.reason == IFrameReason::Periodic)
        .count() as u32;

    GopStats {
        i_frame_count: placements.len() as u32,
        mean_gop,
        min_gop,
        max_gop,
        scene_cut_count,
        periodic_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene_cut::{CutDetectionMethod, SceneCut};

    fn make_cut(frame: u64, confidence: f32) -> SceneCut {
        SceneCut::new(frame, confidence, CutDetectionMethod::Histogram)
    }

    // ---------- GopConfig ----------

    #[test]
    fn test_default_config() {
        let c = GopConfig::default();
        assert_eq!(c.min_gop, 12);
        assert_eq!(c.max_gop, 300);
        assert_eq!(c.fixed_period, 120);
        assert_eq!(c.policy, GopPlacementPolicy::CutsPlusFixed);
        assert!((c.cut_confidence_threshold - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_config_builder_methods() {
        let c = GopConfig::new(10, 200, 60)
            .with_policy(GopPlacementPolicy::CutsOnly)
            .with_cut_threshold(0.6);
        assert_eq!(c.min_gop, 10);
        assert_eq!(c.max_gop, 200);
        assert_eq!(c.fixed_period, 60);
        assert_eq!(c.policy, GopPlacementPolicy::CutsOnly);
        assert!((c.cut_confidence_threshold - 0.6).abs() < 1e-6);
    }

    // ---------- compute_i_frame_placements ----------

    #[test]
    fn test_empty_stream_returns_empty() {
        let placements = compute_i_frame_placements(0, &[], &GopConfig::default());
        assert!(placements.is_empty());
    }

    #[test]
    fn test_stream_start_always_present() {
        let config = GopConfig::default();
        let placements = compute_i_frame_placements(100, &[], &config);
        assert!(!placements.is_empty());
        assert_eq!(placements[0].frame, 0);
        assert_eq!(placements[0].reason, IFrameReason::StreamStart);
    }

    #[test]
    fn test_scene_cut_inserted() {
        let config = GopConfig::new(5, 1000, 500)
            .with_policy(GopPlacementPolicy::CutsOnly)
            .with_cut_threshold(0.4);
        let cuts = vec![make_cut(50, 0.9)];
        let placements = compute_i_frame_placements(200, &cuts, &config);
        assert!(placements.iter().any(|p| p.frame == 50 && p.reason == IFrameReason::SceneCut));
    }

    #[test]
    fn test_low_confidence_cut_ignored() {
        let config = GopConfig::new(5, 1000, 500)
            .with_policy(GopPlacementPolicy::CutsOnly)
            .with_cut_threshold(0.8);
        let cuts = vec![make_cut(50, 0.3)];
        let placements = compute_i_frame_placements(200, &cuts, &config);
        assert!(!placements.iter().any(|p| p.frame == 50));
    }

    #[test]
    fn test_periodic_insertion() {
        let config = GopConfig::new(5, 10000, 60)
            .with_policy(GopPlacementPolicy::FixedOnly);
        let placements = compute_i_frame_placements(200, &[], &config);
        // Should have I-frames at 0, 60, 120, 180
        assert!(placements.iter().any(|p| p.frame == 60));
        assert!(placements.iter().any(|p| p.frame == 120));
        assert!(placements.iter().any(|p| p.frame == 180));
    }

    #[test]
    fn test_min_gop_enforced() {
        // Two cuts at frame 50 and 55 — gap is 5 but min_gop is 20
        let config = GopConfig::new(20, 1000, 500)
            .with_policy(GopPlacementPolicy::CutsOnly)
            .with_cut_threshold(0.4);
        let cuts = vec![make_cut(50, 0.9), make_cut(55, 0.95)];
        let placements = compute_i_frame_placements(200, &cuts, &config);
        // The second cut should be dropped because gap < min_gop
        let frames: Vec<u64> = placements.iter().map(|p| p.frame).collect();
        // Only one of {50, 55} should appear (both can't since gap < min_gop)
        let fifty = frames.contains(&50);
        let fiftyfive = frames.contains(&55);
        assert!(fifty || fiftyfive, "at least one of the cuts should appear");
        assert!(!(fifty && fiftyfive), "both cannot appear due to min_gop");
    }

    #[test]
    fn test_max_gop_enforced() {
        let config = GopConfig::new(1, 30, 10000)
            .with_policy(GopPlacementPolicy::CutsOnly)
            .with_cut_threshold(0.4);
        // No cuts → huge gap from frame 0 to 300
        let placements = compute_i_frame_placements(300, &[], &config);
        // Should have synthetic I-frames at 30, 60, 90, … to cap gap at 30
        assert!(
            placements.iter().any(|p| p.frame == 30 && p.reason == IFrameReason::Periodic),
            "synthetic I-frame at 30 expected"
        );
    }

    #[test]
    fn test_no_duplicate_frames() {
        let config = GopConfig::new(5, 1000, 60)
            .with_policy(GopPlacementPolicy::CutsPlusFixed);
        // Cut exactly on a periodic boundary
        let cuts = vec![make_cut(60, 0.9)];
        let placements = compute_i_frame_placements(200, &cuts, &config);
        let mut frames: Vec<u64> = placements.iter().map(|p| p.frame).collect();
        frames.sort_unstable();
        let count_before = frames.len();
        frames.dedup();
        assert_eq!(frames.len(), count_before, "no duplicate frames");
    }

    // ---------- gop_stats ----------

    #[test]
    fn test_gop_stats_empty() {
        let s = gop_stats(&[], 100);
        assert_eq!(s.i_frame_count, 0);
        assert_eq!(s.mean_gop, 0.0);
    }

    #[test]
    fn test_gop_stats_single_iframe() {
        let pts = vec![IFramePoint::new(0, IFrameReason::StreamStart)];
        let s = gop_stats(&pts, 60);
        assert_eq!(s.i_frame_count, 1);
        assert_eq!(s.max_gop, 60);
    }

    #[test]
    fn test_gop_stats_counts_reasons() {
        let pts = vec![
            IFramePoint::new(0, IFrameReason::StreamStart),
            IFramePoint::new(30, IFrameReason::SceneCut),
            IFramePoint::new(60, IFrameReason::Periodic),
            IFramePoint::new(90, IFrameReason::SceneCut),
        ];
        let s = gop_stats(&pts, 120);
        assert_eq!(s.scene_cut_count, 2);
        assert_eq!(s.periodic_count, 1);
        assert_eq!(s.i_frame_count, 4);
    }

    #[test]
    fn test_gop_stats_mean_gop() {
        let pts = vec![
            IFramePoint::new(0, IFrameReason::StreamStart),
            IFramePoint::new(60, IFrameReason::Periodic),
        ];
        // GOPs: [0..60] = 60, [60..120] = 60
        let s = gop_stats(&pts, 120);
        assert!((s.mean_gop - 60.0).abs() < 1e-9);
    }

    #[test]
    fn test_full_pipeline_cuts_plus_fixed() {
        let config = GopConfig::new(10, 300, 90).with_policy(GopPlacementPolicy::CutsPlusFixed);
        let cuts = vec![make_cut(45, 0.85), make_cut(180, 0.92)];
        let placements = compute_i_frame_placements(300, &cuts, &config);

        // Check sorted
        for w in placements.windows(2) {
            assert!(w[0].frame < w[1].frame, "placements should be sorted");
        }

        let stats = gop_stats(&placements, 300);
        assert!(
            stats.max_gop <= 300,
            "max GOP {max} should be <= 300",
            max = stats.max_gop
        );
        assert!(
            stats.min_gop >= 10 || stats.i_frame_count == 1,
            "min GOP {min} should be >= 10",
            min = stats.min_gop
        );
    }
}
