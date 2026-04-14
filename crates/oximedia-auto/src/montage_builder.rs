//! Montage and highlight reel builder.
//!
//! Constructs a montage (highlight reel) from a set of scored clips by:
//!
//! - **Score-based selection**: rank clips by importance, break ties by
//!   temporal position (earlier preferred).
//! - **Duration packing**: greedily fill a target duration, respecting per-clip
//!   minimum/maximum lengths.
//! - **Variety enforcement**: optional limit on consecutive clips from the same
//!   source segment to avoid repetitive editing.
//! - **Transition assignment**: assign a `TransitionKind` to each inter-clip
//!   boundary based on the score delta between adjacent clips.
//! - **Ordering strategies**: chronological, score-descending, or dramatic-arc
//!   (low → high with a climax near the end).
//!
//! All computations are pure arithmetic — no external dependencies beyond
//! `oximedia_core`.
//!
//! # Example
//!
//! ```
//! use oximedia_auto::montage_builder::{MontageBuilder, MontageConfig, ClipCandidate};
//! use oximedia_core::{Rational, Timestamp};
//!
//! let config = MontageConfig::default();
//! let builder = MontageBuilder::new(config);
//!
//! let tb = Rational::new(1, 1000);
//! let clips = vec![
//!     ClipCandidate::new(0, Timestamp::new(0, tb), Timestamp::new(5000, tb), 0.8),
//!     ClipCandidate::new(0, Timestamp::new(10_000, tb), Timestamp::new(15_000, tb), 0.6),
//! ];
//!
//! let montage = builder.build(&clips).expect("build montage");
//! assert!(!montage.selected_clips.is_empty());
//! ```

#![allow(dead_code)]

use crate::error::{AutoError, AutoResult};
use oximedia_core::Timestamp;

// ─── Transition types ─────────────────────────────────────────────────────────

/// The kind of transition recommended between two adjacent montage clips.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionKind {
    /// Hard cut — no transition.
    Cut,
    /// Short dissolve (≤0.5 s).
    Dissolve,
    /// Wipe from left to right.
    Wipe,
    /// Fade to black then in.
    FadeThrough,
    /// Zoom-based punch-cut.
    ZoomCut,
}

impl TransitionKind {
    /// Typical duration of this transition in milliseconds.
    #[must_use]
    pub const fn typical_duration_ms(&self) -> i64 {
        match self {
            Self::Cut => 0,
            Self::Dissolve => 400,
            Self::Wipe => 500,
            Self::FadeThrough => 800,
            Self::ZoomCut => 100,
        }
    }

    /// Human-readable label.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Cut => "cut",
            Self::Dissolve => "dissolve",
            Self::Wipe => "wipe",
            Self::FadeThrough => "fade-through",
            Self::ZoomCut => "zoom-cut",
        }
    }
}

// ─── Ordering strategy ────────────────────────────────────────────────────────

/// How selected clips are ordered in the final montage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OrderingStrategy {
    /// Preserve original source order (chronological).
    #[default]
    Chronological,
    /// Highest-score clips first.
    ScoreDescending,
    /// Dramatic arc: start moderate, dip slightly, build to climax, end strong.
    DramaticArc,
    /// Alternating high/low intensity.
    Interleaved,
}

// ─── ClipCandidate ────────────────────────────────────────────────────────────

/// A scored clip that may be included in the montage.
#[derive(Debug, Clone)]
pub struct ClipCandidate {
    /// Source media identifier (e.g. file index or track id).
    pub source_id: u32,
    /// In-point in the source timeline.
    pub start: Timestamp,
    /// Out-point in the source timeline.
    pub end: Timestamp,
    /// Importance score (0.0 – 1.0; higher = more important).
    pub score: f64,
    /// Optional human-readable description.
    pub description: Option<String>,
}

impl ClipCandidate {
    /// Create a new clip candidate.
    #[must_use]
    pub fn new(source_id: u32, start: Timestamp, end: Timestamp, score: f64) -> Self {
        Self {
            source_id,
            start,
            end,
            score: score.clamp(0.0, 1.0),
            description: None,
        }
    }

    /// Set a description for this candidate.
    #[must_use]
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Duration of this clip in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> i64 {
        (self.end.pts - self.start.pts).max(0)
    }
}

// ─── SelectedClip ─────────────────────────────────────────────────────────────

/// A clip that has been selected and positioned in the output montage.
#[derive(Debug, Clone)]
pub struct SelectedClip {
    /// Source clip that was chosen.
    pub source: ClipCandidate,
    /// Output position start (cumulative, ms).
    pub output_start_ms: i64,
    /// Output position end (cumulative, ms).
    pub output_end_ms: i64,
    /// Transition that follows this clip (None for the last clip).
    pub transition: Option<TransitionKind>,
}

impl SelectedClip {
    /// Duration of this clip in the output timeline (ms).
    #[must_use]
    pub fn output_duration_ms(&self) -> i64 {
        self.output_end_ms - self.output_start_ms
    }
}

// ─── MontageConfig ────────────────────────────────────────────────────────────

/// Configuration for the montage builder.
#[derive(Debug, Clone)]
pub struct MontageConfig {
    /// Target total duration of the montage in milliseconds.
    pub target_duration_ms: i64,
    /// Tolerance around the target duration (±ms).  The builder stops packing
    /// once within this tolerance.
    pub duration_tolerance_ms: i64,
    /// Minimum clip duration to include (shorter clips are skipped).
    pub min_clip_duration_ms: i64,
    /// Maximum clip duration to include (longer clips are trimmed).
    pub max_clip_duration_ms: i64,
    /// Ordering strategy for the final clip list.
    pub ordering: OrderingStrategy,
    /// Minimum importance score for a clip to be considered.
    pub min_score: f64,
    /// Maximum number of consecutive clips from the same source.
    pub max_consecutive_same_source: usize,
    /// Whether to assign transitions between clips.
    pub assign_transitions: bool,
    /// Score delta above which a `Wipe` transition is used.
    pub wipe_score_delta_threshold: f64,
    /// Score delta above which a `Dissolve` transition is used.
    pub dissolve_score_delta_threshold: f64,
}

impl Default for MontageConfig {
    fn default() -> Self {
        Self {
            target_duration_ms: 60_000,   // 60-second montage
            duration_tolerance_ms: 2_000, // ±2 s
            min_clip_duration_ms: 1_000,
            max_clip_duration_ms: 8_000,
            ordering: OrderingStrategy::default(),
            min_score: 0.30,
            max_consecutive_same_source: 3,
            assign_transitions: true,
            wipe_score_delta_threshold: 0.40,
            dissolve_score_delta_threshold: 0.20,
        }
    }
}

impl MontageConfig {
    /// Create a preset for a short social-media montage (~15 s).
    #[must_use]
    pub fn social_short() -> Self {
        Self {
            target_duration_ms: 15_000,
            duration_tolerance_ms: 1_000,
            min_clip_duration_ms: 500,
            max_clip_duration_ms: 3_000,
            ordering: OrderingStrategy::DramaticArc,
            min_score: 0.50,
            max_consecutive_same_source: 2,
            ..Self::default()
        }
    }

    /// Create a preset for a standard highlight reel (~90 s).
    #[must_use]
    pub fn highlight_reel() -> Self {
        Self {
            target_duration_ms: 90_000,
            duration_tolerance_ms: 5_000,
            min_clip_duration_ms: 2_000,
            max_clip_duration_ms: 12_000,
            ordering: OrderingStrategy::DramaticArc,
            min_score: 0.25,
            max_consecutive_same_source: 4,
            ..Self::default()
        }
    }

    /// Validate configuration values.
    pub fn validate(&self) -> AutoResult<()> {
        if self.target_duration_ms <= 0 {
            return Err(AutoError::InvalidDuration {
                duration_ms: self.target_duration_ms,
            });
        }
        if self.min_clip_duration_ms <= 0 {
            return Err(AutoError::invalid_parameter(
                "min_clip_duration_ms",
                "must be positive",
            ));
        }
        if self.max_clip_duration_ms < self.min_clip_duration_ms {
            return Err(AutoError::invalid_parameter(
                "max_clip_duration_ms",
                "must be >= min_clip_duration_ms",
            ));
        }
        if !(0.0..=1.0).contains(&self.min_score) {
            return Err(AutoError::InvalidThreshold {
                threshold: self.min_score,
                min: 0.0,
                max: 1.0,
            });
        }
        if self.max_consecutive_same_source == 0 {
            return Err(AutoError::invalid_parameter(
                "max_consecutive_same_source",
                "must be at least 1",
            ));
        }
        Ok(())
    }
}

// ─── MontageResult ────────────────────────────────────────────────────────────

/// The assembled montage returned by the builder.
#[derive(Debug, Clone)]
pub struct MontageResult {
    /// Clips selected and ordered for the montage.
    pub selected_clips: Vec<SelectedClip>,
    /// Total output duration in milliseconds (excluding any trailing transitions).
    pub total_duration_ms: i64,
    /// Number of candidates that were rejected (score, duration, or variety).
    pub rejected_count: usize,
    /// Whether the montage reached the target duration within tolerance.
    pub target_reached: bool,
}

impl MontageResult {
    /// Average clip score in the montage.
    #[must_use]
    pub fn average_score(&self) -> f64 {
        if self.selected_clips.is_empty() {
            return 0.0;
        }
        let total: f64 = self.selected_clips.iter().map(|c| c.source.score).sum();
        total / self.selected_clips.len() as f64
    }

    /// Number of distinct sources used in the montage.
    #[must_use]
    pub fn distinct_source_count(&self) -> usize {
        let mut ids: Vec<u32> = self
            .selected_clips
            .iter()
            .map(|c| c.source.source_id)
            .collect();
        ids.sort_unstable();
        ids.dedup();
        ids.len()
    }
}

// ─── MontageBuilder ───────────────────────────────────────────────────────────

/// Builds a montage from a pool of scored clip candidates.
pub struct MontageBuilder {
    config: MontageConfig,
}

impl MontageBuilder {
    /// Create a new builder with the given configuration.
    #[must_use]
    pub fn new(config: MontageConfig) -> Self {
        Self { config }
    }

    /// Create a builder with default configuration.
    #[must_use]
    pub fn default_builder() -> Self {
        Self::new(MontageConfig::default())
    }

    /// Build a montage from `candidates`.
    ///
    /// Steps:
    /// 1. Filter by minimum score and clip duration.
    /// 2. Sort by score (descending) for greedy selection.
    /// 3. Pack clips until the target duration is reached, enforcing variety.
    /// 4. Re-order using the configured [`OrderingStrategy`].
    /// 5. Assign transitions.
    /// 6. Compute output positions.
    ///
    /// # Errors
    ///
    /// Returns [`AutoError::InsufficientData`] when no candidates pass the
    /// filters, or [`AutoError::InvalidParameter`] when the config is invalid.
    pub fn build(&self, candidates: &[ClipCandidate]) -> AutoResult<MontageResult> {
        self.config.validate()?;

        if candidates.is_empty() {
            return Err(AutoError::insufficient_data(
                "No clip candidates provided for montage building",
            ));
        }

        // 1. Filter by quality gates
        let filtered: Vec<&ClipCandidate> = candidates
            .iter()
            .filter(|c| {
                c.score >= self.config.min_score
                    && c.duration_ms() >= self.config.min_clip_duration_ms
            })
            .collect();

        let _rejected_by_filter = candidates.len() - filtered.len();

        if filtered.is_empty() {
            return Err(AutoError::insufficient_data(
                "All clip candidates were filtered out (too short or low score)",
            ));
        }

        // 2. Sort by score descending for greedy packing
        let mut by_score: Vec<&ClipCandidate> = filtered.clone();
        by_score.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.start.pts.cmp(&b.start.pts)) // earlier first on tie
        });

        // 3. Greedy pack
        let mut packed: Vec<ClipCandidate> = Vec::new();
        let mut cumulative_ms: i64 = 0;
        let mut consecutive_same: usize = 0;
        let mut last_source: Option<u32> = None;

        for clip in &by_score {
            let remaining = self.config.target_duration_ms - cumulative_ms;
            if remaining <= 0 {
                break;
            }

            // Enforce variety
            if let Some(ls) = last_source {
                if ls == clip.source_id {
                    consecutive_same += 1;
                    if consecutive_same >= self.config.max_consecutive_same_source {
                        continue;
                    }
                } else {
                    consecutive_same = 0;
                }
            }

            // Trim clip to fit max_clip_duration_ms and remaining space
            let raw_duration = clip.duration_ms();
            let capped = raw_duration
                .min(self.config.max_clip_duration_ms)
                .min(remaining);

            // Don't use if trimmed below minimum
            if capped < self.config.min_clip_duration_ms {
                continue;
            }

            // Build a (possibly trimmed) copy
            let out_end_pts = clip.start.pts + capped;
            let tb = clip.start.timebase;
            let trimmed = ClipCandidate {
                source_id: clip.source_id,
                start: clip.start,
                end: Timestamp::new(out_end_pts, tb),
                score: clip.score,
                description: clip.description.clone(),
            };

            last_source = Some(trimmed.source_id);
            cumulative_ms += capped;
            packed.push(trimmed);

            // Stop early if within tolerance of the target
            let diff = (cumulative_ms - self.config.target_duration_ms).abs();
            if diff <= self.config.duration_tolerance_ms {
                break;
            }
        }

        let rejected_count = candidates.len() - packed.len();
        let target_reached = {
            let diff = (cumulative_ms - self.config.target_duration_ms).abs();
            diff <= self.config.duration_tolerance_ms
        };

        // 4. Re-order
        let ordered = self.apply_ordering(packed);

        // 5. Assign transitions & 6. Compute output positions
        let selected_clips = self.build_selected_clips(ordered);
        let total_duration_ms = selected_clips.last().map(|c| c.output_end_ms).unwrap_or(0);

        Ok(MontageResult {
            selected_clips,
            total_duration_ms,
            rejected_count,
            target_reached,
        })
    }

    /// Get the current configuration.
    #[must_use]
    pub const fn config(&self) -> &MontageConfig {
        &self.config
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Re-order packed clips according to [`MontageConfig::ordering`].
    fn apply_ordering(&self, mut clips: Vec<ClipCandidate>) -> Vec<ClipCandidate> {
        match self.config.ordering {
            OrderingStrategy::Chronological => {
                clips.sort_by_key(|c| c.start.pts);
            }
            OrderingStrategy::ScoreDescending => {
                clips.sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            OrderingStrategy::DramaticArc => {
                self.apply_dramatic_arc(&mut clips);
            }
            OrderingStrategy::Interleaved => {
                self.apply_interleaved_ordering(&mut clips);
            }
        }
        clips
    }

    /// Place clips in a classic three-act dramatic arc:
    ///  Act 1 (first ~25%): medium scores — establish context
    ///  Act 2 (middle 50%): lower → rising scores — build tension
    ///  Act 3 (last 25%): peak climax + short resolution
    fn apply_dramatic_arc(&self, clips: &mut Vec<ClipCandidate>) {
        let n = clips.len();
        if n <= 2 {
            return; // Not enough clips to arrange
        }

        // Sort by score ascending so we can assign positions
        clips.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let act1_end = n / 4;
        let act3_start = n * 3 / 4;

        // Act 1: medium range (take from the middle of the sorted list)
        let act1_range: Vec<ClipCandidate> = clips.drain(n / 2 - act1_end..n / 2).collect();
        // Act 3: highest scoring clips
        let act3_range: Vec<ClipCandidate> =
            clips.drain(clips.len() - (n - act3_start)..).collect();

        // Remaining (lowest scores) form act 2
        let act2: Vec<ClipCandidate> = std::mem::take(clips);

        // Reassemble: act1 → act2 → act3 (climax)
        clips.extend(act1_range);
        clips.extend(act2);
        clips.extend(act3_range);
    }

    /// Alternate high-intensity and low-intensity clips.
    fn apply_interleaved_ordering(&self, clips: &mut Vec<ClipCandidate>) {
        clips.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let n = clips.len();
        let (evens, odds): (Vec<_>, Vec<_>) =
            clips.drain(..).enumerate().partition(|(i, _)| i % 2 == 0);

        let mut result: Vec<ClipCandidate> = Vec::with_capacity(n);
        let mut ei = evens.into_iter().map(|(_, c)| c);
        let mut oi = odds.into_iter().map(|(_, c)| c);

        loop {
            match (ei.next(), oi.next()) {
                (Some(e), Some(o)) => {
                    result.push(e);
                    result.push(o);
                }
                (Some(e), None) => {
                    result.push(e);
                }
                (None, Some(o)) => {
                    result.push(o);
                }
                (None, None) => break,
            }
        }

        *clips = result;
    }

    /// Build the final `SelectedClip` list with output positions and transitions.
    fn build_selected_clips(&self, ordered: Vec<ClipCandidate>) -> Vec<SelectedClip> {
        let n = ordered.len();
        let mut result = Vec::with_capacity(n);
        let mut cursor_ms: i64 = 0;

        for (i, clip) in ordered.into_iter().enumerate() {
            let duration = clip.duration_ms();
            let out_start = cursor_ms;
            let out_end = cursor_ms + duration;

            // Assign transition: None for last clip
            let transition = if i + 1 < n && self.config.assign_transitions {
                Some(TransitionKind::Cut) // placeholder; refined below
            } else {
                None
            };

            result.push(SelectedClip {
                source: clip,
                output_start_ms: out_start,
                output_end_ms: out_end,
                transition,
            });

            cursor_ms = out_end;
        }

        // Refine transitions based on score delta between adjacent clips
        if self.config.assign_transitions {
            self.refine_transitions(&mut result);
        }

        result
    }

    /// Assign `TransitionKind` based on the score delta between adjacent clips.
    fn refine_transitions(&self, clips: &mut Vec<SelectedClip>) {
        let n = clips.len();
        for i in 0..n.saturating_sub(1) {
            let score_a = clips[i].source.score;
            let score_b = clips[i + 1].source.score;
            let delta = (score_b - score_a).abs();

            clips[i].transition = Some(if delta >= self.config.wipe_score_delta_threshold {
                TransitionKind::Wipe
            } else if delta >= self.config.dissolve_score_delta_threshold {
                TransitionKind::Dissolve
            } else {
                TransitionKind::Cut
            });
        }
    }
}

impl Default for MontageBuilder {
    fn default() -> Self {
        Self::default_builder()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::Rational;

    fn tb() -> Rational {
        Rational::new(1, 1000)
    }

    fn ts(ms: i64) -> Timestamp {
        Timestamp::new(ms, tb())
    }

    fn make_clip(source_id: u32, start_ms: i64, end_ms: i64, score: f64) -> ClipCandidate {
        ClipCandidate::new(source_id, ts(start_ms), ts(end_ms), score)
    }

    fn default_candidates() -> Vec<ClipCandidate> {
        vec![
            make_clip(0, 0, 4_000, 0.9),
            make_clip(0, 10_000, 14_000, 0.8),
            make_clip(1, 20_000, 23_000, 0.7),
            make_clip(1, 30_000, 33_000, 0.6),
            make_clip(2, 40_000, 43_000, 0.5),
            make_clip(2, 50_000, 53_000, 0.4),
        ]
    }

    #[test]
    fn test_default_config_is_valid() {
        assert!(MontageConfig::default().validate().is_ok());
    }

    #[test]
    fn test_social_short_preset_is_valid() {
        assert!(MontageConfig::social_short().validate().is_ok());
    }

    #[test]
    fn test_highlight_reel_preset_is_valid() {
        assert!(MontageConfig::highlight_reel().validate().is_ok());
    }

    #[test]
    fn test_invalid_target_duration() {
        let mut cfg = MontageConfig::default();
        cfg.target_duration_ms = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_invalid_min_clip_duration() {
        let mut cfg = MontageConfig::default();
        cfg.min_clip_duration_ms = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_max_less_than_min_clip_duration() {
        let mut cfg = MontageConfig::default();
        cfg.min_clip_duration_ms = 3_000;
        cfg.max_clip_duration_ms = 1_000;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_empty_candidates_error() {
        let builder = MontageBuilder::default();
        assert!(builder.build(&[]).is_err());
    }

    #[test]
    fn test_all_filtered_out_error() {
        let builder = MontageBuilder::new(MontageConfig {
            min_score: 0.99,
            ..MontageConfig::default()
        });
        let candidates = vec![make_clip(0, 0, 5_000, 0.10)];
        assert!(builder.build(&candidates).is_err());
    }

    #[test]
    fn test_basic_build_produces_clips() {
        let builder = MontageBuilder::default();
        let candidates = default_candidates();
        let result = builder.build(&candidates).expect("build should succeed");
        assert!(!result.selected_clips.is_empty());
    }

    #[test]
    fn test_total_duration_within_tolerance() {
        let config = MontageConfig {
            target_duration_ms: 12_000,
            duration_tolerance_ms: 4_000,
            min_clip_duration_ms: 1_000,
            max_clip_duration_ms: 4_000,
            min_score: 0.0,
            ..MontageConfig::default()
        };
        let builder = MontageBuilder::new(config.clone());
        let candidates = default_candidates();
        let result = builder.build(&candidates).expect("build should succeed");
        let tolerance = config.duration_tolerance_ms;
        let diff = (result.total_duration_ms - config.target_duration_ms).abs();
        assert!(
            diff <= tolerance + 4_000, // allow for packing overshoot
            "total={} target={} diff={}",
            result.total_duration_ms,
            config.target_duration_ms,
            diff
        );
    }

    #[test]
    fn test_max_clip_duration_capped() {
        let config = MontageConfig {
            max_clip_duration_ms: 2_000,
            min_score: 0.0,
            ..MontageConfig::default()
        };
        let builder = MontageBuilder::new(config);
        let candidates = vec![
            make_clip(0, 0, 10_000, 0.8), // long clip → should be capped
        ];
        let result = builder.build(&candidates).expect("build should succeed");
        for clip in &result.selected_clips {
            assert!(
                clip.output_duration_ms() <= 2_000,
                "clip duration {} exceeded cap",
                clip.output_duration_ms()
            );
        }
    }

    #[test]
    fn test_transitions_assigned() {
        let config = MontageConfig {
            assign_transitions: true,
            target_duration_ms: 20_000,
            min_score: 0.0,
            ..MontageConfig::default()
        };
        let builder = MontageBuilder::new(config);
        let candidates = default_candidates();
        let result = builder.build(&candidates).expect("build should succeed");
        // All clips except the last should have a transition
        let n = result.selected_clips.len();
        if n > 1 {
            for (i, clip) in result.selected_clips[..n - 1].iter().enumerate() {
                assert!(
                    clip.transition.is_some(),
                    "clip {i} should have a transition"
                );
            }
            assert!(
                result.selected_clips[n - 1].transition.is_none(),
                "last clip should have no trailing transition"
            );
        }
    }

    #[test]
    fn test_ordering_chronological_preserves_source_order() {
        let config = MontageConfig {
            ordering: OrderingStrategy::Chronological,
            min_score: 0.0,
            target_duration_ms: 30_000,
            ..MontageConfig::default()
        };
        let builder = MontageBuilder::new(config);
        // Provide clips in reverse score order but increasing time order
        let candidates = vec![
            make_clip(0, 0, 4_000, 0.9),
            make_clip(0, 10_000, 14_000, 0.7),
            make_clip(0, 20_000, 24_000, 0.5),
        ];
        let result = builder.build(&candidates).expect("build should succeed");
        let starts: Vec<i64> = result
            .selected_clips
            .iter()
            .map(|c| c.source.start.pts)
            .collect();
        let mut sorted_starts = starts.clone();
        sorted_starts.sort_unstable();
        assert_eq!(starts, sorted_starts, "chronological order violated");
    }

    #[test]
    fn test_variety_limits_consecutive_same_source() {
        let config = MontageConfig {
            max_consecutive_same_source: 1,
            target_duration_ms: 30_000,
            min_clip_duration_ms: 1_000,
            max_clip_duration_ms: 5_000,
            min_score: 0.0,
            ..MontageConfig::default()
        };
        let builder = MontageBuilder::new(config);
        // Five clips from source 0, one from source 1
        let mut candidates = vec![
            make_clip(0, 0, 3_000, 0.9),
            make_clip(0, 5_000, 8_000, 0.85),
            make_clip(0, 10_000, 13_000, 0.80),
            make_clip(0, 15_000, 18_000, 0.75),
            make_clip(0, 20_000, 23_000, 0.70),
            make_clip(1, 25_000, 28_000, 0.65),
        ];
        // Sort so source-0 candidates are selected first
        candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let result = builder.build(&candidates).expect("build should succeed");
        // With max_consecutive=1, after one source-0 clip the builder cannot
        // immediately take another source-0 clip without breaking the rule, so
        // consecutive runs should be at most 1.
        let mut consecutive = 1usize;
        let mut prev_source = result.selected_clips[0].source.source_id;
        for clip in result.selected_clips.iter().skip(1) {
            if clip.source.source_id == prev_source {
                consecutive += 1;
                assert!(
                    consecutive <= 1,
                    "consecutive same-source exceeded limit: {}",
                    consecutive
                );
            } else {
                consecutive = 1;
                prev_source = clip.source.source_id;
            }
        }
    }

    #[test]
    fn test_dramatic_arc_ordering_ends_high() {
        let config = MontageConfig {
            ordering: OrderingStrategy::DramaticArc,
            target_duration_ms: 60_000,
            min_score: 0.0,
            min_clip_duration_ms: 1_000,
            max_clip_duration_ms: 10_000,
            ..MontageConfig::default()
        };
        let builder = MontageBuilder::new(config);
        // 8 clips with varying scores
        let candidates: Vec<ClipCandidate> = (0..8)
            .map(|i| make_clip(0, i * 8_000, (i + 1) * 8_000, (i as f64 + 1.0) / 8.0))
            .collect();
        let result = builder.build(&candidates).expect("build should succeed");
        let n = result.selected_clips.len();
        if n >= 4 {
            // The last quarter should contain the highest-scoring clips
            let last_quarter_avg: f64 = result.selected_clips[n * 3 / 4..]
                .iter()
                .map(|c| c.source.score)
                .sum::<f64>()
                / (n - n * 3 / 4) as f64;
            let first_half_avg: f64 = result.selected_clips[..n / 2]
                .iter()
                .map(|c| c.source.score)
                .sum::<f64>()
                / (n / 2) as f64;
            assert!(
                last_quarter_avg >= first_half_avg,
                "dramatic arc: last quarter avg={last_quarter_avg} should >= first half avg={first_half_avg}"
            );
        }
    }

    #[test]
    fn test_average_score_and_distinct_sources() {
        let config = MontageConfig {
            target_duration_ms: 20_000,
            min_score: 0.0,
            min_clip_duration_ms: 1_000,
            max_clip_duration_ms: 5_000,
            ..MontageConfig::default()
        };
        let builder = MontageBuilder::new(config);
        let candidates = vec![
            make_clip(0, 0, 4_000, 0.8),
            make_clip(1, 5_000, 9_000, 0.6),
            make_clip(2, 10_000, 14_000, 0.4),
        ];
        let result = builder.build(&candidates).expect("build should succeed");
        let avg = result.average_score();
        assert!(
            (0.0..=1.0).contains(&avg),
            "average score out of range: {avg}"
        );
        assert!(
            result.distinct_source_count() >= 1,
            "should have at least one source"
        );
    }

    #[test]
    fn test_transition_wipe_on_large_delta() {
        let config = MontageConfig {
            target_duration_ms: 10_000,
            min_score: 0.0,
            min_clip_duration_ms: 1_000,
            max_clip_duration_ms: 5_000,
            assign_transitions: true,
            wipe_score_delta_threshold: 0.30,
            dissolve_score_delta_threshold: 0.10,
            ..MontageConfig::default()
        };
        let builder = MontageBuilder::new(config);
        let candidates = vec![
            make_clip(0, 0, 4_000, 0.10),     // low score
            make_clip(1, 5_000, 9_000, 0.95), // high score → large delta
        ];
        let result = builder.build(&candidates).expect("build should succeed");
        if result.selected_clips.len() >= 2 {
            // The transition between the two clips should be a Wipe
            let transition = result.selected_clips[0].transition;
            assert_eq!(
                transition,
                Some(TransitionKind::Wipe),
                "expected Wipe transition for large score delta"
            );
        }
    }

    #[test]
    fn test_clip_candidate_with_description() {
        let c = make_clip(0, 0, 3_000, 0.7).with_description("Opening shot");
        assert_eq!(c.description.as_deref(), Some("Opening shot"));
    }

    #[test]
    fn test_transition_kind_duration() {
        assert_eq!(TransitionKind::Cut.typical_duration_ms(), 0);
        assert!(TransitionKind::Dissolve.typical_duration_ms() > 0);
        assert!(
            TransitionKind::FadeThrough.typical_duration_ms()
                > TransitionKind::Dissolve.typical_duration_ms()
        );
    }
}
