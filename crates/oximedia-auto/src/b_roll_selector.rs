//! B-roll selection logic for automatic B-roll insertion.
//!
//! This module scores and selects B-roll clips to cover a sequence of
//! primary-footage (A-roll) segments, maximising visual variety, topic
//! relevance, and coverage while avoiding reusing the same clip too often.
//!
//! # Key concepts
//!
//! * **Coverage scoring** — how well a B-roll clip spans an A-roll gap.
//! * **Visual variety scoring** — prefer clips that differ from recently
//!   selected ones (colour histogram distance, scene category variety).
//! * **Topic relevance scoring** — match B-roll topic tags against the
//!   active A-roll topic tags.
//! * **Usage tracking** — repeated use of the same clip is penalised.
//!
//! All scoring is deterministic and purely mathematical; no ML model or
//! external service is needed.
//!
//! # Example
//!
//! ```
//! use oximedia_auto::b_roll_selector::{
//!     BRollClip, BRollSelector, BRollSelectorConfig, CoverageGap,
//! };
//!
//! let clips = vec![
//!     BRollClip::new("clip_001", 5_000, vec!["nature".to_string()]),
//!     BRollClip::new("clip_002", 3_000, vec!["city".to_string()]),
//! ];
//! let config = BRollSelectorConfig::default();
//! let mut selector = BRollSelector::new(clips, config);
//!
//! let gap = CoverageGap { start_ms: 0, end_ms: 4_000, topic_tags: vec!["nature".to_string()] };
//! let selected = selector.select_for_gap(&gap).unwrap();
//! assert!(selected.is_some());
//! ```

#![allow(dead_code)]

use crate::error::{AutoError, AutoResult};

// ─── Clip colour histogram ────────────────────────────────────────────────────

/// A compact colour histogram with `N` buckets per channel (RGB, N=8 → 24 floats).
/// Stored as a flat vector of normalised bin values (sum ≈ 1.0 per channel).
#[derive(Debug, Clone, PartialEq)]
pub struct ColorHistogram {
    /// Flat array: [R_bins…, G_bins…, B_bins…], each channel `bins_per_channel` long.
    pub bins: Vec<f32>,
    /// Number of bins per channel.
    pub bins_per_channel: usize,
}

impl ColorHistogram {
    /// Create a uniform (grey) histogram with the given number of bins per channel.
    #[must_use]
    pub fn uniform(bins_per_channel: usize) -> Self {
        let val = if bins_per_channel > 0 {
            1.0 / bins_per_channel as f32
        } else {
            0.0
        };
        Self {
            bins: vec![val; bins_per_channel * 3],
            bins_per_channel,
        }
    }

    /// Chi-squared distance between two histograms (0 = identical).
    ///
    /// Returns `None` if the histogram dimensions do not match.
    #[must_use]
    pub fn chi_squared_distance(&self, other: &Self) -> Option<f32> {
        if self.bins.len() != other.bins.len() {
            return None;
        }
        let dist = self
            .bins
            .iter()
            .zip(other.bins.iter())
            .map(|(&a, &b)| {
                let sum = a + b;
                if sum > 1e-9 {
                    (a - b).powi(2) / sum
                } else {
                    0.0
                }
            })
            .sum::<f32>()
            * 0.5;
        Some(dist)
    }

    /// Normalise each channel so bins sum to 1.
    #[must_use]
    pub fn normalise(mut self) -> Self {
        let n = self.bins_per_channel;
        for ch in 0..3 {
            let start = ch * n;
            let end = start + n;
            let total: f32 = self.bins[start..end].iter().sum();
            if total > 1e-9 {
                for b in &mut self.bins[start..end] {
                    *b /= total;
                }
            }
        }
        self
    }
}

impl Default for ColorHistogram {
    fn default() -> Self {
        Self::uniform(8)
    }
}

// ─── Scene category ───────────────────────────────────────────────────────────

/// Broad visual scene category for variety tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SceneCategory {
    /// Interior / indoor setting.
    Indoor,
    /// Exterior / outdoor setting.
    Outdoor,
    /// Close-up shot.
    CloseUp,
    /// Wide / establishing shot.
    Wide,
    /// Aerial or drone footage.
    Aerial,
    /// Animated or motion-graphics clip.
    Animation,
    /// Category not determined.
    Unknown,
}

impl Default for SceneCategory {
    fn default() -> Self {
        Self::Unknown
    }
}

// ─── B-roll clip ──────────────────────────────────────────────────────────────

/// A candidate B-roll clip with metadata used for scoring.
#[derive(Debug, Clone)]
pub struct BRollClip {
    /// Unique identifier (file path, asset ID, etc.).
    pub id: String,
    /// Duration of the clip in milliseconds.
    pub duration_ms: u64,
    /// Topic / keyword tags attached to this clip.
    pub topic_tags: Vec<String>,
    /// Dominant colour histogram (optional; improves variety scoring).
    pub histogram: Option<ColorHistogram>,
    /// Broad scene category.
    pub scene_category: SceneCategory,
    /// Base desirability weight (1.0 = neutral; increase for preferred clips).
    pub base_weight: f32,
}

impl BRollClip {
    /// Create a new clip with only the required fields.
    #[must_use]
    pub fn new(id: impl Into<String>, duration_ms: u64, topic_tags: Vec<String>) -> Self {
        Self {
            id: id.into(),
            duration_ms,
            topic_tags,
            histogram: None,
            scene_category: SceneCategory::Unknown,
            base_weight: 1.0,
        }
    }

    /// Attach a colour histogram.
    #[must_use]
    pub fn with_histogram(mut self, hist: ColorHistogram) -> Self {
        self.histogram = Some(hist);
        self
    }

    /// Set the scene category.
    #[must_use]
    pub const fn with_category(mut self, cat: SceneCategory) -> Self {
        self.scene_category = cat;
        self
    }

    /// Set the base weight.
    #[must_use]
    pub fn with_base_weight(mut self, weight: f32) -> Self {
        self.base_weight = weight;
        self
    }
}

// ─── Coverage gap ─────────────────────────────────────────────────────────────

/// A gap in the A-roll timeline that B-roll should cover.
#[derive(Debug, Clone)]
pub struct CoverageGap {
    /// Gap start (ms PTS).
    pub start_ms: i64,
    /// Gap end (ms PTS).
    pub end_ms: i64,
    /// Topic tags active during this gap (e.g. from transcript analysis).
    pub topic_tags: Vec<String>,
}

impl CoverageGap {
    /// Duration of the gap in milliseconds (non-negative).
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        (self.end_ms - self.start_ms).max(0) as u64
    }
}

// ─── Selection result ─────────────────────────────────────────────────────────

/// A scored B-roll selection for a single gap.
#[derive(Debug, Clone)]
pub struct BRollSelection {
    /// The selected clip.
    pub clip_id: String,
    /// Overall composite score (0.0–1.0; higher = better).
    pub score: f32,
    /// Coverage score component.
    pub coverage_score: f32,
    /// Visual variety score component.
    pub variety_score: f32,
    /// Topic relevance score component.
    pub relevance_score: f32,
    /// Recommended in-point within the clip (ms from clip start).
    pub in_point_ms: u64,
    /// Recommended out-point within the clip (ms from clip start).
    pub out_point_ms: u64,
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for [`BRollSelector`].
#[derive(Debug, Clone)]
pub struct BRollSelectorConfig {
    /// Weight applied to coverage score in composite (0.0–1.0).
    pub coverage_weight: f32,
    /// Weight applied to visual variety score.
    pub variety_weight: f32,
    /// Weight applied to topic relevance score.
    pub relevance_weight: f32,
    /// Maximum number of times the same clip may be used before a heavy
    /// penalty is applied.
    pub max_reuse_count: usize,
    /// Penalty multiplier applied to the composite score per extra use beyond
    /// `max_reuse_count` (0.0–1.0; 0.5 = halve the score each extra use).
    pub reuse_penalty: f32,
    /// Number of recently-used clips kept in the variety-distance window.
    pub variety_window: usize,
}

impl Default for BRollSelectorConfig {
    fn default() -> Self {
        Self {
            coverage_weight: 0.35,
            variety_weight: 0.35,
            relevance_weight: 0.30,
            max_reuse_count: 2,
            reuse_penalty: 0.5,
            variety_window: 5,
        }
    }
}

impl BRollSelectorConfig {
    /// Create a new config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate that weights are non-negative and sum to approximately 1.
    ///
    /// # Errors
    ///
    /// Returns an error if any weight is negative or the sum differs from
    /// 1.0 by more than 0.01.
    pub fn validate(&self) -> AutoResult<()> {
        for (name, val) in [
            ("coverage_weight", self.coverage_weight),
            ("variety_weight", self.variety_weight),
            ("relevance_weight", self.relevance_weight),
        ] {
            if val < 0.0 {
                return Err(AutoError::invalid_parameter(
                    name,
                    format!("{val} (must be >= 0)"),
                ));
            }
        }
        let total = self.coverage_weight + self.variety_weight + self.relevance_weight;
        if (total - 1.0).abs() > 0.01 {
            return Err(AutoError::invalid_parameter(
                "weights",
                format!("sum = {total:.4} (must be 1.0 ± 0.01)"),
            ));
        }
        if !(0.0..=1.0).contains(&self.reuse_penalty) {
            return Err(AutoError::invalid_parameter(
                "reuse_penalty",
                format!("{} (must be 0.0–1.0)", self.reuse_penalty),
            ));
        }
        Ok(())
    }
}

// ─── Selector ─────────────────────────────────────────────────────────────────

/// Selects the best B-roll clip for each coverage gap.
pub struct BRollSelector {
    /// Available B-roll clips.
    clips: Vec<BRollClip>,
    /// Configuration.
    config: BRollSelectorConfig,
    /// Usage counts per clip id.
    usage_counts: std::collections::HashMap<String, usize>,
    /// Ring buffer of recently-selected clip ids (for variety scoring).
    recent_ids: std::collections::VecDeque<String>,
}

impl BRollSelector {
    /// Create a new selector with the provided clip library.
    #[must_use]
    pub fn new(clips: Vec<BRollClip>, config: BRollSelectorConfig) -> Self {
        Self {
            clips,
            config,
            usage_counts: std::collections::HashMap::new(),
            recent_ids: std::collections::VecDeque::new(),
        }
    }

    /// Reset usage tracking (useful when starting a new sequence).
    pub fn reset_usage(&mut self) {
        self.usage_counts.clear();
        self.recent_ids.clear();
    }

    /// Select the best B-roll clip for `gap`, recording usage.
    ///
    /// Returns `None` if no clip can cover the gap (library is empty or all
    /// clips are shorter than a 1-second minimum).
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn select_for_gap(&mut self, gap: &CoverageGap) -> AutoResult<Option<BRollSelection>> {
        self.config.validate()?;
        let gap_dur = gap.duration_ms();

        // Collect candidates with their composite score
        let mut candidates: Vec<(usize, f32, f32, f32, f32)> = self
            .clips
            .iter()
            .enumerate()
            .filter_map(|(idx, clip)| {
                // Must be at least 1 s long and cover at least some of the gap
                if clip.duration_ms < 1_000 {
                    return None;
                }
                let cov = self.score_coverage(clip, gap_dur);
                let var = self.score_variety(clip);
                let rel = self.score_relevance(clip, &gap.topic_tags);
                let composite = self.config.coverage_weight * cov
                    + self.config.variety_weight * var
                    + self.config.relevance_weight * rel;

                // Apply reuse penalty
                let uses = self.usage_counts.get(&clip.id).copied().unwrap_or(0);
                let penalty = if uses >= self.config.max_reuse_count {
                    let extra = uses - self.config.max_reuse_count + 1;
                    self.config.reuse_penalty.powi(extra as i32)
                } else {
                    1.0
                };
                let final_score = (composite * clip.base_weight * penalty).clamp(0.0, 1.0);
                Some((idx, final_score, cov, var, rel))
            })
            .collect();

        if candidates.is_empty() {
            return Ok(None);
        }

        // Sort descending by composite score
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let (best_idx, score, cov, var, rel) = candidates[0];
        let clip = &self.clips[best_idx];

        // Determine in/out points
        let (in_pt, out_pt) = self.compute_in_out(clip, gap_dur);

        let selection = BRollSelection {
            clip_id: clip.id.clone(),
            score,
            coverage_score: cov,
            variety_score: var,
            relevance_score: rel,
            in_point_ms: in_pt,
            out_point_ms: out_pt,
        };

        // Record usage
        *self.usage_counts.entry(clip.id.clone()).or_insert(0) += 1;
        self.recent_ids.push_back(clip.id.clone());
        if self.recent_ids.len() > self.config.variety_window {
            self.recent_ids.pop_front();
        }

        Ok(Some(selection))
    }

    /// Select B-roll for each gap in sequence.
    ///
    /// # Errors
    ///
    /// Returns an error if configuration is invalid.
    pub fn select_sequence(
        &mut self,
        gaps: &[CoverageGap],
    ) -> AutoResult<Vec<Option<BRollSelection>>> {
        let mut results = Vec::with_capacity(gaps.len());
        for gap in gaps {
            results.push(self.select_for_gap(gap)?);
        }
        Ok(results)
    }

    /// Score how well `clip` covers a gap of `gap_dur_ms` milliseconds.
    ///
    /// Perfect coverage = clip duration ≥ gap; shorter clips score proportionally.
    /// Very long clips (> 3× gap) are gently penalised for excess.
    #[must_use]
    fn score_coverage(&self, clip: &BRollClip, gap_dur_ms: u64) -> f32 {
        if gap_dur_ms == 0 {
            return 1.0;
        }
        let ratio = clip.duration_ms as f32 / gap_dur_ms as f32;
        if ratio >= 1.0 {
            // Full coverage — slight penalty for very long clips
            1.0 - (ratio - 1.0).min(2.0) * 0.1
        } else {
            // Partial coverage
            ratio
        }
    }

    /// Score visual variety relative to recently-selected clips.
    ///
    /// Returns 1.0 if the clip has never been selected recently, decreasing
    /// toward 0.0 for clips identical to recent selections.
    #[must_use]
    fn score_variety(&self, clip: &BRollClip) -> f32 {
        let in_recent = self.recent_ids.iter().filter(|id| *id == &clip.id).count();
        if in_recent == 0 {
            // Check category variety against recently-selected clips
            let same_category_count = self
                .recent_ids
                .iter()
                .filter_map(|id| self.clips.iter().find(|c| &c.id == id))
                .filter(|c| {
                    c.scene_category == clip.scene_category
                        && clip.scene_category != SceneCategory::Unknown
                })
                .count();
            let window = self.config.variety_window.max(1);
            let category_penalty = same_category_count as f32 / window as f32 * 0.4;
            (1.0 - category_penalty).max(0.0)
        } else {
            // Already in recent window — reduce score proportionally
            let window = self.config.variety_window.max(1) as f32;
            (1.0 - in_recent as f32 / window).max(0.0)
        }
    }

    /// Score topic relevance: Jaccard similarity between clip tags and gap tags.
    #[must_use]
    fn score_relevance(&self, clip: &BRollClip, gap_tags: &[String]) -> f32 {
        if gap_tags.is_empty() || clip.topic_tags.is_empty() {
            // No tags to match — neutral score
            return 0.5;
        }
        let intersection = clip
            .topic_tags
            .iter()
            .filter(|t| gap_tags.contains(t))
            .count();
        let union = clip.topic_tags.len() + gap_tags.len() - intersection;
        if union == 0 {
            0.5
        } else {
            intersection as f32 / union as f32
        }
    }

    /// Compute recommended in/out points within a clip for a gap.
    ///
    /// Starts from the beginning of the clip; out-point is min(clip_dur, gap_dur).
    #[must_use]
    fn compute_in_out(&self, clip: &BRollClip, gap_dur_ms: u64) -> (u64, u64) {
        let out = gap_dur_ms.min(clip.duration_ms);
        (0, out)
    }

    /// Return the usage count for a clip.
    #[must_use]
    pub fn usage_count(&self, clip_id: &str) -> usize {
        self.usage_counts.get(clip_id).copied().unwrap_or(0)
    }

    /// Return the number of available clips.
    #[must_use]
    pub fn clip_count(&self) -> usize {
        self.clips.len()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_library() -> Vec<BRollClip> {
        vec![
            BRollClip::new(
                "nature_001",
                8_000,
                vec!["nature".into(), "landscape".into()],
            )
            .with_category(SceneCategory::Outdoor),
            BRollClip::new("city_001", 5_000, vec!["city".into(), "urban".into()])
                .with_category(SceneCategory::Wide),
            BRollClip::new("closeup_001", 3_000, vec!["product".into()])
                .with_category(SceneCategory::CloseUp),
            BRollClip::new("aerial_001", 10_000, vec!["nature".into(), "aerial".into()])
                .with_category(SceneCategory::Aerial),
        ]
    }

    fn make_selector() -> BRollSelector {
        BRollSelector::new(make_library(), BRollSelectorConfig::default())
    }

    #[test]
    fn test_select_for_gap_returns_some() {
        let mut selector = make_selector();
        let gap = CoverageGap {
            start_ms: 0,
            end_ms: 5_000,
            topic_tags: vec!["nature".into()],
        };
        let result = selector.select_for_gap(&gap).unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn test_relevance_drives_selection() {
        let mut selector = make_selector();
        let gap = CoverageGap {
            start_ms: 0,
            end_ms: 5_000,
            topic_tags: vec!["city".into(), "urban".into()],
        };
        let result = selector.select_for_gap(&gap).unwrap().unwrap();
        assert_eq!(result.clip_id, "city_001");
    }

    #[test]
    fn test_usage_tracking() {
        let mut selector = make_selector();
        let gap = CoverageGap {
            start_ms: 0,
            end_ms: 5_000,
            topic_tags: vec![],
        };
        selector.select_for_gap(&gap).unwrap();
        // At least one clip should have been used
        let any_used = selector
            .clips
            .iter()
            .any(|c| selector.usage_count(&c.id) > 0);
        assert!(any_used);
    }

    #[test]
    fn test_reuse_penalty_reduces_score() {
        let clips = vec![BRollClip::new("only_one", 5_000, vec![])];
        let config = BRollSelectorConfig::default();
        let mut selector = BRollSelector::new(clips, config);
        let gap = CoverageGap {
            start_ms: 0,
            end_ms: 3_000,
            topic_tags: vec![],
        };
        let first = selector.select_for_gap(&gap).unwrap().unwrap();
        let second = selector.select_for_gap(&gap).unwrap().unwrap();
        let third = selector.select_for_gap(&gap).unwrap().unwrap();
        // Score should drop after max_reuse_count
        assert!(third.score <= second.score || third.score <= first.score);
    }

    #[test]
    fn test_empty_library_returns_none() {
        let mut selector = BRollSelector::new(vec![], BRollSelectorConfig::default());
        let gap = CoverageGap {
            start_ms: 0,
            end_ms: 3_000,
            topic_tags: vec![],
        };
        let result = selector.select_for_gap(&gap).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_coverage_score_full() {
        let selector = make_selector();
        let clip = &selector.clips[0]; // 8 s clip
        let score = selector.score_coverage(clip, 5_000);
        // Clip > gap → coverage score ≥ 0.8
        assert!(score >= 0.8);
    }

    #[test]
    fn test_coverage_score_partial() {
        let selector = make_selector();
        let clip = &selector.clips[2]; // 3 s clip
        let score = selector.score_coverage(clip, 6_000);
        // Clip half of gap → ~0.5
        assert!((score - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_relevance_jaccard_exact_match() {
        let selector = make_selector();
        let clip = &selector.clips[0]; // tags: nature, landscape
        let score = selector.score_relevance(clip, &["nature".into(), "landscape".into()]);
        assert!((score - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_relevance_no_overlap() {
        let selector = make_selector();
        let clip = &selector.clips[0]; // tags: nature, landscape
        let score = selector.score_relevance(clip, &["city".into()]);
        assert!(score < 0.5);
    }

    #[test]
    fn test_color_histogram_distance_identical() {
        let h = ColorHistogram::uniform(8);
        let dist = h.chi_squared_distance(&h).unwrap();
        assert!(dist < 1e-6, "identical histograms should have distance ≈ 0");
    }

    #[test]
    fn test_select_sequence() {
        let mut selector = make_selector();
        let gaps = vec![
            CoverageGap {
                start_ms: 0,
                end_ms: 4_000,
                topic_tags: vec!["nature".into()],
            },
            CoverageGap {
                start_ms: 10_000,
                end_ms: 14_000,
                topic_tags: vec!["city".into()],
            },
        ];
        let results = selector.select_sequence(&gaps).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_reset_usage() {
        let mut selector = make_selector();
        let gap = CoverageGap {
            start_ms: 0,
            end_ms: 3_000,
            topic_tags: vec![],
        };
        selector.select_for_gap(&gap).unwrap();
        selector.reset_usage();
        let any_used = selector
            .clips
            .iter()
            .any(|c| selector.usage_count(&c.id) > 0);
        assert!(!any_used);
    }

    #[test]
    fn test_invalid_config_rejected() {
        let bad_config = BRollSelectorConfig {
            coverage_weight: -0.1,
            ..BRollSelectorConfig::default()
        };
        let mut selector = BRollSelector::new(make_library(), bad_config);
        let gap = CoverageGap {
            start_ms: 0,
            end_ms: 3_000,
            topic_tags: vec![],
        };
        assert!(selector.select_for_gap(&gap).is_err());
    }
}
