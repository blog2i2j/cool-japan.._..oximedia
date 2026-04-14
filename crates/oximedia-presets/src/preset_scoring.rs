//! Preset scoring and ranking.
//!
//! Provides a weighted multi-criteria scoring system for ranking presets
//! against a target specification. Users define a [`ScoringProfile`] that
//! weights different criteria (quality, speed, file-size, compatibility),
//! then each candidate preset receives a normalised 0..100 score.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

// ── ScoreCriterion ─────────────────────────────────────────────────────────

/// A criterion used to score a preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScoreCriterion {
    /// Visual quality (higher is better).
    Quality,
    /// Encoding speed (lower time is better).
    Speed,
    /// Output file size (smaller is better).
    FileSize,
    /// Platform compatibility breadth.
    Compatibility,
    /// HDR support / colour-accuracy.
    ColorAccuracy,
    /// Low-latency suitability.
    Latency,
}

impl ScoreCriterion {
    /// All defined criteria (useful for iteration).
    #[must_use]
    pub fn all() -> &'static [ScoreCriterion] {
        &[
            Self::Quality,
            Self::Speed,
            Self::FileSize,
            Self::Compatibility,
            Self::ColorAccuracy,
            Self::Latency,
        ]
    }
}

// ── ScoringProfile ─────────────────────────────────────────────────────────

/// Weights assigned to each [`ScoreCriterion`].
///
/// Weights are arbitrary positive floats; they are normalised internally
/// before applying.
#[derive(Debug, Clone)]
pub struct ScoringProfile {
    /// Human-readable name.
    pub name: String,
    /// Weight for each criterion (missing criteria receive weight 0).
    weights: HashMap<ScoreCriterion, f64>,
}

impl ScoringProfile {
    /// Create a profile with all weights set to zero.
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            weights: HashMap::new(),
        }
    }

    /// Builder-style weight setter.
    #[must_use]
    pub fn with_weight(mut self, criterion: ScoreCriterion, weight: f64) -> Self {
        self.set_weight(criterion, weight);
        self
    }

    /// Set a weight for a specific criterion.
    pub fn set_weight(&mut self, criterion: ScoreCriterion, weight: f64) {
        self.weights.insert(criterion, weight.max(0.0));
    }

    /// Get the raw weight for a criterion.
    #[must_use]
    pub fn weight(&self, criterion: ScoreCriterion) -> f64 {
        self.weights.get(&criterion).copied().unwrap_or(0.0)
    }

    /// Sum of all weights.
    #[must_use]
    pub fn total_weight(&self) -> f64 {
        self.weights.values().sum()
    }

    /// Return a normalised weight (0.0..1.0) for a criterion.
    #[must_use]
    pub fn normalised_weight(&self, criterion: ScoreCriterion) -> f64 {
        let total = self.total_weight();
        if total <= 0.0 {
            return 0.0;
        }
        self.weight(criterion) / total
    }

    /// A pre-built profile optimised for maximum quality.
    #[must_use]
    pub fn quality_focused() -> Self {
        Self::new("quality-focused")
            .with_weight(ScoreCriterion::Quality, 10.0)
            .with_weight(ScoreCriterion::ColorAccuracy, 5.0)
            .with_weight(ScoreCriterion::Speed, 1.0)
            .with_weight(ScoreCriterion::FileSize, 1.0)
    }

    /// A pre-built profile optimised for fast encoding.
    #[must_use]
    pub fn speed_focused() -> Self {
        Self::new("speed-focused")
            .with_weight(ScoreCriterion::Speed, 10.0)
            .with_weight(ScoreCriterion::Quality, 3.0)
            .with_weight(ScoreCriterion::FileSize, 2.0)
    }

    /// A pre-built profile optimised for smallest output.
    #[must_use]
    pub fn size_focused() -> Self {
        Self::new("size-focused")
            .with_weight(ScoreCriterion::FileSize, 10.0)
            .with_weight(ScoreCriterion::Quality, 4.0)
            .with_weight(ScoreCriterion::Speed, 2.0)
    }
}

// ── PresetScore ────────────────────────────────────────────────────────────

/// Raw per-criterion scores for a single preset candidate, each in 0..100.
#[derive(Debug, Clone)]
pub struct PresetScore {
    /// Preset identifier.
    pub preset_id: String,
    /// Per-criterion raw scores.
    scores: HashMap<ScoreCriterion, f64>,
}

impl PresetScore {
    /// Create a new score container for the given preset.
    #[must_use]
    pub fn new(preset_id: &str) -> Self {
        Self {
            preset_id: preset_id.to_string(),
            scores: HashMap::new(),
        }
    }

    /// Set a raw score for a criterion (clamped to 0..100).
    pub fn set(&mut self, criterion: ScoreCriterion, score: f64) {
        self.scores.insert(criterion, score.clamp(0.0, 100.0));
    }

    /// Builder-style raw-score setter.
    #[must_use]
    pub fn with_score(mut self, criterion: ScoreCriterion, score: f64) -> Self {
        self.set(criterion, score);
        self
    }

    /// Get the raw score for a criterion (0.0 if unset).
    #[must_use]
    pub fn get(&self, criterion: ScoreCriterion) -> f64 {
        self.scores.get(&criterion).copied().unwrap_or(0.0)
    }

    /// Compute the weighted aggregate score using a profile.
    ///
    /// Returns a value in 0.0..100.0.
    #[must_use]
    pub fn weighted_total(&self, profile: &ScoringProfile) -> f64 {
        let total_weight = profile.total_weight();
        if total_weight <= 0.0 {
            return 0.0;
        }
        let sum: f64 = self
            .scores
            .iter()
            .map(|(c, v)| v * profile.weight(*c))
            .sum();
        (sum / total_weight).clamp(0.0, 100.0)
    }

    /// Number of criteria with scores set.
    #[must_use]
    pub fn criteria_count(&self) -> usize {
        self.scores.len()
    }
}

// ── PresetRanker ───────────────────────────────────────────────────────────

/// Collects [`PresetScore`] items and ranks them against a
/// [`ScoringProfile`].
#[derive(Debug, Clone)]
pub struct PresetRanker {
    /// Scoring profile to apply.
    profile: ScoringProfile,
    /// Candidate scores.
    candidates: Vec<PresetScore>,
}

impl PresetRanker {
    /// Create a new ranker with the given profile.
    #[must_use]
    pub fn new(profile: ScoringProfile) -> Self {
        Self {
            profile,
            candidates: Vec::new(),
        }
    }

    /// Add a candidate preset score.
    pub fn add(&mut self, score: PresetScore) {
        self.candidates.push(score);
    }

    /// Number of candidates.
    #[must_use]
    pub fn candidate_count(&self) -> usize {
        self.candidates.len()
    }

    /// Return candidates sorted by weighted total (descending).
    ///
    /// Each entry is `(preset_id, weighted_score)`.
    #[must_use]
    pub fn rank(&self) -> Vec<(String, f64)> {
        let mut ranked: Vec<(String, f64)> = self
            .candidates
            .iter()
            .map(|c| (c.preset_id.clone(), c.weighted_total(&self.profile)))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }

    /// Return the top-N candidates by weighted score.
    #[must_use]
    pub fn top_n(&self, n: usize) -> Vec<(String, f64)> {
        let ranked = self.rank();
        ranked.into_iter().take(n).collect()
    }

    /// Return the single best candidate, if any.
    #[must_use]
    pub fn best(&self) -> Option<(String, f64)> {
        self.rank().into_iter().next()
    }

    /// Return the reference to the scoring profile.
    #[must_use]
    pub fn profile(&self) -> &ScoringProfile {
        &self.profile
    }
}

// ── ScoringWeights ─────────────────────────────────────────────────────────

/// Normalised weight vector for the five primary scoring dimensions.
///
/// All five weights should sum to 1.0.  The constructor normalises
/// automatically so callers can pass raw positive ratios.
#[derive(Debug, Clone)]
pub struct ScoringWeights {
    /// Weight for visual quality (higher CRF / lossless = higher score).
    pub quality: f32,
    /// Weight for encoding latency (lower latency = higher score).
    pub latency: f32,
    /// Weight for output file size (smaller = higher score).
    pub file_size: f32,
    /// Weight for platform / device compatibility breadth.
    pub compatibility: f32,
    /// Weight for encoding speed (faster = higher score).
    pub speed: f32,
}

impl ScoringWeights {
    /// Create weights, normalising the provided values so they sum to 1.0.
    ///
    /// If all weights are zero an equal-weight distribution is used.
    #[must_use]
    pub fn new(quality: f32, latency: f32, file_size: f32, compatibility: f32, speed: f32) -> Self {
        let sum = quality + latency + file_size + compatibility + speed;
        if sum <= 0.0 {
            // Fallback: equal weights.
            return Self {
                quality: 0.2,
                latency: 0.2,
                file_size: 0.2,
                compatibility: 0.2,
                speed: 0.2,
            };
        }
        Self {
            quality: quality / sum,
            latency: latency / sum,
            file_size: file_size / sum,
            compatibility: compatibility / sum,
            speed: speed / sum,
        }
    }

    /// Return weights appropriate for the given use case.
    ///
    /// | Use case    | Quality | Latency | File size | Compat | Speed |
    /// |-------------|---------|---------|-----------|--------|-------|
    /// | Streaming   | 0.30    | 0.30    | 0.15      | 0.20   | 0.05  |
    /// | Archive     | 0.50    | 0.05    | 0.25      | 0.10   | 0.10  |
    /// | Social      | 0.25    | 0.10    | 0.30      | 0.30   | 0.05  |
    /// | Broadcast   | 0.40    | 0.25    | 0.10      | 0.20   | 0.05  |
    /// | Mobile      | 0.20    | 0.15    | 0.35      | 0.25   | 0.05  |
    #[must_use]
    pub fn for_use_case(use_case: crate::UseCase) -> Self {
        match use_case {
            crate::UseCase::Streaming => Self::new(0.30, 0.30, 0.15, 0.20, 0.05),
            crate::UseCase::Archive => Self::new(0.50, 0.05, 0.25, 0.10, 0.10),
            crate::UseCase::Social => Self::new(0.25, 0.10, 0.30, 0.30, 0.05),
            crate::UseCase::Broadcast => Self::new(0.40, 0.25, 0.10, 0.20, 0.05),
            crate::UseCase::Mobile => Self::new(0.20, 0.15, 0.35, 0.25, 0.05),
        }
    }

    /// Verify that the weights sum to approximately 1.0 (within 0.001).
    #[must_use]
    pub fn is_normalised(&self) -> bool {
        let sum = self.quality + self.latency + self.file_size + self.compatibility + self.speed;
        (sum - 1.0).abs() < 1e-3
    }
}

// ── PresetScorer ────────────────────────────────────────────────────────────

/// Scores a [`crate::Preset`] against [`crate::SelectionCriteria`] using
/// customisable [`ScoringWeights`].
///
/// # Design
///
/// The scorer derives five raw component scores (0–100 each) from observable
/// preset attributes (tags, bitrate, metadata keywords) and applies the weight
/// vector to produce a single 0–100 composite score.
#[derive(Debug, Clone)]
pub struct PresetScorer {
    weights: ScoringWeights,
}

impl PresetScorer {
    /// Create a scorer with explicit weights.
    #[must_use]
    pub fn new(weights: ScoringWeights) -> Self {
        Self { weights }
    }

    /// Builder-style constructor.
    #[must_use]
    pub fn with_weights(weights: ScoringWeights) -> Self {
        Self { weights }
    }

    /// Create a scorer calibrated for a specific use case.
    #[must_use]
    pub fn for_use_case(use_case: crate::UseCase) -> Self {
        Self::new(ScoringWeights::for_use_case(use_case))
    }

    /// Compute a weighted composite score for `preset` against `criteria`.
    ///
    /// Returns a value in 0.0–100.0 (higher = better match).
    #[must_use]
    pub fn score_preset(&self, preset: &crate::Preset, criteria: &crate::SelectionCriteria) -> f32 {
        let quality_score = self.quality_score(preset);
        let latency_score = self.latency_score(preset);
        let file_size_score = self.file_size_score(preset, criteria);
        let compat_score = self.compatibility_score(preset);
        let speed_score = self.speed_score(preset);

        (quality_score * self.weights.quality
            + latency_score * self.weights.latency
            + file_size_score * self.weights.file_size
            + compat_score * self.weights.compatibility
            + speed_score * self.weights.speed)
            .clamp(0.0, 100.0)
    }

    // ── Component scorers (0–100 each) ──────────────────────────────────────

    fn quality_score(&self, preset: &crate::Preset) -> f32 {
        let name = preset.metadata.name.to_lowercase();
        let desc = preset.metadata.description.to_lowercase();
        if preset.has_tag("lossless") || name.contains("lossless") || desc.contains("lossless") {
            100.0
        } else if preset.has_tag("high")
            || name.contains("high")
            || preset.has_tag("4k")
            || name.contains("4k")
        {
            80.0
        } else if preset.has_tag("medium") || name.contains("medium") {
            55.0
        } else if preset.has_tag("low") || name.contains("low") {
            25.0
        } else {
            50.0 // neutral
        }
    }

    fn latency_score(&self, preset: &crate::Preset) -> f32 {
        let name = preset.metadata.name.to_lowercase();
        if preset.has_tag("low-latency")
            || preset.has_tag("ll-hls")
            || name.contains("low-latency")
            || name.contains("realtime")
        {
            100.0
        } else if preset.has_tag("rtmp")
            || preset.has_tag("srt")
            || name.contains("rtmp")
            || name.contains("srt")
        {
            75.0
        } else if preset.has_tag("lossless") || preset.has_tag("archive") {
            10.0
        } else {
            40.0 // neutral
        }
    }

    fn file_size_score(&self, preset: &crate::Preset, criteria: &crate::SelectionCriteria) -> f32 {
        if criteria.target_bitrate_kbps == 0 {
            return 50.0;
        }
        if let Some(br_bps) = preset.config.video_bitrate {
            let target_bps = criteria.target_bitrate_kbps * 1000;
            // Lower bitrate than target → higher file-size score.
            if br_bps <= target_bps {
                let ratio = br_bps as f32 / target_bps as f32;
                100.0 - ratio * 50.0 // 100 at 0 bps, 50 at target
            } else {
                // Over budget: score drops steeply.
                let over = (br_bps - target_bps) as f32 / target_bps as f32;
                (50.0 - over * 50.0).max(0.0)
            }
        } else {
            50.0
        }
    }

    fn compatibility_score(&self, preset: &crate::Preset) -> f32 {
        // Broadly compatible formats: H.264, VP9, Opus, AAC.
        let name = preset.metadata.name.to_lowercase();
        if name.contains("h264") || name.contains("h.264") || preset.has_tag("h264") {
            90.0
        } else if name.contains("vp9") || preset.has_tag("vp9") {
            70.0
        } else if name.contains("av1") || preset.has_tag("av1") {
            55.0
        } else if name.contains("hevc") || name.contains("h265") || preset.has_tag("hevc") {
            60.0
        } else if preset.has_tag("lossless") || name.contains("lossless") {
            30.0 // lossless codecs have limited decoder support
        } else {
            50.0
        }
    }

    fn speed_score(&self, preset: &crate::Preset) -> f32 {
        let name = preset.metadata.name.to_lowercase();
        if name.contains("fast") || preset.has_tag("fast") || name.contains("realtime") {
            90.0
        } else if preset.has_tag("lossless") || name.contains("lossless") {
            20.0
        } else if name.contains("slow") || preset.has_tag("slow") {
            15.0
        } else {
            50.0
        }
    }

    /// Return the current weight configuration.
    #[must_use]
    pub fn weights(&self) -> &ScoringWeights {
        &self.weights
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn quality_profile() -> ScoringProfile {
        ScoringProfile::quality_focused()
    }

    fn two_candidates() -> (PresetScore, PresetScore) {
        let a = PresetScore::new("preset-a")
            .with_score(ScoreCriterion::Quality, 90.0)
            .with_score(ScoreCriterion::Speed, 40.0)
            .with_score(ScoreCriterion::FileSize, 30.0)
            .with_score(ScoreCriterion::ColorAccuracy, 80.0);
        let b = PresetScore::new("preset-b")
            .with_score(ScoreCriterion::Quality, 60.0)
            .with_score(ScoreCriterion::Speed, 90.0)
            .with_score(ScoreCriterion::FileSize, 70.0)
            .with_score(ScoreCriterion::ColorAccuracy, 50.0);
        (a, b)
    }

    // ── ScoreCriterion ──

    #[test]
    fn test_all_criteria_count() {
        assert_eq!(ScoreCriterion::all().len(), 6);
    }

    // ── ScoringProfile ──

    #[test]
    fn test_profile_empty_total() {
        let p = ScoringProfile::new("empty");
        assert!((p.total_weight() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_profile_normalised_weights_sum_to_one() {
        let p = quality_profile();
        let sum: f64 = ScoreCriterion::all()
            .iter()
            .map(|c| p.normalised_weight(*c))
            .sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_profile_quality_weight_dominates() {
        let p = quality_profile();
        let qw = p.normalised_weight(ScoreCriterion::Quality);
        let sw = p.normalised_weight(ScoreCriterion::Speed);
        assert!(qw > sw);
    }

    #[test]
    fn test_profile_speed_focused() {
        let p = ScoringProfile::speed_focused();
        let sw = p.normalised_weight(ScoreCriterion::Speed);
        let qw = p.normalised_weight(ScoreCriterion::Quality);
        assert!(sw > qw);
    }

    #[test]
    fn test_profile_size_focused() {
        let p = ScoringProfile::size_focused();
        let fw = p.normalised_weight(ScoreCriterion::FileSize);
        let sw = p.normalised_weight(ScoreCriterion::Speed);
        assert!(fw > sw);
    }

    // ── PresetScore ──

    #[test]
    fn test_score_clamp() {
        let s = PresetScore::new("x").with_score(ScoreCriterion::Quality, 150.0);
        assert!((s.get(ScoreCriterion::Quality) - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_score_clamp_negative() {
        let s = PresetScore::new("x").with_score(ScoreCriterion::Speed, -10.0);
        assert!((s.get(ScoreCriterion::Speed) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_score_unset_criterion() {
        let s = PresetScore::new("x");
        assert!((s.get(ScoreCriterion::Latency) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_score_criteria_count() {
        let (a, _) = two_candidates();
        assert_eq!(a.criteria_count(), 4);
    }

    // ── PresetRanker ──

    #[test]
    fn test_ranker_quality_profile_picks_a() {
        let p = quality_profile();
        let (a, b) = two_candidates();
        let mut ranker = PresetRanker::new(p);
        ranker.add(a);
        ranker.add(b);
        let best = ranker.best().expect("best should be valid");
        assert_eq!(best.0, "preset-a");
    }

    #[test]
    fn test_ranker_speed_profile_picks_b() {
        let p = ScoringProfile::speed_focused();
        let (a, b) = two_candidates();
        let mut ranker = PresetRanker::new(p);
        ranker.add(a);
        ranker.add(b);
        let best = ranker.best().expect("best should be valid");
        assert_eq!(best.0, "preset-b");
    }

    #[test]
    fn test_ranker_top_n() {
        let p = quality_profile();
        let (a, b) = two_candidates();
        let mut ranker = PresetRanker::new(p);
        ranker.add(a);
        ranker.add(b);
        let top = ranker.top_n(1);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].0, "preset-a");
    }

    #[test]
    fn test_ranker_empty_best_is_none() {
        let ranker = PresetRanker::new(quality_profile());
        assert!(ranker.best().is_none());
    }

    // ── ScoringWeights ──────────────────────────────────────────────────────

    #[test]
    fn test_weights_normalised_after_construction() {
        let w = ScoringWeights::new(3.0, 2.0, 1.0, 1.0, 1.0);
        assert!(w.is_normalised(), "Weights must normalise to 1.0");
    }

    #[test]
    fn test_weights_all_zero_falls_back_to_equal() {
        let w = ScoringWeights::new(0.0, 0.0, 0.0, 0.0, 0.0);
        assert!(w.is_normalised());
        // Each dimension should be ~0.2.
        let delta = (w.quality - 0.2).abs();
        assert!(delta < 1e-4, "Expected ~0.2, got {}", w.quality);
    }

    #[test]
    fn test_weights_streaming_latency_high() {
        let w = ScoringWeights::for_use_case(crate::UseCase::Streaming);
        assert!(w.latency >= 0.25, "Streaming needs high latency weight");
    }

    #[test]
    fn test_weights_archive_quality_dominant() {
        let w = ScoringWeights::for_use_case(crate::UseCase::Archive);
        assert!(
            w.quality > w.latency,
            "Archive prioritises quality over latency"
        );
        assert!(
            w.quality > w.speed,
            "Archive prioritises quality over speed"
        );
    }

    #[test]
    fn test_weights_mobile_file_size_high() {
        let w = ScoringWeights::for_use_case(crate::UseCase::Mobile);
        assert!(w.file_size >= 0.30, "Mobile needs high file-size weight");
    }

    // ── PresetScorer ────────────────────────────────────────────────────────

    fn make_preset(id: &str, tags: &[&str]) -> crate::Preset {
        use crate::{Preset, PresetCategory, PresetMetadata};
        use oximedia_transcode::PresetConfig;
        let mut meta = PresetMetadata::new(id, id, PresetCategory::Custom);
        for &t in tags {
            meta = meta.with_tag(t);
        }
        Preset::new(meta, PresetConfig::default())
    }

    fn make_low_latency_preset() -> crate::Preset {
        use crate::{Preset, PresetCategory, PresetMetadata};
        use oximedia_transcode::PresetConfig;
        let meta = PresetMetadata::new("low-lat", "Low Latency Streaming", PresetCategory::Custom)
            .with_tag("low-latency")
            .with_tag("rtmp");
        Preset::new(meta, PresetConfig::default())
    }

    fn make_lossless_preset() -> crate::Preset {
        use crate::{Preset, PresetCategory, PresetMetadata};
        use oximedia_transcode::PresetConfig;
        let meta = PresetMetadata::new("lossless", "Lossless Archive", PresetCategory::Custom)
            .with_tag("lossless")
            .with_tag("archive");
        Preset::new(meta, PresetConfig::default())
    }

    fn default_criteria() -> crate::SelectionCriteria {
        crate::SelectionCriteria {
            target_bitrate_kbps: 5_000,
            width: 1920,
            height: 1080,
            frame_rate: 30.0,
            use_case: crate::UseCase::Streaming,
        }
    }

    #[test]
    fn test_scorer_score_in_range() {
        let scorer = PresetScorer::for_use_case(crate::UseCase::Streaming);
        let preset = make_preset("test", &["streaming"]);
        let score = scorer.score_preset(&preset, &default_criteria());
        assert!(
            score >= 0.0 && score <= 100.0,
            "Score out of range: {score}"
        );
    }

    #[test]
    fn test_scorer_latency_weights_rank_low_latency_first() {
        // Create a scorer that cares almost exclusively about latency.
        let weights = ScoringWeights::new(0.0, 1.0, 0.0, 0.0, 0.0);
        let scorer = PresetScorer::with_weights(weights);
        let criteria = default_criteria();

        let ll = make_low_latency_preset();
        let lossless = make_lossless_preset();

        let ll_score = scorer.score_preset(&ll, &criteria);
        let lossless_score = scorer.score_preset(&lossless, &criteria);

        assert!(
            ll_score > lossless_score,
            "Low-latency preset ({ll_score}) should score higher than lossless ({lossless_score}) under latency weights"
        );
    }

    #[test]
    fn test_scorer_quality_weights_rank_lossless_first() {
        let weights = ScoringWeights::new(1.0, 0.0, 0.0, 0.0, 0.0);
        let scorer = PresetScorer::with_weights(weights);
        let criteria = default_criteria();

        let ll = make_low_latency_preset();
        let lossless = make_lossless_preset();

        let ll_score = scorer.score_preset(&ll, &criteria);
        let lossless_score = scorer.score_preset(&lossless, &criteria);

        assert!(
            lossless_score > ll_score,
            "Lossless preset ({lossless_score}) should score higher under quality weights (ll={ll_score})"
        );
    }
}
