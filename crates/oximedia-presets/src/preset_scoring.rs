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
}
