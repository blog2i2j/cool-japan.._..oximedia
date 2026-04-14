//! Confidence scoring for pulldown cadence detection with frame-level accuracy reporting.
//!
//! This module extends [`crate::pulldown_detect`] with a **Bayesian-style
//! confidence estimator** that reports how certain the cadence classifier is
//! about each hypothesis (progressive, interlaced, 2:3, 3:2, 2:3:3:2).
//!
//! # Algorithm
//!
//! For each candidate cadence `C`, the scorer measures how closely the
//! **observed combing-score sequence** matches the *expected* high/low pattern
//! for that cadence using a Gaussian log-likelihood.  The likelihoods are
//! normalised across all hypotheses with a softmax, yielding a proper
//! probability distribution.
//!
//! Frame-level accuracy is reported as the fraction of frames whose
//! **per-frame most-likely cadence phase** matches the cadence with highest
//! overall confidence.
//!
//! # Example
//!
//! ```rust
//! use oximedia_video::cadence_confidence::{CadenceConfidenceScorer, CadenceScore};
//! use oximedia_video::pulldown_detect::{FieldMetrics, Cadence};
//!
//! let mut scorer = CadenceConfidenceScorer::new();
//!
//! // Feed synthetic 3:2-pulldown field metrics.
//! // 3:2 pattern: [H, L, H, L, L] combing pattern per 5-frame cycle.
//! let pattern = [0.8f32, 0.05, 0.8, 0.05, 0.05];
//! for (i, &cs) in pattern.iter().enumerate().cycle().take(20) {
//!     scorer.push(FieldMetrics {
//!         frame_number: i as u64,
//!         combing_score: cs,
//!         tff: true,
//!     });
//! }
//!
//! let scores = scorer.score().expect("enough frames");
//! let best = scores.best_cadence();
//! assert_eq!(best, Cadence::Pulldown32);
//! ```

use std::collections::VecDeque;

use crate::pulldown_detect::{Cadence, FieldMetrics};

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Errors produced by [`CadenceConfidenceScorer`].
#[derive(Debug, thiserror::Error)]
pub enum CadenceConfidenceError {
    /// Not enough frames have been pushed to produce a reliable score.
    #[error("need at least {required} frames, have {available}")]
    InsufficientData {
        /// Minimum required.
        required: usize,
        /// Currently available.
        available: usize,
    },
}

// ---------------------------------------------------------------------------
// Expected combing patterns for each cadence hypothesis
// ---------------------------------------------------------------------------

/// Expected per-frame combing levels for a single cycle of each cadence.
///
/// `HIGH` ≈ 0.7 (interlaced field), `LOW` ≈ 0.05 (progressive field).
const HIGH: f32 = 0.70;
const LOW: f32 = 0.05;

/// For each [`Cadence`] hypothesis, the expected combing cycle pattern.
///
/// The combing score of real frames will be correlated against each of these
/// patterns.  The pattern is repeated cyclically across the history window.
fn cadence_pattern(cadence: Cadence) -> &'static [f32] {
    match cadence {
        Cadence::Progressive => &[LOW],
        Cadence::Interlaced => &[HIGH],
        Cadence::Pulldown23 => &[HIGH, LOW, LOW, HIGH, LOW], // [H,L,L,H,L]
        Cadence::Pulldown32 => &[HIGH, LOW, HIGH, LOW, LOW], // [H,L,H,L,L]
        Cadence::Pulldown2332 => &[HIGH, HIGH, LOW, LOW, HIGH, HIGH, LOW],
        Cadence::Unknown => &[],
    }
}

/// All deterministic cadence hypotheses (excludes `Unknown`).
const ALL_CADENCES: [Cadence; 5] = [
    Cadence::Progressive,
    Cadence::Interlaced,
    Cadence::Pulldown23,
    Cadence::Pulldown32,
    Cadence::Pulldown2332,
];

// ---------------------------------------------------------------------------
// Per-cadence confidence entry
// ---------------------------------------------------------------------------

/// Confidence score for a single cadence hypothesis.
#[derive(Debug, Clone)]
pub struct CadenceHypothesis {
    /// The cadence being scored.
    pub cadence: Cadence,
    /// Posterior probability in [0.0, 1.0].  All hypotheses sum to ≈ 1.0.
    pub probability: f32,
    /// Log-likelihood of the observed combing sequence under this hypothesis.
    pub log_likelihood: f64,
}

// ---------------------------------------------------------------------------
// CadenceScore — overall result
// ---------------------------------------------------------------------------

/// Overall cadence confidence report returned by [`CadenceConfidenceScorer::score`].
#[derive(Debug, Clone)]
pub struct CadenceScore {
    /// Per-hypothesis confidence scores (sorted descending by probability).
    pub hypotheses: Vec<CadenceHypothesis>,
    /// Frame-level accuracy: fraction of frames whose combing score is
    /// consistent with the best hypothesis's expected pattern.
    pub frame_accuracy: f32,
    /// Number of frames used in the analysis.
    pub frame_count: usize,
    /// The cadence with the highest posterior probability.
    pub winner: Cadence,
    /// Probability of the winning hypothesis.
    pub winner_probability: f32,
}

impl CadenceScore {
    /// Return the cadence with the highest probability (same as `winner`).
    pub fn best_cadence(&self) -> Cadence {
        self.winner
    }

    /// Return `true` if the winning cadence has a probability above `threshold`.
    pub fn is_confident(&self, threshold: f32) -> bool {
        self.winner_probability >= threshold
    }
}

// ---------------------------------------------------------------------------
// Per-frame accuracy report
// ---------------------------------------------------------------------------

/// Frame-level cadence classification result.
#[derive(Debug, Clone)]
pub struct FrameCadenceReport {
    /// Frame index.
    pub frame_number: u64,
    /// Observed combing score.
    pub combing_score: f32,
    /// Expected combing score under the winning cadence hypothesis.
    pub expected_score: f32,
    /// Residual (`|observed - expected|`).
    pub residual: f32,
    /// Whether this frame is considered consistent with the winning cadence.
    pub consistent: bool,
}

// ---------------------------------------------------------------------------
// Scorer
// ---------------------------------------------------------------------------

/// Stateful scorer that accumulates [`FieldMetrics`] and computes per-cadence
/// confidence probabilities.
#[derive(Debug)]
pub struct CadenceConfidenceScorer {
    /// History ring-buffer.
    history: VecDeque<FieldMetrics>,
    /// Maximum number of frames to keep.
    window: usize,
    /// Standard deviation assumed for Gaussian combing-score noise.
    sigma: f64,
    /// Minimum frames required before scoring.
    min_frames: usize,
}

impl Default for CadenceConfidenceScorer {
    fn default() -> Self {
        Self::new()
    }
}

impl CadenceConfidenceScorer {
    /// Create a new scorer with sensible defaults.
    ///
    /// - `window = 40`: analyse up to 40 frames.
    /// - `sigma = 0.12`: Gaussian noise standard deviation.
    /// - `min_frames = 5`: minimum frames before scoring.
    pub fn new() -> Self {
        Self {
            history: VecDeque::with_capacity(40),
            window: 40,
            sigma: 0.12,
            min_frames: 5,
        }
    }

    /// Create a scorer with explicit parameters.
    pub fn with_params(window: usize, sigma: f64, min_frames: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(window),
            window,
            sigma,
            min_frames,
        }
    }

    /// Push a new [`FieldMetrics`] observation into the scoring window.
    pub fn push(&mut self, metrics: FieldMetrics) {
        if self.history.len() >= self.window {
            self.history.pop_front();
        }
        self.history.push_back(metrics);
    }

    /// Compute confidence scores over all cadence hypotheses.
    ///
    /// # Errors
    ///
    /// Returns [`CadenceConfidenceError::InsufficientData`] if fewer than
    /// `min_frames` have been pushed.
    pub fn score(&self) -> Result<CadenceScore, CadenceConfidenceError> {
        let n = self.history.len();
        if n < self.min_frames {
            return Err(CadenceConfidenceError::InsufficientData {
                required: self.min_frames,
                available: n,
            });
        }

        let combing: Vec<f32> = self.history.iter().map(|m| m.combing_score).collect();

        // Compute log-likelihood for each hypothesis.
        let log_likelihoods: Vec<(Cadence, f64)> = ALL_CADENCES
            .iter()
            .copied()
            .map(|c| (c, self.log_likelihood(c, &combing)))
            .collect();

        // Softmax to obtain posterior probabilities.
        let max_ll = log_likelihoods
            .iter()
            .map(|(_, ll)| *ll)
            .fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = log_likelihoods
            .iter()
            .map(|(_, ll)| (ll - max_ll).exp())
            .sum();

        let mut hypotheses: Vec<CadenceHypothesis> = log_likelihoods
            .iter()
            .map(|(c, ll)| {
                let prob = ((ll - max_ll).exp() / exp_sum) as f32;
                CadenceHypothesis {
                    cadence: *c,
                    probability: prob,
                    log_likelihood: *ll,
                }
            })
            .collect();

        // Sort descending by probability.
        hypotheses.sort_by(|a, b| {
            b.probability
                .partial_cmp(&a.probability)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let winner = hypotheses
            .first()
            .map(|h| h.cadence)
            .unwrap_or(Cadence::Unknown);
        let winner_probability = hypotheses.first().map(|h| h.probability).unwrap_or(0.0);

        // Compute frame-level accuracy.
        let frame_accuracy = self.frame_accuracy(winner, &combing);

        Ok(CadenceScore {
            hypotheses,
            frame_accuracy,
            frame_count: n,
            winner,
            winner_probability,
        })
    }

    /// Compute per-frame accuracy reports against the winning cadence.
    pub fn frame_reports(&self, winner: Cadence) -> Vec<FrameCadenceReport> {
        let pattern = cadence_pattern(winner);
        if pattern.is_empty() {
            return Vec::new();
        }
        self.history
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let expected = pattern[i % pattern.len()];
                let residual = (m.combing_score - expected).abs();
                FrameCadenceReport {
                    frame_number: m.frame_number,
                    combing_score: m.combing_score,
                    expected_score: expected,
                    residual,
                    consistent: residual < 0.3,
                }
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Gaussian log-likelihood of observing `combing` under `cadence`'s pattern.
    fn log_likelihood(&self, cadence: Cadence, combing: &[f32]) -> f64 {
        let pattern = cadence_pattern(cadence);
        if pattern.is_empty() {
            // Unknown hypothesis: uniform (low) likelihood.
            return -(combing.len() as f64) * 2.0;
        }
        let two_sigma_sq = 2.0 * self.sigma * self.sigma;
        combing
            .iter()
            .enumerate()
            .map(|(i, &obs)| {
                let exp = pattern[i % pattern.len()] as f64;
                let d = obs as f64 - exp;
                -(d * d) / two_sigma_sq
            })
            .sum()
    }

    /// Fraction of frames whose combing is consistent (residual < 0.3) with
    /// the expected pattern for `cadence`.
    fn frame_accuracy(&self, cadence: Cadence, combing: &[f32]) -> f32 {
        let pattern = cadence_pattern(cadence);
        if pattern.is_empty() || combing.is_empty() {
            return 0.0;
        }
        let consistent = combing
            .iter()
            .enumerate()
            .filter(|(i, &obs)| {
                let exp = pattern[i % pattern.len()];
                (obs - exp).abs() < 0.3
            })
            .count();
        consistent as f32 / combing.len() as f32
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build synthetic field metrics for a known cadence pattern.
    fn make_metrics(pattern: &[f32], count: usize) -> Vec<FieldMetrics> {
        (0..count)
            .map(|i| FieldMetrics {
                frame_number: i as u64,
                combing_score: pattern[i % pattern.len()],
                tff: true,
            })
            .collect()
    }

    // ------------------------------------------------------------------
    // 1. Progressive content recognised
    // ------------------------------------------------------------------
    #[test]
    fn test_progressive_detected() {
        let mut scorer = CadenceConfidenceScorer::new();
        for m in make_metrics(&[0.02], 20) {
            scorer.push(m);
        }
        let score = scorer.score().expect("should score");
        assert_eq!(
            score.best_cadence(),
            Cadence::Progressive,
            "flat-low combing should detect as Progressive"
        );
        assert!(
            score.winner_probability > 0.5,
            "confidence should be >50% for clear progressive signal"
        );
    }

    // ------------------------------------------------------------------
    // 2. Fully interlaced content recognised
    // ------------------------------------------------------------------
    #[test]
    fn test_interlaced_detected() {
        let mut scorer = CadenceConfidenceScorer::new();
        for m in make_metrics(&[0.70], 20) {
            scorer.push(m);
        }
        let score = scorer.score().expect("should score");
        assert_eq!(score.best_cadence(), Cadence::Interlaced);
    }

    // ------------------------------------------------------------------
    // 3. 3:2 pulldown pattern recognised
    // ------------------------------------------------------------------
    #[test]
    fn test_pulldown32_detected() {
        let pattern: &[f32] = &[HIGH, LOW, HIGH, LOW, LOW];
        let mut scorer = CadenceConfidenceScorer::new();
        for m in make_metrics(pattern, 30) {
            scorer.push(m);
        }
        let score = scorer.score().expect("should score");
        assert_eq!(
            score.best_cadence(),
            Cadence::Pulldown32,
            "3:2 pulldown pattern should be detected"
        );
    }

    // ------------------------------------------------------------------
    // 4. 2:3 pulldown pattern recognised
    // ------------------------------------------------------------------
    #[test]
    fn test_pulldown23_detected() {
        let pattern: &[f32] = &[HIGH, LOW, LOW, HIGH, LOW];
        let mut scorer = CadenceConfidenceScorer::new();
        for m in make_metrics(pattern, 30) {
            scorer.push(m);
        }
        let score = scorer.score().expect("should score");
        assert_eq!(score.best_cadence(), Cadence::Pulldown23);
    }

    // ------------------------------------------------------------------
    // 5. Insufficient data error
    // ------------------------------------------------------------------
    #[test]
    fn test_insufficient_data_error() {
        let scorer = CadenceConfidenceScorer::new();
        let err = scorer.score();
        assert!(
            matches!(err, Err(CadenceConfidenceError::InsufficientData { .. })),
            "should error with too few frames"
        );
    }

    // ------------------------------------------------------------------
    // 6. Probabilities sum to ≈ 1.0
    // ------------------------------------------------------------------
    #[test]
    fn test_probabilities_sum_to_one() {
        let mut scorer = CadenceConfidenceScorer::new();
        for m in make_metrics(&[0.02], 20) {
            scorer.push(m);
        }
        let score = scorer.score().expect("should score");
        let sum: f32 = score.hypotheses.iter().map(|h| h.probability).sum();
        assert!(
            (sum - 1.0f32).abs() < 1e-4,
            "probabilities should sum to 1.0, got {sum}"
        );
    }

    // ------------------------------------------------------------------
    // 7. Frame accuracy close to 1.0 for clean patterns
    // ------------------------------------------------------------------
    #[test]
    fn test_frame_accuracy_near_one_for_clean_progressive() {
        let mut scorer = CadenceConfidenceScorer::new();
        for m in make_metrics(&[0.02], 20) {
            scorer.push(m);
        }
        let score = scorer.score().expect("should score");
        assert!(
            score.frame_accuracy > 0.9,
            "frame accuracy should be >90% for clean progressive signal, got {}",
            score.frame_accuracy
        );
    }

    // ------------------------------------------------------------------
    // 8. is_confident returns false below threshold
    // ------------------------------------------------------------------
    #[test]
    fn test_is_confident_threshold() {
        let mut scorer = CadenceConfidenceScorer::new();
        for m in make_metrics(&[0.02], 20) {
            scorer.push(m);
        }
        let score = scorer.score().expect("should score");
        // Should be confident at a low threshold
        assert!(
            score.is_confident(0.3),
            "should be confident at 0.3 threshold"
        );
        // Should not be confident at 100% threshold
        assert!(
            !score.is_confident(1.01),
            "probability can never exceed 1.0"
        );
    }

    // ------------------------------------------------------------------
    // 9. Frame reports show correct expected scores
    // ------------------------------------------------------------------
    #[test]
    fn test_frame_reports_expected_scores() {
        let mut scorer = CadenceConfidenceScorer::new();
        for m in make_metrics(&[0.02], 10) {
            scorer.push(m);
        }
        let reports = scorer.frame_reports(Cadence::Progressive);
        assert_eq!(reports.len(), 10);
        for r in &reports {
            // Progressive pattern is [LOW=0.05]; all expected scores should match.
            assert!(
                (r.expected_score - LOW).abs() < 1e-6,
                "expected LOW for progressive"
            );
        }
    }

    // ------------------------------------------------------------------
    // 10. Hypotheses are sorted descending by probability
    // ------------------------------------------------------------------
    #[test]
    fn test_hypotheses_sorted_descending() {
        let mut scorer = CadenceConfidenceScorer::new();
        for m in make_metrics(&[0.7], 20) {
            scorer.push(m);
        }
        let score = scorer.score().expect("should score");
        let probs: Vec<f32> = score.hypotheses.iter().map(|h| h.probability).collect();
        for w in probs.windows(2) {
            assert!(
                w[0] >= w[1],
                "hypotheses should be sorted descending: {} < {}",
                w[0],
                w[1]
            );
        }
    }

    // ------------------------------------------------------------------
    // 11. with_params constructor
    // ------------------------------------------------------------------
    #[test]
    fn test_with_params_constructor() {
        let scorer = CadenceConfidenceScorer::with_params(10, 0.05, 3);
        assert_eq!(scorer.window, 10);
        assert!((scorer.sigma - 0.05).abs() < 1e-12);
        assert_eq!(scorer.min_frames, 3);
    }

    // ------------------------------------------------------------------
    // 12. Scorer respects window size (oldest frames evicted)
    // ------------------------------------------------------------------
    #[test]
    fn test_window_size_respected() {
        let mut scorer = CadenceConfidenceScorer::with_params(5, 0.12, 3);
        for i in 0..10u64 {
            scorer.push(FieldMetrics {
                frame_number: i,
                combing_score: 0.1,
                tff: true,
            });
        }
        assert_eq!(
            scorer.history.len(),
            5,
            "history should not exceed window size"
        );
    }
}
