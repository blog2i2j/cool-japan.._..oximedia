//! Caption quality scoring.
//!
//! Computes accuracy, timing, readability, and completeness scores for a set
//! of captions, and combines them into an overall quality grade.

#![allow(dead_code)]

use crate::live_captions::CaptionSegment;
use serde::{Deserialize, Serialize};

/// Aggregated quality score for a set of captions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptionQualityScore {
    /// Word-level accuracy (0.0–1.0, based on WER against a reference).
    pub accuracy: f32,
    /// Timing score (0.0–1.0, based on caption delay).
    pub timing: f32,
    /// Readability score (0.0–1.0, based on reading speed and line length).
    pub readability: f32,
    /// Completeness score (0.0–1.0, caption coverage fraction).
    pub completeness: f32,
    /// Weighted overall score (0.0–1.0).
    pub overall: f32,
}

impl CaptionQualityScore {
    /// Create a new score, computing the overall as a weighted mean.
    #[must_use]
    pub fn new(accuracy: f32, timing: f32, readability: f32, completeness: f32) -> Self {
        let overall = accuracy * 0.35 + timing * 0.25 + readability * 0.20 + completeness * 0.20;
        Self {
            accuracy,
            timing,
            readability,
            completeness,
            overall: overall.clamp(0.0, 1.0),
        }
    }

    /// Convert the overall score to an alphabetic grade.
    ///
    /// | Range       | Grade |
    /// |-------------|-------|
    /// | 0.90 – 1.00 | A     |
    /// | 0.80 – 0.90 | B     |
    /// | 0.70 – 0.80 | C     |
    /// | 0.60 – 0.70 | D     |
    /// | < 0.60      | F     |
    #[must_use]
    pub fn overall_grade(&self) -> char {
        match self.overall {
            s if s >= 0.90 => 'A',
            s if s >= 0.80 => 'B',
            s if s >= 0.70 => 'C',
            s if s >= 0.60 => 'D',
            _ => 'F',
        }
    }
}

/// Accuracy metrics based on Word Error Rate.
pub struct AccuracyMetrics;

impl AccuracyMetrics {
    /// Compute the Word Error Rate (WER) between a reference and hypothesis.
    ///
    /// WER = (S + D + I) / N  where:
    /// - S = substitutions, D = deletions, I = insertions
    /// - N = number of words in the reference
    ///
    /// Returns a value in [0, ∞). 0.0 = perfect. Values > 1.0 are possible.
    #[must_use]
    pub fn wer(reference: &str, hypothesis: &str) -> f32 {
        let ref_words: Vec<&str> = reference.split_whitespace().collect();
        let hyp_words: Vec<&str> = hypothesis.split_whitespace().collect();

        let n = ref_words.len();
        if n == 0 {
            return if hyp_words.is_empty() { 0.0 } else { 1.0 };
        }

        let edit_dist = levenshtein_words(&ref_words, &hyp_words);
        edit_dist as f32 / n as f32
    }

    /// Convert WER to a quality score in [0, 1].
    ///
    /// A WER of 0 → score 1.0; WER ≥ 1.0 → score 0.0.
    #[must_use]
    pub fn wer_to_score(wer: f32) -> f32 {
        (1.0 - wer).clamp(0.0, 1.0)
    }
}

/// Word-level Levenshtein distance (edit distance).
fn levenshtein_words(a: &[&str], b: &[&str]) -> usize {
    let m = a.len();
    let n = b.len();
    let mut dp = vec![vec![0usize; n + 1]; m + 1];

    for i in 0..=m {
        dp[i][0] = i;
    }
    for j in 0..=n {
        dp[0][j] = j;
    }
    for i in 1..=m {
        for j in 1..=n {
            let cost = usize::from(!a[i - 1].eq_ignore_ascii_case(b[j - 1]));
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }
    dp[m][n]
}

/// Timing quality metrics.
pub struct TimingMetrics;

impl TimingMetrics {
    /// Compute a delay score.
    ///
    /// Returns 1.0 if `caption_ms <= reference_ms`, linearly decaying to 0.0
    /// at 8 seconds of delay.
    #[must_use]
    pub fn delay_score(reference_ms: u64, caption_ms: u64) -> f32 {
        const MAX_DELAY_MS: f64 = 8_000.0;
        let delay = caption_ms.saturating_sub(reference_ms) as f64;
        if delay <= 0.0 {
            1.0
        } else {
            (1.0 - delay / MAX_DELAY_MS).max(0.0) as f32
        }
    }
}

/// Readability quality metrics.
pub struct ReadabilityMetrics;

impl ReadabilityMetrics {
    /// Compute a readability score for a caption segment.
    ///
    /// Based on:
    /// - **Reading speed**: ideal 160–180 WPM. Score drops outside this range.
    /// - **Line length**: ideal ≤ 42 characters/line. Penalty for longer lines.
    ///
    /// Returns a value in [0, 1].
    #[must_use]
    pub fn score(text: &str, duration_ms: u64) -> f32 {
        if duration_ms == 0 {
            return 0.0;
        }

        let word_count = text.split_whitespace().count() as f64;
        let duration_min = duration_ms as f64 / 60_000.0;
        let wpm = if duration_min > 0.0 {
            word_count / duration_min
        } else {
            0.0
        };

        // Score reading speed: ideal 160–180 WPM
        let speed_score = if (160.0..=180.0).contains(&wpm) {
            1.0_f32
        } else if wpm < 160.0 {
            (wpm / 160.0) as f32
        } else {
            // Too fast
            (1.0 - (wpm - 180.0) / 180.0).max(0.0) as f32
        };

        // Score line length: ideal ≤ 42 chars
        let max_line_len = text.lines().map(str::len).max().unwrap_or(0);
        let line_score = if max_line_len <= 42 {
            1.0_f32
        } else {
            (42.0 / max_line_len as f32).clamp(0.0, 1.0)
        };

        ((speed_score + line_score) / 2.0).clamp(0.0, 1.0)
    }
}

/// Completeness (coverage) metrics.
pub struct CompletenessMetrics;

impl CompletenessMetrics {
    /// Compute the caption coverage percentage as a score.
    ///
    /// `caption_secs` is the total duration of captioned content;
    /// `total_secs` is the total programme duration.
    ///
    /// Returns a value in [0.0, 1.0].
    #[must_use]
    pub fn coverage_pct(caption_secs: f64, total_secs: f64) -> f32 {
        if total_secs <= 0.0 {
            return 1.0;
        }
        (caption_secs / total_secs).clamp(0.0, 1.0) as f32
    }
}

/// Evaluates the overall quality of a set of captions.
pub struct CaptionQualityEvaluator;

impl CaptionQualityEvaluator {
    /// Evaluate the quality of `captions`, optionally against `reference` captions.
    ///
    /// When a reference is provided, WER-based accuracy and timing delay are
    /// computed. Without a reference, accuracy defaults to the mean confidence
    /// of the words (if available), and timing defaults to 1.0.
    #[must_use]
    pub fn evaluate(
        captions: &[CaptionSegment],
        reference: Option<&[CaptionSegment]>,
    ) -> CaptionQualityScore {
        if captions.is_empty() {
            return CaptionQualityScore::new(0.0, 0.0, 0.0, 0.0);
        }

        // --- Accuracy ---
        let accuracy = if let Some(ref_caps) = reference {
            let hyp_text: String = captions
                .iter()
                .map(|s| s.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            let ref_text: String = ref_caps
                .iter()
                .map(|s| s.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            let wer = AccuracyMetrics::wer(&ref_text, &hyp_text);
            AccuracyMetrics::wer_to_score(wer)
        } else {
            // Use mean word confidence as a proxy for accuracy
            let (sum, count) = captions.iter().fold((0.0f32, 0usize), |(s, c), seg| {
                let seg_conf: f32 = if seg.words.is_empty() {
                    0.8 // default if no word-level confidence available
                } else {
                    seg.words.iter().map(|w| w.confidence).sum::<f32>() / seg.words.len() as f32
                };
                (s + seg_conf, c + 1)
            });
            if count > 0 {
                sum / count as f32
            } else {
                0.0
            }
        };

        // --- Timing ---
        let timing = if let Some(ref_caps) = reference {
            // Average delay score over segments
            let scores: Vec<f32> = captions
                .iter()
                .zip(ref_caps.iter())
                .map(|(cap, rref)| TimingMetrics::delay_score(rref.start_ms, cap.start_ms))
                .collect();
            if scores.is_empty() {
                1.0
            } else {
                scores.iter().sum::<f32>() / scores.len() as f32
            }
        } else {
            1.0 // assume on-time when no reference
        };

        // --- Readability ---
        let readability_scores: Vec<f32> = captions
            .iter()
            .map(|s| ReadabilityMetrics::score(&s.text, s.duration_ms()))
            .collect();
        let readability = if readability_scores.is_empty() {
            0.0
        } else {
            readability_scores.iter().sum::<f32>() / readability_scores.len() as f32
        };

        // --- Completeness ---
        let total_duration: u64 = if let Some(ref_caps) = reference {
            ref_caps
                .iter()
                .map(super::live_captions::CaptionSegment::duration_ms)
                .sum()
        } else {
            // Estimate from hypothetical captions
            captions
                .iter()
                .map(super::live_captions::CaptionSegment::duration_ms)
                .sum()
        };
        let caption_duration: u64 = captions
            .iter()
            .map(super::live_captions::CaptionSegment::duration_ms)
            .sum();
        let completeness = CompletenessMetrics::coverage_pct(
            caption_duration as f64 / 1000.0,
            total_duration as f64 / 1000.0,
        );

        CaptionQualityScore::new(accuracy, timing, readability, completeness)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::live_captions::{CaptionSegment, CaptionWord};

    fn make_segment(text: &str, start_ms: u64, end_ms: u64) -> CaptionSegment {
        let word = CaptionWord::new(text, 0.9, start_ms, end_ms, None);
        CaptionSegment::from_words(vec![word])
            .expect("caption segment creation should succeed")
            .finalized()
    }

    #[test]
    fn test_score_grade_a() {
        let score = CaptionQualityScore::new(0.95, 0.95, 0.95, 0.95);
        assert_eq!(score.overall_grade(), 'A');
    }

    #[test]
    fn test_score_grade_b() {
        let score = CaptionQualityScore::new(0.80, 0.80, 0.80, 0.80);
        assert_eq!(score.overall_grade(), 'B');
    }

    #[test]
    fn test_score_grade_f() {
        let score = CaptionQualityScore::new(0.0, 0.0, 0.0, 0.0);
        assert_eq!(score.overall_grade(), 'F');
    }

    #[test]
    fn test_wer_identical() {
        let wer = AccuracyMetrics::wer("hello world", "hello world");
        assert_eq!(wer, 0.0);
    }

    #[test]
    fn test_wer_completely_wrong() {
        let wer = AccuracyMetrics::wer("hello world", "foo bar");
        // Both words substituted → WER = 2/2 = 1.0
        assert!((wer - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_wer_one_substitution() {
        let wer = AccuracyMetrics::wer("hello world foo", "hello earth foo");
        // 1 substitution in 3 words → WER ≈ 0.333
        assert!((wer - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_wer_empty_reference() {
        assert_eq!(AccuracyMetrics::wer("", ""), 0.0);
        assert_eq!(AccuracyMetrics::wer("", "hello"), 1.0);
    }

    #[test]
    fn test_wer_to_score() {
        assert_eq!(AccuracyMetrics::wer_to_score(0.0), 1.0);
        assert_eq!(AccuracyMetrics::wer_to_score(1.0), 0.0);
        assert!((AccuracyMetrics::wer_to_score(0.5) - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_timing_delay_zero() {
        let score = TimingMetrics::delay_score(1000, 1000);
        assert_eq!(score, 1.0);
    }

    #[test]
    fn test_timing_delay_early() {
        // Caption before reference → score 1.0
        let score = TimingMetrics::delay_score(5000, 3000);
        assert_eq!(score, 1.0);
    }

    #[test]
    fn test_timing_delay_8_seconds() {
        let score = TimingMetrics::delay_score(0, 8000);
        assert!(score.abs() < 0.01); // should be 0.0
    }

    #[test]
    fn test_timing_delay_partial() {
        let score = TimingMetrics::delay_score(0, 4000);
        assert!((score - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_readability_ideal_wpm() {
        // 3 words in ~1 second = 180 WPM → ideal score
        let score = ReadabilityMetrics::score("one two three", 1000);
        assert!(score > 0.5, "expected reasonable score, got {score}");
    }

    #[test]
    fn test_readability_zero_duration() {
        let score = ReadabilityMetrics::score("hello", 0);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_completeness_full_coverage() {
        let score = CompletenessMetrics::coverage_pct(100.0, 100.0);
        assert_eq!(score, 1.0);
    }

    #[test]
    fn test_completeness_half_coverage() {
        let score = CompletenessMetrics::coverage_pct(50.0, 100.0);
        assert!((score - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_completeness_zero_total() {
        let score = CompletenessMetrics::coverage_pct(0.0, 0.0);
        assert_eq!(score, 1.0);
    }

    #[test]
    fn test_evaluator_empty_captions() {
        let score = CaptionQualityEvaluator::evaluate(&[], None);
        assert_eq!(score.overall_grade(), 'F');
    }

    #[test]
    fn test_evaluator_without_reference() {
        let captions = vec![make_segment("Hello world.", 0, 2000)];
        let score = CaptionQualityEvaluator::evaluate(&captions, None);
        assert!(score.overall > 0.0);
    }

    #[test]
    fn test_evaluator_with_perfect_reference() {
        let captions = vec![make_segment("Hello world.", 0, 2000)];
        let reference = vec![make_segment("Hello world.", 0, 2000)];
        let score = CaptionQualityEvaluator::evaluate(&captions, Some(&reference));
        // Accuracy should be perfect
        assert!((score.accuracy - 1.0).abs() < 0.01);
    }
}
