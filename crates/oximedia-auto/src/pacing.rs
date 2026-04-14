//! Shot pacing analysis for automated video editing.
//!
//! The [`PacingAnalyzer`] classifies the pacing of an edit sequence as
//! **slow**, **medium**, or **fast** based on the distribution of shot
//! durations. It also computes descriptive statistics useful for downstream
//! edit decisions.
//!
//! # Classification thresholds
//!
//! The classifier uses average shot duration:
//!
//! | Average shot duration | Classification |
//! |-----------------------|----------------|
//! | > 5 000 ms            | Slow           |
//! | 1 500 ms – 5 000 ms   | Medium         |
//! | < 1 500 ms            | Fast           |
//!
//! The thresholds are chosen to match common broadcast and social-media
//! editing conventions:
//! - Documentaries / drama → Slow
//! - Corporate / narrative → Medium
//! - Action / sports / social-media → Fast
//!
//! # Example
//!
//! ```rust
//! use oximedia_auto::pacing::{PacingAnalyzer, PacingClass};
//!
//! // Average is 4 700 ms → falls in the Medium range (1 500–5 000 ms).
//! let durations = [4_000_u64, 3_500, 5_200, 6_000, 4_800];
//! let report = PacingAnalyzer::analyze(&durations);
//!
//! assert_eq!(report.classification, PacingClass::Medium);
//! assert!(report.average_ms > 0.0);
//! ```

#![allow(dead_code, clippy::cast_precision_loss)]

// ---------------------------------------------------------------------------
// PacingClass
// ---------------------------------------------------------------------------

/// Broad pacing classification for an edit sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PacingClass {
    /// Average shot duration > 5 000 ms (documentary, drama).
    Slow,
    /// Average shot duration 1 500–5 000 ms (corporate, narrative).
    Medium,
    /// Average shot duration < 1 500 ms (action, sports, social media).
    Fast,
}

impl PacingClass {
    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Slow => "slow",
            Self::Medium => "medium",
            Self::Fast => "fast",
        }
    }
}

impl std::fmt::Display for PacingClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

// ---------------------------------------------------------------------------
// PacingReport
// ---------------------------------------------------------------------------

/// Detailed pacing analysis report for a sequence of shots.
#[derive(Debug, Clone)]
pub struct PacingReport {
    /// Overall pacing classification.
    pub classification: PacingClass,
    /// Arithmetic mean of all shot durations (ms).
    pub average_ms: f64,
    /// Median shot duration (ms).
    pub median_ms: f64,
    /// Minimum shot duration (ms).
    pub min_ms: u64,
    /// Maximum shot duration (ms).
    pub max_ms: u64,
    /// Standard deviation of shot durations (ms).
    pub std_dev_ms: f64,
    /// Total number of shots analysed.
    pub shot_count: usize,
    /// Total duration of all shots combined (ms).
    pub total_ms: u64,
    /// Fraction of shots classified as fast (< 1 500 ms).
    pub fast_fraction: f64,
    /// Fraction of shots classified as slow (> 5 000 ms).
    pub slow_fraction: f64,
}

// ---------------------------------------------------------------------------
// PacingAnalyzer
// ---------------------------------------------------------------------------

/// Shot pacing analyzer.
///
/// All methods are stateless (take `&[u64]` directly) so no persistent
/// instance is required.
pub struct PacingAnalyzer;

impl PacingAnalyzer {
    // Pacing classification thresholds (milliseconds).
    const FAST_THRESHOLD_MS: u64 = 1_500;
    const SLOW_THRESHOLD_MS: u64 = 5_000;

    /// Analyse a sequence of shot durations and return a [`PacingReport`].
    ///
    /// Returns a zeroed report (with classification [`PacingClass::Medium`])
    /// for empty input.
    #[must_use]
    pub fn analyze(shot_durations_ms: &[u64]) -> PacingReport {
        let n = shot_durations_ms.len();
        if n == 0 {
            return PacingReport {
                classification: PacingClass::Medium,
                average_ms: 0.0,
                median_ms: 0.0,
                min_ms: 0,
                max_ms: 0,
                std_dev_ms: 0.0,
                shot_count: 0,
                total_ms: 0,
                fast_fraction: 0.0,
                slow_fraction: 0.0,
            };
        }

        let total_ms: u64 = shot_durations_ms.iter().sum();
        let average_ms = total_ms as f64 / n as f64;

        let min_ms = *shot_durations_ms.iter().min().unwrap_or(&0);
        let max_ms = *shot_durations_ms.iter().max().unwrap_or(&0);

        // Median.
        let mut sorted = shot_durations_ms.to_vec();
        sorted.sort_unstable();
        let median_ms = if n % 2 == 1 {
            sorted[n / 2] as f64
        } else {
            (sorted[n / 2 - 1] + sorted[n / 2]) as f64 / 2.0
        };

        // Standard deviation.
        let variance = shot_durations_ms
            .iter()
            .map(|&d| (d as f64 - average_ms).powi(2))
            .sum::<f64>()
            / n as f64;
        let std_dev_ms = variance.sqrt();

        // Fraction slow / fast.
        let fast_count = shot_durations_ms
            .iter()
            .filter(|&&d| d < Self::FAST_THRESHOLD_MS)
            .count();
        let slow_count = shot_durations_ms
            .iter()
            .filter(|&&d| d > Self::SLOW_THRESHOLD_MS)
            .count();
        let fast_fraction = fast_count as f64 / n as f64;
        let slow_fraction = slow_count as f64 / n as f64;

        // Overall classification using average duration.
        let classification = if average_ms < Self::FAST_THRESHOLD_MS as f64 {
            PacingClass::Fast
        } else if average_ms > Self::SLOW_THRESHOLD_MS as f64 {
            PacingClass::Slow
        } else {
            PacingClass::Medium
        };

        PacingReport {
            classification,
            average_ms,
            median_ms,
            min_ms,
            max_ms,
            std_dev_ms,
            shot_count: n,
            total_ms,
            fast_fraction,
            slow_fraction,
        }
    }

    /// Return the pacing class for a single shot duration in milliseconds.
    #[must_use]
    pub fn classify_shot(duration_ms: u64) -> PacingClass {
        if duration_ms < Self::FAST_THRESHOLD_MS {
            PacingClass::Fast
        } else if duration_ms > Self::SLOW_THRESHOLD_MS {
            PacingClass::Slow
        } else {
            PacingClass::Medium
        }
    }

    /// Suggest a target average shot duration (ms) for a given pacing preset.
    #[must_use]
    pub fn suggested_duration_ms(class: PacingClass) -> u64 {
        match class {
            PacingClass::Fast => 800,
            PacingClass::Medium => 3_000,
            PacingClass::Slow => 7_000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_input_returns_medium() {
        let report = PacingAnalyzer::analyze(&[]);
        assert_eq!(report.classification, PacingClass::Medium);
        assert_eq!(report.shot_count, 0);
        assert_eq!(report.total_ms, 0);
    }

    #[test]
    fn test_all_fast_shots() {
        let durations = vec![500_u64; 10];
        let report = PacingAnalyzer::analyze(&durations);
        assert_eq!(report.classification, PacingClass::Fast);
        assert!((report.average_ms - 500.0).abs() < 1e-6);
        assert!((report.fast_fraction - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_all_slow_shots() {
        let durations = vec![6_000_u64; 8];
        let report = PacingAnalyzer::analyze(&durations);
        assert_eq!(report.classification, PacingClass::Slow);
        assert!((report.average_ms - 6_000.0).abs() < 1e-6);
        assert!((report.slow_fraction - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_medium_pacing() {
        let durations = vec![2_000_u64, 3_000, 2_500, 4_000];
        let report = PacingAnalyzer::analyze(&durations);
        assert_eq!(report.classification, PacingClass::Medium);
    }

    #[test]
    fn test_statistics_computed() {
        let durations = vec![1_000_u64, 2_000, 3_000, 4_000, 5_000];
        let report = PacingAnalyzer::analyze(&durations);
        assert_eq!(report.shot_count, 5);
        assert_eq!(report.total_ms, 15_000);
        assert!((report.average_ms - 3_000.0).abs() < 1e-6);
        assert!((report.median_ms - 3_000.0).abs() < 1e-6);
        assert_eq!(report.min_ms, 1_000);
        assert_eq!(report.max_ms, 5_000);
    }

    #[test]
    fn test_std_dev_correct() {
        // Uniform distribution [1000, 2000, 3000] → mean=2000, variance=2/3*10^6.
        let durations = vec![1_000_u64, 2_000, 3_000];
        let report = PacingAnalyzer::analyze(&durations);
        let expected_std = ((2.0 / 3.0) * 1_000_000_f64).sqrt();
        assert!(
            (report.std_dev_ms - expected_std).abs() < 1.0,
            "std_dev: expected ~{expected_std:.1}, got {:.1}",
            report.std_dev_ms
        );
    }

    #[test]
    fn test_median_even_count() {
        let durations = vec![1_000_u64, 2_000, 3_000, 4_000];
        let report = PacingAnalyzer::analyze(&durations);
        assert!((report.median_ms - 2_500.0).abs() < 1e-6);
    }

    #[test]
    fn test_classify_shot_boundaries() {
        assert_eq!(PacingAnalyzer::classify_shot(500), PacingClass::Fast);
        assert_eq!(PacingAnalyzer::classify_shot(1_500), PacingClass::Medium);
        assert_eq!(PacingAnalyzer::classify_shot(3_000), PacingClass::Medium);
        assert_eq!(PacingAnalyzer::classify_shot(5_000), PacingClass::Medium);
        assert_eq!(PacingAnalyzer::classify_shot(7_000), PacingClass::Slow);
    }

    #[test]
    fn test_mixed_pacing_fractions() {
        // 2 fast, 2 medium, 2 slow.
        let durations = vec![500_u64, 800, 2_500, 3_000, 6_000, 8_000];
        let report = PacingAnalyzer::analyze(&durations);
        assert!((report.fast_fraction - 2.0 / 6.0).abs() < 1e-6);
        assert!((report.slow_fraction - 2.0 / 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_label_display() {
        assert_eq!(PacingClass::Fast.label(), "fast");
        assert_eq!(PacingClass::Medium.label(), "medium");
        assert_eq!(PacingClass::Slow.label(), "slow");
        assert_eq!(format!("{}", PacingClass::Fast), "fast");
    }

    #[test]
    fn test_suggested_duration() {
        assert_eq!(
            PacingAnalyzer::suggested_duration_ms(PacingClass::Fast),
            800
        );
        assert_eq!(
            PacingAnalyzer::suggested_duration_ms(PacingClass::Medium),
            3_000
        );
        assert_eq!(
            PacingAnalyzer::suggested_duration_ms(PacingClass::Slow),
            7_000
        );
    }
}
