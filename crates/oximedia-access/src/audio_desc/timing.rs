//! Audio description timing analysis and placement.

use crate::audio_desc::AudioDescriptionQuality;
use crate::error::{AccessError, AccessResult};
use serde::{Deserialize, Serialize};

/// Timing constraints for audio description placement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingConstraints {
    /// Minimum gap duration in milliseconds.
    pub min_gap_ms: i64,
    /// Minimum description duration in milliseconds.
    pub min_description_ms: i64,
    /// Maximum description duration in milliseconds.
    pub max_description_ms: i64,
    /// Minimum time before next dialogue in milliseconds.
    pub min_gap_after_ms: i64,
    /// Allow extended descriptions (pause video).
    pub allow_extended: bool,
}

impl Default for TimingConstraints {
    fn default() -> Self {
        Self {
            min_gap_ms: 1000,
            min_description_ms: 500,
            max_description_ms: 10000,
            min_gap_after_ms: 200,
            allow_extended: false,
        }
    }
}

impl TimingConstraints {
    /// Create constraints from quality level.
    #[must_use]
    pub fn from_quality(quality: AudioDescriptionQuality) -> Self {
        Self {
            min_gap_ms: quality.min_duration_ms(),
            min_description_ms: quality.min_duration_ms(),
            max_description_ms: 15000,
            min_gap_after_ms: quality.min_gap_after_ms(),
            allow_extended: false,
        }
    }

    /// Validate constraints.
    pub fn validate(&self) -> AccessResult<()> {
        if self.min_gap_ms <= 0 {
            return Err(AccessError::InvalidTiming(
                "Minimum gap must be positive".to_string(),
            ));
        }

        if self.min_description_ms <= 0 {
            return Err(AccessError::InvalidTiming(
                "Minimum description duration must be positive".to_string(),
            ));
        }

        if self.max_description_ms < self.min_description_ms {
            return Err(AccessError::InvalidTiming(
                "Maximum description must be >= minimum".to_string(),
            ));
        }

        Ok(())
    }
}

/// Analyzes audio timing to find suitable gaps for audio descriptions.
pub struct TimingAnalyzer {
    constraints: TimingConstraints,
}

impl TimingAnalyzer {
    /// Create a new timing analyzer.
    #[must_use]
    pub fn new(constraints: TimingConstraints) -> Self {
        Self { constraints }
    }

    /// Create analyzer with default constraints.
    #[must_use]
    pub fn default() -> Self {
        Self::new(TimingConstraints::default())
    }

    /// Analyze dialogue timing to find gaps.
    ///
    /// Returns gaps where audio descriptions can be placed.
    #[must_use]
    pub fn find_gaps(&self, dialogue_segments: &[DialogueSegment]) -> Vec<Gap> {
        let mut gaps = Vec::new();

        for i in 0..dialogue_segments.len().saturating_sub(1) {
            let current = &dialogue_segments[i];
            let next = &dialogue_segments[i + 1];

            let gap_start = current.end_time_ms;
            let gap_end = next.start_time_ms;
            let gap_duration = gap_end - gap_start;

            if gap_duration >= self.constraints.min_gap_ms {
                // Calculate available duration considering minimum gap after
                let available = gap_duration - self.constraints.min_gap_after_ms;

                if available >= self.constraints.min_description_ms {
                    gaps.push(Gap {
                        start_time_ms: gap_start,
                        end_time_ms: gap_end,
                        available_duration_ms: available,
                        context_before: Some(current.clone()),
                        context_after: Some(next.clone()),
                    });
                }
            }
        }

        gaps
    }

    /// Check if a description fits in the given gap.
    #[must_use]
    pub fn fits_in_gap(&self, description_duration_ms: i64, gap: &Gap) -> bool {
        description_duration_ms >= self.constraints.min_description_ms
            && description_duration_ms <= gap.available_duration_ms
            && description_duration_ms <= self.constraints.max_description_ms
    }

    /// Suggest optimal placement for a description.
    #[must_use]
    pub fn suggest_placement(
        &self,
        description_duration_ms: i64,
        gaps: &[Gap],
    ) -> Option<Placement> {
        for gap in gaps {
            if self.fits_in_gap(description_duration_ms, gap) {
                return Some(Placement {
                    start_time_ms: gap.start_time_ms,
                    end_time_ms: gap.start_time_ms + description_duration_ms,
                    gap_used: gap.clone(),
                    requires_pause: false,
                });
            }
        }

        // If extended descriptions are allowed and no gap found
        if self.constraints.allow_extended {
            if let Some(best_gap) = gaps.first() {
                return Some(Placement {
                    start_time_ms: best_gap.start_time_ms,
                    end_time_ms: best_gap.start_time_ms + description_duration_ms,
                    gap_used: best_gap.clone(),
                    requires_pause: true,
                });
            }
        }

        None
    }

    /// Calculate timing score for a placement (0.0 to 1.0, higher is better).
    #[must_use]
    pub fn calculate_score(&self, description_duration_ms: i64, gap: &Gap) -> f64 {
        if !self.fits_in_gap(description_duration_ms, gap) {
            return 0.0;
        }

        let utilization = description_duration_ms as f64 / gap.available_duration_ms as f64;
        let gap_comfort = (gap.available_duration_ms - description_duration_ms) as f64
            / self.constraints.min_gap_after_ms as f64;

        // Score based on utilization (prefer 60-80% utilization) and comfort margin
        let utilization_score = if (0.6..=0.8).contains(&utilization) {
            1.0
        } else if utilization < 0.6 {
            utilization / 0.6
        } else {
            (1.0 - utilization) / 0.2
        };

        let comfort_score = gap_comfort.min(2.0) / 2.0;

        (utilization_score * 0.6 + comfort_score * 0.4).clamp(0.0, 1.0)
    }

    /// Get constraints.
    #[must_use]
    pub const fn constraints(&self) -> &TimingConstraints {
        &self.constraints
    }
}

/// A gap in dialogue where audio description can be placed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gap {
    /// Start time of the gap.
    pub start_time_ms: i64,
    /// End time of the gap.
    pub end_time_ms: i64,
    /// Available duration for description (accounting for min gap after).
    pub available_duration_ms: i64,
    /// Context from dialogue before the gap.
    pub context_before: Option<DialogueSegment>,
    /// Context from dialogue after the gap.
    pub context_after: Option<DialogueSegment>,
}

impl Gap {
    /// Get total gap duration.
    #[must_use]
    pub const fn duration_ms(&self) -> i64 {
        self.end_time_ms - self.start_time_ms
    }
}

/// A dialogue segment (used for gap analysis).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DialogueSegment {
    /// Start time in milliseconds.
    pub start_time_ms: i64,
    /// End time in milliseconds.
    pub end_time_ms: i64,
    /// Speaker name (optional).
    pub speaker: Option<String>,
    /// Dialogue text (optional).
    pub text: Option<String>,
}

impl DialogueSegment {
    /// Create a new dialogue segment.
    #[must_use]
    pub fn new(start_time_ms: i64, end_time_ms: i64) -> Self {
        Self {
            start_time_ms,
            end_time_ms,
            speaker: None,
            text: None,
        }
    }

    /// Get duration.
    #[must_use]
    pub const fn duration_ms(&self) -> i64 {
        self.end_time_ms - self.start_time_ms
    }
}

/// Suggested placement for an audio description.
#[derive(Debug, Clone)]
pub struct Placement {
    /// Start time in milliseconds.
    pub start_time_ms: i64,
    /// End time in milliseconds.
    pub end_time_ms: i64,
    /// Gap that was used.
    pub gap_used: Gap,
    /// Whether video pause is required.
    pub requires_pause: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constraints_default() {
        let constraints = TimingConstraints::default();
        assert_eq!(constraints.min_gap_ms, 1000);
        assert!(constraints.validate().is_ok());
    }

    #[test]
    fn test_constraints_from_quality() {
        let constraints = TimingConstraints::from_quality(AudioDescriptionQuality::Professional);
        assert!(constraints.min_gap_ms >= 2000);
    }

    #[test]
    fn test_find_gaps() {
        let analyzer = TimingAnalyzer::default();

        let dialogue = vec![
            DialogueSegment::new(0, 2000),
            DialogueSegment::new(5000, 7000),
            DialogueSegment::new(10000, 12000),
        ];

        let gaps = analyzer.find_gaps(&dialogue);
        assert_eq!(gaps.len(), 2);

        assert_eq!(gaps[0].start_time_ms, 2000);
        assert_eq!(gaps[0].end_time_ms, 5000);
    }

    #[test]
    fn test_fits_in_gap() {
        let analyzer = TimingAnalyzer::new(TimingConstraints {
            min_gap_ms: 1000,
            min_description_ms: 500,
            max_description_ms: 10000,
            min_gap_after_ms: 200,
            allow_extended: false,
        });

        let gap = Gap {
            start_time_ms: 2000,
            end_time_ms: 5000,
            available_duration_ms: 2800,
            context_before: None,
            context_after: None,
        };

        assert!(analyzer.fits_in_gap(2000, &gap));
        assert!(!analyzer.fits_in_gap(3000, &gap));
        assert!(!analyzer.fits_in_gap(400, &gap));
    }

    #[test]
    fn test_calculate_score() {
        let analyzer = TimingAnalyzer::default();

        let gap = Gap {
            start_time_ms: 0,
            end_time_ms: 3000,
            available_duration_ms: 2800,
            context_before: None,
            context_after: None,
        };

        let score1 = analyzer.calculate_score(2000, &gap);
        let score2 = analyzer.calculate_score(1000, &gap);

        assert!(score1 > score2);
    }

    #[test]
    fn test_suggest_placement() {
        let analyzer = TimingAnalyzer::default();

        let gaps = vec![Gap {
            start_time_ms: 2000,
            end_time_ms: 5000,
            available_duration_ms: 2800,
            context_before: None,
            context_after: None,
        }];

        let placement = analyzer.suggest_placement(2000, &gaps);
        assert!(placement.is_some());

        let placement = placement.expect("placement should be valid");
        assert_eq!(placement.start_time_ms, 2000);
        assert_eq!(placement.end_time_ms, 4000);
        assert!(!placement.requires_pause);
    }
}
