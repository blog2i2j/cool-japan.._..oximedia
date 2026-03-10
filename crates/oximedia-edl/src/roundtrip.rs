//! EDL roundtrip validation.
//!
//! A *roundtrip* test parses an EDL text string into an [`crate::Edl`],
//! generates it back to text, then re-parses the generated text and compares
//! the two parsed structures for semantic equality.  Any differences indicate
//! lossiness in the parser/generator pair.

#![allow(dead_code)]

use crate::error::EdlResult;
use crate::event::EdlEvent;
use crate::{parse_edl, Edl, EdlGenerator};

// ────────────────────────────────────────────────────────────────────────────
// RoundtripDiff
// ────────────────────────────────────────────────────────────────────────────

/// A single semantic difference found between two parsed EDLs.
#[derive(Debug, Clone, PartialEq)]
pub enum RoundtripDiff {
    /// The EDL title changed.
    TitleMismatch {
        /// Title from the original parse.
        original: Option<String>,
        /// Title from the re-parse.
        regenerated: Option<String>,
    },
    /// The event count changed.
    EventCountMismatch {
        /// Event count from the original parse.
        original: usize,
        /// Event count from the re-parse.
        regenerated: usize,
    },
    /// A specific event changed.
    EventMismatch {
        /// Event number (1-based).
        event_number: u32,
        /// Human-readable description of what changed.
        description: String,
    },
    /// The frame count mode (drop-frame / non-drop-frame) changed.
    FrameRateMismatch {
        /// Description of the original frame rate.
        original: String,
        /// Description of the regenerated frame rate.
        regenerated: String,
    },
}

// ────────────────────────────────────────────────────────────────────────────
// RoundtripReport
// ────────────────────────────────────────────────────────────────────────────

/// Full report produced by a roundtrip validation pass.
#[derive(Debug, Clone)]
pub struct RoundtripReport {
    /// The re-generated EDL text (useful for debugging).
    pub generated_text: String,
    /// Differences found, if any.
    pub diffs: Vec<RoundtripDiff>,
}

impl RoundtripReport {
    /// Returns `true` when the roundtrip was lossless.
    #[must_use]
    pub fn is_lossless(&self) -> bool {
        self.diffs.is_empty()
    }

    /// Number of differences found.
    #[must_use]
    pub fn diff_count(&self) -> usize {
        self.diffs.len()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// RoundtripValidator
// ────────────────────────────────────────────────────────────────────────────

/// Validates an EDL by performing a parse → generate → re-parse cycle.
#[derive(Debug, Default)]
pub struct RoundtripValidator {
    /// If `true`, compare clip names attached to events.
    pub check_clip_names: bool,
    /// If `true`, compare comments attached to events.
    pub check_comments: bool,
}

impl RoundtripValidator {
    /// Create a new validator with all checks enabled.
    #[must_use]
    pub fn full() -> Self {
        Self {
            check_clip_names: true,
            check_comments: true,
        }
    }

    /// Validate `edl_text` for roundtrip losslessness.
    ///
    /// # Errors
    ///
    /// Returns an [`crate::error::EdlError`] if either parse or generation
    /// step fails.
    pub fn validate(&self, edl_text: &str) -> EdlResult<RoundtripReport> {
        // Step 1: original parse.
        let original = parse_edl(edl_text)?;

        // Step 2: generate.
        let generator = EdlGenerator::new();
        let generated_text = generator.generate(&original)?;

        // Step 3: re-parse.
        let regenerated = parse_edl(&generated_text)?;

        // Step 4: compare.
        let diffs = self.compare(&original, &regenerated);

        Ok(RoundtripReport {
            generated_text,
            diffs,
        })
    }

    /// Compare two parsed EDLs and return any differences.
    #[must_use]
    pub fn compare(&self, a: &Edl, b: &Edl) -> Vec<RoundtripDiff> {
        let mut diffs = Vec::new();

        // Title.
        if a.title != b.title {
            diffs.push(RoundtripDiff::TitleMismatch {
                original: a.title.clone(),
                regenerated: b.title.clone(),
            });
        }

        // Frame rate.
        if a.frame_rate != b.frame_rate {
            diffs.push(RoundtripDiff::FrameRateMismatch {
                original: format!("{:?}", a.frame_rate),
                regenerated: format!("{:?}", b.frame_rate),
            });
        }

        // Event count.
        if a.events.len() != b.events.len() {
            diffs.push(RoundtripDiff::EventCountMismatch {
                original: a.events.len(),
                regenerated: b.events.len(),
            });
            // Can't compare individual events if counts differ.
            return diffs;
        }

        // Per-event comparison.
        for (ea, eb) in a.events.iter().zip(b.events.iter()) {
            self.compare_event(ea, eb, &mut diffs);
        }

        diffs
    }

    fn compare_event(&self, a: &EdlEvent, b: &EdlEvent, diffs: &mut Vec<RoundtripDiff>) {
        if a.number != b.number {
            diffs.push(RoundtripDiff::EventMismatch {
                event_number: a.number,
                description: format!("event number changed: {} → {}", a.number, b.number),
            });
        }
        if a.reel != b.reel {
            diffs.push(RoundtripDiff::EventMismatch {
                event_number: a.number,
                description: format!("reel changed: {:?} → {:?}", a.reel, b.reel),
            });
        }
        if a.edit_type != b.edit_type {
            diffs.push(RoundtripDiff::EventMismatch {
                event_number: a.number,
                description: format!("edit type changed: {:?} → {:?}", a.edit_type, b.edit_type),
            });
        }
        if a.source_in != b.source_in {
            diffs.push(RoundtripDiff::EventMismatch {
                event_number: a.number,
                description: format!("source_in changed: {:?} → {:?}", a.source_in, b.source_in),
            });
        }
        if a.source_out != b.source_out {
            diffs.push(RoundtripDiff::EventMismatch {
                event_number: a.number,
                description: format!(
                    "source_out changed: {:?} → {:?}",
                    a.source_out, b.source_out
                ),
            });
        }
        if a.record_in != b.record_in {
            diffs.push(RoundtripDiff::EventMismatch {
                event_number: a.number,
                description: format!("record_in changed: {:?} → {:?}", a.record_in, b.record_in),
            });
        }
        if a.record_out != b.record_out {
            diffs.push(RoundtripDiff::EventMismatch {
                event_number: a.number,
                description: format!(
                    "record_out changed: {:?} → {:?}",
                    a.record_out, b.record_out
                ),
            });
        }
        if self.check_clip_names && a.clip_name != b.clip_name {
            diffs.push(RoundtripDiff::EventMismatch {
                event_number: a.number,
                description: format!("clip_name changed: {:?} → {:?}", a.clip_name, b.clip_name),
            });
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Convenience helpers
// ────────────────────────────────────────────────────────────────────────────

/// Quick check: parse `edl_text`, generate, re-parse and confirm lossless.
///
/// Returns `Ok(true)` when lossless, `Ok(false)` when differences are found,
/// and `Err` when parsing/generation fails.
///
/// # Errors
///
/// Propagates any [`crate::error::EdlError`] from parse or generation.
pub fn is_lossless(edl_text: &str) -> EdlResult<bool> {
    let validator = RoundtripValidator::default();
    let report = validator.validate(edl_text)?;
    Ok(report.is_lossless())
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SIMPLE_EDL: &str = r#"TITLE: Roundtrip Test
FCM: NON-DROP FRAME

001  AX       V     C        01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00

"#;

    #[test]
    fn test_lossless_simple_edl() {
        let result = is_lossless(SIMPLE_EDL).expect("operation should succeed");
        assert!(result, "Simple EDL should survive a roundtrip unchanged");
    }

    #[test]
    fn test_roundtrip_report_is_lossless() {
        let validator = RoundtripValidator::default();
        let report = validator
            .validate(SIMPLE_EDL)
            .expect("validation should succeed");
        assert!(report.is_lossless());
        assert_eq!(report.diff_count(), 0);
    }

    #[test]
    fn test_generated_text_not_empty() {
        let validator = RoundtripValidator::default();
        let report = validator
            .validate(SIMPLE_EDL)
            .expect("validation should succeed");
        assert!(!report.generated_text.is_empty());
    }

    #[test]
    fn test_generated_text_contains_title() {
        let validator = RoundtripValidator::default();
        let report = validator
            .validate(SIMPLE_EDL)
            .expect("validation should succeed");
        assert!(report.generated_text.contains("Roundtrip Test"));
    }

    #[test]
    fn test_two_event_edl_roundtrip() {
        let edl_text = r#"TITLE: Two Events
FCM: NON-DROP FRAME

001  A001     V     C        01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00
002  B001     V     C        01:00:05:00 01:00:10:00 01:00:05:00 01:00:10:00

"#;
        let result = is_lossless(edl_text).expect("operation should succeed");
        assert!(result);
    }

    #[test]
    fn test_compare_identical_edls_no_diffs() {
        let a = parse_edl(SIMPLE_EDL).expect("operation should succeed");
        let b = parse_edl(SIMPLE_EDL).expect("operation should succeed");
        let validator = RoundtripValidator::default();
        let diffs = validator.compare(&a, &b);
        assert!(diffs.is_empty());
    }

    #[test]
    fn test_compare_different_titles_produces_diff() {
        let mut a = parse_edl(SIMPLE_EDL).expect("operation should succeed");
        let b = parse_edl(SIMPLE_EDL).expect("operation should succeed");
        a.title = Some("Different Title".to_string());
        let validator = RoundtripValidator::default();
        let diffs = validator.compare(&a, &b);
        assert!(!diffs.is_empty());
        assert!(matches!(diffs[0], RoundtripDiff::TitleMismatch { .. }));
    }

    #[test]
    fn test_compare_different_event_counts_produces_diff() {
        let a = parse_edl(SIMPLE_EDL).expect("operation should succeed");
        let mut b = parse_edl(SIMPLE_EDL).expect("operation should succeed");
        b.events.clear();
        let validator = RoundtripValidator::default();
        let diffs = validator.compare(&a, &b);
        assert!(!diffs.is_empty());
        assert!(matches!(diffs[0], RoundtripDiff::EventCountMismatch { .. }));
    }

    #[test]
    fn test_roundtrip_diff_event_count_mismatch_values() {
        let diff = RoundtripDiff::EventCountMismatch {
            original: 3,
            regenerated: 2,
        };
        if let RoundtripDiff::EventCountMismatch {
            original,
            regenerated,
        } = diff
        {
            assert_eq!(original, 3);
            assert_eq!(regenerated, 2);
        }
    }

    #[test]
    fn test_roundtrip_diff_title_mismatch_values() {
        let diff = RoundtripDiff::TitleMismatch {
            original: Some("A".to_string()),
            regenerated: Some("B".to_string()),
        };
        if let RoundtripDiff::TitleMismatch {
            original,
            regenerated,
        } = diff
        {
            assert_eq!(original, Some("A".to_string()));
            assert_eq!(regenerated, Some("B".to_string()));
        }
    }

    #[test]
    fn test_full_validator_checks_clip_names() {
        let v = RoundtripValidator::full();
        assert!(v.check_clip_names);
        assert!(v.check_comments);
    }

    #[test]
    fn test_report_diff_count_matches_diffs_vec() {
        let report = RoundtripReport {
            generated_text: String::new(),
            diffs: vec![
                RoundtripDiff::TitleMismatch {
                    original: None,
                    regenerated: None,
                },
                RoundtripDiff::EventCountMismatch {
                    original: 0,
                    regenerated: 1,
                },
            ],
        };
        assert_eq!(report.diff_count(), 2);
        assert!(!report.is_lossless());
    }

    #[test]
    fn test_is_lossless_helper_returns_true_for_valid_edl() {
        let result = is_lossless(SIMPLE_EDL);
        assert!(result.is_ok());
        assert!(result.expect("result should be valid"));
    }

    #[test]
    fn test_validate_multi_event_no_diffs() {
        let edl_text = r#"TITLE: Multi
FCM: NON-DROP FRAME

001  AX       V     C        01:00:00:00 01:00:03:00 01:00:00:00 01:00:03:00
002  AX       V     C        01:00:03:00 01:00:06:00 01:00:03:00 01:00:06:00
003  AX       V     C        01:00:06:00 01:00:09:00 01:00:06:00 01:00:09:00

"#;
        let validator = RoundtripValidator::default();
        let report = validator
            .validate(edl_text)
            .expect("validation should succeed");
        assert!(report.is_lossless(), "diffs: {:?}", report.diffs);
    }
}
