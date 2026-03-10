//! EDL validation and compliance checking.
//!
//! This module provides validation functionality to ensure EDLs conform
//! to CMX 3600 and other EDL format specifications.

use crate::error::{EdlError, EdlResult};
use crate::event::{EditType, EdlEvent};
use crate::Edl;
use std::collections::HashSet;

/// Validation level for EDL validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationLevel {
    /// Strict validation (CMX 3600 compliance).
    Strict,
    /// Standard validation (common issues).
    Standard,
    /// Lenient validation (minimal checks).
    Lenient,
}

/// EDL validator for checking compliance and consistency.
#[derive(Debug)]
pub struct EdlValidator {
    /// Validation level.
    pub level: ValidationLevel,
    /// Check for event overlaps.
    pub check_overlaps: bool,
    /// Check for timeline gaps.
    pub check_gaps: bool,
    /// Check timecode validity.
    pub check_timecodes: bool,
    /// Check event numbering.
    pub check_event_numbers: bool,
    /// Maximum allowed gap in frames (0 = no gaps allowed).
    pub max_gap_frames: u64,
}

impl EdlValidator {
    /// Create a new EDL validator with strict settings.
    #[must_use]
    pub const fn strict() -> Self {
        Self {
            level: ValidationLevel::Strict,
            check_overlaps: true,
            check_gaps: true,
            check_timecodes: true,
            check_event_numbers: true,
            max_gap_frames: 0,
        }
    }

    /// Create a new EDL validator with standard settings.
    #[must_use]
    pub const fn standard() -> Self {
        Self {
            level: ValidationLevel::Standard,
            check_overlaps: true,
            check_gaps: false,
            check_timecodes: true,
            check_event_numbers: true,
            max_gap_frames: 1,
        }
    }

    /// Create a new EDL validator with lenient settings.
    #[must_use]
    pub const fn lenient() -> Self {
        Self {
            level: ValidationLevel::Lenient,
            check_overlaps: false,
            check_gaps: false,
            check_timecodes: true,
            check_event_numbers: false,
            max_gap_frames: 100,
        }
    }

    /// Validate an EDL.
    ///
    /// # Errors
    ///
    /// Returns an error if the EDL fails validation.
    pub fn validate(&self, edl: &Edl) -> EdlResult<ValidationReport> {
        let mut report = ValidationReport::new();

        // Check if EDL has events
        if edl.events.is_empty() {
            report.add_warning("EDL has no events".to_string());
        }

        // Validate each event
        for event in &edl.events {
            if let Err(e) = self.validate_event(event) {
                report.add_error(format!("Event {}: {e}", event.number));
            }
        }

        // Check event numbering
        if self.check_event_numbers {
            if let Err(e) = self.check_event_numbering(edl) {
                report.add_error(format!("Event numbering: {e}"));
            }
        }

        // Check for overlaps
        if self.check_overlaps {
            if let Err(e) = self.check_event_overlaps(edl) {
                report.add_error(format!("Overlap detection: {e}"));
            }
        }

        // Check for gaps
        if self.check_gaps {
            if let Err(e) = self.check_timeline_gaps(edl) {
                report.add_error(format!("Gap detection: {e}"));
            }
        }

        // Validate reel table
        if let Err(e) = edl.reel_table.validate() {
            report.add_error(format!("Reel table: {e}"));
        }

        if report.has_errors() {
            Err(EdlError::ValidationError(format!(
                "Validation failed with {} errors",
                report.errors.len()
            )))
        } else {
            Ok(report)
        }
    }

    /// Validate a single event.
    fn validate_event(&self, event: &EdlEvent) -> EdlResult<()> {
        // Basic event validation
        event.validate()?;

        if self.check_timecodes {
            // Check that timecodes are in valid ranges
            if event.source_in.hours() > 23 {
                return Err(EdlError::InvalidTimecode {
                    line: 0,
                    message: format!("Invalid source in hours: {}", event.source_in.hours()),
                });
            }

            if event.record_in.hours() > 23 {
                return Err(EdlError::InvalidTimecode {
                    line: 0,
                    message: format!("Invalid record in hours: {}", event.record_in.hours()),
                });
            }
        }

        // Strict mode checks
        if self.level == ValidationLevel::Strict {
            // Check that reel names are <= 8 characters
            if event.reel.len() > 8 {
                return Err(EdlError::InvalidReelName(format!(
                    "Reel name too long (max 8 characters): {}",
                    event.reel
                )));
            }

            // Check that dissolves and wipes have transition durations
            if matches!(event.edit_type, EditType::Dissolve | EditType::Wipe)
                && event.transition_duration.is_none()
            {
                return Err(EdlError::MissingField(format!(
                    "Event {} missing transition duration",
                    event.number
                )));
            }
        }

        Ok(())
    }

    /// Check event numbering is sequential.
    fn check_event_numbering(&self, edl: &Edl) -> EdlResult<()> {
        let mut expected_num = 1;
        let mut seen_numbers = HashSet::new();

        for event in &edl.events {
            // Check for duplicates
            if !seen_numbers.insert(event.number) {
                return Err(EdlError::ValidationError(format!(
                    "Duplicate event number: {}",
                    event.number
                )));
            }

            // Check for sequential numbering (strict mode only)
            if self.level == ValidationLevel::Strict {
                if event.number != expected_num {
                    return Err(EdlError::ValidationError(format!(
                        "Non-sequential event numbering: expected {expected_num}, got {}",
                        event.number
                    )));
                }
                expected_num += 1;
            }
        }

        Ok(())
    }

    /// Check for event overlaps.
    fn check_event_overlaps(&self, edl: &Edl) -> EdlResult<()> {
        for i in 0..edl.events.len() {
            for j in (i + 1)..edl.events.len() {
                if edl.events[i].overlaps_with(&edl.events[j]) {
                    return Err(EdlError::event_overlap(
                        edl.events[i].number,
                        edl.events[j].number,
                    ));
                }
            }
        }
        Ok(())
    }

    /// Check for timeline gaps.
    fn check_timeline_gaps(&self, edl: &Edl) -> EdlResult<()> {
        // Sort events by record in timecode
        let mut sorted_events: Vec<&EdlEvent> = edl.events.iter().collect();
        sorted_events.sort_by_key(|e| e.record_in.to_frames());

        for i in 0..(sorted_events.len().saturating_sub(1)) {
            let current = sorted_events[i];
            let next = sorted_events[i + 1];

            // Check if same track
            if !current.track.overlaps_with(&next.track) {
                continue;
            }

            let gap = next.record_in.to_frames() as i64 - current.record_out.to_frames() as i64;

            if gap > self.max_gap_frames as i64 {
                return Err(EdlError::timeline_gap(current.number, next.number));
            } else if gap < 0 {
                return Err(EdlError::event_overlap(current.number, next.number));
            }
        }

        Ok(())
    }
}

impl Default for EdlValidator {
    fn default() -> Self {
        Self::standard()
    }
}

/// Validation report containing errors and warnings.
#[derive(Debug, Clone, Default)]
pub struct ValidationReport {
    /// Validation errors.
    pub errors: Vec<String>,
    /// Validation warnings.
    pub warnings: Vec<String>,
}

impl ValidationReport {
    /// Create a new empty validation report.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Add an error to the report.
    pub fn add_error(&mut self, error: String) {
        self.errors.push(error);
    }

    /// Add a warning to the report.
    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }

    /// Check if the report has any errors.
    #[must_use]
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Check if the report has any warnings.
    #[must_use]
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }

    /// Get the number of errors.
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.errors.len()
    }

    /// Get the number of warnings.
    #[must_use]
    pub fn warning_count(&self) -> usize {
        self.warnings.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::TrackType;
    use crate::timecode::{EdlFrameRate, EdlTimecode};
    use crate::EdlFormat;

    #[test]
    fn test_validate_empty_edl() {
        let edl = Edl::new(EdlFormat::Cmx3600);
        let validator = EdlValidator::lenient();
        let report = validator.validate(&edl).expect("validation should succeed");
        assert!(report.has_warnings());
    }

    #[test]
    fn test_validate_simple_edl() {
        let mut edl = Edl::new(EdlFormat::Cmx3600);
        edl.set_frame_rate(EdlFrameRate::Fps25);

        let tc1 = EdlTimecode::new(1, 0, 0, 0, EdlFrameRate::Fps25).expect("failed to create");
        let tc2 = EdlTimecode::new(1, 0, 5, 0, EdlFrameRate::Fps25).expect("failed to create");

        let event = EdlEvent::new(
            1,
            "A001".to_string(),
            TrackType::Video,
            EditType::Cut,
            tc1,
            tc2,
            tc1,
            tc2,
        );

        edl.add_event(event).expect("add_event should succeed");

        let validator = EdlValidator::standard();
        let report = validator.validate(&edl).expect("validation should succeed");
        assert!(!report.has_errors());
    }

    #[test]
    fn test_detect_overlap() {
        let mut edl = Edl::new(EdlFormat::Cmx3600);
        edl.set_frame_rate(EdlFrameRate::Fps25);

        let tc1 = EdlTimecode::new(1, 0, 0, 0, EdlFrameRate::Fps25).expect("failed to create");
        let tc2 = EdlTimecode::new(1, 0, 10, 0, EdlFrameRate::Fps25).expect("failed to create");
        let tc3 = EdlTimecode::new(1, 0, 5, 0, EdlFrameRate::Fps25).expect("failed to create");
        let tc4 = EdlTimecode::new(1, 0, 15, 0, EdlFrameRate::Fps25).expect("failed to create");

        let event1 = EdlEvent::new(
            1,
            "A001".to_string(),
            TrackType::Video,
            EditType::Cut,
            tc1,
            tc2,
            tc1,
            tc2,
        );

        let event2 = EdlEvent::new(
            2,
            "A002".to_string(),
            TrackType::Video,
            EditType::Cut,
            tc3,
            tc4,
            tc3,
            tc4,
        );

        edl.add_event(event1).expect("add_event should succeed");
        edl.add_event(event2).expect("add_event should succeed");

        let validator = EdlValidator::strict();
        assert!(validator.validate(&edl).is_err());
    }

    #[test]
    fn test_detect_gap() {
        let mut edl = Edl::new(EdlFormat::Cmx3600);
        edl.set_frame_rate(EdlFrameRate::Fps25);

        let tc1 = EdlTimecode::new(1, 0, 0, 0, EdlFrameRate::Fps25).expect("failed to create");
        let tc2 = EdlTimecode::new(1, 0, 5, 0, EdlFrameRate::Fps25).expect("failed to create");
        let tc3 = EdlTimecode::new(1, 0, 10, 0, EdlFrameRate::Fps25).expect("failed to create");
        let tc4 = EdlTimecode::new(1, 0, 15, 0, EdlFrameRate::Fps25).expect("failed to create");

        let event1 = EdlEvent::new(
            1,
            "A001".to_string(),
            TrackType::Video,
            EditType::Cut,
            tc1,
            tc2,
            tc1,
            tc2,
        );

        let event2 = EdlEvent::new(
            2,
            "A002".to_string(),
            TrackType::Video,
            EditType::Cut,
            tc3,
            tc4,
            tc3,
            tc4,
        );

        edl.add_event(event1).expect("add_event should succeed");
        edl.add_event(event2).expect("add_event should succeed");

        let validator = EdlValidator::strict();
        assert!(validator.validate(&edl).is_err());
    }

    #[test]
    fn test_event_numbering() {
        let mut edl = Edl::new(EdlFormat::Cmx3600);
        edl.set_frame_rate(EdlFrameRate::Fps25);

        let tc1 = EdlTimecode::new(1, 0, 0, 0, EdlFrameRate::Fps25).expect("failed to create");
        let tc2 = EdlTimecode::new(1, 0, 5, 0, EdlFrameRate::Fps25).expect("failed to create");

        // Non-sequential numbering
        let event1 = EdlEvent::new(
            1,
            "A001".to_string(),
            TrackType::Video,
            EditType::Cut,
            tc1,
            tc2,
            tc1,
            tc2,
        );

        let event2 = EdlEvent::new(
            3,
            "A002".to_string(),
            TrackType::Video,
            EditType::Cut,
            tc1,
            tc2,
            tc1,
            tc2,
        );

        edl.add_event(event1).expect("add_event should succeed");
        edl.add_event(event2).expect("add_event should succeed");

        let validator = EdlValidator::strict();
        assert!(validator.validate(&edl).is_err());

        // Lenient mode should allow this
        let validator = EdlValidator::lenient();
        assert!(validator.validate(&edl).is_ok());
    }
}
