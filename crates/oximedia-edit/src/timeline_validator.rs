//! Timeline validation: detect structural issues such as overlapping clips,
//! gaps, zero-duration clips, inconsistent source ranges, and orphan references.
//!
//! [`TimelineValidator`] inspects a [`crate::timeline::Timeline`] and produces
//! a list of [`ValidationIssue`]s categorized by [`IssueSeverity`].
//!
//! # Example
//! ```rust
//! use oximedia_edit::timeline::Timeline;
//! use oximedia_edit::timeline_validator::{TimelineValidator, IssueSeverity};
//! use oximedia_core::Rational;
//!
//! let timeline = Timeline::new(Rational::new(1, 1000), Rational::new(30, 1));
//! let issues = TimelineValidator::validate(&timeline);
//! let errors = issues.iter().filter(|i| i.severity == IssueSeverity::Error).count();
//! assert_eq!(errors, 0);
//! ```

#![allow(dead_code)]

use crate::clip::Clip;
use crate::timeline::Timeline;

// ─────────────────────────────────────────────────────────────────────────────
// IssueSeverity
// ─────────────────────────────────────────────────────────────────────────────

/// How critical a validation finding is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum IssueSeverity {
    /// Informational — something unusual but not harmful.
    Info,
    /// Warning — may cause unexpected behaviour during playback/export.
    Warning,
    /// Error — will cause incorrect output or crashes.
    Error,
}

impl IssueSeverity {
    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Warning => "warning",
            Self::Error => "error",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// IssueKind
// ─────────────────────────────────────────────────────────────────────────────

/// Specific type of validation finding.
#[derive(Clone, Debug, PartialEq)]
pub enum IssueKind {
    /// Two clips on the same track overlap in time.
    OverlappingClips {
        /// Track index.
        track_index: usize,
        /// First clip ID.
        clip_a: u64,
        /// Second clip ID.
        clip_b: u64,
    },
    /// A gap exists between consecutive clips on a track.
    Gap {
        /// Track index.
        track_index: usize,
        /// Gap start position (timebase units).
        gap_start: i64,
        /// Gap end position (timebase units).
        gap_end: i64,
    },
    /// A clip has zero or negative duration.
    ZeroDurationClip {
        /// Track index.
        track_index: usize,
        /// Offending clip ID.
        clip_id: u64,
    },
    /// A clip's source_in >= source_out.
    InvalidSourceRange {
        /// Track index.
        track_index: usize,
        /// Offending clip ID.
        clip_id: u64,
        /// Source in point.
        source_in: i64,
        /// Source out point.
        source_out: i64,
    },
    /// A clip has a negative timeline start.
    NegativeTimelineStart {
        /// Track index.
        track_index: usize,
        /// Offending clip ID.
        clip_id: u64,
        /// The negative start position.
        start: i64,
    },
    /// Clip speed is zero or negative.
    InvalidSpeed {
        /// Track index.
        track_index: usize,
        /// Offending clip ID.
        clip_id: u64,
        /// The invalid speed value.
        speed: f64,
    },
    /// Duplicate clip ID found across the timeline.
    DuplicateClipId {
        /// The duplicated clip ID.
        clip_id: u64,
    },
    /// A clip's opacity is outside [0.0, 1.0].
    OpacityOutOfRange {
        /// Track index.
        track_index: usize,
        /// Offending clip ID.
        clip_id: u64,
        /// The out-of-range opacity.
        opacity: f32,
    },
    /// An empty track (no clips).
    EmptyTrack {
        /// Track index.
        track_index: usize,
    },
    /// Timeline reported duration does not match actual clip extents.
    DurationMismatch {
        /// Duration stored on the timeline.
        reported: i64,
        /// Actual maximum clip end.
        actual: i64,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// ValidationIssue
// ─────────────────────────────────────────────────────────────────────────────

/// A single validation finding.
#[derive(Clone, Debug)]
pub struct ValidationIssue {
    /// Severity of the issue.
    pub severity: IssueSeverity,
    /// What was found.
    pub kind: IssueKind,
    /// Human-readable description.
    pub message: String,
}

impl ValidationIssue {
    /// Create a new issue.
    #[must_use]
    fn new(severity: IssueSeverity, kind: IssueKind, message: impl Into<String>) -> Self {
        Self {
            severity,
            kind,
            message: message.into(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ValidationReport
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregated validation result.
#[derive(Clone, Debug, Default)]
pub struct ValidationReport {
    /// All issues found.
    pub issues: Vec<ValidationIssue>,
}

impl ValidationReport {
    /// Number of errors.
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Error)
            .count()
    }

    /// Number of warnings.
    #[must_use]
    pub fn warning_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Warning)
            .count()
    }

    /// Whether the timeline passed validation with no errors.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.error_count() == 0
    }

    /// Whether the timeline is fully clean (no issues at all).
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.issues.is_empty()
    }

    /// Filter issues by severity.
    #[must_use]
    pub fn issues_of(&self, severity: IssueSeverity) -> Vec<&ValidationIssue> {
        self.issues
            .iter()
            .filter(|i| i.severity == severity)
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TimelineValidator
// ─────────────────────────────────────────────────────────────────────────────

/// Validates a [`Timeline`] for structural correctness.
pub struct TimelineValidator;

impl TimelineValidator {
    /// Run all validation checks on `timeline` and return a flat list of issues.
    #[must_use]
    pub fn validate(timeline: &Timeline) -> Vec<ValidationIssue> {
        let report = Self::validate_full(timeline);
        report.issues
    }

    /// Run all validation checks and return a [`ValidationReport`].
    #[must_use]
    pub fn validate_full(timeline: &Timeline) -> ValidationReport {
        let mut report = ValidationReport::default();

        Self::check_duplicate_ids(timeline, &mut report);
        Self::check_duration_mismatch(timeline, &mut report);

        for (track_idx, track) in timeline.tracks.iter().enumerate() {
            if track.clips.is_empty() {
                report.issues.push(ValidationIssue::new(
                    IssueSeverity::Info,
                    IssueKind::EmptyTrack {
                        track_index: track_idx,
                    },
                    format!("Track {track_idx} has no clips"),
                ));
                continue;
            }

            Self::check_clips(track_idx, &track.clips, &mut report);
            Self::check_overlaps_and_gaps(track_idx, &track.clips, &mut report);
        }

        report
    }

    /// Check per-clip invariants.
    fn check_clips(track_idx: usize, clips: &[Clip], report: &mut ValidationReport) {
        for clip in clips {
            // Zero or negative duration
            if clip.timeline_duration <= 0 {
                report.issues.push(ValidationIssue::new(
                    IssueSeverity::Error,
                    IssueKind::ZeroDurationClip {
                        track_index: track_idx,
                        clip_id: clip.id,
                    },
                    format!(
                        "Clip {} on track {track_idx} has duration {}",
                        clip.id, clip.timeline_duration
                    ),
                ));
            }

            // Negative start
            if clip.timeline_start < 0 {
                report.issues.push(ValidationIssue::new(
                    IssueSeverity::Warning,
                    IssueKind::NegativeTimelineStart {
                        track_index: track_idx,
                        clip_id: clip.id,
                        start: clip.timeline_start,
                    },
                    format!(
                        "Clip {} on track {track_idx} starts at negative position {}",
                        clip.id, clip.timeline_start
                    ),
                ));
            }

            // Invalid source range
            if clip.source_in >= clip.source_out {
                report.issues.push(ValidationIssue::new(
                    IssueSeverity::Error,
                    IssueKind::InvalidSourceRange {
                        track_index: track_idx,
                        clip_id: clip.id,
                        source_in: clip.source_in,
                        source_out: clip.source_out,
                    },
                    format!(
                        "Clip {} on track {track_idx}: source_in ({}) >= source_out ({})",
                        clip.id, clip.source_in, clip.source_out
                    ),
                ));
            }

            // Invalid speed
            if clip.speed <= 0.0 {
                report.issues.push(ValidationIssue::new(
                    IssueSeverity::Error,
                    IssueKind::InvalidSpeed {
                        track_index: track_idx,
                        clip_id: clip.id,
                        speed: clip.speed,
                    },
                    format!(
                        "Clip {} on track {track_idx} has non-positive speed {}",
                        clip.id, clip.speed
                    ),
                ));
            }

            // Opacity out of range
            if !(0.0..=1.0).contains(&clip.opacity) {
                report.issues.push(ValidationIssue::new(
                    IssueSeverity::Warning,
                    IssueKind::OpacityOutOfRange {
                        track_index: track_idx,
                        clip_id: clip.id,
                        opacity: clip.opacity,
                    },
                    format!(
                        "Clip {} on track {track_idx} has opacity {} (expected 0.0–1.0)",
                        clip.id, clip.opacity
                    ),
                ));
            }
        }
    }

    /// Check for overlapping or gapped clips on a single track.
    fn check_overlaps_and_gaps(track_idx: usize, clips: &[Clip], report: &mut ValidationReport) {
        for window in clips.windows(2) {
            let a = &window[0];
            let b = &window[1];
            let a_end = a.timeline_start + a.timeline_duration;

            if a_end > b.timeline_start {
                // Overlap
                report.issues.push(ValidationIssue::new(
                    IssueSeverity::Error,
                    IssueKind::OverlappingClips {
                        track_index: track_idx,
                        clip_a: a.id,
                        clip_b: b.id,
                    },
                    format!(
                        "Clips {} and {} overlap on track {track_idx} ({}..{} vs {}..{})",
                        a.id,
                        b.id,
                        a.timeline_start,
                        a_end,
                        b.timeline_start,
                        b.timeline_start + b.timeline_duration,
                    ),
                ));
            } else if a_end < b.timeline_start {
                // Gap
                report.issues.push(ValidationIssue::new(
                    IssueSeverity::Info,
                    IssueKind::Gap {
                        track_index: track_idx,
                        gap_start: a_end,
                        gap_end: b.timeline_start,
                    },
                    format!(
                        "Gap on track {track_idx} between clips {} and {} ({}..{})",
                        a.id, b.id, a_end, b.timeline_start,
                    ),
                ));
            }
        }
    }

    /// Check for duplicate clip IDs across the entire timeline.
    fn check_duplicate_ids(timeline: &Timeline, report: &mut ValidationReport) {
        let mut seen = std::collections::HashSet::new();
        for track in &timeline.tracks {
            for clip in &track.clips {
                if !seen.insert(clip.id) {
                    report.issues.push(ValidationIssue::new(
                        IssueSeverity::Error,
                        IssueKind::DuplicateClipId { clip_id: clip.id },
                        format!("Duplicate clip ID {} found in timeline", clip.id),
                    ));
                }
            }
        }
    }

    /// Check if reported timeline duration matches actual extent.
    fn check_duration_mismatch(timeline: &Timeline, report: &mut ValidationReport) {
        let mut actual_end: i64 = 0;
        for track in &timeline.tracks {
            for clip in &track.clips {
                let end = clip.timeline_start + clip.timeline_duration;
                if end > actual_end {
                    actual_end = end;
                }
            }
        }
        if timeline.duration != actual_end {
            report.issues.push(ValidationIssue::new(
                IssueSeverity::Warning,
                IssueKind::DurationMismatch {
                    reported: timeline.duration,
                    actual: actual_end,
                },
                format!(
                    "Timeline duration ({}) does not match actual clip extent ({})",
                    timeline.duration, actual_end,
                ),
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::{Clip, ClipType};
    use crate::timeline::{Timeline, TrackType};
    use oximedia_core::Rational;

    fn make_timeline() -> Timeline {
        Timeline::new(Rational::new(1, 1000), Rational::new(30, 1))
    }

    #[test]
    fn test_empty_timeline_is_valid() {
        let tl = make_timeline();
        let report = TimelineValidator::validate_full(&tl);
        assert!(report.is_valid());
        assert!(report.is_clean());
    }

    #[test]
    fn test_single_clip_no_issues() {
        let mut tl = make_timeline();
        let track = tl.add_track(TrackType::Video);
        let clip = Clip::new(1, ClipType::Video, 0, 1000);
        tl.add_clip(track, clip).ok();
        let report = TimelineValidator::validate_full(&tl);
        assert!(report.is_valid());
    }

    #[test]
    fn test_overlapping_clips_detected() {
        let mut tl = make_timeline();
        let track = tl.add_track(TrackType::Video);
        // Manually push overlapping clips (bypass add_clip validation)
        tl.tracks[track]
            .clips
            .push(Clip::new(10, ClipType::Video, 0, 500));
        tl.tracks[track]
            .clips
            .push(Clip::new(11, ClipType::Video, 400, 500));
        let report = TimelineValidator::validate_full(&tl);
        assert!(!report.is_valid());
        let overlaps: Vec<_> = report
            .issues
            .iter()
            .filter(|i| matches!(i.kind, IssueKind::OverlappingClips { .. }))
            .collect();
        assert_eq!(overlaps.len(), 1);
    }

    #[test]
    fn test_gap_detected() {
        let mut tl = make_timeline();
        let track = tl.add_track(TrackType::Video);
        let c1 = Clip::new(1, ClipType::Video, 0, 100);
        let c2 = Clip::new(2, ClipType::Video, 200, 100);
        tl.add_clip(track, c1).ok();
        tl.add_clip(track, c2).ok();
        let issues = TimelineValidator::validate(&tl);
        let gaps: Vec<_> = issues
            .iter()
            .filter(|i| matches!(i.kind, IssueKind::Gap { .. }))
            .collect();
        assert_eq!(gaps.len(), 1);
    }

    #[test]
    fn test_zero_duration_clip_error() {
        let mut tl = make_timeline();
        let track = tl.add_track(TrackType::Video);
        tl.tracks[track]
            .clips
            .push(Clip::new(50, ClipType::Video, 0, 0));
        let report = TimelineValidator::validate_full(&tl);
        assert!(!report.is_valid());
        assert!(report
            .issues
            .iter()
            .any(|i| matches!(i.kind, IssueKind::ZeroDurationClip { clip_id: 50, .. })));
    }

    #[test]
    fn test_invalid_source_range_error() {
        let mut tl = make_timeline();
        let track = tl.add_track(TrackType::Video);
        let mut clip = Clip::new(60, ClipType::Video, 0, 100);
        clip.source_in = 200;
        clip.source_out = 100; // invalid: in >= out
        tl.tracks[track].clips.push(clip);
        let report = TimelineValidator::validate_full(&tl);
        assert!(!report.is_valid());
        assert!(report
            .issues
            .iter()
            .any(|i| matches!(i.kind, IssueKind::InvalidSourceRange { clip_id: 60, .. })));
    }

    #[test]
    fn test_negative_speed_error() {
        let mut tl = make_timeline();
        let track = tl.add_track(TrackType::Video);
        let mut clip = Clip::new(70, ClipType::Video, 0, 100);
        clip.speed = -1.0;
        tl.tracks[track].clips.push(clip);
        let report = TimelineValidator::validate_full(&tl);
        assert!(!report.is_valid());
    }

    #[test]
    fn test_opacity_out_of_range_warning() {
        let mut tl = make_timeline();
        let track = tl.add_track(TrackType::Video);
        let mut clip = Clip::new(80, ClipType::Video, 0, 100);
        clip.opacity = 1.5;
        tl.tracks[track].clips.push(clip);
        tl.duration = 100; // match actual extent to avoid duration mismatch warning
        let report = TimelineValidator::validate_full(&tl);
        assert!(report.is_valid()); // warnings don't make it invalid
        assert_eq!(report.warning_count(), 1);
    }

    #[test]
    fn test_duplicate_clip_id_error() {
        let mut tl = make_timeline();
        let t1 = tl.add_track(TrackType::Video);
        let t2 = tl.add_track(TrackType::Audio);
        tl.tracks[t1]
            .clips
            .push(Clip::new(99, ClipType::Video, 0, 100));
        tl.tracks[t2]
            .clips
            .push(Clip::new(99, ClipType::Audio, 0, 100));
        let report = TimelineValidator::validate_full(&tl);
        assert!(!report.is_valid());
        assert!(report
            .issues
            .iter()
            .any(|i| matches!(i.kind, IssueKind::DuplicateClipId { clip_id: 99 })));
    }

    #[test]
    fn test_empty_track_info() {
        let mut tl = make_timeline();
        tl.add_track(TrackType::Video);
        let report = TimelineValidator::validate_full(&tl);
        assert!(report.is_valid());
        let infos = report.issues_of(IssueSeverity::Info);
        assert!(!infos.is_empty());
        assert!(infos
            .iter()
            .any(|i| matches!(i.kind, IssueKind::EmptyTrack { track_index: 0 })));
    }

    #[test]
    fn test_duration_mismatch_warning() {
        let mut tl = make_timeline();
        let track = tl.add_track(TrackType::Video);
        let clip = Clip::new(1, ClipType::Video, 0, 500);
        tl.add_clip(track, clip).ok();
        // Tamper with reported duration
        tl.duration = 999;
        let report = TimelineValidator::validate_full(&tl);
        assert!(report.issues.iter().any(|i| matches!(
            i.kind,
            IssueKind::DurationMismatch {
                reported: 999,
                actual: 500
            }
        )));
    }

    #[test]
    fn test_negative_timeline_start_warning() {
        let mut tl = make_timeline();
        let track = tl.add_track(TrackType::Video);
        let clip = Clip::new(42, ClipType::Video, -100, 200);
        tl.tracks[track].clips.push(clip);
        let report = TimelineValidator::validate_full(&tl);
        assert!(report.warning_count() >= 1);
        assert!(report.issues.iter().any(|i| matches!(
            i.kind,
            IssueKind::NegativeTimelineStart {
                clip_id: 42,
                start: -100,
                ..
            }
        )));
    }

    #[test]
    fn test_report_is_clean_when_no_issues() {
        let tl = make_timeline();
        let report = TimelineValidator::validate_full(&tl);
        assert!(report.is_clean());
        assert_eq!(report.error_count(), 0);
        assert_eq!(report.warning_count(), 0);
    }

    #[test]
    fn test_severity_ordering() {
        assert!(IssueSeverity::Info < IssueSeverity::Warning);
        assert!(IssueSeverity::Warning < IssueSeverity::Error);
    }

    #[test]
    fn test_severity_label() {
        assert_eq!(IssueSeverity::Info.label(), "info");
        assert_eq!(IssueSeverity::Warning.label(), "warning");
        assert_eq!(IssueSeverity::Error.label(), "error");
    }
}
