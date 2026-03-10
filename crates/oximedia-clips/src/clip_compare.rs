#![allow(dead_code)]
//! Clip comparison and difference analysis.
//!
//! This module provides tools for comparing two or more clips side by side,
//! detecting differences in metadata, trim points, ratings, keywords, and
//! other properties. Useful for version tracking, QC workflows, and
//! identifying changes between edit passes.

use std::collections::HashSet;
use std::fmt;

/// The type of difference found between clips.
#[derive(Debug, Clone, PartialEq)]
pub enum DiffKind {
    /// Name differs.
    Name {
        /// Value from the left/first clip.
        left: String,
        /// Value from the right/second clip.
        right: String,
    },
    /// In-point differs (in frames).
    InPoint {
        /// In-point of the left clip.
        left: u64,
        /// In-point of the right clip.
        right: u64,
    },
    /// Out-point differs (in frames).
    OutPoint {
        /// Out-point of the left clip.
        left: u64,
        /// Out-point of the right clip.
        right: u64,
    },
    /// Duration differs (in frames).
    Duration {
        /// Duration of the left clip.
        left: u64,
        /// Duration of the right clip.
        right: u64,
    },
    /// Rating differs.
    Rating {
        /// Rating of the left clip.
        left: u8,
        /// Rating of the right clip.
        right: u8,
    },
    /// Keywords differ (shows added/removed).
    Keywords {
        /// Keywords only in the left clip.
        only_left: Vec<String>,
        /// Keywords only in the right clip.
        only_right: Vec<String>,
    },
    /// Codec differs.
    Codec {
        /// Codec of the left clip.
        left: String,
        /// Codec of the right clip.
        right: String,
    },
    /// Resolution differs.
    Resolution {
        /// Width and height of the left clip.
        left: (u32, u32),
        /// Width and height of the right clip.
        right: (u32, u32),
    },
    /// Frame rate differs.
    FrameRate {
        /// Frame rate of the left clip.
        left: f64,
        /// Frame rate of the right clip.
        right: f64,
    },
    /// Color label differs.
    ColorLabel {
        /// Color label of the left clip.
        left: String,
        /// Color label of the right clip.
        right: String,
    },
    /// Note / comment differs.
    Note {
        /// Note from the left clip.
        left: String,
        /// Note from the right clip.
        right: String,
    },
}

impl fmt::Display for DiffKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Name { left, right } => write!(f, "Name: '{left}' vs '{right}'"),
            Self::InPoint { left, right } => write!(f, "In Point: {left} vs {right}"),
            Self::OutPoint { left, right } => write!(f, "Out Point: {left} vs {right}"),
            Self::Duration { left, right } => write!(f, "Duration: {left} vs {right}"),
            Self::Rating { left, right } => write!(f, "Rating: {left} vs {right}"),
            Self::Keywords {
                only_left,
                only_right,
            } => {
                write!(
                    f,
                    "Keywords: only left=[{}], only right=[{}]",
                    only_left.join(", "),
                    only_right.join(", ")
                )
            }
            Self::Codec { left, right } => write!(f, "Codec: '{left}' vs '{right}'"),
            Self::Resolution { left, right } => {
                write!(
                    f,
                    "Resolution: {}x{} vs {}x{}",
                    left.0, left.1, right.0, right.1
                )
            }
            Self::FrameRate { left, right } => {
                write!(f, "Frame Rate: {left:.3} vs {right:.3}")
            }
            Self::ColorLabel { left, right } => {
                write!(f, "Color Label: '{left}' vs '{right}'")
            }
            Self::Note { left: _, right: _ } => write!(f, "Note changed"),
        }
    }
}

/// Severity of a difference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DiffSeverity {
    /// Informational (cosmetic changes like name, color).
    Info,
    /// Warning (metadata changes like keywords, rating).
    Warning,
    /// Critical (structural changes like duration, codec, resolution).
    Critical,
}

/// A single difference record between two clips.
#[derive(Debug, Clone)]
pub struct ClipDiff {
    /// The kind of difference.
    pub kind: DiffKind,
    /// Severity of this difference.
    pub severity: DiffSeverity,
}

impl ClipDiff {
    /// Creates a new diff record.
    #[must_use]
    pub fn new(kind: DiffKind, severity: DiffSeverity) -> Self {
        Self { kind, severity }
    }
}

/// Comparable clip properties used for comparison.
#[derive(Debug, Clone)]
pub struct ComparableClip {
    /// Clip identifier.
    pub id: u64,
    /// Clip name.
    pub name: String,
    /// In-point in frames.
    pub in_point: u64,
    /// Out-point in frames.
    pub out_point: u64,
    /// Rating (0-5).
    pub rating: u8,
    /// Keywords.
    pub keywords: Vec<String>,
    /// Codec name.
    pub codec: String,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Frame rate.
    pub frame_rate: f64,
    /// Color label.
    pub color_label: String,
    /// Note text.
    pub note: String,
}

impl ComparableClip {
    /// Creates a new comparable clip with minimal information.
    #[must_use]
    pub fn new(id: u64, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            in_point: 0,
            out_point: 0,
            rating: 0,
            keywords: Vec::new(),
            codec: String::new(),
            width: 0,
            height: 0,
            frame_rate: 0.0,
            color_label: String::new(),
            note: String::new(),
        }
    }

    /// Returns the duration in frames.
    #[must_use]
    pub fn duration(&self) -> u64 {
        self.out_point.saturating_sub(self.in_point)
    }
}

/// Result of comparing two clips.
#[derive(Debug, Clone)]
pub struct CompareResult {
    /// Left clip ID.
    pub left_id: u64,
    /// Right clip ID.
    pub right_id: u64,
    /// All differences found.
    pub diffs: Vec<ClipDiff>,
}

impl CompareResult {
    /// Creates a new comparison result.
    #[must_use]
    pub fn new(left_id: u64, right_id: u64) -> Self {
        Self {
            left_id,
            right_id,
            diffs: Vec::new(),
        }
    }

    /// Returns true if the clips are identical (no differences).
    #[must_use]
    pub fn is_identical(&self) -> bool {
        self.diffs.is_empty()
    }

    /// Returns the number of differences.
    #[must_use]
    pub fn diff_count(&self) -> usize {
        self.diffs.len()
    }

    /// Returns only critical differences.
    #[must_use]
    pub fn critical_diffs(&self) -> Vec<&ClipDiff> {
        self.diffs
            .iter()
            .filter(|d| d.severity == DiffSeverity::Critical)
            .collect()
    }

    /// Returns the highest severity found.
    #[must_use]
    pub fn max_severity(&self) -> Option<DiffSeverity> {
        self.diffs.iter().map(|d| d.severity).max()
    }

    /// Returns a summary string of all differences.
    #[must_use]
    pub fn summary(&self) -> String {
        if self.is_identical() {
            return format!("Clips {} and {} are identical", self.left_id, self.right_id);
        }
        let mut lines = vec![format!(
            "Comparing clip {} vs {}: {} difference(s)",
            self.left_id,
            self.right_id,
            self.diff_count()
        )];
        for diff in &self.diffs {
            lines.push(format!("  [{:?}] {}", diff.severity, diff.kind));
        }
        lines.join("\n")
    }
}

/// Clip comparison engine.
#[derive(Debug)]
pub struct ClipComparer {
    /// Whether to compare names.
    pub compare_names: bool,
    /// Whether to compare trim points.
    pub compare_trim: bool,
    /// Whether to compare ratings.
    pub compare_ratings: bool,
    /// Whether to compare keywords.
    pub compare_keywords: bool,
    /// Whether to compare technical metadata.
    pub compare_technical: bool,
    /// Whether to compare notes.
    pub compare_notes: bool,
    /// Frame rate comparison tolerance.
    pub fps_tolerance: f64,
}

impl Default for ClipComparer {
    fn default() -> Self {
        Self {
            compare_names: true,
            compare_trim: true,
            compare_ratings: true,
            compare_keywords: true,
            compare_technical: true,
            compare_notes: true,
            fps_tolerance: 0.001,
        }
    }
}

impl ClipComparer {
    /// Creates a new comparer with default settings (compare everything).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a comparer that only compares trim points and technical data.
    #[must_use]
    pub fn technical_only() -> Self {
        Self {
            compare_names: false,
            compare_trim: true,
            compare_ratings: false,
            compare_keywords: false,
            compare_technical: true,
            compare_notes: false,
            fps_tolerance: 0.001,
        }
    }

    /// Compares two clips and returns all differences.
    #[must_use]
    pub fn compare(&self, left: &ComparableClip, right: &ComparableClip) -> CompareResult {
        let mut result = CompareResult::new(left.id, right.id);

        if self.compare_names && left.name != right.name {
            result.diffs.push(ClipDiff::new(
                DiffKind::Name {
                    left: left.name.clone(),
                    right: right.name.clone(),
                },
                DiffSeverity::Info,
            ));
        }

        if self.compare_trim {
            if left.in_point != right.in_point {
                result.diffs.push(ClipDiff::new(
                    DiffKind::InPoint {
                        left: left.in_point,
                        right: right.in_point,
                    },
                    DiffSeverity::Critical,
                ));
            }
            if left.out_point != right.out_point {
                result.diffs.push(ClipDiff::new(
                    DiffKind::OutPoint {
                        left: left.out_point,
                        right: right.out_point,
                    },
                    DiffSeverity::Critical,
                ));
            }
            let ld = left.duration();
            let rd = right.duration();
            if ld != rd {
                result.diffs.push(ClipDiff::new(
                    DiffKind::Duration {
                        left: ld,
                        right: rd,
                    },
                    DiffSeverity::Critical,
                ));
            }
        }

        if self.compare_ratings && left.rating != right.rating {
            result.diffs.push(ClipDiff::new(
                DiffKind::Rating {
                    left: left.rating,
                    right: right.rating,
                },
                DiffSeverity::Warning,
            ));
        }

        if self.compare_keywords {
            let left_set: HashSet<&str> = left.keywords.iter().map(|s| s.as_str()).collect();
            let right_set: HashSet<&str> = right.keywords.iter().map(|s| s.as_str()).collect();
            let only_left: Vec<String> = left_set
                .difference(&right_set)
                .map(|s| (*s).to_string())
                .collect();
            let only_right: Vec<String> = right_set
                .difference(&left_set)
                .map(|s| (*s).to_string())
                .collect();
            if !only_left.is_empty() || !only_right.is_empty() {
                result.diffs.push(ClipDiff::new(
                    DiffKind::Keywords {
                        only_left,
                        only_right,
                    },
                    DiffSeverity::Warning,
                ));
            }
        }

        if self.compare_technical {
            if left.codec != right.codec {
                result.diffs.push(ClipDiff::new(
                    DiffKind::Codec {
                        left: left.codec.clone(),
                        right: right.codec.clone(),
                    },
                    DiffSeverity::Critical,
                ));
            }
            if left.width != right.width || left.height != right.height {
                result.diffs.push(ClipDiff::new(
                    DiffKind::Resolution {
                        left: (left.width, left.height),
                        right: (right.width, right.height),
                    },
                    DiffSeverity::Critical,
                ));
            }
            if (left.frame_rate - right.frame_rate).abs() > self.fps_tolerance {
                result.diffs.push(ClipDiff::new(
                    DiffKind::FrameRate {
                        left: left.frame_rate,
                        right: right.frame_rate,
                    },
                    DiffSeverity::Critical,
                ));
            }
        }

        if self.compare_notes && left.note != right.note {
            result.diffs.push(ClipDiff::new(
                DiffKind::Note {
                    left: left.note.clone(),
                    right: right.note.clone(),
                },
                DiffSeverity::Info,
            ));
        }

        if self.compare_names && left.color_label != right.color_label {
            result.diffs.push(ClipDiff::new(
                DiffKind::ColorLabel {
                    left: left.color_label.clone(),
                    right: right.color_label.clone(),
                },
                DiffSeverity::Info,
            ));
        }

        result
    }

    /// Compares multiple clips against a reference clip.
    #[must_use]
    pub fn compare_against_reference(
        &self,
        reference: &ComparableClip,
        others: &[ComparableClip],
    ) -> Vec<CompareResult> {
        others
            .iter()
            .map(|other| self.compare(reference, other))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_clip_a() -> ComparableClip {
        let mut c = ComparableClip::new(1, "Clip A");
        c.in_point = 0;
        c.out_point = 100;
        c.rating = 4;
        c.keywords = vec!["interview".into(), "john".into()];
        c.codec = "H.264".into();
        c.width = 1920;
        c.height = 1080;
        c.frame_rate = 24.0;
        c.color_label = "blue".into();
        c.note = "Good take".into();
        c
    }

    fn make_clip_b() -> ComparableClip {
        let mut c = ComparableClip::new(2, "Clip B");
        c.in_point = 10;
        c.out_point = 110;
        c.rating = 3;
        c.keywords = vec!["interview".into(), "jane".into()];
        c.codec = "H.264".into();
        c.width = 1920;
        c.height = 1080;
        c.frame_rate = 24.0;
        c.color_label = "red".into();
        c.note = "Needs retake".into();
        c
    }

    #[test]
    fn test_comparable_clip_duration() {
        let c = make_clip_a();
        assert_eq!(c.duration(), 100);
    }

    #[test]
    fn test_identical_clips() {
        let a = make_clip_a();
        let b = make_clip_a();
        let comparer = ClipComparer::new();
        let result = comparer.compare(&a, &b);
        assert!(result.is_identical());
    }

    #[test]
    fn test_name_diff() {
        let a = make_clip_a();
        let b = make_clip_b();
        let comparer = ClipComparer::new();
        let result = comparer.compare(&a, &b);
        assert!(!result.is_identical());
        let has_name = result
            .diffs
            .iter()
            .any(|d| matches!(&d.kind, DiffKind::Name { .. }));
        assert!(has_name);
    }

    #[test]
    fn test_trim_diff() {
        let a = make_clip_a();
        let b = make_clip_b();
        let comparer = ClipComparer::new();
        let result = comparer.compare(&a, &b);
        let has_in = result
            .diffs
            .iter()
            .any(|d| matches!(&d.kind, DiffKind::InPoint { .. }));
        assert!(has_in);
    }

    #[test]
    fn test_rating_diff() {
        let a = make_clip_a();
        let b = make_clip_b();
        let comparer = ClipComparer::new();
        let result = comparer.compare(&a, &b);
        let has_rating = result
            .diffs
            .iter()
            .any(|d| matches!(&d.kind, DiffKind::Rating { .. }));
        assert!(has_rating);
    }

    #[test]
    fn test_keyword_diff() {
        let a = make_clip_a();
        let b = make_clip_b();
        let comparer = ClipComparer::new();
        let result = comparer.compare(&a, &b);
        let kw_diff = result
            .diffs
            .iter()
            .find(|d| matches!(&d.kind, DiffKind::Keywords { .. }));
        assert!(kw_diff.is_some());
    }

    #[test]
    fn test_technical_only_comparer() {
        let a = make_clip_a();
        let b = make_clip_b();
        let comparer = ClipComparer::technical_only();
        let result = comparer.compare(&a, &b);
        // Should not contain name or rating diffs
        let has_name = result
            .diffs
            .iter()
            .any(|d| matches!(&d.kind, DiffKind::Name { .. }));
        assert!(!has_name);
        let has_rating = result
            .diffs
            .iter()
            .any(|d| matches!(&d.kind, DiffKind::Rating { .. }));
        assert!(!has_rating);
    }

    #[test]
    fn test_diff_severity_ordering() {
        assert!(DiffSeverity::Info < DiffSeverity::Warning);
        assert!(DiffSeverity::Warning < DiffSeverity::Critical);
    }

    #[test]
    fn test_max_severity() {
        let a = make_clip_a();
        let b = make_clip_b();
        let comparer = ClipComparer::new();
        let result = comparer.compare(&a, &b);
        let max = result.max_severity().expect("max_severity should succeed");
        assert_eq!(max, DiffSeverity::Critical);
    }

    #[test]
    fn test_critical_diffs() {
        let a = make_clip_a();
        let b = make_clip_b();
        let comparer = ClipComparer::new();
        let result = comparer.compare(&a, &b);
        let critical = result.critical_diffs();
        assert!(!critical.is_empty());
        for d in &critical {
            assert_eq!(d.severity, DiffSeverity::Critical);
        }
    }

    #[test]
    fn test_summary_identical() {
        let a = make_clip_a();
        let b = make_clip_a();
        let comparer = ClipComparer::new();
        let result = comparer.compare(&a, &b);
        let summary = result.summary();
        assert!(summary.contains("identical"));
    }

    #[test]
    fn test_summary_with_diffs() {
        let a = make_clip_a();
        let b = make_clip_b();
        let comparer = ClipComparer::new();
        let result = comparer.compare(&a, &b);
        let summary = result.summary();
        assert!(summary.contains("difference(s)"));
    }

    #[test]
    fn test_compare_against_reference() {
        let reference = make_clip_a();
        let others = vec![make_clip_b(), make_clip_a()];
        let comparer = ClipComparer::new();
        let results = comparer.compare_against_reference(&reference, &others);
        assert_eq!(results.len(), 2);
        assert!(!results[0].is_identical()); // a vs b
        assert!(results[1].is_identical()); // a vs a
    }

    #[test]
    fn test_diff_kind_display() {
        let dk = DiffKind::InPoint { left: 0, right: 10 };
        let s = format!("{dk}");
        assert!(s.contains("0"));
        assert!(s.contains("10"));
    }

    #[test]
    fn test_frame_rate_tolerance() {
        let mut a = make_clip_a();
        let mut b = make_clip_a();
        a.frame_rate = 23.976;
        b.frame_rate = 23.976_024; // within tolerance
        let comparer = ClipComparer {
            fps_tolerance: 0.001,
            ..ClipComparer::new()
        };
        let result = comparer.compare(&a, &b);
        let has_fps = result
            .diffs
            .iter()
            .any(|d| matches!(&d.kind, DiffKind::FrameRate { .. }));
        assert!(!has_fps);
    }
}
