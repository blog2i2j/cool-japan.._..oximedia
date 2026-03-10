//! IMF marker list: editorial and technical markers within a CPL timeline.
//!
//! Markers flag specific timecode positions for editorial review, QC, and
//! chapter-navigation purposes.  This module is a self-contained, pure-Rust
//! implementation that requires no external crates.

#![allow(dead_code)]

/// Classification of a marker's purpose.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MarkerKind {
    /// FFOC – First Frame Of Content (SMPTE ST 2067-3 convention).
    FirstFrameOfContent,
    /// LFOC – Last Frame Of Content.
    LastFrameOfContent,
    /// Chapter or scene boundary.
    Chapter,
    /// General editorial annotation.
    Editorial,
    /// QC / compliance marker.
    Qc,
    /// Custom marker with user-defined label.
    Custom(String),
}

impl MarkerKind {
    /// Returns a stable string code for this marker kind.
    #[must_use]
    pub fn code(&self) -> &str {
        match self {
            Self::FirstFrameOfContent => "FFOC",
            Self::LastFrameOfContent => "LFOC",
            Self::Chapter => "CHAPTER",
            Self::Editorial => "EDITORIAL",
            Self::Qc => "QC",
            Self::Custom(label) => label.as_str(),
        }
    }

    /// Returns `true` for structural boundary markers (FFOC / LFOC / Chapter).
    #[must_use]
    pub fn is_structural(&self) -> bool {
        matches!(
            self,
            Self::FirstFrameOfContent | Self::LastFrameOfContent | Self::Chapter
        )
    }
}

/// A single marker placed at a timeline position.
#[derive(Debug, Clone)]
pub struct Marker {
    /// Unique identifier.
    pub id: String,
    /// Kind / purpose of the marker.
    pub kind: MarkerKind,
    /// Timeline offset in edit-rate ticks from the CPL origin.
    pub offset: u64,
    /// Human-readable annotation text.
    pub annotation: String,
}

impl Marker {
    /// Create a new marker.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        kind: MarkerKind,
        offset: u64,
        annotation: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            kind,
            offset,
            annotation: annotation.into(),
        }
    }

    /// Returns `true` when the marker has a non-empty ID.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        !self.id.is_empty()
    }
}

/// An ordered list of markers attached to a CPL.
#[derive(Debug, Clone, Default)]
pub struct MarkerList {
    markers: Vec<Marker>,
}

impl MarkerList {
    /// Create an empty `MarkerList`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a marker.
    pub fn add(&mut self, marker: Marker) {
        self.markers.push(marker);
    }

    /// Total number of markers.
    #[must_use]
    pub fn count(&self) -> usize {
        self.markers.len()
    }

    /// Return markers sorted by their timeline offset (ascending).
    #[must_use]
    pub fn sorted_by_offset(&self) -> Vec<&Marker> {
        let mut v: Vec<&Marker> = self.markers.iter().collect();
        v.sort_by_key(|m| m.offset);
        v
    }

    /// Find all markers of a specific kind.
    #[must_use]
    pub fn filter_by_kind(&self, kind: &MarkerKind) -> Vec<&Marker> {
        self.markers.iter().filter(|m| &m.kind == kind).collect()
    }

    /// Find a marker by its ID.
    #[must_use]
    pub fn find_by_id(&self, id: &str) -> Option<&Marker> {
        self.markers.iter().find(|m| m.id == id)
    }

    /// Validate the list; returns a list of error messages.
    ///
    /// Checks:
    /// * No duplicate IDs.
    /// * All markers have a non-empty ID.
    #[must_use]
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        let mut seen_ids = std::collections::HashSet::new();
        for m in &self.markers {
            if m.id.is_empty() {
                errors.push("Marker has an empty ID".to_string());
                continue;
            }
            if !seen_ids.insert(m.id.as_str()) {
                errors.push(format!("Duplicate marker ID: '{}'", m.id));
            }
        }
        errors
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- MarkerKind ---

    #[test]
    fn test_marker_kind_codes() {
        assert_eq!(MarkerKind::FirstFrameOfContent.code(), "FFOC");
        assert_eq!(MarkerKind::LastFrameOfContent.code(), "LFOC");
        assert_eq!(MarkerKind::Chapter.code(), "CHAPTER");
        assert_eq!(MarkerKind::Editorial.code(), "EDITORIAL");
        assert_eq!(MarkerKind::Qc.code(), "QC");
        assert_eq!(MarkerKind::Custom("MYMARK".to_string()).code(), "MYMARK");
    }

    #[test]
    fn test_marker_kind_is_structural_true() {
        assert!(MarkerKind::FirstFrameOfContent.is_structural());
        assert!(MarkerKind::LastFrameOfContent.is_structural());
        assert!(MarkerKind::Chapter.is_structural());
    }

    #[test]
    fn test_marker_kind_is_structural_false() {
        assert!(!MarkerKind::Editorial.is_structural());
        assert!(!MarkerKind::Qc.is_structural());
        assert!(!MarkerKind::Custom("X".to_string()).is_structural());
    }

    // --- Marker ---

    #[test]
    fn test_marker_is_valid_true() {
        let m = Marker::new("m1", MarkerKind::Chapter, 100, "Scene 2");
        assert!(m.is_valid());
    }

    #[test]
    fn test_marker_is_valid_false_empty_id() {
        let m = Marker::new("", MarkerKind::Chapter, 100, "Scene 2");
        assert!(!m.is_valid());
    }

    #[test]
    fn test_marker_fields() {
        let m = Marker::new("m1", MarkerKind::Qc, 500, "Check color");
        assert_eq!(m.id, "m1");
        assert_eq!(m.kind, MarkerKind::Qc);
        assert_eq!(m.offset, 500);
        assert_eq!(m.annotation, "Check color");
    }

    // --- MarkerList ---

    #[test]
    fn test_marker_list_starts_empty() {
        let list = MarkerList::new();
        assert_eq!(list.count(), 0);
    }

    #[test]
    fn test_marker_list_add_increases_count() {
        let mut list = MarkerList::new();
        list.add(Marker::new("m1", MarkerKind::Chapter, 0, ""));
        list.add(Marker::new("m2", MarkerKind::Editorial, 100, ""));
        assert_eq!(list.count(), 2);
    }

    #[test]
    fn test_sorted_by_offset() {
        let mut list = MarkerList::new();
        list.add(Marker::new("m1", MarkerKind::Chapter, 300, ""));
        list.add(Marker::new("m2", MarkerKind::Chapter, 100, ""));
        list.add(Marker::new("m3", MarkerKind::Chapter, 200, ""));
        let sorted = list.sorted_by_offset();
        assert_eq!(sorted[0].offset, 100);
        assert_eq!(sorted[1].offset, 200);
        assert_eq!(sorted[2].offset, 300);
    }

    #[test]
    fn test_filter_by_kind_returns_matching() {
        let mut list = MarkerList::new();
        list.add(Marker::new("m1", MarkerKind::Chapter, 0, ""));
        list.add(Marker::new("m2", MarkerKind::Qc, 100, ""));
        list.add(Marker::new("m3", MarkerKind::Chapter, 200, ""));
        let chapters = list.filter_by_kind(&MarkerKind::Chapter);
        assert_eq!(chapters.len(), 2);
    }

    #[test]
    fn test_filter_by_kind_returns_empty() {
        let list = MarkerList::new();
        let res = list.filter_by_kind(&MarkerKind::Qc);
        assert!(res.is_empty());
    }

    #[test]
    fn test_find_by_id_found() {
        let mut list = MarkerList::new();
        list.add(Marker::new("target", MarkerKind::Editorial, 50, "note"));
        let found = list.find_by_id("target");
        assert!(found.is_some());
        assert_eq!(found.expect("test expectation failed").annotation, "note");
    }

    #[test]
    fn test_find_by_id_not_found() {
        let list = MarkerList::new();
        assert!(list.find_by_id("missing").is_none());
    }

    #[test]
    fn test_validate_clean_list() {
        let mut list = MarkerList::new();
        list.add(Marker::new("m1", MarkerKind::Chapter, 0, ""));
        list.add(Marker::new("m2", MarkerKind::Qc, 1, ""));
        assert!(list.validate().is_empty());
    }

    #[test]
    fn test_validate_detects_duplicate_ids() {
        let mut list = MarkerList::new();
        list.add(Marker::new("dup", MarkerKind::Chapter, 0, ""));
        list.add(Marker::new("dup", MarkerKind::Editorial, 100, ""));
        let errors = list.validate();
        assert!(!errors.is_empty());
        assert!(errors.iter().any(|e| e.contains("Duplicate")));
    }

    #[test]
    fn test_validate_detects_empty_id() {
        let mut list = MarkerList::new();
        list.add(Marker::new("", MarkerKind::Qc, 0, ""));
        let errors = list.validate();
        assert!(!errors.is_empty());
    }
}
