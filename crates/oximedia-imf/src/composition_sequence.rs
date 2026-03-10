//! IMF composition sequence management.
//!
//! This module provides structures for managing compositions and segments
//! within an IMF package, including duration calculations and validation.

#![allow(dead_code)]

/// A single segment within a composition sequence.
///
/// Each segment references a track file essence resource and specifies
/// its temporal placement via offset and duration expressed in edit-rate units.
#[derive(Debug, Clone)]
pub struct CompositionSegment {
    /// Unique identifier for this segment.
    pub id: String,
    /// Reference to the sequence track file.
    pub sequence_ref: String,
    /// Offset from the start of the composition in edit-rate units.
    pub offset: u64,
    /// Duration of this segment in edit-rate units.
    pub duration: u64,
    /// Edit rate as a rational (numerator, denominator), e.g. (24, 1) for 24 fps.
    pub edit_rate: (u32, u32),
}

impl CompositionSegment {
    /// Creates a new composition segment.
    pub fn new(
        id: impl Into<String>,
        sequence_ref: impl Into<String>,
        offset: u64,
        duration: u64,
        edit_rate: (u32, u32),
    ) -> Self {
        Self {
            id: id.into(),
            sequence_ref: sequence_ref.into(),
            offset,
            duration,
            edit_rate,
        }
    }

    /// Converts the segment duration to milliseconds given the edit rate in fps.
    ///
    /// `edit_rate_fps` is provided separately to allow for non-integer frame rates
    /// such as 29.97 or 23.976.
    pub fn duration_ms(&self, edit_rate_fps: f64) -> f64 {
        if edit_rate_fps <= 0.0 {
            return 0.0;
        }
        (self.duration as f64 / edit_rate_fps) * 1000.0
    }

    /// Returns the number of frames in this segment, which equals its duration
    /// in edit-rate units.
    pub fn frame_count(&self) -> u64 {
        self.duration
    }

    /// Computes the rational edit rate as a floating-point value.
    ///
    /// Returns `None` if the denominator is zero.
    pub fn edit_rate_fps(&self) -> Option<f64> {
        let (num, den) = self.edit_rate;
        if den == 0 {
            None
        } else {
            Some(f64::from(num) / f64::from(den))
        }
    }

    /// Returns the end offset (exclusive) of this segment.
    pub fn end_offset(&self) -> u64 {
        self.offset + self.duration
    }

    /// Checks whether this segment overlaps with another.
    pub fn overlaps_with(&self, other: &CompositionSegment) -> bool {
        self.offset < other.end_offset() && other.offset < self.end_offset()
    }
}

/// A complete composition sequence consisting of ordered segments.
#[derive(Debug, Clone)]
pub struct CompositionSequence {
    /// Unique identifier for this composition sequence.
    pub id: String,
    /// Human-readable title.
    pub title: String,
    /// Ordered list of composition segments.
    pub segments: Vec<CompositionSegment>,
}

impl CompositionSequence {
    /// Creates a new, empty composition sequence.
    pub fn new(id: &str, title: &str) -> Self {
        Self {
            id: id.to_string(),
            title: title.to_string(),
            segments: Vec::new(),
        }
    }

    /// Adds a segment to the sequence.
    pub fn add_segment(&mut self, seg: CompositionSegment) {
        self.segments.push(seg);
    }

    /// Returns the total duration of all segments in edit-rate frames.
    ///
    /// This is the sum of all segment durations (not accounting for gaps/overlaps).
    pub fn total_duration_frames(&self) -> u64 {
        self.segments.iter().map(|s| s.duration).sum()
    }

    /// Returns the total duration in milliseconds at the given frame rate.
    pub fn total_duration_ms(&self, fps: f64) -> f64 {
        self.segments.iter().map(|s| s.duration_ms(fps)).sum()
    }

    /// Returns the number of segments in this sequence.
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Returns the segment with the given ID, if present.
    pub fn find_segment(&self, id: &str) -> Option<&CompositionSegment> {
        self.segments.iter().find(|s| s.id == id)
    }
}

/// Validates a composition sequence for structural correctness.
///
/// Returns a list of error/warning strings describing any problems found.
/// An empty list means the sequence is valid.
pub fn validate_sequence(seq: &CompositionSequence) -> Vec<String> {
    let mut errors = Vec::new();

    if seq.id.is_empty() {
        errors.push("Sequence ID must not be empty".to_string());
    }

    if seq.title.is_empty() {
        errors.push("Sequence title must not be empty".to_string());
    }

    for (i, seg) in seq.segments.iter().enumerate() {
        if seg.id.is_empty() {
            errors.push(format!("Segment [{}] has an empty ID", i));
        }

        if seg.sequence_ref.is_empty() {
            errors.push(format!("Segment '{}' has an empty sequence_ref", seg.id));
        }

        if seg.duration == 0 {
            errors.push(format!("Segment '{}' has zero duration", seg.id));
        }

        let (num, den) = seg.edit_rate;
        if den == 0 {
            errors.push(format!(
                "Segment '{}' has invalid edit rate {}/{} (denominator is zero)",
                seg.id, num, den
            ));
        }

        // Check for common valid edit rates
        if den != 0 {
            let fps = f64::from(num) / f64::from(den);
            if fps < 1.0 || fps > 300.0 {
                errors.push(format!(
                    "Segment '{}' has unusual edit rate {:.4} fps (expected 1–300)",
                    seg.id, fps
                ));
            }
        }
    }

    // Check for overlapping segments (by offset range within same sequence_ref)
    for i in 0..seq.segments.len() {
        for j in (i + 1)..seq.segments.len() {
            let a = &seq.segments[i];
            let b = &seq.segments[j];
            if a.sequence_ref == b.sequence_ref && a.overlaps_with(b) {
                errors.push(format!(
                    "Segments '{}' and '{}' overlap (same sequence_ref '{}')",
                    a.id, b.id, a.sequence_ref
                ));
            }
        }
    }

    // Check for duplicate segment IDs
    let mut seen_ids = std::collections::HashSet::new();
    for seg in &seq.segments {
        if !seen_ids.insert(seg.id.clone()) {
            errors.push(format!("Duplicate segment ID '{}'", seg.id));
        }
    }

    errors
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_segment(id: &str, offset: u64, duration: u64) -> CompositionSegment {
        CompositionSegment::new(id, "seq-001", offset, duration, (24, 1))
    }

    #[test]
    fn test_segment_duration_ms_24fps() {
        let seg = make_segment("s1", 0, 24);
        // 24 frames at 24 fps = 1000 ms
        let ms = seg.duration_ms(24.0);
        assert!((ms - 1000.0).abs() < 0.001);
    }

    #[test]
    fn test_segment_duration_ms_zero_fps() {
        let seg = make_segment("s1", 0, 100);
        assert_eq!(seg.duration_ms(0.0), 0.0);
    }

    #[test]
    fn test_segment_frame_count() {
        let seg = make_segment("s1", 0, 48);
        assert_eq!(seg.frame_count(), 48);
    }

    #[test]
    fn test_segment_edit_rate_fps() {
        let seg = CompositionSegment::new("s1", "ref", 0, 10, (30000, 1001));
        let fps = seg.edit_rate_fps().expect("fps should be valid");
        assert!((fps - 29.97).abs() < 0.01);
    }

    #[test]
    fn test_segment_edit_rate_fps_zero_denominator() {
        let seg = CompositionSegment::new("s1", "ref", 0, 10, (24, 0));
        assert!(seg.edit_rate_fps().is_none());
    }

    #[test]
    fn test_segment_end_offset() {
        let seg = make_segment("s1", 100, 50);
        assert_eq!(seg.end_offset(), 150);
    }

    #[test]
    fn test_segments_overlap_detection() {
        let a = make_segment("a", 0, 100);
        let b = make_segment("b", 50, 100);
        assert!(a.overlaps_with(&b));
    }

    #[test]
    fn test_segments_no_overlap() {
        let a = make_segment("a", 0, 100);
        let b = make_segment("b", 100, 100);
        assert!(!a.overlaps_with(&b));
    }

    #[test]
    fn test_composition_sequence_total_frames() {
        let mut seq = CompositionSequence::new("cpl-001", "Test Sequence");
        seq.add_segment(make_segment("s1", 0, 100));
        seq.add_segment(make_segment("s2", 100, 200));
        assert_eq!(seq.total_duration_frames(), 300);
    }

    #[test]
    fn test_composition_sequence_total_ms() {
        let mut seq = CompositionSequence::new("cpl-001", "Test Sequence");
        seq.add_segment(make_segment("s1", 0, 24)); // 1 second at 24fps
        seq.add_segment(make_segment("s2", 24, 48)); // 2 seconds at 24fps
        let ms = seq.total_duration_ms(24.0);
        assert!((ms - 3000.0).abs() < 0.001);
    }

    #[test]
    fn test_sequence_segment_count() {
        let mut seq = CompositionSequence::new("id", "title");
        assert_eq!(seq.segment_count(), 0);
        seq.add_segment(make_segment("s1", 0, 10));
        assert_eq!(seq.segment_count(), 1);
    }

    #[test]
    fn test_validate_sequence_empty_ok() {
        let seq = CompositionSequence::new("id", "title");
        let errors = validate_sequence(&seq);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_validate_sequence_zero_duration_error() {
        let mut seq = CompositionSequence::new("id", "title");
        seq.add_segment(CompositionSegment::new("s1", "ref", 0, 0, (24, 1)));
        let errors = validate_sequence(&seq);
        assert!(errors.iter().any(|e| e.contains("zero duration")));
    }

    #[test]
    fn test_validate_sequence_overlap_error() {
        let mut seq = CompositionSequence::new("id", "title");
        // Both reference the same sequence_ref and overlap
        seq.add_segment(CompositionSegment::new("s1", "ref-A", 0, 100, (24, 1)));
        seq.add_segment(CompositionSegment::new("s2", "ref-A", 50, 100, (24, 1)));
        let errors = validate_sequence(&seq);
        assert!(errors.iter().any(|e| e.contains("overlap")));
    }

    #[test]
    fn test_validate_sequence_duplicate_ids() {
        let mut seq = CompositionSequence::new("id", "title");
        seq.add_segment(make_segment("s1", 0, 50));
        seq.add_segment(make_segment("s1", 50, 50));
        let errors = validate_sequence(&seq);
        assert!(errors.iter().any(|e| e.contains("Duplicate segment ID")));
    }

    #[test]
    fn test_find_segment() {
        let mut seq = CompositionSequence::new("id", "title");
        seq.add_segment(make_segment("seg-alpha", 0, 24));
        seq.add_segment(make_segment("seg-beta", 24, 48));
        assert!(seq.find_segment("seg-alpha").is_some());
        assert!(seq.find_segment("seg-gamma").is_none());
    }
}
