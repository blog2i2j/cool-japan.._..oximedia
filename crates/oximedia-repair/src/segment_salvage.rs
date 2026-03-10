#![allow(dead_code)]
//! Segment-level salvage for partially corrupted media files.
//!
//! When a media file is too damaged for a full repair, this module identifies
//! contiguous valid segments and extracts them individually. The segments can
//! then be concatenated or used as-is for partial content recovery.

/// Validation status of a byte range within a file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentValidity {
    /// The segment is fully valid and decodable.
    Valid,
    /// The segment has minor issues but is likely recoverable.
    Degraded,
    /// The segment is corrupt and should be discarded.
    Corrupt,
    /// Validity could not be determined (e.g., unknown format region).
    Unknown,
}

impl std::fmt::Display for SegmentValidity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Valid => write!(f, "valid"),
            Self::Degraded => write!(f, "degraded"),
            Self::Corrupt => write!(f, "corrupt"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

/// A contiguous segment within a media file.
#[derive(Debug, Clone)]
pub struct Segment {
    /// Start byte offset (inclusive).
    pub start: u64,
    /// End byte offset (exclusive).
    pub end: u64,
    /// Validity status of this segment.
    pub validity: SegmentValidity,
    /// Number of decodable frames within this segment, if known.
    pub frame_count: Option<u64>,
    /// Whether this segment starts with a keyframe.
    pub starts_with_keyframe: bool,
    /// Duration in seconds if known.
    pub duration_secs: Option<f64>,
    /// Human-readable notes about the segment.
    pub notes: String,
}

impl Segment {
    /// Create a new segment descriptor.
    pub fn new(start: u64, end: u64, validity: SegmentValidity) -> Self {
        Self {
            start,
            end,
            validity,
            frame_count: None,
            starts_with_keyframe: false,
            duration_secs: None,
            notes: String::new(),
        }
    }

    /// Length of the segment in bytes.
    pub fn len(&self) -> u64 {
        self.end.saturating_sub(self.start)
    }

    /// Check whether the segment is zero-length.
    pub fn is_empty(&self) -> bool {
        self.end <= self.start
    }

    /// Set the frame count.
    pub fn with_frame_count(mut self, count: u64) -> Self {
        self.frame_count = Some(count);
        self
    }

    /// Mark this segment as starting with a keyframe.
    pub fn with_keyframe_start(mut self) -> Self {
        self.starts_with_keyframe = true;
        self
    }

    /// Set the duration in seconds.
    pub fn with_duration(mut self, secs: f64) -> Self {
        self.duration_secs = Some(secs);
        self
    }

    /// Add a note.
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes = note.into();
        self
    }

    /// Returns `true` if this segment is usable for extraction.
    pub fn is_extractable(&self) -> bool {
        !self.is_empty()
            && matches!(
                self.validity,
                SegmentValidity::Valid | SegmentValidity::Degraded
            )
    }

    /// Returns `true` if this segment overlaps the given byte range.
    pub fn overlaps(&self, other_start: u64, other_end: u64) -> bool {
        self.start < other_end && other_start < self.end
    }
}

/// Result of a salvage scan across a file.
#[derive(Debug, Clone)]
pub struct SalvageMap {
    /// Total file size in bytes.
    pub file_size: u64,
    /// Identified segments in byte-offset order.
    pub segments: Vec<Segment>,
}

impl SalvageMap {
    /// Create a new empty salvage map.
    pub fn new(file_size: u64) -> Self {
        Self {
            file_size,
            segments: Vec::new(),
        }
    }

    /// Add a segment to the map.
    pub fn add_segment(&mut self, segment: Segment) {
        self.segments.push(segment);
        self.segments.sort_by_key(|s| s.start);
    }

    /// Return all extractable (valid or degraded) segments.
    pub fn extractable_segments(&self) -> Vec<&Segment> {
        self.segments.iter().filter(|s| s.is_extractable()).collect()
    }

    /// Total bytes of extractable content.
    pub fn extractable_bytes(&self) -> u64 {
        self.extractable_segments().iter().map(|s| s.len()).sum()
    }

    /// Fraction of the file that is extractable (0.0 to 1.0).
    #[allow(clippy::cast_precision_loss)]
    pub fn recovery_ratio(&self) -> f64 {
        if self.file_size == 0 {
            return 0.0;
        }
        self.extractable_bytes() as f64 / self.file_size as f64
    }

    /// Return the total number of segments.
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Find segments that contain the given byte offset.
    pub fn segments_at_offset(&self, offset: u64) -> Vec<&Segment> {
        self.segments
            .iter()
            .filter(|s| offset >= s.start && offset < s.end)
            .collect()
    }

    /// Merge adjacent extractable segments that are within `gap_tolerance` bytes of each other.
    pub fn merge_adjacent(&mut self, gap_tolerance: u64) {
        if self.segments.len() < 2 {
            return;
        }

        let mut merged: Vec<Segment> = Vec::new();
        let mut iter = self.segments.drain(..);

        if let Some(first) = iter.next() {
            let mut current = first;
            for next in iter {
                let gap = next.start.saturating_sub(current.end);
                if current.is_extractable()
                    && next.is_extractable()
                    && gap <= gap_tolerance
                {
                    // Merge: extend current to cover next
                    current.end = next.end;
                    // Take the better validity
                    if next.validity == SegmentValidity::Valid
                        && current.validity == SegmentValidity::Degraded
                    {
                        current.validity = SegmentValidity::Degraded;
                    }
                    // Merge frame counts
                    if let (Some(a), Some(b)) = (current.frame_count, next.frame_count) {
                        current.frame_count = Some(a + b);
                    }
                    // Merge durations
                    if let (Some(a), Some(b)) = (current.duration_secs, next.duration_secs) {
                        current.duration_secs = Some(a + b);
                    }
                } else {
                    merged.push(current);
                    current = next;
                }
            }
            merged.push(current);
        }

        self.segments = merged;
    }

    /// Identify gaps (unscanned/corrupt regions) between segments.
    pub fn gaps(&self) -> Vec<(u64, u64)> {
        let mut result = Vec::new();
        let mut pos: u64 = 0;

        for seg in &self.segments {
            if seg.start > pos {
                result.push((pos, seg.start));
            }
            pos = seg.end;
        }

        if pos < self.file_size {
            result.push((pos, self.file_size));
        }

        result
    }
}

/// Extraction plan for salvaged segments.
#[derive(Debug, Clone)]
pub struct ExtractionPlan {
    /// Ordered list of segment indices to extract.
    pub segment_indices: Vec<usize>,
    /// Total bytes to extract.
    pub total_bytes: u64,
    /// Estimated output duration in seconds, if known.
    pub estimated_duration_secs: Option<f64>,
    /// Whether re-muxing will be needed after extraction.
    pub needs_remux: bool,
}

impl ExtractionPlan {
    /// Build an extraction plan from a salvage map, taking all extractable segments.
    #[allow(clippy::cast_precision_loss)]
    pub fn from_salvage_map(map: &SalvageMap) -> Self {
        let mut indices = Vec::new();
        let mut total_bytes = 0u64;
        let mut total_duration = 0.0_f64;
        let mut has_duration = false;

        for (i, seg) in map.segments.iter().enumerate() {
            if seg.is_extractable() {
                indices.push(i);
                total_bytes += seg.len();
                if let Some(d) = seg.duration_secs {
                    total_duration += d;
                    has_duration = true;
                }
            }
        }

        let needs_remux = indices.len() > 1;
        let estimated_duration_secs = if has_duration {
            Some(total_duration)
        } else {
            None
        };

        Self {
            segment_indices: indices,
            total_bytes,
            estimated_duration_secs,
            needs_remux,
        }
    }

    /// Number of segments in the extraction plan.
    pub fn segment_count(&self) -> usize {
        self.segment_indices.len()
    }

    /// Whether the plan is empty (nothing to extract).
    pub fn is_empty(&self) -> bool {
        self.segment_indices.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment_validity_display() {
        assert_eq!(format!("{}", SegmentValidity::Valid), "valid");
        assert_eq!(format!("{}", SegmentValidity::Corrupt), "corrupt");
    }

    #[test]
    fn test_segment_basic() {
        let seg = Segment::new(100, 500, SegmentValidity::Valid);
        assert_eq!(seg.len(), 400);
        assert!(!seg.is_empty());
        assert!(seg.is_extractable());
    }

    #[test]
    fn test_segment_empty() {
        let seg = Segment::new(100, 100, SegmentValidity::Valid);
        assert!(seg.is_empty());
        assert!(!seg.is_extractable());
    }

    #[test]
    fn test_segment_corrupt_not_extractable() {
        let seg = Segment::new(0, 1000, SegmentValidity::Corrupt);
        assert!(!seg.is_extractable());
    }

    #[test]
    fn test_segment_degraded_extractable() {
        let seg = Segment::new(0, 500, SegmentValidity::Degraded);
        assert!(seg.is_extractable());
    }

    #[test]
    fn test_segment_builder_chain() {
        let seg = Segment::new(0, 1000, SegmentValidity::Valid)
            .with_frame_count(30)
            .with_keyframe_start()
            .with_duration(1.0)
            .with_note("First GOP");

        assert_eq!(seg.frame_count, Some(30));
        assert!(seg.starts_with_keyframe);
        assert_eq!(seg.duration_secs, Some(1.0));
        assert_eq!(seg.notes, "First GOP");
    }

    #[test]
    fn test_segment_overlaps() {
        let seg = Segment::new(100, 300, SegmentValidity::Valid);
        assert!(seg.overlaps(200, 400));
        assert!(seg.overlaps(0, 150));
        assert!(!seg.overlaps(300, 400));
        assert!(!seg.overlaps(0, 100));
    }

    #[test]
    fn test_salvage_map_add_and_sort() {
        let mut map = SalvageMap::new(10000);
        map.add_segment(Segment::new(500, 1000, SegmentValidity::Valid));
        map.add_segment(Segment::new(0, 500, SegmentValidity::Valid));
        assert_eq!(map.segments[0].start, 0);
        assert_eq!(map.segments[1].start, 500);
    }

    #[test]
    fn test_salvage_map_extractable_bytes() {
        let mut map = SalvageMap::new(3000);
        map.add_segment(Segment::new(0, 1000, SegmentValidity::Valid));
        map.add_segment(Segment::new(1000, 2000, SegmentValidity::Corrupt));
        map.add_segment(Segment::new(2000, 3000, SegmentValidity::Degraded));
        assert_eq!(map.extractable_bytes(), 2000);
    }

    #[test]
    fn test_recovery_ratio() {
        let mut map = SalvageMap::new(1000);
        map.add_segment(Segment::new(0, 500, SegmentValidity::Valid));
        let ratio = map.recovery_ratio();
        assert!((ratio - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_recovery_ratio_empty() {
        let map = SalvageMap::new(0);
        assert!((map.recovery_ratio() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_segments_at_offset() {
        let mut map = SalvageMap::new(2000);
        map.add_segment(Segment::new(0, 500, SegmentValidity::Valid));
        map.add_segment(Segment::new(500, 1000, SegmentValidity::Corrupt));
        assert_eq!(map.segments_at_offset(250).len(), 1);
        assert_eq!(map.segments_at_offset(750).len(), 1);
        assert!(map.segments_at_offset(1500).is_empty());
    }

    #[test]
    fn test_merge_adjacent() {
        let mut map = SalvageMap::new(5000);
        map.add_segment(Segment::new(0, 1000, SegmentValidity::Valid).with_frame_count(30));
        map.add_segment(Segment::new(1010, 2000, SegmentValidity::Valid).with_frame_count(30));
        map.add_segment(Segment::new(3000, 4000, SegmentValidity::Valid));

        map.merge_adjacent(20);
        // First two should merge (gap = 10 <= 20), third stays separate
        assert_eq!(map.segments.len(), 2);
        assert_eq!(map.segments[0].start, 0);
        assert_eq!(map.segments[0].end, 2000);
        assert_eq!(map.segments[0].frame_count, Some(60));
    }

    #[test]
    fn test_gaps() {
        let mut map = SalvageMap::new(3000);
        map.add_segment(Segment::new(500, 1000, SegmentValidity::Valid));
        map.add_segment(Segment::new(2000, 2500, SegmentValidity::Valid));
        let gaps = map.gaps();
        assert_eq!(gaps, vec![(0, 500), (1000, 2000), (2500, 3000)]);
    }

    #[test]
    fn test_extraction_plan_from_map() {
        let mut map = SalvageMap::new(4000);
        map.add_segment(Segment::new(0, 1000, SegmentValidity::Valid).with_duration(1.0));
        map.add_segment(Segment::new(1000, 2000, SegmentValidity::Corrupt));
        map.add_segment(Segment::new(2000, 3000, SegmentValidity::Valid).with_duration(1.0));

        let plan = ExtractionPlan::from_salvage_map(&map);
        assert_eq!(plan.segment_count(), 2);
        assert_eq!(plan.total_bytes, 2000);
        assert!(plan.needs_remux);
        assert!((plan.estimated_duration_secs.expect("expected estimated_duration_secs to be Some/Ok") - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_extraction_plan_empty() {
        let map = SalvageMap::new(1000);
        let plan = ExtractionPlan::from_salvage_map(&map);
        assert!(plan.is_empty());
        assert!(!plan.needs_remux);
    }
}
