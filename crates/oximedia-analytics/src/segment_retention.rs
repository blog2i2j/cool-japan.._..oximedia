//! Segment-level retention analysis for content chapters and sections.
//!
//! Provides types and functions for measuring how many viewers reached
//! each content segment, and for detecting significant drop-off points.

// ─── Data types ───────────────────────────────────────────────────────────────

/// A named segment (chapter/section) of a content item identified by time range.
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    /// Unique identifier for the segment (e.g. "intro", "chapter_3").
    pub id: String,
    /// Start time of the segment in seconds.
    pub start_sec: f32,
    /// End time of the segment in seconds (exclusive boundary).
    pub end_sec: f32,
}

impl Segment {
    /// Create a new segment with the given parameters.
    pub fn new(id: impl Into<String>, start_sec: f32, end_sec: f32) -> Self {
        Self {
            id: id.into(),
            start_sec,
            end_sec,
        }
    }

    /// Duration of this segment in seconds.
    pub fn duration_sec(&self) -> f32 {
        (self.end_sec - self.start_sec).max(0.0)
    }
}

/// Viewer-level playback data recording how far a viewer reached in the content.
#[derive(Debug, Clone)]
pub struct ViewerSegmentData {
    /// Unique identifier for the viewer's session.
    pub session_id: String,
    /// The furthest position in seconds the viewer reached during playback.
    pub furthest_reach_sec: f32,
}

impl ViewerSegmentData {
    /// Create a new viewer data entry.
    pub fn new(session_id: impl Into<String>, furthest_reach_sec: f32) -> Self {
        Self {
            session_id: session_id.into(),
            furthest_reach_sec,
        }
    }
}

/// Result of segment-level retention analysis.
///
/// `retention_by_segment` contains `(segment_id, fraction)` pairs where
/// `fraction` is in [0, 1] — the proportion of total viewers who watched
/// past the end of each segment.
#[derive(Debug, Clone)]
pub struct SegmentRetentionAnalysis {
    /// The segments that were analysed (same order as input).
    pub segments: Vec<Segment>,
    /// Per-segment retention: `(segment_id, retention_fraction)`.
    ///
    /// `retention_fraction` is the number of viewers who reached past
    /// `segment.end_sec` divided by the total number of viewers.  A value
    /// of 1.0 means every viewer watched through the end of this segment.
    pub retention_by_segment: Vec<(String, f32)>,
}

// ─── Core computation ─────────────────────────────────────────────────────────

/// Compute segment-level retention from a list of segments and viewer data.
///
/// For each segment the retention fraction is:
/// `viewers who reached past segment.end_sec / total viewers`
///
/// Returns an [`SegmentRetentionAnalysis`] with empty `retention_by_segment`
/// when either `segments` or `views` is empty.
pub fn analyze_segment_retention(
    segments: &[Segment],
    views: &[ViewerSegmentData],
) -> SegmentRetentionAnalysis {
    if segments.is_empty() || views.is_empty() {
        return SegmentRetentionAnalysis {
            segments: segments.to_vec(),
            retention_by_segment: Vec::new(),
        };
    }

    let total = views.len() as f32;

    let retention_by_segment = segments
        .iter()
        .map(|seg| {
            // Count viewers whose furthest reach extends past the segment end.
            let reached = views
                .iter()
                .filter(|v| v.furthest_reach_sec >= seg.end_sec)
                .count() as f32;
            (seg.id.clone(), reached / total)
        })
        .collect();

    SegmentRetentionAnalysis {
        segments: segments.to_vec(),
        retention_by_segment,
    }
}

/// Identify segments where the retention dropped by more than `threshold`
/// compared to the previous segment.
///
/// Returns a list of segment IDs where a significant drop-off occurred.
/// The first segment is never included (no previous segment to compare against).
///
/// `threshold` is an absolute retention fraction (e.g. 0.1 means a 10 pp drop).
pub fn drop_off_points(analysis: &SegmentRetentionAnalysis, threshold: f32) -> Vec<String> {
    let mut result = Vec::new();
    let data = &analysis.retention_by_segment;

    for window in data.windows(2) {
        let (_, prev_retention) = &window[0];
        let (seg_id, curr_retention) = &window[1];
        let drop = prev_retention - curr_retention;
        if drop > threshold {
            result.push(seg_id.clone());
        }
    }

    result
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_segments() -> Vec<Segment> {
        vec![
            Segment::new("intro", 0.0, 60.0),
            Segment::new("act_1", 60.0, 300.0),
            Segment::new("act_2", 300.0, 600.0),
            Segment::new("outro", 600.0, 660.0),
        ]
    }

    fn make_views_full_retention() -> Vec<ViewerSegmentData> {
        (0..100)
            .map(|i| ViewerSegmentData::new(format!("s{i}"), 660.0))
            .collect()
    }

    fn make_views_drop_at_act2() -> Vec<ViewerSegmentData> {
        // 100 viewers start, all pass intro and act_1 (reach >= 300),
        // but only 50 pass act_2 (reach >= 600).
        let mut views: Vec<ViewerSegmentData> = (0..100)
            .map(|i| ViewerSegmentData::new(format!("s{i}"), 300.0))
            .collect();
        // First 50 viewers watch through act_2 and outro.
        for v in views.iter_mut().take(50) {
            v.furthest_reach_sec = 660.0;
        }
        views
    }

    #[test]
    fn analyze_segment_retention_empty_segments() {
        let views = make_views_full_retention();
        let result = analyze_segment_retention(&[], &views);
        assert!(result.retention_by_segment.is_empty());
    }

    #[test]
    fn analyze_segment_retention_empty_views() {
        let segments = make_segments();
        let result = analyze_segment_retention(&segments, &[]);
        assert!(result.retention_by_segment.is_empty());
    }

    #[test]
    fn analyze_segment_retention_full_retention() {
        let segments = make_segments();
        let views = make_views_full_retention();
        let result = analyze_segment_retention(&segments, &views);
        assert_eq!(result.retention_by_segment.len(), 4);
        for (_, r) in &result.retention_by_segment {
            assert!((*r - 1.0).abs() < 1e-5, "expected 1.0, got {r}");
        }
    }

    #[test]
    fn analyze_segment_retention_partial() {
        let segments = make_segments();
        let views = make_views_drop_at_act2();
        let result = analyze_segment_retention(&segments, &views);
        // All 100 viewers reach past intro (60s) and act_1 (300s).
        let intro_r = result.retention_by_segment[0].1;
        let act1_r = result.retention_by_segment[1].1;
        // Only 50 viewers reach past act_2 (600s) and outro (660s).
        let act2_r = result.retention_by_segment[2].1;
        let outro_r = result.retention_by_segment[3].1;
        assert!((intro_r - 1.0).abs() < 1e-5, "intro retention={intro_r}");
        assert!((act1_r - 1.0).abs() < 1e-5, "act1 retention={act1_r}");
        assert!((act2_r - 0.5).abs() < 1e-5, "act2 retention={act2_r}");
        assert!((outro_r - 0.5).abs() < 1e-5, "outro retention={outro_r}");
    }

    #[test]
    fn drop_off_points_detects_large_drop() {
        let segments = make_segments();
        let views = make_views_drop_at_act2();
        let analysis = analyze_segment_retention(&segments, &views);
        // There should be a 50 pp drop going from act_1 → act_2.
        let drops = drop_off_points(&analysis, 0.1);
        assert!(
            drops.contains(&"act_2".to_string()),
            "expected act_2 in drops, got {:?}",
            drops
        );
    }

    #[test]
    fn drop_off_points_no_drops_when_threshold_high() {
        let segments = make_segments();
        let views = make_views_drop_at_act2();
        let analysis = analyze_segment_retention(&segments, &views);
        // With threshold > 0.5 nothing should be flagged.
        let drops = drop_off_points(&analysis, 0.6);
        assert!(drops.is_empty(), "expected no drops, got {:?}", drops);
    }

    #[test]
    fn drop_off_points_empty_when_single_segment() {
        let segments = vec![Segment::new("only", 0.0, 60.0)];
        let views = vec![ViewerSegmentData::new("v1", 60.0)];
        let analysis = analyze_segment_retention(&segments, &views);
        let drops = drop_off_points(&analysis, 0.05);
        assert!(drops.is_empty());
    }

    #[test]
    fn segment_duration_sec() {
        let seg = Segment::new("test", 30.0, 90.0);
        assert!((seg.duration_sec() - 60.0).abs() < 1e-5);
    }
}
