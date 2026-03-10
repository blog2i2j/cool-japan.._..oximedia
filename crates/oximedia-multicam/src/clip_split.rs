//! Multicam clip splitting, sequencing, and grouping operations.

#![allow(dead_code)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::module_name_repetitions)]

// ── SplitPoint ────────────────────────────────────────────────────────────────

/// A detected (or manual) point at which a clip should be split
#[derive(Debug, Clone)]
pub struct SplitPoint {
    /// Frame number at which to split
    pub frame: u64,
    /// Camera that is the intended source at this point
    pub source_camera: u32,
    /// Confidence score in [0, 1]; values above 0.8 indicate a clean cut
    pub confidence: f32,
}

impl SplitPoint {
    /// Create a new split point
    pub fn new(frame: u64, source_camera: u32, confidence: f32) -> Self {
        Self {
            frame,
            source_camera,
            confidence,
        }
    }

    /// Returns `true` when confidence > 0.8 (clean cut threshold)
    pub fn is_clean_cut(&self) -> bool {
        self.confidence > 0.8
    }
}

// ── MultiCamClip ──────────────────────────────────────────────────────────────

/// A single clip segment from one camera in a multicam sequence
#[derive(Debug, Clone)]
pub struct MultiCamClip {
    /// Unique clip identifier
    pub id: u64,
    /// First frame (inclusive)
    pub start_frame: u64,
    /// Last frame (exclusive)
    pub end_frame: u64,
    /// Camera this clip came from
    pub camera_id: u32,
    /// Position of this clip in the sequence
    pub sequence_number: u32,
}

impl MultiCamClip {
    /// Create a new clip
    pub fn new(
        id: u64,
        start_frame: u64,
        end_frame: u64,
        camera_id: u32,
        sequence_number: u32,
    ) -> Self {
        Self {
            id,
            start_frame,
            end_frame,
            camera_id,
            sequence_number,
        }
    }

    /// Number of frames in this clip
    pub fn duration_frames(&self) -> u64 {
        self.end_frame.saturating_sub(self.start_frame)
    }

    /// Whether this clip's time range overlaps with `other`'s time range
    pub fn overlaps(&self, other: &MultiCamClip) -> bool {
        self.start_frame < other.end_frame && other.start_frame < self.end_frame
    }
}

// ── ClipSplitter ──────────────────────────────────────────────────────────────

/// Splits `MultiCamClip` objects at detected or manual cut points
#[derive(Debug, Clone)]
pub struct ClipSplitter {
    /// Frame rate used for timing calculations
    pub frame_rate: f64,
}

impl ClipSplitter {
    /// Create a new splitter for the given frame rate
    pub fn new(frame_rate: f64) -> Self {
        Self { frame_rate }
    }

    /// Split `clip` at `split_frame`.
    ///
    /// Returns `None` if `split_frame` is not strictly inside `[start_frame, end_frame)`.
    /// The second piece receives `clip.id + 1` and `sequence_number + 1`.
    pub fn split_at(
        &self,
        clip: &MultiCamClip,
        split_frame: u64,
    ) -> Option<(MultiCamClip, MultiCamClip)> {
        if split_frame <= clip.start_frame || split_frame >= clip.end_frame {
            return None;
        }
        let first = MultiCamClip::new(
            clip.id,
            clip.start_frame,
            split_frame,
            clip.camera_id,
            clip.sequence_number,
        );
        let second = MultiCamClip::new(
            clip.id + 1,
            split_frame,
            clip.end_frame,
            clip.camera_id,
            clip.sequence_number + 1,
        );
        Some((first, second))
    }

    /// Split `clip` at every split point whose frame lies inside the clip.
    ///
    /// Split points are sorted by frame before processing so the order of the
    /// input slice does not matter.
    pub fn auto_split(
        &self,
        clip: &MultiCamClip,
        split_points: &[SplitPoint],
    ) -> Vec<MultiCamClip> {
        let mut frames: Vec<u64> = split_points
            .iter()
            .map(|sp| sp.frame)
            .filter(|&f| f > clip.start_frame && f < clip.end_frame)
            .collect();
        frames.sort_unstable();
        frames.dedup();

        if frames.is_empty() {
            return vec![clip.clone()];
        }

        let mut result = Vec::with_capacity(frames.len() + 1);
        let mut current_start = clip.start_frame;
        let mut current_id = clip.id;
        let mut seq = clip.sequence_number;

        for &split in &frames {
            result.push(MultiCamClip::new(
                current_id,
                current_start,
                split,
                clip.camera_id,
                seq,
            ));
            current_start = split;
            current_id += 1;
            seq += 1;
        }
        // Final segment
        result.push(MultiCamClip::new(
            current_id,
            current_start,
            clip.end_frame,
            clip.camera_id,
            seq,
        ));
        result
    }
}

// ── MultiCamSequence ─────────────────────────────────────────────────────────

/// An ordered sequence of multicam clips (potentially from multiple cameras)
#[derive(Debug, Default)]
pub struct MultiCamSequence {
    /// All clips in this sequence
    pub clips: Vec<MultiCamClip>,
    /// Total number of distinct cameras in the sequence
    pub total_cameras: u32,
}

impl MultiCamSequence {
    /// Create an empty sequence
    pub fn new(total_cameras: u32) -> Self {
        Self {
            clips: Vec::new(),
            total_cameras,
        }
    }

    /// Add a clip to the sequence
    pub fn add_clip(&mut self, clip: MultiCamClip) {
        self.clips.push(clip);
    }

    /// All clips belonging to `camera_id`
    pub fn clips_for_camera(&self, camera_id: u32) -> Vec<&MultiCamClip> {
        self.clips
            .iter()
            .filter(|c| c.camera_id == camera_id)
            .collect()
    }

    /// Sum of `duration_frames()` for every clip in the sequence
    pub fn total_duration_frames(&self) -> u64 {
        self.clips.iter().map(MultiCamClip::duration_frames).sum()
    }

    /// Number of camera switches (adjacent clips with different `camera_id`)
    pub fn camera_cut_count(&self) -> usize {
        self.clips
            .windows(2)
            .filter(|w| w[0].camera_id != w[1].camera_id)
            .count()
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn clip(id: u64, start: u64, end: u64, cam: u32) -> MultiCamClip {
        MultiCamClip::new(id, start, end, cam, 0)
    }

    fn splitter() -> ClipSplitter {
        ClipSplitter::new(25.0)
    }

    // ── SplitPoint ───────────────────────────────────────────────────────────

    #[test]
    fn test_split_point_clean_cut_above_threshold() {
        let sp = SplitPoint::new(100, 1, 0.9);
        assert!(sp.is_clean_cut());
    }

    #[test]
    fn test_split_point_not_clean_at_threshold() {
        // Exactly 0.8 is NOT > 0.8
        let sp = SplitPoint::new(100, 1, 0.8);
        assert!(!sp.is_clean_cut());
    }

    #[test]
    fn test_split_point_not_clean_below_threshold() {
        let sp = SplitPoint::new(100, 1, 0.5);
        assert!(!sp.is_clean_cut());
    }

    // ── MultiCamClip ─────────────────────────────────────────────────────────

    #[test]
    fn test_clip_duration_frames() {
        let c = clip(1, 0, 100, 1);
        assert_eq!(c.duration_frames(), 100);
    }

    #[test]
    fn test_clip_overlaps_true() {
        let a = clip(1, 0, 50, 1);
        let b = clip(2, 30, 80, 2);
        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));
    }

    #[test]
    fn test_clip_overlaps_false_adjacent() {
        let a = clip(1, 0, 50, 1);
        let b = clip(2, 50, 100, 2);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_clip_overlaps_false_gap() {
        let a = clip(1, 0, 40, 1);
        let b = clip(2, 60, 100, 2);
        assert!(!a.overlaps(&b));
    }

    // ── ClipSplitter ─────────────────────────────────────────────────────────

    #[test]
    fn test_split_at_middle() {
        let c = clip(10, 0, 100, 1);
        let result = splitter().split_at(&c, 40);
        assert!(result.is_some());
        let (first, second) = result.expect("multicam test operation should succeed");
        assert_eq!(first.start_frame, 0);
        assert_eq!(first.end_frame, 40);
        assert_eq!(second.start_frame, 40);
        assert_eq!(second.end_frame, 100);
        assert_eq!(second.id, 11);
        assert_eq!(second.sequence_number, 1);
    }

    #[test]
    fn test_split_at_boundary_returns_none() {
        let c = clip(1, 0, 100, 1);
        assert!(splitter().split_at(&c, 0).is_none());
        assert!(splitter().split_at(&c, 100).is_none());
    }

    #[test]
    fn test_auto_split_single_point() {
        let c = clip(1, 0, 100, 1);
        let pts = vec![SplitPoint::new(50, 1, 0.9)];
        let pieces = splitter().auto_split(&c, &pts);
        assert_eq!(pieces.len(), 2);
        assert_eq!(pieces[0].end_frame, 50);
        assert_eq!(pieces[1].start_frame, 50);
    }

    #[test]
    fn test_auto_split_no_valid_points_returns_original() {
        let c = clip(1, 0, 100, 1);
        // Split points outside the clip range
        let pts = vec![SplitPoint::new(200, 1, 0.9)];
        let pieces = splitter().auto_split(&c, &pts);
        assert_eq!(pieces.len(), 1);
        assert_eq!(pieces[0].start_frame, 0);
        assert_eq!(pieces[0].end_frame, 100);
    }

    #[test]
    fn test_auto_split_multiple_points_sorted() {
        let c = clip(1, 0, 100, 1);
        let pts = vec![
            SplitPoint::new(75, 1, 0.9),
            SplitPoint::new(25, 1, 0.9),
            SplitPoint::new(50, 1, 0.9),
        ];
        let pieces = splitter().auto_split(&c, &pts);
        assert_eq!(pieces.len(), 4);
        assert_eq!(pieces[0].end_frame, 25);
        assert_eq!(pieces[1].end_frame, 50);
        assert_eq!(pieces[2].end_frame, 75);
        assert_eq!(pieces[3].end_frame, 100);
    }

    // ── MultiCamSequence ─────────────────────────────────────────────────────

    #[test]
    fn test_sequence_add_and_total_duration() {
        let mut seq = MultiCamSequence::new(2);
        seq.add_clip(clip(1, 0, 50, 1));
        seq.add_clip(clip(2, 50, 100, 2));
        assert_eq!(seq.total_duration_frames(), 100);
    }

    #[test]
    fn test_sequence_clips_for_camera() {
        let mut seq = MultiCamSequence::new(2);
        seq.add_clip(clip(1, 0, 50, 1));
        seq.add_clip(clip(2, 50, 100, 2));
        seq.add_clip(clip(3, 100, 150, 1));
        assert_eq!(seq.clips_for_camera(1).len(), 2);
        assert_eq!(seq.clips_for_camera(2).len(), 1);
    }

    #[test]
    fn test_sequence_camera_cut_count() {
        let mut seq = MultiCamSequence::new(2);
        seq.add_clip(clip(1, 0, 50, 1));
        seq.add_clip(clip(2, 50, 100, 2));
        seq.add_clip(clip(3, 100, 150, 1));
        // Cam1→Cam2 and Cam2→Cam1 = 2 cuts
        assert_eq!(seq.camera_cut_count(), 2);
    }

    #[test]
    fn test_sequence_no_cuts_same_camera() {
        let mut seq = MultiCamSequence::new(1);
        seq.add_clip(clip(1, 0, 50, 1));
        seq.add_clip(clip(2, 50, 100, 1));
        assert_eq!(seq.camera_cut_count(), 0);
    }
}
