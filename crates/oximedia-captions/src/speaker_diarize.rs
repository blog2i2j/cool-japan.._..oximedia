//! Speaker diarization for captions.
//!
//! Provides types for labelling caption segments with speaker identities and
//! for querying or merging those segments.

#![allow(dead_code)]
#![allow(missing_docs)]

// ── SpeakerLabel ─────────────────────────────────────────────────────────────

/// An identifier and optional human-readable name for a speaker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeakerLabel {
    /// Numeric speaker identifier (0-indexed within a diarization result).
    pub id: u8,
    /// Optional display name assigned by the operator.
    pub name: Option<String>,
}

impl SpeakerLabel {
    /// Create a new speaker label.
    pub fn new(id: u8, name: Option<impl Into<String>>) -> Self {
        Self {
            id,
            name: name.map(Into::into),
        }
    }

    /// Returns the human-readable name, falling back to `"Speaker N"` when no
    /// name has been assigned.
    #[must_use]
    pub fn display_name(&self) -> String {
        match &self.name {
            Some(n) => n.clone(),
            None => format!("Speaker {}", self.id),
        }
    }
}

// ── SpeakerSegment ────────────────────────────────────────────────────────────

/// A contiguous caption segment attributed to a single speaker.
#[derive(Debug, Clone)]
pub struct SpeakerSegment {
    /// Speaker who produced this segment.
    pub speaker: SpeakerLabel,
    /// Start timestamp in milliseconds.
    pub start_ms: u64,
    /// End timestamp in milliseconds.
    pub end_ms: u64,
    /// Caption text for this segment.
    pub text: String,
}

impl SpeakerSegment {
    /// Create a new speaker segment.
    pub fn new(speaker: SpeakerLabel, start_ms: u64, end_ms: u64, text: impl Into<String>) -> Self {
        Self {
            speaker,
            start_ms,
            end_ms,
            text: text.into(),
        }
    }

    /// Duration of this segment in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// Number of whitespace-separated words in this segment.
    #[must_use]
    pub fn word_count(&self) -> usize {
        if self.text.is_empty() {
            0
        } else {
            self.text.split_whitespace().count()
        }
    }
}

// ── DiarizationResult ─────────────────────────────────────────────────────────

/// The complete diarization output: an ordered list of speaker segments.
#[derive(Debug, Default)]
pub struct DiarizationResult {
    /// Segments in presentation order (by `start_ms`).
    pub segments: Vec<SpeakerSegment>,
}

impl DiarizationResult {
    /// Create an empty diarization result.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of distinct speaker IDs present in the result.
    #[must_use]
    pub fn speaker_count(&self) -> usize {
        let mut ids: Vec<u8> = self.segments.iter().map(|s| s.speaker.id).collect();
        ids.sort_unstable();
        ids.dedup();
        ids.len()
    }

    /// All segments attributed to `speaker_id`.
    #[must_use]
    pub fn segments_for(&self, speaker_id: u8) -> Vec<&SpeakerSegment> {
        self.segments
            .iter()
            .filter(|s| s.speaker.id == speaker_id)
            .collect()
    }

    /// Merge consecutive segments for `speaker_id` where the gap between
    /// adjacent segments is less than 500 ms.
    ///
    /// The merged segment's text is formed by joining the constituent texts
    /// with a single space.  The speaker label of the first segment in each
    /// merged group is used.
    #[must_use]
    pub fn merge_consecutive(&self, speaker_id: u8) -> Vec<SpeakerSegment> {
        const GAP_THRESHOLD_MS: u64 = 500;

        let mut speaker_segs: Vec<&SpeakerSegment> = self.segments_for(speaker_id);
        // Sort by start time to ensure correct adjacency checks.
        speaker_segs.sort_by_key(|s| s.start_ms);

        let mut merged: Vec<SpeakerSegment> = Vec::new();

        for seg in speaker_segs {
            if let Some(last) = merged.last_mut() {
                let gap = seg.start_ms.saturating_sub(last.end_ms);
                if gap < GAP_THRESHOLD_MS {
                    // Merge into the current group.
                    last.end_ms = seg.end_ms;
                    if !last.text.is_empty() {
                        last.text.push(' ');
                    }
                    last.text.push_str(&seg.text);
                    continue;
                }
            }
            merged.push(seg.clone());
        }

        merged
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn label(id: u8, name: Option<&str>) -> SpeakerLabel {
        SpeakerLabel::new(id, name)
    }

    fn seg(speaker_id: u8, start: u64, end: u64, text: &str) -> SpeakerSegment {
        SpeakerSegment::new(label(speaker_id, None), start, end, text)
    }

    // ── SpeakerLabel ──

    #[test]
    fn test_label_display_name_with_name() {
        let l = label(0, Some("Alice"));
        assert_eq!(l.display_name(), "Alice");
    }

    #[test]
    fn test_label_display_name_fallback() {
        let l = label(3, None::<&str>);
        assert_eq!(l.display_name(), "Speaker 3");
    }

    #[test]
    fn test_label_display_name_speaker_zero() {
        let l = label(0, None::<&str>);
        assert_eq!(l.display_name(), "Speaker 0");
    }

    // ── SpeakerSegment ──

    #[test]
    fn test_segment_duration_ms() {
        let s = seg(0, 1000, 3000, "hello");
        assert_eq!(s.duration_ms(), 2000);
    }

    #[test]
    fn test_segment_word_count() {
        let s = seg(0, 0, 1000, "one two three");
        assert_eq!(s.word_count(), 3);
    }

    #[test]
    fn test_segment_word_count_empty() {
        let s = seg(0, 0, 100, "");
        assert_eq!(s.word_count(), 0);
    }

    // ── DiarizationResult ──

    #[test]
    fn test_speaker_count_empty() {
        let dr = DiarizationResult::new();
        assert_eq!(dr.speaker_count(), 0);
    }

    #[test]
    fn test_speaker_count_two_speakers() {
        let mut dr = DiarizationResult::new();
        dr.segments.push(seg(0, 0, 1000, "hello"));
        dr.segments.push(seg(1, 1200, 2000, "hi"));
        dr.segments.push(seg(0, 2100, 3000, "world"));
        assert_eq!(dr.speaker_count(), 2);
    }

    #[test]
    fn test_segments_for_filters_correctly() {
        let mut dr = DiarizationResult::new();
        dr.segments.push(seg(0, 0, 500, "A"));
        dr.segments.push(seg(1, 600, 1000, "B"));
        dr.segments.push(seg(0, 1100, 1500, "C"));
        let s0 = dr.segments_for(0);
        assert_eq!(s0.len(), 2);
    }

    #[test]
    fn test_segments_for_unknown_speaker() {
        let mut dr = DiarizationResult::new();
        dr.segments.push(seg(0, 0, 500, "A"));
        assert!(dr.segments_for(9).is_empty());
    }

    #[test]
    fn test_merge_consecutive_within_gap() {
        let mut dr = DiarizationResult::new();
        // gap = 200 ms < 500 ms → should merge
        dr.segments.push(seg(0, 0, 1000, "Hello"));
        dr.segments.push(seg(0, 1200, 2000, "world"));
        let merged = dr.merge_consecutive(0);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].text, "Hello world");
        assert_eq!(merged[0].end_ms, 2000);
    }

    #[test]
    fn test_merge_consecutive_exceeds_gap() {
        let mut dr = DiarizationResult::new();
        // gap = 600 ms > 500 ms → separate segments
        dr.segments.push(seg(0, 0, 1000, "First"));
        dr.segments.push(seg(0, 1600, 2200, "Second"));
        let merged = dr.merge_consecutive(0);
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn test_merge_consecutive_ignores_other_speakers() {
        let mut dr = DiarizationResult::new();
        dr.segments.push(seg(0, 0, 500, "Alice"));
        dr.segments.push(seg(1, 600, 1000, "Bob"));
        dr.segments.push(seg(0, 700, 1200, "Alice again"));
        let merged = dr.merge_consecutive(0);
        // gap between Alice segs = 700 - 500 = 200 ms → merged
        assert_eq!(merged.len(), 1);
    }
}
