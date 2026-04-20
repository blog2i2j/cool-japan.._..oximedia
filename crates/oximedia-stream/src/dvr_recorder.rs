//! DVR recorder with sliding time-window for time-shift functionality.
//!
//! Maintains a bounded deque of [`DvrSegment`]s within a configurable
//! look-back window, enforces a storage-size cap, and can synthesise
//! an HLS VOD playlist for any sub-range of the retained content.

use std::collections::VecDeque;
use std::path::PathBuf;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for a [`DvrRecorder`].
#[derive(Debug, Clone)]
pub struct DvrConfig {
    /// How far back users can seek, in seconds (e.g. `7200` = 2 hours).
    pub window_duration_secs: u64,
    /// Target segment size in seconds (used for bookkeeping / playlist tags).
    pub segment_duration_secs: u32,
    /// Base directory under which segment files would be written.
    pub storage_path: PathBuf,
    /// Hard cap on in-memory segment storage, in bytes.
    pub max_storage_bytes: u64,
}

impl Default for DvrConfig {
    fn default() -> Self {
        Self {
            window_duration_secs: 7200,
            segment_duration_secs: 2,
            storage_path: std::env::temp_dir().join("oximedia-dvr"),
            max_storage_bytes: 4 * 1024 * 1024 * 1024, // 4 GiB
        }
    }
}

// ─── Segment ──────────────────────────────────────────────────────────────────

/// A single packaged segment held in the DVR ring buffer.
#[derive(Debug, Clone)]
pub struct DvrSegment {
    /// Monotonically increasing sequence number assigned by the recorder.
    pub sequence_number: u64,
    /// Presentation timestamp of the first sample in this segment (90 kHz ticks
    /// or arbitrary integer time-base — the recorder treats these as opaque
    /// integers and only compares them).
    pub start_pts: i64,
    /// Presentation timestamp of the last sample in this segment.
    pub end_pts: i64,
    /// Actual wall-clock duration of this segment, in seconds.
    pub duration_secs: f64,
    /// Number of bytes occupied by [`data`](DvrSegment::data).
    pub size_bytes: u64,
    /// Raw segment bytes (fMP4 fragment or MPEG-TS packet payload).
    pub data: Vec<u8>,
    /// `true` when the segment begins on a video key-frame boundary.
    pub is_keyframe_aligned: bool,
}

impl DvrSegment {
    /// Returns `true` if this segment's PTS range overlaps `[range_start, range_end]`.
    ///
    /// Overlap exists unless the segment ends strictly before `range_start` or
    /// begins strictly after `range_end`.
    #[inline]
    fn overlaps(&self, range_start: i64, range_end: i64) -> bool {
        self.end_pts >= range_start && self.start_pts <= range_end
    }
}

// ─── Recorder ─────────────────────────────────────────────────────────────────

/// DVR recorder that maintains a sliding time-window of recorded stream content.
///
/// Segments are stored in a [`VecDeque`] ordered by sequence number (oldest at
/// the front).  When a new segment is pushed the recorder:
///
/// 1. Assigns the next sequence number.
/// 2. Appends the segment to the back of the deque.
/// 3. Evicts segments from the front whose `end_pts` falls outside the
///    configured look-back window (relative to the newest segment's `start_pts`).
/// 4. Enforces the storage cap by evicting oldest segments until the total
///    byte count is within [`DvrConfig::max_storage_bytes`].
pub struct DvrRecorder {
    config: DvrConfig,
    segments: VecDeque<DvrSegment>,
    total_size: u64,
    next_seq: u64,
}

impl DvrRecorder {
    /// Creates a new recorder with the given configuration.
    pub fn new(config: DvrConfig) -> Self {
        Self {
            config,
            segments: VecDeque::new(),
            total_size: 0,
            next_seq: 0,
        }
    }

    // ── Mutation ──────────────────────────────────────────────────────────────

    /// Appends a segment, assigning its sequence number, then evicts any
    /// segments that fall outside the look-back window or exceed the size cap.
    ///
    /// The `sequence_number` field of the supplied `segment` is **overwritten**
    /// by the recorder so that sequence numbers are always assigned
    /// monotonically by this struct alone.
    pub fn push_segment(&mut self, mut segment: DvrSegment) {
        segment.sequence_number = self.next_seq;
        self.next_seq += 1;

        self.total_size += segment.size_bytes;
        self.segments.push_back(segment);

        self.evict_by_window();
        self.evict_by_size();
    }

    /// Removes segments whose `end_pts` is older than the window boundary
    /// computed from the newest segment's `start_pts`.
    fn evict_by_window(&mut self) {
        let newest_start = match self.segments.back() {
            Some(s) => s.start_pts,
            None => return,
        };

        // Convert window_duration_secs to the same unit as PTS.
        // We treat PTS as arbitrary integers; convert seconds to the
        // "90 kHz ticks" convention only if the caller uses it.  Because we
        // cannot know the time-base here we store window_pts_span as the
        // product of window_duration_secs × 90_000 when PTS is in 90 kHz
        // ticks, or the caller can pre-set the pts values accordingly.
        // To keep this generic we expose `window_duration_secs` and assume
        // PTS is already expressed in the same integer unit, so the window
        // boundary is simply the difference between the newest start PTS and
        // the window length *measured in PTS units*.
        //
        // The convention adopted here: PTS is in 90,000 ticks per second
        // (as used by MPEG-TS / HLS).
        let window_pts = (self.config.window_duration_secs as i64).saturating_mul(90_000);
        let window_boundary = newest_start.saturating_sub(window_pts);

        while let Some(front) = self.segments.front() {
            if front.end_pts < window_boundary {
                let removed_size = front.size_bytes;
                self.segments.pop_front();
                self.total_size = self.total_size.saturating_sub(removed_size);
            } else {
                break;
            }
        }
    }

    /// Removes oldest segments until `total_size <= max_storage_bytes`.
    fn evict_by_size(&mut self) {
        while self.total_size > self.config.max_storage_bytes {
            match self.segments.pop_front() {
                Some(seg) => {
                    self.total_size = self.total_size.saturating_sub(seg.size_bytes);
                }
                None => break,
            }
        }
    }

    // ── Queries ───────────────────────────────────────────────────────────────

    /// Returns references to all segments whose PTS range overlaps
    /// `[start_pts, end_pts]`.
    pub fn get_segments_in_range(&self, start_pts: i64, end_pts: i64) -> Vec<&DvrSegment> {
        self.segments
            .iter()
            .filter(|s| s.overlaps(start_pts, end_pts))
            .collect()
    }

    /// Returns the sequence number of the segment whose PTS range contains
    /// `pts`, or `None` if `pts` is outside the retained window.
    pub fn seek_to_pts(&self, pts: i64) -> Option<u64> {
        self.segments
            .iter()
            .find(|s| s.start_pts <= pts && pts <= s.end_pts)
            .map(|s| s.sequence_number)
    }

    /// Generates an HLS VOD M3U8 playlist covering the time range
    /// `[start_pts, end_pts]`.
    ///
    /// Only segments overlapping the range are included.  The playlist uses
    /// `EXT-X-TARGETDURATION` derived from the configured segment duration and
    /// marks the playlist with `EXT-X-PLAYLIST-TYPE:VOD` plus `EXT-X-ENDLIST`.
    pub fn generate_vod_playlist(&self, start_pts: i64, end_pts: i64) -> String {
        let segments = self.get_segments_in_range(start_pts, end_pts);

        let target_duration = self.config.segment_duration_secs;
        // Compute the true max segment duration from the selected segments, at
        // least as large as the configured target.
        let actual_target = segments
            .iter()
            .map(|s| s.duration_secs.ceil() as u32)
            .max()
            .unwrap_or(target_duration)
            .max(target_duration);

        let first_seq = segments.first().map(|s| s.sequence_number).unwrap_or(0);

        let mut playlist = String::with_capacity(512 + segments.len() * 80);
        playlist.push_str("#EXTM3U\n");
        playlist.push_str("#EXT-X-VERSION:3\n");
        playlist.push_str("#EXT-X-PLAYLIST-TYPE:VOD\n");
        playlist.push_str(&format!("#EXT-X-TARGETDURATION:{actual_target}\n"));
        playlist.push_str(&format!("#EXT-X-MEDIA-SEQUENCE:{first_seq}\n"));

        for seg in &segments {
            // Flag discontinuities for non-keyframe-aligned segments.
            if !seg.is_keyframe_aligned {
                playlist.push_str("#EXT-X-DISCONTINUITY\n");
            }
            playlist.push_str(&format!("#EXTINF:{:.6},\n", seg.duration_secs));
            playlist.push_str(&format!(
                "{}/segment_{:010}.ts\n",
                self.config.storage_path.display(),
                seg.sequence_number
            ));
        }

        playlist.push_str("#EXT-X-ENDLIST\n");
        playlist
    }

    /// Returns the `start_pts` of the oldest retained segment, or `None` if
    /// the recorder is empty.
    pub fn oldest_available_pts(&self) -> Option<i64> {
        self.segments.front().map(|s| s.start_pts)
    }

    /// Returns the `end_pts` of the newest retained segment, or `None` if the
    /// recorder is empty.
    pub fn newest_available_pts(&self) -> Option<i64> {
        self.segments.back().map(|s| s.end_pts)
    }

    /// Total wall-clock duration (seconds) spanned by all retained segments.
    pub fn total_duration_secs(&self) -> f64 {
        self.segments.iter().map(|s| s.duration_secs).sum()
    }

    /// Number of segments currently in the buffer.
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Total bytes consumed by all retained segment payloads.
    pub fn storage_used_bytes(&self) -> u64 {
        self.total_size
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a minimal `DvrConfig` suitable for unit tests.
    fn test_config() -> DvrConfig {
        DvrConfig {
            window_duration_secs: 10, // 10 s window
            segment_duration_secs: 2,
            storage_path: std::env::temp_dir().join("oximedia-dvr-test"),
            max_storage_bytes: 1024 * 1024 * 100, // 100 MiB
        }
    }

    /// Returns a PTS value in 90 kHz ticks for a given second offset.
    fn pts(secs: i64) -> i64 {
        secs * 90_000
    }

    /// Creates a synthetic 2-second segment starting at `start_secs`.
    fn make_segment(start_secs: i64, size_bytes: u64) -> DvrSegment {
        DvrSegment {
            sequence_number: 0, // overwritten by push_segment
            start_pts: pts(start_secs),
            end_pts: pts(start_secs + 2) - 1,
            duration_secs: 2.0,
            size_bytes,
            data: vec![0u8; size_bytes as usize],
            is_keyframe_aligned: true,
        }
    }

    // ── 1. Empty recorder ────────────────────────────────────────────────────

    #[test]
    fn test_empty_recorder_queries_return_none_or_zero() {
        let rec = DvrRecorder::new(test_config());
        assert!(rec.oldest_available_pts().is_none());
        assert!(rec.newest_available_pts().is_none());
        assert_eq!(rec.total_duration_secs(), 0.0);
        assert_eq!(rec.segment_count(), 0);
        assert_eq!(rec.storage_used_bytes(), 0);
    }

    // ── 2. Single segment push ───────────────────────────────────────────────

    #[test]
    fn test_single_segment_push_and_query() {
        let mut rec = DvrRecorder::new(test_config());
        rec.push_segment(make_segment(0, 1024));

        assert_eq!(rec.segment_count(), 1);
        assert_eq!(rec.storage_used_bytes(), 1024);
        assert_eq!(rec.total_duration_secs(), 2.0);
        assert_eq!(rec.oldest_available_pts(), Some(pts(0)));
        assert_eq!(rec.newest_available_pts(), Some(pts(2) - 1));
    }

    // ── 3. Sequence numbers are assigned monotonically ───────────────────────

    #[test]
    fn test_sequence_numbers_are_monotonically_assigned() {
        let mut rec = DvrRecorder::new(test_config());
        for i in 0..5_i64 {
            rec.push_segment(make_segment(i * 2, 100));
        }
        let seqs: Vec<u64> = rec.segments.iter().map(|s| s.sequence_number).collect();
        for (i, &seq) in seqs.iter().enumerate() {
            assert_eq!(seq, i as u64, "sequence mismatch at index {i}");
        }
    }

    // ── 4. Window eviction removes oldest segments ───────────────────────────

    #[test]
    fn test_window_eviction_removes_old_segments() {
        // window = 10 s → 900_000 pts ticks.
        // Push segments at t=0, 2, 4, ..., 20 (11 segments × 2 s each).
        // When the segment at t=20 arrives, oldest_start = 20*90000.
        // boundary = 20*90000 − 10*90000 = 10*90000 = pts(10).
        // Segments ending before pts(10) should be evicted (t=0..8 end at
        // pts(2)−1 .. pts(10)−1, so segments starting at t=0,2,4,6,8 end
        // at pts(2)−1 .. pts(10)−1 which are all < pts(10)).
        let mut rec = DvrRecorder::new(test_config());
        for i in 0..=10_i64 {
            rec.push_segment(make_segment(i * 2, 100));
        }
        // Newest start_pts = pts(20); boundary = pts(10).
        // Segments with end_pts < pts(10):  t=0 (ends pts(2)-1), t=2 (ends pts(4)-1),
        // t=4 (ends pts(6)-1), t=6 (ends pts(8)-1), t=8 (ends pts(10)-1).
        // pts(10)-1 < pts(10) → evicted.  t=10 ends pts(12)-1 → kept.
        let oldest = rec.oldest_available_pts().expect("should have segments");
        assert!(
            oldest >= pts(10),
            "oldest PTS {oldest} should be >= pts(10)={} after eviction",
            pts(10)
        );
    }

    // ── 5. Size-cap eviction ─────────────────────────────────────────────────

    #[test]
    fn test_size_cap_eviction_keeps_total_within_limit() {
        let config = DvrConfig {
            window_duration_secs: 3600,
            segment_duration_secs: 2,
            storage_path: std::env::temp_dir().join("oximedia-dvr-test"),
            max_storage_bytes: 500, // very tight cap
        };
        let mut rec = DvrRecorder::new(config);
        for i in 0..20_i64 {
            rec.push_segment(make_segment(i * 2, 100));
        }
        assert!(
            rec.storage_used_bytes() <= 500,
            "storage {} exceeds cap 500",
            rec.storage_used_bytes()
        );
    }

    // ── 6. get_segments_in_range ─────────────────────────────────────────────

    #[test]
    fn test_get_segments_in_range_returns_correct_subset() {
        let mut rec = DvrRecorder::new(test_config());
        // Push t=0,2,4,6,8 (all within 10 s window).
        for i in 0..5_i64 {
            rec.push_segment(make_segment(i * 2, 100));
        }
        // Query range covering t=2..t=5 (pts(2)..pts(5)).
        let segs = rec.get_segments_in_range(pts(2), pts(5));
        // Segments at t=2 and t=4 overlap [pts(2), pts(5)].
        assert_eq!(
            segs.len(),
            2,
            "expected 2 overlapping segments, got {}",
            segs.len()
        );
        assert_eq!(segs[0].start_pts, pts(2));
        assert_eq!(segs[1].start_pts, pts(4));
    }

    // ── 7. seek_to_pts finds correct segment ────────────────────────────────

    #[test]
    fn test_seek_to_pts_finds_enclosing_segment() {
        let mut rec = DvrRecorder::new(test_config());
        // Push segments at t=0,2,4,6,8.  Sequences 0,1,2,3,4 respectively.
        // The 10-second window does not evict anything (newest start pts(8),
        // boundary = pts(-2) which is negative, so all are retained).
        // The segment at t=2 is seq=1; pts(3) falls inside it.
        for i in 0..5_i64 {
            rec.push_segment(make_segment(i * 2, 100));
        }
        // pts(3) falls inside the segment starting at t=2 (seq 1).
        let seq = rec.seek_to_pts(pts(3)).expect("should find segment");
        assert_eq!(seq, 1, "expected sequence 1, got {seq}");
    }

    // ── 8. seek_to_pts returns None for out-of-range PTS ────────────────────

    #[test]
    fn test_seek_to_pts_returns_none_for_out_of_range() {
        let mut rec = DvrRecorder::new(test_config());
        rec.push_segment(make_segment(0, 100));
        // pts(100) is far beyond the single retained segment.
        assert!(rec.seek_to_pts(pts(100)).is_none());
    }

    // ── 9. VOD playlist contains correct segments ────────────────────────────

    #[test]
    fn test_generate_vod_playlist_structure() {
        let mut rec = DvrRecorder::new(test_config());
        for i in 0..5_i64 {
            rec.push_segment(make_segment(i * 2, 100));
        }
        let playlist = rec.generate_vod_playlist(pts(0), pts(9));
        assert!(playlist.starts_with("#EXTM3U\n"), "missing #EXTM3U header");
        assert!(
            playlist.contains("#EXT-X-PLAYLIST-TYPE:VOD"),
            "missing VOD type tag"
        );
        assert!(
            playlist.contains("#EXT-X-ENDLIST"),
            "missing #EXT-X-ENDLIST"
        );
        assert!(playlist.contains("#EXTINF:"), "missing #EXTINF entries");
        // All 5 segments should appear (pts 0..9 covers t=0,2,4,6,8).
        let extinf_count = playlist.matches("#EXTINF:").count();
        assert_eq!(
            extinf_count, 5,
            "expected 5 #EXTINF entries, got {extinf_count}"
        );
    }

    // ── 10. VOD playlist for empty range ────────────────────────────────────

    #[test]
    fn test_generate_vod_playlist_empty_range() {
        let mut rec = DvrRecorder::new(test_config());
        for i in 0..3_i64 {
            rec.push_segment(make_segment(i * 2, 100));
        }
        // Range that does not overlap any retained segment.
        let playlist = rec.generate_vod_playlist(pts(1000), pts(2000));
        assert!(playlist.contains("#EXT-X-ENDLIST"));
        assert!(
            !playlist.contains("#EXTINF:"),
            "should have no segments in out-of-range playlist"
        );
    }

    // ── 11. total_duration_secs reflects actual segments ────────────────────

    #[test]
    fn test_total_duration_secs_accumulates_correctly() {
        let mut rec = DvrRecorder::new(test_config());
        for i in 0..4_i64 {
            rec.push_segment(make_segment(i * 2, 100));
        }
        // 4 segments × 2.0 s each = 8.0 s
        let dur = rec.total_duration_secs();
        assert!((dur - 8.0).abs() < 1e-9, "expected 8.0 s total, got {dur}");
    }

    // ── 12. Push after eviction re-uses correct sequence numbering ───────────

    #[test]
    fn test_sequence_numbers_continue_after_eviction() {
        let config = DvrConfig {
            window_duration_secs: 4, // 4 s window → 360_000 pts ticks
            segment_duration_secs: 2,
            storage_path: std::env::temp_dir().join("oximedia-dvr-test"),
            max_storage_bytes: 1024 * 1024,
        };
        let mut rec = DvrRecorder::new(config);
        // Push 6 segments (t=0..10 step 2).  The 4 s window keeps only the last 2–3.
        for i in 0..6_i64 {
            rec.push_segment(make_segment(i * 2, 50));
        }
        // All sequence numbers so far: 0..5.
        let last_seq = rec
            .segments
            .back()
            .map(|s| s.sequence_number)
            .expect("recorder should not be empty");
        assert_eq!(last_seq, 5, "last seq should be 5, got {last_seq}");

        // Push one more and verify it gets seq 6.
        rec.push_segment(make_segment(12, 50));
        let new_seq = rec
            .segments
            .back()
            .map(|s| s.sequence_number)
            .expect("recorder should not be empty");
        assert_eq!(new_seq, 6, "new seq should be 6, got {new_seq}");
    }

    // ── 13. Non-keyframe-aligned segment adds DISCONTINUITY tag ─────────────

    #[test]
    fn test_vod_playlist_discontinuity_tag_for_non_keyframe_segment() {
        let mut rec = DvrRecorder::new(test_config());
        let mut seg = make_segment(0, 100);
        seg.is_keyframe_aligned = false;
        rec.push_segment(seg);
        let playlist = rec.generate_vod_playlist(pts(0), pts(2));
        assert!(
            playlist.contains("#EXT-X-DISCONTINUITY"),
            "missing #EXT-X-DISCONTINUITY for non-keyframe-aligned segment"
        );
    }
}
