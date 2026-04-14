#![allow(dead_code)]
//! Batch PTS/DTS timestamp rewriting and processing.
//!
//! Provides high-throughput timestamp operations on packet slices, designed for
//! remuxing and transcoding pipelines where thousands of packets need their
//! timestamps adjusted in bulk.
//!
//! # Key Capabilities
//!
//! - **Batch offset**: shift all timestamps by a fixed delta (e.g. to start a
//!   stream at zero).
//! - **Batch rescale**: convert an entire stream from one timebase to another
//!   (e.g. 90 kHz -> 1 kHz).
//! - **Batch repair**: fix negative DTS, enforce monotonicity, and fill gaps.
//! - **Batch statistics**: min/max/mean/jitter over large packet arrays.
//! - **Batch filter**: remove packets outside a PTS window.
//!
//! All operations are designed to work on contiguous slices for cache-friendly
//! access patterns.
//!
//! # Example
//!
//! ```
//! use oximedia_container::pts_dts_batch::{TimestampEntry, BatchProcessor};
//!
//! let mut entries = vec![
//!     TimestampEntry::new(0, 0, 1000),
//!     TimestampEntry::new(1, 3600, 4600),
//!     TimestampEntry::new(2, 7200, 8200),
//! ];
//!
//! let mut proc = BatchProcessor::new();
//! proc.apply_offset(&mut entries, 1000);
//! assert_eq!(entries[0].pts, 1000);
//! assert_eq!(entries[0].dts, 2000);
//! ```

#![forbid(unsafe_code)]

use std::collections::HashMap;

// ─── Timestamp entry ───────────────────────────────────────────────────────

/// A lightweight timestamp record for batch processing.
///
/// Unlike [`PtsDts`](super::pts_dts::PtsDts) which uses `Option<i64>`, this
/// struct uses sentinel values to avoid branching in tight loops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimestampEntry {
    /// Packet sequence number (monotonically increasing).
    pub seq: u64,
    /// Presentation timestamp in timebase ticks.
    pub pts: i64,
    /// Decode timestamp in timebase ticks.
    pub dts: i64,
    /// Stream index this packet belongs to.
    pub stream_index: u32,
    /// Whether this packet is a keyframe.
    pub is_keyframe: bool,
}

impl TimestampEntry {
    /// Creates a new entry with default stream index 0 and non-keyframe.
    #[must_use]
    pub fn new(seq: u64, pts: i64, dts: i64) -> Self {
        Self {
            seq,
            pts,
            dts,
            stream_index: 0,
            is_keyframe: false,
        }
    }

    /// Creates an entry with stream index and keyframe flag.
    #[must_use]
    pub fn with_stream(seq: u64, pts: i64, dts: i64, stream_index: u32, is_keyframe: bool) -> Self {
        Self {
            seq,
            pts,
            dts,
            stream_index,
            is_keyframe,
        }
    }

    /// Returns the PTS-DTS delta (B-frame reorder distance).
    #[must_use]
    pub fn reorder_delta(&self) -> i64 {
        self.pts - self.dts
    }
}

// ─── Timebase ──────────────────────────────────────────────────────────────

/// Represents a timebase as a rational number (numerator / denominator).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Timebase {
    /// Numerator (typically 1).
    pub num: i64,
    /// Denominator (e.g. 90000 for MPEG-TS, 1000 for milliseconds).
    pub den: i64,
}

impl Timebase {
    /// Creates a new timebase.
    ///
    /// # Panics
    ///
    /// Panics if `den` is zero.
    #[must_use]
    pub fn new(num: i64, den: i64) -> Self {
        assert!(den != 0, "timebase denominator must not be zero");
        Self { num, den }
    }

    /// Common timebase: milliseconds (1/1000).
    pub const MILLISECONDS: Self = Self { num: 1, den: 1000 };

    /// Common timebase: MPEG-TS 90 kHz (1/90000).
    pub const MPEG_TS: Self = Self {
        num: 1,
        den: 90_000,
    };

    /// Common timebase: microseconds (1/1000000).
    pub const MICROSECONDS: Self = Self {
        num: 1,
        den: 1_000_000,
    };

    /// Common timebase: Matroska nanosecond-based (1/1000000000).
    pub const NANOSECONDS: Self = Self {
        num: 1,
        den: 1_000_000_000,
    };

    /// Rescale a single timestamp from this timebase to another.
    #[must_use]
    pub fn rescale_to(&self, ts: i64, dst: &Timebase) -> i64 {
        if self.den == 0 || dst.num == 0 {
            return 0;
        }
        // ts * (self.num / self.den) / (dst.num / dst.den)
        // = ts * self.num * dst.den / (self.den * dst.num)
        let numer = ts as i128 * self.num as i128 * dst.den as i128;
        let denom = self.den as i128 * dst.num as i128;
        if denom == 0 {
            return 0;
        }
        (numer / denom) as i64
    }
}

// ─── Batch statistics ──────────────────────────────────────────────────────

/// Statistics computed over a batch of timestamp entries.
#[derive(Debug, Clone, Copy)]
pub struct BatchStats {
    /// Number of entries analysed.
    pub count: usize,
    /// Minimum PTS value.
    pub min_pts: i64,
    /// Maximum PTS value.
    pub max_pts: i64,
    /// Minimum DTS value.
    pub min_dts: i64,
    /// Maximum DTS value.
    pub max_dts: i64,
    /// Average PTS delta between consecutive packets.
    pub mean_pts_delta: f64,
    /// Maximum PTS delta (jitter indicator).
    pub max_pts_delta: i64,
    /// Number of packets with negative DTS.
    pub negative_dts_count: usize,
    /// Number of non-monotonic DTS transitions.
    pub non_monotonic_dts_count: usize,
    /// Number of keyframes.
    pub keyframe_count: usize,
}

impl Default for BatchStats {
    fn default() -> Self {
        Self {
            count: 0,
            min_pts: i64::MAX,
            max_pts: i64::MIN,
            min_dts: i64::MAX,
            max_dts: i64::MIN,
            mean_pts_delta: 0.0,
            max_pts_delta: 0,
            negative_dts_count: 0,
            non_monotonic_dts_count: 0,
            keyframe_count: 0,
        }
    }
}

// ─── Batch processor ───────────────────────────────────────────────────────

/// High-throughput batch processor for timestamp arrays.
///
/// Provides in-place mutations on `&mut [TimestampEntry]` slices for common
/// remuxing operations.
#[derive(Debug, Clone, Default)]
pub struct BatchProcessor {
    /// Number of entries processed (lifetime counter).
    processed_count: u64,
    /// Number of repairs applied (lifetime counter).
    repair_count: u64,
}

impl BatchProcessor {
    /// Creates a new batch processor.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the total number of entries processed.
    #[must_use]
    pub fn processed_count(&self) -> u64 {
        self.processed_count
    }

    /// Returns the total number of repairs applied.
    #[must_use]
    pub fn repair_count(&self) -> u64 {
        self.repair_count
    }

    /// Shifts all PTS and DTS values by a fixed offset.
    ///
    /// Positive `offset` shifts timestamps forward; negative shifts backward.
    pub fn apply_offset(&mut self, entries: &mut [TimestampEntry], offset: i64) {
        for entry in entries.iter_mut() {
            entry.pts += offset;
            entry.dts += offset;
        }
        self.processed_count += entries.len() as u64;
    }

    /// Rescales all timestamps from one timebase to another.
    ///
    /// Uses 128-bit intermediate arithmetic to avoid overflow.
    pub fn rescale(&mut self, entries: &mut [TimestampEntry], src: &Timebase, dst: &Timebase) {
        for entry in entries.iter_mut() {
            entry.pts = src.rescale_to(entry.pts, dst);
            entry.dts = src.rescale_to(entry.dts, dst);
        }
        self.processed_count += entries.len() as u64;
    }

    /// Clamps all negative DTS values to zero, adjusting PTS accordingly.
    ///
    /// Returns the number of entries that were repaired.
    pub fn fix_negative_dts(&mut self, entries: &mut [TimestampEntry]) -> usize {
        let mut repaired = 0usize;
        for entry in entries.iter_mut() {
            if entry.dts < 0 {
                let shift = -entry.dts;
                entry.pts += shift;
                entry.dts = 0;
                repaired += 1;
            }
        }
        self.repair_count += repaired as u64;
        self.processed_count += entries.len() as u64;
        repaired
    }

    /// Enforces monotonically non-decreasing DTS within each stream.
    ///
    /// When a DTS is less than the previous DTS for the same stream, it is
    /// clamped to the previous value.  Returns the number of entries repaired.
    pub fn enforce_dts_monotonicity(&mut self, entries: &mut [TimestampEntry]) -> usize {
        let mut repaired = 0usize;
        let mut last_dts: HashMap<u32, i64> = HashMap::new();

        for entry in entries.iter_mut() {
            if let Some(&prev) = last_dts.get(&entry.stream_index) {
                if entry.dts < prev {
                    let delta = prev - entry.dts;
                    entry.dts = prev;
                    entry.pts += delta;
                    repaired += 1;
                }
            }
            last_dts.insert(entry.stream_index, entry.dts);
        }
        self.repair_count += repaired as u64;
        self.processed_count += entries.len() as u64;
        repaired
    }

    /// Rebase all timestamps so that the earliest DTS becomes zero.
    ///
    /// This is useful when remuxing a clip extracted from the middle of a
    /// longer file.
    pub fn rebase_to_zero(&mut self, entries: &mut [TimestampEntry]) {
        if entries.is_empty() {
            return;
        }
        let min_dts = entries.iter().map(|e| e.dts).min().unwrap_or(0);
        if min_dts != 0 {
            self.apply_offset(entries, -min_dts);
        }
    }

    /// Filters entries to only keep those within a PTS window [start, end).
    ///
    /// Returns a new `Vec` containing only the matching entries; the original
    /// slice is not modified.
    #[must_use]
    pub fn filter_pts_range(
        &mut self,
        entries: &[TimestampEntry],
        start: i64,
        end: i64,
    ) -> Vec<TimestampEntry> {
        self.processed_count += entries.len() as u64;
        entries
            .iter()
            .filter(|e| e.pts >= start && e.pts < end)
            .copied()
            .collect()
    }

    /// Computes batch statistics over the given entries.
    #[must_use]
    pub fn compute_stats(&self, entries: &[TimestampEntry]) -> BatchStats {
        if entries.is_empty() {
            return BatchStats::default();
        }

        let mut stats = BatchStats {
            count: entries.len(),
            ..Default::default()
        };

        let mut prev_pts: Option<i64> = None;
        let mut prev_dts_per_stream: HashMap<u32, i64> = HashMap::new();
        let mut total_delta: i64 = 0;
        let mut delta_count: usize = 0;

        for entry in entries {
            if entry.pts < stats.min_pts {
                stats.min_pts = entry.pts;
            }
            if entry.pts > stats.max_pts {
                stats.max_pts = entry.pts;
            }
            if entry.dts < stats.min_dts {
                stats.min_dts = entry.dts;
            }
            if entry.dts > stats.max_dts {
                stats.max_dts = entry.dts;
            }
            if entry.dts < 0 {
                stats.negative_dts_count += 1;
            }
            if entry.is_keyframe {
                stats.keyframe_count += 1;
            }

            if let Some(prev) = prev_pts {
                let delta = (entry.pts - prev).abs();
                total_delta += delta;
                delta_count += 1;
                if delta > stats.max_pts_delta {
                    stats.max_pts_delta = delta;
                }
            }
            prev_pts = Some(entry.pts);

            if let Some(&prev_dts) = prev_dts_per_stream.get(&entry.stream_index) {
                if entry.dts < prev_dts {
                    stats.non_monotonic_dts_count += 1;
                }
            }
            prev_dts_per_stream.insert(entry.stream_index, entry.dts);
        }

        if delta_count > 0 {
            stats.mean_pts_delta = total_delta as f64 / delta_count as f64;
        }

        stats
    }

    /// Groups entries by stream index.
    #[must_use]
    pub fn group_by_stream(entries: &[TimestampEntry]) -> HashMap<u32, Vec<TimestampEntry>> {
        let mut groups: HashMap<u32, Vec<TimestampEntry>> = HashMap::new();
        for entry in entries {
            groups.entry(entry.stream_index).or_default().push(*entry);
        }
        groups
    }

    /// Fills DTS gaps by linear interpolation between keyframes.
    ///
    /// When a gap (jump) in DTS exceeding `max_gap` ticks is detected, the
    /// intermediate packets are re-stamped by linearly interpolating between
    /// the last good DTS and the next keyframe DTS.
    ///
    /// Returns the number of entries that were adjusted.
    pub fn fill_dts_gaps(&mut self, entries: &mut [TimestampEntry], max_gap: i64) -> usize {
        if entries.len() < 2 {
            return 0;
        }
        let mut repaired = 0usize;

        // Find gap boundaries
        let mut gap_ranges: Vec<(usize, usize, i64, i64)> = Vec::new();
        let mut i = 1;
        while i < entries.len() {
            let delta = entries[i].dts - entries[i - 1].dts;
            if delta.abs() > max_gap {
                // Find next keyframe to anchor interpolation
                let anchor_dts = entries[i - 1].dts;
                let mut end = i;
                while end < entries.len() && !entries[end].is_keyframe {
                    end += 1;
                }
                if end < entries.len() {
                    gap_ranges.push((i, end, anchor_dts, entries[end].dts));
                }
                i = end + 1;
            } else {
                i += 1;
            }
        }

        for (start, end, from_dts, to_dts) in gap_ranges {
            let span = end - start;
            if span == 0 {
                continue;
            }
            for (offset, entry) in entries[start..end].iter_mut().enumerate() {
                let fraction = (offset + 1) as f64 / (span + 1) as f64;
                let interpolated = from_dts + (((to_dts - from_dts) as f64) * fraction) as i64;
                let old_delta = entry.pts - entry.dts;
                entry.dts = interpolated;
                entry.pts = interpolated + old_delta;
                repaired += 1;
            }
        }

        self.repair_count += repaired as u64;
        self.processed_count += entries.len() as u64;
        repaired
    }

    /// Applies a multiplicative speed factor to all timestamps.
    ///
    /// A `factor` of 2.0 doubles the duration (slow motion); 0.5 halves it
    /// (fast forward).
    pub fn apply_speed_factor(&mut self, entries: &mut [TimestampEntry], factor: f64) {
        for entry in entries.iter_mut() {
            entry.pts = (entry.pts as f64 * factor) as i64;
            entry.dts = (entry.dts as f64 * factor) as i64;
        }
        self.processed_count += entries.len() as u64;
    }

    /// Renumbers sequence numbers starting from `start_seq`.
    pub fn renumber_sequences(&mut self, entries: &mut [TimestampEntry], start_seq: u64) {
        for (idx, entry) in entries.iter_mut().enumerate() {
            entry.seq = start_seq + idx as u64;
        }
        self.processed_count += entries.len() as u64;
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_entries() -> Vec<TimestampEntry> {
        vec![
            TimestampEntry::with_stream(0, 0, 0, 0, true),
            TimestampEntry::with_stream(1, 3600, 3600, 0, false),
            TimestampEntry::with_stream(2, 7200, 7200, 0, false),
            TimestampEntry::with_stream(3, 10800, 10800, 0, true),
            TimestampEntry::with_stream(4, 14400, 14400, 0, false),
        ]
    }

    #[test]
    fn test_apply_offset_forward() {
        let mut entries = sample_entries();
        let mut proc = BatchProcessor::new();
        proc.apply_offset(&mut entries, 1000);
        assert_eq!(entries[0].pts, 1000);
        assert_eq!(entries[0].dts, 1000);
        assert_eq!(entries[4].pts, 15400);
        assert_eq!(proc.processed_count(), 5);
    }

    #[test]
    fn test_apply_offset_negative() {
        let mut entries = sample_entries();
        let mut proc = BatchProcessor::new();
        proc.apply_offset(&mut entries, -1000);
        assert_eq!(entries[0].pts, -1000);
        assert_eq!(entries[1].pts, 2600);
    }

    #[test]
    fn test_rescale_90khz_to_ms() {
        let mut entries = vec![
            TimestampEntry::new(0, 90_000, 90_000),
            TimestampEntry::new(1, 180_000, 180_000),
        ];
        let mut proc = BatchProcessor::new();
        proc.rescale(&mut entries, &Timebase::MPEG_TS, &Timebase::MILLISECONDS);
        assert_eq!(entries[0].pts, 1000);
        assert_eq!(entries[1].pts, 2000);
    }

    #[test]
    fn test_rescale_ms_to_90khz() {
        let mut entries = vec![
            TimestampEntry::new(0, 1000, 1000),
            TimestampEntry::new(1, 2000, 2000),
        ];
        let mut proc = BatchProcessor::new();
        proc.rescale(&mut entries, &Timebase::MILLISECONDS, &Timebase::MPEG_TS);
        assert_eq!(entries[0].pts, 90_000);
        assert_eq!(entries[1].pts, 180_000);
    }

    #[test]
    fn test_fix_negative_dts() {
        let mut entries = vec![
            TimestampEntry::new(0, 100, -200),
            TimestampEntry::new(1, 500, 300),
            TimestampEntry::new(2, 1000, -50),
        ];
        let mut proc = BatchProcessor::new();
        let repaired = proc.fix_negative_dts(&mut entries);
        assert_eq!(repaired, 2);
        assert_eq!(entries[0].dts, 0);
        assert_eq!(entries[0].pts, 300); // shifted by 200
        assert_eq!(entries[1].dts, 300); // untouched
        assert_eq!(entries[2].dts, 0);
        assert_eq!(entries[2].pts, 1050); // shifted by 50
    }

    #[test]
    fn test_enforce_dts_monotonicity() {
        let mut entries = vec![
            TimestampEntry::with_stream(0, 100, 100, 0, true),
            TimestampEntry::with_stream(1, 200, 200, 0, false),
            TimestampEntry::with_stream(2, 300, 150, 0, false), // out of order
            TimestampEntry::with_stream(3, 400, 400, 0, false),
        ];
        let mut proc = BatchProcessor::new();
        let repaired = proc.enforce_dts_monotonicity(&mut entries);
        assert_eq!(repaired, 1);
        assert_eq!(entries[2].dts, 200); // clamped to previous
        assert_eq!(entries[2].pts, 350); // adjusted by same delta
    }

    #[test]
    fn test_rebase_to_zero() {
        let mut entries = vec![
            TimestampEntry::new(0, 5100, 5000),
            TimestampEntry::new(1, 8700, 8600),
        ];
        let mut proc = BatchProcessor::new();
        proc.rebase_to_zero(&mut entries);
        assert_eq!(entries[0].dts, 0);
        assert_eq!(entries[0].pts, 100);
        assert_eq!(entries[1].dts, 3600);
    }

    #[test]
    fn test_filter_pts_range() {
        let entries = sample_entries();
        let mut proc = BatchProcessor::new();
        let filtered = proc.filter_pts_range(&entries, 3000, 11000);
        assert_eq!(filtered.len(), 3);
        assert_eq!(filtered[0].pts, 3600);
        assert_eq!(filtered[1].pts, 7200);
        assert_eq!(filtered[2].pts, 10800);
    }

    #[test]
    fn test_compute_stats() {
        let entries = sample_entries();
        let proc = BatchProcessor::new();
        let stats = proc.compute_stats(&entries);

        assert_eq!(stats.count, 5);
        assert_eq!(stats.min_pts, 0);
        assert_eq!(stats.max_pts, 14400);
        assert_eq!(stats.negative_dts_count, 0);
        assert_eq!(stats.non_monotonic_dts_count, 0);
        assert_eq!(stats.keyframe_count, 2);
        assert_eq!(stats.max_pts_delta, 3600);
        assert!((stats.mean_pts_delta - 3600.0).abs() < 0.1);
    }

    #[test]
    fn test_compute_stats_negative_dts() {
        let entries = vec![
            TimestampEntry::new(0, 0, -100),
            TimestampEntry::new(1, 100, 100),
        ];
        let proc = BatchProcessor::new();
        let stats = proc.compute_stats(&entries);
        assert_eq!(stats.negative_dts_count, 1);
    }

    #[test]
    fn test_compute_stats_non_monotonic() {
        let entries = vec![
            TimestampEntry::with_stream(0, 100, 100, 0, false),
            TimestampEntry::with_stream(1, 200, 200, 0, false),
            TimestampEntry::with_stream(2, 300, 150, 0, false), // non-monotonic
        ];
        let proc = BatchProcessor::new();
        let stats = proc.compute_stats(&entries);
        assert_eq!(stats.non_monotonic_dts_count, 1);
    }

    #[test]
    fn test_group_by_stream() {
        let entries = vec![
            TimestampEntry::with_stream(0, 0, 0, 0, true),
            TimestampEntry::with_stream(1, 100, 100, 1, true),
            TimestampEntry::with_stream(2, 200, 200, 0, false),
            TimestampEntry::with_stream(3, 300, 300, 1, false),
        ];
        let groups = BatchProcessor::group_by_stream(&entries);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups.get(&0).map(|g| g.len()), Some(2));
        assert_eq!(groups.get(&1).map(|g| g.len()), Some(2));
    }

    #[test]
    fn test_apply_speed_factor() {
        let mut entries = vec![
            TimestampEntry::new(0, 1000, 1000),
            TimestampEntry::new(1, 2000, 2000),
        ];
        let mut proc = BatchProcessor::new();
        proc.apply_speed_factor(&mut entries, 0.5);
        assert_eq!(entries[0].pts, 500);
        assert_eq!(entries[1].pts, 1000);
    }

    #[test]
    fn test_renumber_sequences() {
        let mut entries = sample_entries();
        let mut proc = BatchProcessor::new();
        proc.renumber_sequences(&mut entries, 100);
        assert_eq!(entries[0].seq, 100);
        assert_eq!(entries[4].seq, 104);
    }

    #[test]
    fn test_fill_dts_gaps() {
        let mut entries = vec![
            TimestampEntry::with_stream(0, 0, 0, 0, true),
            TimestampEntry::with_stream(1, 3600, 3600, 0, false),
            // Gap: jump from 3600 to 100000
            TimestampEntry::with_stream(2, 100100, 100000, 0, false),
            TimestampEntry::with_stream(3, 100200, 100100, 0, false),
            TimestampEntry::with_stream(4, 103700, 103600, 0, true), // keyframe anchor
        ];
        let mut proc = BatchProcessor::new();
        let repaired = proc.fill_dts_gaps(&mut entries, 50_000);
        assert!(repaired >= 1);
        // Entries between the gap should be interpolated
        assert!(entries[2].dts > 3600);
        assert!(entries[2].dts < 103600);
    }

    #[test]
    fn test_timebase_rescale() {
        let src = Timebase::new(1, 48000);
        let dst = Timebase::MILLISECONDS;
        let result = src.rescale_to(48000, &dst);
        assert_eq!(result, 1000);
    }

    #[test]
    fn test_reorder_delta() {
        let entry = TimestampEntry::new(0, 7200, 3600);
        assert_eq!(entry.reorder_delta(), 3600);
    }

    #[test]
    fn test_empty_entries_operations() {
        let mut entries: Vec<TimestampEntry> = Vec::new();
        let mut proc = BatchProcessor::new();

        proc.apply_offset(&mut entries, 1000);
        proc.rebase_to_zero(&mut entries);
        let repaired = proc.fix_negative_dts(&mut entries);
        assert_eq!(repaired, 0);

        let stats = proc.compute_stats(&entries);
        assert_eq!(stats.count, 0);
    }

    #[test]
    fn test_chained_operations() {
        let mut entries = vec![
            TimestampEntry::new(0, 90_000, -90_000),
            TimestampEntry::new(1, 180_000, 90_000),
            TimestampEntry::new(2, 270_000, 180_000),
        ];
        let mut proc = BatchProcessor::new();

        // 1. Fix negative DTS
        let repaired = proc.fix_negative_dts(&mut entries);
        assert_eq!(repaired, 1);
        assert_eq!(entries[0].dts, 0);

        // 2. Rescale from 90kHz to ms
        proc.rescale(&mut entries, &Timebase::MPEG_TS, &Timebase::MILLISECONDS);

        // 3. Rebase to zero
        proc.rebase_to_zero(&mut entries);

        // All DTS should be non-negative and in ms
        for entry in &entries {
            assert!(entry.dts >= 0);
        }
        // fix_negative_dts (3) + rescale (3) = 6 minimum
        // rebase_to_zero may or may not add more depending on min_dts
        assert!(proc.processed_count() >= 6);
    }
}
