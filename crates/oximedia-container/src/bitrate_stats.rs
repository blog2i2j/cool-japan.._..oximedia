#![allow(dead_code)]
//! Container-level bitrate statistics and analysis.
//!
//! Computes per-track and aggregate bitrate metrics from a stream of
//! [`PacketSizeRecord`] entries. Useful for quality-control checks, encoding
//! verification, and ABR ladder validation.

use std::collections::HashMap;
use std::fmt;

/// Index used to identify a track inside a container.
pub type TrackIndex = u32;

/// A record of one packet's size and timing used for bitrate accounting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PacketSizeRecord {
    /// Track this packet belongs to.
    pub track: TrackIndex,
    /// Byte size of the compressed packet (excluding container overhead).
    pub size_bytes: u32,
    /// Decode timestamp in timescale units.
    pub dts: u64,
    /// `true` if this is a key-frame / sync sample.
    pub is_keyframe: bool,
}

impl PacketSizeRecord {
    /// Creates a new packet size record.
    #[must_use]
    pub const fn new(track: TrackIndex, size_bytes: u32, dts: u64, is_keyframe: bool) -> Self {
        Self {
            track,
            size_bytes,
            dts,
            is_keyframe,
        }
    }
}

/// Per-track bitrate statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct TrackBitrateStats {
    /// Track index.
    pub track: TrackIndex,
    /// Number of packets analysed.
    pub packet_count: u64,
    /// Total bytes across all packets.
    pub total_bytes: u64,
    /// Smallest packet in bytes.
    pub min_packet_bytes: u32,
    /// Largest packet in bytes.
    pub max_packet_bytes: u32,
    /// First DTS seen (timescale units).
    pub first_dts: u64,
    /// Last DTS seen (timescale units).
    pub last_dts: u64,
    /// Number of keyframes.
    pub keyframe_count: u64,
    /// Total bytes in keyframe packets.
    pub keyframe_bytes: u64,
}

impl TrackBitrateStats {
    /// Returns the average bitrate in kilobits per second.
    ///
    /// `timescale` converts DTS units to seconds.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn avg_bitrate_kbps(&self, timescale: u32) -> f64 {
        if timescale == 0 || self.first_dts == self.last_dts {
            return 0.0;
        }
        let duration_s = (self.last_dts - self.first_dts) as f64 / f64::from(timescale);
        if duration_s <= 0.0 {
            return 0.0;
        }
        (self.total_bytes as f64 * 8.0) / (duration_s * 1000.0)
    }

    /// Returns the average packet size in bytes.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn avg_packet_size(&self) -> f64 {
        if self.packet_count == 0 {
            return 0.0;
        }
        self.total_bytes as f64 / self.packet_count as f64
    }

    /// Returns the keyframe ratio (0.0..=1.0).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn keyframe_ratio(&self) -> f64 {
        if self.packet_count == 0 {
            return 0.0;
        }
        self.keyframe_count as f64 / self.packet_count as f64
    }
}

impl fmt::Display for TrackBitrateStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Track {} — {} packets, {} bytes, {} keyframes",
            self.track, self.packet_count, self.total_bytes, self.keyframe_count,
        )
    }
}

/// Accumulates packet records and computes per-track bitrate statistics.
#[derive(Debug, Default)]
pub struct BitrateAnalyzer {
    /// Per-track accumulators.
    accumulators: HashMap<TrackIndex, TrackAccumulator>,
}

/// Internal per-track state.
#[derive(Debug)]
struct TrackAccumulator {
    packet_count: u64,
    total_bytes: u64,
    min_packet_bytes: u32,
    max_packet_bytes: u32,
    first_dts: u64,
    last_dts: u64,
    keyframe_count: u64,
    keyframe_bytes: u64,
}

impl TrackAccumulator {
    fn new(rec: &PacketSizeRecord) -> Self {
        Self {
            packet_count: 1,
            total_bytes: u64::from(rec.size_bytes),
            min_packet_bytes: rec.size_bytes,
            max_packet_bytes: rec.size_bytes,
            first_dts: rec.dts,
            last_dts: rec.dts,
            keyframe_count: u64::from(rec.is_keyframe),
            keyframe_bytes: if rec.is_keyframe {
                u64::from(rec.size_bytes)
            } else {
                0
            },
        }
    }

    fn update(&mut self, rec: &PacketSizeRecord) {
        self.packet_count += 1;
        self.total_bytes += u64::from(rec.size_bytes);
        if rec.size_bytes < self.min_packet_bytes {
            self.min_packet_bytes = rec.size_bytes;
        }
        if rec.size_bytes > self.max_packet_bytes {
            self.max_packet_bytes = rec.size_bytes;
        }
        if rec.dts < self.first_dts {
            self.first_dts = rec.dts;
        }
        if rec.dts > self.last_dts {
            self.last_dts = rec.dts;
        }
        if rec.is_keyframe {
            self.keyframe_count += 1;
            self.keyframe_bytes += u64::from(rec.size_bytes);
        }
    }

    fn to_stats(&self, track: TrackIndex) -> TrackBitrateStats {
        TrackBitrateStats {
            track,
            packet_count: self.packet_count,
            total_bytes: self.total_bytes,
            min_packet_bytes: self.min_packet_bytes,
            max_packet_bytes: self.max_packet_bytes,
            first_dts: self.first_dts,
            last_dts: self.last_dts,
            keyframe_count: self.keyframe_count,
            keyframe_bytes: self.keyframe_bytes,
        }
    }
}

impl BitrateAnalyzer {
    /// Creates a new, empty analyzer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            accumulators: HashMap::new(),
        }
    }

    /// Feeds a single packet record into the analyzer.
    pub fn feed(&mut self, rec: &PacketSizeRecord) {
        self.accumulators
            .entry(rec.track)
            .and_modify(|acc| acc.update(rec))
            .or_insert_with(|| TrackAccumulator::new(rec));
    }

    /// Feeds a slice of records.
    pub fn feed_batch(&mut self, records: &[PacketSizeRecord]) {
        for rec in records {
            self.feed(rec);
        }
    }

    /// Returns the set of track indices that have been seen.
    #[must_use]
    pub fn tracks(&self) -> Vec<TrackIndex> {
        let mut v: Vec<_> = self.accumulators.keys().copied().collect();
        v.sort_unstable();
        v
    }

    /// Returns statistics for a single track, or `None` if unseen.
    #[must_use]
    pub fn stats_for(&self, track: TrackIndex) -> Option<TrackBitrateStats> {
        self.accumulators.get(&track).map(|acc| acc.to_stats(track))
    }

    /// Returns statistics for all tracks.
    #[must_use]
    pub fn all_stats(&self) -> Vec<TrackBitrateStats> {
        let mut out: Vec<_> = self
            .accumulators
            .iter()
            .map(|(&t, acc)| acc.to_stats(t))
            .collect();
        out.sort_by_key(|s| s.track);
        out
    }

    /// Returns the total number of packets fed across all tracks.
    #[must_use]
    pub fn total_packets(&self) -> u64 {
        self.accumulators.values().map(|a| a.packet_count).sum()
    }

    /// Resets the analyzer, discarding all accumulated data.
    pub fn reset(&mut self) {
        self.accumulators.clear();
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn pkt(track: u32, size: u32, dts: u64, kf: bool) -> PacketSizeRecord {
        PacketSizeRecord::new(track, size, dts, kf)
    }

    // 1. new analyzer is empty
    #[test]
    fn test_new_analyzer_empty() {
        let a = BitrateAnalyzer::new();
        assert!(a.tracks().is_empty());
        assert_eq!(a.total_packets(), 0);
    }

    // 2. feed single packet
    #[test]
    fn test_feed_single() {
        let mut a = BitrateAnalyzer::new();
        a.feed(&pkt(0, 1000, 0, true));
        assert_eq!(a.total_packets(), 1);
        assert_eq!(a.tracks(), vec![0]);
    }

    // 3. feed_batch multiple tracks
    #[test]
    fn test_feed_batch() {
        let mut a = BitrateAnalyzer::new();
        a.feed_batch(&[
            pkt(0, 500, 0, true),
            pkt(1, 200, 0, true),
            pkt(0, 300, 3000, false),
        ]);
        assert_eq!(a.total_packets(), 3);
        assert_eq!(a.tracks(), vec![0, 1]);
    }

    // 4. per-track total_bytes
    #[test]
    fn test_total_bytes() {
        let mut a = BitrateAnalyzer::new();
        a.feed(&pkt(0, 500, 0, true));
        a.feed(&pkt(0, 700, 3000, false));
        let s = a.stats_for(0).expect("operation should succeed");
        assert_eq!(s.total_bytes, 1200);
    }

    // 5. min / max packet bytes
    #[test]
    fn test_min_max() {
        let mut a = BitrateAnalyzer::new();
        a.feed_batch(&[
            pkt(0, 100, 0, true),
            pkt(0, 500, 3000, false),
            pkt(0, 300, 6000, false),
        ]);
        let s = a.stats_for(0).expect("operation should succeed");
        assert_eq!(s.min_packet_bytes, 100);
        assert_eq!(s.max_packet_bytes, 500);
    }

    // 6. first_dts / last_dts
    #[test]
    fn test_dts_range() {
        let mut a = BitrateAnalyzer::new();
        a.feed_batch(&[pkt(0, 100, 1000, true), pkt(0, 100, 5000, false)]);
        let s = a.stats_for(0).expect("operation should succeed");
        assert_eq!(s.first_dts, 1000);
        assert_eq!(s.last_dts, 5000);
    }

    // 7. avg_bitrate_kbps
    #[test]
    fn test_avg_bitrate_kbps() {
        let mut a = BitrateAnalyzer::new();
        // 1000 bytes over 1 second at timescale 90000
        a.feed_batch(&[pkt(0, 500, 0, true), pkt(0, 500, 90000, false)]);
        let s = a.stats_for(0).expect("operation should succeed");
        let br = s.avg_bitrate_kbps(90000);
        // 1000 * 8 / (1.0 * 1000) = 8.0
        assert!((br - 8.0).abs() < 1e-6);
    }

    // 8. avg_bitrate_kbps zero timescale
    #[test]
    fn test_avg_bitrate_zero_timescale() {
        let s = TrackBitrateStats {
            track: 0,
            packet_count: 1,
            total_bytes: 1000,
            min_packet_bytes: 1000,
            max_packet_bytes: 1000,
            first_dts: 0,
            last_dts: 90000,
            keyframe_count: 1,
            keyframe_bytes: 1000,
        };
        assert_eq!(s.avg_bitrate_kbps(0), 0.0);
    }

    // 9. avg_packet_size
    #[test]
    fn test_avg_packet_size() {
        let mut a = BitrateAnalyzer::new();
        a.feed_batch(&[pkt(0, 200, 0, true), pkt(0, 400, 3000, false)]);
        let s = a.stats_for(0).expect("operation should succeed");
        assert!((s.avg_packet_size() - 300.0).abs() < f64::EPSILON);
    }

    // 10. keyframe_ratio
    #[test]
    fn test_keyframe_ratio() {
        let mut a = BitrateAnalyzer::new();
        a.feed_batch(&[
            pkt(0, 500, 0, true),
            pkt(0, 100, 3000, false),
            pkt(0, 100, 6000, false),
        ]);
        let s = a.stats_for(0).expect("operation should succeed");
        assert!((s.keyframe_ratio() - 1.0 / 3.0).abs() < 1e-9);
    }

    // 11. stats_for unknown track
    #[test]
    fn test_stats_for_unknown() {
        let a = BitrateAnalyzer::new();
        assert!(a.stats_for(99).is_none());
    }

    // 12. all_stats sorted by track
    #[test]
    fn test_all_stats_sorted() {
        let mut a = BitrateAnalyzer::new();
        a.feed(&pkt(2, 100, 0, true));
        a.feed(&pkt(0, 100, 0, true));
        let all = a.all_stats();
        assert_eq!(all[0].track, 0);
        assert_eq!(all[1].track, 2);
    }

    // 13. reset clears state
    #[test]
    fn test_reset() {
        let mut a = BitrateAnalyzer::new();
        a.feed(&pkt(0, 100, 0, true));
        a.reset();
        assert_eq!(a.total_packets(), 0);
    }

    // 14. keyframe bytes tracked
    #[test]
    fn test_keyframe_bytes() {
        let mut a = BitrateAnalyzer::new();
        a.feed_batch(&[pkt(0, 800, 0, true), pkt(0, 100, 3000, false)]);
        let s = a.stats_for(0).expect("operation should succeed");
        assert_eq!(s.keyframe_bytes, 800);
    }

    // 15. TrackBitrateStats display
    #[test]
    fn test_stats_display() {
        let s = TrackBitrateStats {
            track: 1,
            packet_count: 10,
            total_bytes: 5000,
            min_packet_bytes: 200,
            max_packet_bytes: 800,
            first_dts: 0,
            last_dts: 90000,
            keyframe_count: 2,
            keyframe_bytes: 1600,
        };
        let text = format!("{s}");
        assert!(text.contains("Track 1"));
        assert!(text.contains("10 packets"));
    }

    // 16. avg_packet_size zero packets
    #[test]
    fn test_avg_packet_size_zero() {
        let s = TrackBitrateStats {
            track: 0,
            packet_count: 0,
            total_bytes: 0,
            min_packet_bytes: 0,
            max_packet_bytes: 0,
            first_dts: 0,
            last_dts: 0,
            keyframe_count: 0,
            keyframe_bytes: 0,
        };
        assert_eq!(s.avg_packet_size(), 0.0);
    }
}
