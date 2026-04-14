//! Media segment data types for HLS/DASH/CMAF streaming pipelines.
//!
//! A *segment* is a contiguous slice of a media stream — either an initialisation
//! segment containing codec configuration, or a media segment containing
//! presentation samples in a well-defined time range.
//!
//! # Key Types
//!
//! - [`SegmentKind`] — discriminates init vs. media vs. partial media segments
//! - [`ByteRange`] — inclusive byte-range into a parent resource
//! - [`SegmentTimeRange`] — half-open presentation time range `[start, end)` in
//!   rational ticks
//! - [`SegmentMetadata`] — optional per-segment descriptors (sequence, bitrate, …)
//! - [`MediaSegment`] — the top-level descriptor combining all of the above
//!
//! # Example
//!
//! ```
//! use oximedia_core::media_segment::{
//!     ByteRange, MediaSegment, SegmentKind, SegmentMetadata, SegmentTimeRange,
//! };
//! use oximedia_core::types::Rational;
//!
//! let time_base = Rational::new(1, 90_000);
//! let range = SegmentTimeRange::new(0, 270_000, time_base).expect("valid range");
//! let seg = MediaSegment::builder(SegmentKind::Media, range)
//!     .with_byte_range(ByteRange::new(0, 49_999))
//!     .with_metadata(SegmentMetadata {
//!         sequence: 1,
//!         bitrate_bps: Some(4_000_000),
//!         independent: true,
//!         label: None,
//!     })
//!     .build();
//!
//! assert!((seg.time_range().duration_secs() - 3.0).abs() < 1e-6);
//! ```

use crate::error::{OxiError, OxiResult};
use crate::types::Rational;

// ---------------------------------------------------------------------------
// SegmentKind
// ---------------------------------------------------------------------------

/// Discriminator for the role of a segment within a streaming manifest.
///
/// In HLS parlance an *init* segment carries the `ftyp`/`moov` boxes while a
/// *media* segment carries `moof`/`mdat` boxes. Partial segments are used by
/// Low-Latency HLS (LL-HLS, RFC 8216bis).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SegmentKind {
    /// Initialisation segment (codec config, no media samples).
    Init,
    /// Regular media segment (contains one or more GOPs).
    Media,
    /// Partial segment as defined by LL-HLS.
    Partial,
}

impl SegmentKind {
    /// Returns `true` if this is an initialisation segment.
    #[must_use]
    pub fn is_init(self) -> bool {
        matches!(self, Self::Init)
    }

    /// Returns `true` if this segment carries presentable media data.
    #[must_use]
    pub fn has_samples(self) -> bool {
        matches!(self, Self::Media | Self::Partial)
    }
}

impl std::fmt::Display for SegmentKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Init => f.write_str("init"),
            Self::Media => f.write_str("media"),
            Self::Partial => f.write_str("partial"),
        }
    }
}

// ---------------------------------------------------------------------------
// ByteRange
// ---------------------------------------------------------------------------

/// An inclusive byte range `[first, last]` into a parent resource.
///
/// Follows the semantics of the HTTP `Range` header and HLS `EXT-X-BYTERANGE`
/// tags where both endpoints are inclusive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ByteRange {
    /// First byte offset (inclusive).
    pub first: u64,
    /// Last byte offset (inclusive).
    pub last: u64,
}

impl ByteRange {
    /// Creates a new [`ByteRange`].
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `first > last`.
    #[must_use]
    pub fn new(first: u64, last: u64) -> Self {
        debug_assert!(first <= last, "ByteRange: first must not exceed last");
        Self { first, last }
    }

    /// Tries to create a [`ByteRange`], returning an error if `first > last`.
    pub fn try_new(first: u64, last: u64) -> OxiResult<Self> {
        if first > last {
            return Err(OxiError::InvalidData(format!(
                "ByteRange: first ({first}) > last ({last})"
            )));
        }
        Ok(Self { first, last })
    }

    /// Returns the number of bytes in this range.
    #[must_use]
    pub fn length(self) -> u64 {
        self.last - self.first + 1
    }

    /// Returns `true` if the given offset falls within `[first, last]`.
    #[must_use]
    pub fn contains(self, offset: u64) -> bool {
        offset >= self.first && offset <= self.last
    }

    /// Returns `true` if this range and `other` share at least one byte.
    #[must_use]
    pub fn overlaps(self, other: Self) -> bool {
        self.first <= other.last && other.first <= self.last
    }

    /// Returns the intersection of `self` and `other`, or `None` if they do
    /// not overlap.
    #[must_use]
    pub fn intersection(self, other: Self) -> Option<Self> {
        let first = self.first.max(other.first);
        let last = self.last.min(other.last);
        if first <= last {
            Some(Self { first, last })
        } else {
            None
        }
    }
}

impl std::fmt::Display for ByteRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}-{}", self.first, self.last)
    }
}

// ---------------------------------------------------------------------------
// SegmentTimeRange
// ---------------------------------------------------------------------------

/// A half-open presentation time range `[start_ticks, end_ticks)` expressed in
/// a [`Rational`] time base.
///
/// The time base follows MPEG convention: a tick is `time_base.num /
/// time_base.den` seconds.  For 90 kHz MPEG timestamps this is `1/90000`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SegmentTimeRange {
    /// Start of the range in ticks (inclusive).
    pub start_ticks: i64,
    /// End of the range in ticks (exclusive).
    pub end_ticks: i64,
    /// Time base for tick-to-seconds conversion.
    pub time_base: Rational,
}

impl SegmentTimeRange {
    /// Creates a new `SegmentTimeRange`.
    ///
    /// Returns `OxiError::InvalidParameter` if `end_ticks < start_ticks`.
    pub fn new(start_ticks: i64, end_ticks: i64, time_base: Rational) -> OxiResult<Self> {
        if end_ticks < start_ticks {
            return Err(OxiError::InvalidData(format!(
                "SegmentTimeRange: end_ticks ({end_ticks}) < start_ticks ({start_ticks})"
            )));
        }
        Ok(Self {
            start_ticks,
            end_ticks,
            time_base,
        })
    }

    /// Creates a zero-duration range at `ticks`.
    #[must_use]
    pub fn point(ticks: i64, time_base: Rational) -> Self {
        Self {
            start_ticks: ticks,
            end_ticks: ticks,
            time_base,
        }
    }

    /// Duration of this range in seconds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn duration_secs(self) -> f64 {
        let ticks = (self.end_ticks - self.start_ticks) as f64;
        ticks * self.time_base.to_f64()
    }

    /// Duration in ticks.
    #[must_use]
    pub fn duration_ticks(self) -> i64 {
        self.end_ticks - self.start_ticks
    }

    /// Returns `true` if the given tick value falls within `[start, end)`.
    #[must_use]
    pub fn contains(self, ticks: i64) -> bool {
        ticks >= self.start_ticks && ticks < self.end_ticks
    }

    /// Returns `true` if `self` and `other` overlap (share at least one tick).
    #[must_use]
    pub fn overlaps(self, other: Self) -> bool {
        self.start_ticks < other.end_ticks && other.start_ticks < self.end_ticks
    }

    /// Returns the wall-clock start time in seconds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn start_secs(self) -> f64 {
        self.start_ticks as f64 * self.time_base.to_f64()
    }

    /// Returns the wall-clock end time in seconds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn end_secs(self) -> f64 {
        self.end_ticks as f64 * self.time_base.to_f64()
    }

    /// Rescales this time range to a different time base, rounding half-up.
    ///
    /// Uses 128-bit intermediate arithmetic to avoid overflow.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn rebase(self, new_base: Rational) -> Self {
        let rescale = |ticks: i64| -> i64 {
            let scale_num = i128::from(self.time_base.num) * i128::from(new_base.den);
            let scale_den = i128::from(self.time_base.den) * i128::from(new_base.num);
            let half = scale_den / 2;
            ((i128::from(ticks) * scale_num + half) / scale_den) as i64
        };
        Self {
            start_ticks: rescale(self.start_ticks),
            end_ticks: rescale(self.end_ticks),
            time_base: new_base,
        }
    }
}

impl std::fmt::Display for SegmentTimeRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:.6}s, {:.6}s)", self.start_secs(), self.end_secs())
    }
}

// ---------------------------------------------------------------------------
// SegmentMetadata
// ---------------------------------------------------------------------------

/// Optional per-segment descriptors useful for manifest generation and ABR.
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentMetadata {
    /// Monotonically increasing sequence number within the stream.
    pub sequence: u64,
    /// Instantaneous bitrate in bits per second, if known.
    pub bitrate_bps: Option<u64>,
    /// Whether this segment can be decoded independently (random access point).
    pub independent: bool,
    /// Optional human-readable label (e.g. `"video_1080p"`, `"audio_en"`).
    pub label: Option<String>,
}

impl SegmentMetadata {
    /// Creates a minimal `SegmentMetadata` with only a sequence number.
    #[must_use]
    pub fn with_sequence(sequence: u64) -> Self {
        Self {
            sequence,
            bitrate_bps: None,
            independent: false,
            label: None,
        }
    }
}

// ---------------------------------------------------------------------------
// MediaSegment
// ---------------------------------------------------------------------------

/// A complete descriptor for a single segment of a streaming media asset.
///
/// Combines the segment kind, time range, optional byte range into a parent
/// resource, and optional metadata. Use [`MediaSegmentBuilder`] (obtained via
/// [`MediaSegment::builder`]) to construct instances.
#[derive(Debug, Clone, PartialEq)]
pub struct MediaSegment {
    kind: SegmentKind,
    time_range: SegmentTimeRange,
    byte_range: Option<ByteRange>,
    metadata: Option<SegmentMetadata>,
}

impl MediaSegment {
    /// Returns a [`MediaSegmentBuilder`] for the given `kind` and `time_range`.
    #[must_use]
    pub fn builder(kind: SegmentKind, time_range: SegmentTimeRange) -> MediaSegmentBuilder {
        MediaSegmentBuilder {
            kind,
            time_range,
            byte_range: None,
            metadata: None,
        }
    }

    /// The segment kind.
    #[must_use]
    pub fn kind(&self) -> SegmentKind {
        self.kind
    }

    /// The presentation time range of this segment.
    #[must_use]
    pub fn time_range(&self) -> SegmentTimeRange {
        self.time_range
    }

    /// The byte range within the parent resource, if applicable.
    #[must_use]
    pub fn byte_range(&self) -> Option<ByteRange> {
        self.byte_range
    }

    /// Segment metadata, if present.
    #[must_use]
    pub fn metadata(&self) -> Option<&SegmentMetadata> {
        self.metadata.as_ref()
    }

    /// Convenience: returns the sequence number from metadata, if any.
    #[must_use]
    pub fn sequence(&self) -> Option<u64> {
        self.metadata.as_ref().map(|m| m.sequence)
    }

    /// Convenience: returns the instantaneous bitrate from metadata, if any.
    #[must_use]
    pub fn bitrate_bps(&self) -> Option<u64> {
        self.metadata.as_ref().and_then(|m| m.bitrate_bps)
    }
}

// ---------------------------------------------------------------------------
// MediaSegmentBuilder
// ---------------------------------------------------------------------------

/// Builder for [`MediaSegment`].
#[derive(Debug)]
pub struct MediaSegmentBuilder {
    kind: SegmentKind,
    time_range: SegmentTimeRange,
    byte_range: Option<ByteRange>,
    metadata: Option<SegmentMetadata>,
}

impl MediaSegmentBuilder {
    /// Sets the byte range within the parent resource.
    #[must_use]
    pub fn with_byte_range(mut self, range: ByteRange) -> Self {
        self.byte_range = Some(range);
        self
    }

    /// Attaches segment metadata.
    #[must_use]
    pub fn with_metadata(mut self, meta: SegmentMetadata) -> Self {
        self.metadata = Some(meta);
        self
    }

    /// Consumes the builder and produces a [`MediaSegment`].
    #[must_use]
    pub fn build(self) -> MediaSegment {
        MediaSegment {
            kind: self.kind,
            time_range: self.time_range,
            byte_range: self.byte_range,
            metadata: self.metadata,
        }
    }
}

// ---------------------------------------------------------------------------
// SegmentList
// ---------------------------------------------------------------------------

/// An ordered, append-only list of [`MediaSegment`]s, supporting time-indexed
/// lookup and sequence-based retrieval.
///
/// Segments are stored in insertion order; it is the caller's responsibility to
/// insert them in chronological order for correct time-indexed operations.
#[derive(Debug, Default, Clone)]
pub struct SegmentList {
    segments: Vec<MediaSegment>,
}

impl SegmentList {
    /// Creates an empty `SegmentList`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a segment to the list.
    pub fn push(&mut self, seg: MediaSegment) {
        self.segments.push(seg);
    }

    /// Returns the number of segments in the list.
    #[must_use]
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    /// Returns `true` if the list contains no segments.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// Returns the segment at position `index`, or `None` if out of bounds.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&MediaSegment> {
        self.segments.get(index)
    }

    /// Returns an iterator over all segments.
    pub fn iter(&self) -> impl Iterator<Item = &MediaSegment> {
        self.segments.iter()
    }

    /// Finds the first segment whose time range contains `ticks`.
    ///
    /// Assumes segments are stored in chronological order.
    #[must_use]
    pub fn at_tick(&self, ticks: i64) -> Option<&MediaSegment> {
        self.segments.iter().find(|s| s.time_range.contains(ticks))
    }

    /// Finds the segment with the given `sequence` number, or `None`.
    #[must_use]
    pub fn by_sequence(&self, sequence: u64) -> Option<&MediaSegment> {
        self.segments
            .iter()
            .find(|s| s.metadata().map_or(false, |m| m.sequence == sequence))
    }

    /// Returns a slice of all segments.
    #[must_use]
    pub fn as_slice(&self) -> &[MediaSegment] {
        &self.segments
    }

    /// Returns the total presentation duration of all segments in seconds.
    #[must_use]
    pub fn total_duration_secs(&self) -> f64 {
        self.segments
            .iter()
            .map(|s| s.time_range.duration_secs())
            .sum()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Rational;

    fn tb() -> Rational {
        Rational::new(1, 90_000)
    }

    fn make_range(start: i64, end: i64) -> SegmentTimeRange {
        SegmentTimeRange::new(start, end, tb()).expect("valid range")
    }

    // --- SegmentKind ---

    #[test]
    fn test_segment_kind_is_init() {
        assert!(SegmentKind::Init.is_init());
        assert!(!SegmentKind::Media.is_init());
        assert!(!SegmentKind::Partial.is_init());
    }

    #[test]
    fn test_segment_kind_has_samples() {
        assert!(!SegmentKind::Init.has_samples());
        assert!(SegmentKind::Media.has_samples());
        assert!(SegmentKind::Partial.has_samples());
    }

    #[test]
    fn test_segment_kind_display() {
        assert_eq!(format!("{}", SegmentKind::Init), "init");
        assert_eq!(format!("{}", SegmentKind::Media), "media");
        assert_eq!(format!("{}", SegmentKind::Partial), "partial");
    }

    // --- ByteRange ---

    #[test]
    fn test_byte_range_length() {
        let r = ByteRange::new(0, 999);
        assert_eq!(r.length(), 1000);
    }

    #[test]
    fn test_byte_range_contains() {
        let r = ByteRange::new(100, 200);
        assert!(r.contains(100));
        assert!(r.contains(150));
        assert!(r.contains(200));
        assert!(!r.contains(99));
        assert!(!r.contains(201));
    }

    #[test]
    fn test_byte_range_overlaps() {
        let r1 = ByteRange::new(0, 100);
        let r2 = ByteRange::new(50, 150);
        let r3 = ByteRange::new(200, 300);
        assert!(r1.overlaps(r2));
        assert!(!r1.overlaps(r3));
    }

    #[test]
    fn test_byte_range_intersection() {
        let r1 = ByteRange::new(0, 100);
        let r2 = ByteRange::new(50, 150);
        let inter = r1.intersection(r2).expect("should intersect");
        assert_eq!(inter.first, 50);
        assert_eq!(inter.last, 100);

        let r3 = ByteRange::new(200, 300);
        assert!(r1.intersection(r3).is_none());
    }

    #[test]
    fn test_byte_range_try_new_error() {
        assert!(ByteRange::try_new(100, 50).is_err());
        assert!(ByteRange::try_new(50, 100).is_ok());
    }

    // --- SegmentTimeRange ---

    #[test]
    fn test_time_range_duration_secs() {
        let r = make_range(0, 270_000);
        let dur = r.duration_secs();
        assert!((dur - 3.0).abs() < 1e-6, "expected 3s, got {dur}");
    }

    #[test]
    fn test_time_range_contains() {
        let r = make_range(0, 90_000);
        assert!(r.contains(0));
        assert!(r.contains(45_000));
        assert!(!r.contains(90_000)); // exclusive end
        assert!(!r.contains(-1));
    }

    #[test]
    fn test_time_range_overlaps() {
        let r1 = make_range(0, 90_000);
        let r2 = make_range(45_000, 135_000);
        let r3 = make_range(90_000, 180_000);
        assert!(r1.overlaps(r2));
        assert!(!r1.overlaps(r3)); // r1 ends where r3 starts (half-open)
    }

    #[test]
    fn test_time_range_rebase() {
        let r = make_range(90_000, 270_000);
        let new_base = Rational::new(1, 1000);
        let rebased = r.rebase(new_base);
        assert_eq!(rebased.start_ticks, 1000);
        assert_eq!(rebased.end_ticks, 3000);
    }

    #[test]
    fn test_time_range_invalid() {
        let res = SegmentTimeRange::new(100, 50, tb());
        assert!(res.is_err());
    }

    // --- MediaSegment builder ---

    #[test]
    fn test_media_segment_builder() {
        let r = make_range(0, 90_000);
        let seg = MediaSegment::builder(SegmentKind::Media, r)
            .with_byte_range(ByteRange::new(0, 4999))
            .with_metadata(SegmentMetadata {
                sequence: 7,
                bitrate_bps: Some(4_000_000),
                independent: true,
                label: Some("video".to_string()),
            })
            .build();

        assert_eq!(seg.kind(), SegmentKind::Media);
        assert_eq!(seg.sequence(), Some(7));
        assert_eq!(seg.bitrate_bps(), Some(4_000_000));
        assert!(seg.byte_range().is_some());
    }

    #[test]
    fn test_media_segment_no_metadata() {
        let seg = MediaSegment::builder(SegmentKind::Init, make_range(0, 0)).build();
        assert!(seg.metadata().is_none());
        assert!(seg.sequence().is_none());
        assert!(seg.bitrate_bps().is_none());
    }

    // --- SegmentList ---

    #[test]
    fn test_segment_list_push_and_len() {
        let mut list = SegmentList::new();
        assert!(list.is_empty());
        list.push(MediaSegment::builder(SegmentKind::Media, make_range(0, 90_000)).build());
        list.push(MediaSegment::builder(SegmentKind::Media, make_range(90_000, 180_000)).build());
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_segment_list_at_tick() {
        let mut list = SegmentList::new();
        list.push(
            MediaSegment::builder(SegmentKind::Media, make_range(0, 90_000))
                .with_metadata(SegmentMetadata::with_sequence(1))
                .build(),
        );
        list.push(
            MediaSegment::builder(SegmentKind::Media, make_range(90_000, 180_000))
                .with_metadata(SegmentMetadata::with_sequence(2))
                .build(),
        );

        let found = list.at_tick(45_000).expect("should find segment");
        assert_eq!(found.sequence(), Some(1));

        let found2 = list.at_tick(120_000).expect("should find segment");
        assert_eq!(found2.sequence(), Some(2));

        assert!(list.at_tick(300_000).is_none());
    }

    #[test]
    fn test_segment_list_by_sequence() {
        let mut list = SegmentList::new();
        for seq in 1u64..=3 {
            list.push(
                MediaSegment::builder(
                    SegmentKind::Media,
                    make_range((seq as i64 - 1) * 90_000, seq as i64 * 90_000),
                )
                .with_metadata(SegmentMetadata::with_sequence(seq))
                .build(),
            );
        }
        let seg = list.by_sequence(2).expect("sequence 2 should exist");
        assert_eq!(seg.sequence(), Some(2));
        assert!(list.by_sequence(99).is_none());
    }

    #[test]
    fn test_segment_list_total_duration() {
        let mut list = SegmentList::new();
        list.push(MediaSegment::builder(SegmentKind::Media, make_range(0, 90_000)).build());
        list.push(MediaSegment::builder(SegmentKind::Media, make_range(90_000, 180_000)).build());
        let dur = list.total_duration_secs();
        assert!((dur - 2.0).abs() < 1e-6, "expected 2.0s, got {dur}");
    }
}
