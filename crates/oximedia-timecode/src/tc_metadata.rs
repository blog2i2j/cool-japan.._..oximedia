#![allow(dead_code)]
//! Timecode metadata for embedding and extracting timecode-related info
//! alongside media streams.
//!
//! Provides structures for tagging media with timecode origins, recording dates,
//! reel identifiers, and user-bits payloads conforming to SMPTE 12M.

use crate::{FrameRate, Timecode, TimecodeError};
use std::collections::HashMap;

/// Source type that originally generated the timecode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimecodeSource {
    /// Linear Timecode from audio track
    Ltc,
    /// Vertical Interval Timecode
    Vitc,
    /// MIDI Time Code
    Mtc,
    /// Network Time Protocol derived
    Ntp,
    /// Precision Time Protocol derived
    Ptp,
    /// Manually entered / free-run generator
    FreeRun,
    /// Timecode reconstructed from file metadata
    FileMetadata,
}

/// Recording date associated with a timecode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordDate {
    /// Year (e.g. 2026)
    pub year: u16,
    /// Month (1-12)
    pub month: u8,
    /// Day (1-31)
    pub day: u8,
}

impl RecordDate {
    /// Creates a new recording date.
    ///
    /// # Errors
    ///
    /// Returns `TimecodeError::InvalidConfiguration` for out-of-range values.
    pub fn new(year: u16, month: u8, day: u8) -> Result<Self, TimecodeError> {
        if month == 0 || month > 12 {
            return Err(TimecodeError::InvalidConfiguration);
        }
        if day == 0 || day > 31 {
            return Err(TimecodeError::InvalidConfiguration);
        }
        Ok(Self { year, month, day })
    }

    /// Formats the date as ISO 8601 (YYYY-MM-DD).
    pub fn to_iso_string(&self) -> String {
        format!("{:04}-{:02}-{:02}", self.year, self.month, self.day)
    }
}

/// User bits payload from SMPTE 12M (32 bits split into 8 nibbles).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UserBitsPayload {
    /// Raw 32-bit user bits value
    pub raw: u32,
    /// Whether the user bits encode a date (BG flag)
    pub is_date: bool,
}

impl UserBitsPayload {
    /// Creates a new user bits payload from a raw value.
    pub fn new(raw: u32, is_date: bool) -> Self {
        Self { raw, is_date }
    }

    /// Extracts a nibble (0-7) from the user bits.
    pub fn nibble(&self, index: u8) -> u8 {
        if index > 7 {
            return 0;
        }
        ((self.raw >> (index * 4)) & 0x0F) as u8
    }

    /// Sets a nibble (0-7) in the user bits.
    pub fn set_nibble(&mut self, index: u8, value: u8) {
        if index > 7 {
            return;
        }
        let shift = index * 4;
        self.raw &= !(0x0F << shift);
        self.raw |= ((value & 0x0F) as u32) << shift;
    }

    /// Decodes the user bits as a BCD date (if applicable).
    ///
    /// SMPTE 12M encodes dates as: nibbles 0-1 = day, 2-3 = month, 4-7 = year.
    pub fn decode_date(&self) -> Option<RecordDate> {
        if !self.is_date {
            return None;
        }
        let day = self.nibble(0) * 10 + self.nibble(1);
        let month = self.nibble(2) * 10 + self.nibble(3);
        let year_hi = self.nibble(4) as u16 * 10 + self.nibble(5) as u16;
        let year_lo = self.nibble(6) as u16 * 10 + self.nibble(7) as u16;
        let year = year_hi * 100 + year_lo;
        RecordDate::new(year, month, day).ok()
    }

    /// Encodes a date into user bits in BCD format.
    pub fn encode_date(date: &RecordDate) -> Self {
        let mut payload = Self::new(0, true);
        let day_hi = date.day / 10;
        let day_lo = date.day % 10;
        let month_hi = date.month / 10;
        let month_lo = date.month % 10;
        let year_hi_hi = (date.year / 1000) as u8;
        let year_hi_lo = ((date.year / 100) % 10) as u8;
        let year_lo_hi = ((date.year / 10) % 10) as u8;
        let year_lo_lo = (date.year % 10) as u8;
        payload.set_nibble(0, day_hi);
        payload.set_nibble(1, day_lo);
        payload.set_nibble(2, month_hi);
        payload.set_nibble(3, month_lo);
        payload.set_nibble(4, year_hi_hi);
        payload.set_nibble(5, year_hi_lo);
        payload.set_nibble(6, year_lo_hi);
        payload.set_nibble(7, year_lo_lo);
        payload
    }
}

/// Reel identifier associated with a timecode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReelId {
    /// Reel name or number
    pub name: String,
    /// Optional sequence index within the reel
    pub sequence: Option<u32>,
}

impl ReelId {
    /// Creates a new reel identifier.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            sequence: None,
        }
    }

    /// Sets the sequence number.
    pub fn with_sequence(mut self, seq: u32) -> Self {
        self.sequence = Some(seq);
        self
    }
}

/// Comprehensive timecode metadata block.
///
/// Bundles a timecode with all associated metadata such as source, reel, date,
/// user bits, and custom key-value tags.
#[derive(Debug, Clone)]
pub struct TcMetadata {
    /// The timecode value
    pub timecode: Timecode,
    /// Frame rate used for the timecode
    pub frame_rate: FrameRate,
    /// Source of the timecode
    pub source: TimecodeSource,
    /// Optional reel identifier
    pub reel: Option<ReelId>,
    /// Optional recording date
    pub record_date: Option<RecordDate>,
    /// User bits payload
    pub user_bits: Option<UserBitsPayload>,
    /// Arbitrary string key-value tags
    pub tags: HashMap<String, String>,
    /// Scene label
    pub scene: Option<String>,
    /// Take number
    pub take: Option<u32>,
}

impl TcMetadata {
    /// Creates new metadata for a timecode.
    pub fn new(timecode: Timecode, frame_rate: FrameRate, source: TimecodeSource) -> Self {
        Self {
            timecode,
            frame_rate,
            source,
            reel: None,
            record_date: None,
            user_bits: None,
            tags: HashMap::new(),
            scene: None,
            take: None,
        }
    }

    /// Sets the reel identifier.
    pub fn with_reel(mut self, reel: ReelId) -> Self {
        self.reel = Some(reel);
        self
    }

    /// Sets the recording date.
    pub fn with_record_date(mut self, date: RecordDate) -> Self {
        self.record_date = Some(date);
        self
    }

    /// Sets the user bits.
    pub fn with_user_bits(mut self, ub: UserBitsPayload) -> Self {
        self.user_bits = Some(ub);
        self
    }

    /// Adds a custom tag.
    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.insert(key.into(), value.into());
        self
    }

    /// Sets the scene label.
    pub fn with_scene(mut self, scene: impl Into<String>) -> Self {
        self.scene = Some(scene.into());
        self
    }

    /// Sets the take number.
    pub fn with_take(mut self, take: u32) -> Self {
        self.take = Some(take);
        self
    }

    /// Formats metadata as a human-readable summary string.
    pub fn summary(&self) -> String {
        let mut parts = vec![format!("TC={}", self.timecode)];
        parts.push(format!("src={:?}", self.source));
        if let Some(ref reel) = self.reel {
            parts.push(format!("reel={}", reel.name));
        }
        if let Some(ref date) = self.record_date {
            parts.push(format!("date={}", date.to_iso_string()));
        }
        if let Some(ref scene) = self.scene {
            parts.push(format!("scene={scene}"));
        }
        if let Some(take) = self.take {
            parts.push(format!("take={take}"));
        }
        parts.join(" | ")
    }

    /// Validates that the metadata is internally consistent.
    ///
    /// # Errors
    ///
    /// Returns an error if the timecode frame rate info does not match the declared frame rate.
    pub fn validate(&self) -> Result<(), TimecodeError> {
        let expected_fps = self.frame_rate.frames_per_second() as u8;
        if self.timecode.frame_rate.fps != expected_fps {
            return Err(TimecodeError::InvalidConfiguration);
        }
        if self.timecode.frame_rate.drop_frame != self.frame_rate.is_drop_frame() {
            return Err(TimecodeError::InvalidConfiguration);
        }
        Ok(())
    }
}

/// A timeline of metadata entries keyed by frame number.
#[derive(Debug, Clone)]
pub struct MetadataTimeline {
    /// Entries sorted by frame number
    entries: Vec<(u64, TcMetadata)>,
}

impl MetadataTimeline {
    /// Creates an empty metadata timeline.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Adds a metadata entry at the given frame.
    pub fn insert(&mut self, frame: u64, meta: TcMetadata) {
        let pos = self.entries.partition_point(|(f, _)| *f < frame);
        self.entries.insert(pos, (frame, meta));
    }

    /// Finds the metadata entry at or before the given frame.
    pub fn lookup(&self, frame: u64) -> Option<&TcMetadata> {
        let pos = self.entries.partition_point(|(f, _)| *f <= frame);
        if pos == 0 {
            return None;
        }
        Some(&self.entries[pos - 1].1)
    }

    /// Returns the number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns whether the timeline is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns all entries as a slice.
    pub fn entries(&self) -> &[(u64, TcMetadata)] {
        &self.entries
    }
}

impl Default for MetadataTimeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tc() -> Timecode {
        Timecode::new(1, 2, 3, 4, FrameRate::Fps25).expect("valid timecode")
    }

    #[test]
    fn test_record_date_valid() {
        let d = RecordDate::new(2026, 3, 2).expect("valid record date");
        assert_eq!(d.to_iso_string(), "2026-03-02");
    }

    #[test]
    fn test_record_date_invalid_month() {
        assert!(RecordDate::new(2026, 13, 1).is_err());
    }

    #[test]
    fn test_record_date_invalid_day() {
        assert!(RecordDate::new(2026, 1, 0).is_err());
    }

    #[test]
    fn test_user_bits_nibble() {
        let mut ub = UserBitsPayload::new(0, false);
        ub.set_nibble(0, 0x0A);
        assert_eq!(ub.nibble(0), 0x0A);
        assert_eq!(ub.nibble(1), 0);
    }

    #[test]
    fn test_user_bits_date_encode_decode() {
        let date = RecordDate::new(2026, 3, 15).expect("valid record date");
        let ub = UserBitsPayload::encode_date(&date);
        let decoded = ub.decode_date().expect("decode should succeed");
        assert_eq!(decoded.year, 2026);
        assert_eq!(decoded.month, 3);
        assert_eq!(decoded.day, 15);
    }

    #[test]
    fn test_user_bits_no_date() {
        let ub = UserBitsPayload::new(0x12345678, false);
        assert!(ub.decode_date().is_none());
    }

    #[test]
    fn test_reel_id() {
        let reel = ReelId::new("A001").with_sequence(1);
        assert_eq!(reel.name, "A001");
        assert_eq!(reel.sequence, Some(1));
    }

    #[test]
    fn test_tc_metadata_new() {
        let tc = make_tc();
        let meta = TcMetadata::new(tc, FrameRate::Fps25, TimecodeSource::Ltc);
        assert_eq!(meta.source, TimecodeSource::Ltc);
        assert!(meta.reel.is_none());
    }

    #[test]
    fn test_tc_metadata_with_builders() {
        let tc = make_tc();
        let meta = TcMetadata::new(tc, FrameRate::Fps25, TimecodeSource::Vitc)
            .with_reel(ReelId::new("B002"))
            .with_scene("42A")
            .with_take(3)
            .with_tag("camera", "A");
        assert_eq!(meta.scene.as_deref(), Some("42A"));
        assert_eq!(meta.take, Some(3));
        assert_eq!(meta.tags.get("camera").expect("key should exist"), "A");
    }

    #[test]
    fn test_tc_metadata_summary() {
        let tc = make_tc();
        let meta = TcMetadata::new(tc, FrameRate::Fps25, TimecodeSource::Ltc).with_scene("1A");
        let s = meta.summary();
        assert!(s.contains("TC=01:02:03:04"));
        assert!(s.contains("scene=1A"));
    }

    #[test]
    fn test_tc_metadata_validate_ok() {
        let tc = make_tc();
        let meta = TcMetadata::new(tc, FrameRate::Fps25, TimecodeSource::Ltc);
        assert!(meta.validate().is_ok());
    }

    #[test]
    fn test_tc_metadata_validate_mismatch() {
        let tc = make_tc();
        let meta = TcMetadata::new(tc, FrameRate::Fps30, TimecodeSource::Ltc);
        assert!(meta.validate().is_err());
    }

    #[test]
    fn test_metadata_timeline_insert_and_lookup() {
        let tc = make_tc();
        let meta = TcMetadata::new(tc, FrameRate::Fps25, TimecodeSource::FreeRun);
        let mut tl = MetadataTimeline::new();
        tl.insert(100, meta.clone());
        tl.insert(200, meta);
        assert_eq!(tl.len(), 2);
        let found = tl.lookup(150).expect("lookup should succeed");
        assert_eq!(found.timecode.hours, 1);
    }

    #[test]
    fn test_metadata_timeline_empty_lookup() {
        let tl = MetadataTimeline::new();
        assert!(tl.lookup(0).is_none());
        assert!(tl.is_empty());
    }

    #[test]
    fn test_timecode_source_variants() {
        let sources = [
            TimecodeSource::Ltc,
            TimecodeSource::Vitc,
            TimecodeSource::Mtc,
            TimecodeSource::Ntp,
            TimecodeSource::Ptp,
            TimecodeSource::FreeRun,
            TimecodeSource::FileMetadata,
        ];
        assert_eq!(sources.len(), 7);
    }
}
