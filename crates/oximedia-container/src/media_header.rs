//! ISOBMFF `mdhd` (Media Header Box) parsing and writing.
//!
//! The media header box defines the timescale and duration of a track's media,
//! along with a BCP-47 / ISO 639-2/T language code and creation/modification times.

#![allow(dead_code)]

/// Seconds between the Mac/QuickTime epoch (1904-01-01) and the Unix epoch (1970-01-01).
const MAC_EPOCH_OFFSET_SECS: u64 = 2_082_844_800;

/// A parsed `mdhd` (Media Header) box.
#[derive(Debug, Clone)]
pub struct MediaHeaderBox {
    /// Box version: 0 = 32-bit timestamps, 1 = 64-bit timestamps.
    pub version: u8,
    /// Creation time in seconds since 1904-01-01 00:00:00 UTC.
    pub creation_time: u64,
    /// Modification time in seconds since 1904-01-01 00:00:00 UTC.
    pub modification_time: u64,
    /// Number of time units per second for this media.
    pub timescale: u32,
    /// Duration of the media in `timescale` units.
    pub duration: u64,
    /// ISO 639-2/T language code packed as 3 × 5-bit values (BCP-47 style).
    pub language_packed: u16,
    /// Pre-defined field (always 0 in conforming files).
    pub pre_defined: u16,
}

impl MediaHeaderBox {
    /// Decodes the packed language field into a 3-character ISO 639-2/T string.
    ///
    /// Each character is encoded as `(value + 0x60)` in 5-bit fields.
    #[must_use]
    pub fn language_code(&self) -> [u8; 3] {
        let p = self.language_packed;
        let c1 = (((p >> 10) & 0x1F) as u8).wrapping_add(0x60);
        let c2 = (((p >> 5) & 0x1F) as u8).wrapping_add(0x60);
        let c3 = ((p & 0x1F) as u8).wrapping_add(0x60);
        [c1, c2, c3]
    }

    /// Returns the creation time as milliseconds since the Unix epoch.
    ///
    /// Returns `None` if the Mac-epoch timestamp predates the Unix epoch.
    #[must_use]
    pub fn creation_time_ms(&self) -> Option<u64> {
        self.creation_time
            .checked_sub(MAC_EPOCH_OFFSET_SECS)
            .map(|s| s * 1000)
    }

    /// Returns the modification time as milliseconds since the Unix epoch.
    #[must_use]
    pub fn modification_time_ms(&self) -> Option<u64> {
        self.modification_time
            .checked_sub(MAC_EPOCH_OFFSET_SECS)
            .map(|s| s * 1000)
    }

    /// Returns the media duration in whole seconds (truncated).
    #[must_use]
    pub fn duration_secs(&self) -> u64 {
        if self.timescale == 0 {
            return 0;
        }
        self.duration / u64::from(self.timescale)
    }

    /// Returns the media duration in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        if self.timescale == 0 {
            return 0;
        }
        self.duration * 1000 / u64::from(self.timescale)
    }
}

/// Parses a raw `mdhd` box payload (after the 8-byte box header) into a
/// [`MediaHeaderBox`].
#[derive(Debug, Default)]
pub struct MediaHeaderParser;

impl MediaHeaderParser {
    /// Creates a new parser.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Parses `data` (the payload bytes after the box header) into a
    /// [`MediaHeaderBox`].
    ///
    /// Returns `None` if `data` is too short for the declared version.
    #[must_use]
    pub fn parse(&self, data: &[u8]) -> Option<MediaHeaderBox> {
        if data.is_empty() {
            return None;
        }
        let version = data[0];
        // Skip flags (3 bytes) after version.
        match version {
            0 => {
                // v0: 4+4+4+4+2+2 = 20 bytes after version+flags
                if data.len() < 24 {
                    return None;
                }
                let creation_time =
                    u64::from(u32::from_be_bytes([data[4], data[5], data[6], data[7]]));
                let modification_time =
                    u64::from(u32::from_be_bytes([data[8], data[9], data[10], data[11]]));
                let timescale = u32::from_be_bytes([data[12], data[13], data[14], data[15]]);
                let duration =
                    u64::from(u32::from_be_bytes([data[16], data[17], data[18], data[19]]));
                let language_packed = u16::from_be_bytes([data[20], data[21]]);
                let pre_defined = u16::from_be_bytes([data[22], data[23]]);
                Some(MediaHeaderBox {
                    version,
                    creation_time,
                    modification_time,
                    timescale,
                    duration,
                    language_packed,
                    pre_defined,
                })
            }
            1 => {
                // v1: 8+8+4+8+2+2 = 32 bytes after version+flags
                if data.len() < 36 {
                    return None;
                }
                let creation_time = u64::from_be_bytes([
                    data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11],
                ]);
                let modification_time = u64::from_be_bytes([
                    data[12], data[13], data[14], data[15], data[16], data[17], data[18], data[19],
                ]);
                let timescale = u32::from_be_bytes([data[20], data[21], data[22], data[23]]);
                let duration = u64::from_be_bytes([
                    data[24], data[25], data[26], data[27], data[28], data[29], data[30], data[31],
                ]);
                let language_packed = u16::from_be_bytes([data[32], data[33]]);
                let pre_defined = u16::from_be_bytes([data[34], data[35]]);
                Some(MediaHeaderBox {
                    version,
                    creation_time,
                    modification_time,
                    timescale,
                    duration,
                    language_packed,
                    pre_defined,
                })
            }
            _ => None,
        }
    }
}

/// Writes a [`MediaHeaderBox`] into a byte vector (version 0 format).
#[derive(Debug, Default)]
pub struct MediaHeaderWriter;

impl MediaHeaderWriter {
    /// Creates a new writer.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Serialises `mdhd` to a byte vector (version 0, 32-bit timestamps).
    ///
    /// Timestamps are clamped to `u32::MAX` if they overflow 32 bits.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn write(&self, mdhd: &MediaHeaderBox) -> Vec<u8> {
        let mut out = Vec::with_capacity(32);
        // version + flags (3 bytes)
        out.push(0u8);
        out.extend_from_slice(&[0u8; 3]);
        // creation_time
        out.extend_from_slice(&(mdhd.creation_time.min(u64::from(u32::MAX)) as u32).to_be_bytes());
        // modification_time
        out.extend_from_slice(
            &(mdhd.modification_time.min(u64::from(u32::MAX)) as u32).to_be_bytes(),
        );
        // timescale
        out.extend_from_slice(&mdhd.timescale.to_be_bytes());
        // duration
        out.extend_from_slice(&(mdhd.duration.min(u64::from(u32::MAX)) as u32).to_be_bytes());
        // language_packed + pre_defined
        out.extend_from_slice(&mdhd.language_packed.to_be_bytes());
        out.extend_from_slice(&mdhd.pre_defined.to_be_bytes());
        out
    }
}

/// Encodes a 3-character ASCII language string into the packed 15-bit field.
///
/// Each character must be a lower-case ASCII letter (a–z).
#[must_use]
pub fn encode_language(lang: &[u8; 3]) -> u16 {
    let c1 = u16::from(lang[0].saturating_sub(0x60) & 0x1F);
    let c2 = u16::from(lang[1].saturating_sub(0x60) & 0x1F);
    let c3 = u16::from(lang[2].saturating_sub(0x60) & 0x1F);
    (c1 << 10) | (c2 << 5) | c3
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_v0_payload(
        creation: u32,
        modification: u32,
        timescale: u32,
        duration: u32,
        lang: u16,
    ) -> Vec<u8> {
        let mut v = vec![0u8; 24]; // version=0, flags=0
        v[4..8].copy_from_slice(&creation.to_be_bytes());
        v[8..12].copy_from_slice(&modification.to_be_bytes());
        v[12..16].copy_from_slice(&timescale.to_be_bytes());
        v[16..20].copy_from_slice(&duration.to_be_bytes());
        v[20..22].copy_from_slice(&lang.to_be_bytes());
        v
    }

    #[test]
    fn test_parse_v0_basic() {
        let lang = encode_language(b"eng");
        let payload = make_v0_payload(
            MAC_EPOCH_OFFSET_SECS as u32,
            MAC_EPOCH_OFFSET_SECS as u32,
            90000,
            270000,
            lang,
        );
        let parser = MediaHeaderParser::new();
        let mdhd = parser.parse(&payload).expect("operation should succeed");
        assert_eq!(mdhd.version, 0);
        assert_eq!(mdhd.timescale, 90000);
        assert_eq!(mdhd.duration, 270000);
    }

    #[test]
    fn test_language_code_english() {
        let lang = encode_language(b"eng");
        let mdhd = MediaHeaderBox {
            version: 0,
            creation_time: 0,
            modification_time: 0,
            timescale: 90000,
            duration: 0,
            language_packed: lang,
            pre_defined: 0,
        };
        assert_eq!(mdhd.language_code(), *b"eng");
    }

    #[test]
    fn test_language_code_und() {
        let lang = encode_language(b"und");
        let mdhd = MediaHeaderBox {
            version: 0,
            creation_time: 0,
            modification_time: 0,
            timescale: 1,
            duration: 0,
            language_packed: lang,
            pre_defined: 0,
        };
        assert_eq!(mdhd.language_code(), *b"und");
    }

    #[test]
    fn test_creation_time_ms_at_unix_epoch() {
        let mdhd = MediaHeaderBox {
            version: 0,
            creation_time: MAC_EPOCH_OFFSET_SECS,
            modification_time: 0,
            timescale: 1,
            duration: 0,
            language_packed: 0,
            pre_defined: 0,
        };
        assert_eq!(mdhd.creation_time_ms(), Some(0));
    }

    #[test]
    fn test_creation_time_ms_before_unix_epoch_returns_none() {
        let mdhd = MediaHeaderBox {
            version: 0,
            creation_time: 0,
            modification_time: 0,
            timescale: 1,
            duration: 0,
            language_packed: 0,
            pre_defined: 0,
        };
        assert!(mdhd.creation_time_ms().is_none());
    }

    #[test]
    fn test_duration_secs() {
        let mdhd = MediaHeaderBox {
            version: 0,
            creation_time: 0,
            modification_time: 0,
            timescale: 90000,
            duration: 90000 * 30,
            language_packed: 0,
            pre_defined: 0,
        };
        assert_eq!(mdhd.duration_secs(), 30);
    }

    #[test]
    fn test_duration_ms() {
        let mdhd = MediaHeaderBox {
            version: 0,
            creation_time: 0,
            modification_time: 0,
            timescale: 1000,
            duration: 5000,
            language_packed: 0,
            pre_defined: 0,
        };
        assert_eq!(mdhd.duration_ms(), 5000);
    }

    #[test]
    fn test_duration_secs_zero_timescale() {
        let mdhd = MediaHeaderBox {
            version: 0,
            creation_time: 0,
            modification_time: 0,
            timescale: 0,
            duration: 1000,
            language_packed: 0,
            pre_defined: 0,
        };
        assert_eq!(mdhd.duration_secs(), 0);
    }

    #[test]
    fn test_parse_empty_returns_none() {
        let parser = MediaHeaderParser::new();
        assert!(parser.parse(&[]).is_none());
    }

    #[test]
    fn test_parse_too_short_v0_returns_none() {
        let parser = MediaHeaderParser::new();
        assert!(parser.parse(&[0u8; 10]).is_none());
    }

    #[test]
    fn test_parse_unknown_version_returns_none() {
        let parser = MediaHeaderParser::new();
        let mut data = vec![2u8; 36];
        data[0] = 2;
        assert!(parser.parse(&data).is_none());
    }

    #[test]
    fn test_writer_roundtrip() {
        let lang = encode_language(b"fra");
        let original = MediaHeaderBox {
            version: 0,
            creation_time: MAC_EPOCH_OFFSET_SECS + 1000,
            modification_time: MAC_EPOCH_OFFSET_SECS + 2000,
            timescale: 44100,
            duration: 44100 * 60,
            language_packed: lang,
            pre_defined: 0,
        };
        let writer = MediaHeaderWriter::new();
        let bytes = writer.write(&original);
        let parser = MediaHeaderParser::new();
        let parsed = parser.parse(&bytes).expect("operation should succeed");
        assert_eq!(parsed.timescale, original.timescale);
        assert_eq!(parsed.duration, original.duration);
        assert_eq!(parsed.language_code(), *b"fra");
    }

    #[test]
    fn test_encode_language_roundtrip() {
        let lang = encode_language(b"jpn");
        let mdhd = MediaHeaderBox {
            version: 0,
            creation_time: 0,
            modification_time: 0,
            timescale: 1,
            duration: 0,
            language_packed: lang,
            pre_defined: 0,
        };
        assert_eq!(mdhd.language_code(), *b"jpn");
    }

    #[test]
    fn test_modification_time_ms() {
        let mdhd = MediaHeaderBox {
            version: 0,
            creation_time: 0,
            modification_time: MAC_EPOCH_OFFSET_SECS + 500,
            timescale: 1,
            duration: 0,
            language_packed: 0,
            pre_defined: 0,
        };
        assert_eq!(mdhd.modification_time_ms(), Some(500_000));
    }
}
