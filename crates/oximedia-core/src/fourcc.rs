//! `FourCC` (Four Character Code) types and registry for codec/format identification.
//!
//! `FourCC` codes are four-byte identifiers used in multimedia containers and
//! codecs to identify stream types, pixel formats, and compression schemes.

#![allow(dead_code)]

use std::collections::HashMap;
use std::fmt;

/// A four-character code (`FourCC`) identifier.
///
/// Used to identify codec types, pixel formats, and container formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FourCC([u8; 4]);

impl FourCC {
    /// Creates a `FourCC` from a 4-byte array.
    ///
    /// # Example
    /// ```
    /// use oximedia_core::fourcc::FourCC;
    /// let fcc = FourCC::from_bytes(*b"AV01");
    /// assert_eq!(fcc.as_bytes(), b"AV01");
    /// ```
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 4]) -> Self {
        Self(bytes)
    }

    /// Creates a `FourCC` from a string slice.
    ///
    /// Returns `None` if the string is not exactly 4 bytes (ASCII).
    ///
    /// # Example
    /// ```
    /// use oximedia_core::fourcc::FourCC;
    /// let fcc = FourCC::parse("VP90")?;
    /// assert!(fcc.is_video());
    /// ```
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        let b = s.as_bytes();
        if b.len() != 4 {
            return None;
        }
        Some(Self([b[0], b[1], b[2], b[3]]))
    }

    /// Returns the raw bytes of this `FourCC`.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 4] {
        &self.0
    }

    /// Returns `true` if this `FourCC` identifies a known video codec.
    ///
    /// Recognises patent-free codecs: AV1, VP9, VP8, Theora.
    #[must_use]
    pub fn is_video(&self) -> bool {
        matches!(&self.0, b"AV01" | b"VP90" | b"VP80" | b"theo")
    }

    /// Returns `true` if this `FourCC` identifies a known audio codec.
    ///
    /// Recognises patent-free codecs: Opus, Vorbis, FLAC, PCM variants.
    #[must_use]
    pub fn is_audio(&self) -> bool {
        matches!(&self.0, b"Opus" | b"Vorb" | b"fLaC" | b"araw" | b"sowt")
    }

    /// Returns the numeric representation of this `FourCC` as a `u32`.
    #[must_use]
    pub fn as_u32(&self) -> u32 {
        u32::from_be_bytes(self.0)
    }
}

impl fmt::Display for FourCC {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for &b in &self.0 {
            if b.is_ascii_graphic() || b == b' ' {
                write!(f, "{}", b as char)?;
            } else {
                write!(f, ".")?;
            }
        }
        Ok(())
    }
}

impl From<[u8; 4]> for FourCC {
    fn from(bytes: [u8; 4]) -> Self {
        Self::from_bytes(bytes)
    }
}

impl TryFrom<&str> for FourCC {
    type Error = &'static str;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        Self::parse(s).ok_or("FourCC string must be exactly 4 ASCII bytes")
    }
}

/// A registry mapping `FourCC` codes to human-readable names.
///
/// Provides lookup and registration of `FourCC` codes used within a pipeline.
#[derive(Debug, Default)]
pub struct FourCCRegistry {
    entries: HashMap<FourCC, String>,
}

impl FourCCRegistry {
    /// Creates a new, empty `FourCCRegistry`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Creates a registry pre-populated with well-known patent-free codecs.
    #[must_use]
    pub fn with_defaults() -> Self {
        let mut reg = Self::new();
        for (code, name) in Self::known_video_codecs() {
            reg.register(code, name);
        }
        for (code, name) in Self::known_audio_codecs() {
            reg.register(code, name);
        }
        reg
    }

    /// Registers a `FourCC` with an associated name.
    pub fn register(&mut self, code: FourCC, name: impl Into<String>) {
        self.entries.insert(code, name.into());
    }

    /// Looks up the name for a `FourCC`, returning `None` if not registered.
    #[must_use]
    pub fn lookup(&self, code: &FourCC) -> Option<&str> {
        self.entries.get(code).map(String::as_str)
    }

    /// Returns the total number of registered `FourCC` entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the registry contains no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns well-known patent-free video codec FourCC/name pairs.
    #[must_use]
    pub fn known_video_codecs() -> Vec<(FourCC, &'static str)> {
        vec![
            (FourCC::from_bytes(*b"AV01"), "AV1"),
            (FourCC::from_bytes(*b"VP90"), "VP9"),
            (FourCC::from_bytes(*b"VP80"), "VP8"),
            (FourCC::from_bytes(*b"theo"), "Theora"),
        ]
    }

    /// Returns well-known patent-free audio codec FourCC/name pairs.
    #[must_use]
    pub fn known_audio_codecs() -> Vec<(FourCC, &'static str)> {
        vec![
            (FourCC::from_bytes(*b"Opus"), "Opus"),
            (FourCC::from_bytes(*b"Vorb"), "Vorbis"),
            (FourCC::from_bytes(*b"fLaC"), "FLAC"),
            (FourCC::from_bytes(*b"araw"), "PCM"),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_bytes_roundtrip() {
        let fcc = FourCC::from_bytes(*b"AV01");
        assert_eq!(fcc.as_bytes(), b"AV01");
    }

    #[test]
    fn test_from_str_valid() {
        let fcc = FourCC::parse("VP90").expect("should succeed");
        assert_eq!(fcc.as_bytes(), b"VP90");
    }

    #[test]
    fn test_from_str_too_short() {
        assert!(FourCC::parse("VP9").is_none());
    }

    #[test]
    fn test_from_str_too_long() {
        assert!(FourCC::parse("VP901").is_none());
    }

    #[test]
    fn test_is_video_av1() {
        let fcc = FourCC::from_bytes(*b"AV01");
        assert!(fcc.is_video());
    }

    #[test]
    fn test_is_video_vp9() {
        let fcc = FourCC::from_bytes(*b"VP90");
        assert!(fcc.is_video());
    }

    #[test]
    fn test_is_video_vp8() {
        let fcc = FourCC::from_bytes(*b"VP80");
        assert!(fcc.is_video());
    }

    #[test]
    fn test_is_video_theora() {
        let fcc = FourCC::from_bytes(*b"theo");
        assert!(fcc.is_video());
    }

    #[test]
    fn test_is_audio_opus() {
        let fcc = FourCC::from_bytes(*b"Opus");
        assert!(fcc.is_audio());
        assert!(!fcc.is_video());
    }

    #[test]
    fn test_is_audio_flac() {
        let fcc = FourCC::from_bytes(*b"fLaC");
        assert!(fcc.is_audio());
    }

    #[test]
    fn test_display_ascii() {
        let fcc = FourCC::from_bytes(*b"AV01");
        assert_eq!(format!("{fcc}"), "AV01");
    }

    #[test]
    fn test_display_non_ascii() {
        // Non-printable bytes should render as '.'
        let fcc = FourCC::from_bytes([0x01, 0x02, 0x03, 0x04]);
        let s = format!("{fcc}");
        assert!(s.chars().all(|c| c == '.'));
    }

    #[test]
    fn test_as_u32() {
        let fcc = FourCC::from_bytes(*b"AV01");
        assert_eq!(fcc.as_u32(), u32::from_be_bytes(*b"AV01"));
    }

    #[test]
    fn test_from_array_trait() {
        let fcc: FourCC = [b'V', b'P', b'8', b'0'].into();
        assert!(fcc.is_video());
    }

    #[test]
    fn test_try_from_str_ok() {
        let fcc = FourCC::try_from("Opus").expect("try_from should succeed");
        assert!(fcc.is_audio());
    }

    #[test]
    fn test_try_from_str_err() {
        assert!(FourCC::try_from("XY").is_err());
    }

    #[test]
    fn test_registry_register_lookup() {
        let mut reg = FourCCRegistry::new();
        let code = FourCC::from_bytes(*b"AV01");
        reg.register(code, "AV1 Video");
        assert_eq!(reg.lookup(&code), Some("AV1 Video"));
    }

    #[test]
    fn test_registry_lookup_missing() {
        let reg = FourCCRegistry::new();
        let code = FourCC::from_bytes(*b"XXXX");
        assert!(reg.lookup(&code).is_none());
    }

    #[test]
    fn test_registry_len_and_empty() {
        let mut reg = FourCCRegistry::new();
        assert!(reg.is_empty());
        reg.register(FourCC::from_bytes(*b"AV01"), "AV1");
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn test_registry_with_defaults_not_empty() {
        let reg = FourCCRegistry::with_defaults();
        assert!(!reg.is_empty());
    }

    #[test]
    fn test_known_video_codecs_count() {
        assert_eq!(FourCCRegistry::known_video_codecs().len(), 4);
    }

    #[test]
    fn test_known_audio_codecs_count() {
        assert_eq!(FourCCRegistry::known_audio_codecs().len(), 4);
    }

    #[test]
    fn test_known_video_codecs_all_is_video() {
        for (fcc, _) in FourCCRegistry::known_video_codecs() {
            assert!(fcc.is_video(), "{fcc} should be video");
        }
    }

    #[test]
    fn test_equality() {
        let a = FourCC::from_bytes(*b"AV01");
        let b = FourCC::parse("AV01").expect("should succeed");
        assert_eq!(a, b);
    }
}
