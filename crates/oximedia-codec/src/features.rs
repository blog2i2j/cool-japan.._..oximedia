//! Codec capability and feature detection.
//!
//! [`CodecFeatures`] provides a runtime query interface for the capabilities
//! of a named codec (e.g. `"av1"`, `"vp9"`, `"opus"`).  The information is
//! based on a static capability table and can be used by higher-level pipeline
//! code to select the correct encoder/decoder configuration.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// Codec identifier
// ---------------------------------------------------------------------------

/// A codec name as a borrowed or owned string.
pub type CodecId = String;

// ---------------------------------------------------------------------------
// Codec feature flags
// ---------------------------------------------------------------------------

/// Capability flags for a specific codec.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodecFeatures {
    /// Codec identifier (e.g. `"av1"`, `"vp9"`, `"opus"`).
    pub codec: CodecId,
    /// Whether the codec supports HDR content (PQ/HLG transfer functions).
    pub hdr: bool,
    /// Whether the codec supports 10-bit (or deeper) bit depth.
    pub ten_bit: bool,
    /// Whether the codec supports B-frames (bi-directionally predicted frames).
    pub bframe: bool,
    /// Whether the codec supports lossless encoding.
    pub lossless: bool,
    /// Whether the codec is an audio codec (as opposed to video or image).
    pub audio: bool,
    /// Whether the codec supports scalable video coding (SVC / temporal layers).
    pub svc: bool,
    /// Maximum bit depth supported (e.g. 8, 10, 12).
    pub max_bit_depth: u8,
}

impl CodecFeatures {
    /// Create a [`CodecFeatures`] record for a named codec.
    ///
    /// Codec names are matched case-insensitively.  Unknown codecs return a
    /// conservative feature set (no HDR, no 10-bit, no B-frames).
    ///
    /// # Parameters
    /// - `codec` – codec identifier string (e.g. `"av1"`, `"vp9"`, `"h264"`).
    #[must_use]
    pub fn new(codec: &str) -> Self {
        let id = codec.to_lowercase();
        match id.as_str() {
            "av1" => Self {
                codec: id,
                hdr: true,
                ten_bit: true,
                bframe: true,
                lossless: true,
                audio: false,
                svc: true,
                max_bit_depth: 12,
            },
            "vp9" => Self {
                codec: id,
                hdr: true,
                ten_bit: true,
                bframe: false,
                lossless: true,
                audio: false,
                svc: true,
                max_bit_depth: 12,
            },
            "vp8" => Self {
                codec: id,
                hdr: false,
                ten_bit: false,
                bframe: false,
                lossless: false,
                audio: false,
                svc: false,
                max_bit_depth: 8,
            },
            "theora" => Self {
                codec: id,
                hdr: false,
                ten_bit: false,
                bframe: false,
                lossless: false,
                audio: false,
                svc: false,
                max_bit_depth: 8,
            },
            "h264" | "avc" => Self {
                codec: id,
                hdr: false,
                ten_bit: true,
                bframe: true,
                lossless: false,
                audio: false,
                svc: true,
                max_bit_depth: 10,
            },
            "h265" | "hevc" => Self {
                codec: id,
                hdr: true,
                ten_bit: true,
                bframe: true,
                lossless: false,
                audio: false,
                svc: true,
                max_bit_depth: 12,
            },
            "opus" => Self {
                codec: id,
                hdr: false,
                ten_bit: false,
                bframe: false,
                lossless: false,
                audio: true,
                svc: false,
                max_bit_depth: 16,
            },
            "vorbis" => Self {
                codec: id,
                hdr: false,
                ten_bit: false,
                bframe: false,
                lossless: false,
                audio: true,
                svc: false,
                max_bit_depth: 16,
            },
            "flac" => Self {
                codec: id,
                hdr: false,
                ten_bit: false,
                bframe: false,
                lossless: true,
                audio: true,
                svc: false,
                max_bit_depth: 24,
            },
            "ffv1" => Self {
                codec: id,
                hdr: true,
                ten_bit: true,
                bframe: false,
                lossless: true,
                audio: false,
                svc: false,
                max_bit_depth: 16,
            },
            "jpegxl" | "jxl" => Self {
                codec: id,
                hdr: true,
                ten_bit: true,
                bframe: false,
                lossless: true,
                audio: false,
                svc: false,
                max_bit_depth: 16,
            },
            _ => Self {
                codec: id,
                hdr: false,
                ten_bit: false,
                bframe: false,
                lossless: false,
                audio: false,
                svc: false,
                max_bit_depth: 8,
            },
        }
    }

    /// Returns `true` if the codec supports HDR content.
    #[must_use]
    pub fn supports_hdr(&self) -> bool {
        self.hdr
    }

    /// Returns `true` if the codec supports 10-bit (or deeper) bit depth.
    #[must_use]
    pub fn supports_10bit(&self) -> bool {
        self.ten_bit
    }

    /// Returns `true` if the codec supports B-frames.
    #[must_use]
    pub fn supports_bframe(&self) -> bool {
        self.bframe
    }

    /// Returns `true` if the codec supports lossless encoding.
    #[must_use]
    pub fn supports_lossless(&self) -> bool {
        self.lossless
    }

    /// Returns `true` if this is an audio codec.
    #[must_use]
    pub fn is_audio(&self) -> bool {
        self.audio
    }

    /// Returns `true` if the codec supports scalable video coding.
    #[must_use]
    pub fn supports_svc(&self) -> bool {
        self.svc
    }

    /// Returns the maximum bit depth supported by this codec.
    #[must_use]
    pub fn max_bit_depth(&self) -> u8 {
        self.max_bit_depth
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn av1_features_correct() {
        let f = CodecFeatures::new("av1");
        assert!(f.supports_hdr());
        assert!(f.supports_10bit());
        assert!(f.supports_bframe());
        assert!(f.supports_lossless());
        assert!(!f.is_audio());
        assert_eq!(f.max_bit_depth(), 12);
    }

    #[test]
    fn vp9_no_bframe() {
        let f = CodecFeatures::new("vp9");
        assert!(!f.supports_bframe());
        assert!(f.supports_10bit());
    }

    #[test]
    fn vp8_basic_only() {
        let f = CodecFeatures::new("vp8");
        assert!(!f.supports_hdr());
        assert!(!f.supports_10bit());
        assert!(!f.supports_bframe());
        assert_eq!(f.max_bit_depth(), 8);
    }

    #[test]
    fn opus_is_audio() {
        let f = CodecFeatures::new("opus");
        assert!(f.is_audio());
        assert!(!f.supports_bframe());
    }

    #[test]
    fn flac_lossless_audio() {
        let f = CodecFeatures::new("flac");
        assert!(f.supports_lossless());
        assert!(f.is_audio());
        assert_eq!(f.max_bit_depth(), 24);
    }

    #[test]
    fn ffv1_lossless_video_hdr() {
        let f = CodecFeatures::new("ffv1");
        assert!(f.supports_hdr());
        assert!(f.supports_lossless());
        assert!(!f.is_audio());
    }

    #[test]
    fn case_insensitive_lookup() {
        let lower = CodecFeatures::new("AV1");
        let upper = CodecFeatures::new("av1");
        assert_eq!(lower.supports_hdr(), upper.supports_hdr());
    }

    #[test]
    fn unknown_codec_conservative_defaults() {
        let f = CodecFeatures::new("my_unknown_codec");
        assert!(!f.supports_hdr());
        assert!(!f.supports_10bit());
        assert!(!f.supports_bframe());
        assert_eq!(f.max_bit_depth(), 8);
    }

    #[test]
    fn h265_hdr_and_bframe() {
        let f = CodecFeatures::new("h265");
        assert!(f.supports_hdr());
        assert!(f.supports_bframe());
    }
}
