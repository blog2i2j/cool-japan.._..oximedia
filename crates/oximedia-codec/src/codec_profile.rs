//! Codec profile and level definitions, and constraint validation.
//!
//! This module provides:
//! - Profile enumerations for AV1, VP9, and Opus.
//! - The AV1 level table (levels 2.0 – 6.3) with decoded capability limits.
//! - [`CodecConstraints`] — a codec-agnostic constraints bag and validation
//!   helpers for both video and audio streams.
//!
//! # Example
//!
//! ```rust
//! use oximedia_codec::codec_profile::{Av1Level, CodecConstraints};
//!
//! // Find the lowest AV1 level sufficient for 4K @ 60 fps / 20 Mbps.
//! let level = Av1Level::select_for(3840, 2160, 60.0, 20_000);
//! assert!(level.is_some());
//!
//! // Validate a stream against AV1 Main profile constraints.
//! let c = CodecConstraints::av1_main();
//! let result = CodecConstraints::validate_video(&c, 1920, 1080, 60.0, 10_000);
//! assert!(result.is_ok());
//! ```

#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_precision_loss)]

// ──────────────────────────────────────────────
// Profile enumerations
// ──────────────────────────────────────────────

/// AV1 codec profile.
///
/// Defined in the AV1 specification, Section 6.4.1.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Av1Profile {
    /// Main profile — 4:2:0 chroma subsampling, 8-bit and 10-bit.
    Main,
    /// High profile — 4:4:4 chroma subsampling, 8-bit and 10-bit.
    High,
    /// Professional profile — 4:2:2, 4:4:4, up to 12-bit.
    Professional,
}

impl std::fmt::Display for Av1Profile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Main => write!(f, "Main"),
            Self::High => write!(f, "High"),
            Self::Professional => write!(f, "Professional"),
        }
    }
}

/// VP9 codec profile.
///
/// Defined in the VP9 bitstream specification.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Vp9Profile {
    /// Profile 0 — 4:2:0, 8-bit.
    Profile0,
    /// Profile 1 — 4:2:2 / 4:4:4 / 4:4:0, 8-bit.
    Profile1,
    /// Profile 2 — 4:2:0, 10-bit or 12-bit.
    Profile2,
    /// Profile 3 — 4:2:2 / 4:4:4 / 4:4:0, 10-bit or 12-bit.
    Profile3,
}

impl std::fmt::Display for Vp9Profile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Profile0 => write!(f, "Profile 0"),
            Self::Profile1 => write!(f, "Profile 1"),
            Self::Profile2 => write!(f, "Profile 2"),
            Self::Profile3 => write!(f, "Profile 3"),
        }
    }
}

/// Opus application / profile mode.
///
/// Controls the encoder's optimisation target.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OpusProfile {
    /// VOIP — optimised for speech, allows aggressive packet loss concealment.
    Voip,
    /// Audio — optimised for music and general audio, fullband.
    Audio,
    /// Restricted Low Delay — no look-ahead, lowest latency.
    RestrictedLowDelay,
}

impl std::fmt::Display for OpusProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Voip => write!(f, "VOIP"),
            Self::Audio => write!(f, "Audio"),
            Self::RestrictedLowDelay => write!(f, "Restricted Low-Delay"),
        }
    }
}

// ──────────────────────────────────────────────
// AV1 level table
// ──────────────────────────────────────────────

/// An entry in the AV1 level table.
///
/// Level numbers are encoded as `major * 10 + minor`; e.g., level 5.1 is
/// encoded as `51`.  The table is derived from the AV1 specification,
/// Annex A.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Av1Level {
    /// Level code: `major * 10 + minor`.  Level 4.0 → `40`, 6.3 → `63`.
    pub level: u8,
    /// Maximum picture size in pixels (width × height).
    pub max_pic_size: u64,
    /// Maximum horizontal picture dimension in pixels.
    pub max_h_size: u32,
    /// Maximum vertical picture dimension in pixels.
    pub max_v_size: u32,
    /// Maximum display sample rate in samples per second.
    pub max_display_rate: u64,
    /// Maximum bitrate in Mbps (megabits per second).
    pub max_bitrate_mbps: u32,
}

impl Av1Level {
    /// Return the AV1 level entry for the given `code` (e.g., `20` = level 2.0).
    ///
    /// Returns `None` if the code does not correspond to a defined level.
    #[must_use]
    pub fn from_code(code: u8) -> Option<Self> {
        AV1_LEVEL_TABLE.iter().find(|l| l.level == code).cloned()
    }

    /// Find the lowest AV1 level sufficient for the specified video parameters.
    ///
    /// Returns `None` if no defined level can accommodate the request.
    #[must_use]
    pub fn select_for(width: u32, height: u32, fps: f32, bitrate_kbps: u32) -> Option<Self> {
        let pic_size = width as u64 * height as u64;
        let display_rate = (pic_size as f64 * fps as f64).ceil() as u64;
        let bitrate_mbps = (bitrate_kbps as f64 / 1000.0).ceil() as u32;

        AV1_LEVEL_TABLE
            .iter()
            .find(|l| {
                l.max_pic_size >= pic_size
                    && l.max_h_size >= width
                    && l.max_v_size >= height
                    && l.max_display_rate >= display_rate
                    && l.max_bitrate_mbps >= bitrate_mbps
            })
            .cloned()
    }

    /// Return a human-readable level string such as `"4.1"`.
    #[must_use]
    pub fn level_str(&self) -> String {
        format!("{}.{}", self.level / 10, self.level % 10)
    }
}

/// AV1 level table derived from the AV1 specification, Annex A.
///
/// Levels 2.0 through 6.3 are listed in ascending order of capability.
static AV1_LEVEL_TABLE: &[Av1Level] = &[
    Av1Level {
        level: 20,
        max_pic_size: 147_456,
        max_h_size: 2048,
        max_v_size: 1152,
        max_display_rate: 4_423_680,
        max_bitrate_mbps: 1,
    },
    Av1Level {
        level: 21,
        max_pic_size: 278_784,
        max_h_size: 2816,
        max_v_size: 1584,
        max_display_rate: 8_363_520,
        max_bitrate_mbps: 2,
    },
    Av1Level {
        level: 30,
        max_pic_size: 665_856,
        max_h_size: 4352,
        max_v_size: 2448,
        max_display_rate: 19_975_680,
        max_bitrate_mbps: 5,
    },
    Av1Level {
        level: 31,
        max_pic_size: 1_065_024,
        max_h_size: 5504,
        max_v_size: 3096,
        max_display_rate: 31_950_720,
        max_bitrate_mbps: 10,
    },
    Av1Level {
        level: 40,
        max_pic_size: 2_359_296,
        max_h_size: 6144,
        max_v_size: 3456,
        max_display_rate: 70_778_880,
        max_bitrate_mbps: 12,
    },
    Av1Level {
        level: 41,
        max_pic_size: 2_359_296,
        max_h_size: 6144,
        max_v_size: 3456,
        max_display_rate: 141_557_760,
        max_bitrate_mbps: 20,
    },
    Av1Level {
        level: 50,
        max_pic_size: 8_912_896,
        max_h_size: 8192,
        max_v_size: 4352,
        max_display_rate: 267_386_880,
        max_bitrate_mbps: 30,
    },
    Av1Level {
        level: 51,
        max_pic_size: 8_912_896,
        max_h_size: 8192,
        max_v_size: 4352,
        max_display_rate: 534_773_760,
        max_bitrate_mbps: 40,
    },
    Av1Level {
        level: 52,
        max_pic_size: 8_912_896,
        max_h_size: 8192,
        max_v_size: 4352,
        max_display_rate: 1_069_547_520,
        max_bitrate_mbps: 60,
    },
    Av1Level {
        level: 53,
        max_pic_size: 8_912_896,
        max_h_size: 8192,
        max_v_size: 4352,
        max_display_rate: 1_069_547_520,
        max_bitrate_mbps: 60,
    },
    Av1Level {
        level: 60,
        max_pic_size: 35_651_584,
        max_h_size: 16384,
        max_v_size: 8704,
        max_display_rate: 1_069_547_520,
        max_bitrate_mbps: 100,
    },
    Av1Level {
        level: 61,
        max_pic_size: 35_651_584,
        max_h_size: 16384,
        max_v_size: 8704,
        max_display_rate: 2_139_095_040,
        max_bitrate_mbps: 160,
    },
    Av1Level {
        level: 62,
        max_pic_size: 35_651_584,
        max_h_size: 16384,
        max_v_size: 8704,
        max_display_rate: 4_278_190_080,
        max_bitrate_mbps: 240,
    },
    Av1Level {
        level: 63,
        max_pic_size: 35_651_584,
        max_h_size: 16384,
        max_v_size: 8704,
        max_display_rate: 4_278_190_080,
        max_bitrate_mbps: 240,
    },
];

// ──────────────────────────────────────────────
// CodecConstraints
// ──────────────────────────────────────────────

/// A codec-agnostic constraints record used for profile/level validation.
///
/// Callers obtain pre-filled instances via the associated constructor functions
/// (e.g., [`CodecConstraints::av1_main`]) and then call
/// [`CodecConstraints::validate_video`] or [`CodecConstraints::validate_audio`]
/// to collect any violations.
#[derive(Debug, Clone)]
pub struct CodecConstraints {
    /// Codec name (e.g., `"AV1"`, `"VP9"`, `"Opus"`, `"FLAC"`).
    pub codec: String,
    /// Profile name (e.g., `"Main"`, `"Profile 0"`, `"Audio"`).
    pub profile: String,
    /// Level name (e.g., `"4.0"`, `"N/A"`).
    pub level: String,
    /// Maximum picture width in pixels (0 = unlimited).
    pub max_width: u32,
    /// Maximum picture height in pixels (0 = unlimited).
    pub max_height: u32,
    /// Maximum frames per second (0.0 = unlimited).
    pub max_fps: f32,
    /// Maximum video / audio bitrate in kbps (0 = unlimited).
    pub max_bitrate_kbps: u32,
    /// Maximum audio sample rate in Hz (0 = unlimited).
    pub max_sample_rate: u32,
    /// Supported bit depths (empty = all).
    pub bit_depths: Vec<u8>,
    /// Supported colour space names (empty = all).
    pub color_spaces: Vec<String>,
}

impl CodecConstraints {
    // ── Preset constructors ─────────────────────────────────────────────────

    /// Constraints for AV1 Main profile (4:2:0, 8-bit or 10-bit).
    #[must_use]
    pub fn av1_main() -> Self {
        Self {
            codec: "AV1".to_owned(),
            profile: "Main".to_owned(),
            level: "6.3".to_owned(),
            max_width: 16384,
            max_height: 8704,
            max_fps: 300.0,
            max_bitrate_kbps: 240_000,
            max_sample_rate: 0,
            bit_depths: vec![8, 10],
            color_spaces: vec![
                "BT.601".to_owned(),
                "BT.709".to_owned(),
                "BT.2020".to_owned(),
            ],
        }
    }

    /// Constraints for VP9 Profile 0 (4:2:0, 8-bit).
    #[must_use]
    pub fn vp9_profile0() -> Self {
        Self {
            codec: "VP9".to_owned(),
            profile: "Profile 0".to_owned(),
            level: "6.2".to_owned(),
            max_width: 16384,
            max_height: 8704,
            max_fps: 240.0,
            max_bitrate_kbps: 180_000,
            max_sample_rate: 0,
            bit_depths: vec![8],
            color_spaces: vec!["BT.601".to_owned(), "BT.709".to_owned()],
        }
    }

    /// Constraints for Opus Audio profile.
    #[must_use]
    pub fn opus_audio() -> Self {
        Self {
            codec: "Opus".to_owned(),
            profile: "Audio".to_owned(),
            level: "N/A".to_owned(),
            max_width: 0,
            max_height: 0,
            max_fps: 0.0,
            max_bitrate_kbps: 512,
            max_sample_rate: 48_000,
            bit_depths: vec![16, 24, 32],
            color_spaces: Vec::new(),
        }
    }

    /// Constraints for FLAC lossless audio.
    #[must_use]
    pub fn flac_standard() -> Self {
        Self {
            codec: "FLAC".to_owned(),
            profile: "Standard".to_owned(),
            level: "N/A".to_owned(),
            max_width: 0,
            max_height: 0,
            max_fps: 0.0,
            max_bitrate_kbps: 0, // lossless — no fixed cap
            max_sample_rate: 655_350,
            bit_depths: vec![8, 16, 20, 24, 32],
            color_spaces: Vec::new(),
        }
    }

    // ── Validation helpers ──────────────────────────────────────────────────

    /// Validate video stream parameters against `constraints`.
    ///
    /// Returns `Ok(())` when all constraints are satisfied, or
    /// `Err(violations)` with a list of human-readable violation messages.
    pub fn validate_video(
        constraints: &Self,
        width: u32,
        height: u32,
        fps: f32,
        bitrate_kbps: u32,
    ) -> Result<(), Vec<String>> {
        let mut violations = Vec::new();

        if constraints.max_width > 0 && width > constraints.max_width {
            violations.push(format!(
                "width {} exceeds maximum {} for {} {}",
                width, constraints.max_width, constraints.codec, constraints.profile
            ));
        }

        if constraints.max_height > 0 && height > constraints.max_height {
            violations.push(format!(
                "height {} exceeds maximum {} for {} {}",
                height, constraints.max_height, constraints.codec, constraints.profile
            ));
        }

        if constraints.max_fps > 0.0 && fps > constraints.max_fps {
            violations.push(format!(
                "fps {:.2} exceeds maximum {:.2} for {} {}",
                fps, constraints.max_fps, constraints.codec, constraints.profile
            ));
        }

        if constraints.max_bitrate_kbps > 0 && bitrate_kbps > constraints.max_bitrate_kbps {
            violations.push(format!(
                "bitrate {} kbps exceeds maximum {} kbps for {} {}",
                bitrate_kbps, constraints.max_bitrate_kbps, constraints.codec, constraints.profile
            ));
        }

        if violations.is_empty() {
            Ok(())
        } else {
            Err(violations)
        }
    }

    /// Validate audio stream parameters against `constraints`.
    ///
    /// Returns `Ok(())` when all constraints are satisfied, or
    /// `Err(violations)` with a list of human-readable violation messages.
    pub fn validate_audio(
        constraints: &Self,
        sample_rate: u32,
        channels: u8,
        bitrate_kbps: u32,
    ) -> Result<(), Vec<String>> {
        let mut violations = Vec::new();

        if constraints.max_sample_rate > 0 && sample_rate > constraints.max_sample_rate {
            violations.push(format!(
                "sample rate {} Hz exceeds maximum {} Hz for {} {}",
                sample_rate, constraints.max_sample_rate, constraints.codec, constraints.profile
            ));
        }

        // Opus supports up to 255 channels; FLAC up to 8 in standard mode.
        let max_channels: u8 = if constraints.codec == "FLAC" { 8 } else { 255 };
        if channels > max_channels {
            violations.push(format!(
                "channel count {} exceeds maximum {} for {} {}",
                channels, max_channels, constraints.codec, constraints.profile
            ));
        }

        if constraints.max_bitrate_kbps > 0 && bitrate_kbps > constraints.max_bitrate_kbps {
            violations.push(format!(
                "bitrate {} kbps exceeds maximum {} kbps for {} {}",
                bitrate_kbps, constraints.max_bitrate_kbps, constraints.codec, constraints.profile
            ));
        }

        if violations.is_empty() {
            Ok(())
        } else {
            Err(violations)
        }
    }
}

// ──────────────────────────────────────────────
// VP9 Profile Definitions
// ──────────────────────────────────────────────

/// Chroma subsampling format.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ChromaSubsampling {
    /// 4:2:0 — both chroma channels halved in both dimensions.
    Yuv420,
    /// 4:2:2 — chroma halved horizontally only.
    Yuv422,
    /// 4:4:4 — no chroma subsampling.
    Yuv444,
    /// 4:4:0 — chroma halved vertically only.
    Yuv440,
}

impl std::fmt::Display for ChromaSubsampling {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Yuv420 => write!(f, "4:2:0"),
            Self::Yuv422 => write!(f, "4:2:2"),
            Self::Yuv444 => write!(f, "4:4:4"),
            Self::Yuv440 => write!(f, "4:4:0"),
        }
    }
}

/// Full VP9 profile definition with bit depth and chroma constraints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Vp9ProfileDef {
    /// Profile enum variant.
    pub profile: Vp9Profile,
    /// Allowed bit depths.
    pub bit_depths: Vec<u8>,
    /// Allowed chroma subsampling formats.
    pub chroma_formats: Vec<ChromaSubsampling>,
    /// Maximum resolution width.
    pub max_width: u32,
    /// Maximum resolution height.
    pub max_height: u32,
}

impl Vp9ProfileDef {
    /// VP9 Profile 0: 4:2:0, 8-bit only.
    pub fn profile0() -> Self {
        Self {
            profile: Vp9Profile::Profile0,
            bit_depths: vec![8],
            chroma_formats: vec![ChromaSubsampling::Yuv420],
            max_width: 16384,
            max_height: 16384,
        }
    }

    /// VP9 Profile 1: 4:2:2, 4:4:4, 4:4:0, 8-bit.
    pub fn profile1() -> Self {
        Self {
            profile: Vp9Profile::Profile1,
            bit_depths: vec![8],
            chroma_formats: vec![
                ChromaSubsampling::Yuv422,
                ChromaSubsampling::Yuv444,
                ChromaSubsampling::Yuv440,
            ],
            max_width: 16384,
            max_height: 16384,
        }
    }

    /// VP9 Profile 2: 4:2:0, 10-bit or 12-bit.
    pub fn profile2() -> Self {
        Self {
            profile: Vp9Profile::Profile2,
            bit_depths: vec![10, 12],
            chroma_formats: vec![ChromaSubsampling::Yuv420],
            max_width: 16384,
            max_height: 16384,
        }
    }

    /// VP9 Profile 3: 4:2:2, 4:4:4, 4:4:0, 10-bit or 12-bit.
    pub fn profile3() -> Self {
        Self {
            profile: Vp9Profile::Profile3,
            bit_depths: vec![10, 12],
            chroma_formats: vec![
                ChromaSubsampling::Yuv422,
                ChromaSubsampling::Yuv444,
                ChromaSubsampling::Yuv440,
            ],
            max_width: 16384,
            max_height: 16384,
        }
    }

    /// Validate that a given stream matches this profile's constraints.
    pub fn validate(
        &self,
        width: u32,
        height: u32,
        bit_depth: u8,
        chroma: &ChromaSubsampling,
    ) -> Result<(), Vec<String>> {
        let mut violations = Vec::new();

        if !self.bit_depths.contains(&bit_depth) {
            violations.push(format!(
                "bit depth {} not allowed in VP9 {} (allowed: {:?})",
                bit_depth, self.profile, self.bit_depths
            ));
        }

        if !self.chroma_formats.contains(chroma) {
            violations.push(format!(
                "chroma format {} not allowed in VP9 {} (allowed: {:?})",
                chroma,
                self.profile,
                self.chroma_formats
                    .iter()
                    .map(|c| c.to_string())
                    .collect::<Vec<_>>()
            ));
        }

        if width > self.max_width {
            violations.push(format!(
                "width {} exceeds max {} for VP9 {}",
                width, self.max_width, self.profile
            ));
        }

        if height > self.max_height {
            violations.push(format!(
                "height {} exceeds max {} for VP9 {}",
                height, self.max_height, self.profile
            ));
        }

        if violations.is_empty() {
            Ok(())
        } else {
            Err(violations)
        }
    }
}

// ──────────────────────────────────────────────
// AV1 Profile Constraints Validation
// ──────────────────────────────────────────────

/// Full AV1 profile definition with chroma and bit depth constraints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Av1ProfileDef {
    /// Profile enum variant.
    pub profile: Av1Profile,
    /// Allowed bit depths.
    pub bit_depths: Vec<u8>,
    /// Allowed chroma subsampling formats.
    pub chroma_formats: Vec<ChromaSubsampling>,
    /// Whether monochrome is allowed.
    pub mono_allowed: bool,
}

impl Av1ProfileDef {
    /// AV1 Main profile: 4:2:0, 8/10-bit, mono allowed.
    pub fn main() -> Self {
        Self {
            profile: Av1Profile::Main,
            bit_depths: vec![8, 10],
            chroma_formats: vec![ChromaSubsampling::Yuv420],
            mono_allowed: true,
        }
    }

    /// AV1 High profile: 4:2:0, 4:4:4, 8/10-bit, mono allowed.
    pub fn high() -> Self {
        Self {
            profile: Av1Profile::High,
            bit_depths: vec![8, 10],
            chroma_formats: vec![ChromaSubsampling::Yuv420, ChromaSubsampling::Yuv444],
            mono_allowed: true,
        }
    }

    /// AV1 Professional profile: all formats, 8/10/12-bit.
    pub fn professional() -> Self {
        Self {
            profile: Av1Profile::Professional,
            bit_depths: vec![8, 10, 12],
            chroma_formats: vec![
                ChromaSubsampling::Yuv420,
                ChromaSubsampling::Yuv422,
                ChromaSubsampling::Yuv444,
            ],
            mono_allowed: true,
        }
    }

    /// Validate a stream against this profile's constraints.
    pub fn validate(
        &self,
        bit_depth: u8,
        chroma: &ChromaSubsampling,
        is_monochrome: bool,
    ) -> Result<(), Vec<String>> {
        let mut violations = Vec::new();

        if !self.bit_depths.contains(&bit_depth) {
            violations.push(format!(
                "bit depth {} not allowed in AV1 {} (allowed: {:?})",
                bit_depth, self.profile, self.bit_depths
            ));
        }

        if !is_monochrome && !self.chroma_formats.contains(chroma) {
            violations.push(format!(
                "chroma format {} not allowed in AV1 {}",
                chroma, self.profile
            ));
        }

        if is_monochrome && !self.mono_allowed {
            violations.push(format!("monochrome not allowed in AV1 {}", self.profile));
        }

        if violations.is_empty() {
            Ok(())
        } else {
            Err(violations)
        }
    }
}

// ──────────────────────────────────────────────
// Codec Capability Negotiation
// ──────────────────────────────────────────────

/// Codec capability descriptor for negotiation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodecCapability {
    /// Codec name (e.g., "AV1", "VP9").
    pub codec: String,
    /// Supported profiles.
    pub profiles: Vec<String>,
    /// Supported bit depths.
    pub bit_depths: Vec<u8>,
    /// Supported chroma formats.
    pub chroma_formats: Vec<ChromaSubsampling>,
    /// Maximum resolution (width, height).
    pub max_resolution: (u32, u32),
    /// Maximum bitrate in kbps.
    pub max_bitrate_kbps: u32,
}

/// Result of capability negotiation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NegotiatedCapability {
    /// Codec name.
    pub codec: String,
    /// Best common profile (highest tier both support).
    pub profile: String,
    /// Common bit depths.
    pub bit_depths: Vec<u8>,
    /// Common chroma formats.
    pub chroma_formats: Vec<ChromaSubsampling>,
    /// Minimum of both max resolutions.
    pub max_resolution: (u32, u32),
    /// Minimum of both max bitrates.
    pub max_bitrate_kbps: u32,
}

/// Negotiate the best common capability between an encoder and decoder.
///
/// Returns `None` if the two sides have no compatible configuration.
pub fn negotiate_capabilities(
    encoder: &CodecCapability,
    decoder: &CodecCapability,
) -> Option<NegotiatedCapability> {
    if encoder.codec != decoder.codec {
        return None;
    }

    // Find common profiles.
    let common_profiles: Vec<String> = encoder
        .profiles
        .iter()
        .filter(|p| decoder.profiles.contains(p))
        .cloned()
        .collect();

    if common_profiles.is_empty() {
        return None;
    }

    // Find common bit depths.
    let common_depths: Vec<u8> = encoder
        .bit_depths
        .iter()
        .filter(|d| decoder.bit_depths.contains(d))
        .copied()
        .collect();

    if common_depths.is_empty() {
        return None;
    }

    // Find common chroma formats.
    let common_chroma: Vec<ChromaSubsampling> = encoder
        .chroma_formats
        .iter()
        .filter(|c| decoder.chroma_formats.contains(c))
        .cloned()
        .collect();

    if common_chroma.is_empty() {
        return None;
    }

    // Use the last (highest) common profile.
    let best_profile = common_profiles.last().cloned().unwrap_or_default();

    Some(NegotiatedCapability {
        codec: encoder.codec.clone(),
        profile: best_profile,
        bit_depths: common_depths,
        chroma_formats: common_chroma,
        max_resolution: (
            encoder.max_resolution.0.min(decoder.max_resolution.0),
            encoder.max_resolution.1.min(decoder.max_resolution.1),
        ),
        max_bitrate_kbps: encoder.max_bitrate_kbps.min(decoder.max_bitrate_kbps),
    })
}

// ──────────────────────────────────────────────
// Profile Compatibility Matrix
// ──────────────────────────────────────────────

/// Check if a decoder supporting `decoder_profile` can handle content
/// encoded with `content_profile`.
///
/// AV1 profiles form a strict hierarchy: Professional > High > Main.
/// VP9 profiles: Profile 2/3 can decode 0/1 respectively (same chroma, higher bit depth).
pub fn is_profile_compatible(codec: &str, decoder_profile: &str, content_profile: &str) -> bool {
    match codec {
        "AV1" => {
            let decoder_rank = av1_profile_rank(decoder_profile);
            let content_rank = av1_profile_rank(content_profile);
            decoder_rank >= content_rank
        }
        "VP9" => {
            let decoder_rank = vp9_profile_rank(decoder_profile);
            let content_rank = vp9_profile_rank(content_profile);
            // VP9 profiles are only compatible within the same chroma class:
            // Profile 0 ↔ Profile 2 (both 4:2:0)
            // Profile 1 ↔ Profile 3 (both 4:2:2/4:4:4)
            let same_chroma_class =
                (decoder_rank % 2) == (content_rank % 2) || decoder_rank >= content_rank;
            same_chroma_class && decoder_rank >= content_rank
        }
        _ => decoder_profile == content_profile,
    }
}

fn av1_profile_rank(profile: &str) -> u8 {
    match profile {
        "Main" => 0,
        "High" => 1,
        "Professional" => 2,
        _ => 0,
    }
}

fn vp9_profile_rank(profile: &str) -> u8 {
    match profile {
        "Profile 0" => 0,
        "Profile 1" => 1,
        "Profile 2" => 2,
        "Profile 3" => 3,
        _ => 0,
    }
}

// ──────────────────────────────────────────────
// Hardware Decode Profile Detection
// ──────────────────────────────────────────────

/// Hardware decoder tier.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum HwDecoderTier {
    /// Entry-level (mobile, embedded): AV1 Main 8-bit, VP9 Profile 0.
    Entry,
    /// Mid-range (laptop, desktop): AV1 Main/High 10-bit, VP9 Profile 0/2.
    MidRange,
    /// High-end (workstation, pro): AV1 all profiles, VP9 all profiles.
    HighEnd,
    /// Broadcast / professional (studio hardware).
    Broadcast,
}

impl std::fmt::Display for HwDecoderTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Entry => write!(f, "Entry"),
            Self::MidRange => write!(f, "Mid-Range"),
            Self::HighEnd => write!(f, "High-End"),
            Self::Broadcast => write!(f, "Broadcast"),
        }
    }
}

/// Hardware decode capability for a specific decoder tier.
#[derive(Debug, Clone)]
pub struct HwDecodeCapability {
    /// Tier of the hardware.
    pub tier: HwDecoderTier,
    /// Supported AV1 profiles.
    pub av1_profiles: Vec<Av1Profile>,
    /// Supported VP9 profiles.
    pub vp9_profiles: Vec<Vp9Profile>,
    /// Maximum supported bit depth.
    pub max_bit_depth: u8,
    /// Maximum resolution (width, height).
    pub max_resolution: (u32, u32),
    /// Whether HDR / HLG is supported.
    pub hdr_supported: bool,
}

impl HwDecodeCapability {
    /// Entry-level hardware capability.
    pub fn entry() -> Self {
        Self {
            tier: HwDecoderTier::Entry,
            av1_profiles: vec![Av1Profile::Main],
            vp9_profiles: vec![Vp9Profile::Profile0],
            max_bit_depth: 8,
            max_resolution: (1920, 1080),
            hdr_supported: false,
        }
    }

    /// Mid-range hardware capability.
    pub fn mid_range() -> Self {
        Self {
            tier: HwDecoderTier::MidRange,
            av1_profiles: vec![Av1Profile::Main, Av1Profile::High],
            vp9_profiles: vec![Vp9Profile::Profile0, Vp9Profile::Profile2],
            max_bit_depth: 10,
            max_resolution: (3840, 2160),
            hdr_supported: true,
        }
    }

    /// High-end hardware capability.
    pub fn high_end() -> Self {
        Self {
            tier: HwDecoderTier::HighEnd,
            av1_profiles: vec![Av1Profile::Main, Av1Profile::High, Av1Profile::Professional],
            vp9_profiles: vec![
                Vp9Profile::Profile0,
                Vp9Profile::Profile1,
                Vp9Profile::Profile2,
                Vp9Profile::Profile3,
            ],
            max_bit_depth: 12,
            max_resolution: (7680, 4320),
            hdr_supported: true,
        }
    }

    /// Broadcast / professional hardware capability.
    pub fn broadcast() -> Self {
        Self {
            tier: HwDecoderTier::Broadcast,
            av1_profiles: vec![Av1Profile::Main, Av1Profile::High, Av1Profile::Professional],
            vp9_profiles: vec![
                Vp9Profile::Profile0,
                Vp9Profile::Profile1,
                Vp9Profile::Profile2,
                Vp9Profile::Profile3,
            ],
            max_bit_depth: 12,
            max_resolution: (16384, 8704),
            hdr_supported: true,
        }
    }

    /// Check if this hardware can decode the given AV1 profile and bit depth.
    pub fn can_decode_av1(&self, profile: &Av1Profile, bit_depth: u8) -> bool {
        self.av1_profiles.contains(profile) && bit_depth <= self.max_bit_depth
    }

    /// Check if this hardware can decode the given VP9 profile and bit depth.
    pub fn can_decode_vp9(&self, profile: &Vp9Profile, bit_depth: u8) -> bool {
        self.vp9_profiles.contains(profile) && bit_depth <= self.max_bit_depth
    }

    /// Check if resolution is within hardware limits.
    pub fn can_handle_resolution(&self, width: u32, height: u32) -> bool {
        width <= self.max_resolution.0 && height <= self.max_resolution.1
    }
}

/// Determine the minimum hardware tier required for a given stream.
pub fn required_hw_tier(
    codec: &str,
    profile: &str,
    bit_depth: u8,
    width: u32,
    height: u32,
) -> HwDecoderTier {
    let tiers = [
        HwDecodeCapability::entry(),
        HwDecodeCapability::mid_range(),
        HwDecodeCapability::high_end(),
        HwDecodeCapability::broadcast(),
    ];

    for cap in &tiers {
        let profile_ok = match codec {
            "AV1" => {
                let av1p = match profile {
                    "Main" => Some(Av1Profile::Main),
                    "High" => Some(Av1Profile::High),
                    "Professional" => Some(Av1Profile::Professional),
                    _ => None,
                };
                av1p.map_or(false, |p| cap.can_decode_av1(&p, bit_depth))
            }
            "VP9" => {
                let vp9p = match profile {
                    "Profile 0" => Some(Vp9Profile::Profile0),
                    "Profile 1" => Some(Vp9Profile::Profile1),
                    "Profile 2" => Some(Vp9Profile::Profile2),
                    "Profile 3" => Some(Vp9Profile::Profile3),
                    _ => None,
                };
                vp9p.map_or(false, |p| cap.can_decode_vp9(&p, bit_depth))
            }
            _ => bit_depth <= cap.max_bit_depth,
        };

        if profile_ok && cap.can_handle_resolution(width, height) {
            return cap.tier.clone();
        }
    }

    HwDecoderTier::Broadcast
}

// ──────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── 1. Av1Level::from_code: known level ──────────────────────────────────

    #[test]
    fn av1_level_from_code_known() {
        let l = Av1Level::from_code(40);
        assert!(l.is_some(), "level 4.0 (code 40) must exist");
        let l = l.expect("level should exist");
        assert_eq!(l.level, 40);
        assert_eq!(l.level_str(), "4.0");
    }

    // ── 2. Av1Level::from_code: unknown level ────────────────────────────────

    #[test]
    fn av1_level_from_code_unknown() {
        let l = Av1Level::from_code(99);
        assert!(l.is_none(), "code 99 should not correspond to any level");
    }

    // ── 3. Av1Level::select_for: 1080p/30 should fit in level 4.0 ───────────

    #[test]
    fn av1_level_select_1080p30() {
        let l = Av1Level::select_for(1920, 1080, 30.0, 10_000);
        assert!(l.is_some(), "1080p/30/10Mbps should resolve to some level");
        // Level 4.0 has max_bitrate_mbps=12, which is ≥ 10 Mbps.
        assert!(
            l.expect("level should be found").level >= 40,
            "should be at least level 4.0"
        );
    }

    // ── 4. Av1Level::select_for: 8K/120 should exceed level 6.3 ────────────

    #[test]
    fn av1_level_select_exceeds_table() {
        // 8K at 120fps with 1Gbps — no defined level covers this.
        let l = Av1Level::select_for(7680, 4320, 120.0, 1_000_000);
        assert!(
            l.is_none(),
            "extreme parameters should find no matching level"
        );
    }

    // ── 5. CodecConstraints::av1_main: valid 1080p passes ───────────────────

    #[test]
    fn av1_main_valid_1080p() {
        let c = CodecConstraints::av1_main();
        let r = CodecConstraints::validate_video(&c, 1920, 1080, 60.0, 10_000);
        assert!(
            r.is_ok(),
            "1080p/60/10Mbps should pass AV1 Main constraints"
        );
    }

    // ── 6. CodecConstraints::av1_main: oversized width fails ─────────────────

    #[test]
    fn av1_main_oversized_width_fails() {
        let c = CodecConstraints::av1_main();
        let r = CodecConstraints::validate_video(&c, 20_000, 1080, 30.0, 5_000);
        assert!(r.is_err());
        let v = r.expect_err("validation should return errors");
        assert!(v.iter().any(|s| s.contains("width")));
    }

    // ── 7. CodecConstraints::vp9_profile0: 8-bit accepted ───────────────────

    #[test]
    fn vp9_profile0_8bit_accepted() {
        let c = CodecConstraints::vp9_profile0();
        assert!(c.bit_depths.contains(&8));
    }

    // ── 8. CodecConstraints::vp9_profile0: 10-bit not in bit_depths ─────────

    #[test]
    fn vp9_profile0_no_10bit() {
        let c = CodecConstraints::vp9_profile0();
        assert!(!c.bit_depths.contains(&10));
    }

    // ── 9. CodecConstraints::opus_audio: valid parameters pass ──────────────

    #[test]
    fn opus_audio_valid() {
        let c = CodecConstraints::opus_audio();
        let r = CodecConstraints::validate_audio(&c, 48_000, 2, 128);
        assert!(
            r.is_ok(),
            "48kHz/2ch/128kbps should pass Opus Audio constraints"
        );
    }

    // ── 10. CodecConstraints::opus_audio: bitrate over limit fails ───────────

    #[test]
    fn opus_audio_bitrate_exceeded() {
        let c = CodecConstraints::opus_audio();
        let r = CodecConstraints::validate_audio(&c, 48_000, 2, 1_000);
        assert!(r.is_err());
        let v = r.expect_err("validation should return errors");
        assert!(v.iter().any(|s| s.contains("bitrate")));
    }

    // ── 11. CodecConstraints::flac_standard: high sample rate passes ─────────

    #[test]
    fn flac_high_sample_rate_passes() {
        let c = CodecConstraints::flac_standard();
        let r = CodecConstraints::validate_audio(&c, 192_000, 2, 0);
        assert!(r.is_ok(), "192kHz FLAC should pass");
    }

    // ── 12. validate_video returns all violations at once ────────────────────

    #[test]
    fn validate_video_multiple_violations() {
        let c = CodecConstraints::vp9_profile0();
        // Width + height + fps all exceed VP9 Profile 0 limits.
        let r = CodecConstraints::validate_video(&c, 99_999, 99_999, 9999.0, 99_999);
        assert!(r.is_err());
        let v = r.expect_err("validation should return errors");
        // Expect at least three violations (width, height, fps or bitrate).
        assert!(
            v.len() >= 3,
            "expected ≥3 violations, got {}: {:?}",
            v.len(),
            v
        );
    }

    // ── VP9 Profile Definitions ─────────────────────────────────────────────

    #[test]
    fn vp9_profile0_valid_420_8bit() {
        let p = Vp9ProfileDef::profile0();
        let r = p.validate(1920, 1080, 8, &ChromaSubsampling::Yuv420);
        assert!(r.is_ok());
    }

    #[test]
    fn vp9_profile0_rejects_10bit() {
        let p = Vp9ProfileDef::profile0();
        let r = p.validate(1920, 1080, 10, &ChromaSubsampling::Yuv420);
        assert!(r.is_err());
    }

    #[test]
    fn vp9_profile0_rejects_444() {
        let p = Vp9ProfileDef::profile0();
        let r = p.validate(1920, 1080, 8, &ChromaSubsampling::Yuv444);
        assert!(r.is_err());
    }

    #[test]
    fn vp9_profile2_accepts_10bit_420() {
        let p = Vp9ProfileDef::profile2();
        let r = p.validate(3840, 2160, 10, &ChromaSubsampling::Yuv420);
        assert!(r.is_ok());
    }

    #[test]
    fn vp9_profile3_accepts_12bit_444() {
        let p = Vp9ProfileDef::profile3();
        let r = p.validate(1920, 1080, 12, &ChromaSubsampling::Yuv444);
        assert!(r.is_ok());
    }

    // ── AV1 Profile Constraints Validation ───────────────────────────────────

    #[test]
    fn av1_main_accepts_420_10bit() {
        let p = Av1ProfileDef::main();
        let r = p.validate(10, &ChromaSubsampling::Yuv420, false);
        assert!(r.is_ok());
    }

    #[test]
    fn av1_main_rejects_444() {
        let p = Av1ProfileDef::main();
        let r = p.validate(8, &ChromaSubsampling::Yuv444, false);
        assert!(r.is_err());
    }

    #[test]
    fn av1_high_accepts_444() {
        let p = Av1ProfileDef::high();
        let r = p.validate(10, &ChromaSubsampling::Yuv444, false);
        assert!(r.is_ok());
    }

    #[test]
    fn av1_professional_accepts_422_12bit() {
        let p = Av1ProfileDef::professional();
        let r = p.validate(12, &ChromaSubsampling::Yuv422, false);
        assert!(r.is_ok());
    }

    #[test]
    fn av1_main_rejects_12bit() {
        let p = Av1ProfileDef::main();
        let r = p.validate(12, &ChromaSubsampling::Yuv420, false);
        assert!(r.is_err());
    }

    // ── Capability Negotiation ───────────────────────────────────────────────

    #[test]
    fn negotiate_compatible() {
        let enc = CodecCapability {
            codec: "AV1".to_string(),
            profiles: vec!["Main".to_string(), "High".to_string()],
            bit_depths: vec![8, 10],
            chroma_formats: vec![ChromaSubsampling::Yuv420, ChromaSubsampling::Yuv444],
            max_resolution: (3840, 2160),
            max_bitrate_kbps: 50_000,
        };
        let dec = CodecCapability {
            codec: "AV1".to_string(),
            profiles: vec!["Main".to_string()],
            bit_depths: vec![8, 10],
            chroma_formats: vec![ChromaSubsampling::Yuv420],
            max_resolution: (1920, 1080),
            max_bitrate_kbps: 20_000,
        };
        let result = negotiate_capabilities(&enc, &dec);
        assert!(result.is_some());
        let neg = result.expect("negotiation should succeed");
        assert_eq!(neg.profile, "Main");
        assert_eq!(neg.max_resolution, (1920, 1080));
        assert_eq!(neg.max_bitrate_kbps, 20_000);
    }

    #[test]
    fn negotiate_incompatible_codec() {
        let enc = CodecCapability {
            codec: "AV1".to_string(),
            profiles: vec!["Main".to_string()],
            bit_depths: vec![8],
            chroma_formats: vec![ChromaSubsampling::Yuv420],
            max_resolution: (1920, 1080),
            max_bitrate_kbps: 10_000,
        };
        let dec = CodecCapability {
            codec: "VP9".to_string(),
            profiles: vec!["Profile 0".to_string()],
            bit_depths: vec![8],
            chroma_formats: vec![ChromaSubsampling::Yuv420],
            max_resolution: (1920, 1080),
            max_bitrate_kbps: 10_000,
        };
        assert!(negotiate_capabilities(&enc, &dec).is_none());
    }

    #[test]
    fn negotiate_no_common_depth() {
        let enc = CodecCapability {
            codec: "AV1".to_string(),
            profiles: vec!["Main".to_string()],
            bit_depths: vec![8],
            chroma_formats: vec![ChromaSubsampling::Yuv420],
            max_resolution: (1920, 1080),
            max_bitrate_kbps: 10_000,
        };
        let dec = CodecCapability {
            codec: "AV1".to_string(),
            profiles: vec!["Main".to_string()],
            bit_depths: vec![10],
            chroma_formats: vec![ChromaSubsampling::Yuv420],
            max_resolution: (1920, 1080),
            max_bitrate_kbps: 10_000,
        };
        assert!(negotiate_capabilities(&enc, &dec).is_none());
    }

    // ── Profile Compatibility Matrix ─────────────────────────────────────────

    #[test]
    fn av1_professional_decodes_main() {
        assert!(is_profile_compatible("AV1", "Professional", "Main"));
    }

    #[test]
    fn av1_main_cannot_decode_high() {
        assert!(!is_profile_compatible("AV1", "Main", "High"));
    }

    #[test]
    fn vp9_profile2_decodes_profile0() {
        assert!(is_profile_compatible("VP9", "Profile 2", "Profile 0"));
    }

    #[test]
    fn vp9_profile0_cannot_decode_profile2() {
        assert!(!is_profile_compatible("VP9", "Profile 0", "Profile 2"));
    }

    // ── Hardware Decode Profile Detection ────────────────────────────────────

    #[test]
    fn hw_entry_supports_av1_main_8bit() {
        let cap = HwDecodeCapability::entry();
        assert!(cap.can_decode_av1(&Av1Profile::Main, 8));
        assert!(!cap.can_decode_av1(&Av1Profile::High, 10));
    }

    #[test]
    fn hw_mid_range_supports_4k() {
        let cap = HwDecodeCapability::mid_range();
        assert!(cap.can_handle_resolution(3840, 2160));
        assert!(!cap.can_handle_resolution(7680, 4320));
    }

    #[test]
    fn hw_required_tier_1080p_main() {
        let tier = required_hw_tier("AV1", "Main", 8, 1920, 1080);
        assert_eq!(tier, HwDecoderTier::Entry);
    }

    #[test]
    fn hw_required_tier_4k_hdr() {
        let tier = required_hw_tier("AV1", "Main", 10, 3840, 2160);
        assert_eq!(tier, HwDecoderTier::MidRange);
    }

    #[test]
    fn hw_required_tier_8k_professional() {
        let tier = required_hw_tier("AV1", "Professional", 12, 7680, 4320);
        assert_eq!(tier, HwDecoderTier::HighEnd);
    }
}
