//! SMPTE ST 2110 stream metadata — parameter sets for ST 2110-20 (video),
//! ST 2110-30 (audio) and ST 2110-40 (ancillary data) streams.
//!
//! This module models the metadata that describes each essence type in a
//! SMPTE ST 2110 IP-media infrastructure.  It intentionally avoids any
//! network I/O and can therefore be compiled and tested entirely in isolation.
//!
//! # References
//!
//! * SMPTE ST 2110-20:2017 — *Uncompressed Active Video*
//! * SMPTE ST 2110-30:2017 — *PCM Digital Audio*
//! * SMPTE ST 2110-40:2023 — *Ancillary Data*
//! * SMPTE ST 2110-21:2017 — *Traffic Shaping and Delivery Timing for Video*

use crate::error::{VideoIpError, VideoIpResult};
use serde::{Deserialize, Serialize};
use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Rational frame-rate representation following RFC 4175 / ST 2110-20 syntax
/// (e.g. `60000/1001` for 59.94 fps).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Rational {
    /// Numerator.
    pub num: u32,
    /// Denominator (must be non-zero).
    pub den: u32,
}

impl Rational {
    /// Creates a new rational.
    ///
    /// # Errors
    ///
    /// Returns [`VideoIpError::InvalidVideoConfig`] when `den` is zero.
    pub fn new(num: u32, den: u32) -> VideoIpResult<Self> {
        if den == 0 {
            return Err(VideoIpError::InvalidVideoConfig(
                "frame-rate denominator must be non-zero".to_string(),
            ));
        }
        Ok(Self { num, den })
    }

    /// Converts to a floating-point value.
    #[must_use]
    pub fn to_f64(self) -> f64 {
        f64::from(self.num) / f64::from(self.den)
    }

    // Common frame rates ─────────────────────────────────────────────────────

    /// 24000/1001 ≈ 23.976 fps.
    pub const FPS_23_976: Self = Self { num: 24000, den: 1001 };
    /// 24/1 fps.
    pub const FPS_24: Self = Self { num: 24, den: 1 };
    /// 25/1 fps.
    pub const FPS_25: Self = Self { num: 25, den: 1 };
    /// 30000/1001 ≈ 29.97 fps.
    pub const FPS_29_97: Self = Self { num: 30000, den: 1001 };
    /// 30/1 fps.
    pub const FPS_30: Self = Self { num: 30, den: 1 };
    /// 50/1 fps.
    pub const FPS_50: Self = Self { num: 50, den: 1 };
    /// 60000/1001 ≈ 59.94 fps.
    pub const FPS_59_94: Self = Self { num: 60000, den: 1001 };
    /// 60/1 fps.
    pub const FPS_60: Self = Self { num: 60, den: 1 };
}

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.den == 1 {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{}/{}", self.num, self.den)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ST 2110-20 — Uncompressed Active Video
// ─────────────────────────────────────────────────────────────────────────────

/// Colour sampling structure as defined by ST 2110-20 §7.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Sampling {
    /// 4:4:4 RGB or YCbCr.
    Yuv444,
    /// 4:2:2 YCbCr (most common for broadcast video).
    Yuv422,
    /// 4:2:0 YCbCr.
    Yuv420,
    /// RGB (full colour, no chroma subsampling).
    Rgb,
    /// RGBA (with alpha channel).
    Rgba,
    /// 4:4:4:4 YCbCrA with alpha.
    Yuva4444,
}

impl fmt::Display for Sampling {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Yuv444 => "YCbCr-4:4:4",
            Self::Yuv422 => "YCbCr-4:2:2",
            Self::Yuv420 => "YCbCr-4:2:0",
            Self::Rgb => "RGB",
            Self::Rgba => "RGBA",
            Self::Yuva4444 => "YCbCrA-4:4:4:4",
        };
        f.write_str(s)
    }
}

/// Transfer characteristics (colorimetry) used by ST 2110-20.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Colorimetry {
    /// BT.601 SD.
    Bt601,
    /// BT.709 HD.
    Bt709,
    /// BT.2020 wide colour gamut.
    Bt2020,
    /// DCI-P3 cinema.
    DciP3,
    /// Unknown / unspecified.
    Unknown,
}

impl fmt::Display for Colorimetry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Bt601 => "BT601",
            Self::Bt709 => "BT709",
            Self::Bt2020 => "BT2020",
            Self::DciP3 => "P3D65",
            Self::Unknown => "UNSPECIFIED",
        };
        f.write_str(s)
    }
}

/// Depth (bit-depth per component) for ST 2110-20.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BitDepth {
    /// 8-bit components.
    Depth8,
    /// 10-bit components.
    Depth10,
    /// 12-bit components.
    Depth12,
    /// 16-bit components.
    Depth16,
}

impl fmt::Display for BitDepth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = match self {
            Self::Depth8 => 8u8,
            Self::Depth10 => 10,
            Self::Depth12 => 12,
            Self::Depth16 => 16,
        };
        write!(f, "{n}")
    }
}

/// Scanning mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScanMode {
    /// Progressive scan.
    Progressive,
    /// Interlaced scan (field-based transport).
    Interlaced,
    /// Progressive segmented frame (PsF).
    Psf,
}

impl fmt::Display for ScanMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Progressive => "progressive",
            Self::Interlaced => "interlace",
            Self::Psf => "psf",
        };
        f.write_str(s)
    }
}

/// SMPTE ST 2110-20 video stream parameter set.
///
/// These parameters are carried in the SDP `fmtp` attribute and are needed by
/// a receiver to correctly decode incoming RTP packets.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct St211020Params {
    /// Active picture width in pixels.
    pub width: u32,
    /// Active picture height in lines.
    pub height: u32,
    /// Frame rate (exact rational).
    pub frame_rate: Rational,
    /// Scanning mode.
    pub scan_mode: ScanMode,
    /// Colour sampling structure.
    pub sampling: Sampling,
    /// Bit depth per component.
    pub depth: BitDepth,
    /// Colorimetry / transfer characteristics.
    pub colorimetry: Colorimetry,
    /// Whether to include the RTP timing marker (TP field) in packets.
    pub tp_marker: bool,
    /// Maximum transmission unit in bytes used for RTP packetisation.
    /// Typically 1460 (jumbo-less) or 8960 (jumbo frames).
    pub mtu: u16,
}

impl St211020Params {
    /// Constructs a standard HD 1080p59.94 YCbCr 4:2:2 10-bit parameter set.
    #[must_use]
    pub fn hd_1080p59_94() -> Self {
        Self {
            width: 1920,
            height: 1080,
            frame_rate: Rational::FPS_59_94,
            scan_mode: ScanMode::Progressive,
            sampling: Sampling::Yuv422,
            depth: BitDepth::Depth10,
            colorimetry: Colorimetry::Bt709,
            tp_marker: true,
            mtu: 1460,
        }
    }

    /// Constructs a standard UHD 2160p50 YCbCr 4:2:2 10-bit parameter set.
    #[must_use]
    pub fn uhd_2160p50() -> Self {
        Self {
            width: 3840,
            height: 2160,
            frame_rate: Rational::FPS_50,
            scan_mode: ScanMode::Progressive,
            sampling: Sampling::Yuv422,
            depth: BitDepth::Depth10,
            colorimetry: Colorimetry::Bt2020,
            tp_marker: true,
            mtu: 8960,
        }
    }

    /// Validates that all fields are within the ranges permitted by ST 2110-20.
    ///
    /// # Errors
    ///
    /// Returns [`VideoIpError::InvalidVideoConfig`] on any constraint violation.
    pub fn validate(&self) -> VideoIpResult<()> {
        if self.width == 0 || self.height == 0 {
            return Err(VideoIpError::InvalidVideoConfig(
                "width and height must be non-zero".to_string(),
            ));
        }
        if self.width > 32768 || self.height > 32768 {
            return Err(VideoIpError::InvalidVideoConfig(format!(
                "resolution {}×{} exceeds ST 2110-20 maximum",
                self.width, self.height
            )));
        }
        if self.mtu < 28 {
            // Minimum: 20 (IP) + 8 (UDP) headers
            return Err(VideoIpError::InvalidVideoConfig(
                "MTU must be at least 28 bytes".to_string(),
            ));
        }
        Ok(())
    }

    /// Computes the uncompressed video line length in bytes (one active line).
    ///
    /// Returns the number of bytes required to hold a single active scan-line
    /// at this sampling/depth, ignoring horizontal blanking.
    ///
    /// Returns `None` for currently unsupported sampling/depth combinations.
    #[must_use]
    pub fn line_size_bytes(&self) -> Option<usize> {
        let bits_per_pixel: u32 = match (&self.sampling, &self.depth) {
            (Sampling::Yuv422, BitDepth::Depth10) => 20, // 2 pixels = 40 bits
            (Sampling::Yuv422, BitDepth::Depth8) => 16,
            (Sampling::Yuv422, BitDepth::Depth12) => 24,
            (Sampling::Yuv444, BitDepth::Depth10) => 30,
            (Sampling::Yuv444, BitDepth::Depth8) => 24,
            (Sampling::Rgb, BitDepth::Depth10) => 30,
            (Sampling::Rgb, BitDepth::Depth8) => 24,
            (Sampling::Rgb, BitDepth::Depth12) => 36,
            (Sampling::Rgb, BitDepth::Depth16) => 48,
            _ => return None,
        };
        // Round up to whole bytes
        let total_bits = u64::from(self.width) * u64::from(bits_per_pixel);
        Some(((total_bits + 7) / 8) as usize)
    }

    /// Estimates the uncompressed bitrate in bits per second.
    #[must_use]
    pub fn bitrate_bps(&self) -> Option<u64> {
        let line_bytes = self.line_size_bytes()? as u64;
        let lines = if matches!(self.scan_mode, ScanMode::Interlaced) {
            u64::from(self.height) / 2
        } else {
            u64::from(self.height)
        };
        let frame_bytes = line_bytes * lines;
        let fps = self.frame_rate.to_f64();
        Some((frame_bytes as f64 * fps * 8.0) as u64)
    }

    /// Formats the parameter set as an SDP `fmtp` attribute value, following
    /// RFC 4175 and ST 2110-20.
    ///
    /// Example: `width=1920; height=1080; exactframerate=60000/1001;
    ///            sampling=YCbCr-4:2:2; depth=10; colorimetry=BT709;
    ///            interlace=false`
    #[must_use]
    pub fn to_fmtp(&self) -> String {
        format!(
            "width={}; height={}; exactframerate={}; sampling={}; \
             depth={}; colorimetry={}; {}",
            self.width,
            self.height,
            self.frame_rate,
            self.sampling,
            self.depth,
            self.colorimetry,
            match self.scan_mode {
                ScanMode::Progressive => "interlace=false",
                ScanMode::Interlaced => "interlace=true",
                ScanMode::Psf => "segmented=true",
            }
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ST 2110-30 — PCM Digital Audio
// ─────────────────────────────────────────────────────────────────────────────

/// Permitted sample rates for ST 2110-30.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AudioSampleRate {
    /// 48 000 Hz (professional broadcast standard).
    Hz48000,
    /// 96 000 Hz (extended professional).
    Hz96000,
}

impl fmt::Display for AudioSampleRate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n: u32 = match self {
            Self::Hz48000 => 48_000,
            Self::Hz96000 => 96_000,
        };
        write!(f, "{n}")
    }
}

impl AudioSampleRate {
    /// Converts to a `u32` sample-rate value.
    #[must_use]
    pub const fn to_hz(self) -> u32 {
        match self {
            Self::Hz48000 => 48_000,
            Self::Hz96000 => 96_000,
        }
    }
}

/// Bit depth for ST 2110-30 audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AudioBitDepth {
    /// 16-bit PCM.
    Depth16,
    /// 24-bit PCM (broadcast standard).
    Depth24,
}

impl fmt::Display for AudioBitDepth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n: u8 = match self {
            Self::Depth16 => 16,
            Self::Depth24 => 24,
        };
        write!(f, "{n}")
    }
}

/// SMPTE ST 2110-30 audio stream parameter set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct St211030Params {
    /// Number of audio channels.
    pub channels: u8,
    /// Sample rate.
    pub sample_rate: AudioSampleRate,
    /// Bit depth per sample.
    pub depth: AudioBitDepth,
    /// Number of audio samples packed per RTP packet.
    /// ST 2110-30 mandates exactly 1 ms of audio per packet at 48 kHz ⟹
    /// 48 samples for 48 kHz.
    pub samples_per_packet: u16,
}

impl St211030Params {
    /// Constructs a standard stereo 48 kHz 24-bit parameter set.
    #[must_use]
    pub fn stereo_48k() -> Self {
        Self {
            channels: 2,
            sample_rate: AudioSampleRate::Hz48000,
            depth: AudioBitDepth::Depth24,
            samples_per_packet: 48,
        }
    }

    /// Constructs a standard 16-channel 48 kHz 24-bit parameter set.
    #[must_use]
    pub fn multi_channel_48k() -> Self {
        Self {
            channels: 16,
            sample_rate: AudioSampleRate::Hz48000,
            depth: AudioBitDepth::Depth24,
            samples_per_packet: 48,
        }
    }

    /// Validates against ST 2110-30 constraints.
    ///
    /// # Errors
    ///
    /// Returns [`VideoIpError::InvalidAudioConfig`] on constraint violation.
    pub fn validate(&self) -> VideoIpResult<()> {
        if self.channels == 0 || self.channels > 64 {
            return Err(VideoIpError::InvalidAudioConfig(format!(
                "channel count {} is outside 1..=64",
                self.channels
            )));
        }
        if self.samples_per_packet == 0 {
            return Err(VideoIpError::InvalidAudioConfig(
                "samples_per_packet must be non-zero".to_string(),
            ));
        }
        Ok(())
    }

    /// Computes the uncompressed audio bitrate in bits per second.
    #[must_use]
    pub fn bitrate_bps(&self) -> u64 {
        let depth: u64 = match self.depth {
            AudioBitDepth::Depth16 => 16,
            AudioBitDepth::Depth24 => 24,
        };
        u64::from(self.channels) * u64::from(self.sample_rate.to_hz()) * depth
    }

    /// Computes the byte size of a single RTP payload carrying exactly
    /// `samples_per_packet` frames.
    #[must_use]
    pub fn packet_payload_bytes(&self) -> usize {
        let depth_bytes: usize = match self.depth {
            AudioBitDepth::Depth16 => 2,
            AudioBitDepth::Depth24 => 3,
        };
        usize::from(self.channels) * usize::from(self.samples_per_packet) * depth_bytes
    }

    /// Formats as an SDP `fmtp` attribute value.
    ///
    /// Example: `channel-order=ST; rate=48000`
    #[must_use]
    pub fn to_fmtp(&self) -> String {
        format!(
            "channel-order=SMPTE2110.({}CH); rate={}; depth={}",
            self.channels, self.sample_rate, self.depth
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ST 2110-40 — Ancillary Data
// ─────────────────────────────────────────────────────────────────────────────

/// SMPTE VANC (Vertical Ancillary Data Space) DID/SDID pair identifying the
/// type of ancillary data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AncDataId {
    /// Data Identification word (10-bit, stored as u16).
    pub did: u16,
    /// Secondary Data Identification word (10-bit, stored as u16).
    pub sdid: u16,
}

impl AncDataId {
    /// Creates a new ANC data-ID pair.
    ///
    /// # Errors
    ///
    /// Returns [`VideoIpError::InvalidPacket`] when either value exceeds 10
    /// bits (0x3FF).
    pub fn new(did: u16, sdid: u16) -> VideoIpResult<Self> {
        if did > 0x3FF || sdid > 0x3FF {
            return Err(VideoIpError::InvalidPacket(format!(
                "DID/SDID values must fit in 10 bits, got DID=0x{did:03X} SDID=0x{sdid:03X}"
            )));
        }
        Ok(Self { did, sdid })
    }

    // ── Commonly used ANC types ──────────────────────────────────────────────

    /// SMPTE ST 291M ANC packet (generic).
    pub const ST291: Self = Self { did: 0x41, sdid: 0x01 };
    /// CEA-708 captions (CDP carrier, SMPTE RP 207).
    pub const CEA708_CDP: Self = Self { did: 0x61, sdid: 0x01 };
    /// AFD/Bar data (SMPTE ST 2016-3).
    pub const AFD_BAR: Self = Self { did: 0x41, sdid: 0x05 };
    /// Time code (RP 188/SMPTE ST 12-2).
    pub const TIMECODE: Self = Self { did: 0x60, sdid: 0x60 };
}

impl fmt::Display for AncDataId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DID=0x{:03X}/SDID=0x{:03X}", self.did, self.sdid)
    }
}

/// A single ANC stream declaration within a ST 2110-40 flow.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AncStreamDecl {
    /// The DID/SDID identifying this ANC stream.
    pub id: AncDataId,
    /// Whether this is a HANC (horizontal) vs VANC (vertical) packet.
    pub is_hanc: bool,
    /// Line number(s) in the video frame associated with this ANC.
    /// `None` means "any line" (unspecified placement).
    pub line: Option<u16>,
}

/// SMPTE ST 2110-40 ancillary-data stream parameter set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct St211040Params {
    /// List of ANC stream declarations carried by this flow.
    pub streams: Vec<AncStreamDecl>,
    /// Frame rate of the associated video frame (required for packet pacing).
    pub frame_rate: Rational,
}

impl St211040Params {
    /// Creates a minimal ST 2110-40 parameter set (no streams declared yet).
    #[must_use]
    pub fn new(frame_rate: Rational) -> Self {
        Self {
            streams: Vec::new(),
            frame_rate,
        }
    }

    /// Adds an ANC stream declaration.
    pub fn add_stream(&mut self, decl: AncStreamDecl) {
        self.streams.push(decl);
    }

    /// Validates that at least one stream is declared.
    ///
    /// # Errors
    ///
    /// Returns [`VideoIpError::InvalidPacket`] when no streams are declared.
    pub fn validate(&self) -> VideoIpResult<()> {
        if self.streams.is_empty() {
            return Err(VideoIpError::InvalidPacket(
                "ST 2110-40 flow must declare at least one ANC stream".to_string(),
            ));
        }
        Ok(())
    }

    /// Formats as an SDP `fmtp` attribute value.
    #[must_use]
    pub fn to_fmtp(&self) -> String {
        let anc_list: Vec<String> = self
            .streams
            .iter()
            .map(|s| {
                let space = if s.is_hanc { "HANC" } else { "VANC" };
                let line = s
                    .line
                    .map(|l| format!(" line={l}"))
                    .unwrap_or_default();
                format!("DID_SDID={{0x{:02X},0x{:02X}}} {}{}", s.id.did, s.id.sdid, space, line)
            })
            .collect();
        format!(
            "exactframerate={}; {}",
            self.frame_rate,
            anc_list.join("; ")
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unified ST 2110 stream descriptor
// ─────────────────────────────────────────────────────────────────────────────

/// Identifies which ST 2110 essence type a stream carries.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum St2110EssenceType {
    /// Uncompressed video (ST 2110-20).
    Video(St211020Params),
    /// PCM audio (ST 2110-30).
    Audio(St211030Params),
    /// Ancillary data (ST 2110-40).
    Ancillary(St211040Params),
}

/// A fully described ST 2110 stream, combining network addressing metadata
/// with essence-type parameters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct St2110StreamDescriptor {
    /// Human-readable name for this flow (e.g. "CAM-01 VIDEO").
    pub name: String,
    /// Flow identifier (UUID-style, stored as a 128-bit value).
    pub flow_id: u128,
    /// Multicast (or unicast) destination IP address as a string.
    pub destination_ip: String,
    /// Destination UDP port.
    pub destination_port: u16,
    /// RTP payload type number assigned to this flow.
    pub payload_type: u8,
    /// Essence-type specific parameters.
    pub essence: St2110EssenceType,
}

impl St2110StreamDescriptor {
    /// Creates a new stream descriptor.
    ///
    /// # Errors
    ///
    /// Returns [`VideoIpError::InvalidPacket`] when `name` is empty.
    pub fn new(
        name: impl Into<String>,
        flow_id: u128,
        destination_ip: impl Into<String>,
        destination_port: u16,
        payload_type: u8,
        essence: St2110EssenceType,
    ) -> VideoIpResult<Self> {
        let name = name.into();
        if name.is_empty() {
            return Err(VideoIpError::InvalidPacket(
                "stream name must not be empty".to_string(),
            ));
        }
        Ok(Self {
            name,
            flow_id,
            destination_ip: destination_ip.into(),
            destination_port,
            payload_type,
            essence,
        })
    }

    /// Validates the descriptor and its essence parameters.
    ///
    /// # Errors
    ///
    /// Returns an error when any parameter is out of range.
    pub fn validate(&self) -> VideoIpResult<()> {
        if self.destination_port == 0 {
            return Err(VideoIpError::InvalidPacket(
                "destination port must be non-zero".to_string(),
            ));
        }
        match &self.essence {
            St2110EssenceType::Video(p) => p.validate(),
            St2110EssenceType::Audio(p) => p.validate(),
            St2110EssenceType::Ancillary(p) => p.validate(),
        }
    }

    /// Returns a brief one-line summary of this descriptor.
    #[must_use]
    pub fn summary(&self) -> String {
        let kind = match &self.essence {
            St2110EssenceType::Video(p) => format!(
                "ST2110-20 {}×{} @ {} {}",
                p.width, p.height, p.frame_rate, p.sampling
            ),
            St2110EssenceType::Audio(p) => format!(
                "ST2110-30 {}ch {}Hz {}b",
                p.channels, p.sample_rate, p.depth
            ),
            St2110EssenceType::Ancillary(p) => format!(
                "ST2110-40 {} ANC streams @ {}fps",
                p.streams.len(),
                p.frame_rate
            ),
        };
        format!(
            "{} [{}→{}:{} PT={}]",
            kind, self.name, self.destination_ip, self.destination_port, self.payload_type
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Rational ─────────────────────────────────────────────────────────────

    #[test]
    fn test_rational_zero_den_error() {
        assert!(Rational::new(30000, 0).is_err());
    }

    #[test]
    fn test_rational_display_integer() {
        let r = Rational { num: 50, den: 1 };
        assert_eq!(r.to_string(), "50");
    }

    #[test]
    fn test_rational_display_fraction() {
        let r = Rational::FPS_59_94;
        assert_eq!(r.to_string(), "60000/1001");
    }

    // ── ST 2110-20 ────────────────────────────────────────────────────────────

    #[test]
    fn test_st2110_20_validate_ok() {
        let p = St211020Params::hd_1080p59_94();
        assert!(p.validate().is_ok());
    }

    #[test]
    fn test_st2110_20_validate_zero_size() {
        let mut p = St211020Params::hd_1080p59_94();
        p.width = 0;
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_st2110_20_line_size_bytes_1080p_yuv422_10b() {
        let p = St211020Params::hd_1080p59_94(); // 1920×1080 YCbCr 4:2:2 10-bit
        // 1920 pixels × 20 bits/pixel = 38400 bits → 4800 bytes
        assert_eq!(p.line_size_bytes(), Some(4800));
    }

    #[test]
    fn test_st2110_20_bitrate_estimate_reasonable() {
        let p = St211020Params::hd_1080p59_94();
        let bps = p.bitrate_bps().expect("should return a bitrate");
        // Rough sanity: 1080p60 YCbCr 4:2:2 10-bit ≈ 2.5 Gbit/s
        assert!(bps > 2_000_000_000, "bitrate {bps} too low");
        assert!(bps < 4_000_000_000, "bitrate {bps} too high");
    }

    #[test]
    fn test_st2110_20_fmtp_contains_fields() {
        let p = St211020Params::hd_1080p59_94();
        let fmtp = p.to_fmtp();
        assert!(fmtp.contains("width=1920"));
        assert!(fmtp.contains("height=1080"));
        assert!(fmtp.contains("YCbCr-4:2:2"));
    }

    // ── ST 2110-30 ────────────────────────────────────────────────────────────

    #[test]
    fn test_st2110_30_validate_ok() {
        assert!(St211030Params::stereo_48k().validate().is_ok());
    }

    #[test]
    fn test_st2110_30_validate_zero_channels() {
        let mut p = St211030Params::stereo_48k();
        p.channels = 0;
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_st2110_30_bitrate_stereo_48k_24b() {
        let p = St211030Params::stereo_48k();
        // 2 ch × 48000 × 24 bit = 2 304 000 bps
        assert_eq!(p.bitrate_bps(), 2_304_000);
    }

    #[test]
    fn test_st2110_30_packet_payload_stereo() {
        let p = St211030Params::stereo_48k();
        // 2 channels × 48 samples × 3 bytes = 288 bytes per packet
        assert_eq!(p.packet_payload_bytes(), 288);
    }

    #[test]
    fn test_st2110_30_fmtp_contains_rate() {
        let p = St211030Params::stereo_48k();
        assert!(p.to_fmtp().contains("48000"));
    }

    // ── ST 2110-40 ────────────────────────────────────────────────────────────

    #[test]
    fn test_st2110_40_validate_empty_error() {
        let p = St211040Params::new(Rational::FPS_60);
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_st2110_40_validate_with_stream_ok() {
        let mut p = St211040Params::new(Rational::FPS_60);
        p.add_stream(AncStreamDecl {
            id: AncDataId::CEA708_CDP,
            is_hanc: false,
            line: Some(10),
        });
        assert!(p.validate().is_ok());
    }

    #[test]
    fn test_anc_data_id_10bit_overflow() {
        assert!(AncDataId::new(0x400, 0x00).is_err());
        assert!(AncDataId::new(0x00, 0x400).is_err());
        assert!(AncDataId::new(0x3FF, 0x3FF).is_ok());
    }

    // ── Unified descriptor ────────────────────────────────────────────────────

    #[test]
    fn test_stream_descriptor_video_roundtrip() {
        let params = St211020Params::hd_1080p59_94();
        let desc = St2110StreamDescriptor::new(
            "CAM-01",
            0xDEAD_BEEF_CAFE_BABE_0000_0000_0000_0001u128,
            "239.0.0.1",
            5004,
            96,
            St2110EssenceType::Video(params),
        )
        .unwrap();
        assert!(desc.validate().is_ok());
        let summary = desc.summary();
        assert!(summary.contains("ST2110-20"));
        assert!(summary.contains("CAM-01"));
    }

    #[test]
    fn test_stream_descriptor_empty_name_error() {
        let params = St211020Params::hd_1080p59_94();
        assert!(St2110StreamDescriptor::new(
            "",
            1,
            "239.0.0.1",
            5004,
            96,
            St2110EssenceType::Video(params)
        )
        .is_err());
    }

    #[test]
    fn test_stream_descriptor_zero_port_error() {
        let params = St211030Params::stereo_48k();
        let desc = St2110StreamDescriptor::new(
            "AUD-01",
            2,
            "239.0.0.2",
            0, // invalid
            97,
            St2110EssenceType::Audio(params),
        )
        .unwrap();
        assert!(desc.validate().is_err());
    }
}
