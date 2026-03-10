#![allow(dead_code)]
//! Sample entry / codec configuration record parsing for ISO BMFF containers.
//!
//! Parses the `SampleEntry` boxes that describe the encoding parameters of each
//! track in an MP4/MOV container. Covers video (`avc1`/`hev1`/`av01`) and audio
//! (`mp4a`/`Opus`/`fLaC`) sample entries.

use std::fmt;

/// Codec family encoded in a sample entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SampleEntryCodec {
    /// H.264 / AVC.
    Avc,
    /// H.265 / HEVC.
    Hevc,
    /// AV1.
    Av1,
    /// VP9.
    Vp9,
    /// AAC (MPEG-4 Audio).
    Aac,
    /// Opus.
    Opus,
    /// FLAC.
    Flac,
    /// Unknown / unsupported codec.
    Unknown([u8; 4]),
}

impl SampleEntryCodec {
    /// Creates a codec variant from a four-character code.
    #[must_use]
    pub fn from_fourcc(cc: [u8; 4]) -> Self {
        match &cc {
            b"avc1" | b"avc3" => Self::Avc,
            b"hev1" | b"hvc1" => Self::Hevc,
            b"av01" => Self::Av1,
            b"vp09" => Self::Vp9,
            b"mp4a" => Self::Aac,
            b"Opus" => Self::Opus,
            b"fLaC" => Self::Flac,
            _ => Self::Unknown(cc),
        }
    }

    /// Returns `true` for video codecs.
    #[must_use]
    pub fn is_video(self) -> bool {
        matches!(self, Self::Avc | Self::Hevc | Self::Av1 | Self::Vp9)
    }

    /// Returns `true` for audio codecs.
    #[must_use]
    pub fn is_audio(self) -> bool {
        matches!(self, Self::Aac | Self::Opus | Self::Flac)
    }
}

impl fmt::Display for SampleEntryCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Avc => "AVC/H.264",
            Self::Hevc => "HEVC/H.265",
            Self::Av1 => "AV1",
            Self::Vp9 => "VP9",
            Self::Aac => "AAC",
            Self::Opus => "Opus",
            Self::Flac => "FLAC",
            Self::Unknown(_) => "Unknown",
        };
        f.write_str(label)
    }
}

/// Parsed video sample entry parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoSampleEntry {
    /// Codec family.
    pub codec: SampleEntryCodec,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Bit depth (usually 8, 10, or 12).
    pub bit_depth: u8,
    /// Raw codec-specific configuration data (e.g. SPS/PPS for AVC).
    pub config_data: Vec<u8>,
}

impl VideoSampleEntry {
    /// Creates a new video sample entry.
    #[must_use]
    pub fn new(codec: SampleEntryCodec, width: u32, height: u32, bit_depth: u8) -> Self {
        Self {
            codec,
            width,
            height,
            bit_depth,
            config_data: Vec::new(),
        }
    }

    /// Returns the number of pixels (width * height).
    #[must_use]
    pub fn pixel_count(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Builder: attaches codec-specific configuration bytes.
    #[must_use]
    pub fn with_config(mut self, data: Vec<u8>) -> Self {
        self.config_data = data;
        self
    }

    /// Returns `true` if configuration data has been set.
    #[must_use]
    pub fn has_config(&self) -> bool {
        !self.config_data.is_empty()
    }
}

/// Parsed audio sample entry parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioSampleEntry {
    /// Codec family.
    pub codec: SampleEntryCodec,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of audio channels.
    pub channels: u16,
    /// Bits per sample.
    pub bits_per_sample: u16,
    /// Raw codec-specific configuration data.
    pub config_data: Vec<u8>,
}

impl AudioSampleEntry {
    /// Creates a new audio sample entry.
    #[must_use]
    pub fn new(
        codec: SampleEntryCodec,
        sample_rate: u32,
        channels: u16,
        bits_per_sample: u16,
    ) -> Self {
        Self {
            codec,
            sample_rate,
            channels,
            bits_per_sample,
            config_data: Vec::new(),
        }
    }

    /// Builder: attaches codec-specific configuration bytes.
    #[must_use]
    pub fn with_config(mut self, data: Vec<u8>) -> Self {
        self.config_data = data;
        self
    }

    /// Returns `true` if configuration data has been set.
    #[must_use]
    pub fn has_config(&self) -> bool {
        !self.config_data.is_empty()
    }

    /// Computes raw PCM bitrate for uncompressed audio (informational).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn pcm_bitrate_kbps(&self) -> f64 {
        (u64::from(self.sample_rate) * u64::from(self.channels) * u64::from(self.bits_per_sample))
            as f64
            / 1000.0
    }
}

/// A unified sample entry that can be either video or audio.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SampleEntry {
    /// Video sample entry.
    Video(VideoSampleEntry),
    /// Audio sample entry.
    Audio(AudioSampleEntry),
}

impl SampleEntry {
    /// Returns the codec family regardless of media type.
    #[must_use]
    pub fn codec(&self) -> SampleEntryCodec {
        match self {
            Self::Video(v) => v.codec,
            Self::Audio(a) => a.codec,
        }
    }

    /// Returns `true` for video entries.
    #[must_use]
    pub fn is_video(&self) -> bool {
        matches!(self, Self::Video(_))
    }

    /// Returns `true` for audio entries.
    #[must_use]
    pub fn is_audio(&self) -> bool {
        matches!(self, Self::Audio(_))
    }
}

/// Tries to parse a sample entry from a raw box payload.
///
/// `fourcc` is the 4-byte box type that identifies the codec.
/// `data` is the payload after the standard 8-byte sample entry header.
///
/// Returns `None` if the data is too short for a valid sample entry.
#[must_use]
pub fn parse_sample_entry(fourcc: [u8; 4], data: &[u8]) -> Option<SampleEntry> {
    let codec = SampleEntryCodec::from_fourcc(fourcc);
    if codec.is_video() {
        // Minimum video sample entry: 6 reserved + 2 data_ref + 16 pre-defined + 4 w/h + ...
        if data.len() < 28 {
            return None;
        }
        let width = u16::from_be_bytes([data[24], data[25]]);
        let height = u16::from_be_bytes([data[26], data[27]]);
        let entry = VideoSampleEntry::new(codec, u32::from(width), u32::from(height), 8);
        Some(SampleEntry::Video(entry))
    } else if codec.is_audio() {
        // Minimum audio sample entry: 6 reserved + 2 data_ref + 8 pre-defined + 2 ch + 2 bps + 4 sr
        if data.len() < 24 {
            return None;
        }
        let channels = u16::from_be_bytes([data[16], data[17]]);
        let bps = u16::from_be_bytes([data[18], data[19]]);
        let sr_fixed = u32::from_be_bytes([data[20], data[21], data[22], data[23]]);
        let sample_rate = sr_fixed >> 16; // fixed-point 16.16
        let entry = AudioSampleEntry::new(codec, sample_rate, channels, bps);
        Some(SampleEntry::Audio(entry))
    } else {
        None
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    // 1. codec from known fourcc
    #[test]
    fn test_codec_from_fourcc_avc() {
        assert_eq!(
            SampleEntryCodec::from_fourcc(*b"avc1"),
            SampleEntryCodec::Avc
        );
    }

    // 2. codec from unknown fourcc
    #[test]
    fn test_codec_from_fourcc_unknown() {
        let cc = *b"xxxx";
        assert_eq!(
            SampleEntryCodec::from_fourcc(cc),
            SampleEntryCodec::Unknown(cc)
        );
    }

    // 3. is_video
    #[test]
    fn test_is_video() {
        assert!(SampleEntryCodec::Avc.is_video());
        assert!(SampleEntryCodec::Av1.is_video());
        assert!(!SampleEntryCodec::Aac.is_video());
    }

    // 4. is_audio
    #[test]
    fn test_is_audio() {
        assert!(SampleEntryCodec::Aac.is_audio());
        assert!(SampleEntryCodec::Opus.is_audio());
        assert!(!SampleEntryCodec::Hevc.is_audio());
    }

    // 5. video sample entry pixel count
    #[test]
    fn test_video_pixel_count() {
        let v = VideoSampleEntry::new(SampleEntryCodec::Av1, 1920, 1080, 10);
        assert_eq!(v.pixel_count(), 1920 * 1080);
    }

    // 6. video with_config
    #[test]
    fn test_video_with_config() {
        let v = VideoSampleEntry::new(SampleEntryCodec::Avc, 640, 480, 8)
            .with_config(vec![0x67, 0x42, 0x00, 0x1e]);
        assert!(v.has_config());
        assert_eq!(v.config_data.len(), 4);
    }

    // 7. audio pcm_bitrate_kbps
    #[test]
    fn test_audio_pcm_bitrate() {
        let a = AudioSampleEntry::new(SampleEntryCodec::Flac, 48000, 2, 16);
        // 48000 * 2 * 16 = 1_536_000 / 1000 = 1536.0
        assert!((a.pcm_bitrate_kbps() - 1536.0).abs() < f64::EPSILON);
    }

    // 8. audio has_config default false
    #[test]
    fn test_audio_no_config() {
        let a = AudioSampleEntry::new(SampleEntryCodec::Opus, 48000, 2, 16);
        assert!(!a.has_config());
    }

    // 9. SampleEntry codec accessor
    #[test]
    fn test_sample_entry_codec() {
        let se = SampleEntry::Video(VideoSampleEntry::new(
            SampleEntryCodec::Hevc,
            3840,
            2160,
            10,
        ));
        assert_eq!(se.codec(), SampleEntryCodec::Hevc);
    }

    // 10. SampleEntry is_video / is_audio
    #[test]
    fn test_sample_entry_variant() {
        let v = SampleEntry::Video(VideoSampleEntry::new(SampleEntryCodec::Av1, 1280, 720, 8));
        assert!(v.is_video());
        assert!(!v.is_audio());
        let a = SampleEntry::Audio(AudioSampleEntry::new(SampleEntryCodec::Aac, 44100, 2, 16));
        assert!(a.is_audio());
    }

    // 11. parse_sample_entry video
    #[test]
    fn test_parse_video_entry() {
        let mut data = vec![0u8; 30];
        // width at offset 24-25, height at 26-27
        data[24] = 0x07;
        data[25] = 0x80; // 1920
        data[26] = 0x04;
        data[27] = 0x38; // 1080
        let result = parse_sample_entry(*b"avc1", &data).expect("operation should succeed");
        assert!(result.is_video());
        if let SampleEntry::Video(v) = result {
            assert_eq!(v.width, 1920);
            assert_eq!(v.height, 1080);
        }
    }

    // 12. parse_sample_entry audio
    #[test]
    fn test_parse_audio_entry() {
        let mut data = vec![0u8; 24];
        // channels at offset 16-17
        data[16] = 0x00;
        data[17] = 0x02; // 2 channels
                         // bits per sample at 18-19
        data[18] = 0x00;
        data[19] = 0x10; // 16 bits
                         // sample rate (fixed 16.16) at 20-23: 48000 << 16
        let sr_fixed = 48000u32 << 16;
        data[20..24].copy_from_slice(&sr_fixed.to_be_bytes());
        let result = parse_sample_entry(*b"mp4a", &data).expect("operation should succeed");
        assert!(result.is_audio());
        if let SampleEntry::Audio(a) = result {
            assert_eq!(a.sample_rate, 48000);
            assert_eq!(a.channels, 2);
        }
    }

    // 13. parse_sample_entry too short
    #[test]
    fn test_parse_too_short() {
        assert!(parse_sample_entry(*b"avc1", &[0; 10]).is_none());
    }

    // 14. parse_sample_entry unknown codec returns None
    #[test]
    fn test_parse_unknown_codec() {
        assert!(parse_sample_entry(*b"xxxx", &[0; 30]).is_none());
    }

    // 15. codec display
    #[test]
    fn test_codec_display() {
        assert_eq!(format!("{}", SampleEntryCodec::Av1), "AV1");
        assert_eq!(format!("{}", SampleEntryCodec::Opus), "Opus");
    }

    // 16. hevc alternate fourcc
    #[test]
    fn test_hevc_alternate_fourcc() {
        assert_eq!(
            SampleEntryCodec::from_fourcc(*b"hvc1"),
            SampleEntryCodec::Hevc
        );
    }
}
