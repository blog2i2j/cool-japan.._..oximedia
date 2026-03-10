//! Audio codec metadata and registry.
//!
//! This module provides a registry of known audio codecs with their properties,
//! capabilities, and constraints.

/// Known audio codec types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AudioCodec {
    /// Uncompressed PCM audio.
    Pcm,
    /// MPEG-1/2 Layer III.
    Mp3,
    /// Advanced Audio Coding.
    Aac,
    /// Opus – modern codec for speech and music.
    Opus,
    /// Ogg Vorbis.
    Vorbis,
    /// Free Lossless Audio Codec.
    Flac,
    /// Dolby Digital (AC-3).
    Ac3,
    /// Dolby Digital Plus (Enhanced AC-3).
    Eac3,
    /// DTS Coherent Acoustics.
    Dts,
    /// Dolby TrueHD lossless.
    TrueHd,
    /// Apple Lossless Audio Codec.
    Alac,
    /// Windows Media Audio.
    Wma,
}

/// Metadata describing the capabilities of a single audio codec.
#[derive(Debug, Clone)]
pub struct CodecInfo {
    /// The codec variant.
    pub codec: AudioCodec,
    /// Human-readable name.
    pub name: &'static str,
    /// Whether the codec is lossy (`true`) or lossless (`false`).
    pub lossy: bool,
    /// Maximum bitrate in kilobits per second.
    pub max_bitrate_kbps: u32,
    /// Maximum number of audio channels supported.
    pub max_channels: u8,
    /// Maximum sample rate in Hz.
    pub max_sample_rate: u32,
}

/// Return [`CodecInfo`] for the given codec.
pub fn codec_info(codec: AudioCodec) -> CodecInfo {
    match codec {
        AudioCodec::Pcm => CodecInfo {
            codec,
            name: "PCM",
            lossy: false,
            max_bitrate_kbps: 4608,
            max_channels: 32,
            max_sample_rate: 192_000,
        },
        AudioCodec::Mp3 => CodecInfo {
            codec,
            name: "MP3",
            lossy: true,
            max_bitrate_kbps: 320,
            max_channels: 2,
            max_sample_rate: 48_000,
        },
        AudioCodec::Aac => CodecInfo {
            codec,
            name: "AAC",
            lossy: true,
            max_bitrate_kbps: 512,
            max_channels: 48,
            max_sample_rate: 96_000,
        },
        AudioCodec::Opus => CodecInfo {
            codec,
            name: "Opus",
            lossy: true,
            max_bitrate_kbps: 510,
            max_channels: 255,
            max_sample_rate: 48_000,
        },
        AudioCodec::Vorbis => CodecInfo {
            codec,
            name: "Vorbis",
            lossy: true,
            max_bitrate_kbps: 500,
            max_channels: 255,
            max_sample_rate: 192_000,
        },
        AudioCodec::Flac => CodecInfo {
            codec,
            name: "FLAC",
            lossy: false,
            max_bitrate_kbps: 4608,
            max_channels: 8,
            max_sample_rate: 655_350,
        },
        AudioCodec::Ac3 => CodecInfo {
            codec,
            name: "AC-3",
            lossy: true,
            max_bitrate_kbps: 640,
            max_channels: 6,
            max_sample_rate: 48_000,
        },
        AudioCodec::Eac3 => CodecInfo {
            codec,
            name: "E-AC-3",
            lossy: true,
            max_bitrate_kbps: 6144,
            max_channels: 16,
            max_sample_rate: 48_000,
        },
        AudioCodec::Dts => CodecInfo {
            codec,
            name: "DTS",
            lossy: true,
            max_bitrate_kbps: 1509,
            max_channels: 8,
            max_sample_rate: 96_000,
        },
        AudioCodec::TrueHd => CodecInfo {
            codec,
            name: "TrueHD",
            lossy: false,
            max_bitrate_kbps: 18_000,
            max_channels: 14,
            max_sample_rate: 192_000,
        },
        AudioCodec::Alac => CodecInfo {
            codec,
            name: "ALAC",
            lossy: false,
            max_bitrate_kbps: 4608,
            max_channels: 8,
            max_sample_rate: 384_000,
        },
        AudioCodec::Wma => CodecInfo {
            codec,
            name: "WMA",
            lossy: true,
            max_bitrate_kbps: 768,
            max_channels: 8,
            max_sample_rate: 96_000,
        },
    }
}

/// Return `true` if the codec is lossless.
pub fn is_lossless(codec: AudioCodec) -> bool {
    !codec_info(codec).lossy
}

/// Return `true` if the codec supports at least `channels` channels.
pub fn supports_channels(codec: AudioCodec, channels: u8) -> bool {
    codec_info(codec).max_channels >= channels
}

/// Registry of all known codecs, queryable by name or capability.
pub struct CodecRegistry {
    codecs: Vec<CodecInfo>,
}

impl CodecRegistry {
    /// Build a registry containing all known codecs.
    pub fn new() -> Self {
        let all = [
            AudioCodec::Pcm,
            AudioCodec::Mp3,
            AudioCodec::Aac,
            AudioCodec::Opus,
            AudioCodec::Vorbis,
            AudioCodec::Flac,
            AudioCodec::Ac3,
            AudioCodec::Eac3,
            AudioCodec::Dts,
            AudioCodec::TrueHd,
            AudioCodec::Alac,
            AudioCodec::Wma,
        ];
        Self {
            codecs: all.iter().map(|c| codec_info(*c)).collect(),
        }
    }

    /// Look up a codec by its name (case-insensitive).
    pub fn by_name(&self, name: &str) -> Option<&CodecInfo> {
        let lower = name.to_lowercase();
        self.codecs.iter().find(|c| c.name.to_lowercase() == lower)
    }

    /// Return all lossless codecs in the registry.
    pub fn lossless_codecs(&self) -> Vec<&CodecInfo> {
        self.codecs.iter().filter(|c| !c.lossy).collect()
    }

    /// Return all codecs whose maximum bitrate is at least `kbps`.
    pub fn codecs_for_bitrate(&self, kbps: u32) -> Vec<&CodecInfo> {
        self.codecs
            .iter()
            .filter(|c| c.max_bitrate_kbps >= kbps)
            .collect()
    }

    /// Return all codecs in the registry.
    pub fn all(&self) -> &[CodecInfo] {
        &self.codecs
    }
}

impl Default for CodecRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pcm_is_lossless() {
        assert!(is_lossless(AudioCodec::Pcm));
    }

    #[test]
    fn test_mp3_is_lossy() {
        assert!(!is_lossless(AudioCodec::Mp3));
    }

    #[test]
    fn test_flac_is_lossless() {
        assert!(is_lossless(AudioCodec::Flac));
    }

    #[test]
    fn test_truehd_is_lossless() {
        assert!(is_lossless(AudioCodec::TrueHd));
    }

    #[test]
    fn test_alac_is_lossless() {
        assert!(is_lossless(AudioCodec::Alac));
    }

    #[test]
    fn test_supports_channels_opus() {
        // Opus supports up to 255 channels
        assert!(supports_channels(AudioCodec::Opus, 8));
        assert!(supports_channels(AudioCodec::Opus, 255));
    }

    #[test]
    fn test_mp3_max_two_channels() {
        assert!(supports_channels(AudioCodec::Mp3, 2));
        assert!(!supports_channels(AudioCodec::Mp3, 3));
    }

    #[test]
    fn test_registry_by_name() {
        let reg = CodecRegistry::new();
        let info = reg.by_name("opus").expect("should succeed");
        assert_eq!(info.name, "Opus");
    }

    #[test]
    fn test_registry_by_name_case_insensitive() {
        let reg = CodecRegistry::new();
        assert!(reg.by_name("FLAC").is_some());
        assert!(reg.by_name("flac").is_some());
    }

    #[test]
    fn test_registry_by_name_missing() {
        let reg = CodecRegistry::new();
        assert!(reg.by_name("nonexistent").is_none());
    }

    #[test]
    fn test_lossless_codecs_count() {
        let reg = CodecRegistry::new();
        let lossless = reg.lossless_codecs();
        // PCM, FLAC, TrueHD, ALAC = 4
        assert_eq!(lossless.len(), 4);
    }

    #[test]
    fn test_codecs_for_bitrate_high() {
        let reg = CodecRegistry::new();
        // Only TrueHD (18000) and E-AC-3 (6144) exceed 5000 kbps
        let result = reg.codecs_for_bitrate(5000);
        assert!(!result.is_empty());
        for c in &result {
            assert!(c.max_bitrate_kbps >= 5000);
        }
    }

    #[test]
    fn test_codec_info_name_pcm() {
        let info = codec_info(AudioCodec::Pcm);
        assert_eq!(info.name, "PCM");
    }

    #[test]
    fn test_codec_info_aac_bitrate() {
        let info = codec_info(AudioCodec::Aac);
        assert_eq!(info.max_bitrate_kbps, 512);
    }

    #[test]
    fn test_registry_all_count() {
        let reg = CodecRegistry::new();
        assert_eq!(reg.all().len(), 12);
    }
}
