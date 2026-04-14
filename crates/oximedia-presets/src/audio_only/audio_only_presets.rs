//! `AudioOnlyPresets` — a unified facade for FLAC and Opus audio-only presets.
//!
//! This module exposes a zero-size struct [`AudioOnlyPresets`] whose associated
//! functions return individual [`Preset`] values for the most common audio-only
//! distribution scenarios.  All returned presets have:
//!
//! - `video_codec = None` and `video_bitrate = None` (no video track)
//! - A pure-Rust patent-free codec (FLAC or Opus)
//! - An `"audio-only"` tag for easy catalogue filtering
//!
//! # Preset overview
//!
//! | Method                | Codec | Bitrate   | Channels | Use case                     |
//! |-----------------------|-------|-----------|----------|------------------------------|
//! | `podcast_opus`        | Opus  | 64 kbps   | Mono     | Solo podcast / interview     |
//! | `podcast_flac`        | FLAC  | 1411 kbps | Stereo   | Lossless podcast master      |
//! | `music_opus_hq`       | Opus  | 192 kbps  | Stereo   | Premium music streaming      |
//! | `music_flac_hd`       | FLAC  | 4608 kbps | Stereo   | Hi-Res 24-bit music master   |
//! | `audiobook_opus`      | Opus  | 32 kbps   | Mono     | Data-efficient audiobook     |

use crate::{Preset, PresetCategory, PresetMetadata};
use oximedia_transcode::{audio_channel_map::AudioLayout, PresetConfig, QualityMode};

/// Facade providing named constructors for audio-only distribution presets.
///
/// All methods are free functions (no `self`) — use them as
/// `AudioOnlyPresets::podcast_opus()`.
pub struct AudioOnlyPresets;

impl AudioOnlyPresets {
    /// Opus 64 kbps mono — optimised for solo podcast and interview delivery.
    ///
    /// 64 kbps mono Opus is transparent for speech and produces files roughly
    /// six times smaller than 128 kbps MP3.  Mono eliminates phase issues
    /// common with single-microphone recordings.
    #[must_use]
    pub fn podcast_opus() -> Preset {
        let metadata = PresetMetadata::new(
            "audio-only-podcast-opus",
            "Podcast Opus 64k Mono",
            PresetCategory::Codec("Opus".to_string()),
        )
        .with_description(
            "Opus 64 kbps mono for solo and interview podcasts — speech-optimised, patent-free",
        )
        .with_target("Podcast Distribution")
        .with_tag("opus")
        .with_tag("podcast")
        .with_tag("speech")
        .with_tag("mono")
        .with_tag("audio-only")
        .with_tag("64kbps");

        let config = PresetConfig {
            video_codec: None,
            audio_codec: Some("opus".to_string()),
            video_bitrate: None,
            audio_bitrate: Some(64_000),
            width: None,
            height: None,
            frame_rate: None,
            quality_mode: Some(QualityMode::Medium),
            container: Some("ogg".to_string()),
            audio_channel_layout: Some(AudioLayout::Mono),
        };

        Preset::new(metadata, config)
    }

    /// FLAC lossless stereo — 44.1 kHz / 16-bit podcast production master.
    ///
    /// Preserves every bit of the original recording before any lossy encode.
    /// Compatible with all podcast DAWs and editing tools that accept FLAC.
    /// Uncompressed PCM equivalent at 1 411 200 bit/s (44 100 × 16 × 2).
    #[must_use]
    pub fn podcast_flac() -> Preset {
        let metadata = PresetMetadata::new(
            "audio-only-podcast-flac",
            "Podcast FLAC Lossless Stereo",
            PresetCategory::Codec("FLAC".to_string()),
        )
        .with_description(
            "Lossless FLAC stereo at 44.1 kHz / 16-bit for podcast pre-master archival",
        )
        .with_target("Podcast Production")
        .with_tag("flac")
        .with_tag("podcast")
        .with_tag("lossless")
        .with_tag("stereo")
        .with_tag("audio-only")
        .with_tag("44100");

        let config = PresetConfig {
            video_codec: None,
            audio_codec: Some("flac".to_string()),
            video_bitrate: None,
            // 44100 Hz × 16 bits × 2 channels = 1 411 200 bit/s
            audio_bitrate: Some(1_411_200),
            width: None,
            height: None,
            frame_rate: None,
            quality_mode: Some(QualityMode::VeryHigh),
            container: Some("flac".to_string()),
            audio_channel_layout: Some(AudioLayout::Stereo),
        };

        Preset::new(metadata, config)
    }

    /// Opus 192 kbps stereo — near-lossless high-quality music streaming.
    ///
    /// Virtually indistinguishable from the lossless source in double-blind
    /// listening tests.  Recommended for music distribution services targeting
    /// listeners who require transparent quality without the storage cost of FLAC.
    #[must_use]
    pub fn music_opus_hq() -> Preset {
        let metadata = PresetMetadata::new(
            "audio-only-music-opus-hq",
            "Music Opus HQ 192k Stereo",
            PresetCategory::Codec("Opus".to_string()),
        )
        .with_description(
            "Near-lossless Opus at 192 kbps stereo for premium music distribution",
        )
        .with_target("Music Distribution")
        .with_tag("opus")
        .with_tag("music")
        .with_tag("stereo")
        .with_tag("audio-only")
        .with_tag("192kbps")
        .with_tag("hq");

        let config = PresetConfig {
            video_codec: None,
            audio_codec: Some("opus".to_string()),
            video_bitrate: None,
            audio_bitrate: Some(192_000),
            width: None,
            height: None,
            frame_rate: None,
            quality_mode: Some(QualityMode::High),
            container: Some("ogg".to_string()),
            audio_channel_layout: Some(AudioLayout::Stereo),
        };

        Preset::new(metadata, config)
    }

    /// FLAC 24-bit hi-res stereo — 96 kHz / 24-bit music master.
    ///
    /// The most common hi-res audio format sold by Qobuz, HDtracks, and Apple
    /// Music lossless.  Provides mastering-grade headroom for downstream
    /// processing and future-proof archival at 4 608 000 bit/s (96 000 × 24 × 2).
    #[must_use]
    pub fn music_flac_hd() -> Preset {
        let metadata = PresetMetadata::new(
            "audio-only-music-flac-hd",
            "Music FLAC HD 96kHz/24-bit Stereo",
            PresetCategory::Codec("FLAC".to_string()),
        )
        .with_description(
            "Hi-Res FLAC at 96 kHz / 24-bit stereo for premium lossless music distribution",
        )
        .with_target("Hi-Res Music Distribution")
        .with_tag("flac")
        .with_tag("music")
        .with_tag("lossless")
        .with_tag("stereo")
        .with_tag("audio-only")
        .with_tag("hires")
        .with_tag("96000");

        let config = PresetConfig {
            video_codec: None,
            audio_codec: Some("flac".to_string()),
            video_bitrate: None,
            // 96000 Hz × 24 bits × 2 channels = 4 608 000 bit/s
            audio_bitrate: Some(4_608_000),
            width: None,
            height: None,
            frame_rate: None,
            quality_mode: Some(QualityMode::VeryHigh),
            container: Some("flac".to_string()),
            audio_channel_layout: Some(AudioLayout::Stereo),
        };

        Preset::new(metadata, config)
    }

    /// Opus 32 kbps mono — speech-optimised audiobook delivery.
    ///
    /// Opus is uniquely suited for low-bitrate speech: at 32 kbps mono it
    /// produces intelligible, clean narration while consuming only 14.4 MB/hour.
    /// The SILK layer inside Opus is activated at this bitrate to maximise
    /// speech quality.
    #[must_use]
    pub fn audiobook_opus() -> Preset {
        let metadata = PresetMetadata::new(
            "audio-only-audiobook-opus",
            "Audiobook Opus 32k Mono",
            PresetCategory::Codec("Opus".to_string()),
        )
        .with_description(
            "Data-efficient Opus at 32 kbps mono for audiobook delivery — speech-optimised",
        )
        .with_target("Audiobook Distribution")
        .with_tag("opus")
        .with_tag("audiobook")
        .with_tag("speech")
        .with_tag("mono")
        .with_tag("audio-only")
        .with_tag("32kbps");

        let config = PresetConfig {
            video_codec: None,
            audio_codec: Some("opus".to_string()),
            video_bitrate: None,
            audio_bitrate: Some(32_000),
            width: None,
            height: None,
            frame_rate: None,
            quality_mode: Some(QualityMode::Low),
            container: Some("ogg".to_string()),
            audio_channel_layout: Some(AudioLayout::Mono),
        };

        Preset::new(metadata, config)
    }

    /// Return all five presets as a `Vec` for bulk registration.
    #[must_use]
    pub fn all_presets() -> Vec<Preset> {
        vec![
            Self::podcast_opus(),
            Self::podcast_flac(),
            Self::music_opus_hq(),
            Self::music_flac_hd(),
            Self::audiobook_opus(),
        ]
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_transcode::audio_channel_map::AudioLayout;

    #[test]
    fn test_all_audio_only_presets_have_no_video_track() {
        for preset in AudioOnlyPresets::all_presets() {
            assert!(
                preset.config.video_codec.is_none(),
                "preset '{}' must have no video_codec",
                preset.metadata.id
            );
            assert!(
                preset.config.video_bitrate.is_none(),
                "preset '{}' must have no video_bitrate",
                preset.metadata.id
            );
        }
    }

    #[test]
    fn test_all_audio_only_presets_have_audio_only_tag() {
        for preset in AudioOnlyPresets::all_presets() {
            assert!(
                preset.has_tag("audio-only"),
                "preset '{}' must have 'audio-only' tag",
                preset.metadata.id
            );
        }
    }

    #[test]
    fn test_all_audio_only_presets_use_patent_free_codecs() {
        for preset in AudioOnlyPresets::all_presets() {
            let codec = preset
                .config
                .audio_codec
                .as_deref()
                .expect("audio codec must be set");
            assert!(
                codec == "opus" || codec == "flac",
                "preset '{}' uses non-patent-free codec '{}'",
                preset.metadata.id,
                codec
            );
        }
    }

    #[test]
    fn test_podcast_opus_is_mono_64kbps() {
        let p = AudioOnlyPresets::podcast_opus();
        assert_eq!(p.config.audio_codec.as_deref(), Some("opus"));
        assert_eq!(p.config.audio_bitrate, Some(64_000));
        assert_eq!(p.config.audio_channel_layout, Some(AudioLayout::Mono));
        assert!(p.has_tag("mono"));
        assert!(p.has_tag("podcast"));
    }

    #[test]
    fn test_podcast_flac_is_lossless_stereo() {
        let p = AudioOnlyPresets::podcast_flac();
        assert_eq!(p.config.audio_codec.as_deref(), Some("flac"));
        assert_eq!(p.config.audio_bitrate, Some(1_411_200));
        assert_eq!(p.config.audio_channel_layout, Some(AudioLayout::Stereo));
        assert!(p.has_tag("lossless"));
        assert!(p.has_tag("stereo"));
    }

    #[test]
    fn test_music_opus_hq_is_stereo_192kbps() {
        let p = AudioOnlyPresets::music_opus_hq();
        assert_eq!(p.config.audio_codec.as_deref(), Some("opus"));
        assert_eq!(p.config.audio_bitrate, Some(192_000));
        assert_eq!(p.config.audio_channel_layout, Some(AudioLayout::Stereo));
        assert!(p.has_tag("hq"));
        assert!(p.has_tag("music"));
    }

    #[test]
    fn test_music_flac_hd_is_24bit_96khz() {
        let p = AudioOnlyPresets::music_flac_hd();
        assert_eq!(p.config.audio_codec.as_deref(), Some("flac"));
        // 96000 Hz × 24-bit × 2 channels
        assert_eq!(p.config.audio_bitrate, Some(4_608_000));
        assert_eq!(p.config.audio_channel_layout, Some(AudioLayout::Stereo));
        assert!(p.has_tag("hires"));
        assert!(p.has_tag("96000"));
    }

    #[test]
    fn test_audiobook_opus_is_mono_speech_optimised() {
        let p = AudioOnlyPresets::audiobook_opus();
        assert_eq!(p.config.audio_codec.as_deref(), Some("opus"));
        assert_eq!(p.config.audio_bitrate, Some(32_000));
        assert_eq!(p.config.audio_channel_layout, Some(AudioLayout::Mono));
        assert!(p.has_tag("speech"));
        assert!(p.has_tag("audiobook"));
        assert!(p.has_tag("32kbps"));
    }

    #[test]
    fn test_podcast_mono_vs_music_stereo_channel_count() {
        let podcast = AudioOnlyPresets::podcast_opus();
        let music = AudioOnlyPresets::music_opus_hq();

        assert_eq!(
            podcast.config.audio_channel_layout,
            Some(AudioLayout::Mono),
            "podcast_opus should be mono"
        );
        assert_eq!(
            music.config.audio_channel_layout,
            Some(AudioLayout::Stereo),
            "music_opus_hq should be stereo"
        );
    }

    #[test]
    fn test_all_presets_count() {
        assert_eq!(AudioOnlyPresets::all_presets().len(), 5);
    }
}
