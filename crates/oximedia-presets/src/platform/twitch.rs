//! Twitch streaming presets following official ingest requirements.
//!
//! Provides presets optimized for Twitch's ingest servers:
//! - 720p30 low-latency for bandwidth-constrained streamers
//! - 1080p60 for standard high-quality streaming
//! - Source quality for Twitch Partners with transcoding access

use crate::{Preset, PresetCategory, PresetMetadata};
use oximedia_transcode::{PresetConfig, QualityMode};

/// Get all Twitch presets.
#[must_use]
pub fn all_presets() -> Vec<Preset> {
    vec![
        twitch_720p30_low_latency(),
        twitch_1080p60(),
        twitch_source_quality(),
        twitch_720p60(),
        twitch_480p30(),
    ]
}

/// Twitch 720p30 low-latency preset.
///
/// Optimized for minimal encoding delay: single-pass CBR at 3 Mbps,
/// short keyframe interval (2 s). Ideal for interactive streams on
/// limited bandwidth.
#[must_use]
pub fn twitch_720p30_low_latency() -> Preset {
    let metadata = PresetMetadata::new(
        "twitch-720p30-low-latency",
        "Twitch 720p30 Low Latency",
        PresetCategory::Platform("Twitch".to_string()),
    )
    .with_description("Low-latency 720p30 preset for Twitch (3 Mbps CBR)")
    .with_target("Twitch Ingest")
    .with_tag("twitch")
    .with_tag("720p")
    .with_tag("low-latency")
    .with_tag("rtmp")
    .with_tag("h264");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(3_000_000), // 3 Mbps — Twitch recommended for 720p30
        audio_bitrate: Some(128_000),   // 128 kbps AAC
        width: Some(1280),
        height: Some(720),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::Medium),
        container: Some("flv".to_string()), // RTMP ingest uses FLV
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// Twitch 1080p60 standard streaming preset.
///
/// The most common high-quality Twitch preset: 6 Mbps CBR, 1080p at 60 fps.
/// Requires Twitch Partner/Affiliate transcoding for viewer adaptation.
#[must_use]
pub fn twitch_1080p60() -> Preset {
    let metadata = PresetMetadata::new(
        "twitch-1080p60",
        "Twitch 1080p60",
        PresetCategory::Platform("Twitch".to_string()),
    )
    .with_description("Standard 1080p60 Twitch streaming preset (6 Mbps CBR)")
    .with_target("Twitch Ingest")
    .with_tag("twitch")
    .with_tag("1080p")
    .with_tag("60fps")
    .with_tag("rtmp")
    .with_tag("h264");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(6_000_000), // 6 Mbps — Twitch recommended max
        audio_bitrate: Some(160_000),   // 160 kbps AAC
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((60, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("flv".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// Twitch source quality preset.
///
/// Maximum quality for Twitch Partners with guaranteed transcoding:
/// 8.5 Mbps, 1080p60. Uses the Twitch-recommended upper bitrate ceiling.
#[must_use]
pub fn twitch_source_quality() -> Preset {
    let metadata = PresetMetadata::new(
        "twitch-source-quality",
        "Twitch Source Quality",
        PresetCategory::Platform("Twitch".to_string()),
    )
    .with_description("Maximum quality source preset for Twitch Partners (8.5 Mbps)")
    .with_target("Twitch Ingest")
    .with_tag("twitch")
    .with_tag("1080p")
    .with_tag("60fps")
    .with_tag("source")
    .with_tag("rtmp")
    .with_tag("h264");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(8_500_000), // 8.5 Mbps — Twitch upper ceiling
        audio_bitrate: Some(320_000),   // 320 kbps AAC
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((60, 1)),
        quality_mode: Some(QualityMode::VeryHigh),
        container: Some("flv".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// Twitch 720p60 preset.
///
/// A compromise between low-latency 720p30 and full 1080p60: smooth 60 fps
/// at a moderate bitrate of 4.5 Mbps.
#[must_use]
pub fn twitch_720p60() -> Preset {
    let metadata = PresetMetadata::new(
        "twitch-720p60",
        "Twitch 720p60",
        PresetCategory::Platform("Twitch".to_string()),
    )
    .with_description("720p60 Twitch streaming preset (4.5 Mbps CBR)")
    .with_target("Twitch Ingest")
    .with_tag("twitch")
    .with_tag("720p")
    .with_tag("60fps")
    .with_tag("rtmp")
    .with_tag("h264");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(4_500_000), // 4.5 Mbps
        audio_bitrate: Some(128_000),
        width: Some(1280),
        height: Some(720),
        frame_rate: Some((60, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("flv".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// Twitch 480p30 low-bandwidth preset.
///
/// For streamers with severely limited upload bandwidth: 1.5 Mbps CBR at
/// 480p30. Ensures a watchable stream even on slow connections.
#[must_use]
pub fn twitch_480p30() -> Preset {
    let metadata = PresetMetadata::new(
        "twitch-480p30",
        "Twitch 480p30",
        PresetCategory::Platform("Twitch".to_string()),
    )
    .with_description("Low-bandwidth 480p30 Twitch preset (1.5 Mbps CBR)")
    .with_target("Twitch Ingest")
    .with_tag("twitch")
    .with_tag("480p")
    .with_tag("low-bandwidth")
    .with_tag("rtmp")
    .with_tag("h264");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(1_500_000), // 1.5 Mbps
        audio_bitrate: Some(128_000),   // 128 kbps AAC (Twitch minimum)
        width: Some(854),
        height: Some(480),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::Low),
        container: Some("flv".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_twitch_presets_count() {
        assert_eq!(all_presets().len(), 5);
    }

    #[test]
    fn test_twitch_720p30_low_latency() {
        let p = twitch_720p30_low_latency();
        assert_eq!(p.metadata.id, "twitch-720p30-low-latency");
        assert_eq!(p.config.width, Some(1280));
        assert_eq!(p.config.height, Some(720));
        assert_eq!(p.config.frame_rate, Some((30, 1)));
        assert_eq!(p.config.video_bitrate, Some(3_000_000));
        assert!(p.has_tag("low-latency"));
        assert!(p.has_tag("twitch"));
    }

    #[test]
    fn test_twitch_1080p60() {
        let p = twitch_1080p60();
        assert_eq!(p.metadata.id, "twitch-1080p60");
        assert_eq!(p.config.width, Some(1920));
        assert_eq!(p.config.height, Some(1080));
        assert_eq!(p.config.frame_rate, Some((60, 1)));
        assert_eq!(p.config.video_bitrate, Some(6_000_000));
    }

    #[test]
    fn test_twitch_source_quality() {
        let p = twitch_source_quality();
        assert_eq!(p.metadata.id, "twitch-source-quality");
        assert_eq!(p.config.video_bitrate, Some(8_500_000));
        assert!(p.has_tag("source"));
    }

    #[test]
    fn test_twitch_720p60() {
        let p = twitch_720p60();
        assert_eq!(p.config.video_bitrate, Some(4_500_000));
        assert_eq!(p.config.frame_rate, Some((60, 1)));
    }

    #[test]
    fn test_twitch_480p30() {
        let p = twitch_480p30();
        assert_eq!(p.config.video_bitrate, Some(1_500_000));
        assert_eq!(p.config.height, Some(480));
    }

    #[test]
    fn test_all_twitch_presets_use_flv() {
        for p in all_presets() {
            assert_eq!(
                p.config.container.as_deref(),
                Some("flv"),
                "Twitch preset {} should use FLV container for RTMP ingest",
                p.metadata.id
            );
        }
    }

    #[test]
    fn test_all_twitch_presets_use_h264() {
        for p in all_presets() {
            assert_eq!(
                p.config.video_codec.as_deref(),
                Some("h264"),
                "Twitch preset {} should use H.264",
                p.metadata.id
            );
        }
    }

    #[test]
    fn test_all_twitch_presets_have_twitch_tag() {
        for p in all_presets() {
            assert!(
                p.has_tag("twitch"),
                "Twitch preset {} should have 'twitch' tag",
                p.metadata.id
            );
        }
    }
}
