// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Unified social media encoding presets with cross-platform support.
//!
//! This module provides a higher-level API on top of [`crate::presets::social_media`]
//! that adds:
//!
//! - **Facebook** presets (Feed, Stories, Reels, Ads)
//! - Cross-platform resolution/bitrate negotiation
//! - Platform constraint validation (file size, duration, aspect ratio)
//! - Multi-platform export planning (one source -> multiple platform outputs)
//!
//! All codecs are patent-free (AV1, VP9, VP8, Opus, Vorbis, FLAC).

use crate::formats::{AudioCodec, ChannelLayout, ContainerFormat, VideoCodec};
use crate::presets::{AudioPresetSettings, EncodingSpeed, Preset, VideoPresetSettings};
use crate::Result;
use serde::{Deserialize, Serialize};

// ── Platform enumeration ───────────────────────────────────────────────────

/// Extended social media platform identifiers (superset of `presets::social_media::SocialPlatform`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Platform {
    /// YouTube (all format variants)
    YouTube,
    /// Instagram (Feed, Stories, Reels)
    Instagram,
    /// TikTok
    TikTok,
    /// Twitter / X
    Twitter,
    /// Facebook (Feed, Stories, Reels, Ads)
    Facebook,
}

impl Platform {
    /// Human-readable name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::YouTube => "YouTube",
            Self::Instagram => "Instagram",
            Self::TikTok => "TikTok",
            Self::Twitter => "Twitter/X",
            Self::Facebook => "Facebook",
        }
    }

    /// Maximum upload file size in bytes.
    #[must_use]
    pub const fn max_file_size_bytes(self) -> u64 {
        match self {
            Self::YouTube => 256 * 1024 * 1024 * 1024, // 256 GB
            Self::Instagram => 650 * 1024 * 1024,      // 650 MB
            Self::TikTok => 287 * 1024 * 1024,         // 287 MB
            Self::Twitter => 512 * 1024 * 1024,        // 512 MB
            Self::Facebook => 10 * 1024 * 1024 * 1024, // 10 GB
        }
    }

    /// Maximum video duration in seconds.
    #[must_use]
    pub const fn max_duration_seconds(self) -> u64 {
        match self {
            Self::YouTube => 43200,  // 12 hours
            Self::Instagram => 3600, // 60 min (IGTV/Feed)
            Self::TikTok => 600,     // 10 min
            Self::Twitter => 140,    // 2:20
            Self::Facebook => 14400, // 4 hours (Feed)
        }
    }

    /// Maximum recommended resolution (width, height) for the platform.
    #[must_use]
    pub const fn max_resolution(self) -> (u32, u32) {
        match self {
            Self::YouTube => (3840, 2160),
            Self::Instagram => (1080, 1920),
            Self::TikTok => (1080, 1920),
            Self::Twitter => (1920, 1200),
            Self::Facebook => (1920, 1080),
        }
    }

    /// All available format variants for this platform.
    #[must_use]
    pub fn variants(self) -> Vec<PlatformVariant> {
        match self {
            Self::YouTube => vec![
                PlatformVariant::YouTube1080p,
                PlatformVariant::YouTube1080p60,
                PlatformVariant::YouTube4K,
                PlatformVariant::YouTubeShorts,
                PlatformVariant::YouTubeLive,
            ],
            Self::Instagram => vec![
                PlatformVariant::InstagramFeedSquare,
                PlatformVariant::InstagramFeedPortrait,
                PlatformVariant::InstagramFeedLandscape,
                PlatformVariant::InstagramReels,
                PlatformVariant::InstagramStories,
            ],
            Self::TikTok => vec![
                PlatformVariant::TikTokStandard,
                PlatformVariant::TikTokHd,
                PlatformVariant::TikTokAds,
            ],
            Self::Twitter => vec![
                PlatformVariant::TwitterLandscape,
                PlatformVariant::TwitterPortrait,
                PlatformVariant::TwitterSquare,
            ],
            Self::Facebook => vec![
                PlatformVariant::FacebookFeed,
                PlatformVariant::FacebookStories,
                PlatformVariant::FacebookReels,
                PlatformVariant::FacebookAds,
            ],
        }
    }

    /// All platforms.
    #[must_use]
    pub const fn all() -> &'static [Platform] {
        &[
            Platform::YouTube,
            Platform::Instagram,
            Platform::TikTok,
            Platform::Twitter,
            Platform::Facebook,
        ]
    }
}

// ── Platform variants ──────────────────────────────────────────────────────

/// Specific format variant within a platform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PlatformVariant {
    // YouTube
    /// YouTube 1080p 30fps standard upload
    YouTube1080p,
    /// YouTube 1080p 60fps for gaming/sports
    YouTube1080p60,
    /// YouTube 4K HDR
    YouTube4K,
    /// YouTube Shorts (vertical, max 60s)
    YouTubeShorts,
    /// YouTube Live streaming
    YouTubeLive,

    // Instagram
    /// Instagram Feed 1:1 square
    InstagramFeedSquare,
    /// Instagram Feed 4:5 portrait
    InstagramFeedPortrait,
    /// Instagram Feed 1.91:1 landscape
    InstagramFeedLandscape,
    /// Instagram Reels 9:16
    InstagramReels,
    /// Instagram Stories 9:16
    InstagramStories,

    // TikTok
    /// TikTok standard quality
    TikTokStandard,
    /// TikTok high definition
    TikTokHd,
    /// TikTok advertising
    TikTokAds,

    // Twitter
    /// Twitter 16:9 landscape
    TwitterLandscape,
    /// Twitter 9:16 portrait
    TwitterPortrait,
    /// Twitter 1:1 square
    TwitterSquare,

    // Facebook
    /// Facebook Feed 16:9 or 1:1
    FacebookFeed,
    /// Facebook Stories 9:16
    FacebookStories,
    /// Facebook Reels 9:16
    FacebookReels,
    /// Facebook In-Stream Ads
    FacebookAds,
}

impl PlatformVariant {
    /// The parent platform.
    #[must_use]
    pub const fn platform(self) -> Platform {
        match self {
            Self::YouTube1080p
            | Self::YouTube1080p60
            | Self::YouTube4K
            | Self::YouTubeShorts
            | Self::YouTubeLive => Platform::YouTube,

            Self::InstagramFeedSquare
            | Self::InstagramFeedPortrait
            | Self::InstagramFeedLandscape
            | Self::InstagramReels
            | Self::InstagramStories => Platform::Instagram,

            Self::TikTokStandard | Self::TikTokHd | Self::TikTokAds => Platform::TikTok,

            Self::TwitterLandscape | Self::TwitterPortrait | Self::TwitterSquare => {
                Platform::Twitter
            }

            Self::FacebookFeed
            | Self::FacebookStories
            | Self::FacebookReels
            | Self::FacebookAds => Platform::Facebook,
        }
    }

    /// Human-readable label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::YouTube1080p => "YouTube 1080p",
            Self::YouTube1080p60 => "YouTube 1080p 60fps",
            Self::YouTube4K => "YouTube 4K HDR",
            Self::YouTubeShorts => "YouTube Shorts",
            Self::YouTubeLive => "YouTube Live",
            Self::InstagramFeedSquare => "Instagram Feed (Square)",
            Self::InstagramFeedPortrait => "Instagram Feed (Portrait)",
            Self::InstagramFeedLandscape => "Instagram Feed (Landscape)",
            Self::InstagramReels => "Instagram Reels",
            Self::InstagramStories => "Instagram Stories",
            Self::TikTokStandard => "TikTok Standard",
            Self::TikTokHd => "TikTok HD",
            Self::TikTokAds => "TikTok Ads",
            Self::TwitterLandscape => "Twitter Landscape",
            Self::TwitterPortrait => "Twitter Portrait",
            Self::TwitterSquare => "Twitter Square",
            Self::FacebookFeed => "Facebook Feed",
            Self::FacebookStories => "Facebook Stories",
            Self::FacebookReels => "Facebook Reels",
            Self::FacebookAds => "Facebook Ads",
        }
    }

    /// Target resolution (width, height).
    #[must_use]
    pub const fn resolution(self) -> (u32, u32) {
        match self {
            Self::YouTube1080p | Self::YouTube1080p60 | Self::YouTubeLive => (1920, 1080),
            Self::YouTube4K => (3840, 2160),
            Self::YouTubeShorts => (1080, 1920),
            Self::InstagramFeedSquare => (1080, 1080),
            Self::InstagramFeedPortrait => (1080, 1350),
            Self::InstagramFeedLandscape => (1080, 566),
            Self::InstagramReels | Self::InstagramStories => (1080, 1920),
            Self::TikTokStandard | Self::TikTokHd | Self::TikTokAds => (1080, 1920),
            Self::TwitterLandscape => (1280, 720),
            Self::TwitterPortrait => (720, 1280),
            Self::TwitterSquare => (720, 720),
            Self::FacebookFeed => (1920, 1080),
            Self::FacebookStories | Self::FacebookReels => (1080, 1920),
            Self::FacebookAds => (1280, 720),
        }
    }

    /// Aspect ratio as a string.
    #[must_use]
    pub const fn aspect_ratio(self) -> &'static str {
        match self {
            Self::YouTube1080p
            | Self::YouTube1080p60
            | Self::YouTube4K
            | Self::YouTubeLive
            | Self::TwitterLandscape
            | Self::FacebookFeed
            | Self::FacebookAds => "16:9",

            Self::YouTubeShorts
            | Self::InstagramReels
            | Self::InstagramStories
            | Self::TikTokStandard
            | Self::TikTokHd
            | Self::TikTokAds
            | Self::TwitterPortrait
            | Self::FacebookStories
            | Self::FacebookReels => "9:16",

            Self::InstagramFeedSquare | Self::TwitterSquare => "1:1",
            Self::InstagramFeedPortrait => "4:5",
            Self::InstagramFeedLandscape => "1.91:1",
        }
    }

    /// Maximum duration in seconds for this variant.
    #[must_use]
    pub const fn max_duration_seconds(self) -> u64 {
        match self {
            Self::YouTubeShorts => 60,
            Self::YouTubeLive => 43200,
            Self::YouTube1080p | Self::YouTube1080p60 | Self::YouTube4K => 43200,
            Self::InstagramFeedSquare
            | Self::InstagramFeedPortrait
            | Self::InstagramFeedLandscape => 3600,
            Self::InstagramReels => 90,
            Self::InstagramStories => 60,
            Self::TikTokStandard | Self::TikTokHd | Self::TikTokAds => 600,
            Self::TwitterLandscape | Self::TwitterPortrait | Self::TwitterSquare => 140,
            Self::FacebookFeed => 14400,
            Self::FacebookStories => 120,
            Self::FacebookReels => 90,
            Self::FacebookAds => 241,
        }
    }

    /// Build the encoding preset for this variant.
    pub fn to_preset(self) -> Result<Preset> {
        match self {
            // Delegate existing variants to the social_media module
            Self::YouTube1080p => Ok(Preset {
                name: self.label().to_string(),
                description: format!(
                    "{} preset: {}x{} {}",
                    self.label(),
                    self.resolution().0,
                    self.resolution().1,
                    self.aspect_ratio()
                ),
                container: ContainerFormat::Mp4,
                video: Some(VideoPresetSettings {
                    codec: VideoCodec::Vp9,
                    width: Some(1920),
                    height: Some(1080),
                    frame_rate: Some(30.0),
                    bitrate: Some(8_000_000),
                    quality: None,
                    two_pass: true,
                    speed: EncodingSpeed::Medium,
                }),
                audio: Some(AudioPresetSettings {
                    codec: AudioCodec::Opus,
                    sample_rate: 48000,
                    channels: ChannelLayout::Stereo,
                    bitrate: Some(192_000),
                }),
            }),
            Self::YouTube1080p60 => {
                crate::presets::social_media::social_media_preset("youtube-1080p-60")
            }
            Self::YouTube4K => crate::presets::social_media::social_media_preset("youtube-4k-hdr"),
            Self::YouTubeShorts => {
                crate::presets::social_media::social_media_preset("youtube-shorts")
            }
            Self::YouTubeLive => crate::presets::social_media::social_media_preset("youtube-live"),
            Self::InstagramFeedSquare => {
                crate::presets::social_media::social_media_preset("instagram-feed-square")
            }
            Self::InstagramFeedPortrait => {
                crate::presets::social_media::social_media_preset("instagram-feed-portrait")
            }
            Self::InstagramFeedLandscape => {
                crate::presets::social_media::social_media_preset("instagram-feed-landscape")
            }
            Self::InstagramReels => {
                crate::presets::social_media::social_media_preset("instagram-reels")
            }
            Self::InstagramStories => {
                crate::presets::social_media::social_media_preset("instagram-stories")
            }
            Self::TikTokStandard => {
                crate::presets::social_media::social_media_preset("tiktok-standard")
            }
            Self::TikTokHd => crate::presets::social_media::social_media_preset("tiktok-hd"),
            Self::TikTokAds => crate::presets::social_media::social_media_preset("tiktok-ads"),
            Self::TwitterLandscape => {
                crate::presets::social_media::social_media_preset("twitter-landscape")
            }
            Self::TwitterPortrait => {
                crate::presets::social_media::social_media_preset("twitter-portrait")
            }
            Self::TwitterSquare => {
                crate::presets::social_media::social_media_preset("twitter-square")
            }

            // New Facebook presets
            Self::FacebookFeed => facebook_feed(),
            Self::FacebookStories => facebook_stories(),
            Self::FacebookReels => facebook_reels(),
            Self::FacebookAds => facebook_ads(),
        }
    }
}

// ── Facebook presets ───────────────────────────────────────────────────────

/// Facebook Feed preset: 1920x1080, 16:9, up to 4 hours.
pub fn facebook_feed() -> Result<Preset> {
    Ok(Preset {
        name: "Facebook Feed".to_string(),
        description: "Facebook Feed 1920x1080 (16:9), up to 240 minutes".to_string(),
        container: ContainerFormat::Mp4,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Vp9,
            width: Some(1920),
            height: Some(1080),
            frame_rate: Some(30.0),
            bitrate: Some(6_000_000),
            quality: None,
            two_pass: false,
            speed: EncodingSpeed::Fast,
        }),
        audio: Some(AudioPresetSettings {
            codec: AudioCodec::Opus,
            sample_rate: 48000,
            channels: ChannelLayout::Stereo,
            bitrate: Some(128_000),
        }),
    })
}

/// Facebook Stories preset: 1080x1920, 9:16, up to 120 seconds.
pub fn facebook_stories() -> Result<Preset> {
    Ok(Preset {
        name: "Facebook Stories".to_string(),
        description: "Facebook Stories 1080x1920 (9:16), max 120 seconds".to_string(),
        container: ContainerFormat::Mp4,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Vp9,
            width: Some(1080),
            height: Some(1920),
            frame_rate: Some(30.0),
            bitrate: Some(4_000_000),
            quality: None,
            two_pass: false,
            speed: EncodingSpeed::Fast,
        }),
        audio: Some(AudioPresetSettings {
            codec: AudioCodec::Opus,
            sample_rate: 48000,
            channels: ChannelLayout::Stereo,
            bitrate: Some(128_000),
        }),
    })
}

/// Facebook Reels preset: 1080x1920, 9:16, up to 90 seconds.
pub fn facebook_reels() -> Result<Preset> {
    Ok(Preset {
        name: "Facebook Reels".to_string(),
        description: "Facebook Reels 1080x1920 (9:16), max 90 seconds".to_string(),
        container: ContainerFormat::Mp4,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Vp9,
            width: Some(1080),
            height: Some(1920),
            frame_rate: Some(30.0),
            bitrate: Some(5_000_000),
            quality: None,
            two_pass: false,
            speed: EncodingSpeed::Fast,
        }),
        audio: Some(AudioPresetSettings {
            codec: AudioCodec::Opus,
            sample_rate: 48000,
            channels: ChannelLayout::Stereo,
            bitrate: Some(128_000),
        }),
    })
}

/// Facebook In-Stream Ads preset: 1280x720, 16:9, up to ~4 minutes.
pub fn facebook_ads() -> Result<Preset> {
    Ok(Preset {
        name: "Facebook Ads".to_string(),
        description: "Facebook In-Stream Ads 1280x720 (16:9), max 241 seconds".to_string(),
        container: ContainerFormat::Mp4,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Vp9,
            width: Some(1280),
            height: Some(720),
            frame_rate: Some(30.0),
            bitrate: Some(5_000_000),
            quality: None,
            two_pass: true,
            speed: EncodingSpeed::Medium,
        }),
        audio: Some(AudioPresetSettings {
            codec: AudioCodec::Opus,
            sample_rate: 48000,
            channels: ChannelLayout::Stereo,
            bitrate: Some(128_000),
        }),
    })
}

// ── Platform constraint validation ─────────────────────────────────────────

/// Constraint violation detected during validation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstraintViolation {
    /// Which constraint was violated.
    pub kind: ConstraintKind,
    /// Human-readable description of the violation.
    pub message: String,
}

/// Kinds of platform constraints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConstraintKind {
    /// Duration exceeds platform maximum
    DurationTooLong,
    /// File size exceeds platform maximum
    FileSizeTooLarge,
    /// Resolution exceeds platform maximum
    ResolutionTooHigh,
    /// Frame rate outside platform support
    UnsupportedFrameRate,
}

/// Validate that a video with the given properties fits within platform constraints.
pub fn validate_constraints(
    variant: PlatformVariant,
    duration_seconds: f64,
    file_size_bytes: u64,
    width: u32,
    height: u32,
    frame_rate: f64,
) -> Vec<ConstraintViolation> {
    let mut violations = Vec::new();
    let platform = variant.platform();

    // Duration check
    let max_dur = variant.max_duration_seconds() as f64;
    if duration_seconds > max_dur {
        violations.push(ConstraintViolation {
            kind: ConstraintKind::DurationTooLong,
            message: format!(
                "{}: duration {:.1}s exceeds max {:.0}s",
                variant.label(),
                duration_seconds,
                max_dur
            ),
        });
    }

    // File size check
    let max_size = platform.max_file_size_bytes();
    if file_size_bytes > max_size {
        violations.push(ConstraintViolation {
            kind: ConstraintKind::FileSizeTooLarge,
            message: format!(
                "{}: file size {} bytes exceeds max {} bytes",
                platform.name(),
                file_size_bytes,
                max_size
            ),
        });
    }

    // Resolution check
    let (max_w, max_h) = platform.max_resolution();
    if width > max_w || height > max_h {
        violations.push(ConstraintViolation {
            kind: ConstraintKind::ResolutionTooHigh,
            message: format!(
                "{}: resolution {}x{} exceeds max {}x{}",
                platform.name(),
                width,
                height,
                max_w,
                max_h
            ),
        });
    }

    // Frame rate check (most platforms support 24-60)
    if frame_rate < 1.0 || frame_rate > 120.0 {
        violations.push(ConstraintViolation {
            kind: ConstraintKind::UnsupportedFrameRate,
            message: format!(
                "{}: frame rate {:.1}fps outside supported range",
                platform.name(),
                frame_rate
            ),
        });
    }

    violations
}

// ── Multi-platform export plan ─────────────────────────────────────────────

/// A planned export to one platform variant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExportPlan {
    /// Target variant.
    pub variant: PlatformVariant,
    /// Encoding preset.
    pub preset: Preset,
    /// Any constraint violations for the source material.
    pub violations: Vec<ConstraintViolation>,
    /// Suggested output filename suffix.
    pub filename_suffix: String,
}

/// Plan exports to multiple platform variants from a single source.
///
/// Returns one `ExportPlan` per requested variant.  Each plan includes
/// the encoding preset and any constraint violations detected for the
/// given source properties.
pub fn plan_multi_platform_export(
    variants: &[PlatformVariant],
    source_duration_seconds: f64,
    source_file_size_bytes: u64,
    source_width: u32,
    source_height: u32,
    source_frame_rate: f64,
) -> Result<Vec<ExportPlan>> {
    let mut plans = Vec::with_capacity(variants.len());

    for &variant in variants {
        let preset = variant.to_preset()?;
        let violations = validate_constraints(
            variant,
            source_duration_seconds,
            source_file_size_bytes,
            source_width,
            source_height,
            source_frame_rate,
        );
        let (w, h) = variant.resolution();
        let filename_suffix = format!(
            "_{}_{}x{}",
            variant.label().to_lowercase().replace([' ', '/'], "_"),
            w,
            h
        );

        plans.push(ExportPlan {
            variant,
            preset,
            violations,
            filename_suffix,
        });
    }

    Ok(plans)
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_platforms_have_variants() {
        for &platform in Platform::all() {
            let variants = platform.variants();
            assert!(
                !variants.is_empty(),
                "{} should have at least one variant",
                platform.name()
            );
        }
    }

    #[test]
    fn test_all_variants_produce_presets() {
        for &platform in Platform::all() {
            for variant in platform.variants() {
                let result = variant.to_preset();
                assert!(
                    result.is_ok(),
                    "{:?} should produce a valid preset: {:?}",
                    variant,
                    result.err()
                );
                let preset = result.expect("checked above");
                assert!(!preset.name.is_empty());
                assert!(!preset.description.is_empty());
                assert!(preset.video.is_some(), "{:?} should have video", variant);
                assert!(preset.audio.is_some(), "{:?} should have audio", variant);
            }
        }
    }

    #[test]
    fn test_facebook_feed_preset() {
        let preset = facebook_feed().expect("should be valid");
        let video = preset.video.expect("should have video");
        assert_eq!(video.width, Some(1920));
        assert_eq!(video.height, Some(1080));
        assert_eq!(video.frame_rate, Some(30.0));
        assert_eq!(video.codec, VideoCodec::Vp9);
        let audio = preset.audio.expect("should have audio");
        assert_eq!(audio.codec, AudioCodec::Opus);
        assert_eq!(audio.sample_rate, 48000);
    }

    #[test]
    fn test_facebook_stories_vertical() {
        let preset = facebook_stories().expect("should be valid");
        let video = preset.video.expect("should have video");
        assert_eq!(video.width, Some(1080));
        assert_eq!(video.height, Some(1920));
        assert!(video.height > video.width);
    }

    #[test]
    fn test_facebook_reels_preset() {
        let preset = facebook_reels().expect("should be valid");
        let video = preset.video.expect("should have video");
        assert_eq!(video.width, Some(1080));
        assert_eq!(video.height, Some(1920));
    }

    #[test]
    fn test_facebook_ads_two_pass() {
        let preset = facebook_ads().expect("should be valid");
        let video = preset.video.expect("should have video");
        assert!(video.two_pass, "Ads should use two-pass for quality");
        assert_eq!(video.speed, EncodingSpeed::Medium);
    }

    #[test]
    fn test_constraint_validation_pass() {
        let violations = validate_constraints(
            PlatformVariant::YouTubeShorts,
            30.0,       // 30 seconds (max 60)
            50_000_000, // 50 MB
            1080,
            1920,
            30.0,
        );
        assert!(violations.is_empty(), "Should have no violations");
    }

    #[test]
    fn test_constraint_duration_violation() {
        let violations = validate_constraints(
            PlatformVariant::YouTubeShorts,
            120.0, // 120 seconds (max 60)
            50_000_000,
            1080,
            1920,
            30.0,
        );
        assert!(!violations.is_empty());
        assert!(violations
            .iter()
            .any(|v| v.kind == ConstraintKind::DurationTooLong));
    }

    #[test]
    fn test_constraint_file_size_violation() {
        let violations = validate_constraints(
            PlatformVariant::TikTokStandard,
            60.0,
            500 * 1024 * 1024, // 500 MB (max 287 MB for TikTok)
            1080,
            1920,
            30.0,
        );
        assert!(violations
            .iter()
            .any(|v| v.kind == ConstraintKind::FileSizeTooLarge));
    }

    #[test]
    fn test_constraint_frame_rate_violation() {
        let violations = validate_constraints(
            PlatformVariant::FacebookFeed,
            60.0,
            50_000_000,
            1920,
            1080,
            0.5, // Too low
        );
        assert!(violations
            .iter()
            .any(|v| v.kind == ConstraintKind::UnsupportedFrameRate));
    }

    #[test]
    fn test_multi_platform_export() {
        let variants = vec![
            PlatformVariant::YouTube1080p,
            PlatformVariant::InstagramReels,
            PlatformVariant::TikTokStandard,
            PlatformVariant::FacebookFeed,
        ];
        let result = plan_multi_platform_export(&variants, 60.0, 100_000_000, 1920, 1080, 30.0);
        assert!(result.is_ok());
        let plans = result.expect("should succeed");
        assert_eq!(plans.len(), 4);

        // Each plan should have a unique filename suffix
        let suffixes: Vec<&str> = plans.iter().map(|p| p.filename_suffix.as_str()).collect();
        for (i, s) in suffixes.iter().enumerate() {
            for (j, other) in suffixes.iter().enumerate() {
                if i != j {
                    assert_ne!(s, other, "Filename suffixes should be unique");
                }
            }
        }
    }

    #[test]
    fn test_variant_parent_platform() {
        assert_eq!(PlatformVariant::YouTubeShorts.platform(), Platform::YouTube);
        assert_eq!(
            PlatformVariant::InstagramReels.platform(),
            Platform::Instagram
        );
        assert_eq!(PlatformVariant::TikTokHd.platform(), Platform::TikTok);
        assert_eq!(PlatformVariant::TwitterSquare.platform(), Platform::Twitter);
        assert_eq!(PlatformVariant::FacebookFeed.platform(), Platform::Facebook);
    }

    #[test]
    fn test_variant_aspect_ratios() {
        assert_eq!(PlatformVariant::YouTube1080p.aspect_ratio(), "16:9");
        assert_eq!(PlatformVariant::InstagramFeedSquare.aspect_ratio(), "1:1");
        assert_eq!(PlatformVariant::InstagramFeedPortrait.aspect_ratio(), "4:5");
        assert_eq!(PlatformVariant::TikTokStandard.aspect_ratio(), "9:16");
        assert_eq!(PlatformVariant::FacebookFeed.aspect_ratio(), "16:9");
    }

    #[test]
    fn test_all_presets_patent_free() {
        for &platform in Platform::all() {
            for variant in platform.variants() {
                let preset = variant.to_preset().expect("should produce preset");
                if let Some(v) = &preset.video {
                    assert!(
                        matches!(
                            v.codec,
                            VideoCodec::Av1
                                | VideoCodec::Vp9
                                | VideoCodec::Vp8
                                | VideoCodec::Theora
                        ),
                        "{:?} uses non-patent-free video codec {:?}",
                        variant,
                        v.codec
                    );
                }
                if let Some(a) = &preset.audio {
                    assert!(
                        matches!(
                            a.codec,
                            AudioCodec::Opus
                                | AudioCodec::Vorbis
                                | AudioCodec::Flac
                                | AudioCodec::Pcm
                        ),
                        "{:?} uses non-patent-free audio codec {:?}",
                        variant,
                        a.codec
                    );
                }
            }
        }
    }

    #[test]
    fn test_platform_max_values() {
        // Basic sanity
        assert!(Platform::YouTube.max_file_size_bytes() > Platform::TikTok.max_file_size_bytes());
        assert!(
            Platform::YouTube.max_duration_seconds() > Platform::Twitter.max_duration_seconds()
        );
        assert!(Platform::Facebook.max_file_size_bytes() > Platform::TikTok.max_file_size_bytes());
    }

    #[test]
    fn test_variant_max_durations() {
        assert_eq!(PlatformVariant::YouTubeShorts.max_duration_seconds(), 60);
        assert_eq!(PlatformVariant::InstagramReels.max_duration_seconds(), 90);
        assert_eq!(PlatformVariant::FacebookStories.max_duration_seconds(), 120);
        assert_eq!(PlatformVariant::FacebookReels.max_duration_seconds(), 90);
        assert_eq!(PlatformVariant::FacebookAds.max_duration_seconds(), 241);
    }

    #[test]
    fn test_export_plan_violations_propagated() {
        let variants = vec![PlatformVariant::TwitterLandscape];
        let result = plan_multi_platform_export(
            &variants, 300.0, // 5 minutes (Twitter max 2:20 = 140s)
            50_000_000, 1280, 720, 30.0,
        );
        assert!(result.is_ok());
        let plans = result.expect("should succeed");
        assert_eq!(plans.len(), 1);
        assert!(
            !plans[0].violations.is_empty(),
            "Should have duration violation for Twitter"
        );
        assert!(plans[0]
            .violations
            .iter()
            .any(|v| v.kind == ConstraintKind::DurationTooLong));
    }

    // ── Additional tests for cross-platform negotiation ───────────────────────

    #[test]
    fn test_all_variants_have_non_zero_resolution() {
        for &platform in Platform::all() {
            for variant in platform.variants() {
                let (w, h) = variant.resolution();
                assert!(w > 0, "{:?} should have non-zero width", variant);
                assert!(h > 0, "{:?} should have non-zero height", variant);
            }
        }
    }

    #[test]
    fn test_vertical_variants_have_portrait_resolution() {
        let vertical = [
            PlatformVariant::YouTubeShorts,
            PlatformVariant::InstagramReels,
            PlatformVariant::InstagramStories,
            PlatformVariant::TikTokStandard,
            PlatformVariant::TikTokHd,
            PlatformVariant::FacebookStories,
            PlatformVariant::FacebookReels,
        ];
        for variant in vertical {
            let (w, h) = variant.resolution();
            assert!(
                h > w,
                "{:?} should be portrait (h > w), got {}x{}",
                variant,
                w,
                h
            );
        }
    }

    #[test]
    fn test_landscape_variants_have_landscape_resolution() {
        let landscape = [
            PlatformVariant::YouTube1080p,
            PlatformVariant::YouTube1080p60,
            PlatformVariant::YouTube4K,
            PlatformVariant::FacebookFeed,
            PlatformVariant::FacebookAds,
            PlatformVariant::TwitterLandscape,
        ];
        for variant in landscape {
            let (w, h) = variant.resolution();
            assert!(
                w > h,
                "{:?} should be landscape (w > h), got {}x{}",
                variant,
                w,
                h
            );
        }
    }

    #[test]
    fn test_filename_suffix_contains_resolution() {
        let variants = vec![PlatformVariant::YouTube1080p, PlatformVariant::TikTokHd];
        let plans = plan_multi_platform_export(&variants, 30.0, 50_000_000, 1920, 1080, 30.0)
            .expect("should succeed");
        for plan in &plans {
            let (w, h) = plan.variant.resolution();
            let expected_res = format!("{}x{}", w, h);
            assert!(
                plan.filename_suffix.contains(&expected_res),
                "Suffix '{}' should contain resolution '{}'",
                plan.filename_suffix,
                expected_res
            );
        }
    }

    #[test]
    fn test_youtube4k_bitrate_exceeds_1080p() {
        let p4k = PlatformVariant::YouTube4K.to_preset().expect("valid");
        let p1080 = PlatformVariant::YouTube1080p.to_preset().expect("valid");
        let br4k = p4k.video.as_ref().and_then(|v| v.bitrate).unwrap_or(0);
        let br1080 = p1080.video.as_ref().and_then(|v| v.bitrate).unwrap_or(0);
        assert!(
            br4k >= br1080,
            "4K bitrate ({}) should be >= 1080p bitrate ({})",
            br4k,
            br1080
        );
    }

    #[test]
    fn test_all_platforms_have_correct_name() {
        assert_eq!(Platform::YouTube.name(), "YouTube");
        assert_eq!(Platform::Instagram.name(), "Instagram");
        assert_eq!(Platform::TikTok.name(), "TikTok");
        assert_eq!(Platform::Twitter.name(), "Twitter/X");
        assert_eq!(Platform::Facebook.name(), "Facebook");
    }

    #[test]
    fn test_constraint_resolution_violation() {
        // Oversized resolution for TikTok (max 1080x1920)
        let violations = validate_constraints(
            PlatformVariant::TikTokStandard,
            30.0,
            10_000_000,
            3840, // 4K width > 1080
            2160,
            30.0,
        );
        assert!(
            violations
                .iter()
                .any(|v| v.kind == ConstraintKind::ResolutionTooHigh),
            "Should report resolution violation"
        );
    }

    #[test]
    fn test_plan_with_no_violations() {
        let variants = vec![PlatformVariant::YouTube1080p, PlatformVariant::FacebookFeed];
        let plans = plan_multi_platform_export(
            &variants, 60.0,       // 1 min, well within limits
            10_000_000, // 10 MB
            1920, 1080, 30.0,
        )
        .expect("should succeed");
        for plan in &plans {
            assert!(
                plan.violations.is_empty(),
                "{:?} should have no violations for normal parameters",
                plan.variant
            );
        }
    }

    #[test]
    fn test_facebook_ads_resolution_correct() {
        let preset = facebook_ads().expect("valid");
        let video = preset.video.expect("has video");
        assert_eq!(video.width, Some(1280));
        assert_eq!(video.height, Some(720));
        // Ads use two-pass for quality
        assert!(video.two_pass);
    }

    #[test]
    fn test_empty_variants_produces_empty_plans() {
        let plans = plan_multi_platform_export(&[], 60.0, 10_000_000, 1920, 1080, 30.0)
            .expect("should succeed");
        assert!(plans.is_empty());
    }
}
