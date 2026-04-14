//! High-level preset API: matching, inheritance, validation, export, ABR ladder generation,
//! platform presets, codec profiles, and quality tiers.
//!
//! This module provides the concrete implementations specified in the task:
//! - `PresetMatcher` — score-based content-to-preset matching
//! - `PresetInheritance` — field-level override resolution
//! - `PresetValidator` — required-field and range checks
//! - `PresetExporter` — FFmpeg args and XML export
//! - `AbrLadderGenerator` — content-adaptive ABR ladder
//! - `PlatformPresets` — named platform presets (YouTube, Netflix, Twitch)
//! - `CodecProfileRegistry` — built-in AV1/VP9/VP8 profiles
//! - `QualityTier` — high/medium/low quality convenience presets

#![allow(dead_code)]

use std::collections::HashMap;
use oximedia_transcode::{PresetConfig, QualityMode};

use crate::{AbrLadder, Preset, PresetCategory, PresetMetadata};

// ─── ContentAnalysis ─────────────────────────────────────────────────────────

/// Analysis results of source media used for preset matching and ABR generation.
#[derive(Debug, Clone)]
pub struct ContentAnalysis {
    /// Source video width in pixels.
    pub width: u32,
    /// Source video height in pixels.
    pub height: u32,
    /// Source video bitrate in bits/s.
    pub video_bitrate: u64,
    /// Source codec identifier (e.g. `"h264"`, `"hevc"`, `"av1"`).
    pub codec: String,
    /// Content complexity score in `[0.0, 1.0]`; 1.0 = very complex.
    pub complexity: f32,
    /// Whether the source has an audio stream.
    pub has_audio: bool,
    /// Source frame rate as `(num, den)`.
    pub frame_rate: (u32, u32),
}

impl ContentAnalysis {
    /// Create a new content analysis with sensible defaults.
    #[must_use]
    pub fn new(width: u32, height: u32, video_bitrate: u64, codec: impl Into<String>) -> Self {
        Self {
            width,
            height,
            video_bitrate,
            codec: codec.into(),
            complexity: 0.5,
            has_audio: true,
            frame_rate: (30, 1),
        }
    }

    /// Override complexity.
    #[must_use]
    pub fn with_complexity(mut self, c: f32) -> Self {
        self.complexity = c.clamp(0.0, 1.0);
        self
    }
}

// ─── PresetMatcher ────────────────────────────────────────────────────────────

/// Scores presets against a `ContentAnalysis` and returns the best match.
///
/// Score formula per preset:
/// ```text
/// score = codec_match * 0.4 + resolution_match * 0.3 + bitrate_efficiency * 0.3
/// ```
pub struct PresetMatcher;

impl PresetMatcher {
    /// Find the best-scoring preset from `presets` for the given content.
    ///
    /// Returns `None` when `presets` is empty.
    #[must_use]
    pub fn find_best<'a>(content: &ContentAnalysis, presets: &'a [Preset]) -> Option<&'a Preset> {
        presets
            .iter()
            .max_by(|a, b| {
                let sa = Self::score(content, a);
                let sb = Self::score(content, b);
                sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Compute a score in `[0.0, 1.0]` for a single preset against the content.
    #[must_use]
    pub fn score(content: &ContentAnalysis, preset: &Preset) -> f32 {
        let codec_match = Self::codec_match(content, preset);
        let resolution_match = Self::resolution_match(content, preset);
        let bitrate_efficiency = Self::bitrate_efficiency(content, preset);
        codec_match * 0.4 + resolution_match * 0.3 + bitrate_efficiency * 0.3
    }

    fn codec_match(content: &ContentAnalysis, preset: &Preset) -> f32 {
        match &preset.config.video_codec {
            Some(codec) if codec.to_lowercase() == content.codec.to_lowercase() => 1.0,
            Some(_) => 0.5, // different codec but still usable
            None => 0.3,
        }
    }

    fn resolution_match(content: &ContentAnalysis, preset: &Preset) -> f32 {
        let preset_w = preset.config.width.unwrap_or(content.width);
        let preset_h = preset.config.height.unwrap_or(content.height);

        let w_ratio = if content.width == 0 {
            1.0
        } else {
            (preset_w as f32 / content.width as f32).min(1.0)
        };
        let h_ratio = if content.height == 0 {
            1.0
        } else {
            (preset_h as f32 / content.height as f32).min(1.0)
        };

        (w_ratio + h_ratio) / 2.0
    }

    fn bitrate_efficiency(content: &ContentAnalysis, preset: &Preset) -> f32 {
        match preset.config.video_bitrate {
            Some(br) => {
                if content.video_bitrate == 0 {
                    return 0.5;
                }
                let ratio = br as f32 / content.video_bitrate as f32;
                // Best efficiency when ratio ∈ [0.4, 1.0]; falls off outside
                if ratio <= 1.0 {
                    (ratio / 1.0).max(0.1)
                } else {
                    (1.0 / ratio).max(0.1)
                }
            }
            None => 0.3,
        }
    }
}

// ─── PresetOverride ──────────────────────────────────────────────────────────

/// Field-level overrides to apply on top of a base preset.
#[derive(Debug, Clone, Default)]
pub struct PresetOverride {
    /// Override video codec.
    pub video_codec: Option<String>,
    /// Override audio codec.
    pub audio_codec: Option<String>,
    /// Override video bitrate (bits/s).
    pub video_bitrate: Option<u64>,
    /// Override audio bitrate (bits/s).
    pub audio_bitrate: Option<u64>,
    /// Override output width.
    pub width: Option<u32>,
    /// Override output height.
    pub height: Option<u32>,
    /// Override frame rate `(numerator, denominator)`.
    pub frame_rate: Option<(u32, u32)>,
    /// Override preset name.
    pub name: Option<String>,
    /// Override preset description.
    pub description: Option<String>,
    /// Override container format.
    pub container: Option<String>,
}

impl PresetOverride {
    /// Create an empty override (no changes).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the video bitrate.
    #[must_use]
    pub fn with_video_bitrate(mut self, bps: u64) -> Self {
        self.video_bitrate = Some(bps);
        self
    }

    /// Override the resolution.
    #[must_use]
    pub fn with_resolution(mut self, w: u32, h: u32) -> Self {
        self.width = Some(w);
        self.height = Some(h);
        self
    }

    /// Override the video codec.
    #[must_use]
    pub fn with_video_codec(mut self, codec: impl Into<String>) -> Self {
        self.video_codec = Some(codec.into());
        self
    }
}

// ─── PresetInheritance ────────────────────────────────────────────────────────

/// Resolves a derived preset by merging field-level overrides onto a base preset.
pub struct PresetInheritance;

impl PresetInheritance {
    /// Create a new preset by copying `base` and applying `overrides`.
    ///
    /// Fields in `overrides` that are `Some(...)` replace the corresponding
    /// fields in the copy of `base`; `None` fields leave the base unchanged.
    #[must_use]
    pub fn resolve(base: &Preset, overrides: &PresetOverride) -> Preset {
        let mut result = base.clone();

        if let Some(ref name) = overrides.name {
            result.metadata.name = name.clone();
        }
        if let Some(ref desc) = overrides.description {
            result.metadata.description = desc.clone();
        }
        if let Some(ref vc) = overrides.video_codec {
            result.config.video_codec = Some(vc.clone());
        }
        if let Some(ref ac) = overrides.audio_codec {
            result.config.audio_codec = Some(ac.clone());
        }
        if let Some(vbr) = overrides.video_bitrate {
            result.config.video_bitrate = Some(vbr);
        }
        if let Some(abr) = overrides.audio_bitrate {
            result.config.audio_bitrate = Some(abr);
        }
        if let Some(w) = overrides.width {
            result.config.width = Some(w);
        }
        if let Some(h) = overrides.height {
            result.config.height = Some(h);
        }
        if let Some(fr) = overrides.frame_rate {
            result.config.frame_rate = Some(fr);
        }
        if let Some(ref container) = overrides.container {
            result.config.container = Some(container.clone());
        }

        result
    }
}

// ─── PresetValidator ──────────────────────────────────────────────────────────

/// Validates a preset for required fields and value ranges.
pub struct PresetValidator;

impl PresetValidator {
    /// Validate a preset and return a list of human-readable error messages.
    ///
    /// An empty return value means the preset is valid.
    #[must_use]
    pub fn validate(preset: &Preset) -> Vec<String> {
        let mut errors = Vec::new();

        // Required metadata fields
        if preset.metadata.id.is_empty() {
            errors.push("Preset ID must not be empty".to_string());
        }
        if preset.metadata.name.is_empty() {
            errors.push("Preset name must not be empty".to_string());
        }

        // Video codec
        if preset.config.video_codec.is_none() {
            errors.push("video_codec is required".to_string());
        }

        // Resolution: both or neither
        match (preset.config.width, preset.config.height) {
            (Some(0), _) => {
                errors.push("width must be greater than 0".to_string());
            }
            (_, Some(0)) => {
                errors.push("height must be greater than 0".to_string());
            }
            (Some(w), None) => {
                errors.push(format!("height is required when width ({w}) is set"));
            }
            (None, Some(h)) => {
                errors.push(format!("width is required when height ({h}) is set"));
            }
            _ => {}
        }

        // Bitrates: must be positive if set
        if let Some(vbr) = preset.config.video_bitrate {
            if vbr == 0 {
                errors.push("video_bitrate must be greater than 0".to_string());
            }
        }
        if let Some(abr) = preset.config.audio_bitrate {
            if abr == 0 {
                errors.push("audio_bitrate must be greater than 0".to_string());
            }
        }

        // Frame rate: denominator must not be zero
        if let Some((num, den)) = preset.config.frame_rate {
            if den == 0 {
                errors.push("frame_rate denominator must not be zero".to_string());
            }
            if num == 0 {
                errors.push("frame_rate numerator must not be zero".to_string());
            }
        }

        errors
    }

    /// Return `true` if the preset passes all validation checks.
    #[must_use]
    pub fn is_valid(preset: &Preset) -> bool {
        Self::validate(preset).is_empty()
    }
}

// ─── PresetExporter ──────────────────────────────────────────────────────────

/// Exports a preset to various text formats consumable by external tools.
pub struct PresetExporter;

impl PresetExporter {
    /// Produce a list of FFmpeg command-line argument strings from a preset.
    ///
    /// Example output: `["-c:v", "libsvtav1", "-crf", "30", "-b:v", "5M", ...]`
    #[must_use]
    pub fn to_ffmpeg_args(preset: &Preset) -> Vec<String> {
        let mut args = Vec::new();

        // Video codec mapping
        if let Some(ref codec) = preset.config.video_codec {
            let ffmpeg_codec = Self::codec_to_ffmpeg(codec);
            args.push("-c:v".to_string());
            args.push(ffmpeg_codec);
        }

        // Audio codec
        if let Some(ref codec) = preset.config.audio_codec {
            let ffmpeg_audio = Self::audio_codec_to_ffmpeg(codec);
            args.push("-c:a".to_string());
            args.push(ffmpeg_audio);
        }

        // Bitrates
        if let Some(vbr) = preset.config.video_bitrate {
            args.push("-b:v".to_string());
            args.push(Self::format_bitrate(vbr));
        }
        if let Some(abr) = preset.config.audio_bitrate {
            args.push("-b:a".to_string());
            args.push(Self::format_bitrate(abr));
        }

        // Resolution
        if let (Some(w), Some(h)) = (preset.config.width, preset.config.height) {
            args.push("-vf".to_string());
            args.push(format!("scale={w}:{h}"));
        }

        // Frame rate
        if let Some((num, den)) = preset.config.frame_rate {
            args.push("-r".to_string());
            if den == 1 {
                args.push(format!("{num}"));
            } else {
                args.push(format!("{num}/{den}"));
            }
        }

        // Quality mode → CRF
        if let Some(ref qm) = preset.config.quality_mode {
            let crf = Self::quality_mode_to_crf(qm);
            args.push("-crf".to_string());
            args.push(format!("{crf}"));
        }

        // Container format
        if let Some(ref container) = preset.config.container {
            args.push("-f".to_string());
            args.push(container.clone());
        }

        args
    }

    /// Produce an XML representation of the preset.
    ///
    /// ```xml
    /// <preset id="youtube-1080p" name="YouTube 1080p">
    ///   <video_codec>h264</video_codec>
    ///   ...
    /// </preset>
    /// ```
    #[must_use]
    pub fn to_xml(preset: &Preset) -> String {
        let mut xml = String::new();
        let id = Self::escape_xml(&preset.metadata.id);
        let name = Self::escape_xml(&preset.metadata.name);

        xml.push_str(&format!(
            "<preset id=\"{id}\" name=\"{name}\">\n"
        ));
        xml.push_str(&format!(
            "  <description>{}</description>\n",
            Self::escape_xml(&preset.metadata.description)
        ));

        if let Some(ref vc) = preset.config.video_codec {
            xml.push_str(&format!("  <video_codec>{}</video_codec>\n", Self::escape_xml(vc)));
        }
        if let Some(ref ac) = preset.config.audio_codec {
            xml.push_str(&format!("  <audio_codec>{}</audio_codec>\n", Self::escape_xml(ac)));
        }
        if let Some(vbr) = preset.config.video_bitrate {
            xml.push_str(&format!("  <video_bitrate>{vbr}</video_bitrate>\n"));
        }
        if let Some(abr) = preset.config.audio_bitrate {
            xml.push_str(&format!("  <audio_bitrate>{abr}</audio_bitrate>\n"));
        }
        if let (Some(w), Some(h)) = (preset.config.width, preset.config.height) {
            xml.push_str(&format!("  <width>{w}</width>\n  <height>{h}</height>\n"));
        }
        if let Some((num, den)) = preset.config.frame_rate {
            xml.push_str(&format!("  <frame_rate>{num}/{den}</frame_rate>\n"));
        }
        if let Some(ref container) = preset.config.container {
            xml.push_str(&format!("  <container>{}</container>\n", Self::escape_xml(container)));
        }

        xml.push_str("</preset>\n");
        xml
    }

    // ── Helpers ──

    fn codec_to_ffmpeg(codec: &str) -> String {
        match codec.to_lowercase().as_str() {
            "av1" | "svt-av1" => "libsvtav1",
            "vp9" => "libvpx-vp9",
            "vp8" => "libvpx",
            "h264" | "avc" => "libx264",
            "hevc" | "h265" => "libx265",
            "ffv1" => "ffv1",
            "prores" | "prores_proxy" => "prores_ks",
            "dnxhd" | "dnxhr" => "dnxhd",
            other => other,
        }
        .to_string()
    }

    fn audio_codec_to_ffmpeg(codec: &str) -> String {
        match codec.to_lowercase().as_str() {
            "aac" => "aac",
            "opus" => "libopus",
            "vorbis" => "libvorbis",
            "mp3" => "libmp3lame",
            "flac" => "flac",
            other => other,
        }
        .to_string()
    }

    fn format_bitrate(bps: u64) -> String {
        if bps >= 1_000_000 {
            format!("{}M", bps / 1_000_000)
        } else if bps >= 1_000 {
            format!("{}K", bps / 1_000)
        } else {
            format!("{bps}")
        }
    }

    fn quality_mode_to_crf(qm: &QualityMode) -> u32 {
        match qm {
            QualityMode::VeryHigh => 18,
            QualityMode::High => 20,
            QualityMode::Medium => 28,
            QualityMode::Low => 38,
            QualityMode::Custom => 28,
        }
    }

    fn escape_xml(s: &str) -> String {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
    }
}

// ─── AbrLadderGenerator ──────────────────────────────────────────────────────

/// Generates a 4-rung ABR ladder based on content complexity.
///
/// Simple content (complexity ≤ 0.3) uses lower bitrates; complex content
/// (complexity ≥ 0.7) uses higher bitrates.
pub struct AbrLadderGenerator;

impl AbrLadderGenerator {
    /// Generate a 4-rung ABR ladder for the given content.
    ///
    /// Rung order: 240p → 480p → 720p → 1080p (ascending bitrate).
    #[must_use]
    pub fn generate(content: &ContentAnalysis) -> AbrLadder {
        let complexity = content.complexity;

        // Bitrate multiplier: 0.5× for simple, 1.0× for moderate, 2.0× for complex
        let multiplier = if complexity <= 0.3 {
            0.5_f32
        } else if complexity >= 0.7 {
            2.0
        } else {
            1.0
        };

        let rungs = [
            (240_u32, 400_000_u64),
            (480, 1_500_000),
            (720, 3_000_000),
            (1080, 6_000_000),
        ];

        let mut ladder = AbrLadder::new("adaptive-ladder", "HLS");

        for (height, base_bitrate) in rungs {
            let bitrate = ((base_bitrate as f32 * multiplier) as u64).max(100_000);
            let width = height * 16 / 9;

            let metadata = PresetMetadata::new(
                &format!("abr-{height}p"),
                &format!("ABR {height}p"),
                PresetCategory::Streaming("HLS".to_string()),
            )
            .with_tag("abr")
            .with_tag(&format!("{height}p"));

            let config = PresetConfig {
                video_codec: Some("h264".to_string()),
                audio_codec: Some("aac".to_string()),
                video_bitrate: Some(bitrate),
                audio_bitrate: Some(128_000),
                width: Some(width),
                height: Some(height),
                frame_rate: Some((30, 1)),
                quality_mode: Some(QualityMode::Medium),
                container: Some("mp4".to_string()),
                audio_channel_layout: None,
            };

            ladder = ladder.add_rung(height, bitrate, Preset::new(metadata, config));
        }

        ladder
    }
}

// ─── PlatformPresets ─────────────────────────────────────────────────────────

/// Factory methods for common platform-specific presets.
pub struct PlatformPresets;

impl PlatformPresets {
    /// YouTube 1080p H.264/AAC preset.
    #[must_use]
    pub fn youtube_1080p() -> Preset {
        let metadata = PresetMetadata::new(
            "platform-youtube-1080p",
            "YouTube 1080p",
            PresetCategory::Platform("YouTube".to_string()),
        )
        .with_description("YouTube recommended 1080p H.264 upload preset")
        .with_tag("youtube")
        .with_tag("1080p")
        .with_tag("h264")
        .with_target("YouTube");

        let config = PresetConfig {
            video_codec: Some("h264".to_string()),
            audio_codec: Some("aac".to_string()),
            video_bitrate: Some(8_000_000),
            audio_bitrate: Some(384_000),
            width: Some(1920),
            height: Some(1080),
            frame_rate: Some((30, 1)),
            quality_mode: Some(QualityMode::High),
            container: Some("mp4".to_string()),
            audio_channel_layout: None,
        };

        Preset::new(metadata, config)
    }

    /// YouTube 4K (2160p) VP9/Opus preset.
    #[must_use]
    pub fn youtube_4k() -> Preset {
        let metadata = PresetMetadata::new(
            "platform-youtube-4k",
            "YouTube 4K (2160p)",
            PresetCategory::Platform("YouTube".to_string()),
        )
        .with_description("YouTube 4K VP9 upload preset")
        .with_tag("youtube")
        .with_tag("4k")
        .with_tag("2160p")
        .with_tag("vp9")
        .with_target("YouTube 4K");

        let config = PresetConfig {
            video_codec: Some("vp9".to_string()),
            audio_codec: Some("opus".to_string()),
            video_bitrate: Some(40_000_000),
            audio_bitrate: Some(512_000),
            width: Some(3840),
            height: Some(2160),
            frame_rate: Some((60, 1)),
            quality_mode: Some(QualityMode::High),
            container: Some("webm".to_string()),
            audio_channel_layout: None,
        };

        Preset::new(metadata, config)
    }

    /// Netflix streaming H.264 preset (HD, VMAF-optimised).
    #[must_use]
    pub fn netflix_stream() -> Preset {
        let metadata = PresetMetadata::new(
            "platform-netflix-stream",
            "Netflix Streaming HD",
            PresetCategory::Platform("Netflix".to_string()),
        )
        .with_description("Netflix-optimised H.264 streaming preset")
        .with_tag("netflix")
        .with_tag("1080p")
        .with_tag("h264")
        .with_target("Netflix");

        let config = PresetConfig {
            video_codec: Some("h264".to_string()),
            audio_codec: Some("aac".to_string()),
            video_bitrate: Some(6_000_000),
            audio_bitrate: Some(192_000),
            width: Some(1920),
            height: Some(1080),
            frame_rate: Some((24, 1)),
            quality_mode: Some(QualityMode::VeryHigh),
            container: Some("mp4".to_string()),
            audio_channel_layout: None,
        };

        Preset::new(metadata, config)
    }

    /// Twitch 720p60 live-streaming preset.
    #[must_use]
    pub fn twitch_720p() -> Preset {
        let metadata = PresetMetadata::new(
            "platform-twitch-720p",
            "Twitch 720p60",
            PresetCategory::Platform("Twitch".to_string()),
        )
        .with_description("Twitch live-streaming 720p60 preset")
        .with_tag("twitch")
        .with_tag("720p")
        .with_tag("h264")
        .with_tag("live")
        .with_target("Twitch");

        let config = PresetConfig {
            video_codec: Some("h264".to_string()),
            audio_codec: Some("aac".to_string()),
            video_bitrate: Some(6_000_000),
            audio_bitrate: Some(160_000),
            width: Some(1280),
            height: Some(720),
            frame_rate: Some((60, 1)),
            quality_mode: Some(QualityMode::Medium),
            container: Some("flv".to_string()),
            audio_channel_layout: None,
        };

        Preset::new(metadata, config)
    }
}

// ─── CodecProfile ────────────────────────────────────────────────────────────

/// A codec-specific encoding profile.
#[derive(Debug, Clone)]
pub struct CodecProfile {
    /// Codec identifier (e.g. `"av1"`, `"vp9"`).
    pub codec: String,
    /// Human-readable profile name.
    pub name: String,
    /// Recommended CRF range (min, max).
    pub crf_range: (u32, u32),
    /// Recommended bitrate range in bits/s (min, max).
    pub bitrate_range: (u64, u64),
    /// Supported container formats.
    pub containers: Vec<String>,
    /// Profile-level description.
    pub description: String,
}

impl CodecProfile {
    /// Create a new codec profile.
    #[must_use]
    pub fn new(
        codec: impl Into<String>,
        name: impl Into<String>,
        crf_range: (u32, u32),
        bitrate_range: (u64, u64),
        containers: Vec<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            codec: codec.into(),
            name: name.into(),
            crf_range,
            bitrate_range,
            containers,
            description: description.into(),
        }
    }
}

/// A registry of codec profiles with O(1) lookup.
#[derive(Default)]
pub struct CodecProfileRegistry {
    profiles: HashMap<String, CodecProfile>,
}

impl CodecProfileRegistry {
    /// Create a new registry pre-populated with built-in AV1, VP9, and VP8 profiles.
    #[must_use]
    pub fn with_builtins() -> Self {
        let mut reg = Self::new();
        reg.register_builtins();
        reg
    }

    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self { profiles: HashMap::new() }
    }

    /// Register a codec profile.
    pub fn register(&mut self, codec: &str, profile: CodecProfile) {
        self.profiles.insert(codec.to_lowercase(), profile);
    }

    /// Look up a profile by codec name (case-insensitive).
    #[must_use]
    pub fn get(&self, codec: &str) -> Option<&CodecProfile> {
        self.profiles.get(&codec.to_lowercase())
    }

    /// Number of registered profiles.
    #[must_use]
    pub fn len(&self) -> usize {
        self.profiles.len()
    }

    /// Return `true` if no profiles are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }

    /// Populate the registry with built-in AV1, VP9, and VP8 profiles.
    fn register_builtins(&mut self) {
        self.register(
            "av1",
            CodecProfile::new(
                "av1",
                "AV1 Standard",
                (20, 63),
                (1_000_000, 50_000_000),
                vec!["mp4".to_string(), "webm".to_string(), "mkv".to_string()],
                "AOMedia Video 1 — royalty-free, state-of-the-art compression",
            ),
        );
        self.register(
            "vp9",
            CodecProfile::new(
                "vp9",
                "VP9 Standard",
                (15, 55),
                (500_000, 30_000_000),
                vec!["webm".to_string(), "mp4".to_string()],
                "Google VP9 — widely supported, patent-free",
            ),
        );
        self.register(
            "vp8",
            CodecProfile::new(
                "vp8",
                "VP8 Standard",
                (4, 63),
                (250_000, 10_000_000),
                vec!["webm".to_string()],
                "Google VP8 — legacy codec with broad browser support",
            ),
        );
    }
}

// ─── QualityTier ─────────────────────────────────────────────────────────────

/// Convenience factory for quality-tiered presets.
pub struct QualityTier;

impl QualityTier {
    /// High-quality AV1 preset with CRF 18.
    #[must_use]
    pub fn high() -> Preset {
        let metadata = PresetMetadata::new(
            "quality-tier-high",
            "High Quality",
            PresetCategory::Quality("High".to_string()),
        )
        .with_description("High quality AV1 encoding, CRF 18")
        .with_tag("high")
        .with_tag("av1");

        let config = PresetConfig {
            video_codec: Some("av1".to_string()),
            audio_codec: Some("opus".to_string()),
            video_bitrate: None,
            audio_bitrate: Some(192_000),
            width: None,
            height: None,
            frame_rate: None,
            quality_mode: Some(QualityMode::VeryHigh),
            container: Some("mp4".to_string()),
            audio_channel_layout: None,
        };

        Preset::new(metadata, config)
    }

    /// Medium-quality H.264 preset with CRF 28.
    #[must_use]
    pub fn medium() -> Preset {
        let metadata = PresetMetadata::new(
            "quality-tier-medium",
            "Medium Quality",
            PresetCategory::Quality("Medium".to_string()),
        )
        .with_description("Balanced H.264 encoding, CRF 28")
        .with_tag("medium")
        .with_tag("h264");

        let config = PresetConfig {
            video_codec: Some("h264".to_string()),
            audio_codec: Some("aac".to_string()),
            video_bitrate: None,
            audio_bitrate: Some(128_000),
            width: None,
            height: None,
            frame_rate: None,
            quality_mode: Some(QualityMode::Medium),
            container: Some("mp4".to_string()),
            audio_channel_layout: None,
        };

        Preset::new(metadata, config)
    }

    /// Low-quality H.264 preset (CRF 38 equivalent).
    #[must_use]
    pub fn low() -> Preset {
        let metadata = PresetMetadata::new(
            "quality-tier-low",
            "Low Quality",
            PresetCategory::Quality("Low".to_string()),
        )
        .with_description("Small file size H.264 encoding, CRF 38")
        .with_tag("low")
        .with_tag("h264");

        let config = PresetConfig {
            video_codec: Some("h264".to_string()),
            audio_codec: Some("aac".to_string()),
            video_bitrate: None,
            audio_bitrate: Some(64_000),
            width: None,
            height: None,
            frame_rate: None,
            quality_mode: Some(QualityMode::Low),
            container: Some("mp4".to_string()),
            audio_channel_layout: None,
        };

        Preset::new(metadata, config)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_content_1080p() -> ContentAnalysis {
        ContentAnalysis::new(1920, 1080, 8_000_000, "h264")
    }

    fn make_1080p_preset() -> Preset {
        PlatformPresets::youtube_1080p()
    }

    // ── PlatformPresets ──

    #[test]
    fn test_youtube_1080p_resolution() {
        let p = PlatformPresets::youtube_1080p();
        assert_eq!(p.config.width, Some(1920));
        assert_eq!(p.config.height, Some(1080));
    }

    #[test]
    fn test_youtube_1080p_codec() {
        let p = PlatformPresets::youtube_1080p();
        assert_eq!(p.config.video_codec.as_deref(), Some("h264"));
    }

    #[test]
    fn test_youtube_4k_resolution() {
        let p = PlatformPresets::youtube_4k();
        assert_eq!(p.config.width, Some(3840));
        assert_eq!(p.config.height, Some(2160));
    }

    #[test]
    fn test_twitch_720p_resolution() {
        let p = PlatformPresets::twitch_720p();
        assert_eq!(p.config.height, Some(720));
    }

    // ── AbrLadderGenerator ──

    #[test]
    fn test_abr_ladder_has_4_rungs() {
        let content = make_content_1080p();
        let ladder = AbrLadderGenerator::generate(&content);
        assert_eq!(ladder.rungs.len(), 4);
    }

    #[test]
    fn test_abr_ladder_ascending_bitrates() {
        let content = make_content_1080p();
        let ladder = AbrLadderGenerator::generate(&content);
        for window in ladder.rungs.windows(2) {
            assert!(
                window[0].bitrate <= window[1].bitrate,
                "Bitrates must be non-decreasing: {} > {}",
                window[0].bitrate,
                window[1].bitrate
            );
        }
    }

    #[test]
    fn test_abr_ladder_complex_higher_bitrates() {
        let simple = ContentAnalysis::new(1280, 720, 3_000_000, "h264").with_complexity(0.1);
        let complex = ContentAnalysis::new(1280, 720, 3_000_000, "h264").with_complexity(0.9);

        let simple_ladder = AbrLadderGenerator::generate(&simple);
        let complex_ladder = AbrLadderGenerator::generate(&complex);

        // Top rung of complex should have higher bitrate than simple
        let simple_top = simple_ladder.rungs.last().map(|r| r.bitrate).unwrap_or(0);
        let complex_top = complex_ladder.rungs.last().map(|r| r.bitrate).unwrap_or(0);
        assert!(complex_top > simple_top);
    }

    // ── PresetExporter ──

    #[test]
    fn test_ffmpeg_args_contains_codec_name() {
        let p = PlatformPresets::youtube_1080p();
        let args = PresetExporter::to_ffmpeg_args(&p);
        let joined = args.join(" ");
        // "-c:v" followed by ffmpeg codec name
        assert!(joined.contains("libx264") || joined.contains("h264"), "Should contain codec: {joined}");
    }

    #[test]
    fn test_ffmpeg_args_codec_flag() {
        let p = PlatformPresets::youtube_1080p();
        let args = PresetExporter::to_ffmpeg_args(&p);
        assert!(args.contains(&"-c:v".to_string()));
    }

    #[test]
    fn test_xml_export_contains_preset_name() {
        let p = PlatformPresets::youtube_1080p();
        let xml = PresetExporter::to_xml(&p);
        assert!(xml.contains("YouTube 1080p") || xml.contains("youtube"));
    }

    #[test]
    fn test_xml_export_well_formed() {
        let p = PlatformPresets::youtube_1080p();
        let xml = PresetExporter::to_xml(&p);
        assert!(xml.starts_with("<preset "));
        assert!(xml.contains("</preset>"));
    }

    // ── PresetMatcher ──

    #[test]
    fn test_preset_matcher_finds_best() {
        let content = make_content_1080p();
        let presets = vec![
            PlatformPresets::youtube_1080p(),
            PlatformPresets::twitch_720p(),
            PlatformPresets::youtube_4k(),
        ];
        let best = PresetMatcher::find_best(&content, &presets);
        assert!(best.is_some());
    }

    #[test]
    fn test_preset_matcher_empty_returns_none() {
        let content = make_content_1080p();
        let best = PresetMatcher::find_best(&content, &[]);
        assert!(best.is_none());
    }

    // ── PresetInheritance ──

    #[test]
    fn test_preset_inheritance_overrides_bitrate() {
        let base = PlatformPresets::youtube_1080p();
        let overrides = PresetOverride::new().with_video_bitrate(12_000_000);
        let derived = PresetInheritance::resolve(&base, &overrides);
        assert_eq!(derived.config.video_bitrate, Some(12_000_000));
    }

    #[test]
    fn test_preset_inheritance_keeps_unset_fields() {
        let base = PlatformPresets::youtube_1080p();
        let overrides = PresetOverride::new().with_video_bitrate(12_000_000);
        let derived = PresetInheritance::resolve(&base, &overrides);
        // Resolution should be unchanged from base
        assert_eq!(derived.config.width, Some(1920));
        assert_eq!(derived.config.height, Some(1080));
    }

    #[test]
    fn test_preset_inheritance_overrides_codec() {
        let base = PlatformPresets::youtube_1080p();
        let overrides = PresetOverride::new().with_video_codec("av1");
        let derived = PresetInheritance::resolve(&base, &overrides);
        assert_eq!(derived.config.video_codec.as_deref(), Some("av1"));
    }

    // ── PresetValidator ──

    #[test]
    fn test_validator_valid_preset() {
        let p = PlatformPresets::youtube_1080p();
        let errors = PresetValidator::validate(&p);
        assert!(errors.is_empty(), "Errors: {errors:?}");
    }

    #[test]
    fn test_validator_missing_codec() {
        let mut p = PlatformPresets::youtube_1080p();
        p.config.video_codec = None;
        let errors = PresetValidator::validate(&p);
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_validator_zero_bitrate() {
        let mut p = PlatformPresets::youtube_1080p();
        p.config.video_bitrate = Some(0);
        let errors = PresetValidator::validate(&p);
        assert!(!errors.is_empty());
    }

    // ── CodecProfileRegistry ──

    #[test]
    fn test_codec_profile_registry_has_builtins() {
        let reg = CodecProfileRegistry::with_builtins();
        assert!(reg.get("av1").is_some());
        assert!(reg.get("vp9").is_some());
        assert!(reg.get("vp8").is_some());
    }

    #[test]
    fn test_codec_profile_registry_custom() {
        let mut reg = CodecProfileRegistry::new();
        reg.register("hevc", CodecProfile::new(
            "hevc", "HEVC Main", (18, 51), (1_000_000, 50_000_000),
            vec!["mp4".to_string()], "H.265 Main Profile",
        ));
        assert!(reg.get("hevc").is_some());
    }

    #[test]
    fn test_codec_profile_case_insensitive() {
        let reg = CodecProfileRegistry::with_builtins();
        assert!(reg.get("AV1").is_some());
        assert!(reg.get("VP9").is_some());
    }

    // ── QualityTier ──

    #[test]
    fn test_quality_tier_high_is_very_high() {
        let p = QualityTier::high();
        assert_eq!(p.config.quality_mode, Some(QualityMode::VeryHigh));
    }

    #[test]
    fn test_quality_tier_medium_is_medium() {
        let p = QualityTier::medium();
        assert_eq!(p.config.quality_mode, Some(QualityMode::Medium));
    }

    #[test]
    fn test_quality_tier_low_is_low() {
        let p = QualityTier::low();
        assert_eq!(p.config.quality_mode, Some(QualityMode::Low));
    }
}
