// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Content-aware codec selection engine.
//!
//! Analyzes input media properties (resolution, frame rate, color depth, audio
//! channels, HDR status, content type) and recommends optimal codec/container
//! combinations for the given use case.
//!
//! This module builds on the heuristic classification in [`crate::smart`] and
//! adds a structured recommendation API with ranked alternatives, compatibility
//! notes, and estimated bitrate guidance.

use crate::formats::{AudioCodec, ChannelLayout, ContainerFormat, VideoCodec};
use crate::smart::{ContentType, MediaAnalysis};
use crate::{ConversionError, Result};
use serde::{Deserialize, Serialize};

// ── Input properties ───────────────────────────────────────────────────────

/// Detailed media properties used for codec selection.
///
/// This is a richer description than [`MediaAnalysis`] because it carries
/// information that is already parsed/validated (e.g. bit depth, channel
/// count) rather than raw probe output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MediaInputProperties {
    /// Width in pixels (0 if audio-only).
    pub width: u32,
    /// Height in pixels (0 if audio-only).
    pub height: u32,
    /// Frame rate (frames per second).  `None` for audio-only or still images.
    pub frame_rate: Option<f64>,
    /// Colour bit depth per channel (8, 10, 12, 16).
    pub bit_depth: u32,
    /// Whether the source carries HDR metadata (PQ / HLG / HDR10+).
    pub is_hdr: bool,
    /// Whether the source is interlaced.
    pub is_interlaced: bool,
    /// Whether the source has an alpha channel.
    pub has_alpha: bool,
    /// Number of audio channels (0 if video-only).
    pub audio_channels: u32,
    /// Audio sample rate in Hz (0 if video-only).
    pub audio_sample_rate: u32,
    /// Audio bit depth (16, 24, 32).  0 if video-only.
    pub audio_bit_depth: u32,
    /// Estimated content type (animation, live action, ...).
    pub content_type: ContentType,
    /// Duration in seconds.  `None` if unknown.
    pub duration_seconds: Option<f64>,
    /// Source file size in bytes.
    pub file_size: u64,
}

impl MediaInputProperties {
    /// Total pixel count.
    #[must_use]
    pub fn pixel_count(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Whether the source has video.
    #[must_use]
    pub fn has_video(&self) -> bool {
        self.width > 0 && self.height > 0
    }

    /// Whether the source has audio.
    #[must_use]
    pub fn has_audio(&self) -> bool {
        self.audio_channels > 0
    }

    /// Resolution tier for bitrate estimation.
    #[must_use]
    pub fn resolution_tier(&self) -> ResolutionTier {
        let pixels = self.pixel_count();
        if pixels >= 3840 * 2160 {
            ResolutionTier::Uhd4K
        } else if pixels >= 2560 * 1440 {
            ResolutionTier::Qhd
        } else if pixels >= 1920 * 1080 {
            ResolutionTier::FullHd
        } else if pixels >= 1280 * 720 {
            ResolutionTier::Hd
        } else if pixels >= 640 * 480 {
            ResolutionTier::Sd
        } else if pixels > 0 {
            ResolutionTier::Low
        } else {
            ResolutionTier::AudioOnly
        }
    }

    /// Build from a [`MediaAnalysis`].
    #[must_use]
    pub fn from_analysis(analysis: &MediaAnalysis, content_type: ContentType) -> Self {
        let (width, height) = analysis.resolution.unwrap_or((0, 0));
        Self {
            width,
            height,
            frame_rate: analysis.frame_rate,
            bit_depth: if analysis.is_hdr { 10 } else { 8 },
            is_hdr: analysis.is_hdr,
            is_interlaced: analysis.is_interlaced,
            has_alpha: false,
            audio_channels: if analysis.has_audio { 2 } else { 0 },
            audio_sample_rate: if analysis.has_audio { 48000 } else { 0 },
            audio_bit_depth: if analysis.has_audio { 16 } else { 0 },
            content_type,
            duration_seconds: analysis.duration_seconds,
            file_size: analysis.file_size,
        }
    }
}

/// Resolution tier classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ResolutionTier {
    /// Audio-only content
    AudioOnly,
    /// Below SD (< 480p)
    Low,
    /// SD (480p)
    Sd,
    /// HD (720p)
    Hd,
    /// Full HD (1080p)
    FullHd,
    /// QHD (1440p)
    Qhd,
    /// 4K UHD (2160p)
    Uhd4K,
}

impl ResolutionTier {
    /// Human-readable label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::AudioOnly => "Audio Only",
            Self::Low => "Low (<480p)",
            Self::Sd => "SD (480p)",
            Self::Hd => "HD (720p)",
            Self::FullHd => "Full HD (1080p)",
            Self::Qhd => "QHD (1440p)",
            Self::Uhd4K => "4K UHD (2160p+)",
        }
    }
}

// ── Codec recommendation output ────────────────────────────────────────────

/// A single codec/container recommendation with rationale.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CodecRecommendation {
    /// Recommended video codec (None for audio-only).
    pub video_codec: Option<VideoCodec>,
    /// Recommended audio codec (None for video-only).
    pub audio_codec: Option<AudioCodec>,
    /// Recommended container format.
    pub container: ContainerFormat,
    /// Recommended video bitrate in bits/s (None for audio-only or CRF mode).
    pub video_bitrate_bps: Option<u64>,
    /// Recommended audio bitrate in bits/s (None for lossless).
    pub audio_bitrate_bps: Option<u64>,
    /// Recommended CRF value (None if CBR/VBR is preferred).
    pub crf: Option<u32>,
    /// Whether two-pass encoding is recommended.
    pub two_pass: bool,
    /// Recommended channel layout for output audio.
    pub audio_channels: Option<ChannelLayout>,
    /// Recommended sample rate for output audio.
    pub audio_sample_rate: Option<u32>,
    /// Confidence score (0.0 - 1.0).
    pub confidence: f64,
    /// Human-readable rationale for this recommendation.
    pub rationale: String,
    /// Compatibility notes / caveats.
    pub notes: Vec<String>,
}

/// A ranked set of codec recommendations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CodecRecommendationSet {
    /// Primary (best) recommendation.
    pub primary: CodecRecommendation,
    /// Alternative recommendations, ordered by decreasing preference.
    pub alternatives: Vec<CodecRecommendation>,
    /// The input properties used to generate these recommendations.
    pub input_summary: String,
}

// ── Selector ───────────────────────────────────────────────────────────────

/// Content-aware codec selector.
///
/// Examines detailed input properties and produces ranked codec/container
/// recommendations.
#[derive(Debug, Clone)]
pub struct CodecSelector {
    /// Prefer AV1 even when VP9 would be acceptable.
    prefer_av1: bool,
    /// Allow lossless audio codecs.
    allow_lossless_audio: bool,
    /// Maximum acceptable encoding time multiplier (e.g. 2.0 = at most 2x
    /// slower than realtime).  `None` = no constraint.
    max_encode_time_factor: Option<f64>,
}

impl CodecSelector {
    /// Create a selector with default preferences.
    #[must_use]
    pub fn new() -> Self {
        Self {
            prefer_av1: true,
            allow_lossless_audio: true,
            max_encode_time_factor: None,
        }
    }

    /// Set whether to prefer AV1 over VP9 when both are viable.
    #[must_use]
    pub const fn with_av1_preference(mut self, prefer: bool) -> Self {
        self.prefer_av1 = prefer;
        self
    }

    /// Set whether lossless audio codecs (FLAC) are allowed.
    #[must_use]
    pub const fn with_lossless_audio(mut self, allow: bool) -> Self {
        self.allow_lossless_audio = allow;
        self
    }

    /// Set maximum acceptable encoding time factor.
    #[must_use]
    pub const fn with_max_encode_time(mut self, factor: Option<f64>) -> Self {
        self.max_encode_time_factor = factor;
        self
    }

    /// Produce a full set of recommendations for the given input properties.
    pub fn recommend_codec(&self, props: &MediaInputProperties) -> Result<CodecRecommendationSet> {
        if !props.has_video() && !props.has_audio() {
            return Err(ConversionError::InvalidInput(
                "Input has neither video nor audio".to_string(),
            ));
        }

        let primary = self.build_primary_recommendation(props);
        let alternatives = self.build_alternatives(props, &primary);

        let input_summary = self.summarize_input(props);

        Ok(CodecRecommendationSet {
            primary,
            alternatives,
            input_summary,
        })
    }

    // ── Internal recommendation builders ───────────────────────────────────

    fn build_primary_recommendation(&self, props: &MediaInputProperties) -> CodecRecommendation {
        let video_codec = if props.has_video() {
            Some(self.select_video_codec(props))
        } else {
            None
        };

        let audio_codec = if props.has_audio() {
            Some(self.select_audio_codec(props))
        } else {
            None
        };

        let container = self.select_container(video_codec, audio_codec, props);
        let video_bitrate = video_codec.map(|vc| self.estimate_video_bitrate(vc, props));
        let audio_bitrate = audio_codec.and_then(|ac| self.estimate_audio_bitrate(ac, props));
        let crf = video_codec.map(|vc| self.select_crf(vc, props));
        let two_pass = self.should_use_two_pass(props);

        let audio_channels = if props.has_audio() {
            Some(self.select_channel_layout(props))
        } else {
            None
        };

        let audio_sample_rate = if props.has_audio() {
            Some(self.select_sample_rate(props, audio_codec))
        } else {
            None
        };

        let rationale = self.generate_rationale(props, video_codec, audio_codec, &container);
        let notes = self.generate_notes(props, video_codec, audio_codec);

        CodecRecommendation {
            video_codec,
            audio_codec,
            container,
            video_bitrate_bps: video_bitrate,
            audio_bitrate_bps: audio_bitrate,
            crf,
            two_pass,
            audio_channels,
            audio_sample_rate,
            confidence: self.calculate_confidence(props),
            rationale,
            notes,
        }
    }

    fn build_alternatives(
        &self,
        props: &MediaInputProperties,
        primary: &CodecRecommendation,
    ) -> Vec<CodecRecommendation> {
        let mut alts = Vec::new();

        // Alternative video codecs
        if props.has_video() {
            let alt_codecs: Vec<VideoCodec> = [VideoCodec::Av1, VideoCodec::Vp9, VideoCodec::Vp8]
                .iter()
                .copied()
                .filter(|&c| Some(c) != primary.video_codec)
                .collect();

            for vc in alt_codecs.into_iter().take(2) {
                let audio_codec = primary.audio_codec;
                let container = self.select_container(Some(vc), audio_codec, props);
                let video_bitrate = self.estimate_video_bitrate(vc, props);
                let audio_bitrate =
                    audio_codec.and_then(|ac| self.estimate_audio_bitrate(ac, props));
                let crf = self.select_crf(vc, props);

                let rationale = format!(
                    "Alternative: {} in {} container",
                    vc.name(),
                    container.name()
                );

                alts.push(CodecRecommendation {
                    video_codec: Some(vc),
                    audio_codec,
                    container,
                    video_bitrate_bps: Some(video_bitrate),
                    audio_bitrate_bps: audio_bitrate,
                    crf: Some(crf),
                    two_pass: primary.two_pass,
                    audio_channels: primary.audio_channels,
                    audio_sample_rate: primary.audio_sample_rate,
                    confidence: self.calculate_confidence(props) * 0.8,
                    rationale,
                    notes: Vec::new(),
                });
            }
        }

        // For audio-only: offer Vorbis as alternative to Opus
        if !props.has_video() && props.has_audio() {
            if primary.audio_codec != Some(AudioCodec::Vorbis) {
                let alt_audio = AudioCodec::Vorbis;
                let container = ContainerFormat::Ogg;
                let bitrate = self.estimate_audio_bitrate(alt_audio, props);

                alts.push(CodecRecommendation {
                    video_codec: None,
                    audio_codec: Some(alt_audio),
                    container,
                    video_bitrate_bps: None,
                    audio_bitrate_bps: bitrate,
                    crf: None,
                    two_pass: false,
                    audio_channels: primary.audio_channels,
                    audio_sample_rate: Some(self.select_sample_rate(props, Some(alt_audio))),
                    confidence: self.calculate_confidence(props) * 0.7,
                    rationale: "Alternative: Vorbis in Ogg container for broad compatibility"
                        .to_string(),
                    notes: Vec::new(),
                });
            }

            // FLAC lossless alternative
            if self.allow_lossless_audio && primary.audio_codec != Some(AudioCodec::Flac) {
                alts.push(CodecRecommendation {
                    video_codec: None,
                    audio_codec: Some(AudioCodec::Flac),
                    container: ContainerFormat::Flac,
                    video_bitrate_bps: None,
                    audio_bitrate_bps: None,
                    crf: None,
                    two_pass: false,
                    audio_channels: primary.audio_channels,
                    audio_sample_rate: Some(self.select_sample_rate(props, Some(AudioCodec::Flac))),
                    confidence: self.calculate_confidence(props) * 0.6,
                    rationale: "Alternative: FLAC lossless for archival quality".to_string(),
                    notes: vec!["Lossless encoding produces larger files".to_string()],
                });
            }
        }

        alts
    }

    // ── Video codec selection ──────────────────────────────────────────────

    fn select_video_codec(&self, props: &MediaInputProperties) -> VideoCodec {
        // Speed constraint: if max encode time is tight, avoid AV1
        let speed_constrained = self.max_encode_time_factor.map_or(false, |f| f < 4.0);

        match props.content_type {
            ContentType::Animation => {
                // Animation: VP8 for simple content, VP9 for HD+
                if props.resolution_tier() >= ResolutionTier::FullHd {
                    VideoCodec::Vp9
                } else {
                    VideoCodec::Vp8
                }
            }
            ContentType::LiveAction => {
                // Live action: AV1 for best compression, unless speed-constrained
                if self.prefer_av1 && !speed_constrained {
                    VideoCodec::Av1
                } else {
                    VideoCodec::Vp9
                }
            }
            ContentType::ScreenRecording => {
                // Screen recording: VP9 for sharp text preservation
                VideoCodec::Vp9
            }
            ContentType::Slideshow => {
                // Slideshow: AV1 excels at intra-frame coding
                if !speed_constrained {
                    VideoCodec::Av1
                } else {
                    VideoCodec::Vp9
                }
            }
            ContentType::HighMotion => {
                // High motion: VP9 for speed/quality balance at high frame rates
                if props.frame_rate.unwrap_or(30.0) > 60.0 || speed_constrained {
                    VideoCodec::Vp9
                } else if self.prefer_av1 {
                    VideoCodec::Av1
                } else {
                    VideoCodec::Vp9
                }
            }
            ContentType::AudioOnly => VideoCodec::Vp8, // lightest if forced
            ContentType::Unknown => {
                // Default: VP9 as safe middle ground, AV1 if preferred
                if self.prefer_av1 && !speed_constrained {
                    VideoCodec::Av1
                } else {
                    VideoCodec::Vp9
                }
            }
        }
    }

    // ── Audio codec selection ──────────────────────────────────────────────

    fn select_audio_codec(&self, props: &MediaInputProperties) -> AudioCodec {
        // High bit-depth or lossless preference -> FLAC
        if self.allow_lossless_audio && props.audio_bit_depth >= 24 {
            return AudioCodec::Flac;
        }

        // Multi-channel content benefits from Opus's channel coupling
        if props.audio_channels > 2 {
            return AudioCodec::Opus;
        }

        // Standard: Opus is the best lossy codec
        AudioCodec::Opus
    }

    // ── Container selection ────────────────────────────────────────────────

    fn select_container(
        &self,
        video_codec: Option<VideoCodec>,
        audio_codec: Option<AudioCodec>,
        props: &MediaInputProperties,
    ) -> ContainerFormat {
        match (video_codec, audio_codec) {
            (Some(vc), _) => match vc {
                VideoCodec::Av1 => {
                    // AV1: Matroska for HDR/high-quality, WebM for web
                    if props.is_hdr || props.resolution_tier() >= ResolutionTier::Uhd4K {
                        ContainerFormat::Matroska
                    } else {
                        ContainerFormat::Webm
                    }
                }
                VideoCodec::Vp9 | VideoCodec::Vp8 => ContainerFormat::Webm,
                VideoCodec::Theora => ContainerFormat::Ogg,
            },
            (None, Some(ac)) => match ac {
                AudioCodec::Opus | AudioCodec::Vorbis => ContainerFormat::Ogg,
                AudioCodec::Flac => ContainerFormat::Flac,
                AudioCodec::Pcm => ContainerFormat::Wav,
            },
            (None, None) => ContainerFormat::Webm, // fallback
        }
    }

    // ── Bitrate estimation ─────────────────────────────────────────────────

    fn estimate_video_bitrate(&self, codec: VideoCodec, props: &MediaInputProperties) -> u64 {
        let base_bpp = match codec {
            VideoCodec::Av1 => 0.04,    // AV1 is most efficient
            VideoCodec::Vp9 => 0.06,    // VP9 is moderately efficient
            VideoCodec::Vp8 => 0.10,    // VP8 needs more bits
            VideoCodec::Theora => 0.12, // Theora is least efficient
        };

        // Content type multiplier
        let content_mult = match props.content_type {
            ContentType::Animation => 0.5,       // very compressible
            ContentType::ScreenRecording => 0.6, // large flat areas
            ContentType::Slideshow => 0.3,       // nearly static
            ContentType::LiveAction => 1.0,      // baseline
            ContentType::HighMotion => 1.5,      // needs more bits
            ContentType::AudioOnly => 0.1,       // minimal
            ContentType::Unknown => 1.0,
        };

        // HDR multiplier (10-bit content needs ~20% more bits)
        let hdr_mult = if props.is_hdr { 1.2 } else { 1.0 };

        let fps = props.frame_rate.unwrap_or(30.0);
        let pixels = props.pixel_count().max(1) as f64;

        (pixels * fps * base_bpp * content_mult * hdr_mult) as u64
    }

    fn estimate_audio_bitrate(
        &self,
        codec: AudioCodec,
        props: &MediaInputProperties,
    ) -> Option<u64> {
        match codec {
            AudioCodec::Flac | AudioCodec::Pcm => None, // lossless
            AudioCodec::Opus => {
                let per_channel = match props.audio_sample_rate {
                    0..=16000 => 32_000u64,
                    16001..=32000 => 48_000,
                    _ => 64_000,
                };
                Some(per_channel * u64::from(props.audio_channels.max(1)))
            }
            AudioCodec::Vorbis => {
                let per_channel = 96_000u64;
                Some(per_channel * u64::from(props.audio_channels.max(1)))
            }
        }
    }

    // ── CRF selection ──────────────────────────────────────────────────────

    fn select_crf(&self, codec: VideoCodec, props: &MediaInputProperties) -> u32 {
        let base_crf = match props.content_type {
            ContentType::Animation => 28,
            ContentType::ScreenRecording => 22,
            ContentType::Slideshow => 20,
            ContentType::LiveAction => 30,
            ContentType::HighMotion => 28,
            ContentType::AudioOnly => 40,
            ContentType::Unknown => 30,
        };

        // Scale CRF to codec's range
        let (min_q, max_q) = codec.quality_range();
        let range = max_q - min_q;
        // base_crf is on a 0-63 scale, remap
        let normalized = (base_crf as f64) / 63.0;
        let scaled = min_q as f64 + normalized * range as f64;

        // HDR content: slightly lower CRF for quality preservation
        let hdr_offset = if props.is_hdr {
            -(range as f64 * 0.05)
        } else {
            0.0
        };

        (scaled + hdr_offset)
            .round()
            .clamp(min_q as f64, max_q as f64) as u32
    }

    // ── Two-pass decision ──────────────────────────────────────────────────

    fn should_use_two_pass(&self, props: &MediaInputProperties) -> bool {
        // Two-pass is beneficial for:
        // 1. Long content (> 5 minutes)
        // 2. High resolution (> 1080p)
        // 3. High motion content
        // 4. HDR content
        // But not for:
        // - Speed-constrained scenarios
        // - Animation (compresses well with CRF)
        // - Very short content (< 30s)

        if self.max_encode_time_factor.map_or(false, |f| f < 2.0) {
            return false;
        }

        if props.content_type == ContentType::Animation {
            return false;
        }

        let long_content = props.duration_seconds.unwrap_or(0.0) > 300.0;
        let high_res = props.resolution_tier() >= ResolutionTier::Qhd;
        let high_motion = props.content_type == ContentType::HighMotion;
        let hdr = props.is_hdr;

        long_content || high_res || high_motion || hdr
    }

    // ── Channel layout ─────────────────────────────────────────────────────

    fn select_channel_layout(&self, props: &MediaInputProperties) -> ChannelLayout {
        match props.audio_channels {
            0 | 1 => ChannelLayout::Mono,
            2 => ChannelLayout::Stereo,
            3..=6 => ChannelLayout::Surround5_1,
            _ => ChannelLayout::Surround7_1,
        }
    }

    // ── Sample rate ────────────────────────────────────────────────────────

    fn select_sample_rate(&self, props: &MediaInputProperties, codec: Option<AudioCodec>) -> u32 {
        let supported = codec.unwrap_or(AudioCodec::Opus).supported_sample_rates();

        // Try to match source sample rate, otherwise find closest supported
        let source = props.audio_sample_rate;
        if source == 0 {
            return 48000; // default
        }

        if supported.contains(&source) {
            return source;
        }

        // Find closest supported rate
        supported
            .iter()
            .copied()
            .min_by_key(|&r| (r as i64 - source as i64).unsigned_abs())
            .unwrap_or(48000)
    }

    // ── Confidence scoring ─────────────────────────────────────────────────

    fn calculate_confidence(&self, props: &MediaInputProperties) -> f64 {
        let mut confidence: f64 = 0.5;

        // Known content type boosts confidence
        if props.content_type != ContentType::Unknown {
            confidence += 0.2;
        }

        // Having resolution info boosts confidence
        if props.has_video() && props.width > 0 {
            confidence += 0.1;
        }

        // Having frame rate info boosts confidence
        if props.frame_rate.is_some() {
            confidence += 0.1;
        }

        // Having duration info boosts confidence
        if props.duration_seconds.is_some() {
            confidence += 0.1;
        }

        confidence.min(1.0)
    }

    // ── Rationale generation ───────────────────────────────────────────────

    fn generate_rationale(
        &self,
        props: &MediaInputProperties,
        video_codec: Option<VideoCodec>,
        audio_codec: Option<AudioCodec>,
        container: &ContainerFormat,
    ) -> String {
        let mut parts = Vec::new();

        if let Some(vc) = video_codec {
            let reason = match (vc, props.content_type) {
                (VideoCodec::Av1, ContentType::LiveAction) => {
                    "AV1 selected for superior compression of natural imagery"
                }
                (VideoCodec::Av1, ContentType::Slideshow) => {
                    "AV1 selected for excellent intra-frame coding of static content"
                }
                (VideoCodec::Av1, _) => "AV1 selected for best-in-class compression efficiency",
                (VideoCodec::Vp9, ContentType::ScreenRecording) => {
                    "VP9 selected for sharp text and edge preservation"
                }
                (VideoCodec::Vp9, ContentType::HighMotion) => {
                    "VP9 selected for good speed/quality balance with fast motion"
                }
                (VideoCodec::Vp9, ContentType::Animation) => {
                    "VP9 selected for HD animation with efficient coding"
                }
                (VideoCodec::Vp9, _) => "VP9 selected as a versatile general-purpose codec",
                (VideoCodec::Vp8, ContentType::Animation) => {
                    "VP8 selected for simple animation with alpha support"
                }
                (VideoCodec::Vp8, _) => "VP8 selected for fast encoding of simple content",
                (VideoCodec::Theora, _) => "Theora selected for maximum compatibility",
            };
            parts.push(reason.to_string());
        }

        if let Some(ac) = audio_codec {
            let reason = match ac {
                AudioCodec::Opus => "Opus audio for best lossy quality at any bitrate",
                AudioCodec::Vorbis => "Vorbis audio for broad compatibility",
                AudioCodec::Flac => "FLAC audio for lossless archival quality",
                AudioCodec::Pcm => "PCM audio for uncompressed raw audio",
            };
            parts.push(reason.to_string());
        }

        parts.push(format!("{} container", container.name()));

        if props.is_hdr {
            parts.push("HDR metadata preserved".to_string());
        }

        parts.join(". ") + "."
    }

    fn generate_notes(
        &self,
        props: &MediaInputProperties,
        video_codec: Option<VideoCodec>,
        _audio_codec: Option<AudioCodec>,
    ) -> Vec<String> {
        let mut notes = Vec::new();

        if let Some(vc) = video_codec {
            if vc == VideoCodec::Av1 {
                notes.push(
                    "AV1 encoding is slower than VP9/VP8; consider VP9 for faster turnaround"
                        .to_string(),
                );
            }

            if props.has_alpha && !vc.supports_alpha() {
                notes.push(format!(
                    "{} does not support alpha channel; alpha will be composited over black",
                    vc.name()
                ));
            }
        }

        if props.is_hdr && props.bit_depth < 10 {
            notes
                .push("HDR flagged but bit depth < 10; HDR metadata may be inaccurate".to_string());
        }

        if props.is_interlaced {
            notes.push(
                "Interlaced source detected; deinterlacing is recommended before encoding"
                    .to_string(),
            );
        }

        notes
    }

    fn summarize_input(&self, props: &MediaInputProperties) -> String {
        let mut parts = Vec::new();

        if props.has_video() {
            parts.push(format!(
                "{}x{} {}",
                props.width,
                props.height,
                props.resolution_tier().label()
            ));
            if let Some(fps) = props.frame_rate {
                parts.push(format!("{fps:.1}fps"));
            }
            parts.push(format!("{}bit", props.bit_depth));
            if props.is_hdr {
                parts.push("HDR".to_string());
            }
            if props.has_alpha {
                parts.push("alpha".to_string());
            }
        }

        if props.has_audio() {
            parts.push(format!(
                "{}ch {}Hz {}bit audio",
                props.audio_channels, props.audio_sample_rate, props.audio_bit_depth
            ));
        }

        parts.push(format!("content: {:?}", props.content_type));

        parts.join(", ")
    }
}

impl Default for CodecSelector {
    fn default() -> Self {
        Self::new()
    }
}

// ── Convenience function ───────────────────────────────────────────────────

/// Recommend optimal codecs for the given media properties.
///
/// This is a convenience wrapper around [`CodecSelector::recommend_codec`]
/// using default settings.
pub fn recommend_codec(props: &MediaInputProperties) -> Result<CodecRecommendationSet> {
    CodecSelector::new().recommend_codec(props)
}

/// Recommend codecs from a [`MediaAnalysis`] and classified content type.
pub fn recommend_codec_from_analysis(
    analysis: &MediaAnalysis,
    content_type: ContentType,
) -> Result<CodecRecommendationSet> {
    let props = MediaInputProperties::from_analysis(analysis, content_type);
    recommend_codec(&props)
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_1080p_live_action() -> MediaInputProperties {
        MediaInputProperties {
            width: 1920,
            height: 1080,
            frame_rate: Some(30.0),
            bit_depth: 8,
            is_hdr: false,
            is_interlaced: false,
            has_alpha: false,
            audio_channels: 2,
            audio_sample_rate: 48000,
            audio_bit_depth: 16,
            content_type: ContentType::LiveAction,
            duration_seconds: Some(120.0),
            file_size: 100_000_000,
        }
    }

    fn make_animation_720p() -> MediaInputProperties {
        MediaInputProperties {
            width: 1280,
            height: 720,
            frame_rate: Some(24.0),
            bit_depth: 8,
            is_hdr: false,
            is_interlaced: false,
            has_alpha: true,
            audio_channels: 2,
            audio_sample_rate: 44100,
            audio_bit_depth: 16,
            content_type: ContentType::Animation,
            duration_seconds: Some(600.0),
            file_size: 200_000_000,
        }
    }

    fn make_4k_hdr() -> MediaInputProperties {
        MediaInputProperties {
            width: 3840,
            height: 2160,
            frame_rate: Some(60.0),
            bit_depth: 10,
            is_hdr: true,
            is_interlaced: false,
            has_alpha: false,
            audio_channels: 6,
            audio_sample_rate: 48000,
            audio_bit_depth: 24,
            content_type: ContentType::LiveAction,
            duration_seconds: Some(7200.0),
            file_size: 50_000_000_000,
        }
    }

    fn make_audio_only() -> MediaInputProperties {
        MediaInputProperties {
            width: 0,
            height: 0,
            frame_rate: None,
            bit_depth: 0,
            is_hdr: false,
            is_interlaced: false,
            has_alpha: false,
            audio_channels: 2,
            audio_sample_rate: 44100,
            audio_bit_depth: 16,
            content_type: ContentType::AudioOnly,
            duration_seconds: Some(240.0),
            file_size: 10_000_000,
        }
    }

    #[test]
    fn test_recommend_1080p_live_action() {
        let props = make_1080p_live_action();
        let result = recommend_codec(&props);
        assert!(result.is_ok());
        let set = result.expect("should succeed");

        // Live action at 1080p should recommend AV1 (default prefer_av1=true)
        assert_eq!(set.primary.video_codec, Some(VideoCodec::Av1));
        assert_eq!(set.primary.audio_codec, Some(AudioCodec::Opus));
        assert!(set.primary.confidence > 0.5);
        assert!(!set.primary.rationale.is_empty());
    }

    #[test]
    fn test_recommend_animation() {
        let props = make_animation_720p();
        let result = recommend_codec(&props);
        assert!(result.is_ok());
        let set = result.expect("should succeed");

        // 720p animation should use VP8
        assert_eq!(set.primary.video_codec, Some(VideoCodec::Vp8));
        // Animation should not use two-pass
        assert!(!set.primary.two_pass);
    }

    #[test]
    fn test_recommend_4k_hdr() {
        let props = make_4k_hdr();
        let result = recommend_codec(&props);
        assert!(result.is_ok());
        let set = result.expect("should succeed");

        // 4K HDR should recommend AV1 in Matroska
        assert_eq!(set.primary.video_codec, Some(VideoCodec::Av1));
        assert_eq!(set.primary.container, ContainerFormat::Matroska);
        // HDR + high res + long content => two-pass
        assert!(set.primary.two_pass);
        // FLAC for 24-bit audio
        assert_eq!(set.primary.audio_codec, Some(AudioCodec::Flac));
        // 5.1 surround for 6 channels
        assert_eq!(set.primary.audio_channels, Some(ChannelLayout::Surround5_1));
    }

    #[test]
    fn test_recommend_audio_only() {
        let props = make_audio_only();
        let result = recommend_codec(&props);
        assert!(result.is_ok());
        let set = result.expect("should succeed");

        // Audio-only should have no video codec
        assert!(set.primary.video_codec.is_none());
        assert_eq!(set.primary.audio_codec, Some(AudioCodec::Opus));
        assert!(!set.primary.two_pass);
        // Should have alternatives
        assert!(!set.alternatives.is_empty());
    }

    #[test]
    fn test_empty_input_fails() {
        let props = MediaInputProperties {
            width: 0,
            height: 0,
            frame_rate: None,
            bit_depth: 0,
            is_hdr: false,
            is_interlaced: false,
            has_alpha: false,
            audio_channels: 0,
            audio_sample_rate: 0,
            audio_bit_depth: 0,
            content_type: ContentType::Unknown,
            duration_seconds: None,
            file_size: 0,
        };
        let result = recommend_codec(&props);
        assert!(result.is_err());
    }

    #[test]
    fn test_speed_constrained_avoids_av1() {
        let props = make_1080p_live_action();
        let selector = CodecSelector::new().with_max_encode_time(Some(2.0));
        let result = selector.recommend_codec(&props);
        assert!(result.is_ok());
        let set = result.expect("should succeed");

        // With tight speed constraint, should prefer VP9 over AV1
        assert_eq!(set.primary.video_codec, Some(VideoCodec::Vp9));
    }

    #[test]
    fn test_alternatives_present() {
        let props = make_1080p_live_action();
        let result = recommend_codec(&props);
        assert!(result.is_ok());
        let set = result.expect("should succeed");

        // Should have at least one alternative
        assert!(!set.alternatives.is_empty());
        // Alternatives should have lower confidence
        for alt in &set.alternatives {
            assert!(alt.confidence <= set.primary.confidence);
        }
    }

    #[test]
    fn test_from_analysis() {
        let analysis = MediaAnalysis {
            has_video: true,
            has_audio: true,
            video_codec: Some(VideoCodec::Vp9),
            audio_codec: Some(AudioCodec::Opus),
            resolution: Some((1920, 1080)),
            frame_rate: Some(30.0),
            bitrate: Some(5_000_000),
            duration_seconds: Some(300.0),
            file_size: 625_000_000,
            is_hdr: false,
            is_interlaced: false,
        };
        let result = recommend_codec_from_analysis(&analysis, ContentType::LiveAction);
        assert!(result.is_ok());
        let set = result.expect("should succeed");
        assert!(set.primary.video_codec.is_some());
        assert!(set.primary.audio_codec.is_some());
    }

    #[test]
    fn test_resolution_tier() {
        let mut props = make_1080p_live_action();
        assert_eq!(props.resolution_tier(), ResolutionTier::FullHd);

        props.width = 3840;
        props.height = 2160;
        assert_eq!(props.resolution_tier(), ResolutionTier::Uhd4K);

        props.width = 640;
        props.height = 480;
        assert_eq!(props.resolution_tier(), ResolutionTier::Sd);

        props.width = 0;
        props.height = 0;
        assert_eq!(props.resolution_tier(), ResolutionTier::AudioOnly);
    }

    #[test]
    fn test_screen_recording_uses_vp9() {
        let props = MediaInputProperties {
            width: 2560,
            height: 1440,
            frame_rate: Some(15.0),
            bit_depth: 8,
            is_hdr: false,
            is_interlaced: false,
            has_alpha: false,
            audio_channels: 1,
            audio_sample_rate: 44100,
            audio_bit_depth: 16,
            content_type: ContentType::ScreenRecording,
            duration_seconds: Some(600.0),
            file_size: 500_000_000,
        };
        let result = recommend_codec(&props);
        assert!(result.is_ok());
        let set = result.expect("should succeed");
        assert_eq!(set.primary.video_codec, Some(VideoCodec::Vp9));
    }

    #[test]
    fn test_high_motion_estimation() {
        let props = MediaInputProperties {
            width: 1920,
            height: 1080,
            frame_rate: Some(120.0),
            bit_depth: 8,
            is_hdr: false,
            is_interlaced: false,
            has_alpha: false,
            audio_channels: 2,
            audio_sample_rate: 48000,
            audio_bit_depth: 16,
            content_type: ContentType::HighMotion,
            duration_seconds: Some(60.0),
            file_size: 500_000_000,
        };
        let result = recommend_codec(&props);
        assert!(result.is_ok());
        let set = result.expect("should succeed");
        // 120fps high motion => VP9 (speed/quality balance)
        assert_eq!(set.primary.video_codec, Some(VideoCodec::Vp9));
    }

    #[test]
    fn test_interlaced_note() {
        let mut props = make_1080p_live_action();
        props.is_interlaced = true;
        let result = recommend_codec(&props);
        assert!(result.is_ok());
        let set = result.expect("should succeed");
        assert!(
            set.primary.notes.iter().any(|n| n.contains("interlac")),
            "Should note interlaced source"
        );
    }

    #[test]
    fn test_alpha_note_with_av1() {
        let mut props = make_1080p_live_action();
        props.has_alpha = true;
        let result = recommend_codec(&props);
        assert!(result.is_ok());
        let set = result.expect("should succeed");
        // AV1 does not support alpha; should have a note
        if set.primary.video_codec == Some(VideoCodec::Av1) {
            assert!(
                set.primary.notes.iter().any(|n| n.contains("alpha")),
                "Should note alpha limitation"
            );
        }
    }

    #[test]
    fn test_sample_rate_selection() {
        let selector = CodecSelector::new();

        let props = MediaInputProperties {
            width: 0,
            height: 0,
            frame_rate: None,
            bit_depth: 0,
            is_hdr: false,
            is_interlaced: false,
            has_alpha: false,
            audio_channels: 2,
            audio_sample_rate: 44100,
            audio_bit_depth: 16,
            content_type: ContentType::AudioOnly,
            duration_seconds: Some(60.0),
            file_size: 5_000_000,
        };

        // Opus doesn't support 44100, should pick closest (48000)
        let rate = selector.select_sample_rate(&props, Some(AudioCodec::Opus));
        assert_eq!(rate, 48000);

        // FLAC supports 44100 directly
        let rate_flac = selector.select_sample_rate(&props, Some(AudioCodec::Flac));
        assert_eq!(rate_flac, 44100);
    }

    #[test]
    fn test_no_unwrap_in_selector() {
        // Ensure edge cases don't panic
        let props = MediaInputProperties {
            width: 1,
            height: 1,
            frame_rate: Some(0.001),
            bit_depth: 1,
            is_hdr: false,
            is_interlaced: false,
            has_alpha: false,
            audio_channels: 1,
            audio_sample_rate: 1,
            audio_bit_depth: 1,
            content_type: ContentType::Unknown,
            duration_seconds: Some(0.001),
            file_size: 1,
        };
        let result = recommend_codec(&props);
        assert!(result.is_ok());
    }

    #[test]
    fn test_lossless_disabled() {
        let selector = CodecSelector::new().with_lossless_audio(false);
        let props = make_audio_only();
        let result = selector.recommend_codec(&props);
        assert!(result.is_ok());
        let set = result.expect("should succeed");
        // No FLAC alternative when lossless disabled
        assert!(
            !set.alternatives
                .iter()
                .any(|a| a.audio_codec == Some(AudioCodec::Flac)),
            "Should not include FLAC when lossless disabled"
        );
    }

    #[test]
    fn test_input_summary_format() {
        let props = make_1080p_live_action();
        let selector = CodecSelector::new();
        let result = selector.recommend_codec(&props);
        assert!(result.is_ok());
        let set = result.expect("should succeed");
        assert!(set.input_summary.contains("1920x1080"));
        assert!(set.input_summary.contains("30.0fps"));
        assert!(set.input_summary.contains("LiveAction"));
    }

    #[test]
    fn test_slideshow_low_bitrate() {
        let props = MediaInputProperties {
            width: 1920,
            height: 1080,
            frame_rate: Some(1.0),
            bit_depth: 8,
            is_hdr: false,
            is_interlaced: false,
            has_alpha: false,
            audio_channels: 0,
            audio_sample_rate: 0,
            audio_bit_depth: 0,
            content_type: ContentType::Slideshow,
            duration_seconds: Some(300.0),
            file_size: 50_000_000,
        };
        let result = recommend_codec(&props);
        assert!(result.is_ok());
        let set = result.expect("should succeed");
        // Slideshow: AV1 (good at intra-frame)
        assert_eq!(set.primary.video_codec, Some(VideoCodec::Av1));
        // Bitrate should be relatively low for slideshow
        if let Some(bitrate) = set.primary.video_bitrate_bps {
            // 1fps slideshow should have very low bitrate
            assert!(bitrate < 1_000_000, "Slideshow bitrate should be low");
        }
    }
}
