//! Dolby Atmos and spatial audio passthrough support for the transcode pipeline.
//!
//! This module provides metadata structures, codec descriptors, and a passthrough
//! planner for immersive / object-based audio formats:
//!
//! - **Dolby Atmos** (AC-4, E-AC-3 JOC — Joint Object Coding extension)
//! - **MPEG-H Audio** (ISO 23008-3 — 3D audio)
//! - **Sony 360 Reality Audio** (MPEG-H profile)
//! - **Ambisonics** (First-order through Higher-order, carrying channel bed + objects)
//!
//! # Passthrough vs. re-encode
//!
//! Spatial audio bitstreams carry object metadata and audio objects that can only
//! be faithfully preserved by **passing through** the compressed bitstream without
//! decoding.  This module models the three legal operations:
//!
//! 1. [`PassthroughMode::Copy`] — write the bitstream bytes as-is into the output
//!    container (same codec, same parameters).
//! 2. [`PassthroughMode::Rewrap`] — demux from one container format and remux
//!    into another without touching the elementary stream.
//! 3. [`PassthroughMode::Downmix`] — decode spatial audio and mix down to a
//!    conventional stereo or surround channel bed (lossy, last resort).
//!
//! # Patent-free note
//!
//! This module **describes** the Atmos/spatial codec parameters and passthrough
//! decisions but does not implement any proprietary codec.  Actual bitstream
//! copying/remuxing is delegated to the container layer.

#![allow(clippy::cast_precision_loss)]

use crate::{Result, TranscodeError};

// ─── Spatial audio codec family ───────────────────────────────────────────────

/// Identifies the codec family used for spatial/object-based audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpatialAudioCodec {
    /// Dolby TrueHD with Atmos object layer (lossless, Blu-ray).
    DolbyTrueHdAtmos,
    /// Dolby E-AC-3 (DD+) with Joint Object Coding extension (Atmos for streaming).
    EAc3Joc,
    /// Dolby AC-4 IMS (Immersive Stereo) — low-latency Atmos variant.
    Ac4Ims,
    /// MPEG-H Audio 3D (ISO 23008-3).
    MpegH3dAudio,
    /// Ambisonics B-format (first-order through 7th-order).
    Ambisonics,
    /// Sony 360 Reality Audio (MPEG-H profile).
    Sony360Ra,
    /// DTS:X (object-based extension to DTS-HD MA).
    DtsX,
    /// Conventional PCM channel bed (no object layer — used as passthrough target
    /// for [`PassthroughMode::Downmix`]).
    PcmChannelBed,
}

impl SpatialAudioCodec {
    /// Returns a human-readable name for the codec.
    #[must_use]
    pub fn display_name(self) -> &'static str {
        match self {
            Self::DolbyTrueHdAtmos => "Dolby TrueHD + Atmos",
            Self::EAc3Joc => "E-AC-3 JOC (Atmos streaming)",
            Self::Ac4Ims => "AC-4 IMS (Dolby Atmos)",
            Self::MpegH3dAudio => "MPEG-H 3D Audio",
            Self::Ambisonics => "Ambisonics (B-format)",
            Self::Sony360Ra => "Sony 360 Reality Audio",
            Self::DtsX => "DTS:X",
            Self::PcmChannelBed => "PCM channel bed",
        }
    }

    /// Returns `true` if this codec carries discrete audio objects (not just
    /// a fixed channel bed).
    #[must_use]
    pub fn has_object_layer(self) -> bool {
        matches!(
            self,
            Self::DolbyTrueHdAtmos
                | Self::EAc3Joc
                | Self::Ac4Ims
                | Self::MpegH3dAudio
                | Self::Sony360Ra
                | Self::DtsX
        )
    }

    /// Returns `true` if the codec is lossless (at its core).
    #[must_use]
    pub fn is_lossless(self) -> bool {
        matches!(self, Self::DolbyTrueHdAtmos | Self::PcmChannelBed)
    }

    /// Returns `true` if this codec is patent-free.
    ///
    /// Only Ambisonics PCM B-format is truly patent-free; all others carry
    /// proprietary royalty-bearing extensions.
    #[must_use]
    pub fn is_patent_free(self) -> bool {
        matches!(self, Self::Ambisonics | Self::PcmChannelBed)
    }

    /// Returns the typical maximum number of simultaneous audio objects
    /// supported by this codec variant.  Returns `None` if object-count is
    /// not applicable (PCM) or not publicly specified.
    #[must_use]
    pub fn max_audio_objects(self) -> Option<u32> {
        match self {
            Self::DolbyTrueHdAtmos | Self::EAc3Joc | Self::Ac4Ims => Some(128),
            Self::MpegH3dAudio => Some(24),
            Self::DtsX => Some(32),
            Self::Sony360Ra => Some(22),
            Self::Ambisonics | Self::PcmChannelBed => None,
        }
    }
}

// ─── Ambisonics order ─────────────────────────────────────────────────────────

/// Ambisonics order (FOA = First-Order Ambisonics, HOA = Higher-Order).
///
/// The number of B-format channels = `(order + 1)²`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct AmbisonicsOrder(pub u8);

impl AmbisonicsOrder {
    /// Returns the number of B-format channels required for this order.
    ///
    /// Formula: `(order + 1)²`.
    #[must_use]
    pub fn channel_count(self) -> u32 {
        let n = u32::from(self.0) + 1;
        n * n
    }

    /// Returns `true` if this is First-Order Ambisonics (FOA).
    #[must_use]
    pub fn is_foa(self) -> bool {
        self.0 == 1
    }

    /// Returns `true` if this is Higher-Order Ambisonics (order ≥ 2).
    #[must_use]
    pub fn is_hoa(self) -> bool {
        self.0 >= 2
    }
}

// ─── Spatial audio stream descriptor ─────────────────────────────────────────

/// Describes a spatial audio stream found in (or targeted for) a container.
#[derive(Debug, Clone)]
pub struct SpatialAudioStreamDescriptor {
    /// Stream index within the container (0-based).
    pub stream_index: u32,
    /// Codec family.
    pub codec: SpatialAudioCodec,
    /// Sample rate in Hz.
    pub sample_rate_hz: u32,
    /// Bit depth for PCM representations (0 = N/A for compressed formats).
    pub bit_depth: u8,
    /// Average bitrate in bits per second (0 = lossless / CBR unspecified).
    pub avg_bitrate_bps: u64,
    /// Number of channels in the base bed (e.g. 7.1 = 8).
    pub bed_channels: u8,
    /// Number of rendered audio objects (0 if no object layer).
    pub object_count: u32,
    /// Ambisonics order, if codec is [`SpatialAudioCodec::Ambisonics`].
    pub ambisonics_order: Option<AmbisonicsOrder>,
    /// Language tag (BCP 47), e.g. `"en"`, `"fr"`.
    pub language: Option<String>,
}

impl SpatialAudioStreamDescriptor {
    /// Creates a descriptor for an E-AC-3 JOC (Dolby Atmos streaming) stream.
    #[must_use]
    pub fn eac3_joc(stream_index: u32, avg_bitrate_bps: u64) -> Self {
        Self {
            stream_index,
            codec: SpatialAudioCodec::EAc3Joc,
            sample_rate_hz: 48_000,
            bit_depth: 0,
            avg_bitrate_bps,
            bed_channels: 8, // 7.1 bed
            object_count: 118,
            ambisonics_order: None,
            language: None,
        }
    }

    /// Creates a descriptor for a first-order Ambisonics stream (4 channels PCM).
    #[must_use]
    pub fn foa_ambisonics(stream_index: u32, sample_rate_hz: u32) -> Self {
        let order = AmbisonicsOrder(1);
        Self {
            stream_index,
            codec: SpatialAudioCodec::Ambisonics,
            sample_rate_hz,
            bit_depth: 24,
            avg_bitrate_bps: 0,
            bed_channels: order.channel_count() as u8,
            object_count: 0,
            ambisonics_order: Some(order),
            language: None,
        }
    }

    /// Sets the language tag (builder-style).
    #[must_use]
    pub fn with_language(mut self, lang: impl Into<String>) -> Self {
        self.language = Some(lang.into());
        self
    }

    /// Total channel count: bed channels + rendered objects (for estimation only).
    #[must_use]
    pub fn total_channel_estimate(&self) -> u32 {
        u32::from(self.bed_channels) + self.object_count
    }
}

// ─── Passthrough mode ─────────────────────────────────────────────────────────

/// How to handle a spatial audio stream during transcoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassthroughMode {
    /// Copy the bitstream bytes unchanged into the output container (same codec).
    Copy,
    /// Rewrap the bitstream into a different container without re-encoding.
    Rewrap,
    /// Decode and downmix to a conventional stereo (or 5.1) PCM channel bed.
    Downmix,
    /// Drop the stream entirely from the output.
    Drop,
}

impl PassthroughMode {
    /// Returns `true` if this mode preserves the spatial audio object layer.
    #[must_use]
    pub fn preserves_objects(self) -> bool {
        matches!(self, Self::Copy | Self::Rewrap)
    }

    /// Returns a short human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Copy => "copy",
            Self::Rewrap => "rewrap",
            Self::Downmix => "downmix",
            Self::Drop => "drop",
        }
    }
}

// ─── Container compatibility ──────────────────────────────────────────────────

/// Well-known container formats relevant to spatial audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContainerFormat {
    /// MPEG-4 / MP4 (.mp4, .m4a).
    Mp4,
    /// Matroska (.mkv, .mka).
    Mkv,
    /// Blu-ray BDMV transport stream (.m2ts).
    M2ts,
    /// Dolby Atmos Master format (.atmos).
    AtmosMaster,
    /// Ambisonics B-format WAV.
    BFormatWav,
}

impl ContainerFormat {
    /// Returns `true` if this container format natively supports the given codec.
    #[must_use]
    pub fn supports_codec(self, codec: SpatialAudioCodec) -> bool {
        match self {
            Self::Mp4 => matches!(
                codec,
                SpatialAudioCodec::EAc3Joc
                    | SpatialAudioCodec::Ac4Ims
                    | SpatialAudioCodec::MpegH3dAudio
            ),
            Self::Mkv => matches!(
                codec,
                SpatialAudioCodec::EAc3Joc
                    | SpatialAudioCodec::DolbyTrueHdAtmos
                    | SpatialAudioCodec::MpegH3dAudio
                    | SpatialAudioCodec::Ambisonics
                    | SpatialAudioCodec::DtsX
            ),
            Self::M2ts => matches!(
                codec,
                SpatialAudioCodec::DolbyTrueHdAtmos
                    | SpatialAudioCodec::EAc3Joc
                    | SpatialAudioCodec::DtsX
            ),
            Self::AtmosMaster => matches!(
                codec,
                SpatialAudioCodec::DolbyTrueHdAtmos | SpatialAudioCodec::EAc3Joc
            ),
            Self::BFormatWav => matches!(codec, SpatialAudioCodec::Ambisonics),
        }
    }
}

// ─── Passthrough plan ─────────────────────────────────────────────────────────

/// The decided action for a single spatial audio stream.
#[derive(Debug, Clone)]
pub struct SpatialAudioAction {
    /// The stream this action applies to.
    pub stream: SpatialAudioStreamDescriptor,
    /// Chosen passthrough mode.
    pub mode: PassthroughMode,
    /// Human-readable reason for the chosen mode.
    pub reason: String,
}

/// A complete spatial audio passthrough plan for one transcode job.
#[derive(Debug, Clone)]
pub struct SpatialAudioPlan {
    /// Per-stream actions.
    pub actions: Vec<SpatialAudioAction>,
}

impl SpatialAudioPlan {
    /// Returns the number of streams that will be passed through (copy or rewrap).
    #[must_use]
    pub fn passthrough_count(&self) -> usize {
        self.actions
            .iter()
            .filter(|a| a.mode.preserves_objects())
            .count()
    }

    /// Returns the number of streams that will be downmixed.
    #[must_use]
    pub fn downmix_count(&self) -> usize {
        self.actions
            .iter()
            .filter(|a| a.mode == PassthroughMode::Downmix)
            .count()
    }

    /// Returns the number of streams that will be dropped.
    #[must_use]
    pub fn drop_count(&self) -> usize {
        self.actions
            .iter()
            .filter(|a| a.mode == PassthroughMode::Drop)
            .count()
    }

    /// Returns `true` if the plan preserves at least one spatial audio stream.
    #[must_use]
    pub fn has_spatial_output(&self) -> bool {
        self.passthrough_count() > 0
    }
}

// ─── Planner ─────────────────────────────────────────────────────────────────

/// Configuration for the [`SpatialAudioPlanner`].
#[derive(Debug, Clone)]
pub struct SpatialPlannerConfig {
    /// Target container format.
    pub output_container: ContainerFormat,
    /// When `true`, prefer passthrough even if the container technically supports
    /// downmix as well.
    pub prefer_passthrough: bool,
    /// When `true`, drop streams whose codec is not supported by the output
    /// container rather than downmixing.
    pub drop_unsupported: bool,
    /// When `true`, log a warning (as a `reason` string) for every decision.
    pub verbose_reasons: bool,
}

impl Default for SpatialPlannerConfig {
    fn default() -> Self {
        Self {
            output_container: ContainerFormat::Mp4,
            prefer_passthrough: true,
            drop_unsupported: false,
            verbose_reasons: true,
        }
    }
}

/// Plans the passthrough/downmix/drop action for each spatial audio stream.
///
/// # Example
///
/// ```
/// use oximedia_transcode::spatial_audio_passthrough::{
///     SpatialAudioStreamDescriptor, SpatialAudioPlanner, SpatialPlannerConfig,
///     ContainerFormat, PassthroughMode,
/// };
///
/// let stream = SpatialAudioStreamDescriptor::eac3_joc(0, 768_000);
/// let config = SpatialPlannerConfig {
///     output_container: ContainerFormat::Mp4,
///     prefer_passthrough: true,
///     drop_unsupported: false,
///     verbose_reasons: false,
/// };
/// let plan = SpatialAudioPlanner::new(config)
///     .plan(vec![stream])
///     .expect("plan failed");
/// assert_eq!(plan.actions[0].mode, PassthroughMode::Copy);
/// ```
pub struct SpatialAudioPlanner {
    config: SpatialPlannerConfig,
}

impl SpatialAudioPlanner {
    /// Creates a new planner with the supplied configuration.
    #[must_use]
    pub fn new(config: SpatialPlannerConfig) -> Self {
        Self { config }
    }

    /// Creates a planner with default configuration targeting MP4 output.
    #[must_use]
    pub fn default_mp4() -> Self {
        Self::new(SpatialPlannerConfig::default())
    }

    /// Generates a [`SpatialAudioPlan`] from a list of source streams.
    ///
    /// # Errors
    ///
    /// Returns [`TranscodeError::InvalidInput`] if `streams` is empty.
    pub fn plan(
        &self,
        streams: Vec<SpatialAudioStreamDescriptor>,
    ) -> Result<SpatialAudioPlan> {
        if streams.is_empty() {
            return Err(TranscodeError::InvalidInput(
                "no spatial audio streams provided to planner".to_string(),
            ));
        }

        let actions = streams
            .into_iter()
            .map(|stream| self.plan_stream(stream))
            .collect();

        Ok(SpatialAudioPlan { actions })
    }

    /// Decides the action for a single stream.
    fn plan_stream(&self, stream: SpatialAudioStreamDescriptor) -> SpatialAudioAction {
        let container_ok = self
            .config
            .output_container
            .supports_codec(stream.codec);

        if container_ok && self.config.prefer_passthrough {
            let reason = if self.config.verbose_reasons {
                format!(
                    "Container {:?} supports {}; copying bitstream unchanged.",
                    self.config.output_container,
                    stream.codec.display_name()
                )
            } else {
                String::new()
            };
            return SpatialAudioAction {
                stream,
                mode: PassthroughMode::Copy,
                reason,
            };
        }

        // Container doesn't natively support the codec.
        if self.config.drop_unsupported {
            let reason = if self.config.verbose_reasons {
                format!(
                    "Container {:?} does not support {}; dropping stream.",
                    self.config.output_container,
                    stream.codec.display_name()
                )
            } else {
                String::new()
            };
            return SpatialAudioAction {
                stream,
                mode: PassthroughMode::Drop,
                reason,
            };
        }

        // Fallback: downmix.
        let reason = if self.config.verbose_reasons {
            format!(
                "Container {:?} does not support {}; downmixing to PCM bed.",
                self.config.output_container,
                stream.codec.display_name()
            )
        } else {
            String::new()
        };
        SpatialAudioAction {
            stream,
            mode: PassthroughMode::Downmix,
            reason,
        }
    }
}

// ─── Downmix specification ────────────────────────────────────────────────────

/// Target channel layout for a spatial-audio downmix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DownmixTarget {
    /// Stereo (L + R).
    Stereo,
    /// 5.1 surround (L, C, R, Ls, Rs, LFE).
    FivePointOne,
    /// 7.1 surround (L, C, R, Lss, Rss, Lrs, Rrs, LFE).
    SevenPointOne,
    /// Mono.
    Mono,
}

impl DownmixTarget {
    /// Returns the number of output channels.
    #[must_use]
    pub fn channel_count(self) -> u8 {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
            Self::FivePointOne => 6,
            Self::SevenPointOne => 8,
        }
    }
}

/// Parameters for a spatial audio downmix operation.
#[derive(Debug, Clone)]
pub struct DownmixSpec {
    /// Target channel layout after downmix.
    pub target: DownmixTarget,
    /// Output sample rate (Hz).
    pub output_sample_rate_hz: u32,
    /// Output bit depth (16 or 24).
    pub output_bit_depth: u8,
    /// Apply loudness normalization after downmix (EBU R128).
    pub normalize_loudness: bool,
}

impl DownmixSpec {
    /// Creates a default stereo downmix spec at 48 kHz / 24-bit.
    #[must_use]
    pub fn stereo_48k() -> Self {
        Self {
            target: DownmixTarget::Stereo,
            output_sample_rate_hz: 48_000,
            output_bit_depth: 24,
            normalize_loudness: true,
        }
    }

    /// Creates a 5.1 downmix spec at 48 kHz / 24-bit.
    #[must_use]
    pub fn five_point_one_48k() -> Self {
        Self {
            target: DownmixTarget::FivePointOne,
            output_sample_rate_hz: 48_000,
            output_bit_depth: 24,
            normalize_loudness: false,
        }
    }

    /// Validates the downmix spec.
    ///
    /// # Errors
    ///
    /// Returns [`TranscodeError::InvalidInput`] if bit depth or sample rate are out of range.
    pub fn validate(&self) -> Result<()> {
        if self.output_bit_depth != 16 && self.output_bit_depth != 24 && self.output_bit_depth != 32 {
            return Err(TranscodeError::InvalidInput(format!(
                "invalid bit depth {}; must be 16, 24, or 32",
                self.output_bit_depth
            )));
        }
        if self.output_sample_rate_hz == 0 {
            return Err(TranscodeError::InvalidInput(
                "output sample rate must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── SpatialAudioCodec ────────────────────────────────────────────────────

    #[test]
    fn test_codec_has_object_layer() {
        assert!(SpatialAudioCodec::EAc3Joc.has_object_layer());
        assert!(SpatialAudioCodec::DolbyTrueHdAtmos.has_object_layer());
        assert!(!SpatialAudioCodec::Ambisonics.has_object_layer());
        assert!(!SpatialAudioCodec::PcmChannelBed.has_object_layer());
    }

    #[test]
    fn test_codec_is_lossless() {
        assert!(SpatialAudioCodec::DolbyTrueHdAtmos.is_lossless());
        assert!(SpatialAudioCodec::PcmChannelBed.is_lossless());
        assert!(!SpatialAudioCodec::EAc3Joc.is_lossless());
        assert!(!SpatialAudioCodec::MpegH3dAudio.is_lossless());
    }

    #[test]
    fn test_codec_patent_free() {
        assert!(SpatialAudioCodec::Ambisonics.is_patent_free());
        assert!(SpatialAudioCodec::PcmChannelBed.is_patent_free());
        assert!(!SpatialAudioCodec::EAc3Joc.is_patent_free());
        assert!(!SpatialAudioCodec::DolbyTrueHdAtmos.is_patent_free());
    }

    #[test]
    fn test_codec_max_audio_objects() {
        assert_eq!(SpatialAudioCodec::EAc3Joc.max_audio_objects(), Some(128));
        assert_eq!(SpatialAudioCodec::MpegH3dAudio.max_audio_objects(), Some(24));
        assert_eq!(SpatialAudioCodec::Ambisonics.max_audio_objects(), None);
        assert_eq!(SpatialAudioCodec::PcmChannelBed.max_audio_objects(), None);
    }

    // ── AmbisonicsOrder ──────────────────────────────────────────────────────

    #[test]
    fn test_ambisonics_channel_count() {
        assert_eq!(AmbisonicsOrder(1).channel_count(), 4); // (1+1)² = 4 FOA
        assert_eq!(AmbisonicsOrder(2).channel_count(), 9); // (2+1)² = 9 SOA
        assert_eq!(AmbisonicsOrder(3).channel_count(), 16); // (3+1)² = 16 TOA
        assert_eq!(AmbisonicsOrder(7).channel_count(), 64); // (7+1)² = 64
    }

    #[test]
    fn test_ambisonics_is_foa_hoa() {
        assert!(AmbisonicsOrder(1).is_foa());
        assert!(!AmbisonicsOrder(2).is_foa());
        assert!(AmbisonicsOrder(2).is_hoa());
        assert!(!AmbisonicsOrder(1).is_hoa());
    }

    // ── PassthroughMode ──────────────────────────────────────────────────────

    #[test]
    fn test_passthrough_mode_preserves_objects() {
        assert!(PassthroughMode::Copy.preserves_objects());
        assert!(PassthroughMode::Rewrap.preserves_objects());
        assert!(!PassthroughMode::Downmix.preserves_objects());
        assert!(!PassthroughMode::Drop.preserves_objects());
    }

    #[test]
    fn test_passthrough_mode_labels() {
        assert_eq!(PassthroughMode::Copy.label(), "copy");
        assert_eq!(PassthroughMode::Rewrap.label(), "rewrap");
        assert_eq!(PassthroughMode::Downmix.label(), "downmix");
        assert_eq!(PassthroughMode::Drop.label(), "drop");
    }

    // ── ContainerFormat ──────────────────────────────────────────────────────

    #[test]
    fn test_container_supports_codec_mp4() {
        assert!(ContainerFormat::Mp4.supports_codec(SpatialAudioCodec::EAc3Joc));
        assert!(ContainerFormat::Mp4.supports_codec(SpatialAudioCodec::Ac4Ims));
        assert!(!ContainerFormat::Mp4.supports_codec(SpatialAudioCodec::DolbyTrueHdAtmos));
        assert!(!ContainerFormat::Mp4.supports_codec(SpatialAudioCodec::Ambisonics));
    }

    #[test]
    fn test_container_supports_codec_mkv() {
        assert!(ContainerFormat::Mkv.supports_codec(SpatialAudioCodec::Ambisonics));
        assert!(ContainerFormat::Mkv.supports_codec(SpatialAudioCodec::DolbyTrueHdAtmos));
        assert!(ContainerFormat::Mkv.supports_codec(SpatialAudioCodec::DtsX));
    }

    #[test]
    fn test_container_supports_codec_bformat_wav() {
        assert!(ContainerFormat::BFormatWav.supports_codec(SpatialAudioCodec::Ambisonics));
        assert!(!ContainerFormat::BFormatWav.supports_codec(SpatialAudioCodec::EAc3Joc));
    }

    // ── SpatialAudioStreamDescriptor ─────────────────────────────────────────

    #[test]
    fn test_eac3_joc_descriptor_defaults() {
        let desc = SpatialAudioStreamDescriptor::eac3_joc(0, 768_000);
        assert_eq!(desc.codec, SpatialAudioCodec::EAc3Joc);
        assert_eq!(desc.sample_rate_hz, 48_000);
        assert_eq!(desc.bed_channels, 8);
        assert_eq!(desc.object_count, 118);
    }

    #[test]
    fn test_foa_ambisonics_descriptor() {
        let desc = SpatialAudioStreamDescriptor::foa_ambisonics(1, 48_000);
        assert_eq!(desc.codec, SpatialAudioCodec::Ambisonics);
        assert_eq!(desc.bed_channels, 4); // (1+1)² = 4
        assert_eq!(desc.object_count, 0);
        assert_eq!(desc.ambisonics_order, Some(AmbisonicsOrder(1)));
    }

    #[test]
    fn test_stream_total_channel_estimate() {
        let desc = SpatialAudioStreamDescriptor::eac3_joc(0, 384_000);
        // 8 bed + 118 objects = 126
        assert_eq!(desc.total_channel_estimate(), 126);
    }

    #[test]
    fn test_stream_with_language() {
        let desc = SpatialAudioStreamDescriptor::eac3_joc(0, 384_000).with_language("en");
        assert_eq!(desc.language.as_deref(), Some("en"));
    }

    // ── SpatialAudioPlanner ──────────────────────────────────────────────────

    #[test]
    fn test_planner_copy_for_supported_codec() {
        let stream = SpatialAudioStreamDescriptor::eac3_joc(0, 768_000);
        let config = SpatialPlannerConfig {
            output_container: ContainerFormat::Mp4,
            prefer_passthrough: true,
            drop_unsupported: false,
            verbose_reasons: false,
        };
        let plan = SpatialAudioPlanner::new(config)
            .plan(vec![stream])
            .expect("plan should succeed");
        assert_eq!(plan.actions[0].mode, PassthroughMode::Copy);
        assert!(plan.has_spatial_output());
    }

    #[test]
    fn test_planner_downmix_for_unsupported_codec() {
        // TrueHD Atmos is not supported in MP4.
        let stream = SpatialAudioStreamDescriptor {
            stream_index: 0,
            codec: SpatialAudioCodec::DolbyTrueHdAtmos,
            sample_rate_hz: 48_000,
            bit_depth: 0,
            avg_bitrate_bps: 0,
            bed_channels: 8,
            object_count: 60,
            ambisonics_order: None,
            language: None,
        };
        let config = SpatialPlannerConfig {
            output_container: ContainerFormat::Mp4,
            prefer_passthrough: true,
            drop_unsupported: false,
            verbose_reasons: true,
        };
        let plan = SpatialAudioPlanner::new(config)
            .plan(vec![stream])
            .expect("plan should succeed");
        assert_eq!(plan.actions[0].mode, PassthroughMode::Downmix);
        assert!(!plan.has_spatial_output());
        assert_eq!(plan.downmix_count(), 1);
    }

    #[test]
    fn test_planner_drop_for_unsupported_when_configured() {
        let stream = SpatialAudioStreamDescriptor {
            stream_index: 0,
            codec: SpatialAudioCodec::DolbyTrueHdAtmos,
            sample_rate_hz: 48_000,
            bit_depth: 0,
            avg_bitrate_bps: 0,
            bed_channels: 8,
            object_count: 60,
            ambisonics_order: None,
            language: None,
        };
        let config = SpatialPlannerConfig {
            output_container: ContainerFormat::Mp4,
            prefer_passthrough: true,
            drop_unsupported: true,
            verbose_reasons: false,
        };
        let plan = SpatialAudioPlanner::new(config)
            .plan(vec![stream])
            .expect("plan should succeed");
        assert_eq!(plan.actions[0].mode, PassthroughMode::Drop);
        assert_eq!(plan.drop_count(), 1);
    }

    #[test]
    fn test_planner_empty_streams_error() {
        let plan = SpatialAudioPlanner::default_mp4().plan(vec![]);
        assert!(plan.is_err());
    }

    #[test]
    fn test_planner_mixed_streams() {
        let eac3 = SpatialAudioStreamDescriptor::eac3_joc(0, 768_000);
        let truehd = SpatialAudioStreamDescriptor {
            stream_index: 1,
            codec: SpatialAudioCodec::DolbyTrueHdAtmos,
            sample_rate_hz: 48_000,
            bit_depth: 0,
            avg_bitrate_bps: 0,
            bed_channels: 8,
            object_count: 60,
            ambisonics_order: None,
            language: None,
        };
        let config = SpatialPlannerConfig {
            output_container: ContainerFormat::Mp4,
            prefer_passthrough: true,
            drop_unsupported: false,
            verbose_reasons: false,
        };
        let plan = SpatialAudioPlanner::new(config)
            .plan(vec![eac3, truehd])
            .expect("plan should succeed");
        assert_eq!(plan.actions.len(), 2);
        assert_eq!(plan.passthrough_count(), 1);
        assert_eq!(plan.downmix_count(), 1);
    }

    #[test]
    fn test_planner_mkv_truehd_passthrough() {
        let stream = SpatialAudioStreamDescriptor {
            stream_index: 0,
            codec: SpatialAudioCodec::DolbyTrueHdAtmos,
            sample_rate_hz: 48_000,
            bit_depth: 0,
            avg_bitrate_bps: 0,
            bed_channels: 8,
            object_count: 60,
            ambisonics_order: None,
            language: None,
        };
        let config = SpatialPlannerConfig {
            output_container: ContainerFormat::Mkv,
            prefer_passthrough: true,
            drop_unsupported: false,
            verbose_reasons: false,
        };
        let plan = SpatialAudioPlanner::new(config)
            .plan(vec![stream])
            .expect("plan should succeed");
        assert_eq!(plan.actions[0].mode, PassthroughMode::Copy);
        assert!(plan.has_spatial_output());
    }

    // ── DownmixSpec ──────────────────────────────────────────────────────────

    #[test]
    fn test_downmix_spec_stereo_defaults() {
        let spec = DownmixSpec::stereo_48k();
        assert_eq!(spec.target, DownmixTarget::Stereo);
        assert_eq!(spec.output_sample_rate_hz, 48_000);
        assert_eq!(spec.target.channel_count(), 2);
        assert!(spec.normalize_loudness);
    }

    #[test]
    fn test_downmix_spec_five_one() {
        let spec = DownmixSpec::five_point_one_48k();
        assert_eq!(spec.target.channel_count(), 6);
    }

    #[test]
    fn test_downmix_spec_validate_ok() {
        let spec = DownmixSpec::stereo_48k();
        assert!(spec.validate().is_ok());
    }

    #[test]
    fn test_downmix_spec_validate_bad_bit_depth() {
        let spec = DownmixSpec {
            target: DownmixTarget::Stereo,
            output_sample_rate_hz: 48_000,
            output_bit_depth: 7,
            normalize_loudness: false,
        };
        assert!(spec.validate().is_err());
    }

    #[test]
    fn test_downmix_spec_validate_zero_sample_rate() {
        let spec = DownmixSpec {
            target: DownmixTarget::Stereo,
            output_sample_rate_hz: 0,
            output_bit_depth: 24,
            normalize_loudness: false,
        };
        assert!(spec.validate().is_err());
    }

    #[test]
    fn test_downmix_target_channel_counts() {
        assert_eq!(DownmixTarget::Mono.channel_count(), 1);
        assert_eq!(DownmixTarget::Stereo.channel_count(), 2);
        assert_eq!(DownmixTarget::FivePointOne.channel_count(), 6);
        assert_eq!(DownmixTarget::SevenPointOne.channel_count(), 8);
    }
}
