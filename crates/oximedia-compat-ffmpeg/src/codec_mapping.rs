//! Comprehensive codec and container format name mapping tables.
//!
//! This module supplements the simpler [`crate::codec_map`] with structured
//! records that carry rich metadata: whether a codec is video/audio, what the
//! corresponding OxiMedia name is, and human-readable notes on the mapping
//! rationale.  A parallel set of container/format mappings is provided for
//! `-f` / output-extension resolution.
//!
//! ## Design notes
//!
//! * All lookup is case-insensitive and hyphen/underscore-normalised.
//! * The tables are static slices — no heap allocation at startup.
//! * Reverse lookup (OxiMedia name → FFmpeg name) returns the first match in
//!   table order, so keep canonical entries before aliases.

use std::collections::HashMap;
use std::sync::OnceLock;

// ─────────────────────────────────────────────────────────────────────────────
// Codec mapping types
// ─────────────────────────────────────────────────────────────────────────────

/// Structured metadata for a single FFmpeg → OxiMedia codec mapping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodecMapping {
    /// The FFmpeg codec name (lowercase, hyphens normalised to underscores).
    pub ffmpeg_name: &'static str,
    /// The OxiMedia canonical codec identifier.
    pub oximedia_codec: &'static str,
    /// `true` if the codec encodes/decodes video streams.
    pub is_video: bool,
    /// `true` if the codec encodes/decodes audio streams.
    pub is_audio: bool,
    /// Human-readable mapping rationale or notes.
    pub notes: &'static str,
}

impl CodecMapping {
    const fn video(
        ffmpeg_name: &'static str,
        oximedia_codec: &'static str,
        notes: &'static str,
    ) -> Self {
        Self {
            ffmpeg_name,
            oximedia_codec,
            is_video: true,
            is_audio: false,
            notes,
        }
    }

    const fn audio(
        ffmpeg_name: &'static str,
        oximedia_codec: &'static str,
        notes: &'static str,
    ) -> Self {
        Self {
            ffmpeg_name,
            oximedia_codec,
            is_video: false,
            is_audio: true,
            notes,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Format mapping types
// ─────────────────────────────────────────────────────────────────────────────

/// Structured metadata for a single FFmpeg → OxiMedia container mapping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormatMapping {
    /// The FFmpeg format name (as used with `-f` or inferred from extension).
    pub ffmpeg_name: &'static str,
    /// The OxiMedia container identifier.
    pub oximedia_container: &'static str,
    /// Common file extension (without leading dot).
    pub extension: &'static str,
    /// Human-readable notes.
    pub notes: &'static str,
}

impl FormatMapping {
    const fn new(
        ffmpeg_name: &'static str,
        oximedia_container: &'static str,
        extension: &'static str,
        notes: &'static str,
    ) -> Self {
        Self {
            ffmpeg_name,
            oximedia_container,
            extension,
            notes,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Static codec table  (>= 30 entries)
// ─────────────────────────────────────────────────────────────────────────────

static CODEC_MAPPINGS: &[CodecMapping] = &[
    // ── AV1 — direct ─────────────────────────────────────────────────────────
    CodecMapping::video("libaom_av1", "av1", "libaom encoder — direct match"),
    CodecMapping::video("libsvtav1", "av1", "SVT-AV1 encoder — direct match"),
    CodecMapping::video("librav1e", "av1", "rav1e encoder — direct match"),
    CodecMapping::video("av1", "av1", "native AV1 muxed identifier"),
    CodecMapping::video("av1_vaapi", "av1", "VAAPI AV1 HW encode → software AV1"),
    CodecMapping::video("av1_nvenc", "av1", "NVENC AV1 HW encode → software AV1"),
    CodecMapping::video("av1_amf", "av1", "AMD AMF AV1 encode → software AV1"),
    // ── AV1 — patent substitutions (H.264 / H.265 / proprietary) ─────────────
    CodecMapping::video(
        "libx264",
        "av1",
        "H.264 patent-encumbered → AV1 substitution",
    ),
    CodecMapping::video("h264", "av1", "H.264 native → AV1 substitution"),
    CodecMapping::video(
        "libx265",
        "av1",
        "H.265/HEVC patent-encumbered → AV1 substitution",
    ),
    CodecMapping::video("hevc", "av1", "HEVC native → AV1 substitution"),
    CodecMapping::video("h265", "av1", "H.265 alias → AV1 substitution"),
    CodecMapping::video(
        "prores",
        "av1",
        "Apple ProRes proprietary → AV1 substitution",
    ),
    CodecMapping::video("prores_ks", "av1", "ProRes (KS variant) → AV1 substitution"),
    CodecMapping::video("dnxhd", "av1", "Avid DNxHD proprietary → AV1 substitution"),
    CodecMapping::video("dnxhr", "av1", "Avid DNxHR proprietary → AV1 substitution"),
    CodecMapping::video("mpeg1video", "av1", "MPEG-1 video → AV1 substitution"),
    CodecMapping::video("mpeg2video", "av1", "MPEG-2 video → AV1 substitution"),
    CodecMapping::video("mpeg4", "av1", "MPEG-4 part 2 → AV1 substitution"),
    CodecMapping::video("wmv1", "av1", "WMV1 → AV1 substitution"),
    CodecMapping::video("wmv2", "av1", "WMV2 → AV1 substitution"),
    CodecMapping::video("wmv3", "av1", "WMV3/VC-1 → AV1 substitution"),
    CodecMapping::video("vc1", "av1", "VC-1 → AV1 substitution"),
    CodecMapping::video("h263", "av1", "H.263 → AV1 substitution"),
    CodecMapping::video("h261", "av1", "H.261 → AV1 substitution"),
    CodecMapping::video("mjpeg", "mjpeg", "Motion JPEG — direct match"),
    CodecMapping::video("cfhd", "av1", "CineForm HD → AV1 substitution"),
    // ── APV — direct match (ISO/IEC 23009-13, royalty-free intra-frame) ──────
    CodecMapping::video("apv", "apv", "APV intra-frame codec — direct match"),
    CodecMapping::video("apv1", "apv", "APV alias apv1 — direct match"),
    CodecMapping::video("ffv1", "ffv1", "FFV1 lossless — direct match"),
    // ── VP9 ──────────────────────────────────────────────────────────────────
    CodecMapping::video("libvpx_vp9", "vp9", "libvpx VP9 encoder — direct match"),
    CodecMapping::video("vp9", "vp9", "VP9 native identifier — direct match"),
    CodecMapping::video("vp9_vaapi", "vp9", "VAAPI VP9 HW encode → software VP9"),
    CodecMapping::video("vp9_cuvid", "vp9", "CUVID VP9 HW decode → software VP9"),
    // ── VP8 ──────────────────────────────────────────────────────────────────
    CodecMapping::video("libvpx", "vp8", "libvpx VP8 encoder — direct match"),
    CodecMapping::video("vp8", "vp8", "VP8 native identifier — direct match"),
    CodecMapping::video("libtheora", "vp8", "Theora → VP8 (closest open equivalent)"),
    CodecMapping::video("theora", "vp8", "Theora → VP8 (closest open equivalent)"),
    // ── Opus — direct ────────────────────────────────────────────────────────
    CodecMapping::audio("libopus", "opus", "libopus encoder — direct match"),
    CodecMapping::audio("opus", "opus", "Opus native identifier — direct match"),
    // ── Opus — patent substitutions (AAC, MP3, AMR, WMA) ─────────────────────
    CodecMapping::audio("aac", "opus", "AAC patent-encumbered → Opus substitution"),
    CodecMapping::audio("libfdk_aac", "opus", "FDK AAC → Opus substitution"),
    CodecMapping::audio("aac_at", "opus", "Core Audio AAC → Opus substitution"),
    CodecMapping::audio("mp3", "opus", "MP3 patent-encumbered → Opus substitution"),
    CodecMapping::audio("libmp3lame", "opus", "LAME MP3 → Opus substitution"),
    CodecMapping::audio("mp2", "opus", "MP2 → Opus substitution"),
    CodecMapping::audio("amr_nb", "opus", "AMR-NB → Opus substitution"),
    CodecMapping::audio("amr_wb", "opus", "AMR-WB → Opus substitution"),
    CodecMapping::audio("gsm", "opus", "GSM → Opus substitution"),
    CodecMapping::audio("wma", "opus", "WMA → Opus substitution"),
    CodecMapping::audio("wmav2", "opus", "WMA v2 → Opus substitution"),
    CodecMapping::audio("wmapro", "opus", "WMA Pro → Opus substitution"),
    CodecMapping::audio("atrac3", "opus", "ATRAC3 → Opus substitution"),
    CodecMapping::audio("atrac3p", "opus", "ATRAC3+ → Opus substitution"),
    // ── FLAC — direct and substitutions ──────────────────────────────────────
    CodecMapping::audio("flac", "flac", "FLAC lossless — direct match"),
    CodecMapping::audio("alac", "flac", "Apple Lossless → FLAC substitution"),
    CodecMapping::audio(
        "pcm_s16le",
        "flac",
        "PCM signed 16-bit LE → FLAC lossless path",
    ),
    CodecMapping::audio(
        "pcm_s24le",
        "flac",
        "PCM signed 24-bit LE → FLAC lossless path",
    ),
    CodecMapping::audio(
        "pcm_s32le",
        "flac",
        "PCM signed 32-bit LE → FLAC lossless path",
    ),
    CodecMapping::audio(
        "pcm_f32le",
        "flac",
        "PCM float 32-bit LE → FLAC lossless path",
    ),
    CodecMapping::audio(
        "pcm_f64le",
        "flac",
        "PCM float 64-bit LE → FLAC lossless path",
    ),
    CodecMapping::audio("pcm_u8", "flac", "PCM unsigned 8-bit → FLAC lossless path"),
    CodecMapping::audio("pcm_alaw", "flac", "PCM A-law → FLAC lossless path"),
    CodecMapping::audio("pcm_mulaw", "flac", "PCM mu-law → FLAC lossless path"),
    CodecMapping::audio("ac3", "flac", "Dolby AC-3 → FLAC substitution"),
    CodecMapping::audio("eac3", "flac", "Dolby E-AC-3 → FLAC substitution"),
    CodecMapping::audio("truehd", "flac", "Dolby TrueHD → FLAC substitution"),
    CodecMapping::audio("dts", "flac", "DTS → FLAC substitution"),
    CodecMapping::audio("dca", "flac", "DCA (DTS) → FLAC substitution"),
    CodecMapping::audio("mlp", "flac", "MLP → FLAC substitution"),
    // ── Vorbis ────────────────────────────────────────────────────────────────
    CodecMapping::audio("libvorbis", "vorbis", "libvorbis encoder — direct match"),
    CodecMapping::audio(
        "vorbis",
        "vorbis",
        "Vorbis native identifier — direct match",
    ),
    // ── PCM (additional big-endian variants) ─────────────────────────────────
    CodecMapping::audio(
        "pcm_s16be",
        "flac",
        "PCM signed 16-bit BE → FLAC lossless path",
    ),
    CodecMapping::audio(
        "pcm_s24be",
        "flac",
        "PCM signed 24-bit BE → FLAC lossless path",
    ),
    CodecMapping::audio(
        "pcm_s32be",
        "flac",
        "PCM signed 32-bit BE → FLAC lossless path",
    ),
    CodecMapping::audio(
        "pcm_f32be",
        "flac",
        "PCM float 32-bit BE → FLAC lossless path",
    ),
];

// ─────────────────────────────────────────────────────────────────────────────
// Static format table  (>= 20 entries)
// ─────────────────────────────────────────────────────────────────────────────

static FORMAT_MAPPINGS: &[FormatMapping] = &[
    FormatMapping::new("matroska", "mkv", "mkv", "Matroska muxer — direct match"),
    FormatMapping::new(
        "webm",
        "webm",
        "webm",
        "WebM (Matroska profile) — direct match",
    ),
    FormatMapping::new("ogg", "ogg", "ogg", "Ogg container — direct match"),
    FormatMapping::new("opus", "ogg", "opus", "Opus in Ogg — ogg container"),
    FormatMapping::new(
        "mp4",
        "mp4",
        "mp4",
        "MPEG-4 part 14 container — direct match",
    ),
    FormatMapping::new("mov", "mp4", "mov", "QuickTime MOV → mp4 container"),
    FormatMapping::new("m4v", "mp4", "m4v", "iTunes M4V → mp4 container"),
    FormatMapping::new("avi", "avi", "avi", "AVI container — direct match"),
    FormatMapping::new("flv", "flv", "flv", "Flash Video container — direct match"),
    FormatMapping::new("mpegts", "ts", "ts", "MPEG-TS container — direct match"),
    FormatMapping::new("ts", "ts", "ts", "MPEG-TS short alias"),
    FormatMapping::new("flac", "flac", "flac", "FLAC container — direct match"),
    FormatMapping::new("wav", "wav", "wav", "RIFF/WAV container — direct match"),
    FormatMapping::new("aiff", "aiff", "aiff", "AIFF container — direct match"),
    FormatMapping::new("mp3", "mp3", "mp3", "MP3 elementary stream container"),
    FormatMapping::new("aac", "adts", "aac", "AAC ADTS container"),
    FormatMapping::new("m4a", "mp4", "m4a", "iTunes M4A audio → mp4 container"),
    FormatMapping::new(
        "mxf",
        "mxf",
        "mxf",
        "MXF professional container — direct match",
    ),
    FormatMapping::new("mxf_opatom", "mxf", "mxf", "MXF OP-Atom variant"),
    FormatMapping::new("gxf", "gxf", "gxf", "General eXchange Format"),
    FormatMapping::new("rm", "rm", "rm", "RealMedia container"),
    FormatMapping::new("rmvb", "rm", "rmvb", "RealMedia VBR container"),
    FormatMapping::new("3gp", "mp4", "3gp", "3GPP (mp4 profile)"),
    FormatMapping::new("3g2", "mp4", "3g2", "3GPP2 (mp4 profile)"),
    FormatMapping::new("asf", "asf", "asf", "Advanced Systems Format"),
    FormatMapping::new("wmv", "asf", "wmv", "Windows Media Video (ASF container)"),
    FormatMapping::new("wma", "asf", "wma", "Windows Media Audio (ASF container)"),
    FormatMapping::new("nut", "nut", "nut", "NUT container — direct match"),
    FormatMapping::new("f4v", "mp4", "f4v", "Flash MP4 → mp4 container"),
    FormatMapping::new("hevc", "ts", "hevc", "Raw HEVC bitstream → TS container"),
];

// ─────────────────────────────────────────────────────────────────────────────
// Preset / Tune / Profile translation types
// ─────────────────────────────────────────────────────────────────────────────

/// A translated encoding preset for OxiMedia.
///
/// FFmpeg presets like `ultrafast`, `medium`, `veryslow` control the speed/quality
/// tradeoff. This maps them to OxiMedia's equivalent speed parameter (0-13 for AV1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PresetMapping {
    /// The FFmpeg preset name (e.g. `"ultrafast"`, `"medium"`, `"veryslow"`).
    pub ffmpeg_name: &'static str,
    /// The OxiMedia speed level (0 = slowest/best quality, 13 = fastest).
    pub oxi_speed: u8,
    /// Human-readable description.
    pub notes: &'static str,
}

impl PresetMapping {
    const fn new(ffmpeg_name: &'static str, oxi_speed: u8, notes: &'static str) -> Self {
        Self {
            ffmpeg_name,
            oxi_speed,
            notes,
        }
    }
}

/// A translated encoding tune setting for OxiMedia.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuneMapping {
    /// The FFmpeg tune name (e.g. `"film"`, `"animation"`, `"grain"`).
    pub ffmpeg_name: &'static str,
    /// The OxiMedia tune profile identifier.
    pub oxi_tune: &'static str,
    /// Human-readable description.
    pub notes: &'static str,
}

impl TuneMapping {
    const fn new(ffmpeg_name: &'static str, oxi_tune: &'static str, notes: &'static str) -> Self {
        Self {
            ffmpeg_name,
            oxi_tune,
            notes,
        }
    }
}

/// A translated encoding profile for OxiMedia.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileMapping {
    /// The FFmpeg profile name (e.g. `"baseline"`, `"main"`, `"high"`).
    pub ffmpeg_name: &'static str,
    /// The OxiMedia profile identifier.
    pub oxi_profile: &'static str,
    /// Human-readable description.
    pub notes: &'static str,
}

impl ProfileMapping {
    const fn new(
        ffmpeg_name: &'static str,
        oxi_profile: &'static str,
        notes: &'static str,
    ) -> Self {
        Self {
            ffmpeg_name,
            oxi_profile,
            notes,
        }
    }
}

// ── Static preset table ──────────────────────────────────────────────────────

static PRESET_MAPPINGS: &[PresetMapping] = &[
    PresetMapping::new("ultrafast", 13, "x264/x265 ultrafast -> AV1 speed 13"),
    PresetMapping::new("superfast", 12, "x264/x265 superfast -> AV1 speed 12"),
    PresetMapping::new("veryfast", 10, "x264/x265 veryfast -> AV1 speed 10"),
    PresetMapping::new("faster", 9, "x264/x265 faster -> AV1 speed 9"),
    PresetMapping::new("fast", 8, "x264/x265 fast -> AV1 speed 8"),
    PresetMapping::new("medium", 6, "x264/x265 medium -> AV1 speed 6"),
    PresetMapping::new("slow", 4, "x264/x265 slow -> AV1 speed 4"),
    PresetMapping::new("slower", 2, "x264/x265 slower -> AV1 speed 2"),
    PresetMapping::new("veryslow", 1, "x264/x265 veryslow -> AV1 speed 1"),
    PresetMapping::new("placebo", 0, "x264/x265 placebo -> AV1 speed 0"),
    // AV1-specific presets (SVT-AV1 numeric)
    PresetMapping::new("0", 0, "SVT-AV1 preset 0 -> AV1 speed 0"),
    PresetMapping::new("1", 1, "SVT-AV1 preset 1 -> AV1 speed 1"),
    PresetMapping::new("2", 2, "SVT-AV1 preset 2 -> AV1 speed 2"),
    PresetMapping::new("3", 3, "SVT-AV1 preset 3 -> AV1 speed 3"),
    PresetMapping::new("4", 4, "SVT-AV1 preset 4 -> AV1 speed 4"),
    PresetMapping::new("5", 5, "SVT-AV1 preset 5 -> AV1 speed 5"),
    PresetMapping::new("6", 6, "SVT-AV1 preset 6 -> AV1 speed 6"),
    PresetMapping::new("7", 7, "SVT-AV1 preset 7 -> AV1 speed 7"),
    PresetMapping::new("8", 8, "SVT-AV1 preset 8 -> AV1 speed 8"),
    PresetMapping::new("9", 9, "SVT-AV1 preset 9 -> AV1 speed 9"),
    PresetMapping::new("10", 10, "SVT-AV1 preset 10 -> AV1 speed 10"),
    PresetMapping::new("11", 11, "SVT-AV1 preset 11 -> AV1 speed 11"),
    PresetMapping::new("12", 12, "SVT-AV1 preset 12 -> AV1 speed 12"),
    PresetMapping::new("13", 13, "SVT-AV1 preset 13 -> AV1 speed 13"),
];

// ── Static tune table ────────────────────────────────────────────────────────

static TUNE_MAPPINGS: &[TuneMapping] = &[
    TuneMapping::new(
        "film",
        "film",
        "x264 film -> OxiMedia film (low noise, detail preservation)",
    ),
    TuneMapping::new(
        "animation",
        "animation",
        "x264 animation -> OxiMedia animation (flat areas, strong edges)",
    ),
    TuneMapping::new(
        "grain",
        "grain",
        "x264 grain -> OxiMedia grain (preserve film grain)",
    ),
    TuneMapping::new(
        "stillimage",
        "stillimage",
        "x264 stillimage -> OxiMedia stillimage (I-frame heavy)",
    ),
    TuneMapping::new(
        "fastdecode",
        "fastdecode",
        "x264 fastdecode -> OxiMedia fastdecode (no CABAC/deblock)",
    ),
    TuneMapping::new(
        "zerolatency",
        "zerolatency",
        "x264 zerolatency -> OxiMedia zerolatency (no B-frames, no lookahead)",
    ),
    TuneMapping::new(
        "psnr",
        "psnr",
        "x264 psnr -> OxiMedia psnr (PSNR optimization)",
    ),
    TuneMapping::new(
        "ssim",
        "ssim",
        "x264 ssim -> OxiMedia ssim (SSIM optimization)",
    ),
    TuneMapping::new(
        "screen",
        "screen",
        "HEVC screen -> OxiMedia screen (screen content)",
    ),
];

// ── Static profile table ─────────────────────────────────────────────────────

static PROFILE_MAPPINGS: &[ProfileMapping] = &[
    // H.264 profiles -> AV1 equivalents
    ProfileMapping::new(
        "baseline",
        "main",
        "H.264 baseline -> AV1 main (8-bit 4:2:0)",
    ),
    ProfileMapping::new("main", "main", "H.264/HEVC main -> AV1 main (8-bit 4:2:0)"),
    ProfileMapping::new(
        "high",
        "high",
        "H.264 high -> AV1 high (8-bit 4:2:0 + tools)",
    ),
    ProfileMapping::new(
        "high10",
        "professional",
        "H.264 high10 -> AV1 professional (10-bit)",
    ),
    ProfileMapping::new(
        "high422",
        "professional",
        "H.264 high422 -> AV1 professional (4:2:2)",
    ),
    ProfileMapping::new(
        "high444",
        "professional",
        "H.264 high444 -> AV1 professional (4:4:4)",
    ),
    // HEVC profiles
    ProfileMapping::new(
        "main10",
        "professional",
        "HEVC main10 -> AV1 professional (10-bit)",
    ),
    ProfileMapping::new(
        "main12",
        "professional",
        "HEVC main12 -> AV1 professional (12-bit)",
    ),
    ProfileMapping::new(
        "main422-10",
        "professional",
        "HEVC main422-10 -> AV1 professional",
    ),
    ProfileMapping::new(
        "main444-10",
        "professional",
        "HEVC main444-10 -> AV1 professional",
    ),
    // VP9 profiles
    ProfileMapping::new("profile0", "main", "VP9 profile0 -> main (8-bit 4:2:0)"),
    ProfileMapping::new(
        "profile1",
        "main",
        "VP9 profile1 -> main (8-bit 4:2:2/4:4:4)",
    ),
    ProfileMapping::new(
        "profile2",
        "professional",
        "VP9 profile2 -> professional (10/12-bit 4:2:0)",
    ),
    ProfileMapping::new(
        "profile3",
        "professional",
        "VP9 profile3 -> professional (10/12-bit 4:2:2/4:4:4)",
    ),
];

static CODEC_LOOKUP: OnceLock<HashMap<String, &'static CodecMapping>> = OnceLock::new();
static FORMAT_LOOKUP: OnceLock<HashMap<String, &'static FormatMapping>> = OnceLock::new();
static PRESET_LOOKUP: OnceLock<HashMap<String, &'static PresetMapping>> = OnceLock::new();
static TUNE_LOOKUP: OnceLock<HashMap<String, &'static TuneMapping>> = OnceLock::new();
static PROFILE_LOOKUP: OnceLock<HashMap<String, &'static ProfileMapping>> = OnceLock::new();

// ─────────────────────────────────────────────────────────────────────────────
// CodecMapper
// ─────────────────────────────────────────────────────────────────────────────

/// Zero-cost static lookup service for codec and format name mappings.
///
/// All methods perform a linear scan of the static tables, which is efficient
/// in practice because the tables are small and CPU cache-resident.  For
/// hot-path repeated lookups, callers should cache the returned references.
pub struct CodecMapper;

impl CodecMapper {
    /// Look up the [`CodecMapping`] for an FFmpeg codec name.
    ///
    /// The lookup is case-insensitive and normalises `-` to `_`.
    pub fn codec(ffmpeg_name: &str) -> Option<&'static CodecMapping> {
        let key = normalise(ffmpeg_name);
        codec_lookup().get(key.as_str()).copied()
    }

    /// Look up the [`FormatMapping`] for an FFmpeg format/container name.
    ///
    /// The lookup is case-insensitive and normalises `-` to `_`.
    pub fn format(ffmpeg_name: &str) -> Option<&'static FormatMapping> {
        let key = normalise(ffmpeg_name);
        format_lookup().get(key.as_str()).copied()
    }

    /// Reverse lookup: given an OxiMedia codec name, return the first
    /// [`CodecMapping`] whose `oximedia_codec` matches.
    ///
    /// Returns the *canonical* (first table entry) for that OxiMedia name.
    pub fn reverse_codec(oximedia_name: &str) -> Option<&'static CodecMapping> {
        let key = oximedia_name.to_lowercase();
        CODEC_MAPPINGS
            .iter()
            .find(|m| m.oximedia_codec.eq_ignore_ascii_case(&key))
    }

    /// Return all video [`CodecMapping`] entries.
    pub fn all_video_codecs() -> Vec<&'static CodecMapping> {
        CODEC_MAPPINGS.iter().filter(|m| m.is_video).collect()
    }

    /// Return all audio [`CodecMapping`] entries.
    pub fn all_audio_codecs() -> Vec<&'static CodecMapping> {
        CODEC_MAPPINGS.iter().filter(|m| m.is_audio).collect()
    }

    /// Return all format [`FormatMapping`] entries.
    pub fn all_formats() -> Vec<&'static FormatMapping> {
        FORMAT_MAPPINGS.iter().collect()
    }

    /// Look up a [`PresetMapping`] for an FFmpeg preset name.
    ///
    /// Case-insensitive lookup.
    pub fn preset(ffmpeg_preset: &str) -> Option<&'static PresetMapping> {
        let key = ffmpeg_preset.to_lowercase();
        preset_lookup().get(key.as_str()).copied()
    }

    /// Look up a [`TuneMapping`] for an FFmpeg tune name.
    ///
    /// Case-insensitive lookup.
    pub fn tune(ffmpeg_tune: &str) -> Option<&'static TuneMapping> {
        let key = ffmpeg_tune.to_lowercase();
        tune_lookup().get(key.as_str()).copied()
    }

    /// Look up a [`ProfileMapping`] for an FFmpeg profile name.
    ///
    /// Case-insensitive, hyphen/underscore-normalised lookup.
    pub fn profile(ffmpeg_profile: &str) -> Option<&'static ProfileMapping> {
        let key = normalise(ffmpeg_profile);
        profile_lookup().get(key.as_str()).copied()
    }

    /// Return all preset mappings.
    pub fn all_presets() -> &'static [PresetMapping] {
        PRESET_MAPPINGS
    }

    /// Return all tune mappings.
    pub fn all_tunes() -> &'static [TuneMapping] {
        TUNE_MAPPINGS
    }

    /// Return all profile mappings.
    pub fn all_profiles() -> &'static [ProfileMapping] {
        PROFILE_MAPPINGS
    }
}

/// Normalise a codec/format name for table lookup.
fn normalise(s: &str) -> String {
    s.to_lowercase().replace('-', "_")
}

fn codec_lookup() -> &'static HashMap<String, &'static CodecMapping> {
    CODEC_LOOKUP.get_or_init(|| {
        CODEC_MAPPINGS
            .iter()
            .map(|mapping| (mapping.ffmpeg_name.to_string(), mapping))
            .collect()
    })
}

fn format_lookup() -> &'static HashMap<String, &'static FormatMapping> {
    FORMAT_LOOKUP.get_or_init(|| {
        FORMAT_MAPPINGS
            .iter()
            .map(|mapping| (mapping.ffmpeg_name.to_string(), mapping))
            .collect()
    })
}

fn preset_lookup() -> &'static HashMap<String, &'static PresetMapping> {
    PRESET_LOOKUP.get_or_init(|| {
        PRESET_MAPPINGS
            .iter()
            .map(|mapping| (mapping.ffmpeg_name.to_string(), mapping))
            .collect()
    })
}

fn tune_lookup() -> &'static HashMap<String, &'static TuneMapping> {
    TUNE_LOOKUP.get_or_init(|| {
        TUNE_MAPPINGS
            .iter()
            .map(|mapping| (mapping.ffmpeg_name.to_string(), mapping))
            .collect()
    })
}

fn profile_lookup() -> &'static HashMap<String, &'static ProfileMapping> {
    PROFILE_LOOKUP.get_or_init(|| {
        PROFILE_MAPPINGS
            .iter()
            .map(|mapping| (normalise(mapping.ffmpeg_name), mapping))
            .collect()
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── codec lookups ─────────────────────────────────────────────────────────

    #[test]
    fn test_codec_av1_direct() {
        let m = CodecMapper::codec("libaom-av1").expect("should be found");
        assert_eq!(m.oximedia_codec, "av1");
        assert!(m.is_video);
        assert!(!m.is_audio);
    }

    #[test]
    fn test_codec_libx264_substitution() {
        let m = CodecMapper::codec("libx264").expect("should be found");
        assert_eq!(m.oximedia_codec, "av1");
        assert!(m.is_video);
        assert!(m.notes.contains("substitution") || m.notes.contains("patent"));
    }

    #[test]
    fn test_codec_opus_direct() {
        let m = CodecMapper::codec("libopus").expect("should be found");
        assert_eq!(m.oximedia_codec, "opus");
        assert!(m.is_audio);
        assert!(!m.is_video);
    }

    #[test]
    fn test_codec_aac_substitution() {
        let m = CodecMapper::codec("aac").expect("should be found");
        assert_eq!(m.oximedia_codec, "opus");
        assert!(m.is_audio);
    }

    #[test]
    fn test_codec_flac_direct() {
        let m = CodecMapper::codec("flac").expect("should be found");
        assert_eq!(m.oximedia_codec, "flac");
        assert!(m.is_audio);
    }

    #[test]
    fn test_codec_vp9() {
        let m = CodecMapper::codec("vp9").expect("should be found");
        assert_eq!(m.oximedia_codec, "vp9");
        assert!(m.is_video);
    }

    #[test]
    fn test_codec_vorbis() {
        let m = CodecMapper::codec("libvorbis").expect("should be found");
        assert_eq!(m.oximedia_codec, "vorbis");
        assert!(m.is_audio);
    }

    #[test]
    fn test_codec_case_insensitive() {
        assert!(CodecMapper::codec("LIBX264").is_some());
        assert!(CodecMapper::codec("AV1").is_some());
        assert!(CodecMapper::codec("OPUS").is_some());
    }

    #[test]
    fn test_codec_hyphen_normalised() {
        // libaom-av1 should resolve despite the hyphen.
        let m = CodecMapper::codec("libaom-av1").expect("hyphen should normalise");
        assert_eq!(m.oximedia_codec, "av1");
    }

    #[test]
    fn test_codec_unknown_returns_none() {
        assert!(CodecMapper::codec("totally_made_up_codec_xyz").is_none());
    }

    // ── format lookups ────────────────────────────────────────────────────────

    #[test]
    fn test_format_matroska() {
        let f = CodecMapper::format("matroska").expect("matroska should be found");
        assert_eq!(f.oximedia_container, "mkv");
        assert_eq!(f.extension, "mkv");
    }

    #[test]
    fn test_format_webm() {
        let f = CodecMapper::format("webm").expect("webm should be found");
        assert_eq!(f.oximedia_container, "webm");
    }

    #[test]
    fn test_format_mp4() {
        let f = CodecMapper::format("mp4").expect("mp4 should be found");
        assert_eq!(f.oximedia_container, "mp4");
    }

    #[test]
    fn test_format_mpegts() {
        let f = CodecMapper::format("mpegts").expect("mpegts should be found");
        assert_eq!(f.oximedia_container, "ts");
    }

    #[test]
    fn test_format_unknown_returns_none() {
        assert!(CodecMapper::format("totally_made_up_format_xyz").is_none());
    }

    // ── reverse lookup ────────────────────────────────────────────────────────

    #[test]
    fn test_reverse_codec_av1() {
        let m = CodecMapper::reverse_codec("av1").expect("reverse av1 should exist");
        assert_eq!(m.oximedia_codec, "av1");
        // Should be the first (canonical) entry.
        assert!(m.ffmpeg_name.contains("libaom") || m.ffmpeg_name == "av1");
    }

    #[test]
    fn test_reverse_codec_opus() {
        let m = CodecMapper::reverse_codec("opus").expect("reverse opus should exist");
        assert_eq!(m.oximedia_codec, "opus");
    }

    // ── category helpers ──────────────────────────────────────────────────────

    #[test]
    fn test_all_video_codecs_non_empty() {
        let vids = CodecMapper::all_video_codecs();
        assert!(!vids.is_empty());
        // Every entry should have is_video == true.
        for m in &vids {
            assert!(m.is_video, "{} should be video", m.ffmpeg_name);
        }
    }

    #[test]
    fn test_all_audio_codecs_non_empty() {
        let auds = CodecMapper::all_audio_codecs();
        assert!(!auds.is_empty());
        for m in &auds {
            assert!(m.is_audio, "{} should be audio", m.ffmpeg_name);
        }
    }

    // ── preset lookup tests ────────────────────────────────────────────────

    #[test]
    fn test_preset_medium() {
        let p = CodecMapper::preset("medium").expect("medium should exist");
        assert_eq!(p.oxi_speed, 6);
    }

    #[test]
    fn test_preset_ultrafast() {
        let p = CodecMapper::preset("ultrafast").expect("ultrafast should exist");
        assert_eq!(p.oxi_speed, 13);
    }

    #[test]
    fn test_preset_veryslow() {
        let p = CodecMapper::preset("veryslow").expect("veryslow should exist");
        assert_eq!(p.oxi_speed, 1);
    }

    #[test]
    fn test_preset_placebo() {
        let p = CodecMapper::preset("placebo").expect("placebo should exist");
        assert_eq!(p.oxi_speed, 0);
    }

    #[test]
    fn test_preset_svtav1_numeric() {
        let p = CodecMapper::preset("6").expect("6 should exist");
        assert_eq!(p.oxi_speed, 6);
    }

    #[test]
    fn test_preset_unknown() {
        assert!(CodecMapper::preset("nonexistent_preset").is_none());
    }

    #[test]
    fn test_preset_case_insensitive() {
        assert!(CodecMapper::preset("Medium").is_some());
        assert!(CodecMapper::preset("FAST").is_some());
    }

    // ── tune lookup tests ────────────────────────────────────────────────────

    #[test]
    fn test_tune_film() {
        let t = CodecMapper::tune("film").expect("film should exist");
        assert_eq!(t.oxi_tune, "film");
    }

    #[test]
    fn test_tune_grain() {
        let t = CodecMapper::tune("grain").expect("grain should exist");
        assert_eq!(t.oxi_tune, "grain");
    }

    #[test]
    fn test_tune_animation() {
        let t = CodecMapper::tune("animation").expect("animation should exist");
        assert_eq!(t.oxi_tune, "animation");
    }

    #[test]
    fn test_tune_zerolatency() {
        let t = CodecMapper::tune("zerolatency").expect("zerolatency should exist");
        assert_eq!(t.oxi_tune, "zerolatency");
    }

    #[test]
    fn test_tune_unknown() {
        assert!(CodecMapper::tune("nonexistent_tune_xyz").is_none());
    }

    // ── profile lookup tests ─────────────────────────────────────────────────

    #[test]
    fn test_profile_main() {
        let p = CodecMapper::profile("main").expect("main should exist");
        assert_eq!(p.oxi_profile, "main");
    }

    #[test]
    fn test_profile_high() {
        let p = CodecMapper::profile("high").expect("high should exist");
        assert_eq!(p.oxi_profile, "high");
    }

    #[test]
    fn test_profile_baseline_maps_to_main() {
        let p = CodecMapper::profile("baseline").expect("baseline should exist");
        assert_eq!(p.oxi_profile, "main");
    }

    #[test]
    fn test_profile_high10_maps_to_professional() {
        let p = CodecMapper::profile("high10").expect("high10 should exist");
        assert_eq!(p.oxi_profile, "professional");
    }

    #[test]
    fn test_profile_main10_maps_to_professional() {
        let p = CodecMapper::profile("main10").expect("main10 should exist");
        assert_eq!(p.oxi_profile, "professional");
    }

    #[test]
    fn test_profile_unknown() {
        assert!(CodecMapper::profile("nonexistent_profile").is_none());
    }

    #[test]
    fn test_all_presets_non_empty() {
        assert!(!CodecMapper::all_presets().is_empty());
    }

    #[test]
    fn test_all_tunes_non_empty() {
        assert!(!CodecMapper::all_tunes().is_empty());
    }

    #[test]
    fn test_all_profiles_non_empty() {
        assert!(!CodecMapper::all_profiles().is_empty());
    }

    #[test]
    fn test_table_sizes() {
        // Verify the tables meet the documented minimums.
        assert!(
            CODEC_MAPPINGS.len() >= 30,
            "need >= 30 codec mappings, got {}",
            CODEC_MAPPINGS.len()
        );
        assert!(
            FORMAT_MAPPINGS.len() >= 20,
            "need >= 20 format mappings, got {}",
            FORMAT_MAPPINGS.len()
        );
    }
}
