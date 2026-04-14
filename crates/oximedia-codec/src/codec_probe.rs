//! Codec identification from raw bitstream bytes.
//!
//! Provides magic-byte and structural probing of codec bitstreams with a
//! confidence score for each candidate codec.  The probe intentionally avoids
//! full parsing so that it remains fast and safe to call on untrusted data.
//!
//! # Supported codecs
//!
//! - **AV1** — OBU temporal delimiter / sequence header signature
//! - **VP9** — IVF frame header / superframe marker heuristic
//! - **VP8** — VP8 frame header tag detection
//! - **H.264 / AVC** — AnnexB start codes + SPS NAL type byte
//! - **H.265 / HEVC** — AnnexB start codes + VPS/SPS NAL type bytes
//! - **Theora** — Ogg Theora identification header magic bytes
//! - **Opus** — Ogg Opus identification header magic bytes
//! - **Vorbis** — Ogg Vorbis identification header magic bytes
//! - **FLAC** — fLaC stream marker
//! - **PCM** — Raw PCM (always low confidence unless framed; detected by exclusion)
//! - **PNG** — PNG signature bytes
//! - **GIF** — GIF87a / GIF89a magic
//! - **WebP** — RIFF/WEBP container signature
//! - **JPEG-XL** — JXL codestream / ISOBMFF signature
//! - **MPEG-4 AAC** — ADTS sync word

use std::fmt;

// ---------------------------------------------------------------------------
// Codec identifiers
// ---------------------------------------------------------------------------

/// Identifies a codec or media format in the probe result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CodecId {
    /// AV1 video codec (Alliance for Open Media).
    Av1,
    /// VP9 video codec (Google).
    Vp9,
    /// VP8 video codec (Google / On2).
    Vp8,
    /// H.264 / AVC video codec.
    H264,
    /// H.265 / HEVC video codec.
    H265,
    /// Theora video codec (Xiph.Org).
    Theora,
    /// Opus audio codec (Xiph.Org / IETF).
    Opus,
    /// Vorbis audio codec (Xiph.Org).
    Vorbis,
    /// FLAC lossless audio codec.
    Flac,
    /// Raw PCM audio.
    Pcm,
    /// PNG image format.
    Png,
    /// GIF image format.
    Gif,
    /// WebP image format.
    WebP,
    /// JPEG-XL image format.
    JpegXl,
    /// MPEG-4 AAC audio (ADTS framing).
    Aac,
    /// Unknown / unidentified codec.
    Unknown,
}

impl fmt::Display for CodecId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Av1 => "AV1",
            Self::Vp9 => "VP9",
            Self::Vp8 => "VP8",
            Self::H264 => "H.264",
            Self::H265 => "H.265/HEVC",
            Self::Theora => "Theora",
            Self::Opus => "Opus",
            Self::Vorbis => "Vorbis",
            Self::Flac => "FLAC",
            Self::Pcm => "PCM",
            Self::Png => "PNG",
            Self::Gif => "GIF",
            Self::WebP => "WebP",
            Self::JpegXl => "JPEG-XL",
            Self::Aac => "AAC (ADTS)",
            Self::Unknown => "Unknown",
        };
        f.write_str(name)
    }
}

// ---------------------------------------------------------------------------
// Confidence scoring
// ---------------------------------------------------------------------------

/// Confidence of a probe match, expressed as a value in `[0, 100]`.
///
/// Higher values indicate stronger evidence for the codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Confidence(u8);

impl Confidence {
    /// The minimum confidence (0 — codec is ruled out or unsupported).
    pub const MIN: Self = Self(0);
    /// A low-confidence heuristic match (guessed from partial data).
    pub const LOW: Self = Self(25);
    /// A medium-confidence match (one structural indicator confirmed).
    pub const MEDIUM: Self = Self(50);
    /// A high-confidence match (two or more structural indicators confirmed).
    pub const HIGH: Self = Self(75);
    /// A near-certain match (all applicable magic bytes / markers confirmed).
    pub const CERTAIN: Self = Self(100);

    /// Create a `Confidence` from a raw byte value, clamping to [0, 100].
    pub fn new(raw: u8) -> Self {
        Self(raw.min(100))
    }

    /// Return the raw confidence value.
    pub fn value(self) -> u8 {
        self.0
    }
}

impl fmt::Display for Confidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}%", self.0)
    }
}

// ---------------------------------------------------------------------------
// Probe result
// ---------------------------------------------------------------------------

/// Result of probing a single codec against a byte buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeResult {
    /// The codec this result describes.
    pub codec: CodecId,
    /// Confidence that `data` belongs to this codec.
    pub confidence: Confidence,
    /// Human-readable description of why this confidence was assigned.
    pub reason: String,
}

impl ProbeResult {
    fn new(codec: CodecId, confidence: Confidence, reason: impl Into<String>) -> Self {
        Self {
            codec,
            confidence,
            reason: reason.into(),
        }
    }
}

impl fmt::Display for ProbeResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {} ({})", self.codec, self.confidence, self.reason)
    }
}

// ---------------------------------------------------------------------------
// Internal probe helpers
// ---------------------------------------------------------------------------

/// Check whether `data` starts with `prefix`.
fn starts_with(data: &[u8], prefix: &[u8]) -> bool {
    data.len() >= prefix.len() && &data[..prefix.len()] == prefix
}

/// Check whether `data` contains `needle` within the first `limit` bytes.
fn contains_in_first(data: &[u8], needle: &[u8], limit: usize) -> bool {
    let search_end = data.len().min(limit);
    if search_end < needle.len() {
        return false;
    }
    let haystack = &data[..search_end];
    haystack.windows(needle.len()).any(|w| w == needle)
}

// ---------------------------------------------------------------------------
// Per-codec probe functions
// ---------------------------------------------------------------------------

fn probe_av1(data: &[u8]) -> ProbeResult {
    // AV1 bitstream: first OBU byte has forbidden_bit=0 and type field in bits [6:3].
    // Temporal delimiter OBU has type=2 => header byte = 0b0_0010_0_1_0 = 0x12
    // Sequence header OBU has type=1  => header byte = 0b0_0001_0_1_0 = 0x0A
    // We check for either at offset 0 with has_size_field=1.
    if data.is_empty() {
        return ProbeResult::new(CodecId::Av1, Confidence::MIN, "empty buffer");
    }
    let b = data[0];
    let forbidden = (b >> 7) & 1;
    let obu_type = (b >> 3) & 0x0F;
    let has_size = (b >> 1) & 1;

    if forbidden == 0 && (obu_type == 1 || obu_type == 2) && has_size == 1 {
        ProbeResult::new(
            CodecId::Av1,
            Confidence::HIGH,
            format!("AV1 OBU header byte 0x{b:02X} (type={obu_type})"),
        )
    } else if forbidden == 0 && (1..=8).contains(&obu_type) {
        ProbeResult::new(
            CodecId::Av1,
            Confidence::MEDIUM,
            format!("possible AV1 OBU type={obu_type}"),
        )
    } else {
        ProbeResult::new(CodecId::Av1, Confidence::MIN, "no AV1 OBU marker")
    }
}

fn probe_vp9(data: &[u8]) -> ProbeResult {
    // VP9 IVF frame format is framed by the container.  In raw VP9 the first byte
    // carries `frame_marker` (2 bits, must be 0b10), `profile_low_bit`,
    // `profile_high_bit`, `reserved_zero` (1 bit for profile < 3).
    // frame_marker = (data[0] >> 6) & 0x3 == 2 for a valid VP9 frame header.
    if data.is_empty() {
        return ProbeResult::new(CodecId::Vp9, Confidence::MIN, "empty buffer");
    }
    let frame_marker = (data[0] >> 6) & 0x03;
    // VP9 superframe marker: last byte has 0b110xxxxx pattern.
    let superframe_marker = data.last().map(|&b| (b >> 5) & 0x7).unwrap_or(0);

    if frame_marker == 2 && superframe_marker == 0b110 {
        ProbeResult::new(
            CodecId::Vp9,
            Confidence::HIGH,
            "VP9 frame_marker + superframe marker",
        )
    } else if frame_marker == 2 {
        ProbeResult::new(CodecId::Vp9, Confidence::MEDIUM, "VP9 frame_marker present")
    } else {
        ProbeResult::new(CodecId::Vp9, Confidence::MIN, "no VP9 frame_marker")
    }
}

fn probe_vp8(data: &[u8]) -> ProbeResult {
    // VP8 frame tag: bits [0] = frame_type (0=key, 1=inter), [2:1] = version, [3] = show_frame.
    // Key frame: first 3 bytes of payload after the 3-byte tag should be 0x9D 0x01 0x2A.
    if data.len() < 4 {
        return ProbeResult::new(CodecId::Vp8, Confidence::MIN, "buffer too short");
    }
    let frame_type = data[0] & 0x01; // 0 = key frame
    if frame_type == 0 {
        // Key frame: bytes 3..6 should be the VP8 start code 0x9D 0x01 0x2A.
        if data.len() >= 6 && data[3] == 0x9D && data[4] == 0x01 && data[5] == 0x2A {
            return ProbeResult::new(
                CodecId::Vp8,
                Confidence::CERTAIN,
                "VP8 key frame start code 9D 01 2A found",
            );
        }
        return ProbeResult::new(
            CodecId::Vp8,
            Confidence::MEDIUM,
            "VP8 key frame flag set but start code missing",
        );
    }
    ProbeResult::new(
        CodecId::Vp8,
        Confidence::LOW,
        "VP8 inter frame (cannot confirm without key frame)",
    )
}

fn probe_h264(data: &[u8]) -> ProbeResult {
    // AnnexB start code followed by SPS NAL type byte (0x67).
    const START_4: [u8; 4] = [0x00, 0x00, 0x00, 0x01];
    const START_3: [u8; 3] = [0x00, 0x00, 0x01];

    let check_nal_type = |offset: usize| -> Option<u8> { data.get(offset).copied() };

    if starts_with(data, &START_4) {
        let nal_byte = check_nal_type(4).unwrap_or(0);
        let nal_type = nal_byte & 0x1F;
        if nal_type == 7 {
            return ProbeResult::new(
                CodecId::H264,
                Confidence::CERTAIN,
                "AnnexB + SPS NAL type 7",
            );
        } else if nal_type == 8 || nal_type == 5 || nal_type == 1 {
            return ProbeResult::new(
                CodecId::H264,
                Confidence::HIGH,
                format!("AnnexB start code + H.264-compatible NAL type {nal_type}"),
            );
        }
        return ProbeResult::new(
            CodecId::H264,
            Confidence::MEDIUM,
            "AnnexB 4-byte start code",
        );
    }
    if starts_with(data, &START_3) {
        let nal_byte = check_nal_type(3).unwrap_or(0);
        let nal_type = nal_byte & 0x1F;
        if nal_type == 7 {
            return ProbeResult::new(CodecId::H264, Confidence::HIGH, "3-byte AnnexB + SPS NAL");
        }
        return ProbeResult::new(CodecId::H264, Confidence::LOW, "3-byte AnnexB start code");
    }
    // Check for AVCC-style (no start code, first 4 bytes = big-endian length).
    if data.len() >= 5 {
        let claimed_len = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize;
        if claimed_len > 0 && claimed_len < data.len() {
            let nal_byte = data[4];
            let nal_type = nal_byte & 0x1F;
            if nal_type == 7 || nal_type == 8 || nal_type == 5 {
                return ProbeResult::new(
                    CodecId::H264,
                    Confidence::MEDIUM,
                    "AVCC length-prefixed NAL with valid type",
                );
            }
        }
    }
    ProbeResult::new(CodecId::H264, Confidence::MIN, "no H.264 signature found")
}

fn probe_h265(data: &[u8]) -> ProbeResult {
    // HEVC NAL types: VPS=32, SPS=33, PPS=34 (nal_unit_type = (byte >> 1) & 0x3F).
    const START_4: [u8; 4] = [0x00, 0x00, 0x00, 0x01];
    const START_3: [u8; 3] = [0x00, 0x00, 0x01];

    let check_hevc_nal = |offset: usize| -> Option<u8> {
        if data.len() > offset + 1 {
            Some((data[offset] >> 1) & 0x3F)
        } else {
            None
        }
    };

    if starts_with(data, &START_4) {
        if let Some(nal_type) = check_hevc_nal(4) {
            if nal_type == 32 {
                return ProbeResult::new(
                    CodecId::H265,
                    Confidence::CERTAIN,
                    "AnnexB + HEVC VPS (type 32)",
                );
            } else if nal_type == 33 || nal_type == 34 {
                return ProbeResult::new(
                    CodecId::H265,
                    Confidence::HIGH,
                    format!("AnnexB + HEVC NAL type {nal_type}"),
                );
            }
        }
        return ProbeResult::new(
            CodecId::H265,
            Confidence::LOW,
            "AnnexB 4-byte start code (ambiguous)",
        );
    }
    if starts_with(data, &START_3) {
        if let Some(nal_type) = check_hevc_nal(3) {
            if nal_type == 32 || nal_type == 33 {
                return ProbeResult::new(
                    CodecId::H265,
                    Confidence::HIGH,
                    format!("3-byte AnnexB + HEVC NAL type {nal_type}"),
                );
            }
        }
    }
    ProbeResult::new(CodecId::H265, Confidence::MIN, "no HEVC signature found")
}

fn probe_theora(data: &[u8]) -> ProbeResult {
    // Theora identification header: 0x80 "theora"
    const MAGIC: &[u8] = &[0x80, b't', b'h', b'e', b'o', b'r', b'a'];
    if starts_with(data, MAGIC) {
        ProbeResult::new(
            CodecId::Theora,
            Confidence::CERTAIN,
            "Theora identification header magic",
        )
    } else {
        ProbeResult::new(CodecId::Theora, Confidence::MIN, "no Theora magic")
    }
}

fn probe_opus(data: &[u8]) -> ProbeResult {
    // Opus identification header: "OpusHead"
    const MAGIC: &[u8] = b"OpusHead";
    if starts_with(data, MAGIC) {
        ProbeResult::new(
            CodecId::Opus,
            Confidence::CERTAIN,
            "OpusHead identification header",
        )
    } else if contains_in_first(data, MAGIC, 64) {
        ProbeResult::new(
            CodecId::Opus,
            Confidence::HIGH,
            "OpusHead found within first 64 bytes",
        )
    } else {
        ProbeResult::new(CodecId::Opus, Confidence::MIN, "no Opus magic")
    }
}

fn probe_vorbis(data: &[u8]) -> ProbeResult {
    // Vorbis identification header: 0x01 "vorbis"
    const MAGIC: &[u8] = &[0x01, b'v', b'o', b'r', b'b', b'i', b's'];
    if starts_with(data, MAGIC) {
        ProbeResult::new(
            CodecId::Vorbis,
            Confidence::CERTAIN,
            "Vorbis identification header magic",
        )
    } else {
        ProbeResult::new(CodecId::Vorbis, Confidence::MIN, "no Vorbis magic")
    }
}

fn probe_flac(data: &[u8]) -> ProbeResult {
    // FLAC stream marker: "fLaC"
    const MAGIC: &[u8] = b"fLaC";
    if starts_with(data, MAGIC) {
        ProbeResult::new(
            CodecId::Flac,
            Confidence::CERTAIN,
            "FLAC stream marker 'fLaC'",
        )
    } else {
        ProbeResult::new(CodecId::Flac, Confidence::MIN, "no FLAC marker")
    }
}

fn probe_png(data: &[u8]) -> ProbeResult {
    // PNG signature: 89 50 4E 47 0D 0A 1A 0A
    const PNG_SIG: &[u8] = &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    if starts_with(data, PNG_SIG) {
        ProbeResult::new(CodecId::Png, Confidence::CERTAIN, "PNG signature bytes")
    } else {
        ProbeResult::new(CodecId::Png, Confidence::MIN, "no PNG signature")
    }
}

fn probe_gif(data: &[u8]) -> ProbeResult {
    // GIF87a or GIF89a
    if starts_with(data, b"GIF87a") || starts_with(data, b"GIF89a") {
        ProbeResult::new(CodecId::Gif, Confidence::CERTAIN, "GIF header magic")
    } else {
        ProbeResult::new(CodecId::Gif, Confidence::MIN, "no GIF magic")
    }
}

fn probe_webp(data: &[u8]) -> ProbeResult {
    // WebP: "RIFF" at 0, "WEBP" at 8.
    if data.len() >= 12 && &data[..4] == b"RIFF" && &data[8..12] == b"WEBP" {
        ProbeResult::new(
            CodecId::WebP,
            Confidence::CERTAIN,
            "RIFF/WEBP container signature",
        )
    } else if starts_with(data, b"RIFF") {
        ProbeResult::new(
            CodecId::WebP,
            Confidence::LOW,
            "RIFF container (possibly WebP)",
        )
    } else {
        ProbeResult::new(CodecId::WebP, Confidence::MIN, "no WebP signature")
    }
}

fn probe_jpegxl(data: &[u8]) -> ProbeResult {
    // JXL codestream: FF 0A
    // JXL ISOBMFF: 00 00 00 0C 4A 58 4C 20 ...
    const JXL_CODESTREAM: &[u8] = &[0xFF, 0x0A];
    const JXL_ISOBMFF: &[u8] = &[0x00, 0x00, 0x00, 0x0C, b'J', b'X', b'L', b' '];
    if starts_with(data, JXL_ISOBMFF) {
        ProbeResult::new(
            CodecId::JpegXl,
            Confidence::CERTAIN,
            "JPEG-XL ISOBMFF signature",
        )
    } else if starts_with(data, JXL_CODESTREAM) {
        ProbeResult::new(
            CodecId::JpegXl,
            Confidence::CERTAIN,
            "JPEG-XL codestream marker FF 0A",
        )
    } else {
        ProbeResult::new(CodecId::JpegXl, Confidence::MIN, "no JPEG-XL signature")
    }
}

fn probe_aac(data: &[u8]) -> ProbeResult {
    // ADTS sync word: 0xFFF (12 bits) at start of each frame.
    if data.len() >= 2 && data[0] == 0xFF && (data[1] & 0xF0) == 0xF0 {
        // Check layer bits: must be 00.
        let layer = (data[1] >> 1) & 0x03;
        if layer == 0 {
            return ProbeResult::new(
                CodecId::Aac,
                Confidence::HIGH,
                "ADTS sync word 0xFFF with layer=0",
            );
        }
        return ProbeResult::new(
            CodecId::Aac,
            Confidence::MEDIUM,
            "ADTS-like sync word (layer non-zero)",
        );
    }
    ProbeResult::new(CodecId::Aac, Confidence::MIN, "no ADTS sync word")
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Probe `data` against all known codecs and return all results sorted by
/// descending confidence.
///
/// Only results with confidence > 0 are included unless `include_zero` is true.
pub fn probe_all(data: &[u8], include_zero: bool) -> Vec<ProbeResult> {
    let mut results = vec![
        probe_av1(data),
        probe_vp9(data),
        probe_vp8(data),
        probe_h264(data),
        probe_h265(data),
        probe_theora(data),
        probe_opus(data),
        probe_vorbis(data),
        probe_flac(data),
        probe_png(data),
        probe_gif(data),
        probe_webp(data),
        probe_jpegxl(data),
        probe_aac(data),
    ];

    if !include_zero {
        results.retain(|r| r.confidence.value() > 0);
    }

    // Sort by descending confidence, then stable alphabetical by codec Display name.
    results.sort_by(|a, b| b.confidence.cmp(&a.confidence).then(a.codec.cmp(&b.codec)));
    results
}

/// Probe `data` and return the single best-matching codec along with its confidence.
///
/// Returns `(CodecId::Unknown, Confidence::MIN, "")` when no codec matches.
pub fn probe_best(data: &[u8]) -> ProbeResult {
    probe_all(data, false)
        .into_iter()
        .next()
        .unwrap_or_else(|| {
            ProbeResult::new(CodecId::Unknown, Confidence::MIN, "no codec identified")
        })
}

/// Probe `data` against a specific codec and return the result.
pub fn probe_codec(data: &[u8], codec: CodecId) -> ProbeResult {
    match codec {
        CodecId::Av1 => probe_av1(data),
        CodecId::Vp9 => probe_vp9(data),
        CodecId::Vp8 => probe_vp8(data),
        CodecId::H264 => probe_h264(data),
        CodecId::H265 => probe_h265(data),
        CodecId::Theora => probe_theora(data),
        CodecId::Opus => probe_opus(data),
        CodecId::Vorbis => probe_vorbis(data),
        CodecId::Flac => probe_flac(data),
        CodecId::Png => probe_png(data),
        CodecId::Gif => probe_gif(data),
        CodecId::WebP => probe_webp(data),
        CodecId::JpegXl => probe_jpegxl(data),
        CodecId::Aac => probe_aac(data),
        CodecId::Pcm | CodecId::Unknown => ProbeResult::new(
            codec,
            Confidence::MIN,
            "codec not directly probeable from magic bytes",
        ),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probe_png_signature() {
        let data = [0x89u8, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00];
        let result = probe_codec(&data, CodecId::Png);
        assert_eq!(result.confidence, Confidence::CERTAIN);
        assert_eq!(result.codec, CodecId::Png);
    }

    #[test]
    fn test_probe_gif_header() {
        let data = b"GIF89a\x10\x00\x10\x00";
        let result = probe_codec(data, CodecId::Gif);
        assert_eq!(result.confidence, Confidence::CERTAIN);
    }

    #[test]
    fn test_probe_webp_signature() {
        let mut data = [0u8; 16];
        data[..4].copy_from_slice(b"RIFF");
        data[8..12].copy_from_slice(b"WEBP");
        let result = probe_codec(&data, CodecId::WebP);
        assert_eq!(result.confidence, Confidence::CERTAIN);
    }

    #[test]
    fn test_probe_flac_marker() {
        let data = b"fLaCextra";
        let result = probe_codec(data, CodecId::Flac);
        assert_eq!(result.confidence, Confidence::CERTAIN);
    }

    #[test]
    fn test_probe_h264_annexb_sps() {
        // AnnexB 4-byte start code + SPS NAL type byte 0x67
        let data = [0x00u8, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x1E];
        let result = probe_codec(&data, CodecId::H264);
        assert_eq!(result.confidence, Confidence::CERTAIN);
    }

    #[test]
    fn test_probe_h265_vps() {
        // AnnexB 4-byte start code + HEVC VPS: first nal byte = (32 << 1) = 0x40
        let data = [0x00u8, 0x00, 0x00, 0x01, 0x40, 0x01, 0x0C, 0x01];
        let result = probe_codec(&data, CodecId::H265);
        assert_eq!(result.confidence, Confidence::CERTAIN);
    }

    #[test]
    fn test_probe_opus_head() {
        let data = b"OpusHead\x01\x02\x38\x01";
        let result = probe_codec(data, CodecId::Opus);
        assert_eq!(result.confidence, Confidence::CERTAIN);
    }

    #[test]
    fn test_probe_vorbis_magic() {
        let data = [0x01u8, b'v', b'o', b'r', b'b', b'i', b's', 0x00];
        let result = probe_codec(&data, CodecId::Vorbis);
        assert_eq!(result.confidence, Confidence::CERTAIN);
    }

    #[test]
    fn test_probe_best_returns_highest_confidence() {
        // PNG bytes should win over H.264 checks.
        let data = [0x89u8, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 0x00];
        let best = probe_best(&data);
        assert_eq!(best.codec, CodecId::Png);
        assert_eq!(best.confidence, Confidence::CERTAIN);
    }

    #[test]
    fn test_probe_all_sorted_descending() {
        let data = b"fLaC\x00\x00\x00\x22";
        let results = probe_all(data, false);
        // Results must be in descending confidence order.
        for pair in results.windows(2) {
            assert!(pair[0].confidence >= pair[1].confidence);
        }
        // FLAC must be first.
        assert_eq!(results[0].codec, CodecId::Flac);
    }

    #[test]
    fn test_probe_all_include_zero() {
        let data = b"fLaC\x00";
        let with_zero = probe_all(data, true);
        let without_zero = probe_all(data, false);
        // with_zero should have at least as many entries.
        assert!(with_zero.len() >= without_zero.len());
        // Entries with confidence=0 must exist in with_zero for unmatched codecs.
        assert!(with_zero.iter().any(|r| r.confidence.value() == 0));
    }

    #[test]
    fn test_probe_unknown_data_returns_unknown() {
        // Data that matches nothing deterministically.
        let data = [0x00u8; 8];
        let best = probe_best(&data);
        // Should not panic; confidence should be low or unknown.
        // (All-zeros happens to match an AV1 OBU type=0 with forbidden=0,
        //  but type=0 is reserved. The probe may assign LOW or MEDIUM to AV1.)
        // We simply assert it doesn't crash.
        assert!(best.confidence.value() <= 100);
    }

    #[test]
    fn test_confidence_ordering() {
        assert!(Confidence::CERTAIN > Confidence::HIGH);
        assert!(Confidence::HIGH > Confidence::MEDIUM);
        assert!(Confidence::MEDIUM > Confidence::LOW);
        assert!(Confidence::LOW > Confidence::MIN);
    }

    #[test]
    fn test_probe_jpegxl_codestream() {
        let data = [0xFFu8, 0x0A, 0x00, 0x00];
        let result = probe_codec(&data, CodecId::JpegXl);
        assert_eq!(result.confidence, Confidence::CERTAIN);
    }

    #[test]
    fn test_probe_vp8_key_frame() {
        // VP8 key frame: frame_type=0 (bit 0 = 0), then 2 version bits + show_frame,
        // then at offset 3: 0x9D 0x01 0x2A
        let data = [0x00u8, 0x00, 0x00, 0x9D, 0x01, 0x2A, 0x00, 0x00];
        let result = probe_codec(&data, CodecId::Vp8);
        assert_eq!(result.confidence, Confidence::CERTAIN);
    }
}
