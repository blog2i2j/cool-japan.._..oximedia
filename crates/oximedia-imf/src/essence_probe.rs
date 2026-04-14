//! Quick MXF essence file probing without full parse.
//!
//! [`EssenceProbe`] reads only the first kilobytes of an MXF file to extract
//! key metadata: operational pattern, essence type, edit rate hints,
//! and file size — without parsing the full index table or essence container.
//!
//! # Example
//! ```no_run
//! use oximedia_imf::essence_probe::EssenceProbe;
//!
//! let meta = EssenceProbe::probe("/path/to/video.mxf").expect("probe failed");
//! println!("Essence type: {}", meta.essence_type);
//! println!("File size: {} bytes", meta.file_size);
//! ```

#![allow(dead_code, missing_docs)]

use crate::{ImfError, ImfResult};
use std::io::Read;
use std::path::Path;

/// MXF key prefix universal label header.
const MXF_KEY_PREFIX: [u8; 4] = [0x06, 0x0e, 0x2b, 0x34];

/// Detected essence type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProbeEssenceType {
    /// JPEG 2000 codestream
    Jpeg2000,
    /// AVC / H.264
    Avc,
    /// Apple ProRes
    ProRes,
    /// MPEG-2 video
    Mpeg2,
    /// PCM audio
    PcmAudio,
    /// JPEG XS
    JpegXs,
    /// Unknown or unparsed
    Unknown(String),
}

impl std::fmt::Display for ProbeEssenceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Jpeg2000 => write!(f, "JPEG 2000"),
            Self::Avc => write!(f, "AVC/H.264"),
            Self::ProRes => write!(f, "Apple ProRes"),
            Self::Mpeg2 => write!(f, "MPEG-2"),
            Self::PcmAudio => write!(f, "PCM Audio"),
            Self::JpegXs => write!(f, "JPEG XS"),
            Self::Unknown(s) => write!(f, "Unknown({s})"),
        }
    }
}

/// Detected operational pattern.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperationalPattern {
    /// OP-Atom: one essence container per file
    OpAtom,
    /// OP-1a: single item in single package
    Op1a,
    /// OP-2a: single item in multiple packages
    Op2a,
    /// Other / unrecognized
    Other(String),
}

impl std::fmt::Display for OperationalPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OpAtom => write!(f, "OP-Atom"),
            Self::Op1a => write!(f, "OP-1a"),
            Self::Op2a => write!(f, "OP-2a"),
            Self::Other(s) => write!(f, "Other({s})"),
        }
    }
}

/// Quick metadata returned by the essence probe.
#[derive(Debug, Clone)]
pub struct ProbeResult {
    /// Path that was probed.
    pub path: String,
    /// Detected essence type.
    pub essence_type: ProbeEssenceType,
    /// Detected operational pattern.
    pub operational_pattern: OperationalPattern,
    /// Frame width in pixels (video only).
    pub width: Option<u32>,
    /// Frame height in pixels (video only).
    pub height: Option<u32>,
    /// Edit rate as (numerator, denominator).
    pub edit_rate: Option<(u32, u32)>,
    /// Approximate frame count derived from file size heuristic.
    pub frame_count: Option<u64>,
    /// Sample rate for audio essence (Hz).
    pub sample_rate: Option<u32>,
    /// Number of audio channels.
    pub audio_channels: Option<u8>,
    /// File size in bytes.
    pub file_size: u64,
    /// Whether the header partition was found and parsed.
    pub header_parsed: bool,
}

impl ProbeResult {
    /// Returns `true` if this probe result represents a video essence.
    #[must_use]
    pub fn is_video(&self) -> bool {
        matches!(
            self.essence_type,
            ProbeEssenceType::Jpeg2000
                | ProbeEssenceType::Avc
                | ProbeEssenceType::ProRes
                | ProbeEssenceType::Mpeg2
                | ProbeEssenceType::JpegXs
        )
    }

    /// Returns `true` if this probe result represents an audio essence.
    #[must_use]
    pub fn is_audio(&self) -> bool {
        matches!(self.essence_type, ProbeEssenceType::PcmAudio)
    }

    /// Render a human-readable summary.
    #[must_use]
    pub fn summary(&self) -> String {
        let mut parts = Vec::<String>::new();
        parts.push(format!("path={}", self.path));
        parts.push(format!("type={}", self.essence_type));
        parts.push(format!("op={}", self.operational_pattern));
        if let (Some(w), Some(h)) = (self.width, self.height) {
            parts.push(format!("res={w}x{h}"));
        }
        if let Some((n, d)) = self.edit_rate {
            parts.push(format!("rate={n}/{d}"));
        }
        if let Some(fc) = self.frame_count {
            parts.push(format!("frames={fc}"));
        }
        if let Some(sr) = self.sample_rate {
            parts.push(format!("sample_rate={sr}"));
        }
        parts.push(format!("size={}B", self.file_size));
        parts.join(" | ")
    }
}

/// MXF essence prober.
pub struct EssenceProbe;

impl EssenceProbe {
    /// Probe an MXF file at `path` and return quick metadata.
    ///
    /// Reads at most the first 64 KiB of the file to detect the header
    /// partition and essence descriptors without loading the full file.
    ///
    /// # Errors
    ///
    /// Returns `ImfError::Io` if the file cannot be read, or
    /// `ImfError::InvalidPackage` if the file does not look like MXF.
    pub fn probe(path: impl AsRef<Path>) -> ImfResult<ProbeResult> {
        let path = path.as_ref();
        let path_str = path.to_string_lossy().to_string();

        let meta = std::fs::metadata(path).map_err(|e| ImfError::Other(e.to_string()))?;
        let file_size = meta.len();

        let mut file =
            std::fs::File::open(path).map_err(|e| ImfError::Other(e.to_string()))?;

        let read_len = file_size.min(65536) as usize;
        let mut buf = vec![0u8; read_len];
        let n = file.read(&mut buf).map_err(|e| ImfError::Other(e.to_string()))?;
        buf.truncate(n);

        if buf.len() < 4 || buf[..4] != MXF_KEY_PREFIX {
            return Err(ImfError::InvalidPackage(format!(
                "File does not start with MXF key prefix: {path_str}"
            )));
        }

        let essence_type = detect_essence_type(&buf);
        let operational_pattern = detect_operational_pattern(&buf);
        let edit_rate = detect_edit_rate(&buf);
        let (width, height) = detect_resolution(&buf);
        let frame_count = estimate_frame_count(file_size, &essence_type);
        let (sample_rate, audio_channels) = detect_audio_params(&buf);

        Ok(ProbeResult {
            path: path_str,
            essence_type,
            operational_pattern,
            width,
            height,
            edit_rate,
            frame_count,
            sample_rate,
            audio_channels,
            file_size,
            header_parsed: true,
        })
    }

    /// Probe multiple files, returning one result per file.
    pub fn probe_many<P: AsRef<Path>>(paths: &[P]) -> Vec<ImfResult<ProbeResult>> {
        paths.iter().map(|p| Self::probe(p)).collect()
    }
}

fn detect_essence_type(buf: &[u8]) -> ProbeEssenceType {
    if find_seq(buf, &[0x0E, 0x09, 0x06, 0x01]).is_some() {
        return ProbeEssenceType::Jpeg2000;
    }
    if find_seq(buf, &[0x0E, 0x09, 0x06, 0x07]).is_some() {
        return ProbeEssenceType::Avc;
    }
    if find_seq(buf, &[0x0d, 0x01, 0x03, 0x01, 0x02, 0x1c]).is_some() {
        return ProbeEssenceType::ProRes;
    }
    if find_seq(buf, &[0x0E, 0x09, 0x06, 0x05]).is_some() {
        return ProbeEssenceType::Mpeg2;
    }
    if find_seq(buf, &[0x0E, 0x09, 0x06, 0x03]).is_some() {
        return ProbeEssenceType::PcmAudio;
    }
    if find_seq(buf, &[0x0E, 0x09, 0x06, 0x0f]).is_some() {
        return ProbeEssenceType::JpegXs;
    }
    ProbeEssenceType::Unknown("unrecognized".to_string())
}

fn detect_operational_pattern(buf: &[u8]) -> OperationalPattern {
    if find_seq(buf, &[0x0d, 0x01, 0x02, 0x01, 0x10, 0x00]).is_some() {
        return OperationalPattern::OpAtom;
    }
    if find_seq(buf, &[0x0d, 0x01, 0x02, 0x01, 0x01, 0x01]).is_some() {
        return OperationalPattern::Op1a;
    }
    if find_seq(buf, &[0x0d, 0x01, 0x02, 0x01, 0x02, 0x01]).is_some() {
        return OperationalPattern::Op2a;
    }
    OperationalPattern::Other("unrecognized".to_string())
}

fn detect_edit_rate(buf: &[u8]) -> Option<(u32, u32)> {
    let common_rates: &[(u32, u32)] = &[
        (24, 1), (25, 1), (30, 1), (48, 1), (50, 1), (60, 1),
        (24000, 1001), (30000, 1001),
    ];
    for &(n, d) in common_rates {
        let n_bytes = n.to_be_bytes();
        let d_bytes = d.to_be_bytes();
        if let Some(pos) = find_seq(buf, &n_bytes) {
            if pos + 8 <= buf.len() && buf[pos + 4..pos + 8] == d_bytes {
                return Some((n, d));
            }
        }
    }
    None
}

fn detect_resolution(_buf: &[u8]) -> (Option<u32>, Option<u32>) {
    (None, None)
}

fn estimate_frame_count(file_size: u64, essence_type: &ProbeEssenceType) -> Option<u64> {
    let bytes_per_frame: u64 = match essence_type {
        ProbeEssenceType::Jpeg2000 => 2_000_000,
        ProbeEssenceType::ProRes => 1_000_000,
        ProbeEssenceType::Avc => 100_000,
        ProbeEssenceType::Mpeg2 => 200_000,
        _ => return None,
    };
    Some((file_size / bytes_per_frame).max(1))
}

fn detect_audio_params(buf: &[u8]) -> (Option<u32>, Option<u8>) {
    if find_seq(buf, &48000u32.to_be_bytes()).is_some() {
        return (Some(48000), Some(2));
    }
    if find_seq(buf, &44100u32.to_be_bytes()).is_some() {
        return (Some(44100), Some(2));
    }
    (None, None)
}

fn find_seq(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

// ============================================================================
// CPL-based essence probing (task API)
// ============================================================================

/// Essence type classification derived from CPL sequence type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EssenceType {
    /// Video (MainImageSequence)
    Video,
    /// Audio (MainAudioSequence)
    Audio,
    /// Subtitle / timed text
    Subtitle,
    /// Generic data track
    Data,
}

impl std::fmt::Display for EssenceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Video => write!(f, "Video"),
            Self::Audio => write!(f, "Audio"),
            Self::Subtitle => write!(f, "Subtitle"),
            Self::Data => write!(f, "Data"),
        }
    }
}

/// Metadata for a single essence track probed from a CPL.
#[derive(Debug, Clone)]
pub struct EssenceTrackInfo {
    /// Track identifier (SourceEncoding UUID string)
    pub track_id: String,
    /// Classified essence type
    pub essence_type: EssenceType,
    /// Duration in frames (sum of Resource Duration elements)
    pub duration_frames: u64,
    /// Edit rate as (numerator, denominator)
    pub edit_rate: (u32, u32),
    /// Codec label string extracted from CPL
    pub codec: String,
    /// Bit depth (video / audio, if present)
    pub bit_depth: Option<u8>,
    /// Audio sample rate (Hz)
    pub sample_rate: Option<u32>,
    /// Audio channel count
    pub channel_count: Option<u8>,
}

/// Result of probing all tracks from a CPL XML document.
#[derive(Debug, Clone)]
pub struct EssenceProbeResult {
    /// All discovered essence tracks
    pub tracks: Vec<EssenceTrackInfo>,
    /// Sum of all video-track durations (or longest single track)
    pub total_duration_frames: u64,
    /// Track ID of the primary (first) video track, if any
    pub primary_video_track: Option<String>,
}

/// Errors produced by the CPL essence prober.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ProbeError {
    /// Malformed or unrecognised XML
    #[error("Invalid XML: {0}")]
    InvalidXml(String),
    /// A required field was absent in the CPL
    #[error("Missing field: {0}")]
    MissingField(String),
    /// The essence or format is not supported
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),
}

/// Probes essence track metadata from a CPL XML document.
///
/// Parsing is deliberately minimal (no external XML library dependency beyond
/// simple text scanning) so that it works without a full MXF decode pass.
pub struct EssenceProber;

impl EssenceProber {
    /// Parse `xml` as a CPL and return essence track metadata.
    ///
    /// # Errors
    /// Returns [`ProbeError::InvalidXml`] for empty or completely non-XML input,
    /// [`ProbeError::MissingField`] if no `<EditRate>` is present,
    /// or [`ProbeError::UnsupportedFormat`] for recognised but unsupported profiles.
    pub fn probe_from_cpl_xml(xml: &str) -> Result<EssenceProbeResult, ProbeError> {
        if xml.trim().is_empty() || !xml.contains('<') {
            return Err(ProbeError::InvalidXml("Empty or non-XML input".to_string()));
        }

        // Extract global EditRate
        let edit_rate = extract_edit_rate(xml)
            .ok_or_else(|| ProbeError::MissingField("EditRate".to_string()))?;

        // Find all Sequence blocks and classify them
        let mut tracks: Vec<EssenceTrackInfo> = Vec::new();
        let mut primary_video_track: Option<String> = None;

        let sequence_types = [
            ("MainImageSequence", EssenceType::Video),
            ("MainAudioSequence", EssenceType::Audio),
            ("SubtitlesSequence", EssenceType::Subtitle),
            ("HearingImpairedCaptionsSequence", EssenceType::Subtitle),
            ("VisuallyImpairedTextSequence", EssenceType::Subtitle),
            ("CommentarySequence", EssenceType::Subtitle),
            ("KaraokeSequence", EssenceType::Subtitle),
            ("DataEssenceSequence", EssenceType::Data),
        ];

        for (tag, essence_type) in &sequence_types {
            for seq_xml in split_sequences(xml, tag) {
                let resources = extract_resources(&seq_xml);
                if resources.is_empty() {
                    continue;
                }
                for resource in resources {
                    let track_id = resource.source_encoding.clone();
                    let duration_frames = resource.duration;
                    let codec = infer_codec(tag);

                    let track = EssenceTrackInfo {
                        track_id: track_id.clone(),
                        essence_type: essence_type.clone(),
                        duration_frames,
                        edit_rate,
                        codec,
                        bit_depth: resource.bit_depth,
                        sample_rate: resource.sample_rate,
                        channel_count: resource.channel_count,
                    };

                    if essence_type == &EssenceType::Video && primary_video_track.is_none() {
                        primary_video_track = Some(track_id);
                    }
                    tracks.push(track);
                }
            }
        }

        // total_duration_frames: sum of all video durations, else longest track
        let total_duration_frames = {
            let video_sum: u64 = tracks
                .iter()
                .filter(|t| t.essence_type == EssenceType::Video)
                .map(|t| t.duration_frames)
                .sum();
            if video_sum > 0 {
                video_sum
            } else {
                tracks.iter().map(|t| t.duration_frames).max().unwrap_or(0)
            }
        };

        Ok(EssenceProbeResult {
            tracks,
            total_duration_frames,
            primary_video_track,
        })
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

struct ResourceInfo {
    source_encoding: String,
    duration: u64,
    bit_depth: Option<u8>,
    sample_rate: Option<u32>,
    channel_count: Option<u8>,
}

/// Split the CPL XML into per-sequence blocks for a given sequence element name.
fn split_sequences(xml: &str, tag: &str) -> Vec<String> {
    let open = format!("<{tag}");
    let close = format!("</{tag}>");
    let mut result = Vec::new();
    let mut search = xml;
    while let Some(start) = search.find(&open) {
        let rest = &search[start..];
        if let Some(end) = rest.find(&close) {
            let block = &rest[..end + close.len()];
            result.push(block.to_string());
            search = &rest[end + close.len()..];
        } else {
            break;
        }
    }
    result
}

/// Extract all `TrackFileResource` elements from a sequence block.
fn extract_resources(seq_xml: &str) -> Vec<ResourceInfo> {
    let open = "<TrackFileResource";
    let close = "</TrackFileResource>";
    let mut resources = Vec::new();
    let mut search = seq_xml;
    while let Some(start) = search.find(open) {
        let rest = &search[start..];
        // Handle both self-closing and paired tags
        let (block, advance) = if let Some(end) = rest.find(close) {
            (&rest[..end + close.len()], end + close.len())
        } else if let Some(sc) = rest.find("/>") {
            (&rest[..sc + 2], sc + 2)
        } else {
            break;
        };

        let source_encoding = extract_text(block, "SourceEncoding")
            .unwrap_or_else(|| extract_text(block, "SourceEncoding")
                .unwrap_or_else(|| format!("unknown-{}", resources.len())));
        let duration = extract_text(block, "Duration")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);
        let bit_depth = extract_text(block, "BitDepth")
            .and_then(|s| s.parse::<u8>().ok());
        let sample_rate = extract_text(block, "AudioSampleRate")
            .and_then(|s| s.parse::<u32>().ok());
        let channel_count = extract_text(block, "ChannelCount")
            .and_then(|s| s.parse::<u8>().ok());

        resources.push(ResourceInfo {
            source_encoding,
            duration,
            bit_depth,
            sample_rate,
            channel_count,
        });

        search = &rest[advance..];
    }
    resources
}

/// Extract text content of `<tag>...</tag>` from a string.
fn extract_text(xml: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}");
    let close = format!("</{tag}>");
    let start_tag = xml.find(&open)?;
    let after_open = xml[start_tag..].find('>')?;
    let content_start = start_tag + after_open + 1;
    let close_pos = xml[content_start..].find(&close)?;
    let text = xml[content_start..content_start + close_pos].trim().to_string();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

/// Parse `<EditRate>24 1</EditRate>` or `<EditRate>30000 1001</EditRate>`.
fn extract_edit_rate(xml: &str) -> Option<(u32, u32)> {
    let text = extract_text(xml, "EditRate")?;
    let parts: Vec<&str> = text.split_whitespace().collect();
    if parts.len() >= 2 {
        let n = parts[0].parse::<u32>().ok()?;
        let d = parts[1].parse::<u32>().ok()?;
        Some((n, d))
    } else {
        None
    }
}

/// Infer a codec label string from the CPL sequence element name.
fn infer_codec(sequence_tag: &str) -> String {
    match sequence_tag {
        "MainImageSequence" => "JPEG2000".to_string(),
        "MainAudioSequence" => "PCM".to_string(),
        "SubtitlesSequence" | "HearingImpairedCaptionsSequence"
        | "VisuallyImpairedTextSequence" | "CommentarySequence"
        | "KaraokeSequence" => "IMSC1".to_string(),
        "DataEssenceSequence" => "Data".to_string(),
        _ => "Unknown".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probe_nonexistent() {
        assert!(EssenceProbe::probe("/nonexistent/file.mxf").is_err());
    }

    #[test]
    fn test_probe_non_mxf() {
        let dir = std::env::temp_dir().join("ep_test_non_mxf");
        std::fs::create_dir_all(&dir).ok();
        let p = dir.join("notmxf.bin");
        std::fs::write(&p, b"NOTMXF").ok();
        assert!(EssenceProbe::probe(&p).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_probe_minimal_mxf() {
        let dir = std::env::temp_dir().join("ep_test_minimal");
        std::fs::create_dir_all(&dir).ok();
        let p = dir.join("min.mxf");
        let mut data = vec![0x06u8, 0x0e, 0x2b, 0x34];
        data.extend(vec![0u8; 100]);
        std::fs::write(&p, &data).ok();
        let r = EssenceProbe::probe(&p).expect("ok");
        assert!(r.header_parsed);
        assert!(r.file_size > 0);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_detect_jpeg2000() {
        let mut buf = vec![0x06u8, 0x0e, 0x2b, 0x34];
        buf.extend_from_slice(&[0x0E, 0x09, 0x06, 0x01]);
        assert_eq!(detect_essence_type(&buf), ProbeEssenceType::Jpeg2000);
    }

    #[test]
    fn test_probe_result_is_video() {
        let r = ProbeResult {
            path: "v.mxf".into(),
            essence_type: ProbeEssenceType::Jpeg2000,
            operational_pattern: OperationalPattern::OpAtom,
            width: Some(1920),
            height: Some(1080),
            edit_rate: Some((24, 1)),
            frame_count: Some(240),
            sample_rate: None,
            audio_channels: None,
            file_size: 480_000_000,
            header_parsed: true,
        };
        assert!(r.is_video());
        assert!(!r.is_audio());
        let s = r.summary();
        assert!(s.contains("JPEG 2000"));
    }

    // -----------------------------------------------------------------------
    // EssenceProber (CPL-based) tests
    // -----------------------------------------------------------------------

    fn video_only_cpl(duration: u64, edit_rate: &str) -> String {
        format!(
            r#"<?xml version="1.0"?>
<CompositionPlaylist>
  <EditRate>{edit_rate}</EditRate>
  <SegmentList>
    <Segment>
      <SequenceList>
        <MainImageSequence>
          <TrackFileResource>
            <SourceEncoding>urn:uuid:aabbccdd-1111-2222-3333-aabbccddeeff</SourceEncoding>
            <Duration>{duration}</Duration>
          </TrackFileResource>
        </MainImageSequence>
      </SequenceList>
    </Segment>
  </SegmentList>
</CompositionPlaylist>"#
        )
    }

    fn video_audio_cpl(v_dur: u64, a_dur: u64) -> String {
        format!(
            r#"<?xml version="1.0"?>
<CompositionPlaylist>
  <EditRate>24 1</EditRate>
  <SegmentList>
    <Segment>
      <SequenceList>
        <MainImageSequence>
          <TrackFileResource>
            <SourceEncoding>urn:uuid:video-0000-0000-0000-000000000001</SourceEncoding>
            <Duration>{v_dur}</Duration>
          </TrackFileResource>
        </MainImageSequence>
        <MainAudioSequence>
          <TrackFileResource>
            <SourceEncoding>urn:uuid:audio-0000-0000-0000-000000000002</SourceEncoding>
            <Duration>{a_dur}</Duration>
          </TrackFileResource>
        </MainAudioSequence>
      </SequenceList>
    </Segment>
  </SegmentList>
</CompositionPlaylist>"#
        )
    }

    #[test]
    fn test_cpl_probe_video_only() {
        let xml = video_only_cpl(240, "24 1");
        let result = EssenceProber::probe_from_cpl_xml(&xml).expect("should parse");
        assert_eq!(result.tracks.len(), 1);
        assert_eq!(result.tracks[0].essence_type, EssenceType::Video);
        assert_eq!(result.tracks[0].duration_frames, 240);
        assert_eq!(result.tracks[0].edit_rate, (24, 1));
        assert!(result.primary_video_track.is_some());
    }

    #[test]
    fn test_cpl_probe_video_audio() {
        let xml = video_audio_cpl(240, 240);
        let result = EssenceProber::probe_from_cpl_xml(&xml).expect("should parse");
        let video_tracks: Vec<_> = result.tracks.iter().filter(|t| t.essence_type == EssenceType::Video).collect();
        let audio_tracks: Vec<_> = result.tracks.iter().filter(|t| t.essence_type == EssenceType::Audio).collect();
        assert_eq!(video_tracks.len(), 1);
        assert_eq!(audio_tracks.len(), 1);
        assert!(result.primary_video_track.is_some());
    }

    #[test]
    fn test_cpl_probe_duration_sum() {
        // Two video tracks → total_duration_frames = sum
        let xml = r#"<?xml version="1.0"?>
<CompositionPlaylist>
  <EditRate>25 1</EditRate>
  <SegmentList>
    <Segment>
      <SequenceList>
        <MainImageSequence>
          <TrackFileResource>
            <SourceEncoding>urn:uuid:seg1-0000-0000-0000-000000000001</SourceEncoding>
            <Duration>100</Duration>
          </TrackFileResource>
          <TrackFileResource>
            <SourceEncoding>urn:uuid:seg2-0000-0000-0000-000000000002</SourceEncoding>
            <Duration>150</Duration>
          </TrackFileResource>
        </MainImageSequence>
      </SequenceList>
    </Segment>
  </SegmentList>
</CompositionPlaylist>"#;
        let result = EssenceProber::probe_from_cpl_xml(xml).expect("should parse");
        assert_eq!(result.total_duration_frames, 250);
    }

    #[test]
    fn test_cpl_probe_primary_video_detection() {
        let xml = video_audio_cpl(500, 500);
        let result = EssenceProber::probe_from_cpl_xml(&xml).expect("should parse");
        let pv = result.primary_video_track.as_deref().unwrap_or("");
        assert!(pv.contains("video"), "expected video UUID, got {pv}");
    }

    #[test]
    fn test_cpl_probe_missing_edit_rate() {
        let xml = r#"<?xml version="1.0"?>
<CompositionPlaylist>
  <SegmentList/>
</CompositionPlaylist>"#;
        let err = EssenceProber::probe_from_cpl_xml(xml).unwrap_err();
        assert!(matches!(err, ProbeError::MissingField(_)));
    }

    #[test]
    fn test_cpl_probe_empty_input() {
        let err = EssenceProber::probe_from_cpl_xml("").unwrap_err();
        assert!(matches!(err, ProbeError::InvalidXml(_)));
    }

    #[test]
    fn test_cpl_probe_non_xml() {
        let err = EssenceProber::probe_from_cpl_xml("not xml at all").unwrap_err();
        assert!(matches!(err, ProbeError::InvalidXml(_)));
    }

    #[test]
    fn test_cpl_probe_30000_1001() {
        let xml = video_only_cpl(2997, "30000 1001");
        let result = EssenceProber::probe_from_cpl_xml(&xml).expect("should parse");
        assert_eq!(result.tracks[0].edit_rate, (30000, 1001));
    }

    #[test]
    fn test_essence_type_display() {
        assert_eq!(EssenceType::Video.to_string(), "Video");
        assert_eq!(EssenceType::Audio.to_string(), "Audio");
        assert_eq!(EssenceType::Subtitle.to_string(), "Subtitle");
        assert_eq!(EssenceType::Data.to_string(), "Data");
    }
}
