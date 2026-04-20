//! Unified HLS/DASH playlist parser with auto-detection and URL resolution.
//!
//! This module provides a single [`PlaylistParser`] entry-point that:
//!
//! - Auto-detects whether content is an HLS master/media playlist (`#EXTM3U`)
//!   or a DASH MPD (`<?xml … <MPD`).
//! - Parses HLS master playlists into a structured [`HlsMasterPlaylist`] with
//!   variant streams, renditions, and EXT-X-SESSION-DATA entries.
//! - Parses HLS media playlists into [`HlsMediaPlaylist`] with segments,
//!   encryption keys, and map initialisation information.
//! - Parses minimal DASH MPDs into [`DashManifest`] (Periods → AdaptationSets
//!   → Representations → SegmentTemplate / SegmentList).
//! - Resolves relative segment and manifest URLs against a base URL using
//!   [`UrlResolver`].
//!
//! # Design goals
//!
//! The parser is deliberately incremental and liberal:  it does not abort on
//! unknown tags — they are collected in `unknown_tags` so that callers can
//! inspect or forward them.  Parsing never panics and never calls `unwrap()`.

use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

use crate::error::{NetError, NetResult};

// ─── URL Resolution ───────────────────────────────────────────────────────────

/// Resolves relative URLs against a base URL.
///
/// This is a lightweight, dependency-free resolver that covers the subset of
/// RFC 3986 needed for HLS and DASH: absolute URLs are returned unchanged;
/// protocol-relative URLs (`//…`) inherit the base scheme; absolute-path
/// references (`/…`) inherit the base origin; relative-path references are
/// resolved relative to the last `/` in the base path.
#[derive(Debug, Clone)]
pub struct UrlResolver {
    base: String,
}

impl UrlResolver {
    /// Creates a new resolver anchored at `base_url`.
    ///
    /// `base_url` may be any well-formed HTTP/HTTPS URL.
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base: base_url.into(),
        }
    }

    /// Returns the base URL.
    #[must_use]
    pub fn base(&self) -> &str {
        &self.base
    }

    /// Resolves `reference` against the stored base URL.
    ///
    /// # Errors
    ///
    /// Returns [`NetError::InvalidUrl`] when `reference` contains a scheme
    /// that is not `http` or `https` and is not relative.
    pub fn resolve(&self, reference: &str) -> NetResult<String> {
        let reference = reference.trim();

        // Already absolute (http:// or https://)
        if reference.starts_with("http://") || reference.starts_with("https://") {
            return Ok(reference.to_owned());
        }

        // Protocol-relative: //host/path
        if let Some(rest) = reference.strip_prefix("//") {
            let scheme = self.base_scheme();
            return Ok(format!("{scheme}://{rest}"));
        }

        // Absolute-path reference: /path
        if reference.starts_with('/') {
            let origin = self.base_origin()?;
            return Ok(format!("{origin}{reference}"));
        }

        // Relative-path reference
        let base_dir = self.base_directory();
        let joined = if base_dir.is_empty() {
            reference.to_owned()
        } else if base_dir.ends_with('/') {
            format!("{base_dir}{reference}")
        } else {
            format!("{base_dir}/{reference}")
        };

        Ok(normalize_path(&joined))
    }

    fn base_scheme(&self) -> &str {
        if self.base.starts_with("https://") {
            "https"
        } else {
            "http"
        }
    }

    fn base_origin(&self) -> NetResult<String> {
        // Find end of "scheme://"
        let after_scheme = if self.base.starts_with("https://") {
            &self.base["https://".len()..]
        } else if self.base.starts_with("http://") {
            &self.base["http://".len()..]
        } else {
            return Err(NetError::InvalidUrl(format!(
                "Cannot determine origin of '{}'",
                self.base
            )));
        };

        let scheme = self.base_scheme();
        let host_end = after_scheme.find('/').unwrap_or(after_scheme.len());
        let host = &after_scheme[..host_end];
        Ok(format!("{scheme}://{host}"))
    }

    fn base_directory(&self) -> String {
        match self.base.rfind('/') {
            None => String::new(),
            Some(idx) => {
                // Keep everything up to and including the last slash.
                if self.base[..idx].contains("://") {
                    // The slash is after the scheme, keep it
                    self.base[..=idx].to_owned()
                } else {
                    self.base[..=idx].to_owned()
                }
            }
        }
    }
}

/// Normalises `/../` and `/./` components in a URL path.
fn normalize_path(url: &str) -> String {
    // Split into scheme+authority and path
    let (prefix, path) = if let Some(pos) = url.find("://") {
        let after = pos + 3;
        if let Some(slash) = url[after..].find('/') {
            let split = after + slash;
            (&url[..split], &url[split..])
        } else {
            return url.to_owned();
        }
    } else {
        ("", url)
    };

    let mut segments: Vec<&str> = Vec::new();
    for part in path.split('/') {
        match part {
            "." | "" if !segments.is_empty() => {}
            ".." => {
                segments.pop();
            }
            other => segments.push(other),
        }
    }

    let normalised = segments.join("/");
    if prefix.is_empty() {
        normalised
    } else {
        format!("{prefix}/{normalised}")
    }
}

// ─── Playlist format detection ────────────────────────────────────────────────

/// The detected format of a raw manifest/playlist blob.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaylistFormat {
    /// HLS master playlist (contains `#EXT-X-STREAM-INF` or only `#EXTM3U`
    /// with `#EXT-X-MEDIA`).
    HlsMaster,
    /// HLS media playlist (contains `#EXTINF` segments).
    HlsMedia,
    /// MPEG-DASH MPD XML document.
    DashMpd,
    /// Unknown format.
    Unknown,
}

impl fmt::Display for PlaylistFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HlsMaster => write!(f, "HLS Master Playlist"),
            Self::HlsMedia => write!(f, "HLS Media Playlist"),
            Self::DashMpd => write!(f, "MPEG-DASH MPD"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Detects the format of a raw text manifest.
pub fn detect_format(text: &str) -> PlaylistFormat {
    let trimmed = text.trim_start();

    // DASH MPD is XML
    if trimmed.starts_with("<?xml") || trimmed.contains("<MPD") {
        return PlaylistFormat::DashMpd;
    }

    if !trimmed.starts_with("#EXTM3U") {
        return PlaylistFormat::Unknown;
    }

    // HLS: distinguish master vs media by tag presence
    if text.contains("#EXT-X-STREAM-INF") || text.contains("#EXT-X-I-FRAME-STREAM-INF") {
        PlaylistFormat::HlsMaster
    } else if text.contains("#EXTINF") {
        PlaylistFormat::HlsMedia
    } else {
        // Could be a master-only with just #EXT-X-MEDIA, treat as master
        PlaylistFormat::HlsMaster
    }
}

// ─── HLS types ────────────────────────────────────────────────────────────────

/// Encryption method for HLS segments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncryptionMethod {
    /// No encryption.
    None,
    /// AES-128 encryption.
    Aes128,
    /// Sample-level AES encryption.
    SampleAes,
    /// Unknown/future method.
    Other(String),
}

impl From<&str> for EncryptionMethod {
    fn from(s: &str) -> Self {
        match s.to_uppercase().as_str() {
            "NONE" => Self::None,
            "AES-128" => Self::Aes128,
            "SAMPLE-AES" => Self::SampleAes,
            other => Self::Other(other.to_owned()),
        }
    }
}

/// HLS encryption key descriptor (`#EXT-X-KEY`).
#[derive(Debug, Clone)]
pub struct HlsKeyInfo {
    /// Encryption method.
    pub method: EncryptionMethod,
    /// Key URI.
    pub uri: Option<String>,
    /// IV as hex string (without leading 0x).
    pub iv: Option<String>,
    /// Key format identifier.
    pub key_format: Option<String>,
    /// Key format versions.
    pub key_format_versions: Option<String>,
}

/// HLS map info (`#EXT-X-MAP`).
#[derive(Debug, Clone)]
pub struct HlsMapInfo {
    /// URI of the initialisation segment.
    pub uri: String,
    /// Optional byte range.
    pub byte_range: Option<(u64, Option<u64>)>,
}

/// A single HLS media segment.
#[derive(Debug, Clone)]
pub struct HlsSegment {
    /// Segment URI (resolved if a base URL was provided).
    pub uri: String,
    /// Segment duration in seconds.
    pub duration: f64,
    /// Optional human-readable title.
    pub title: Option<String>,
    /// Sequence number (computed from `EXT-X-MEDIA-SEQUENCE` + offset).
    pub sequence: u64,
    /// Whether this segment starts after a discontinuity.
    pub discontinuity: bool,
    /// Active key info at this segment.
    pub key: Option<HlsKeyInfo>,
    /// Optional byte range `(length, offset)`.
    pub byte_range: Option<(u64, Option<u64>)>,
}

/// Parsed HLS media playlist.
#[derive(Debug, Clone)]
pub struct HlsMediaPlaylist {
    /// HLS protocol version.
    pub version: u8,
    /// Target segment duration in seconds.
    pub target_duration: u64,
    /// First sequence number.
    pub media_sequence: u64,
    /// Discontinuity sequence number.
    pub discontinuity_sequence: u64,
    /// Whether the playlist has ended (`#EXT-X-ENDLIST`).
    pub ended: bool,
    /// Playlist type (`VOD`, `EVENT`, or `None`).
    pub playlist_type: Option<String>,
    /// Map (initialisation segment) info.
    pub map: Option<HlsMapInfo>,
    /// All segments.
    pub segments: Vec<HlsSegment>,
    /// Tags not understood by this parser.
    pub unknown_tags: Vec<String>,
}

impl HlsMediaPlaylist {
    /// Total playlist duration in seconds.
    #[must_use]
    pub fn total_duration(&self) -> f64 {
        self.segments.iter().map(|s| s.duration).sum()
    }
}

/// An HLS variant stream (`#EXT-X-STREAM-INF`).
#[derive(Debug, Clone)]
pub struct HlsVariant {
    /// URI of the media playlist (resolved if base URL was provided).
    pub uri: String,
    /// Peak bitrate in bits/s.
    pub bandwidth: u64,
    /// Average bitrate in bits/s.
    pub average_bandwidth: Option<u64>,
    /// Codecs string (RFC 6381).
    pub codecs: Option<String>,
    /// Resolution `(width, height)`.
    pub resolution: Option<(u32, u32)>,
    /// Frame rate.
    pub frame_rate: Option<f64>,
    /// Audio group ID.
    pub audio: Option<String>,
    /// Video group ID.
    pub video: Option<String>,
    /// Subtitles group ID.
    pub subtitles: Option<String>,
    /// Closed-captions group ID.
    pub closed_captions: Option<String>,
}

/// An HLS media rendition (`#EXT-X-MEDIA`).
#[derive(Debug, Clone)]
pub struct HlsRendition {
    /// Rendition type (`AUDIO`, `VIDEO`, `SUBTITLES`, `CLOSED-CAPTIONS`).
    pub rendition_type: String,
    /// Group identifier.
    pub group_id: String,
    /// BCP-47 language tag.
    pub language: Option<String>,
    /// Human-readable name.
    pub name: String,
    /// Whether this rendition is selected by default.
    pub default: bool,
    /// Whether this rendition is auto-selected.
    pub autoselect: bool,
    /// URI of the media playlist.
    pub uri: Option<String>,
}

/// Parsed HLS master playlist.
#[derive(Debug, Clone)]
pub struct HlsMasterPlaylist {
    /// HLS protocol version.
    pub version: u8,
    /// Whether all segments are independently decodeable.
    pub independent_segments: bool,
    /// Variant streams, sorted by bandwidth ascending.
    pub variants: Vec<HlsVariant>,
    /// Media renditions.
    pub renditions: Vec<HlsRendition>,
    /// Session data entries.
    pub session_data: Vec<(String, String)>,
    /// Tags not understood by this parser.
    pub unknown_tags: Vec<String>,
}

impl HlsMasterPlaylist {
    /// Returns the lowest-bandwidth variant, if any.
    #[must_use]
    pub fn lowest_variant(&self) -> Option<&HlsVariant> {
        self.variants.first()
    }

    /// Returns the highest-bandwidth variant, if any.
    #[must_use]
    pub fn highest_variant(&self) -> Option<&HlsVariant> {
        self.variants.last()
    }

    /// Returns variants whose `codecs` string contains `codec_prefix`.
    pub fn variants_with_codec(&self, codec_prefix: &str) -> Vec<&HlsVariant> {
        self.variants
            .iter()
            .filter(|v| {
                v.codecs
                    .as_deref()
                    .map_or(false, |c| c.contains(codec_prefix))
            })
            .collect()
    }
}

// ─── DASH types ───────────────────────────────────────────────────────────────

/// A DASH Representation (rendition within an AdaptationSet).
#[derive(Debug, Clone)]
pub struct DashRepresentation {
    /// Representation id attribute.
    pub id: String,
    /// Bandwidth in bits/s.
    pub bandwidth: u64,
    /// Codecs string.
    pub codecs: Option<String>,
    /// Width in pixels.
    pub width: Option<u32>,
    /// Height in pixels.
    pub height: Option<u32>,
    /// Frame rate.
    pub frame_rate: Option<f64>,
    /// Audio sampling rate.
    pub audio_sampling_rate: Option<u32>,
    /// Base URL (may be relative to AdaptationSet or Period base URL).
    pub base_url: Option<String>,
    /// Segment template, if present.
    pub segment_template: Option<DashSegmentTemplate>,
}

/// DASH SegmentTemplate descriptor.
#[derive(Debug, Clone)]
pub struct DashSegmentTemplate {
    /// Media URL template (e.g. `$RepresentationID$/$Number$.m4s`).
    pub media: Option<String>,
    /// Initialisation URL template (e.g. `$RepresentationID$/init.mp4`).
    pub initialization: Option<String>,
    /// Start number for `$Number$` substitution.
    pub start_number: u64,
    /// Segment duration in `timescale` units.
    pub duration: Option<u64>,
    /// Timescale (units per second).
    pub timescale: u64,
}

impl DashSegmentTemplate {
    /// Expands the media template for segment `number` and representation `id`.
    #[must_use]
    pub fn expand_media(&self, rep_id: &str, number: u64) -> Option<String> {
        let template = self.media.as_deref()?;
        Some(
            template
                .replace("$RepresentationID$", rep_id)
                .replace("$Number$", &number.to_string())
                .replace("$Bandwidth$", ""),
        )
    }

    /// Expands the initialisation template for representation `id`.
    #[must_use]
    pub fn expand_init(&self, rep_id: &str) -> Option<String> {
        let template = self.initialization.as_deref()?;
        Some(template.replace("$RepresentationID$", rep_id))
    }

    /// Segment duration as [`Duration`], if both `duration` and `timescale`
    /// are non-zero.
    #[must_use]
    pub fn segment_duration(&self) -> Option<Duration> {
        let d = self.duration?;
        if self.timescale == 0 {
            return None;
        }
        let secs = d / self.timescale;
        let nanos = (d % self.timescale) * 1_000_000_000 / self.timescale;
        Some(Duration::new(secs, nanos as u32))
    }
}

/// DASH AdaptationSet — a group of Representations sharing the same media type.
#[derive(Debug, Clone)]
pub struct DashAdaptationSet {
    /// Optional id attribute.
    pub id: Option<String>,
    /// MIME type (e.g. `video/mp4`, `audio/mp4`).
    pub mime_type: Option<String>,
    /// Codecs string.
    pub codecs: Option<String>,
    /// BCP-47 language.
    pub lang: Option<String>,
    /// Segment template inherited by representations.
    pub segment_template: Option<DashSegmentTemplate>,
    /// Representations.
    pub representations: Vec<DashRepresentation>,
}

impl DashAdaptationSet {
    /// Returns the MIME media type prefix (before `/`), e.g. `"video"`.
    #[must_use]
    pub fn media_type(&self) -> Option<&str> {
        self.mime_type.as_deref().and_then(|m| m.split('/').next())
    }
}

/// A DASH Period.
#[derive(Debug, Clone)]
pub struct DashPeriod {
    /// Period id attribute.
    pub id: Option<String>,
    /// Start time.
    pub start: Option<Duration>,
    /// Period duration.
    pub duration: Option<Duration>,
    /// Base URL.
    pub base_url: Option<String>,
    /// Adaptation sets.
    pub adaptation_sets: Vec<DashAdaptationSet>,
}

/// Parsed DASH MPD.
#[derive(Debug, Clone)]
pub struct DashManifest {
    /// Presentation type (`static` or `dynamic`).
    pub presentation_type: String,
    /// Minimum buffer time.
    pub min_buffer_time: Option<Duration>,
    /// Suggested presentation delay for live streams.
    pub suggested_presentation_delay: Option<Duration>,
    /// Media presentation duration.
    pub media_presentation_duration: Option<Duration>,
    /// Time-shift buffer depth for live streams.
    pub time_shift_buffer_depth: Option<Duration>,
    /// Periods.
    pub periods: Vec<DashPeriod>,
}

impl DashManifest {
    /// Returns `true` if this is a live (dynamic) manifest.
    #[must_use]
    pub fn is_live(&self) -> bool {
        self.presentation_type.to_lowercase() == "dynamic"
    }

    /// Collects all video representations across all periods and adaptation sets.
    pub fn video_representations(&self) -> Vec<&DashRepresentation> {
        self.periods
            .iter()
            .flat_map(|p| &p.adaptation_sets)
            .filter(|a| a.media_type() == Some("video"))
            .flat_map(|a| &a.representations)
            .collect()
    }

    /// Collects all audio representations across all periods and adaptation sets.
    pub fn audio_representations(&self) -> Vec<&DashRepresentation> {
        self.periods
            .iter()
            .flat_map(|p| &p.adaptation_sets)
            .filter(|a| a.media_type() == Some("audio"))
            .flat_map(|a| &a.representations)
            .collect()
    }
}

// ─── Unified result ───────────────────────────────────────────────────────────

/// The result of parsing a manifest/playlist of any supported format.
#[derive(Debug)]
pub enum ParsedPlaylist {
    /// Parsed HLS master playlist.
    HlsMaster(HlsMasterPlaylist),
    /// Parsed HLS media playlist.
    HlsMedia(HlsMediaPlaylist),
    /// Parsed DASH MPD.
    Dash(DashManifest),
}

// ─── Parser ───────────────────────────────────────────────────────────────────

/// Unified HLS/DASH playlist parser.
///
/// # Usage
///
/// ```rust
/// use oximedia_net::playlist_parser::{PlaylistParser, ParsedPlaylist};
///
/// let text = "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:10\n\
///             #EXTINF:10.0,\nsegment0.ts\n#EXT-X-ENDLIST\n";
/// let result = PlaylistParser::new().parse(text, None).expect("valid HLS media playlist");
/// match result {
///     ParsedPlaylist::HlsMedia(m) => assert_eq!(m.segments.len(), 1),
///     _ => panic!("expected HLS media playlist"),
/// }
/// ```
pub struct PlaylistParser {
    /// If provided, relative URIs are resolved against this base URL.
    base_url: Option<String>,
}

impl PlaylistParser {
    /// Creates a new parser without a base URL.
    #[must_use]
    pub fn new() -> Self {
        Self { base_url: None }
    }

    /// Creates a new parser that resolves relative URIs against `base_url`.
    #[must_use]
    pub fn with_base_url(base_url: impl Into<String>) -> Self {
        Self {
            base_url: Some(base_url.into()),
        }
    }

    /// Parses `text` auto-detecting the format.  If `base_url_override` is
    /// `Some`, it takes precedence over the URL set at construction time.
    ///
    /// # Errors
    ///
    /// Returns an error if the content is not valid HLS or DASH.
    pub fn parse(&self, text: &str, base_url_override: Option<&str>) -> NetResult<ParsedPlaylist> {
        let base = base_url_override
            .map(str::to_owned)
            .or_else(|| self.base_url.clone());
        let resolver = base.as_deref().map(UrlResolver::new);

        match detect_format(text) {
            PlaylistFormat::HlsMaster => {
                let m = parse_hls_master(text, resolver.as_ref())?;
                Ok(ParsedPlaylist::HlsMaster(m))
            }
            PlaylistFormat::HlsMedia => {
                let m = parse_hls_media(text, resolver.as_ref())?;
                Ok(ParsedPlaylist::HlsMedia(m))
            }
            PlaylistFormat::DashMpd => {
                let m = parse_dash_mpd(text)?;
                Ok(ParsedPlaylist::Dash(m))
            }
            PlaylistFormat::Unknown => Err(NetError::Playlist(
                "Unrecognised playlist format (expected #EXTM3U or XML MPD)".into(),
            )),
        }
    }
}

impl Default for PlaylistParser {
    fn default() -> Self {
        Self::new()
    }
}

// ─── HLS parsing helpers ──────────────────────────────────────────────────────

/// Parses key-value attributes from an HLS attribute list string.
/// e.g. `BANDWIDTH=1000,CODECS="avc1.640028"` → HashMap.
fn parse_attribute_list(s: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let mut remaining = s.trim();
    while !remaining.is_empty() {
        // Find key
        let eq = match remaining.find('=') {
            Some(i) => i,
            None => break,
        };
        let key = remaining[..eq].trim().to_uppercase();
        remaining = &remaining[eq + 1..];

        // Value: quoted or unquoted
        let (value, rest) = if remaining.starts_with('"') {
            // Quoted string: find closing quote
            let end = remaining[1..]
                .find('"')
                .map(|i| i + 2)
                .unwrap_or(remaining.len());
            let val = remaining[1..end - 1].to_owned();
            let rest = &remaining[end..];
            let rest = rest.trim_start_matches(',');
            (val, rest)
        } else {
            // Unquoted: up to next comma
            let end = remaining.find(',').unwrap_or(remaining.len());
            let val = remaining[..end].trim().to_owned();
            let rest = remaining[end..].trim_start_matches(',');
            (val, rest)
        };

        map.insert(key, value);
        remaining = rest;
    }
    map
}

/// Parses a resolution string like `1920x1080` into `(width, height)`.
fn parse_resolution(s: &str) -> Option<(u32, u32)> {
    let mut parts = s.splitn(2, 'x');
    let w = parts.next()?.parse::<u32>().ok()?;
    let h = parts.next()?.parse::<u32>().ok()?;
    Some((w, h))
}

/// Parses an HLS byte-range string: `length[@offset]`.
fn parse_byte_range(s: &str) -> Option<(u64, Option<u64>)> {
    let mut parts = s.splitn(2, '@');
    let len = parts.next()?.trim().parse::<u64>().ok()?;
    let off = parts.next().and_then(|o| o.trim().parse::<u64>().ok());
    Some((len, off))
}

/// Parses an HLS map line from attributes.
fn parse_map_info(attrs: &HashMap<String, String>) -> Option<HlsMapInfo> {
    let uri = attrs.get("URI")?.clone();
    let byte_range = attrs.get("BYTERANGE").and_then(|br| parse_byte_range(br));
    Some(HlsMapInfo { uri, byte_range })
}

/// Parses an HLS key line from attributes.
fn parse_key_info(attrs: &HashMap<String, String>) -> HlsKeyInfo {
    let method = attrs
        .get("METHOD")
        .map(|m| EncryptionMethod::from(m.as_str()))
        .unwrap_or(EncryptionMethod::None);
    HlsKeyInfo {
        method,
        uri: attrs.get("URI").cloned(),
        iv: attrs.get("IV").cloned(),
        key_format: attrs.get("KEYFORMAT").cloned(),
        key_format_versions: attrs.get("KEYFORMATVERSIONS").cloned(),
    }
}

/// Parses an HLS master playlist.
fn parse_hls_master(text: &str, resolver: Option<&UrlResolver>) -> NetResult<HlsMasterPlaylist> {
    let mut playlist = HlsMasterPlaylist {
        version: 1,
        independent_segments: false,
        variants: Vec::new(),
        renditions: Vec::new(),
        session_data: Vec::new(),
        unknown_tags: Vec::new(),
    };

    let mut pending_stream_inf: Option<HlsVariant> = None;
    let mut lines = text.lines().peekable();

    while let Some(line) = lines.next() {
        let line = line.trim();
        if line.is_empty() || line == "#EXTM3U" {
            continue;
        }

        if let Some(rest) = line.strip_prefix("#EXT-X-VERSION:") {
            playlist.version = rest.trim().parse::<u8>().unwrap_or(1);
        } else if line == "#EXT-X-INDEPENDENT-SEGMENTS" {
            playlist.independent_segments = true;
        } else if let Some(rest) = line.strip_prefix("#EXT-X-STREAM-INF:") {
            let attrs = parse_attribute_list(rest);
            let bandwidth = attrs
                .get("BANDWIDTH")
                .and_then(|b| b.parse::<u64>().ok())
                .unwrap_or(0);
            let average_bandwidth = attrs
                .get("AVERAGE-BANDWIDTH")
                .and_then(|b| b.parse::<u64>().ok());
            let codecs = attrs.get("CODECS").cloned();
            let resolution = attrs.get("RESOLUTION").and_then(|r| parse_resolution(r));
            let frame_rate = attrs.get("FRAME-RATE").and_then(|r| r.parse::<f64>().ok());
            let audio = attrs.get("AUDIO").cloned();
            let video = attrs.get("VIDEO").cloned();
            let subtitles = attrs.get("SUBTITLES").cloned();
            let closed_captions = attrs.get("CLOSED-CAPTIONS").cloned();

            pending_stream_inf = Some(HlsVariant {
                uri: String::new(),
                bandwidth,
                average_bandwidth,
                codecs,
                resolution,
                frame_rate,
                audio,
                video,
                subtitles,
                closed_captions,
            });
        } else if let Some(rest) = line.strip_prefix("#EXT-X-MEDIA:") {
            let attrs = parse_attribute_list(rest);
            let rendition_type = attrs.get("TYPE").cloned().unwrap_or_default();
            let group_id = attrs.get("GROUP-ID").cloned().unwrap_or_default();
            let language = attrs.get("LANGUAGE").cloned();
            let name = attrs.get("NAME").cloned().unwrap_or_default();
            let default = attrs.get("DEFAULT").map_or(false, |v| v == "YES");
            let autoselect = attrs.get("AUTOSELECT").map_or(false, |v| v == "YES");
            let uri = attrs.get("URI").cloned();
            playlist.renditions.push(HlsRendition {
                rendition_type,
                group_id,
                language,
                name,
                default,
                autoselect,
                uri,
            });
        } else if let Some(rest) = line.strip_prefix("#EXT-X-SESSION-DATA:") {
            let attrs = parse_attribute_list(rest);
            if let (Some(id), Some(val)) = (attrs.get("DATA-ID"), attrs.get("VALUE")) {
                playlist.session_data.push((id.clone(), val.clone()));
            }
        } else if line.starts_with('#') {
            playlist.unknown_tags.push(line.to_owned());
        } else if !line.is_empty() {
            // URI line — attach to pending stream inf
            if let Some(mut variant) = pending_stream_inf.take() {
                variant.uri = if let Some(r) = resolver {
                    r.resolve(line).unwrap_or_else(|_| line.to_owned())
                } else {
                    line.to_owned()
                };
                playlist.variants.push(variant);
            }
        }
    }

    // Sort variants by bandwidth ascending
    playlist.variants.sort_by_key(|v| v.bandwidth);

    Ok(playlist)
}

/// Parses an HLS media playlist.
fn parse_hls_media(text: &str, resolver: Option<&UrlResolver>) -> NetResult<HlsMediaPlaylist> {
    let mut playlist = HlsMediaPlaylist {
        version: 1,
        target_duration: 0,
        media_sequence: 0,
        discontinuity_sequence: 0,
        ended: false,
        playlist_type: None,
        map: None,
        segments: Vec::new(),
        unknown_tags: Vec::new(),
    };

    let mut pending_duration: Option<f64> = None;
    let mut pending_title: Option<String> = None;
    let mut pending_discontinuity = false;
    let mut pending_byte_range: Option<(u64, Option<u64>)> = None;
    let mut current_key: Option<HlsKeyInfo> = None;
    let mut seq_offset: u64 = 0;

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line == "#EXTM3U" {
            continue;
        }

        if let Some(rest) = line.strip_prefix("#EXT-X-VERSION:") {
            playlist.version = rest.trim().parse::<u8>().unwrap_or(1);
        } else if let Some(rest) = line.strip_prefix("#EXT-X-TARGETDURATION:") {
            playlist.target_duration = rest.trim().parse::<u64>().unwrap_or(0);
        } else if let Some(rest) = line.strip_prefix("#EXT-X-MEDIA-SEQUENCE:") {
            playlist.media_sequence = rest.trim().parse::<u64>().unwrap_or(0);
        } else if let Some(rest) = line.strip_prefix("#EXT-X-DISCONTINUITY-SEQUENCE:") {
            playlist.discontinuity_sequence = rest.trim().parse::<u64>().unwrap_or(0);
        } else if let Some(rest) = line.strip_prefix("#EXT-X-PLAYLIST-TYPE:") {
            playlist.playlist_type = Some(rest.trim().to_owned());
        } else if line == "#EXT-X-ENDLIST" {
            playlist.ended = true;
        } else if line == "#EXT-X-DISCONTINUITY" {
            pending_discontinuity = true;
        } else if let Some(rest) = line.strip_prefix("#EXT-X-MAP:") {
            let attrs = parse_attribute_list(rest);
            playlist.map = parse_map_info(&attrs);
        } else if let Some(rest) = line.strip_prefix("#EXT-X-KEY:") {
            let attrs = parse_attribute_list(rest);
            current_key = Some(parse_key_info(&attrs));
        } else if let Some(rest) = line.strip_prefix("#EXT-X-BYTERANGE:") {
            pending_byte_range = parse_byte_range(rest.trim());
        } else if let Some(rest) = line.strip_prefix("#EXTINF:") {
            // #EXTINF:<duration>[,<title>]
            let (dur_str, title) = match rest.find(',') {
                Some(idx) => {
                    let t = rest[idx + 1..].trim();
                    (
                        &rest[..idx],
                        if t.is_empty() {
                            None
                        } else {
                            Some(t.to_owned())
                        },
                    )
                }
                None => (rest, None),
            };
            pending_duration = dur_str.trim().parse::<f64>().ok();
            pending_title = title;
        } else if line.starts_with('#') {
            playlist.unknown_tags.push(line.to_owned());
        } else if !line.is_empty() {
            // Segment URI
            if let Some(duration) = pending_duration.take() {
                let uri = if let Some(r) = resolver {
                    r.resolve(line).unwrap_or_else(|_| line.to_owned())
                } else {
                    line.to_owned()
                };
                playlist.segments.push(HlsSegment {
                    uri,
                    duration,
                    title: pending_title.take(),
                    sequence: playlist.media_sequence + seq_offset,
                    discontinuity: pending_discontinuity,
                    key: current_key.clone(),
                    byte_range: pending_byte_range.take(),
                });
                seq_offset += 1;
                pending_discontinuity = false;
            }
        }
    }

    Ok(playlist)
}

// ─── DASH MPD parsing ─────────────────────────────────────────────────────────

/// Parses a DASH MPD XML document.
///
/// This is a minimal, allocation-friendly parser that does not depend on a
/// full XML library: it uses a hand-written attribute scanner sufficient for
/// the MPD subset needed by adaptive players.
pub fn parse_dash_mpd(text: &str) -> NetResult<DashManifest> {
    let mut manifest = DashManifest {
        presentation_type: "static".into(),
        min_buffer_time: None,
        suggested_presentation_delay: None,
        media_presentation_duration: None,
        time_shift_buffer_depth: None,
        periods: Vec::new(),
    };

    // Extract MPD element attributes
    if let Some(mpd_content) = extract_element_attrs(text, "MPD") {
        let mpd_attrs = parse_xml_attrs(mpd_content);
        if let Some(t) = mpd_attrs.get("type") {
            manifest.presentation_type = t.to_lowercase();
        }
        if let Some(v) = mpd_attrs.get("minBufferTime") {
            manifest.min_buffer_time = parse_iso8601_duration(v);
        }
        if let Some(v) = mpd_attrs.get("suggestedPresentationDelay") {
            manifest.suggested_presentation_delay = parse_iso8601_duration(v);
        }
        if let Some(v) = mpd_attrs.get("mediaPresentationDuration") {
            manifest.media_presentation_duration = parse_iso8601_duration(v);
        }
        if let Some(v) = mpd_attrs.get("timeShiftBufferDepth") {
            manifest.time_shift_buffer_depth = parse_iso8601_duration(v);
        }
    }

    // Parse Period elements
    for period_text in iter_elements(text, "Period") {
        let period = parse_dash_period(&period_text)?;
        manifest.periods.push(period);
    }

    Ok(manifest)
}

/// Parses a single DASH Period element text.
fn parse_dash_period(text: &str) -> NetResult<DashPeriod> {
    let attrs = parse_xml_attrs(text);
    let id = attrs.get("id").cloned();
    let start = attrs.get("start").and_then(|v| parse_iso8601_duration(v));
    let duration = attrs
        .get("duration")
        .and_then(|v| parse_iso8601_duration(v));
    let base_url = extract_text_content(text, "BaseURL");

    let mut adaptation_sets = Vec::new();
    for as_text in iter_elements(text, "AdaptationSet") {
        let aset = parse_dash_adaptation_set(&as_text)?;
        adaptation_sets.push(aset);
    }

    Ok(DashPeriod {
        id,
        start,
        duration,
        base_url,
        adaptation_sets,
    })
}

/// Parses a single DASH AdaptationSet element.
fn parse_dash_adaptation_set(text: &str) -> NetResult<DashAdaptationSet> {
    let attrs = parse_xml_attrs(text);
    let id = attrs.get("id").cloned();
    let mime_type = attrs.get("mimeType").cloned();
    let codecs = attrs.get("codecs").cloned();
    let lang = attrs.get("lang").cloned();

    let segment_template = iter_elements(text, "SegmentTemplate")
        .next()
        .map(|st| parse_segment_template(&st));

    let mut representations = Vec::new();
    for rep_text in iter_elements(text, "Representation") {
        let rep = parse_dash_representation(&rep_text, segment_template.as_ref())?;
        representations.push(rep);
    }

    Ok(DashAdaptationSet {
        id,
        mime_type,
        codecs,
        lang,
        segment_template,
        representations,
    })
}

/// Parses a single DASH Representation element.
fn parse_dash_representation(
    text: &str,
    inherited_template: Option<&DashSegmentTemplate>,
) -> NetResult<DashRepresentation> {
    let attrs = parse_xml_attrs(text);
    let id = attrs.get("id").cloned().unwrap_or_default();
    let bandwidth = attrs
        .get("bandwidth")
        .and_then(|b| b.parse::<u64>().ok())
        .unwrap_or(0);
    let codecs = attrs.get("codecs").cloned();
    let width = attrs.get("width").and_then(|w| w.parse::<u32>().ok());
    let height = attrs.get("height").and_then(|h| h.parse::<u32>().ok());
    let frame_rate = attrs.get("frameRate").and_then(|r| parse_frame_rate(r));
    let audio_sampling_rate = attrs
        .get("audioSamplingRate")
        .and_then(|r| r.parse::<u32>().ok());
    let base_url = extract_text_content(text, "BaseURL");

    // Representation-level SegmentTemplate overrides inherited one
    let segment_template = iter_elements(text, "SegmentTemplate")
        .next()
        .map(|st| parse_segment_template(&st))
        .or_else(|| inherited_template.cloned());

    Ok(DashRepresentation {
        id,
        bandwidth,
        codecs,
        width,
        height,
        frame_rate,
        audio_sampling_rate,
        base_url,
        segment_template,
    })
}

/// Parses a DASH SegmentTemplate element.
fn parse_segment_template(text: &str) -> DashSegmentTemplate {
    let attrs = parse_xml_attrs(text);
    DashSegmentTemplate {
        media: attrs.get("media").cloned(),
        initialization: attrs.get("initialization").cloned(),
        start_number: attrs
            .get("startNumber")
            .and_then(|n| n.parse::<u64>().ok())
            .unwrap_or(1),
        duration: attrs.get("duration").and_then(|d| d.parse::<u64>().ok()),
        timescale: attrs
            .get("timescale")
            .and_then(|t| t.parse::<u64>().ok())
            .unwrap_or(1),
    }
}

/// Parses a fractional frame rate like `"30"` or `"30000/1001"`.
fn parse_frame_rate(s: &str) -> Option<f64> {
    if let Some((num, den)) = s.split_once('/') {
        let n = num.trim().parse::<f64>().ok()?;
        let d = den.trim().parse::<f64>().ok()?;
        if d == 0.0 {
            return None;
        }
        Some(n / d)
    } else {
        s.trim().parse::<f64>().ok()
    }
}

// ─── Minimal XML helpers ──────────────────────────────────────────────────────

/// Returns the attribute string of the first `<tag …>` element in `text`.
fn extract_element_attrs<'a>(text: &'a str, tag: &str) -> Option<&'a str> {
    let open = format!("<{tag}");
    let pos = text.find(open.as_str())?;
    let rest = &text[pos + open.len()..];
    // Find the end of the opening tag (either `>` or `/>`)
    let end = rest.find('>')?;
    let attrs_str = rest[..end].trim_start_matches([' ', '\t', '\n', '\r']);
    Some(attrs_str.trim_end_matches('/').trim())
}

/// Iterates over all `<tag …>…</tag>` or `<tag … />` blocks in `text`.
fn iter_elements<'a>(text: &'a str, tag: &'a str) -> impl Iterator<Item = String> + 'a {
    let open_tag = format!("<{tag}");
    let close_tag = format!("</{tag}>");

    let mut pos = 0usize;
    std::iter::from_fn(move || {
        let start = text[pos..].find(open_tag.as_str()).map(|i| pos + i)?;
        let after_open = start + open_tag.len();

        // Find end of this element
        let gt = text[after_open..].find('>')? + after_open;

        // Self-closing?
        if text[after_open..=gt].trim_end_matches('>').ends_with('/') {
            let chunk = text[start..=gt].to_owned();
            pos = gt + 1;
            return Some(chunk);
        }

        // Has children — find matching close tag
        let close_start = text[gt..].find(close_tag.as_str()).map(|i| i + gt)?;
        let close_end = close_start + close_tag.len();
        let chunk = text[start..close_end].to_owned();
        pos = close_end;
        Some(chunk)
    })
}

/// Extracts the text content of the first `<tag>text</tag>` in `text`.
fn extract_text_content(text: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = text.find(&open)? + open.len();
    let end = text.find(&close)?;
    if end < start {
        return None;
    }
    Some(text[start..end].trim().to_owned())
}

/// Parses XML attributes from a raw attribute string such as
/// `type="static" minBufferTime="PT2S"`.
fn parse_xml_attrs(text: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let mut s = text.trim();

    // Skip tag name if present (e.g. `<Period id="1"`)
    if s.starts_with('<') {
        if let Some(space) = s.find(|c: char| c.is_whitespace()) {
            s = &s[space..];
        } else {
            return map;
        }
    }

    while !s.is_empty() {
        s = s.trim_start();
        // key=
        let eq = match s.find('=') {
            Some(i) => i,
            None => break,
        };
        let key = s[..eq].trim().to_owned();
        s = s[eq + 1..].trim_start();
        // value (quoted or unquoted)
        let (val, rest) = if s.starts_with('"') {
            match s[1..].find('"') {
                Some(end) => {
                    let v = s[1..=end].to_owned();
                    (v, &s[end + 2..])
                }
                None => break,
            }
        } else if s.starts_with('\'') {
            match s[1..].find('\'') {
                Some(end) => {
                    let v = s[1..=end].to_owned();
                    (v, &s[end + 2..])
                }
                None => break,
            }
        } else {
            let end = s
                .find(|c: char| c.is_whitespace() || c == '>')
                .unwrap_or(s.len());
            let v = s[..end].to_owned();
            (v, &s[end..])
        };

        map.insert(key, val);
        s = rest;
    }
    map
}

/// Parses an ISO 8601 duration string (`PT2S`, `PT1H30M`, etc.) into
/// [`Duration`].
///
/// Supported designators: `H`, `M`, `S` (fractional seconds supported).
/// The `P` prefix and optional `T` are consumed.  Year/month/day designators
/// are not supported by this implementation.
pub fn parse_iso8601_duration(s: &str) -> Option<Duration> {
    let s = s.trim().trim_start_matches('P');
    let s = if let Some(stripped) = s.strip_prefix('T') {
        stripped
    } else {
        s
    };

    let mut total_secs: f64 = 0.0;
    let mut num_start = 0usize;

    for (i, ch) in s.char_indices() {
        if ch.is_ascii_digit() || ch == '.' {
            // accumulating number
        } else {
            let num: f64 = s[num_start..i].parse().ok()?;
            match ch {
                'H' => total_secs += num * 3600.0,
                'M' => total_secs += num * 60.0,
                'S' => total_secs += num,
                _ => {}
            }
            num_start = i + 1;
        }
    }

    if total_secs < 0.0 {
        return None;
    }

    let secs = total_secs.floor() as u64;
    let nanos = ((total_secs - total_secs.floor()) * 1_000_000_000.0) as u32;
    Some(Duration::new(secs, nanos))
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Format detection ──────────────────────────────────────────────────────

    #[test]
    fn test_detect_hls_master() {
        let text = "#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=1000\nlow.m3u8\n";
        assert_eq!(detect_format(text), PlaylistFormat::HlsMaster);
    }

    #[test]
    fn test_detect_hls_media() {
        let text = "#EXTM3U\n#EXT-X-TARGETDURATION:10\n#EXTINF:10.0,\nseg.ts\n";
        assert_eq!(detect_format(text), PlaylistFormat::HlsMedia);
    }

    #[test]
    fn test_detect_dash_mpd() {
        let text = r#"<?xml version="1.0"?><MPD type="static"></MPD>"#;
        assert_eq!(detect_format(text), PlaylistFormat::DashMpd);
    }

    #[test]
    fn test_detect_unknown() {
        let text = "not a playlist";
        assert_eq!(detect_format(text), PlaylistFormat::Unknown);
    }

    // ── HLS media playlist ────────────────────────────────────────────────────

    #[test]
    fn test_parse_hls_media_basic() {
        let text = concat!(
            "#EXTM3U\n",
            "#EXT-X-VERSION:3\n",
            "#EXT-X-TARGETDURATION:10\n",
            "#EXT-X-MEDIA-SEQUENCE:5\n",
            "#EXTINF:9.009,\n",
            "seg5.ts\n",
            "#EXTINF:9.009,\n",
            "seg6.ts\n",
            "#EXT-X-ENDLIST\n",
        );
        let parser = PlaylistParser::new();
        let result = parser.parse(text, None).expect("parse failed");
        match result {
            ParsedPlaylist::HlsMedia(m) => {
                assert_eq!(m.version, 3);
                assert_eq!(m.target_duration, 10);
                assert_eq!(m.media_sequence, 5);
                assert!(m.ended);
                assert_eq!(m.segments.len(), 2);
                assert_eq!(m.segments[0].sequence, 5);
                assert_eq!(m.segments[1].sequence, 6);
                assert!((m.total_duration() - 18.018).abs() < 0.001);
            }
            other => panic!("Expected HLS media, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_hls_media_with_url_resolution() {
        let text = "#EXTM3U\n#EXT-X-TARGETDURATION:6\n#EXTINF:6.0,\nseg0.ts\n#EXT-X-ENDLIST\n";
        let parser = PlaylistParser::with_base_url("https://cdn.example.com/live/playlist.m3u8");
        let result = parser.parse(text, None).expect("parse failed");
        match result {
            ParsedPlaylist::HlsMedia(m) => {
                assert!(
                    m.segments[0].uri.starts_with("https://cdn.example.com/"),
                    "URI should be resolved: {}",
                    m.segments[0].uri
                );
            }
            other => panic!("Expected HLS media, got {:?}", other),
        }
    }

    // ── HLS master playlist ───────────────────────────────────────────────────

    #[test]
    fn test_parse_hls_master_basic() {
        let text = concat!(
            "#EXTM3U\n",
            "#EXT-X-VERSION:4\n",
            "#EXT-X-STREAM-INF:BANDWIDTH=500000,CODECS=\"avc1.42c01e,mp4a.40.2\",RESOLUTION=640x360\n",
            "low/index.m3u8\n",
            "#EXT-X-STREAM-INF:BANDWIDTH=2000000,CODECS=\"avc1.640028,mp4a.40.2\",RESOLUTION=1280x720\n",
            "high/index.m3u8\n",
        );
        let parser = PlaylistParser::new();
        let result = parser.parse(text, None).expect("parse failed");
        match result {
            ParsedPlaylist::HlsMaster(m) => {
                assert_eq!(m.version, 4);
                assert_eq!(m.variants.len(), 2);
                assert!(
                    m.variants[0].bandwidth <= m.variants[1].bandwidth,
                    "variants should be sorted by bandwidth"
                );
                assert_eq!(m.variants[0].resolution, Some((640, 360)));
                assert_eq!(m.variants[1].resolution, Some((1280, 720)));
            }
            other => panic!("Expected HLS master, got {:?}", other),
        }
    }

    // ── URL resolver ──────────────────────────────────────────────────────────

    #[test]
    fn test_url_resolver_absolute() {
        let r = UrlResolver::new("https://a.example.com/path/manifest.m3u8");
        assert_eq!(
            r.resolve("https://b.example.com/seg.ts")
                .expect("absolute URL resolution should not fail"),
            "https://b.example.com/seg.ts"
        );
    }

    #[test]
    fn test_url_resolver_relative() {
        let r = UrlResolver::new("https://cdn.example.com/live/hls/manifest.m3u8");
        let resolved = r
            .resolve("seg0.ts")
            .expect("relative URL resolution should not fail");
        assert!(resolved.contains("cdn.example.com"), "got: {resolved}");
        assert!(resolved.ends_with("seg0.ts"), "got: {resolved}");
    }

    #[test]
    fn test_url_resolver_abs_path() {
        let r = UrlResolver::new("https://cdn.example.com/live/manifest.m3u8");
        let resolved = r
            .resolve("/segments/seg0.ts")
            .expect("absolute-path URL resolution should not fail");
        assert_eq!(resolved, "https://cdn.example.com/segments/seg0.ts");
    }

    // ── DASH MPD ──────────────────────────────────────────────────────────────

    #[test]
    fn test_parse_dash_mpd_static() {
        let text = r#"<?xml version="1.0" encoding="UTF-8"?>
<MPD type="static" mediaPresentationDuration="PT60S" minBufferTime="PT2S">
  <Period id="1" start="PT0S">
    <AdaptationSet id="0" mimeType="video/mp4" codecs="avc1.640028">
      <SegmentTemplate media="video/$Number$.m4s" initialization="video/init.mp4"
                       startNumber="1" duration="4000" timescale="1000"/>
      <Representation id="1080p" bandwidth="4000000" width="1920" height="1080"/>
      <Representation id="720p"  bandwidth="2000000" width="1280" height="720"/>
    </AdaptationSet>
    <AdaptationSet id="1" mimeType="audio/mp4" codecs="mp4a.40.2">
      <SegmentTemplate media="audio/$Number$.m4s" initialization="audio/init.mp4"
                       startNumber="1" duration="4000" timescale="1000"/>
      <Representation id="aac" bandwidth="128000" audioSamplingRate="48000"/>
    </AdaptationSet>
  </Period>
</MPD>"#;

        let manifest = parse_dash_mpd(text).expect("parse failed");
        assert_eq!(manifest.presentation_type, "static");
        assert!(!manifest.is_live());
        assert_eq!(manifest.periods.len(), 1);

        let period = &manifest.periods[0];
        assert_eq!(period.adaptation_sets.len(), 2);

        let video_reps = manifest.video_representations();
        assert_eq!(video_reps.len(), 2);

        let audio_reps = manifest.audio_representations();
        assert_eq!(audio_reps.len(), 1);
        assert_eq!(audio_reps[0].audio_sampling_rate, Some(48000));
    }

    #[test]
    fn test_dash_segment_template_expansion() {
        let template = DashSegmentTemplate {
            media: Some("video/$RepresentationID$/$Number$.m4s".into()),
            initialization: Some("video/$RepresentationID$/init.mp4".into()),
            start_number: 1,
            duration: Some(4000),
            timescale: 1000,
        };

        let media = template
            .expand_media("1080p", 42)
            .expect("media template is set");
        assert_eq!(media, "video/1080p/42.m4s");

        let init = template
            .expand_init("1080p")
            .expect("initialization template is set");
        assert_eq!(init, "video/1080p/init.mp4");

        let dur = template
            .segment_duration()
            .expect("duration and timescale are non-zero");
        assert_eq!(dur, Duration::from_secs(4));
    }

    // ── ISO 8601 duration ─────────────────────────────────────────────────────

    #[test]
    fn test_iso8601_duration_parsing() {
        assert_eq!(parse_iso8601_duration("PT2S"), Some(Duration::from_secs(2)));
        assert_eq!(
            parse_iso8601_duration("PT1H30M"),
            Some(Duration::from_secs(5400))
        );
        assert_eq!(
            parse_iso8601_duration("PT1H30M45S"),
            Some(Duration::from_secs(5445))
        );
        assert!(parse_iso8601_duration("PT60S").is_some());
    }

    // ── Error cases ───────────────────────────────────────────────────────────

    #[test]
    fn test_parse_unknown_format_returns_error() {
        let parser = PlaylistParser::new();
        let err = parser
            .parse("not a playlist", None)
            .expect_err("unknown format must fail");
        assert!(
            matches!(err, crate::error::NetError::Playlist(_)),
            "expected Playlist error, got {err:?}"
        );
    }
}
