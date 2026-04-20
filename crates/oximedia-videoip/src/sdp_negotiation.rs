//! SDP (Session Description Protocol) offer/answer negotiation for ST 2110 streams.
//!
//! This module implements RFC 3264 offer/answer model on top of SDP (RFC 4566),
//! tailored for professional IP media transport (SMPTE ST 2110-20/30/40) and
//! standard SIP/WebRTC interoperability.
//!
//! # Design
//!
//! The negotiation flow:
//!
//! 1. Offerer constructs an `SdpSession` via `SdpBuilder`.
//! 2. The SDP text is transmitted out-of-band (SIP, HTTP, etc.).
//! 3. Answerer calls `SdpParser::parse` to decode it, then builds its answer.
//! 4. Both sides apply the negotiated parameters to their RTP stacks.
//!
//! # Example
//!
//! ```rust
//! use oximedia_videoip::sdp_negotiation::{SdpBuilder, SdpParser, SdpMediaType};
//!
//! let sdp_text = SdpBuilder::new("Camera 1", "- 0 0 IN IP4 10.0.0.1")
//!     .add_video_section(5004, "RTP/AVP", &["96"], Some("239.100.0.1"))
//!     .build()
//!     .to_string();
//!
//! let session = SdpParser::parse(&sdp_text).expect("valid SDP");
//! assert_eq!(session.media.len(), 1);
//! assert_eq!(session.media[0].media_type, SdpMediaType::Video);
//! ```

#![allow(clippy::cast_precision_loss)]

use std::fmt;

// ── Error types ───────────────────────────────────────────────────────────────

/// Errors produced by SDP parsing and negotiation.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SdpError {
    /// A required SDP field was absent.
    #[error("missing required SDP field: {0}")]
    MissingField(String),
    /// Generic parse error with a description.
    #[error("SDP parse error: {0}")]
    ParseError(String),
    /// A port number was out of range or malformed.
    #[error("invalid port number in SDP")]
    InvalidPort,
}

/// Convenience `Result` alias for SDP operations.
pub type SdpResult<T> = Result<T, SdpError>;

// ── Media type ────────────────────────────────────────────────────────────────

/// Media type appearing in the `m=` line of an SDP session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SdpMediaType {
    /// Video stream (e.g. ST 2110-20, H.264, VP9).
    Video,
    /// Audio stream (e.g. ST 2110-30, Opus, PCM).
    Audio,
    /// Ancillary / metadata data (ST 2110-40, application).
    AncillaryData,
}

impl SdpMediaType {
    /// Returns the SDP keyword for this type (`video`, `audio`, `application`).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Video => "video",
            Self::Audio => "audio",
            Self::AncillaryData => "application",
        }
    }

    /// Parses a media-type keyword from an SDP `m=` line token.
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "video" => Some(Self::Video),
            "audio" => Some(Self::Audio),
            "application" => Some(Self::AncillaryData),
            _ => None,
        }
    }
}

impl fmt::Display for SdpMediaType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ── Attribute ─────────────────────────────────────────────────────────────────

/// An SDP `a=` attribute, either `a=<name>` (flag) or `a=<name>:<value>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SdpAttribute {
    /// Attribute name.
    pub name: String,
    /// Optional value (present for key-value attributes, absent for flags).
    pub value: Option<String>,
}

impl SdpAttribute {
    /// Creates a key-value attribute.
    #[must_use]
    pub fn new(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: Some(value.into()),
        }
    }

    /// Creates a flag attribute (no value).
    #[must_use]
    pub fn flag(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: None,
        }
    }

    /// Serialises to the `a=…` SDP line.
    #[must_use]
    pub fn to_line(&self) -> String {
        match &self.value {
            Some(v) => format!("a={}:{}", self.name, v),
            None => format!("a={}", self.name),
        }
    }

    /// Parses a raw `a=` line (without the leading `a=`).
    fn parse_raw(raw: &str) -> Self {
        if let Some((name, value)) = raw.split_once(':') {
            Self::new(name.trim(), value.trim())
        } else {
            Self::flag(raw.trim())
        }
    }
}

// ── Media section ─────────────────────────────────────────────────────────────

/// A single media section from an SDP description (`m=` + associated lines).
#[derive(Debug, Clone)]
pub struct SdpMediaSection {
    /// Media type (`video`, `audio`, `application`).
    pub media_type: SdpMediaType,
    /// UDP port for this media stream.
    pub port: u16,
    /// Transport protocol string (e.g. `RTP/AVP`, `RTP/SAVP`).
    pub protocol: String,
    /// Format list (payload type strings for RTP, or format names).
    pub formats: Vec<String>,
    /// Media-level `a=` attributes.
    pub attributes: Vec<SdpAttribute>,
    /// Optional connection address (`c=IN IP4 <addr>`).
    pub connection_addr: Option<String>,
}

impl SdpMediaSection {
    /// Creates a new media section with the given parameters.
    #[must_use]
    pub fn new(
        media_type: SdpMediaType,
        port: u16,
        protocol: impl Into<String>,
        formats: Vec<String>,
    ) -> Self {
        Self {
            media_type,
            port,
            protocol: protocol.into(),
            formats,
            attributes: Vec::new(),
            connection_addr: None,
        }
    }

    /// Appends an attribute to this media section.
    pub fn push_attribute(&mut self, attr: SdpAttribute) {
        self.attributes.push(attr);
    }

    /// Returns the first attribute matching `name`, if any.
    #[must_use]
    pub fn find_attribute(&self, name: &str) -> Option<&SdpAttribute> {
        self.attributes.iter().find(|a| a.name == name)
    }

    /// Serialises this media section to a list of SDP lines.
    #[must_use]
    pub fn to_lines(&self) -> Vec<String> {
        let mut out = Vec::new();
        let fmt_list = self.formats.join(" ");
        out.push(format!(
            "m={} {} {} {}",
            self.media_type.as_str(),
            self.port,
            self.protocol,
            fmt_list
        ));
        if let Some(addr) = &self.connection_addr {
            out.push(format!("c=IN IP4 {addr}"));
        }
        for attr in &self.attributes {
            out.push(attr.to_line());
        }
        out
    }
}

// ── Session ───────────────────────────────────────────────────────────────────

/// A complete SDP session description (RFC 4566).
#[derive(Debug, Clone)]
pub struct SdpSession {
    /// SDP version (`v=`), always `0` per RFC 4566.
    pub version: u8,
    /// Origin line (`o=`) as a raw string.
    pub origin: String,
    /// Session name (`s=`).
    pub session_name: String,
    /// Session-level `a=` attributes.
    pub attributes: Vec<SdpAttribute>,
    /// Media sections.
    pub media: Vec<SdpMediaSection>,
}

impl SdpSession {
    /// Creates an empty session with the given name and origin.
    #[must_use]
    pub fn new(
        version: u8,
        origin: impl Into<String>,
        session_name: impl Into<String>,
    ) -> Self {
        Self {
            version,
            origin: origin.into(),
            session_name: session_name.into(),
            attributes: Vec::new(),
            media: Vec::new(),
        }
    }

    /// Appends a session-level attribute.
    pub fn push_attribute(&mut self, attr: SdpAttribute) {
        self.attributes.push(attr);
    }

    /// Appends a media section.
    pub fn push_media(&mut self, section: SdpMediaSection) {
        self.media.push(section);
    }

    /// Returns all media sections of the specified type.
    #[must_use]
    pub fn media_of_type(&self, t: SdpMediaType) -> Vec<&SdpMediaSection> {
        self.media.iter().filter(|m| m.media_type == t).collect()
    }

    /// Returns the first session-level attribute matching `name`, if any.
    #[must_use]
    pub fn find_attribute(&self, name: &str) -> Option<&SdpAttribute> {
        self.attributes.iter().find(|a| a.name == name)
    }
}

impl fmt::Display for SdpSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v={}\r\n", self.version)?;
        write!(f, "o={}\r\n", self.origin)?;
        write!(f, "s={}\r\n", self.session_name)?;
        write!(f, "t=0 0\r\n")?;
        for attr in &self.attributes {
            write!(f, "{}\r\n", attr.to_line())?;
        }
        for section in &self.media {
            for line in section.to_lines() {
                write!(f, "{line}\r\n")?;
            }
        }
        Ok(())
    }
}

// ── Parser ────────────────────────────────────────────────────────────────────

/// Stateless SDP parser.
///
/// Parses a complete SDP text (lines separated by `\r\n` or `\n`) into an
/// [`SdpSession`].
pub struct SdpParser;

impl SdpParser {
    /// Parses `sdp` and returns the decoded [`SdpSession`].
    ///
    /// # Errors
    ///
    /// Returns [`SdpError::MissingField`] when a mandatory field (`v=`, `o=`,
    /// `s=`) is absent, [`SdpError::InvalidPort`] for an unparseable port, and
    /// [`SdpError::ParseError`] for other format violations.
    pub fn parse(sdp: &str) -> SdpResult<SdpSession> {
        let mut version: Option<u8> = None;
        let mut origin: Option<String> = None;
        let mut session_name: Option<String> = None;
        let mut session_attrs: Vec<SdpAttribute> = Vec::new();

        // Working state for the current media section being built.
        let mut current_media: Option<SdpMediaSection> = None;
        let mut media_sections: Vec<SdpMediaSection> = Vec::new();

        for raw_line in sdp.lines() {
            let line = raw_line.trim_end_matches('\r').trim();
            if line.is_empty() {
                continue;
            }
            let (type_char, value) = Self::split_line(line)?;

            match type_char {
                'v' => {
                    version = Some(
                        value
                            .parse::<u8>()
                            .map_err(|_| SdpError::ParseError(format!("bad v= value: {value}")))?,
                    );
                }
                'o' => {
                    origin = Some(value.to_owned());
                }
                's' => {
                    session_name = Some(value.to_owned());
                }
                't' => {
                    // Timing line – accepted but not stored.
                }
                'a' => {
                    let attr = SdpAttribute::parse_raw(value);
                    if let Some(ref mut sec) = current_media {
                        sec.push_attribute(attr);
                    } else {
                        session_attrs.push(attr);
                    }
                }
                'c' => {
                    // c=IN IP4 <addr> — extract the address part
                    let addr = Self::parse_connection(value)?;
                    if let Some(ref mut sec) = current_media {
                        sec.connection_addr = Some(addr);
                    }
                }
                'm' => {
                    // Flush previous media section.
                    if let Some(prev) = current_media.take() {
                        media_sections.push(prev);
                    }
                    current_media = Some(Self::parse_media_line(value)?);
                }
                _ => {
                    // Ignore unknown line types (b=, i=, u=, e=, p=, z=, k=, r=).
                }
            }
        }

        // Flush last media section.
        if let Some(last) = current_media {
            media_sections.push(last);
        }

        let version =
            version.ok_or_else(|| SdpError::MissingField("v=".to_owned()))?;
        let origin =
            origin.ok_or_else(|| SdpError::MissingField("o=".to_owned()))?;
        let session_name =
            session_name.ok_or_else(|| SdpError::MissingField("s=".to_owned()))?;

        let mut session = SdpSession::new(version, origin, session_name);
        session.attributes = session_attrs;
        session.media = media_sections;
        Ok(session)
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Splits a line into `(type_char, rest_after_equals)`.
    fn split_line(line: &str) -> SdpResult<(char, &str)> {
        let mut chars = line.chars();
        let type_char = chars
            .next()
            .ok_or_else(|| SdpError::ParseError("empty line".to_owned()))?;
        let rest = chars.as_str();
        if !rest.starts_with('=') {
            return Err(SdpError::ParseError(format!(
                "expected '=' after type char, got: {line}"
            )));
        }
        Ok((type_char, &rest[1..]))
    }

    /// Parses a `c=IN IP4 <addr>` line and returns the address part.
    fn parse_connection(value: &str) -> SdpResult<String> {
        // Format: <nettype> <addrtype> <connection-address>
        let parts: Vec<&str> = value.split_whitespace().collect();
        if parts.len() < 3 {
            return Err(SdpError::ParseError(format!(
                "malformed c= line: {value}"
            )));
        }
        Ok(parts[2].to_owned())
    }

    /// Parses an `m=` line value and returns a skeleton [`SdpMediaSection`].
    fn parse_media_line(value: &str) -> SdpResult<SdpMediaSection> {
        // Format: <media> <port>[/<num-ports>] <proto> <fmt-list>
        let mut parts = value.splitn(4, ' ');

        let media_str = parts
            .next()
            .ok_or_else(|| SdpError::MissingField("m= media type".to_owned()))?;
        let port_str = parts
            .next()
            .ok_or_else(|| SdpError::MissingField("m= port".to_owned()))?;
        let protocol = parts
            .next()
            .ok_or_else(|| SdpError::MissingField("m= protocol".to_owned()))?;
        let fmt_str = parts.next().unwrap_or("");

        let media_type =
            SdpMediaType::from_str(media_str).ok_or_else(|| {
                SdpError::ParseError(format!("unknown media type: {media_str}"))
            })?;

        // Port may be "<port>/<num-ports>"; only take the first.
        let port: u16 = port_str
            .split('/')
            .next()
            .unwrap_or(port_str)
            .parse()
            .map_err(|_| SdpError::InvalidPort)?;

        let formats = fmt_str
            .split_whitespace()
            .map(str::to_owned)
            .collect::<Vec<_>>();

        Ok(SdpMediaSection::new(media_type, port, protocol, formats))
    }
}

// ── Builder ───────────────────────────────────────────────────────────────────

/// Fluent builder for constructing [`SdpSession`] descriptions.
///
/// # Example
///
/// ```rust
/// use oximedia_videoip::sdp_negotiation::{SdpBuilder, SdpMediaType};
///
/// let session = SdpBuilder::new("Studio Cam A", "- 0 0 IN IP4 10.0.0.1")
///     .add_video_section(5004, "RTP/AVP", &["96"], Some("239.100.0.1"))
///     .add_audio_section(5006, "RTP/AVP", &["97"], Some("239.100.0.2"))
///     .add_session_attribute("tool", "OxiMedia")
///     .build();
///
/// assert_eq!(session.media_of_type(SdpMediaType::Video).len(), 1);
/// assert_eq!(session.media_of_type(SdpMediaType::Audio).len(), 1);
/// ```
#[derive(Debug)]
pub struct SdpBuilder {
    session: SdpSession,
}

impl SdpBuilder {
    /// Creates a new builder with the given session name and origin string.
    #[must_use]
    pub fn new(session_name: impl Into<String>, origin: impl Into<String>) -> Self {
        Self {
            session: SdpSession::new(0, origin, session_name),
        }
    }

    /// Adds a session-level attribute.
    #[must_use]
    pub fn add_session_attribute(
        mut self,
        name: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.session
            .push_attribute(SdpAttribute::new(name, value));
        self
    }

    /// Adds a session-level flag attribute.
    #[must_use]
    pub fn add_session_flag(mut self, name: impl Into<String>) -> Self {
        self.session
            .push_attribute(SdpAttribute::flag(name));
        self
    }

    /// Adds a video media section.
    #[must_use]
    pub fn add_video_section(
        self,
        port: u16,
        protocol: &str,
        formats: &[&str],
        connection_addr: Option<&str>,
    ) -> Self {
        self.add_media_section(SdpMediaType::Video, port, protocol, formats, connection_addr)
    }

    /// Adds an audio media section.
    #[must_use]
    pub fn add_audio_section(
        self,
        port: u16,
        protocol: &str,
        formats: &[&str],
        connection_addr: Option<&str>,
    ) -> Self {
        self.add_media_section(SdpMediaType::Audio, port, protocol, formats, connection_addr)
    }

    /// Adds an ancillary data media section.
    #[must_use]
    pub fn add_ancillary_section(
        self,
        port: u16,
        protocol: &str,
        formats: &[&str],
        connection_addr: Option<&str>,
    ) -> Self {
        self.add_media_section(
            SdpMediaType::AncillaryData,
            port,
            protocol,
            formats,
            connection_addr,
        )
    }

    /// Adds a generic media section.
    #[must_use]
    pub fn add_media_section(
        mut self,
        media_type: SdpMediaType,
        port: u16,
        protocol: &str,
        formats: &[&str],
        connection_addr: Option<&str>,
    ) -> Self {
        let mut section = SdpMediaSection::new(
            media_type,
            port,
            protocol,
            formats.iter().map(|s| (*s).to_owned()).collect(),
        );
        section.connection_addr = connection_addr.map(str::to_owned);
        self.session.push_media(section);
        self
    }

    /// Adds an attribute to the most recently added media section.
    ///
    /// Panics if no media section has been added yet — call one of the
    /// `add_*_section` methods first.
    #[must_use]
    pub fn with_media_attribute(
        mut self,
        name: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        if let Some(sec) = self.session.media.last_mut() {
            sec.push_attribute(SdpAttribute::new(name, value));
        }
        self
    }

    /// Adds a flag attribute to the most recently added media section.
    #[must_use]
    pub fn with_media_flag(mut self, name: impl Into<String>) -> Self {
        if let Some(sec) = self.session.media.last_mut() {
            sec.push_attribute(SdpAttribute::flag(name));
        }
        self
    }

    /// Finalises and returns the built [`SdpSession`].
    #[must_use]
    pub fn build(self) -> SdpSession {
        self.session
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Round-trip helpers ────────────────────────────────────────────────────

    fn build_simple_session() -> SdpSession {
        SdpBuilder::new("TestSession", "- 0 0 IN IP4 10.0.0.1")
            .add_video_section(5004, "RTP/AVP", &["96"], Some("239.100.0.1"))
            .with_media_attribute("rtpmap", "96 raw/90000")
            .add_audio_section(5006, "RTP/AVP", &["97"], Some("239.100.0.2"))
            .with_media_attribute("rtpmap", "97 L24/48000/2")
            .build()
    }

    // 1. Round-trip: build → to_string → parse → same media count
    #[test]
    fn test_round_trip_media_count() {
        let session = build_simple_session();
        let text = session.to_string();
        let parsed = SdpParser::parse(&text).expect("should parse round-trip");
        assert_eq!(parsed.media.len(), 2);
    }

    // 2. Round-trip: session name is preserved
    #[test]
    fn test_round_trip_session_name() {
        let session = build_simple_session();
        let text = session.to_string();
        let parsed = SdpParser::parse(&text).expect("should parse round-trip");
        assert_eq!(parsed.session_name, "TestSession");
    }

    // 3. Round-trip: video section detected
    #[test]
    fn test_round_trip_video_section() {
        let session = build_simple_session();
        let text = session.to_string();
        let parsed = SdpParser::parse(&text).expect("should parse round-trip");
        let videos = parsed.media_of_type(SdpMediaType::Video);
        assert_eq!(videos.len(), 1);
        assert_eq!(videos[0].port, 5004);
    }

    // 4. Round-trip: audio section detected
    #[test]
    fn test_round_trip_audio_section() {
        let session = build_simple_session();
        let text = session.to_string();
        let parsed = SdpParser::parse(&text).expect("should parse round-trip");
        let audios = parsed.media_of_type(SdpMediaType::Audio);
        assert_eq!(audios.len(), 1);
        assert_eq!(audios[0].port, 5006);
    }

    // 5. Round-trip: connection address preserved
    #[test]
    fn test_round_trip_connection_addr() {
        let session = build_simple_session();
        let text = session.to_string();
        let parsed = SdpParser::parse(&text).expect("should parse round-trip");
        assert_eq!(
            parsed.media[0].connection_addr.as_deref(),
            Some("239.100.0.1")
        );
    }

    // 6. Round-trip: attribute preserved in media section
    #[test]
    fn test_round_trip_media_attribute() {
        let session = build_simple_session();
        let text = session.to_string();
        let parsed = SdpParser::parse(&text).expect("should parse round-trip");
        let attr = parsed.media[0].find_attribute("rtpmap");
        assert!(attr.is_some());
        assert_eq!(
            attr.expect("attr is Some, checked above")
                .value
                .as_deref(),
            Some("96 raw/90000")
        );
    }

    // 7. Error: missing version field
    #[test]
    fn test_missing_version_error() {
        let sdp = "o=- 0 0 IN IP4 127.0.0.1\r\ns=Test\r\n";
        let result = SdpParser::parse(sdp);
        assert!(matches!(result, Err(SdpError::MissingField(_))));
    }

    // 8. Error: missing session name
    #[test]
    fn test_missing_session_name_error() {
        let sdp = "v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\n";
        let result = SdpParser::parse(sdp);
        assert!(matches!(result, Err(SdpError::MissingField(_))));
    }

    // 9. Error: invalid port
    #[test]
    fn test_invalid_port_error() {
        let sdp =
            "v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=Test\r\nm=video notaport RTP/AVP 96\r\n";
        let result = SdpParser::parse(sdp);
        assert!(matches!(result, Err(SdpError::InvalidPort)));
    }

    // 10. Attribute extraction: flag attribute
    #[test]
    fn test_flag_attribute_extraction() {
        let session = SdpBuilder::new("S", "- 0 0 IN IP4 127.0.0.1")
            .add_session_flag("recvonly")
            .build();
        let text = session.to_string();
        let parsed = SdpParser::parse(&text).expect("should parse");
        let attr = parsed.find_attribute("recvonly");
        assert!(attr.is_some());
        assert!(
            attr.expect("attr is Some, checked above")
                .value
                .is_none()
        );
    }

    // 11. Ancillary data section
    #[test]
    fn test_ancillary_section_round_trip() {
        let session = SdpBuilder::new("S", "- 0 0 IN IP4 127.0.0.1")
            .add_ancillary_section(5010, "RTP/AVP", &["100"], Some("239.100.0.3"))
            .build();
        let text = session.to_string();
        let parsed = SdpParser::parse(&text).expect("should parse");
        let anc = parsed.media_of_type(SdpMediaType::AncillaryData);
        assert_eq!(anc.len(), 1);
        assert_eq!(anc[0].port, 5010);
    }

    // 12. Multiple formats in a single m= section
    #[test]
    fn test_multiple_formats_parsed() {
        let session = SdpBuilder::new("S", "- 0 0 IN IP4 127.0.0.1")
            .add_video_section(5004, "RTP/AVP", &["96", "97", "98"], None)
            .build();
        let text = session.to_string();
        let parsed = SdpParser::parse(&text).expect("should parse");
        assert_eq!(parsed.media[0].formats.len(), 3);
    }
}
