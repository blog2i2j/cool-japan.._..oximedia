//! FFmpeg stream specifier parsing.
//!
//! Stream specifiers allow users to target specific streams within a file.
//! FFmpeg supports a rich syntax for stream specifiers:
//!
//! ```text
//! 0:v:0       — first video stream in file 0
//! 0:a:#0x1100 — audio stream with PID 0x1100 in file 0
//! 1:a:0       — first audio stream in file 1
//! v:0         — first video stream (any file)
//! a           — all audio streams
//! p:5:0       — stream 0 in program 5
//! ```
//!
//! Reference: <https://ffmpeg.org/ffmpeg.html#Stream-specifiers-1>

/// The media type part of a stream specifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamType {
    /// Video streams (`v`).
    Video,
    /// Audio streams (`a`).
    Audio,
    /// Subtitle streams (`s`).
    Subtitle,
    /// Data streams (`d`).
    Data,
    /// Attachment streams (`t`).
    Attachment,
    /// All stream types (no type discriminator specified).
    Any,
}

impl Default for StreamType {
    fn default() -> Self {
        Self::Any
    }
}

impl std::fmt::Display for StreamType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Video => write!(f, "v"),
            Self::Audio => write!(f, "a"),
            Self::Subtitle => write!(f, "s"),
            Self::Data => write!(f, "d"),
            Self::Attachment => write!(f, "t"),
            Self::Any => write!(f, "*"),
        }
    }
}

/// A stream index selector — either a zero-based position or a PID.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamIndex {
    /// Zero-based positional index within streams of the given type.
    Position(usize),
    /// Stream PID (used in `#0x1100` syntax in MPEG-TS contexts).
    Pid(u32),
    /// Select all streams of the given type (no index specified).
    All,
}

impl Default for StreamIndex {
    fn default() -> Self {
        Self::All
    }
}

impl std::fmt::Display for StreamIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Position(n) => write!(f, "{}", n),
            Self::Pid(pid) => write!(f, "#0x{:x}", pid),
            Self::All => Ok(()),
        }
    }
}

/// A fully parsed FFmpeg stream specifier.
///
/// Supports all common FFmpeg stream specifier forms:
///
/// | Input                 | `file_index` | `stream_type` | `stream_index` | `program_id` |
/// |-----------------------|--------------|---------------|----------------|-------------|
/// | `"v"`                 | `None`       | `Video`       | `All`          | `None`      |
/// | `"v:0"`               | `None`       | `Video`       | `Position(0)`  | `None`      |
/// | `"0:v:0"`             | `Some(0)`    | `Video`       | `Position(0)`  | `None`      |
/// | `"0:a:#0x1100"`       | `Some(0)`    | `Audio`       | `Pid(0x1100)`  | `None`      |
/// | `"p:5:0"`             | `None`       | `Any`         | `Position(0)`  | `Some(5)`   |
/// | `"2"`                 | `None`       | `Any`         | `Position(2)`  | `None`      |
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamSpec {
    /// File index prefix (from `0:v:0` style specifiers). `None` = any file.
    pub file_index: Option<usize>,
    /// The media type, if specified.
    pub stream_type: StreamType,
    /// Stream index within this type, or `All` if none specified.
    pub stream_index: StreamIndex,
    /// Program ID (from `p:<pid>:<stream_idx>` form), if specified.
    pub program_id: Option<u32>,
    /// Legacy: zero-based index (kept for backward compat, mirrors `stream_index` when Position).
    pub index: Option<usize>,
}

impl Default for StreamSpec {
    fn default() -> Self {
        Self {
            file_index: None,
            stream_type: StreamType::Any,
            stream_index: StreamIndex::All,
            program_id: None,
            index: None,
        }
    }
}

impl StreamSpec {
    /// Parse an FFmpeg stream specifier string into a [`StreamSpec`].
    ///
    /// Supported forms:
    /// - `"v"` — all video streams
    /// - `"v:0"` — first video stream
    /// - `"a:1"` — second audio stream
    /// - `"s"` — all subtitle streams
    /// - `"0"` — stream index 0 (type = Any)
    /// - `"0:v:0"` — first video stream in file 0
    /// - `"0:a:#0x1100"` — audio stream with PID 0x1100 in file 0
    /// - `"p:5:0"` — stream 0 in program 5
    pub fn parse(spec: &str) -> anyhow::Result<Self> {
        let spec = spec.trim();

        if spec.is_empty() {
            return Ok(Self::default());
        }

        // Handle program specifier: `p:<pid>:<stream_idx>`
        if let Some(rest) = spec.strip_prefix("p:") {
            let mut parts = rest.splitn(2, ':');
            let pid_str = parts.next().unwrap_or("0");
            let idx_str = parts.next().unwrap_or("");

            let program_id = pid_str
                .parse::<u32>()
                .map_err(|_| anyhow::anyhow!("invalid program id in specifier '{}'", spec))?;
            let (stream_index, index) = if idx_str.is_empty() {
                (StreamIndex::All, None)
            } else {
                let pos = idx_str
                    .parse::<usize>()
                    .map_err(|_| anyhow::anyhow!("invalid stream index in specifier '{}'", spec))?;
                (StreamIndex::Position(pos), Some(pos))
            };

            return Ok(Self {
                file_index: None,
                stream_type: StreamType::Any,
                stream_index,
                program_id: Some(program_id),
                index,
            });
        }

        // Try to detect a file-index prefix: `<num>:<type>:...`
        // The distinguishing heuristic: if the first segment is a pure integer,
        // AND the second segment starts with a known stream type letter (v,a,s,d,t),
        // treat the first segment as a file index.
        let parts: Vec<&str> = spec.splitn(3, ':').collect();

        if parts.len() >= 2 {
            // Check if first part is a numeric file index
            if let Ok(file_idx) = parts[0].parse::<usize>() {
                // Check if second part is a stream type letter
                if matches!(parts[1], "v" | "V" | "a" | "s" | "d" | "t") {
                    let stream_type = parse_stream_type_letter(parts[1]);
                    let stream_index = if parts.len() >= 3 {
                        parse_stream_index(parts[2])?
                    } else {
                        StreamIndex::All
                    };
                    let index = match &stream_index {
                        StreamIndex::Position(n) => Some(*n),
                        _ => None,
                    };
                    return Ok(Self {
                        file_index: Some(file_idx),
                        stream_type,
                        stream_index,
                        program_id: None,
                        index,
                    });
                }
            }
        }

        // Standard form: `[type][:index]`
        let mut split = spec.splitn(2, ':');
        let type_part = split.next().unwrap_or("");
        let index_part = split.next();

        let stream_type = match type_part {
            "v" | "V" => StreamType::Video,
            "a" => StreamType::Audio,
            "s" => StreamType::Subtitle,
            "d" => StreamType::Data,
            "t" => StreamType::Attachment,
            // Pure numeric → positional index with Any type
            n if n.parse::<usize>().is_ok() => {
                let pos = n.parse::<usize>().ok();
                let si = pos.map_or(StreamIndex::All, StreamIndex::Position);
                return Ok(Self {
                    file_index: None,
                    stream_type: StreamType::Any,
                    stream_index: si,
                    program_id: None,
                    index: pos,
                });
            }
            _ => {
                anyhow::bail!(
                    "unknown stream type '{}' in specifier '{}'",
                    type_part,
                    spec
                );
            }
        };

        let (stream_index, index) = match index_part {
            None | Some("") => (StreamIndex::All, None),
            Some(s) => {
                let si = parse_stream_index(s)?;
                let idx = match &si {
                    StreamIndex::Position(n) => Some(*n),
                    _ => None,
                };
                (si, idx)
            }
        };

        Ok(Self {
            file_index: None,
            stream_type,
            stream_index,
            program_id: None,
            index,
        })
    }

    /// Return `true` if this specifier targets only video streams.
    pub fn is_video(&self) -> bool {
        matches!(self.stream_type, StreamType::Video)
    }

    /// Return `true` if this specifier targets only audio streams.
    pub fn is_audio(&self) -> bool {
        matches!(self.stream_type, StreamType::Audio)
    }

    /// Return `true` if this specifier targets subtitle streams.
    pub fn is_subtitle(&self) -> bool {
        matches!(self.stream_type, StreamType::Subtitle)
    }
}

/// Parse a single stream-type letter into [`StreamType`].
fn parse_stream_type_letter(s: &str) -> StreamType {
    match s {
        "v" | "V" => StreamType::Video,
        "a" => StreamType::Audio,
        "s" => StreamType::Subtitle,
        "d" => StreamType::Data,
        "t" => StreamType::Attachment,
        _ => StreamType::Any,
    }
}

/// Parse a stream index sub-specifier: positional integer or `#0x<hex>` PID.
fn parse_stream_index(s: &str) -> anyhow::Result<StreamIndex> {
    let s = s.trim();
    if let Some(hex_str) = s.strip_prefix("#0x").or_else(|| s.strip_prefix("#0X")) {
        let pid = u32::from_str_radix(hex_str, 16)
            .map_err(|_| anyhow::anyhow!("invalid hex PID '{}' in stream specifier", s))?;
        return Ok(StreamIndex::Pid(pid));
    }
    if let Some(hex_str) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        let pid = u32::from_str_radix(hex_str, 16)
            .map_err(|_| anyhow::anyhow!("invalid hex PID '{}' in stream specifier", s))?;
        return Ok(StreamIndex::Pid(pid));
    }
    let pos = s
        .parse::<usize>()
        .map_err(|_| anyhow::anyhow!("invalid stream index '{}' in stream specifier", s))?;
    Ok(StreamIndex::Position(pos))
}

impl std::fmt::Display for StreamSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(pid) = self.program_id {
            write!(f, "p:{}", pid)?;
            if let StreamIndex::Position(n) = &self.stream_index {
                write!(f, ":{}", n)?;
            }
        } else {
            if let Some(fi) = self.file_index {
                write!(f, "{}:", fi)?;
            }
            write!(f, "{}", self.stream_type)?;
            match &self.stream_index {
                StreamIndex::All => {}
                si => write!(f, ":{}", si)?,
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_first() {
        let s = StreamSpec::parse("v:0").expect("parse");
        assert_eq!(s.stream_type, StreamType::Video);
        assert_eq!(s.stream_index, StreamIndex::Position(0));
        assert_eq!(s.index, Some(0));
    }

    #[test]
    fn test_audio_all() {
        let s = StreamSpec::parse("a").expect("parse");
        assert_eq!(s.stream_type, StreamType::Audio);
        assert_eq!(s.stream_index, StreamIndex::All);
    }

    #[test]
    fn test_numeric() {
        let s = StreamSpec::parse("2").expect("parse");
        assert_eq!(s.stream_type, StreamType::Any);
        assert_eq!(s.stream_index, StreamIndex::Position(2));
        assert_eq!(s.index, Some(2));
    }

    #[test]
    fn test_program() {
        let s = StreamSpec::parse("p:5:0").expect("parse");
        assert_eq!(s.program_id, Some(5));
        assert_eq!(s.index, Some(0));
    }

    #[test]
    fn test_file_index_video() {
        // 0:v:0 — first video stream in file 0
        let s = StreamSpec::parse("0:v:0").expect("parse 0:v:0");
        assert_eq!(s.file_index, Some(0));
        assert_eq!(s.stream_type, StreamType::Video);
        assert_eq!(s.stream_index, StreamIndex::Position(0));
    }

    #[test]
    fn test_file_index_audio_pid() {
        // 0:a:#0x1100 — audio PID 0x1100 in file 0
        let s = StreamSpec::parse("0:a:#0x1100").expect("parse 0:a:#0x1100");
        assert_eq!(s.file_index, Some(0));
        assert_eq!(s.stream_type, StreamType::Audio);
        assert_eq!(s.stream_index, StreamIndex::Pid(0x1100));
    }

    #[test]
    fn test_file_index_second_input_first_audio() {
        // 1:a:0 — first audio stream in file 1
        let s = StreamSpec::parse("1:a:0").expect("parse 1:a:0");
        assert_eq!(s.file_index, Some(1));
        assert_eq!(s.stream_type, StreamType::Audio);
        assert_eq!(s.stream_index, StreamIndex::Position(0));
    }

    #[test]
    fn test_subtitle_all() {
        let s = StreamSpec::parse("s").expect("parse");
        assert_eq!(s.stream_type, StreamType::Subtitle);
        assert_eq!(s.stream_index, StreamIndex::All);
    }

    #[test]
    fn test_empty_spec() {
        let s = StreamSpec::parse("").expect("parse empty");
        assert_eq!(s.stream_type, StreamType::Any);
        assert_eq!(s.stream_index, StreamIndex::All);
    }

    #[test]
    fn test_invalid_type_letter() {
        assert!(StreamSpec::parse("x").is_err());
    }

    #[test]
    fn test_display_video_first() {
        let s = StreamSpec {
            file_index: None,
            stream_type: StreamType::Video,
            stream_index: StreamIndex::Position(0),
            program_id: None,
            index: Some(0),
        };
        assert_eq!(s.to_string(), "v:0");
    }

    #[test]
    fn test_display_with_file_index() {
        let s = StreamSpec {
            file_index: Some(0),
            stream_type: StreamType::Video,
            stream_index: StreamIndex::Position(0),
            program_id: None,
            index: Some(0),
        };
        assert_eq!(s.to_string(), "0:v:0");
    }

    #[test]
    fn test_display_program() {
        let s = StreamSpec {
            file_index: None,
            stream_type: StreamType::Any,
            stream_index: StreamIndex::Position(0),
            program_id: Some(10),
            index: Some(0),
        };
        assert_eq!(s.to_string(), "p:10:0");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// StreamSelector — extended `-map` argument parsing
// ─────────────────────────────────────────────────────────────────────────────

/// Error type for stream-selector parsing.
#[derive(Debug, thiserror::Error)]
pub enum StreamSpecError {
    /// The stream-type letter was not recognized.
    #[error("unknown stream type letter '{0}'")]
    UnknownStreamType(String),

    /// A numeric field could not be parsed as an integer.
    #[error("invalid integer in stream specifier: {0}")]
    InvalidInteger(#[from] std::num::ParseIntError),

    /// The specifier string was syntactically malformed.
    #[error("malformed stream specifier '{0}': {1}")]
    Malformed(String, String),

    /// A metadata key=value pair was missing the `=` separator.
    #[error("metadata specifier missing '=': '{0}'")]
    MissingMetadataEquals(String),
}

/// High-level stream selector produced from a `-map` argument.
///
/// FFmpeg's `-map` syntax:
///
/// ```text
/// -map 0           — all streams from input 0
/// -map 0:v         — all video streams from input 0
/// -map 0:v:0       — first video stream from input 0
/// -map 0:a:1       — second audio stream from input 0
/// -map [label]     — output pad of a filter_complex
/// -map 0:m:language:eng — streams with metadata tag language=eng
/// -map -0:a:1      — exclude (negative map)
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamSelector {
    /// Select all streams (optionally filtered by type) from a given input file.
    ///
    /// `file_idx`: index into the `-i` input list.
    /// `stream_type`: `None` means all streams; `Some(t)` limits by media type.
    /// `stream_idx`: further positional refinement within `stream_type`.
    All {
        /// Input file index.
        file_idx: usize,
        /// Optional media-type filter.
        stream_type: Option<StreamType>,
        /// Optional positional index within the type group.
        stream_idx: Option<usize>,
    },

    /// Select streams by metadata key/value pair.
    ///
    /// Corresponds to `-map 0:m:key:value`.
    ByMetadata {
        /// Input file index.
        file_idx: usize,
        /// Optional media-type pre-filter.
        stream_type: Option<StreamType>,
        /// Metadata tag name.
        key: String,
        /// Metadata tag value.
        value: String,
    },

    /// Select a filter-complex output pad by its label.
    ///
    /// Corresponds to `-map [label]`.
    ByLabel {
        /// The pad label (the string between `[` and `]`).
        label: String,
    },

    /// Negative (exclusion) map — wraps another selector.
    ///
    /// Corresponds to the `-map -…` prefix.
    Exclude(Box<StreamSelector>),
}

impl StreamSelector {
    /// Parse a `-map` argument value into a [`StreamSelector`].
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_compat_ffmpeg::stream_spec::StreamSelector;
    ///
    /// let sel = StreamSelector::parse("0:v:0").expect("ok");
    /// let sel_neg = StreamSelector::parse("-0:a:1").expect("ok");
    /// let sel_lbl = StreamSelector::parse("[out_v]").expect("ok");
    /// let sel_meta = StreamSelector::parse("0:m:language:eng").expect("ok");
    /// ```
    pub fn parse(s: &str) -> Result<Self, StreamSpecError> {
        let s = s.trim();

        // ── Negative prefix ────────────────────────────────────────────────
        if let Some(rest) = s.strip_prefix('-') {
            // The rest must not itself start with `-` (double negative is invalid).
            let inner = Self::parse_positive(rest)?;
            return Ok(Self::Exclude(Box::new(inner)));
        }

        Self::parse_positive(s)
    }

    /// Parse a positive (non-negated) selector.
    fn parse_positive(s: &str) -> Result<Self, StreamSpecError> {
        // ── Filter-complex label: [label] ──────────────────────────────────
        if s.starts_with('[') {
            if let Some(inner) = s.strip_prefix('[').and_then(|t| t.strip_suffix(']')) {
                if inner.is_empty() {
                    return Err(StreamSpecError::Malformed(
                        s.to_string(),
                        "empty label".to_string(),
                    ));
                }
                return Ok(Self::ByLabel {
                    label: inner.to_string(),
                });
            }
            return Err(StreamSpecError::Malformed(
                s.to_string(),
                "unclosed '['".to_string(),
            ));
        }

        // All remaining forms start with a file index.
        let parts: Vec<&str> = s.splitn(4, ':').collect();

        let file_idx: usize = parts[0]
            .parse()
            .map_err(|e: std::num::ParseIntError| StreamSpecError::InvalidInteger(e))?;

        if parts.len() == 1 {
            // "0" — all streams from file 0
            return Ok(Self::All {
                file_idx,
                stream_type: None,
                stream_idx: None,
            });
        }

        // Second segment: stream type letter or 'm' (metadata)
        match parts[1] {
            "m" => {
                // Metadata specifier: 0:m:key:value
                let key = parts.get(2).copied().unwrap_or("").to_string();
                let value = parts.get(3).copied().unwrap_or("").to_string();
                if key.is_empty() {
                    return Err(StreamSpecError::Malformed(
                        s.to_string(),
                        "metadata key is empty".to_string(),
                    ));
                }
                Ok(Self::ByMetadata {
                    file_idx,
                    stream_type: None,
                    key,
                    value,
                })
            }
            type_str => {
                // Check if it is a valid stream type
                let st = parse_stream_selector_type(type_str)
                    .ok_or_else(|| StreamSpecError::UnknownStreamType(type_str.to_string()))?;

                // Third segment: optional positional index
                let stream_idx = if let Some(idx_str) = parts.get(2) {
                    if idx_str.is_empty() {
                        None
                    } else {
                        let n: usize = idx_str.parse().map_err(|e: std::num::ParseIntError| {
                            StreamSpecError::InvalidInteger(e)
                        })?;
                        Some(n)
                    }
                } else {
                    None
                };

                Ok(Self::All {
                    file_idx,
                    stream_type: Some(st),
                    stream_idx,
                })
            }
        }
    }
}

/// Parse a stream-type letter into [`StreamType`], returning `None` for unknown letters.
fn parse_stream_selector_type(s: &str) -> Option<StreamType> {
    match s {
        "v" | "V" => Some(StreamType::Video),
        "a" => Some(StreamType::Audio),
        "s" => Some(StreamType::Subtitle),
        "d" => Some(StreamType::Data),
        "t" => Some(StreamType::Attachment),
        _ => None,
    }
}

impl std::fmt::Display for StreamSelector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::All {
                file_idx,
                stream_type,
                stream_idx,
            } => {
                write!(f, "{}", file_idx)?;
                if let Some(st) = stream_type {
                    write!(f, ":{}", st)?;
                }
                if let Some(idx) = stream_idx {
                    write!(f, ":{}", idx)?;
                }
                Ok(())
            }
            Self::ByMetadata {
                file_idx,
                stream_type,
                key,
                value,
            } => {
                write!(f, "{}:", file_idx)?;
                if let Some(st) = stream_type {
                    write!(f, "{}:", st)?;
                }
                write!(f, "m:{}:{}", key, value)
            }
            Self::ByLabel { label } => write!(f, "[{}]", label),
            Self::Exclude(inner) => write!(f, "-{}", inner),
        }
    }
}

#[cfg(test)]
mod stream_selector_tests {
    use super::*;

    #[test]
    fn test_all_streams_from_file() {
        let sel = StreamSelector::parse("0").expect("parse 0");
        assert_eq!(
            sel,
            StreamSelector::All {
                file_idx: 0,
                stream_type: None,
                stream_idx: None
            }
        );
    }

    #[test]
    fn test_all_video_streams() {
        let sel = StreamSelector::parse("0:v").expect("parse 0:v");
        assert_eq!(
            sel,
            StreamSelector::All {
                file_idx: 0,
                stream_type: Some(StreamType::Video),
                stream_idx: None
            }
        );
    }

    #[test]
    fn test_first_video_stream() {
        let sel = StreamSelector::parse("0:v:0").expect("parse 0:v:0");
        assert_eq!(
            sel,
            StreamSelector::All {
                file_idx: 0,
                stream_type: Some(StreamType::Video),
                stream_idx: Some(0)
            }
        );
    }

    #[test]
    fn test_second_audio_stream_from_file1() {
        let sel = StreamSelector::parse("1:a:1").expect("parse 1:a:1");
        assert_eq!(
            sel,
            StreamSelector::All {
                file_idx: 1,
                stream_type: Some(StreamType::Audio),
                stream_idx: Some(1)
            }
        );
    }

    #[test]
    fn test_negative_map() {
        let sel = StreamSelector::parse("-0:a:1").expect("parse -0:a:1");
        if let StreamSelector::Exclude(inner) = &sel {
            assert_eq!(
                **inner,
                StreamSelector::All {
                    file_idx: 0,
                    stream_type: Some(StreamType::Audio),
                    stream_idx: Some(1)
                }
            );
        } else {
            panic!("expected Exclude variant");
        }
    }

    #[test]
    fn test_filter_label() {
        let sel = StreamSelector::parse("[out_v]").expect("parse [out_v]");
        assert_eq!(
            sel,
            StreamSelector::ByLabel {
                label: "out_v".to_string()
            }
        );
    }

    #[test]
    fn test_metadata() {
        let sel = StreamSelector::parse("0:m:language:eng").expect("parse 0:m:language:eng");
        assert_eq!(
            sel,
            StreamSelector::ByMetadata {
                file_idx: 0,
                stream_type: None,
                key: "language".to_string(),
                value: "eng".to_string()
            }
        );
    }

    #[test]
    fn test_display_roundtrip() {
        for spec in &["0", "0:v", "0:v:0", "1:a:1", "[out_v]", "0:m:language:eng"] {
            let sel = StreamSelector::parse(spec).expect("parse");
            assert_eq!(&sel.to_string(), spec, "roundtrip failed for {}", spec);
        }
    }

    #[test]
    fn test_invalid_file_idx() {
        assert!(StreamSelector::parse("x:v").is_err());
    }

    #[test]
    fn test_invalid_type_letter() {
        assert!(StreamSelector::parse("0:z").is_err());
    }
}
