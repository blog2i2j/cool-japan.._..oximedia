//! FFmpeg concat protocol and concat demuxer syntax compatibility.
//!
//! Translates two FFmpeg input-concatenation idioms into structured
//! [`ConcatSpec`] records that OxiMedia can execute natively:
//!
//! 1. **concat protocol** — `concat:file1.ts|file2.ts|file3.ts`
//!    Works with transport streams and container-less formats.
//!
//! 2. **concat demuxer** — a text file with `file 'path'` entries,
//!    read via `-f concat -safe 0 -i list.txt`.
//!    Supports optional `duration`, `inpoint`, and `outpoint` per segment.
//!
//! ## Example
//!
//! ```rust
//! use oximedia_compat_ffmpeg::concat_compat::{parse_concat_protocol, parse_concat_demuxer};
//!
//! // Protocol syntax
//! let spec = parse_concat_protocol("concat:a.ts|b.ts|c.ts").unwrap();
//! assert_eq!(spec.segments.len(), 3);
//!
//! // Demuxer syntax
//! let demuxer_text = "file 'clip1.mp4'\nfile 'clip2.mp4'\n";
//! let spec = parse_concat_demuxer(demuxer_text).unwrap();
//! assert_eq!(spec.segments.len(), 2);
//! ```

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Error returned when a concat specification cannot be parsed.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ConcatError {
    /// The concat protocol prefix `concat:` is missing.
    #[error("missing 'concat:' prefix in protocol string: '{0}'")]
    MissingProtocolPrefix(String),

    /// The concat protocol string has no file paths after the prefix.
    #[error("concat protocol string is empty after 'concat:' prefix")]
    EmptyProtocolList,

    /// The concat demuxer text contains a malformed `file` directive.
    #[error("malformed 'file' directive on line {line}: '{content}'")]
    MalformedFileLine {
        /// 1-based line number.
        line: usize,
        /// The raw line content.
        content: String,
    },

    /// An `inpoint`/`outpoint`/`duration` value is not a valid timestamp.
    #[error("invalid timestamp value '{value}' on line {line}")]
    InvalidTimestamp {
        /// The raw value that could not be parsed.
        value: String,
        /// 1-based line number.
        line: usize,
    },

    /// The concat specification contains no segments.
    #[error("concat specification contains no segments")]
    NoSegments,
}

// ─────────────────────────────────────────────────────────────────────────────
// Timestamp parsing
// ─────────────────────────────────────────────────────────────────────────────

/// A timestamp parsed from FFmpeg's time representation.
///
/// Supports:
/// - Seconds as a plain float: `12.5`
/// - `HH:MM:SS.ms` format: `01:23:45.678`
/// - `HH:MM:SS`: `01:23:45`
#[derive(Debug, Clone, PartialEq)]
pub struct Timestamp {
    /// The total duration in seconds (sub-second precision as fractional part).
    pub seconds: f64,
    /// The original string representation for round-trip fidelity.
    pub raw: String,
}

impl Timestamp {
    /// Parse an FFmpeg timestamp string into a [`Timestamp`].
    ///
    /// Returns `None` if the string is not recognisable as a timestamp.
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim();

        // Try `HH:MM:SS.ms` or `HH:MM:SS`
        let parts: Vec<&str> = s.splitn(3, ':').collect();
        if parts.len() == 3 {
            let hours = parts[0].parse::<f64>().ok()?;
            let minutes = parts[1].parse::<f64>().ok()?;
            let secs = parts[2].parse::<f64>().ok()?;
            let total = hours * 3600.0 + minutes * 60.0 + secs;
            return Some(Self {
                seconds: total,
                raw: s.to_string(),
            });
        }

        // Try `MM:SS.ms` or `MM:SS`
        if parts.len() == 2 {
            let minutes = parts[0].parse::<f64>().ok()?;
            let secs = parts[1].parse::<f64>().ok()?;
            let total = minutes * 60.0 + secs;
            return Some(Self {
                seconds: total,
                raw: s.to_string(),
            });
        }

        // Try plain seconds (float or int)
        let total = s.parse::<f64>().ok()?;
        Some(Self {
            seconds: total,
            raw: s.to_string(),
        })
    }
}

impl std::fmt::Display for Timestamp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.raw)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Segment
// ─────────────────────────────────────────────────────────────────────────────

/// A single segment in a concat specification.
#[derive(Debug, Clone)]
pub struct ConcatSegment {
    /// Path or URL to the input file.
    pub path: String,
    /// Optional inpoint (start offset within the file).
    pub inpoint: Option<Timestamp>,
    /// Optional outpoint (end offset within the file).
    pub outpoint: Option<Timestamp>,
    /// Optional explicit duration override.
    pub duration: Option<Timestamp>,
}

impl ConcatSegment {
    /// Create a simple segment with only a path.
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            inpoint: None,
            outpoint: None,
            duration: None,
        }
    }

    /// Return the effective duration of this segment, if computable.
    ///
    /// If `duration` is set, it is returned directly.
    /// If both `inpoint` and `outpoint` are set, their difference is returned.
    /// Otherwise `None`.
    pub fn effective_duration(&self) -> Option<f64> {
        if let Some(d) = &self.duration {
            return Some(d.seconds);
        }
        if let (Some(i), Some(o)) = (&self.inpoint, &self.outpoint) {
            let diff = o.seconds - i.seconds;
            if diff > 0.0 {
                return Some(diff);
            }
        }
        None
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ConcatSpec
// ─────────────────────────────────────────────────────────────────────────────

/// The source of a concat specification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConcatSource {
    /// Parsed from the `concat:file1|file2|...` protocol string.
    Protocol,
    /// Parsed from a concat demuxer text file.
    DemuxerFile,
}

/// A fully parsed concat specification ready for OxiMedia to execute.
#[derive(Debug, Clone)]
pub struct ConcatSpec {
    /// The ordered list of segments to concatenate.
    pub segments: Vec<ConcatSegment>,
    /// Whether this came from the protocol or the demuxer.
    pub source: ConcatSource,
    /// Total number of segments.
    pub segment_count: usize,
    /// Estimated total duration (sum of effective segment durations).
    /// Will be `None` if any segment lacks duration information.
    pub total_duration: Option<f64>,
}

impl ConcatSpec {
    fn build(segments: Vec<ConcatSegment>, source: ConcatSource) -> Self {
        let segment_count = segments.len();

        // Compute total duration only if all segments have effective durations.
        let total_duration = if segments.iter().all(|s| s.effective_duration().is_some()) {
            Some(segments.iter().filter_map(|s| s.effective_duration()).sum())
        } else {
            None
        };

        Self {
            segments,
            source,
            segment_count,
            total_duration,
        }
    }

    /// Return all segment paths in order.
    pub fn paths(&self) -> Vec<&str> {
        self.segments.iter().map(|s| s.path.as_str()).collect()
    }

    /// Return `true` if any segment has an inpoint or outpoint set.
    pub fn has_trim_points(&self) -> bool {
        self.segments
            .iter()
            .any(|s| s.inpoint.is_some() || s.outpoint.is_some())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Protocol parser
// ─────────────────────────────────────────────────────────────────────────────

/// Parse an FFmpeg concat protocol string (`concat:file1|file2|file3`) into a
/// [`ConcatSpec`].
///
/// The protocol string must begin with `concat:`. File paths are separated by
/// `|`. Leading/trailing whitespace around individual paths is trimmed.
///
/// # Errors
///
/// Returns [`ConcatError::MissingProtocolPrefix`] if the prefix is absent,
/// or [`ConcatError::EmptyProtocolList`] if no paths follow the prefix.
pub fn parse_concat_protocol(input: &str) -> Result<ConcatSpec, ConcatError> {
    let lower = input.trim();
    if !lower.to_lowercase().starts_with("concat:") {
        return Err(ConcatError::MissingProtocolPrefix(input.to_string()));
    }

    // Strip the "concat:" prefix (case-insensitive).
    let rest = &input[input
        .to_lowercase()
        .find("concat:")
        .map_or(0, |p| p + 7)..];

    if rest.trim().is_empty() {
        return Err(ConcatError::EmptyProtocolList);
    }

    let segments: Vec<ConcatSegment> = rest
        .split('|')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|path| ConcatSegment::new(path))
        .collect();

    if segments.is_empty() {
        return Err(ConcatError::EmptyProtocolList);
    }

    Ok(ConcatSpec::build(segments, ConcatSource::Protocol))
}

// ─────────────────────────────────────────────────────────────────────────────
// Demuxer parser
// ─────────────────────────────────────────────────────────────────────────────

/// Parse an FFmpeg concat demuxer text (the contents of a `list.txt` file) into
/// a [`ConcatSpec`].
///
/// Supported directives:
/// - `file 'path/to/file'` or `file path/to/file` — required to start a segment.
/// - `duration SECONDS` — explicit duration for the preceding segment.
/// - `inpoint TIMESTAMP` — start offset.
/// - `outpoint TIMESTAMP` — end offset.
/// - Lines beginning with `#` are comments.
/// - Empty lines are ignored.
///
/// # Errors
///
/// Returns [`ConcatError::MalformedFileLine`] for unrecognised directives, or
/// [`ConcatError::NoSegments`] if the text contains no `file` directives.
pub fn parse_concat_demuxer(text: &str) -> Result<ConcatSpec, ConcatError> {
    let mut segments: Vec<ConcatSegment> = Vec::new();

    for (line_idx, line) in text.lines().enumerate() {
        let line_num = line_idx + 1;
        let trimmed = line.trim();

        // Skip empty lines and comments.
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let lower = trimmed.to_lowercase();

        if lower.starts_with("file ") || lower == "file" {
            let path = parse_file_path(trimmed, line_num)?;
            segments.push(ConcatSegment::new(path));
        } else if lower.starts_with("duration ") {
            let value = &trimmed["duration ".len()..].trim().to_string();
            let ts = Timestamp::parse(value).ok_or_else(|| ConcatError::InvalidTimestamp {
                value: value.clone(),
                line: line_num,
            })?;
            if let Some(seg) = segments.last_mut() {
                seg.duration = Some(ts);
            }
        } else if lower.starts_with("inpoint ") {
            let value = &trimmed["inpoint ".len()..].trim().to_string();
            let ts = Timestamp::parse(value).ok_or_else(|| ConcatError::InvalidTimestamp {
                value: value.clone(),
                line: line_num,
            })?;
            if let Some(seg) = segments.last_mut() {
                seg.inpoint = Some(ts);
            }
        } else if lower.starts_with("outpoint ") {
            let value = &trimmed["outpoint ".len()..].trim().to_string();
            let ts = Timestamp::parse(value).ok_or_else(|| ConcatError::InvalidTimestamp {
                value: value.clone(),
                line: line_num,
            })?;
            if let Some(seg) = segments.last_mut() {
                seg.outpoint = Some(ts);
            }
        } else if lower.starts_with("ffconcat ") || lower.starts_with("ffconcat\t") {
            // Header line — ignore (contains version info).
        } else {
            return Err(ConcatError::MalformedFileLine {
                line: line_num,
                content: trimmed.to_string(),
            });
        }
    }

    if segments.is_empty() {
        return Err(ConcatError::NoSegments);
    }

    Ok(ConcatSpec::build(segments, ConcatSource::DemuxerFile))
}

/// Extract the file path from a `file 'path'` or `file path` directive line.
fn parse_file_path(line: &str, line_num: usize) -> Result<String, ConcatError> {
    // Strip directive keyword.
    let rest = if line.len() > 5 { &line[5..] } else { "" }; // "file ".len() == 5
    let rest = rest.trim();

    if rest.is_empty() {
        return Err(ConcatError::MalformedFileLine {
            line: line_num,
            content: line.to_string(),
        });
    }

    // Handle quoted paths: single or double quotes.
    let path = if (rest.starts_with('\'') && rest.ends_with('\''))
        || (rest.starts_with('"') && rest.ends_with('"'))
    {
        rest[1..rest.len() - 1].to_string()
    } else {
        // Unquoted: strip inline comments.
        rest.split('#').next().unwrap_or(rest).trim().to_string()
    };

    if path.is_empty() {
        return Err(ConcatError::MalformedFileLine {
            line: line_num,
            content: line.to_string(),
        });
    }

    Ok(path)
}

/// Determine whether a string looks like a concat protocol URL.
pub fn is_concat_protocol(s: &str) -> bool {
    s.trim().to_lowercase().starts_with("concat:")
}

/// Build an FFmpeg concat protocol string from a list of file paths.
///
/// The paths are joined with `|` after the `concat:` prefix.
pub fn build_concat_protocol(paths: &[&str]) -> String {
    format!("concat:{}", paths.join("|"))
}

/// Build a minimal ffconcat demuxer text from a list of file paths.
///
/// The resulting text uses single-quoted paths and includes the standard
/// `ffconcat version 1.0` header.
pub fn build_concat_demuxer_text(paths: &[&str]) -> String {
    let mut out = String::from("ffconcat version 1.0\n");
    for path in paths {
        out.push_str(&format!("file '{}'\n", path));
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── concat protocol ──────────────────────────────────────────────────────

    #[test]
    fn test_parse_concat_protocol_basic() {
        let spec = parse_concat_protocol("concat:a.ts|b.ts|c.ts").expect("parse failed");
        assert_eq!(spec.segments.len(), 3);
        assert_eq!(spec.source, ConcatSource::Protocol);
        assert_eq!(spec.segment_count, 3);
        let paths = spec.paths();
        assert_eq!(paths, vec!["a.ts", "b.ts", "c.ts"]);
    }

    #[test]
    fn test_parse_concat_protocol_single_file() {
        let spec = parse_concat_protocol("concat:only.ts").expect("parse failed");
        assert_eq!(spec.segments.len(), 1);
        assert_eq!(spec.segments[0].path, "only.ts");
    }

    #[test]
    fn test_parse_concat_protocol_missing_prefix() {
        let err = parse_concat_protocol("a.ts|b.ts").unwrap_err();
        assert!(matches!(err, ConcatError::MissingProtocolPrefix(_)));
    }

    #[test]
    fn test_parse_concat_protocol_empty_after_prefix() {
        let err = parse_concat_protocol("concat:").unwrap_err();
        assert!(matches!(err, ConcatError::EmptyProtocolList));
    }

    #[test]
    fn test_parse_concat_protocol_spaces_trimmed() {
        let spec =
            parse_concat_protocol("concat: file1.ts | file2.ts ").expect("parse failed");
        assert_eq!(spec.segments[0].path, "file1.ts");
        assert_eq!(spec.segments[1].path, "file2.ts");
    }

    // ── concat demuxer ───────────────────────────────────────────────────────

    #[test]
    fn test_parse_concat_demuxer_simple() {
        let text = "file 'clip1.mp4'\nfile 'clip2.mp4'\n";
        let spec = parse_concat_demuxer(text).expect("parse failed");
        assert_eq!(spec.segments.len(), 2);
        assert_eq!(spec.source, ConcatSource::DemuxerFile);
        assert_eq!(spec.segments[0].path, "clip1.mp4");
        assert_eq!(spec.segments[1].path, "clip2.mp4");
    }

    #[test]
    fn test_parse_concat_demuxer_with_duration() {
        let text = "file 'a.mp4'\nduration 10.5\nfile 'b.mp4'\nduration 5.0\n";
        let spec = parse_concat_demuxer(text).expect("parse failed");
        assert_eq!(spec.segments.len(), 2);
        let dur0 = spec.segments[0].duration.as_ref().expect("duration a");
        assert!((dur0.seconds - 10.5).abs() < 0.001);
        let dur1 = spec.segments[1].duration.as_ref().expect("duration b");
        assert!((dur1.seconds - 5.0).abs() < 0.001);
        let total = spec.total_duration.expect("total duration");
        assert!((total - 15.5).abs() < 0.001);
    }

    #[test]
    fn test_parse_concat_demuxer_with_inpoint_outpoint() {
        let text = "file 'clip.mp4'\ninpoint 00:00:05.0\noutpoint 00:00:15.0\n";
        let spec = parse_concat_demuxer(text).expect("parse failed");
        let seg = &spec.segments[0];
        let inpoint = seg.inpoint.as_ref().expect("inpoint");
        let outpoint = seg.outpoint.as_ref().expect("outpoint");
        assert!((inpoint.seconds - 5.0).abs() < 0.001);
        assert!((outpoint.seconds - 15.0).abs() < 0.001);
        let eff = seg.effective_duration().expect("effective duration");
        assert!((eff - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_parse_concat_demuxer_comments_ignored() {
        let text = "# This is a comment\nfile 'a.mp4'\n# Another comment\nfile 'b.mp4'\n";
        let spec = parse_concat_demuxer(text).expect("parse failed");
        assert_eq!(spec.segments.len(), 2);
    }

    #[test]
    fn test_parse_concat_demuxer_empty_returns_error() {
        let err = parse_concat_demuxer("# only comments\n").unwrap_err();
        assert!(matches!(err, ConcatError::NoSegments));
    }

    #[test]
    fn test_parse_concat_demuxer_malformed_line() {
        let text = "file 'ok.mp4'\nunknown_directive foo\n";
        let err = parse_concat_demuxer(text).unwrap_err();
        assert!(matches!(
            err,
            ConcatError::MalformedFileLine { line: 2, .. }
        ));
    }

    #[test]
    fn test_parse_concat_demuxer_ffconcat_header_ignored() {
        let text = "ffconcat version 1.0\nfile 'a.mp4'\n";
        let spec = parse_concat_demuxer(text).expect("parse failed");
        assert_eq!(spec.segments.len(), 1);
    }

    // ── Timestamp parsing ────────────────────────────────────────────────────

    #[test]
    fn test_timestamp_parse_seconds() {
        let ts = Timestamp::parse("12.5").expect("parse failed");
        assert!((ts.seconds - 12.5).abs() < 0.001);
    }

    #[test]
    fn test_timestamp_parse_hhmmss() {
        let ts = Timestamp::parse("01:02:03").expect("parse failed");
        let expected = 1.0 * 3600.0 + 2.0 * 60.0 + 3.0;
        assert!((ts.seconds - expected).abs() < 0.001);
    }

    #[test]
    fn test_timestamp_parse_hhmmss_ms() {
        let ts = Timestamp::parse("00:00:30.5").expect("parse failed");
        assert!((ts.seconds - 30.5).abs() < 0.001);
    }

    #[test]
    fn test_timestamp_parse_invalid() {
        assert!(Timestamp::parse("not_a_time").is_none());
    }

    // ── Utility functions ─────────────────────────────────────────────────────

    #[test]
    fn test_is_concat_protocol() {
        assert!(is_concat_protocol("concat:a.ts|b.ts"));
        assert!(is_concat_protocol("CONCAT:a.ts|b.ts"));
        assert!(!is_concat_protocol("a.ts|b.ts"));
        assert!(!is_concat_protocol(""));
    }

    #[test]
    fn test_build_concat_protocol() {
        let s = build_concat_protocol(&["a.ts", "b.ts", "c.ts"]);
        assert_eq!(s, "concat:a.ts|b.ts|c.ts");
    }

    #[test]
    fn test_build_concat_demuxer_text() {
        let text = build_concat_demuxer_text(&["a.mp4", "b.mp4"]);
        assert!(text.contains("ffconcat version 1.0"));
        assert!(text.contains("file 'a.mp4'"));
        assert!(text.contains("file 'b.mp4'"));
    }

    #[test]
    fn test_concat_spec_has_trim_points() {
        let mut spec = parse_concat_protocol("concat:a.ts|b.ts").expect("parse failed");
        assert!(!spec.has_trim_points());
        spec.segments[0].inpoint = Some(Timestamp {
            seconds: 5.0,
            raw: "5.0".to_string(),
        });
        assert!(spec.has_trim_points());
    }

    #[test]
    fn test_demuxer_double_quoted_path() {
        let text = "file \"clip with spaces.mp4\"\n";
        let spec = parse_concat_demuxer(text).expect("parse failed");
        assert_eq!(spec.segments[0].path, "clip with spaces.mp4");
    }

    #[test]
    fn test_demuxer_unquoted_path() {
        let text = "file simple.mp4\n";
        let spec = parse_concat_demuxer(text).expect("parse failed");
        assert_eq!(spec.segments[0].path, "simple.mp4");
    }
}
