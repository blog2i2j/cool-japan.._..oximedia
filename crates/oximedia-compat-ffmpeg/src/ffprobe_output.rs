//! ffprobe-compatible output formatter.
//!
//! This module wraps the existing [`crate::ffprobe::ProbeOutput`] type and
//! provides a richer output-format enum — adding XML and human-readable
//! "default" formats on top of the JSON/CSV/Flat modes already implemented in
//! `ffprobe.rs`.
//!
//! ## Supported formats
//!
//! | [`FfprobeOutputFormat`] | ffprobe flag | Description |
//! |------------------------|--------------|-------------|
//! | `Json`                 | `-print_format json` | JSON object (default ffprobe output) |
//! | `Xml`                  | `-print_format xml` | XML document (manually generated) |
//! | `Csv`                  | `-print_format csv` | CSV rows |
//! | `Default`              | *(none / `-print_format flat`)* | Flat key=value pairs |
//!
//! ## Usage
//!
//! ```rust
//! use oximedia_compat_ffmpeg::ffprobe::{ProbeFormat, ProbeStream, ProbeOutput};
//! use oximedia_compat_ffmpeg::ffprobe_output::{FfprobeOutputFormat, format_probe_result};
//!
//! let stream = ProbeStream::new_video("av1", 3840, 2160, "16:9", 60.0);
//! let format = ProbeFormat::new("uhd.mkv", "matroska,webm", 500_000_000, 300.0);
//! let output = ProbeOutput { format: Some(format), streams: vec![stream] };
//!
//! let json = format_probe_result(&output, FfprobeOutputFormat::Json).unwrap();
//! assert!(json.contains("\"codec_name\""));
//!
//! let xml = format_probe_result(&output, FfprobeOutputFormat::Xml).unwrap();
//! assert!(xml.contains("<ffprobe "));
//! ```

use crate::ffprobe::{PrintFormat, ProbeOutput};

/// Error type for ffprobe output formatting.
#[derive(Debug, thiserror::Error)]
pub enum FfprobeOutputError {
    /// An unexpected internal error occurred during formatting.
    #[error("formatting error: {0}")]
    FormatError(String),
}

/// The output format for ffprobe-compatible results.
///
/// Extends [`crate::ffprobe::PrintFormat`] with `Xml` and renames `Flat`
/// to `Default` to match the ffprobe CLI naming.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FfprobeOutputFormat {
    /// JSON object output (`-print_format json`).
    #[default]
    Json,
    /// XML document output (`-print_format xml`).
    Xml,
    /// CSV row output (`-print_format csv`).
    Csv,
    /// Flat key=value pairs (the default/human-readable mode).
    Default,
}

impl FfprobeOutputFormat {
    /// Parse from a `-print_format` value string.
    ///
    /// Recognises: `"json"`, `"xml"`, `"csv"`, `"flat"`, `"default"` (case-insensitive).
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "json" => Some(Self::Json),
            "xml" => Some(Self::Xml),
            "csv" => Some(Self::Csv),
            "flat" | "default" => Some(Self::Default),
            _ => None,
        }
    }
}

impl std::fmt::Display for FfprobeOutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Json => write!(f, "json"),
            Self::Xml => write!(f, "xml"),
            Self::Csv => write!(f, "csv"),
            Self::Default => write!(f, "default"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main formatting function
// ─────────────────────────────────────────────────────────────────────────────

/// Format a [`ProbeOutput`] in the requested [`FfprobeOutputFormat`].
///
/// Returns the formatted string or a [`FfprobeOutputError`] on failure.
pub fn format_probe_result(
    output: &ProbeOutput,
    fmt: FfprobeOutputFormat,
) -> Result<String, FfprobeOutputError> {
    let result = match fmt {
        FfprobeOutputFormat::Json => output.to_print_format(PrintFormat::Json),
        FfprobeOutputFormat::Csv => output.to_print_format(PrintFormat::Csv),
        FfprobeOutputFormat::Default => output.to_print_format(PrintFormat::Flat),
        FfprobeOutputFormat::Xml => render_xml(output),
    };
    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
// XML renderer (hand-built; no external XML dependency needed)
// ─────────────────────────────────────────────────────────────────────────────

/// Render a [`ProbeOutput`] as an XML document compatible with
/// `ffprobe -print_format xml` output.
fn render_xml(output: &ProbeOutput) -> String {
    let mut buf = String::with_capacity(2048);

    buf.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    buf.push_str(
        "<ffprobe xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" \
         xsi:noNamespaceSchemaLocation=\"ffprobe.xsd\">\n",
    );

    // ── streams ──────────────────────────────────────────────────────────────
    if !output.streams.is_empty() {
        buf.push_str("    <streams>\n");
        for (i, stream) in output.streams.iter().enumerate() {
            let mut s = stream.clone();
            s.index = i;

            buf.push_str("        <stream");
            xml_attr(&mut buf, "index", &i.to_string());
            xml_attr(&mut buf, "codec_name", &s.codec_name);
            xml_attr(&mut buf, "codec_long_name", &s.codec_long_name);
            xml_attr(&mut buf, "codec_type", &s.codec_type.to_string());
            xml_attr(&mut buf, "codec_tag_string", &s.codec_tag_string);
            xml_attr(&mut buf, "codec_tag", &s.codec_tag);

            if let Some(v) = s.width {
                xml_attr(&mut buf, "width", &v.to_string());
            }
            if let Some(v) = s.height {
                xml_attr(&mut buf, "height", &v.to_string());
            }
            if let Some(v) = s.coded_width {
                xml_attr(&mut buf, "coded_width", &v.to_string());
            }
            if let Some(v) = s.coded_height {
                xml_attr(&mut buf, "coded_height", &v.to_string());
            }
            if let Some(ref v) = s.display_aspect_ratio {
                xml_attr(&mut buf, "display_aspect_ratio", v);
            }
            if let Some(ref v) = s.pix_fmt {
                xml_attr(&mut buf, "pix_fmt", v);
            }
            if let Some(ref v) = s.r_frame_rate {
                xml_attr(&mut buf, "r_frame_rate", v);
            }
            if let Some(ref v) = s.avg_frame_rate {
                xml_attr(&mut buf, "avg_frame_rate", v);
            }
            if let Some(v) = s.sample_rate {
                xml_attr(&mut buf, "sample_rate", &v.to_string());
            }
            if let Some(v) = s.channels {
                xml_attr(&mut buf, "channels", &v.to_string());
            }
            if let Some(ref v) = s.channel_layout {
                xml_attr(&mut buf, "channel_layout", v);
            }
            if let Some(ref v) = s.sample_fmt {
                xml_attr(&mut buf, "sample_fmt", v);
            }
            if let Some(v) = s.bit_rate {
                xml_attr(&mut buf, "bit_rate", &v.to_string());
            }
            if let Some(ref v) = s.time_base {
                xml_attr(&mut buf, "time_base", v);
            }
            if let Some(v) = s.duration_ts {
                xml_attr(&mut buf, "duration_ts", &v.to_string());
            }
            if let Some(ref v) = s.duration {
                xml_attr(&mut buf, "duration", v);
            }
            if let Some(ref v) = s.profile {
                xml_attr(&mut buf, "profile", v);
            }
            if let Some(v) = s.level {
                xml_attr(&mut buf, "level", &v.to_string());
            }

            if s.tags.is_empty() {
                buf.push_str("/>\n");
            } else {
                buf.push_str(">\n");
                buf.push_str("            <tags>\n");
                let mut tag_pairs: Vec<(&String, &String)> = s.tags.iter().collect();
                tag_pairs.sort_by_key(|(k, _)| k.as_str());
                for (k, v) in tag_pairs {
                    buf.push_str("                <tag");
                    xml_attr(&mut buf, "key", k);
                    xml_attr(&mut buf, "value", v);
                    buf.push_str("/>\n");
                }
                buf.push_str("            </tags>\n");
                buf.push_str("        </stream>\n");
            }
        }
        buf.push_str("    </streams>\n");
    }

    // ── format ───────────────────────────────────────────────────────────────
    if let Some(ref fmt) = output.format {
        buf.push_str("    <format");
        xml_attr(&mut buf, "filename", &fmt.filename);
        xml_attr(&mut buf, "nb_streams", &fmt.nb_streams.to_string());
        xml_attr(&mut buf, "nb_programs", &fmt.nb_programs.to_string());
        xml_attr(&mut buf, "format_name", &fmt.format_name);
        xml_attr(&mut buf, "format_long_name", &fmt.format_long_name);
        if let Some(v) = fmt.start_time {
            xml_attr(&mut buf, "start_time", &format!("{:.6}", v));
        }
        if let Some(v) = fmt.duration {
            xml_attr(&mut buf, "duration", &format!("{:.6}", v));
        }
        if let Some(v) = fmt.size {
            xml_attr(&mut buf, "size", &v.to_string());
        }
        if let Some(v) = fmt.bit_rate {
            xml_attr(&mut buf, "bit_rate", &v.to_string());
        }
        xml_attr(&mut buf, "probe_score", &fmt.probe_score.to_string());

        if fmt.tags.is_empty() {
            buf.push_str("/>\n");
        } else {
            buf.push_str(">\n");
            buf.push_str("        <tags>\n");
            let mut tag_pairs: Vec<(&String, &String)> = fmt.tags.iter().collect();
            tag_pairs.sort_by_key(|(k, _)| k.as_str());
            for (k, v) in tag_pairs {
                buf.push_str("            <tag");
                xml_attr(&mut buf, "key", k);
                xml_attr(&mut buf, "value", v);
                buf.push_str("/>\n");
            }
            buf.push_str("        </tags>\n");
            buf.push_str("    </format>\n");
        }
    }

    buf.push_str("</ffprobe>\n");
    buf
}

/// Append a single XML attribute to the buffer: ` key="value"`.
///
/// Values are XML-attribute-escaped.
fn xml_attr(buf: &mut String, key: &str, value: &str) {
    buf.push(' ');
    buf.push_str(key);
    buf.push_str("=\"");
    for ch in value.chars() {
        match ch {
            '&' => buf.push_str("&amp;"),
            '"' => buf.push_str("&quot;"),
            '<' => buf.push_str("&lt;"),
            '>' => buf.push_str("&gt;"),
            '\'' => buf.push_str("&apos;"),
            other => buf.push(other),
        }
    }
    buf.push('"');
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffprobe::{ProbeFormat, ProbeStream};

    fn make_output() -> ProbeOutput {
        let stream = ProbeStream::new_video("av1", 1920, 1080, "16:9", 24.0);
        let format = ProbeFormat::new("test.mkv", "matroska,webm", 10_000_000, 60.0);
        ProbeOutput {
            format: Some(format),
            streams: vec![stream],
        }
    }

    #[test]
    fn test_json_output_contains_codec() {
        let out = make_output();
        let json = format_probe_result(&out, FfprobeOutputFormat::Json).expect("json");
        assert!(
            json.contains("\"codec_name\""),
            "should contain codec_name key"
        );
        assert!(json.contains("\"av1\""), "should contain av1 codec");
    }

    #[test]
    fn test_xml_output_has_root() {
        let out = make_output();
        let xml = format_probe_result(&out, FfprobeOutputFormat::Xml).expect("xml");
        assert!(
            xml.starts_with("<?xml"),
            "should start with XML declaration"
        );
        assert!(xml.contains("<ffprobe"), "should have <ffprobe> root");
        assert!(xml.contains("</ffprobe>"), "should close <ffprobe>");
    }

    #[test]
    fn test_xml_contains_stream() {
        let out = make_output();
        let xml = format_probe_result(&out, FfprobeOutputFormat::Xml).expect("xml");
        assert!(xml.contains("<stream "), "should have <stream> element");
        assert!(
            xml.contains("codec_name=\"av1\""),
            "should have codec_name attr"
        );
    }

    #[test]
    fn test_xml_contains_format() {
        let out = make_output();
        let xml = format_probe_result(&out, FfprobeOutputFormat::Xml).expect("xml");
        assert!(xml.contains("<format "), "should have <format> element");
        assert!(xml.contains("format_name=\"matroska,webm\""));
    }

    #[test]
    fn test_csv_output() {
        let out = make_output();
        let csv = format_probe_result(&out, FfprobeOutputFormat::Csv).expect("csv");
        assert!(csv.starts_with("stream,"), "CSV should start with stream,");
    }

    #[test]
    fn test_default_output() {
        let out = make_output();
        let flat = format_probe_result(&out, FfprobeOutputFormat::Default).expect("default");
        assert!(
            flat.contains("codec_name"),
            "flat should contain codec_name"
        );
    }

    #[test]
    fn test_format_from_str() {
        assert_eq!(
            FfprobeOutputFormat::from_str("json"),
            Some(FfprobeOutputFormat::Json)
        );
        assert_eq!(
            FfprobeOutputFormat::from_str("xml"),
            Some(FfprobeOutputFormat::Xml)
        );
        assert_eq!(
            FfprobeOutputFormat::from_str("csv"),
            Some(FfprobeOutputFormat::Csv)
        );
        assert_eq!(
            FfprobeOutputFormat::from_str("flat"),
            Some(FfprobeOutputFormat::Default)
        );
        assert_eq!(
            FfprobeOutputFormat::from_str("default"),
            Some(FfprobeOutputFormat::Default)
        );
        assert_eq!(
            FfprobeOutputFormat::from_str("JSON"),
            Some(FfprobeOutputFormat::Json)
        );
        assert_eq!(FfprobeOutputFormat::from_str("unknown"), None);
    }

    #[test]
    fn test_xml_escaping() {
        let mut buf = String::new();
        xml_attr(&mut buf, "value", "a&b\"c<d>e");
        assert_eq!(buf, " value=\"a&amp;b&quot;c&lt;d&gt;e\"");
    }

    #[test]
    fn test_empty_streams() {
        let out = ProbeOutput {
            format: Some(ProbeFormat::new("empty.mp4", "mp4", 0, 0.0)),
            streams: vec![],
        };
        let xml = format_probe_result(&out, FfprobeOutputFormat::Xml).expect("xml");
        assert!(!xml.contains("<streams>"), "no streams section when empty");
        assert!(xml.contains("<format "), "format section still present");
    }

    #[test]
    fn test_display() {
        assert_eq!(FfprobeOutputFormat::Json.to_string(), "json");
        assert_eq!(FfprobeOutputFormat::Xml.to_string(), "xml");
        assert_eq!(FfprobeOutputFormat::Csv.to_string(), "csv");
        assert_eq!(FfprobeOutputFormat::Default.to_string(), "default");
    }
}
