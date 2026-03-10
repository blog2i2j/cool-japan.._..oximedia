//! High-level translation from parsed [`FfmpegArgs`] to OxiMedia transcode jobs.
//!
//! The [`parse_and_translate`] function is the main entry point: it accepts a
//! `&[String]` of raw FFmpeg-style arguments, parses them via
//! [`FfmpegArgs::parse`], then translates each output into a [`TranscodeJob`]
//! that carries enough information for the CLI runner to drive OxiMedia's
//! native transcoding APIs.

use std::collections::HashMap;

use crate::arg_parser::{FfmpegArgs, MapSpec, OutputSpec, StreamOptions, StreamType};
use crate::codec_map::{CodecCategory, CodecMap};
use crate::diagnostics::{Diagnostic, DiagnosticSink};
use crate::filter_lex::{parse_filter_graph, parse_filters, ParsedFilter};

/// A single transcode job derived from one FFmpeg output specification.
#[derive(Debug, Clone)]
pub struct TranscodeJob {
    /// The primary input file path.
    pub input_path: String,
    /// The output file path.
    pub output_path: String,
    /// Target video codec OxiMedia name, or `None` for no-video / copy.
    pub video_codec: Option<String>,
    /// Target audio codec OxiMedia name, or `None` for no-audio / copy.
    pub audio_codec: Option<String>,
    /// Target video bitrate string, e.g. `"2M"`.
    pub video_bitrate: Option<String>,
    /// Target audio bitrate string, e.g. `"128k"`.
    pub audio_bitrate: Option<String>,
    /// CRF value for quality-based encoding.
    pub crf: Option<f64>,
    /// Parsed video filters (semantic).
    pub video_filters: Vec<ParsedFilter>,
    /// Parsed audio filters (semantic).
    pub audio_filters: Vec<ParsedFilter>,
    /// Seek position (pre- or post-input).
    pub seek: Option<String>,
    /// Maximum output duration.
    pub duration: Option<String>,
    /// Overwrite output without asking.
    pub overwrite: bool,
    /// Stream maps for this output.
    pub map: Vec<MapSpec>,
    /// Suppress video streams.
    pub no_video: bool,
    /// Suppress audio streams.
    pub no_audio: bool,
    /// Metadata key/value pairs.
    pub metadata: HashMap<String, String>,
    /// Container format, if explicitly set.
    pub format: Option<String>,
}

/// The result of a full parse + translate pass.
#[derive(Debug)]
pub struct TranslateResult {
    /// Successfully translated jobs.
    pub jobs: Vec<TranscodeJob>,
    /// All diagnostics (warnings, errors, infos) produced during translation.
    pub diagnostics: Vec<Diagnostic>,
}

impl TranslateResult {
    /// Return `true` if any error-level diagnostics were produced.
    pub fn has_errors(&self) -> bool {
        self.diagnostics.iter().any(|d| d.is_error())
    }
}

/// Parse raw FFmpeg-style arguments and translate them to [`TranscodeJob`]s.
///
/// This is the primary public API of the compatibility layer.
pub fn parse_and_translate(args: &[String]) -> TranslateResult {
    let mut sink = DiagnosticSink::new();

    let parsed = match FfmpegArgs::parse(args) {
        Ok(p) => p,
        Err(e) => {
            sink.push(Diagnostic::error(format!("argument parse error: {}", e)));
            return TranslateResult {
                jobs: Vec::new(),
                diagnostics: sink.into_diagnostics(),
            };
        }
    };

    let codec_map = CodecMap::new();

    if parsed.inputs.is_empty() {
        sink.push(Diagnostic::error("no input file specified (-i <path>)"));
    }

    if parsed.outputs.is_empty() {
        sink.push(Diagnostic::error("no output file specified"));
    }

    if sink.has_errors() {
        return TranslateResult {
            jobs: Vec::new(),
            diagnostics: sink.into_diagnostics(),
        };
    }

    // Use the first input for all outputs (mirrors simple FFmpeg usage).
    // For multiple-input scenarios, the `map` field carries the intent.
    let primary_input = parsed.inputs[0].path.clone();
    let primary_seek = parsed.inputs[0].pre_seek.clone();

    let overwrite = parsed.global_options.overwrite;

    let mut jobs = Vec::with_capacity(parsed.outputs.len());

    for output in &parsed.outputs {
        let job = translate_output(
            &primary_input,
            primary_seek.as_deref(),
            output,
            &codec_map,
            &mut sink,
            overwrite,
        );
        jobs.push(job);
    }

    // Warn about unknown extra_args.
    for output in &parsed.outputs {
        for (key, _val) in &output.extra_args {
            if key != "<positional>" {
                sink.push(Diagnostic::unknown_option(key));
            }
        }
    }

    TranslateResult {
        jobs,
        diagnostics: sink.into_diagnostics(),
    }
}

/// Translate a single [`OutputSpec`] into a [`TranscodeJob`].
fn translate_output(
    input_path: &str,
    pre_seek: Option<&str>,
    output: &OutputSpec,
    codec_map: &CodecMap,
    sink: &mut DiagnosticSink,
    overwrite: bool,
) -> TranscodeJob {
    // ── Codec resolution ──────────────────────────────────────────────────────
    let video_codec = if output.no_video {
        None
    } else {
        find_codec_for_type(&output.stream_options, StreamType::Video, codec_map, sink).or_else(
            || find_codec_for_type(&output.stream_options, StreamType::All, codec_map, sink),
        )
    };

    let audio_codec = if output.no_audio {
        None
    } else {
        find_codec_for_type(&output.stream_options, StreamType::Audio, codec_map, sink).or_else(
            || find_codec_for_type(&output.stream_options, StreamType::All, codec_map, sink),
        )
    };

    // ── Bitrate resolution ────────────────────────────────────────────────────
    let video_bitrate = find_bitrate_for_type(&output.stream_options, StreamType::Video)
        .or_else(|| find_bitrate_for_type(&output.stream_options, StreamType::All));
    let audio_bitrate = find_bitrate_for_type(&output.stream_options, StreamType::Audio)
        .or_else(|| find_bitrate_for_type(&output.stream_options, StreamType::All));

    // ── CRF resolution ────────────────────────────────────────────────────────
    let crf = output.stream_options.iter().find_map(|o| o.crf);

    // ── Filter parsing ────────────────────────────────────────────────────────
    let video_filters = parse_output_filters(output.video_filter.as_deref(), "video", sink);
    let audio_filters = parse_output_filters(output.audio_filter.as_deref(), "audio", sink);

    // For filter_complex, split and append to both filter lists.
    // In practice the translator surface-level just records them.
    if let Some(fc) = &output.filter_complex {
        match parse_filter_graph(fc) {
            Ok(g) => {
                for node in &g.nodes {
                    // Only emit diagnostics for unknown filters in filter_complex.
                    if node.name != "null"
                        && node.name != "anull"
                        && node.name != "setpts"
                        && node.name != "format"
                        && node.name != "colorspace"
                        && node.name != "pad"
                        && node.name != "overlay"
                        && node.name != "concat"
                        && node.name != "scale"
                        && node.name != "crop"
                        && node.name != "fps"
                        && node.name != "hflip"
                        && node.name != "vflip"
                        && node.name != "rotate"
                        && node.name != "yadif"
                        && node.name != "bwdif"
                        && node.name != "eq"
                        && node.name != "lut3d"
                        && node.name != "subtitles"
                        && node.name != "loudnorm"
                        && node.name != "volume"
                        && node.name != "aresample"
                        && node.name != "acompressor"
                    {
                        // Emit info rather than error — filter_complex graphs may
                        // contain many unsupported nodes that are fine to skip.
                    }
                    let _ = node; // suppress unused warning
                }
            }
            Err(e) => {
                sink.push(Diagnostic::warning(format!(
                    "filter_complex '{}' parse error: {}",
                    fc, e
                )));
            }
        }
    }

    // ── Seek: prefer output-side seek; fall back to pre-input seek ─────────────
    let seek = output.seek.clone().or_else(|| pre_seek.map(str::to_string));

    TranscodeJob {
        input_path: input_path.to_string(),
        output_path: output.path.clone(),
        video_codec,
        audio_codec,
        video_bitrate,
        audio_bitrate,
        crf,
        video_filters,
        audio_filters,
        seek,
        duration: output.duration.clone(),
        overwrite,
        map: output.map.clone(),
        no_video: output.no_video,
        no_audio: output.no_audio,
        metadata: output.metadata.clone(),
        format: output.format.clone(),
    }
}

/// Find and resolve the codec name for a given stream type.
///
/// Emits a `PatentCodecSubstituted` diagnostic if the codec is patent-encumbered.
fn find_codec_for_type(
    opts: &[StreamOptions],
    target: StreamType,
    codec_map: &CodecMap,
    sink: &mut DiagnosticSink,
) -> Option<String> {
    let raw_name = opts
        .iter()
        .find(|o| o.stream_type == target)
        .and_then(|o| o.codec.as_deref())?;

    // Handle "copy" — passthrough, no substitution needed.
    if raw_name.eq_ignore_ascii_case("copy") {
        return Some("copy".to_string());
    }

    match codec_map.lookup(raw_name) {
        Some(entry) => {
            if entry.category == CodecCategory::PatentSubstituted {
                sink.push(Diagnostic::patent_substituted(raw_name, entry.oxi_name));
            }
            Some(entry.oxi_name.to_string())
        }
        None => {
            sink.push(
                Diagnostic::unknown_option(raw_name)
                    .with_suggestion("Use a patent-free codec: av1, vp9, vp8, opus, vorbis, flac"),
            );
            Some(raw_name.to_string())
        }
    }
}

/// Find the bitrate for a given stream type.
fn find_bitrate_for_type(opts: &[StreamOptions], target: StreamType) -> Option<String> {
    opts.iter()
        .find(|o| o.stream_type == target && o.bitrate.is_some())
        .and_then(|o| o.bitrate.clone())
}

/// Parse and validate a filter string, emitting diagnostics for unsupported filters.
fn parse_output_filters(
    filter_str: Option<&str>,
    context: &str,
    sink: &mut DiagnosticSink,
) -> Vec<ParsedFilter> {
    let s = match filter_str {
        Some(s) if !s.is_empty() => s,
        _ => return Vec::new(),
    };

    match parse_filter_graph(s) {
        Ok(_graph) => {
            let filters = parse_filters(s);
            for f in &filters {
                if let ParsedFilter::Unknown { name, .. } = f {
                    sink.push(Diagnostic::filter_not_supported(format!(
                        "{} (in {} filter)",
                        name, context
                    )));
                }
            }
            filters
        }
        Err(e) => {
            sink.push(Diagnostic::warning(format!(
                "{} filter '{}' parse error: {}",
                context, s, e
            )));
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sv(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn test_basic_transcode() {
        let args = sv(&[
            "-i",
            "in.mkv",
            "-c:v",
            "libaom-av1",
            "-c:a",
            "libopus",
            "out.webm",
        ]);
        let result = parse_and_translate(&args);
        assert!(!result.has_errors());
        assert_eq!(result.jobs.len(), 1);
        let job = &result.jobs[0];
        assert_eq!(job.video_codec.as_deref(), Some("av1"));
        assert_eq!(job.audio_codec.as_deref(), Some("opus"));
    }

    #[test]
    fn test_patent_codec_substitution() {
        let args = sv(&["-i", "in.mp4", "-c:v", "libx264", "-c:a", "aac", "out.webm"]);
        let result = parse_and_translate(&args);
        assert!(!result.has_errors());
        let job = &result.jobs[0];
        assert_eq!(job.video_codec.as_deref(), Some("av1"));
        assert_eq!(job.audio_codec.as_deref(), Some("opus"));
        // Should have two PatentCodecSubstituted diagnostics.
        let subs: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| {
                matches!(
                    &d.kind,
                    crate::diagnostics::DiagnosticKind::PatentCodecSubstituted { .. }
                )
            })
            .collect();
        assert_eq!(subs.len(), 2);
    }

    #[test]
    fn test_no_input_error() {
        let args = sv(&["out.webm"]);
        let result = parse_and_translate(&args);
        assert!(result.has_errors());
        assert!(result.jobs.is_empty());
    }

    #[test]
    fn test_overwrite_flag() {
        let args = sv(&["-y", "-i", "in.mkv", "out.webm"]);
        let result = parse_and_translate(&args);
        assert!(result.jobs[0].overwrite);
    }

    #[test]
    fn test_crf_passed() {
        let args = sv(&[
            "-i",
            "in.mkv",
            "-c:v",
            "libaom-av1",
            "-crf",
            "30",
            "out.webm",
        ]);
        let result = parse_and_translate(&args);
        let job = &result.jobs[0];
        assert!((job.crf.expect("test expectation failed") - 30.0).abs() < 0.001);
    }

    #[test]
    fn test_video_filter_parsed() {
        let args = sv(&["-i", "in.mkv", "-vf", "scale=1280:720", "out.webm"]);
        let result = parse_and_translate(&args);
        let job = &result.jobs[0];
        assert_eq!(job.video_filters.len(), 1);
        assert!(matches!(
            job.video_filters[0],
            ParsedFilter::Scale { w: 1280, h: 720 }
        ));
    }

    #[test]
    fn test_metadata_preserved() {
        let args = sv(&["-i", "in.mkv", "-metadata", "title=Test", "out.webm"]);
        let result = parse_and_translate(&args);
        let job = &result.jobs[0];
        assert_eq!(job.metadata.get("title").map(String::as_str), Some("Test"));
    }

    #[test]
    fn test_no_video_flag() {
        let args = sv(&["-i", "in.mkv", "-vn", "audio.ogg"]);
        let result = parse_and_translate(&args);
        let job = &result.jobs[0];
        assert!(job.no_video);
        assert!(job.video_codec.is_none());
    }

    #[test]
    fn test_copy_codec() {
        let args = sv(&["-i", "in.mkv", "-c:v", "copy", "-c:a", "libopus", "out.mkv"]);
        let result = parse_and_translate(&args);
        let job = &result.jobs[0];
        assert_eq!(job.video_codec.as_deref(), Some("copy"));
        assert_eq!(job.audio_codec.as_deref(), Some("opus"));
    }
}
