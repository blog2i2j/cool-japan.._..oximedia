//! FFmpeg-style argument parser (stateful, left-to-right).
//!
//! Parses a flat `&[String]` of FFmpeg-style arguments into structured
//! [`FfmpegArgs`] — separating global options, input specs, and output specs.
//!
//! ## State machine
//!
//! The parser operates as a state machine moving left to right:
//!
//! ```text
//! PreInput  ──(-i PATH)──▶  BetweenInputs  ──(non-flag)──▶  BuildingOutput  ──▶  Done
//! ```
//!
//! - Options appearing **before** the first `-i` are applied to the *next* input.
//! - Options appearing **after** the last `-i` are applied to the *next* output.
//! - `-f FORMAT` is context-sensitive: before `-i` it becomes `InputSpec::format`;
//!   after all inputs it becomes `OutputSpec::format`.

use std::collections::HashMap;

use crate::diagnostics::TranslationError;
use crate::encoder_options::{
    EncoderProfile, EncoderQualityOptions, EncoderQualityPreset, EncoderTune,
};
use crate::filter_graph::FilterGraph as AdvancedFilterGraph;
use crate::filter_shorthand::{parse_af, parse_vf};
use crate::pass::{phase_from_parts, PassPhase};

/// Global FFmpeg options (apply to the whole session, not a specific stream).
#[derive(Debug, Clone, Default)]
pub struct GlobalOptions {
    /// Overwrite output files without asking (`-y`).
    pub overwrite: bool,
    /// Never overwrite output files (`-n`).
    pub no_overwrite: bool,
    /// Log level string (`-loglevel LEVEL`).
    pub log_level: Option<String>,
    /// Thread count (`-threads N`; `None` = unset, use default).
    pub threads: Option<usize>,
    /// Hardware acceleration method (`-hwaccel METHOD`).
    pub hwaccel: Option<String>,
}

/// Which media type a stream option targets.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum StreamType {
    /// Video streams (`-c:v`, `-b:v`, …).
    Video,
    /// Audio streams (`-c:a`, `-b:a`, …).
    Audio,
    /// Subtitle streams (`-c:s`).
    Subtitle,
    /// Applies to all stream types (`-c`, `-codec`).
    #[default]
    All,
}

impl std::fmt::Display for StreamType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Video => write!(f, "v"),
            Self::Audio => write!(f, "a"),
            Self::Subtitle => write!(f, "s"),
            Self::All => write!(f, "*"),
        }
    }
}

/// Per-stream encoding options for a single stream selector.
#[derive(Debug, Clone, Default)]
pub struct StreamOptions {
    /// Which stream type this option block applies to.
    pub stream_type: StreamType,
    /// Codec name, e.g. `"libaom-av1"`, `"libvpx-vp9"`, `"copy"`.
    pub codec: Option<String>,
    /// Bit-rate string, e.g. `"2M"`, `"128k"`.
    pub bitrate: Option<String>,
    /// Constant-rate factor (quality-based encoding).
    pub crf: Option<f64>,
    /// Quality scale value (`-q:v`).
    pub quality: Option<f64>,
    /// Pixel format, e.g. `"yuv420p"`.
    pub pixel_fmt: Option<String>,
    /// Audio sample rate in Hz (`-ar`).
    pub sample_rate: Option<u32>,
    /// Audio channel count (`-ac`).
    pub channels: Option<u8>,
    /// Frame rate string, e.g. `"24"`, `"30000/1001"` (`-r`).
    pub frame_rate: Option<String>,
    /// Video resolution string, e.g. `"1280x720"` (`-s`).
    pub size: Option<String>,
}

/// A stream-selection mapping (`-map` flag).
#[derive(Debug, Clone)]
pub struct MapSpec {
    /// Input file index (0-based).
    pub input_index: usize,
    /// Stream selector, e.g. `"v"`, `"a"`, `"v:0"`, `"a:1"`, `"s"`.
    pub stream_selector: Option<String>,
    /// Filter-complex output pad label, e.g. `"[out_v]"`.
    pub label: Option<String>,
    /// Negative map (exclude rather than include).
    pub negative: bool,
}

/// A single input file/URL with its pre-input options.
#[derive(Debug, Clone)]
pub struct InputSpec {
    /// The path or URL for this input.
    pub path: String,
    /// Seek position set before `-i` (`-ss` in pre-input position).
    pub pre_seek: Option<String>,
    /// Format override set before `-i` (`-f` in pre-input position).
    pub format: Option<String>,
    /// Stream options attached to this input.
    pub stream_options: Vec<StreamOptions>,
}

/// A single output file/URL with all its associated options.
#[derive(Debug, Clone, Default)]
pub struct OutputSpec {
    /// The output path or URL.
    pub path: String,
    /// Container format override (`-f`).
    pub format: Option<String>,
    /// Stream options (codec, bitrate, etc.) grouped by stream type.
    pub stream_options: Vec<StreamOptions>,
    /// Video filtergraph string (`-vf`/`-filter:v`).
    pub video_filter: Option<String>,
    /// Parsed `-vf` shorthand graph wrapped as a single `[in]...[out]` chain.
    pub video_filter_graph: Option<crate::filter_complex::FilterGraph>,
    /// Audio filtergraph string (`-af`/`-filter:a`).
    pub audio_filter: Option<String>,
    /// Parsed `-af` shorthand graph wrapped as a single `[in]...[out]` chain.
    pub audio_filter_graph: Option<crate::filter_complex::FilterGraph>,
    /// Complex filtergraph string (`-filter_complex`).
    pub filter_complex: Option<String>,
    /// Stream mappings for this output.
    pub map: Vec<MapSpec>,
    /// Suppress all video streams (`-vn`).
    pub no_video: bool,
    /// Suppress all audio streams (`-an`).
    pub no_audio: bool,
    /// Suppress all subtitle streams (`-sn`).
    pub no_subtitle: bool,
    /// Stop encoding when the shortest stream ends (`-shortest`).
    pub shortest: bool,
    /// Seek into input from output side (`-ss` in post-input position).
    pub seek: Option<String>,
    /// Maximum duration of output (`-t`).
    pub duration: Option<String>,
    /// Metadata key/value pairs (`-metadata key=value`).
    pub metadata: HashMap<String, String>,
    /// Encoding preset (e.g. `"medium"`, `"fast"`, `"veryslow"`).
    pub preset: Option<String>,
    /// Encoding tune (e.g. `"film"`, `"animation"`, `"grain"`).
    pub tune: Option<String>,
    /// Encoding profile (e.g. `"baseline"`, `"main"`, `"high"`).
    pub profile: Option<String>,
    /// Parsed encoder quality options.
    pub encoder_quality: EncoderQualityOptions,
    /// Two-pass encoding pass number (1 or 2).
    pub pass: Option<u8>,
    /// Passlogfile prefix for two-pass encoding.
    pub passlogfile: Option<String>,
    /// Parsed two-pass phase details.
    pub pass_phase: Option<PassPhase>,
    /// Muxer options (e.g. `movflags`, `fflags`).
    pub muxer_options: Vec<(String, String)>,
    /// Unrecognised option key/value pairs.
    pub extra_args: Vec<(String, String)>,
}

/// Top-level parsed representation of an FFmpeg command line.
#[derive(Debug, Clone, Default)]
pub struct FfmpegArgs {
    /// Global options (apply to the whole session).
    pub global_options: GlobalOptions,
    /// Ordered list of input specifications.
    pub inputs: Vec<InputSpec>,
    /// Ordered list of output specifications.
    pub outputs: Vec<OutputSpec>,
}

/// Internal parser state tracking where we are in the arg list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParseState {
    /// No `-i` seen yet; options accumulate on the next input.
    PreInput,
    /// At least one `-i` has been seen; options accumulate on the next output.
    AfterInput,
}

impl FfmpegArgs {
    /// Parse a slice of string arguments into [`FfmpegArgs`].
    ///
    /// Follows FFmpeg's left-to-right, context-sensitive parsing convention.
    /// Unknown flags are stored in `OutputSpec::extra_args` rather than failing.
    pub fn parse(args: &[String]) -> anyhow::Result<Self> {
        let mut result = FfmpegArgs::default();
        let mut state = ParseState::PreInput;

        // Pending input-side options (applied when `-i` is encountered).
        let mut pending_pre_seek: Option<String> = None;
        let mut pending_pre_format: Option<String> = None;
        let mut pending_input_stream_opts: Vec<StreamOptions> = Vec::new();

        // Pending output-side options (applied when an output filename is found).
        let mut pending_output: OutputSpec = OutputSpec::default();

        let mut i = 0usize;

        macro_rules! next_arg {
            ($flag:expr) => {{
                i += 1;
                if i >= args.len() {
                    anyhow::bail!("flag '{}' requires an argument", $flag);
                }
                &args[i]
            }};
        }

        while i < args.len() {
            let arg = args[i].as_str();

            match arg {
                // ── Global flags ──────────────────────────────────────────────
                "-y" => {
                    result.global_options.overwrite = true;
                }
                "-n" => {
                    result.global_options.no_overwrite = true;
                }
                "-loglevel" | "-v" => {
                    let val = next_arg!(arg);
                    result.global_options.log_level = Some(val.clone());
                }
                "-threads" => {
                    let val = next_arg!(arg);
                    result.global_options.threads = val.parse::<usize>().ok();
                }
                "-hwaccel" => {
                    let val = next_arg!(arg);
                    result.global_options.hwaccel = Some(val.clone());
                }
                "-hide_banner" | "-nostdin" | "-nostats" | "-benchmark" => {
                    // Recognised flags that take no value; silently accepted.
                }

                // ── Input ─────────────────────────────────────────────────────
                "-i" => {
                    let path = next_arg!("-i").clone();
                    result.inputs.push(InputSpec {
                        path,
                        pre_seek: pending_pre_seek.take(),
                        format: pending_pre_format.take(),
                        stream_options: std::mem::take(&mut pending_input_stream_opts),
                    });
                    state = ParseState::AfterInput;
                    // Reset output pending state (each -i group starts fresh for next output).
                    pending_output = OutputSpec::default();
                }

                // ── Format (context-sensitive) ────────────────────────────────
                "-f" => {
                    let val = next_arg!("-f").clone();
                    if state == ParseState::PreInput {
                        pending_pre_format = Some(val);
                    } else {
                        pending_output.format = Some(val);
                    }
                }

                // ── Seek (context-sensitive) ──────────────────────────────────
                "-ss" => {
                    let val = next_arg!("-ss").clone();
                    if state == ParseState::PreInput {
                        pending_pre_seek = Some(val);
                    } else {
                        pending_output.seek = Some(val);
                    }
                }

                // ── Duration ──────────────────────────────────────────────────
                "-t" => {
                    let val = next_arg!("-t").clone();
                    pending_output.duration = Some(val);
                }

                // ── Codec options ─────────────────────────────────────────────
                "-c" | "-codec" => {
                    let val = next_arg!(arg).clone();
                    let opt = StreamOptions {
                        stream_type: StreamType::All,
                        codec: Some(val),
                        ..Default::default()
                    };
                    push_stream_opt(
                        &mut pending_output,
                        &mut pending_input_stream_opts,
                        opt,
                        state,
                    );
                }
                "-c:v" | "-vcodec" | "-codec:v" => {
                    let val = next_arg!(arg).clone();
                    let opt = StreamOptions {
                        stream_type: StreamType::Video,
                        codec: Some(val),
                        ..Default::default()
                    };
                    push_stream_opt(
                        &mut pending_output,
                        &mut pending_input_stream_opts,
                        opt,
                        state,
                    );
                }
                "-c:a" | "-acodec" | "-codec:a" => {
                    let val = next_arg!(arg).clone();
                    let opt = StreamOptions {
                        stream_type: StreamType::Audio,
                        codec: Some(val),
                        ..Default::default()
                    };
                    push_stream_opt(
                        &mut pending_output,
                        &mut pending_input_stream_opts,
                        opt,
                        state,
                    );
                }
                "-c:s" | "-scodec" | "-codec:s" => {
                    let val = next_arg!(arg).clone();
                    let opt = StreamOptions {
                        stream_type: StreamType::Subtitle,
                        codec: Some(val),
                        ..Default::default()
                    };
                    push_stream_opt(
                        &mut pending_output,
                        &mut pending_input_stream_opts,
                        opt,
                        state,
                    );
                }

                // ── Bitrate options ───────────────────────────────────────────
                "-b" | "-b:v" => {
                    let val = next_arg!(arg).clone();
                    let opt = StreamOptions {
                        stream_type: StreamType::Video,
                        bitrate: Some(val),
                        ..Default::default()
                    };
                    push_stream_opt(
                        &mut pending_output,
                        &mut pending_input_stream_opts,
                        opt,
                        state,
                    );
                }
                "-b:a" => {
                    let val = next_arg!(arg).clone();
                    let opt = StreamOptions {
                        stream_type: StreamType::Audio,
                        bitrate: Some(val),
                        ..Default::default()
                    };
                    push_stream_opt(
                        &mut pending_output,
                        &mut pending_input_stream_opts,
                        opt,
                        state,
                    );
                }

                // ── Quality options ───────────────────────────────────────────
                "-crf" => {
                    let val = next_arg!("-crf");
                    let crf = val.parse::<f64>().map_err(|_| {
                        anyhow::anyhow!("invalid CRF value '{}': must be a number", val)
                    })?;
                    let opt = StreamOptions {
                        stream_type: StreamType::Video,
                        crf: Some(crf),
                        ..Default::default()
                    };
                    push_stream_opt(
                        &mut pending_output,
                        &mut pending_input_stream_opts,
                        opt,
                        state,
                    );
                }
                "-q:v" | "-qv" => {
                    let val = next_arg!(arg);
                    let q = val
                        .parse::<f64>()
                        .map_err(|_| anyhow::anyhow!("invalid quality value '{}'", val))?;
                    let opt = StreamOptions {
                        stream_type: StreamType::Video,
                        quality: Some(q),
                        ..Default::default()
                    };
                    push_stream_opt(
                        &mut pending_output,
                        &mut pending_input_stream_opts,
                        opt,
                        state,
                    );
                }

                // ── Video options ─────────────────────────────────────────────
                "-pix_fmt" | "-pixel_format" => {
                    let val = next_arg!(arg).clone();
                    let opt = StreamOptions {
                        stream_type: StreamType::Video,
                        pixel_fmt: Some(val),
                        ..Default::default()
                    };
                    push_stream_opt(
                        &mut pending_output,
                        &mut pending_input_stream_opts,
                        opt,
                        state,
                    );
                }
                "-r" => {
                    let val = next_arg!("-r").clone();
                    let opt = StreamOptions {
                        stream_type: StreamType::Video,
                        frame_rate: Some(val),
                        ..Default::default()
                    };
                    push_stream_opt(
                        &mut pending_output,
                        &mut pending_input_stream_opts,
                        opt,
                        state,
                    );
                }
                "-s" => {
                    let val = next_arg!("-s").clone();
                    let opt = StreamOptions {
                        stream_type: StreamType::Video,
                        size: Some(val),
                        ..Default::default()
                    };
                    push_stream_opt(
                        &mut pending_output,
                        &mut pending_input_stream_opts,
                        opt,
                        state,
                    );
                }

                // ── Audio options ─────────────────────────────────────────────
                "-ar" => {
                    let val = next_arg!("-ar");
                    let sr = val.parse::<u32>().map_err(|_| {
                        anyhow::anyhow!("invalid sample rate '{}': must be an integer", val)
                    })?;
                    let opt = StreamOptions {
                        stream_type: StreamType::Audio,
                        sample_rate: Some(sr),
                        ..Default::default()
                    };
                    push_stream_opt(
                        &mut pending_output,
                        &mut pending_input_stream_opts,
                        opt,
                        state,
                    );
                }
                "-ac" => {
                    let val = next_arg!("-ac");
                    let ch = val.parse::<u8>().map_err(|_| {
                        anyhow::anyhow!("invalid channel count '{}': must be 1–255", val)
                    })?;
                    let opt = StreamOptions {
                        stream_type: StreamType::Audio,
                        channels: Some(ch),
                        ..Default::default()
                    };
                    push_stream_opt(
                        &mut pending_output,
                        &mut pending_input_stream_opts,
                        opt,
                        state,
                    );
                }

                // ── Filter options ────────────────────────────────────────────
                "-vf" | "-filter:v" => {
                    let val = next_arg!(arg).clone();
                    pending_output.video_filter_graph = Some(parse_vf(&val)?);
                    pending_output.video_filter = Some(val);
                }
                "-af" | "-filter:a" => {
                    let val = next_arg!(arg).clone();
                    pending_output.audio_filter_graph = Some(parse_af(&val)?);
                    pending_output.audio_filter = Some(val);
                }
                "-filter_complex" | "-lavfi" => {
                    let val = next_arg!(arg).clone();
                    pending_output.filter_complex = Some(val);
                }

                // ── Map ───────────────────────────────────────────────────────
                "-map" => {
                    let val = next_arg!("-map");
                    if let Some(ms) = parse_map_spec(val) {
                        pending_output.map.push(ms);
                    }
                }

                // ── Stream disable flags ──────────────────────────────────────
                "-vn" => {
                    pending_output.no_video = true;
                }
                "-an" => {
                    pending_output.no_audio = true;
                }
                "-sn" => {
                    pending_output.no_subtitle = true;
                }

                // ── Preset / Tune / Profile ───────────────────────────────────
                "-preset" => {
                    let val = next_arg!("-preset").clone();
                    pending_output.encoder_quality.preset =
                        Some(val.parse::<EncoderQualityPreset>()?);
                    pending_output.preset = Some(val);
                }
                "-tune" => {
                    let val = next_arg!("-tune").clone();
                    pending_output.encoder_quality.tune = Some(val.parse::<EncoderTune>()?);
                    pending_output.tune = Some(val);
                }
                "-profile" | "-profile:v" => {
                    let val = next_arg!(arg).clone();
                    pending_output.encoder_quality.profile = Some(val.parse::<EncoderProfile>()?);
                    pending_output.profile = Some(val);
                }

                // ── Two-pass encoding ────────────────────────────────────────
                "-pass" => {
                    let val = next_arg!("-pass");
                    pending_output.pass = val.parse::<u8>().ok();
                }
                "-passlogfile" => {
                    let val = next_arg!("-passlogfile").clone();
                    pending_output.passlogfile = Some(val);
                }

                // ── Muxer options ────────────────────────────────────────────
                "-movflags" => {
                    let val = next_arg!("-movflags").clone();
                    pending_output
                        .muxer_options
                        .push(("movflags".to_string(), val));
                }
                "-fflags" => {
                    let val = next_arg!("-fflags").clone();
                    pending_output
                        .muxer_options
                        .push(("fflags".to_string(), val));
                }
                "-brand" => {
                    let val = next_arg!("-brand").clone();
                    pending_output
                        .muxer_options
                        .push(("brand".to_string(), val));
                }
                "-write_tmcd" => {
                    let val = next_arg!("-write_tmcd").clone();
                    pending_output
                        .muxer_options
                        .push(("write_tmcd".to_string(), val));
                }
                "-moov_size" => {
                    let val = next_arg!("-moov_size").clone();
                    pending_output
                        .muxer_options
                        .push(("moov_size".to_string(), val));
                }
                "-fragment_index" => {
                    let val = next_arg!("-fragment_index").clone();
                    pending_output
                        .muxer_options
                        .push(("fragment_index".to_string(), val));
                }
                "-movflags:v" => {
                    let val = next_arg!("-movflags:v").clone();
                    pending_output
                        .muxer_options
                        .push(("movflags".to_string(), val));
                }

                // ── To position ──────────────────────────────────────────────
                "-to" => {
                    let val = next_arg!("-to").clone();
                    // FFmpeg -to sets end position; we store it as duration context
                    pending_output.extra_args.push(("-to".to_string(), val));
                }

                // ── Misc output flags ─────────────────────────────────────────
                "-shortest" => {
                    pending_output.shortest = true;
                }

                // ── Metadata ──────────────────────────────────────────────────
                "-metadata" => {
                    let kv = next_arg!("-metadata");
                    if let Some(eq) = kv.find('=') {
                        let key = kv[..eq].to_string();
                        let value = kv[eq + 1..].to_string();
                        pending_output.metadata.insert(key, value);
                    }
                }

                // ── Non-flag args → output filenames ──────────────────────────
                other if !other.starts_with('-') => {
                    if state == ParseState::PreInput {
                        // Positional arg before any -i; treat as unknown / ignored.
                        pending_output
                            .extra_args
                            .push(("<positional>".to_string(), other.to_string()));
                    } else {
                        // This is an output filename.
                        let mut output = std::mem::take(&mut pending_output);
                        output.path = other.to_string();
                        result.outputs.push(output);
                        // Reset for next possible output.
                        pending_output = OutputSpec::default();
                    }
                }

                // ── Unknown options ───────────────────────────────────────────
                other => {
                    // Peek at next arg: if it exists and doesn't start with `-`,
                    // treat it as the value for this unknown flag.
                    let key = other.to_string();
                    let next_is_value = args
                        .get(i + 1)
                        .map(|a| !a.starts_with('-'))
                        .unwrap_or(false);
                    if next_is_value {
                        i += 1;
                        let value = args[i].clone();
                        pending_output.extra_args.push((key, value));
                    } else {
                        pending_output.extra_args.push((key, String::new()));
                    }
                }
            }

            i += 1;
        }

        for output in &mut result.outputs {
            output.pass_phase = phase_from_parts(output.pass, output.passlogfile.as_deref())?;
            validate_filter_conflicts(output)?;
        }

        Ok(result)
    }
}

fn validate_filter_conflicts(output: &OutputSpec) -> Result<(), TranslationError> {
    let Some(filter_complex) = output.filter_complex.as_deref() else {
        return Ok(());
    };

    let graph = match AdvancedFilterGraph::parse(filter_complex) {
        Ok(graph) => graph,
        Err(_) => return Ok(()),
    };

    if output.video_filter.is_some() && graph.has_video_filters() {
        return Err(TranslationError::ParseError(
            "cannot combine -vf with -filter_complex for video".to_string(),
        ));
    }

    if output.audio_filter.is_some() && graph.has_audio_filters() {
        return Err(TranslationError::ParseError(
            "cannot combine -af with -filter_complex for audio".to_string(),
        ));
    }

    Ok(())
}

/// Push a stream option onto the correct pending collection depending on parse state.
fn push_stream_opt(
    output: &mut OutputSpec,
    input_opts: &mut Vec<StreamOptions>,
    opt: StreamOptions,
    state: ParseState,
) {
    if state == ParseState::PreInput {
        input_opts.push(opt);
    } else {
        output.stream_options.push(opt);
    }
}

/// Parse a single `-map` specifier string.
///
/// Handles forms like:
/// - `"0"` — all streams from input 0
/// - `"0:v:0"` — first video stream of input 0
/// - `"0:a:1"` — second audio stream of input 0
/// - `"-0:a"` — negative map
/// - `"[out_v]"` — filter-complex output label
fn parse_map_spec(spec: &str) -> Option<MapSpec> {
    let spec = spec.trim();

    // Handle filter-complex label form: `[label]`.
    if spec.starts_with('[') && spec.ends_with(']') {
        return Some(MapSpec {
            input_index: 0,
            stream_selector: None,
            label: Some(spec[1..spec.len() - 1].to_string()),
            negative: false,
        });
    }

    let (negative, rest) = if let Some(s) = spec.strip_prefix('-') {
        (true, s)
    } else {
        (false, spec)
    };

    let mut parts = rest.splitn(3, ':');
    let index_str = parts.next()?;
    let input_index = index_str.parse::<usize>().ok()?;

    // Build the stream selector from remaining parts.
    let selector = {
        let type_part = parts.next().unwrap_or("");
        let idx_part = parts.next().unwrap_or("");
        if type_part.is_empty() {
            None
        } else if idx_part.is_empty() {
            Some(type_part.to_string())
        } else {
            Some(format!("{}:{}", type_part, idx_part))
        }
    };

    Some(MapSpec {
        input_index,
        stream_selector: selector,
        label: None,
        negative,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &str) -> String {
        v.to_string()
    }

    fn tmp_str(name: &str) -> String {
        std::env::temp_dir()
            .join(format!("oximedia-compat-ffmpeg-argparse-{name}"))
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn test_simple_transcode() {
        let args = vec![
            s("-i"),
            s("input.mkv"),
            s("-c:v"),
            s("libaom-av1"),
            s("-c:a"),
            s("libopus"),
            s("-crf"),
            s("28"),
            s("output.webm"),
        ];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        assert_eq!(parsed.inputs.len(), 1);
        assert_eq!(parsed.outputs.len(), 1);
        let out = &parsed.outputs[0];
        assert_eq!(out.path, "output.webm");
        let video_opt = out
            .stream_options
            .iter()
            .find(|o| o.stream_type == StreamType::Video && o.codec.is_some())
            .expect("video opt");
        assert_eq!(video_opt.codec.as_deref(), Some("libaom-av1"));
        let crf_opt = out
            .stream_options
            .iter()
            .find(|o| o.crf.is_some())
            .expect("crf opt");
        assert!((crf_opt.crf.expect("test expectation failed") - 28.0).abs() < 0.001);
    }

    #[test]
    fn test_global_overwrite() {
        let args = vec![s("-y"), s("-i"), s("in.mkv"), s("out.webm")];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        assert!(parsed.global_options.overwrite);
        assert!(!parsed.global_options.no_overwrite);
    }

    #[test]
    fn test_format_context_sensitive() {
        let args = vec![
            s("-f"),
            s("matroska"),
            s("-i"),
            s("in.mkv"),
            s("-f"),
            s("webm"),
            s("out.webm"),
        ];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        assert_eq!(parsed.inputs[0].format.as_deref(), Some("matroska"));
        assert_eq!(parsed.outputs[0].format.as_deref(), Some("webm"));
    }

    #[test]
    fn test_seek_context_sensitive() {
        let args = vec![
            s("-ss"),
            s("00:01:00"),
            s("-i"),
            s("in.mkv"),
            s("-ss"),
            s("00:00:10"),
            s("out.webm"),
        ];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        assert_eq!(parsed.inputs[0].pre_seek.as_deref(), Some("00:01:00"));
        assert_eq!(parsed.outputs[0].seek.as_deref(), Some("00:00:10"));
    }

    #[test]
    fn test_no_video_no_audio() {
        let args = vec![s("-i"), s("in.mkv"), s("-vn"), s("-an"), s("out.webm")];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        assert!(parsed.outputs[0].no_video);
        assert!(parsed.outputs[0].no_audio);
    }

    #[test]
    fn test_metadata_parsing() {
        let args = vec![
            s("-i"),
            s("in.mkv"),
            s("-metadata"),
            s("title=My Video"),
            s("out.webm"),
        ];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        assert_eq!(
            parsed.outputs[0].metadata.get("title").map(String::as_str),
            Some("My Video")
        );
    }

    #[test]
    fn test_map_parsing() {
        let args = vec![
            s("-i"),
            s("in.mkv"),
            s("-map"),
            s("0:v:0"),
            s("-map"),
            s("0:a:1"),
            s("out.webm"),
        ];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        assert_eq!(parsed.outputs[0].map.len(), 2);
        assert_eq!(
            parsed.outputs[0].map[0].stream_selector.as_deref(),
            Some("v:0")
        );
        assert_eq!(
            parsed.outputs[0].map[1].stream_selector.as_deref(),
            Some("a:1")
        );
    }

    #[test]
    fn test_filter_complex() {
        let args = vec![
            s("-i"),
            s("in.mkv"),
            s("-filter_complex"),
            s("[0:v]scale=1280:720[out]"),
            s("out.webm"),
        ];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        assert_eq!(
            parsed.outputs[0].filter_complex.as_deref(),
            Some("[0:v]scale=1280:720[out]")
        );
    }

    #[test]
    fn test_threads_hwaccel() {
        let args = vec![
            s("-threads"),
            s("4"),
            s("-hwaccel"),
            s("vaapi"),
            s("-i"),
            s("in.mkv"),
            s("out.webm"),
        ];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        assert_eq!(parsed.global_options.threads, Some(4));
        assert_eq!(parsed.global_options.hwaccel.as_deref(), Some("vaapi"));
    }

    #[test]
    fn test_single_input_output() {
        let args = vec![s("-i"), s("input.mkv"), s("output.webm")];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        assert_eq!(parsed.inputs.len(), 1);
        assert_eq!(parsed.inputs[0].path, "input.mkv");
        assert_eq!(parsed.outputs.len(), 1);
        assert_eq!(parsed.outputs[0].path, "output.webm");
    }

    #[test]
    fn test_multiple_inputs() {
        let args = vec![s("-i"), s("a.mkv"), s("-i"), s("b.mkv"), s("out.webm")];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        assert_eq!(parsed.inputs.len(), 2);
        assert_eq!(parsed.inputs[0].path, "a.mkv");
        assert_eq!(parsed.inputs[1].path, "b.mkv");
        assert_eq!(parsed.outputs.len(), 1);
    }

    #[test]
    fn test_global_no_overwrite() {
        let args = vec![s("-n"), s("-i"), s("in.mkv"), s("out.webm")];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        assert!(parsed.global_options.no_overwrite);
    }

    #[test]
    fn test_codec_options_video_and_audio() {
        let args = vec![
            s("-i"),
            s("in.mkv"),
            s("-c:v"),
            s("libx264"),
            s("-c:a"),
            s("aac"),
            s("out.webm"),
        ];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        let output = &parsed.outputs[0];
        let has_video = output.stream_options.iter().any(|so| {
            so.stream_type == StreamType::Video && so.codec.as_deref() == Some("libx264")
        });
        let has_audio = output
            .stream_options
            .iter()
            .any(|so| so.stream_type == StreamType::Audio && so.codec.as_deref() == Some("aac"));
        assert!(has_video, "should have video codec option");
        assert!(has_audio, "should have audio codec option");
    }

    #[test]
    fn test_video_filter() {
        let args = vec![
            s("-i"),
            s("in.mkv"),
            s("-vf"),
            s("scale=1280:720"),
            s("out.webm"),
        ];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        assert_eq!(
            parsed.outputs[0].video_filter.as_deref(),
            Some("scale=1280:720")
        );
    }

    #[test]
    fn test_filter_complex_with_map() {
        let args = vec![
            s("-i"),
            s("in.mkv"),
            s("-filter_complex"),
            s("[0:v]scale=1280:720[out]"),
            s("-map"),
            s("[out]"),
            s("out.webm"),
        ];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        assert_eq!(
            parsed.outputs[0].filter_complex.as_deref(),
            Some("[0:v]scale=1280:720[out]")
        );
        assert_eq!(parsed.outputs[0].map.len(), 1);
        // label form: map has a label, not a stream_selector
        let map = &parsed.outputs[0].map[0];
        assert_eq!(map.label.as_deref(), Some("out"));
    }

    #[test]
    fn test_seek_before_input() {
        let args = vec![s("-ss"), s("00:01:30"), s("-i"), s("in.mkv"), s("out.webm")];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        assert_eq!(parsed.inputs[0].pre_seek.as_deref(), Some("00:01:30"));
    }

    #[test]
    fn test_no_subtitle_flag() {
        let args = vec![s("-i"), s("in.mkv"), s("-sn"), s("out.webm")];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        assert!(parsed.outputs[0].no_subtitle);
    }

    #[test]
    fn test_crf_value() {
        let args = vec![s("-i"), s("in.mkv"), s("-crf"), s("28"), s("out.webm")];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        let has_crf = parsed.outputs[0]
            .stream_options
            .iter()
            .any(|so| so.crf.map(|c| (c - 28.0).abs() < 0.001).unwrap_or(false));
        assert!(has_crf, "should have crf=28");
    }

    #[test]
    fn test_bitrate_video_and_audio() {
        let args = vec![
            s("-i"),
            s("in.mkv"),
            s("-b:v"),
            s("2M"),
            s("-b:a"),
            s("128k"),
            s("out.webm"),
        ];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        let has_vbr = parsed.outputs[0]
            .stream_options
            .iter()
            .any(|so| so.stream_type == StreamType::Video && so.bitrate.as_deref() == Some("2M"));
        assert!(has_vbr, "should have video bitrate 2M");
        let has_abr = parsed.outputs[0]
            .stream_options
            .iter()
            .any(|so| so.stream_type == StreamType::Audio && so.bitrate.as_deref() == Some("128k"));
        assert!(has_abr, "should have audio bitrate 128k");
    }

    #[test]
    fn test_map_spec_multi_input() {
        let args = vec![
            s("-i"),
            s("a.mkv"),
            s("-i"),
            s("b.mkv"),
            s("-map"),
            s("0:v"),
            s("-map"),
            s("1:a"),
            s("out.webm"),
        ];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        assert_eq!(parsed.outputs[0].map.len(), 2);
        let m0 = &parsed.outputs[0].map[0];
        assert_eq!(m0.input_index, 0);
        assert_eq!(m0.stream_selector.as_deref(), Some("v"));
        let m1 = &parsed.outputs[0].map[1];
        assert_eq!(m1.input_index, 1);
        assert_eq!(m1.stream_selector.as_deref(), Some("a"));
    }

    #[test]
    fn test_format_flag_output() {
        let args = vec![s("-i"), s("in.mkv"), s("-f"), s("webm"), s("out.webm")];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        assert_eq!(parsed.outputs[0].format.as_deref(), Some("webm"));
    }

    #[test]
    fn test_shortest_flag() {
        let args = vec![s("-i"), s("in.mkv"), s("-shortest"), s("out.webm")];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        assert!(parsed.outputs[0].shortest);
    }

    #[test]
    fn test_duration_flag() {
        let args = vec![s("-i"), s("in.mkv"), s("-t"), s("00:02:00"), s("out.webm")];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        assert_eq!(parsed.outputs[0].duration.as_deref(), Some("00:02:00"));
    }

    #[test]
    fn test_audio_sample_rate_and_channels() {
        let args = vec![
            s("-i"),
            s("in.mkv"),
            s("-ar"),
            s("48000"),
            s("-ac"),
            s("2"),
            s("out.webm"),
        ];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        let has_sr = parsed.outputs[0]
            .stream_options
            .iter()
            .any(|so| so.stream_type == StreamType::Audio && so.sample_rate == Some(48000));
        assert!(has_sr, "should have sample rate 48000");
        let has_ch = parsed.outputs[0]
            .stream_options
            .iter()
            .any(|so| so.stream_type == StreamType::Audio && so.channels == Some(2));
        assert!(has_ch, "should have channel count 2");
    }

    #[test]
    fn test_invalid_crf_returns_error() {
        let args = vec![
            s("-i"),
            s("in.mkv"),
            s("-crf"),
            s("not_a_number"),
            s("out.webm"),
        ];
        let result = FfmpegArgs::parse(&args);
        assert!(result.is_err(), "invalid CRF should produce an error");
    }

    #[test]
    fn test_loglevel_flag() {
        let args = vec![
            s("-loglevel"),
            s("quiet"),
            s("-i"),
            s("in.mkv"),
            s("out.webm"),
        ];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        assert_eq!(parsed.global_options.log_level.as_deref(), Some("quiet"));
    }

    #[test]
    fn test_pixel_format_flag() {
        let args = vec![
            s("-i"),
            s("in.mkv"),
            s("-pix_fmt"),
            s("yuv420p"),
            s("out.webm"),
        ];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        let has_pf = parsed.outputs[0].stream_options.iter().any(|so| {
            so.stream_type == StreamType::Video && so.pixel_fmt.as_deref() == Some("yuv420p")
        });
        assert!(has_pf, "should have pixel format yuv420p");
    }

    #[test]
    fn test_negative_map() {
        let args = vec![
            s("-i"),
            s("in.mkv"),
            s("-map"),
            s("0"),
            s("-map"),
            s("-0:a"),
            s("out.webm"),
        ];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        assert_eq!(parsed.outputs[0].map.len(), 2);
        let negative_map = parsed.outputs[0].map.iter().find(|m| m.negative);
        assert!(negative_map.is_some(), "should have a negative map entry");
    }

    #[test]
    fn test_hide_banner_no_effect() {
        // -hide_banner is silently accepted and has no semantic effect
        let args = vec![s("-hide_banner"), s("-i"), s("in.mkv"), s("out.webm")];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        assert_eq!(parsed.inputs.len(), 1);
        assert_eq!(parsed.outputs.len(), 1);
    }

    // ── Preset / Tune / Profile tests ───────────────────────────────────────

    #[test]
    fn test_preset_parsing() {
        let args = vec![
            s("-i"),
            s("in.mkv"),
            s("-preset"),
            s("medium"),
            s("out.webm"),
        ];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        assert_eq!(parsed.outputs[0].preset.as_deref(), Some("medium"));
    }

    #[test]
    fn test_tune_parsing() {
        let args = vec![s("-i"), s("in.mkv"), s("-tune"), s("film"), s("out.webm")];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        assert_eq!(parsed.outputs[0].tune.as_deref(), Some("film"));
    }

    #[test]
    fn test_profile_parsing() {
        let args = vec![
            s("-i"),
            s("in.mkv"),
            s("-profile:v"),
            s("main"),
            s("out.webm"),
        ];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        assert_eq!(parsed.outputs[0].profile.as_deref(), Some("main"));
    }

    #[test]
    fn test_preset_tune_profile_combined() {
        let args = vec![
            s("-i"),
            s("in.mkv"),
            s("-preset"),
            s("slow"),
            s("-tune"),
            s("grain"),
            s("-profile"),
            s("high"),
            s("-c:v"),
            s("libx264"),
            s("out.webm"),
        ];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        let out = &parsed.outputs[0];
        assert_eq!(out.preset.as_deref(), Some("slow"));
        assert_eq!(out.tune.as_deref(), Some("grain"));
        assert_eq!(out.profile.as_deref(), Some("high"));
    }

    // ── Two-pass encoding tests ─────────────────────────────────────────────

    #[test]
    fn test_two_pass_first_pass() {
        let args = vec![
            s("-i"),
            s("in.mkv"),
            s("-c:v"),
            s("libaom-av1"),
            s("-b:v"),
            s("2M"),
            s("-pass"),
            s("1"),
            s("-passlogfile"),
            tmp_str("ffpass"),
            s("out.webm"),
        ];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        let out = &parsed.outputs[0];
        assert_eq!(out.pass, Some(1));
        let expected = tmp_str("ffpass");
        assert_eq!(out.passlogfile.as_deref(), Some(expected.as_str()));
    }

    #[test]
    fn test_two_pass_second_pass() {
        let args = vec![
            s("-i"),
            s("in.mkv"),
            s("-c:v"),
            s("libaom-av1"),
            s("-b:v"),
            s("2M"),
            s("-pass"),
            s("2"),
            s("out.webm"),
        ];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        assert_eq!(parsed.outputs[0].pass, Some(2));
    }

    // ── Muxer options tests ─────────────────────────────────────────────────

    #[test]
    fn test_movflags_parsing() {
        let args = vec![
            s("-i"),
            s("in.mkv"),
            s("-movflags"),
            s("+faststart"),
            s("out.mp4"),
        ];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        let muxer = &parsed.outputs[0].muxer_options;
        assert_eq!(muxer.len(), 1);
        assert_eq!(muxer[0].0, "movflags");
        assert_eq!(muxer[0].1, "+faststart");
    }

    #[test]
    fn test_fflags_parsing() {
        let args = vec![
            s("-i"),
            s("in.mkv"),
            s("-fflags"),
            s("+genpts"),
            s("out.mp4"),
        ];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        let muxer = &parsed.outputs[0].muxer_options;
        assert_eq!(muxer.len(), 1);
        assert_eq!(muxer[0].0, "fflags");
        assert_eq!(muxer[0].1, "+genpts");
    }

    #[test]
    fn test_multiple_muxer_options() {
        let args = vec![
            s("-i"),
            s("in.mkv"),
            s("-movflags"),
            s("+faststart"),
            s("-fflags"),
            s("+genpts"),
            s("-brand"),
            s("mp42"),
            s("out.mp4"),
        ];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        let muxer = &parsed.outputs[0].muxer_options;
        assert_eq!(muxer.len(), 3);
    }

    #[test]
    fn test_to_position_parsing() {
        let args = vec![s("-i"), s("in.mkv"), s("-to"), s("00:05:00"), s("out.webm")];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        let to_entry = parsed.outputs[0]
            .extra_args
            .iter()
            .find(|(k, _)| k == "-to");
        assert!(to_entry.is_some());
        assert_eq!(to_entry.map(|(_, v)| v.as_str()), Some("00:05:00"));
    }
}
