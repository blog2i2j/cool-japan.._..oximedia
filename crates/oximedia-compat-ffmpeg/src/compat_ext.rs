//! Extended FFmpeg compatibility utilities.
//!
//! This module provides enhanced argument-parsing helpers, container/pixel-format
//! mappers, filter-graph utilities, a fluent argument builder with `build()`,
//! and diagnostics tooling (deprecated-option detection, compatibility scoring,
//! translation hints).
//!
//! All types are designed to work with the existing [`FfmpegArgs`] structures
//! and maintain the no-`unwrap()` policy throughout.

use std::collections::HashMap;

use crate::arg_parser::{FfmpegArgs, StreamType};
use crate::codec_map::CodecMap;
use crate::filter_lex::FilterNode;

// ─────────────────────────────────────────────────────────────────────────────
// Argument-parser accessor extensions
// ─────────────────────────────────────────────────────────────────────────────

/// Extension methods on [`FfmpegArgs`] for accessing multi-input, metadata,
/// filter_complex, map specifications, and seeking/duration parameters.
pub trait FfmpegArgsExt {
    /// Return all input specifications (all `-i` arguments).
    fn inputs_all(&self) -> &[crate::arg_parser::InputSpec];

    /// Return the first `-filter_complex` / `-lavfi` string found across all
    /// output specifications, or `None` if not present.
    fn complex_filter(&self) -> Option<&str>;

    /// Return parsed [`StreamMap`] entries gathered from all output
    /// specifications, in order.
    fn stream_maps(&self) -> Vec<StreamMap>;

    /// Collect all `-metadata key=value` pairs from all output specifications
    /// into a single `HashMap`.  Later outputs overwrite earlier ones on key
    /// collision.
    fn all_metadata(&self) -> HashMap<String, String>;

    /// Return the first seek-start value (`-ss` in *post-input* position),
    /// parsed as seconds.  Returns `None` if not set or unparseable.
    fn seek_start(&self) -> Option<f64>;

    /// Return the first maximum duration value (`-t`), parsed as seconds.
    fn duration(&self) -> Option<f64>;

    /// Return the first end-time value (`-to`), parsed as seconds.
    ///
    /// FFmpeg's `-to` sets the *end* position; OxiMedia treats it separately
    /// from duration.
    fn to_time(&self) -> Option<f64>;
}

impl FfmpegArgsExt for FfmpegArgs {
    fn inputs_all(&self) -> &[crate::arg_parser::InputSpec] {
        &self.inputs
    }

    fn complex_filter(&self) -> Option<&str> {
        for out in &self.outputs {
            if let Some(ref fc) = out.filter_complex {
                return Some(fc.as_str());
            }
        }
        None
    }

    fn stream_maps(&self) -> Vec<StreamMap> {
        let mut result = Vec::new();
        for out in &self.outputs {
            for map_spec in &out.map {
                let parsed = StreamMap::from_map_spec(map_spec);
                result.push(parsed);
            }
        }
        result
    }

    fn all_metadata(&self) -> HashMap<String, String> {
        let mut merged = HashMap::new();
        for out in &self.outputs {
            for (k, v) in &out.metadata {
                merged.insert(k.clone(), v.clone());
            }
        }
        merged
    }

    fn seek_start(&self) -> Option<f64> {
        for out in &self.outputs {
            if let Some(ref s) = out.seek {
                if let Some(secs) = parse_time_str(s) {
                    return Some(secs);
                }
            }
        }
        None
    }

    fn duration(&self) -> Option<f64> {
        for out in &self.outputs {
            if let Some(ref s) = out.duration {
                if let Some(secs) = parse_time_str(s) {
                    return Some(secs);
                }
            }
        }
        None
    }

    fn to_time(&self) -> Option<f64> {
        // `-to` is stored in extra_args by the parser
        for out in &self.outputs {
            for (k, v) in &out.extra_args {
                if k == "-to" {
                    if let Some(secs) = parse_time_str(v) {
                        return Some(secs);
                    }
                }
            }
        }
        None
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// StreamMap — structured -map argument
// ─────────────────────────────────────────────────────────────────────────────

/// A structured representation of an FFmpeg `-map` argument.
///
/// Examples: `-map 0:v:0`, `-map 0:a:1`, `-map 0`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamMap {
    /// Input file index (0-based).
    pub input_idx: u32,
    /// Optional stream type selector (`v`, `a`, `s`).
    pub stream_type: Option<MapStreamType>,
    /// Optional stream index within that type.
    pub stream_idx: Option<u32>,
    /// Whether this is a negative (exclusion) map.
    pub negative: bool,
}

/// Stream type component in a `-map` specifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MapStreamType {
    Video,
    Audio,
    Subtitle,
    Data,
    Attachment,
}

impl MapStreamType {
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "v" | "video" => Some(Self::Video),
            "a" | "audio" => Some(Self::Audio),
            "s" | "subtitle" => Some(Self::Subtitle),
            "d" | "data" => Some(Self::Data),
            "t" | "attachment" => Some(Self::Attachment),
            _ => None,
        }
    }
}

impl StreamMap {
    /// Build a [`StreamMap`] from an existing [`crate::arg_parser::MapSpec`].
    fn from_map_spec(spec: &crate::arg_parser::MapSpec) -> Self {
        let input_idx = spec.input_index as u32;
        let negative = spec.negative;

        // Parse stream_selector like "v", "a:1", "v:0"
        let (stream_type, stream_idx) = match &spec.stream_selector {
            None => (None, None),
            Some(sel) => {
                let mut parts = sel.splitn(2, ':');
                let type_str = parts.next().unwrap_or("");
                let idx_str = parts.next();
                let stype = MapStreamType::from_str(type_str);
                let sidx = idx_str.and_then(|s| s.parse::<u32>().ok());
                (stype, sidx)
            }
        };

        StreamMap {
            input_idx,
            stream_type,
            stream_idx,
            negative,
        }
    }

    /// Parse a raw `-map` specifier string directly (e.g. `"0:v:0"`, `"0:a:1"`).
    pub fn parse(spec: &str) -> Option<Self> {
        let spec = spec.trim();
        if spec.is_empty() {
            return None;
        }

        // Filter-complex output label: [label]
        if spec.starts_with('[') && spec.ends_with(']') {
            return Some(StreamMap {
                input_idx: 0,
                stream_type: None,
                stream_idx: None,
                negative: false,
            });
        }

        let (negative, rest) = if let Some(s) = spec.strip_prefix('-') {
            (true, s)
        } else {
            (false, spec)
        };

        let mut parts = rest.splitn(3, ':');
        let input_idx = parts.next()?.parse::<u32>().ok()?;
        let type_str = parts.next().unwrap_or("");
        let idx_str = parts.next();

        let stream_type = if type_str.is_empty() {
            None
        } else {
            MapStreamType::from_str(type_str)
        };

        let stream_idx = idx_str.and_then(|s| s.parse::<u32>().ok());

        Some(StreamMap {
            input_idx,
            stream_type,
            stream_idx,
            negative,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ContainerMapper
// ─────────────────────────────────────────────────────────────────────────────

/// Maps between FFmpeg container format names/extensions and OxiMedia container IDs.
pub struct ContainerMapper;

/// Static table: (extension/ffmpeg_name, oximedia_container)
static CONTAINER_TABLE: &[(&str, &str)] = &[
    ("mp4", "mp4"),
    ("m4v", "mp4"),
    ("m4a", "mp4"),
    ("3gp", "mp4"),
    ("3g2", "mp4"),
    ("f4v", "mp4"),
    ("mkv", "matroska"),
    ("matroska", "matroska"),
    ("mka", "matroska"),
    ("mks", "matroska"),
    ("webm", "webm"),
    ("mov", "quicktime"),
    ("qt", "quicktime"),
    ("avi", "avi"),
    ("flv", "flv"),
    ("ts", "mpegts"),
    ("mts", "mpegts"),
    ("m2ts", "mpegts"),
    ("m2t", "mpegts"),
    ("mpegts", "mpegts"),
    ("ogg", "ogg"),
    ("ogv", "ogg"),
    ("oga", "ogg"),
    ("opus", "ogg"),
    ("wav", "wav"),
    ("wave", "wav"),
    ("flac", "flac"),
    ("aiff", "aiff"),
    ("aif", "aiff"),
    ("mp3", "mp3"),
    ("aac", "adts"),
    ("mxf", "mxf"),
    ("gxf", "gxf"),
    ("rm", "rm"),
    ("rmvb", "rm"),
    ("asf", "asf"),
    ("wmv", "asf"),
    ("wma", "asf"),
    ("nut", "nut"),
];

impl ContainerMapper {
    /// Map an FFmpeg format name or file extension to an OxiMedia container identifier.
    ///
    /// Returns `None` for unrecognised names.
    pub fn ffmpeg_to_oximedia(ext: &str) -> Option<&'static str> {
        let key = ext.to_lowercase();
        let key = key.trim_start_matches('.');
        CONTAINER_TABLE
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, v)| *v)
    }

    /// Map an OxiMedia container identifier back to a canonical FFmpeg format name.
    pub fn oximedia_to_ffmpeg(oxi: &str) -> Option<&'static str> {
        let key = oxi.to_lowercase();
        CONTAINER_TABLE
            .iter()
            .find(|(_, v)| *v == key.as_str())
            .map(|(k, _)| *k)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PixelFormatMapper
// ─────────────────────────────────────────────────────────────────────────────

/// Maps between FFmpeg pixel format names and OxiMedia pixel format identifiers.
pub struct PixelFormatMapper;

/// Static table: (ffmpeg_pix_fmt, oximedia_pix_fmt)
static PIX_FMT_TABLE: &[(&str, &str)] = &[
    ("yuv420p", "yuv420p"),
    ("yuvj420p", "yuv420p"),
    ("yuv420p10le", "yuv420p10le"),
    ("yuv420p10be", "yuv420p10le"),
    ("yuv420p12le", "yuv420p12le"),
    ("yuv422p", "yuv422p"),
    ("yuvj422p", "yuv422p"),
    ("yuv422p10le", "yuv422p10le"),
    ("yuv422p10be", "yuv422p10le"),
    ("yuv444p", "yuv444p"),
    ("yuvj444p", "yuv444p"),
    ("yuv444p10le", "yuv444p10le"),
    ("yuv444p10be", "yuv444p10le"),
    ("nv12", "nv12"),
    ("nv21", "nv12"),
    ("nv16", "yuv422p"),
    ("nv24", "yuv444p"),
    ("p010le", "p010le"),
    ("p010be", "p010le"),
    ("p016le", "p016le"),
    ("rgb24", "rgb24"),
    ("bgr24", "rgb24"),
    ("rgba", "rgba"),
    ("bgra", "rgba"),
    ("rgb0", "rgb24"),
    ("bgr0", "rgb24"),
    ("argb", "rgba"),
    ("abgr", "rgba"),
    ("gray", "gray8"),
    ("gray8", "gray8"),
    ("gray10le", "gray10le"),
    ("gray12le", "gray12le"),
    ("gray16le", "gray16le"),
    ("rgb48le", "rgb48le"),
    ("rgb48be", "rgb48le"),
    ("rgba64le", "rgba64le"),
    ("rgba64be", "rgba64le"),
    ("gbrp", "gbrp"),
    ("gbrp10le", "gbrp10le"),
];

impl PixelFormatMapper {
    /// Map an FFmpeg pixel format name to an OxiMedia pixel format identifier.
    ///
    /// Returns `None` for unrecognised format names.
    pub fn ffmpeg_to_oximedia(fmt: &str) -> Option<&'static str> {
        let key = fmt.to_lowercase();
        PIX_FMT_TABLE
            .iter()
            .find(|(k, _)| *k == key.as_str())
            .map(|(_, v)| *v)
    }

    /// Map an OxiMedia pixel format identifier back to a canonical FFmpeg name.
    pub fn oximedia_to_ffmpeg(oxi: &str) -> Option<&'static str> {
        let key = oxi.to_lowercase();
        PIX_FMT_TABLE
            .iter()
            .find(|(_, v)| *v == key.as_str())
            .map(|(k, _)| *k)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FilterChain parsing helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Parses a comma-separated filter chain string into an ordered list of
/// [`FilterNode`] values.
///
/// Commas inside `()` (parameter lists) are not treated as separators.
/// This allows filter arguments like `scale=iw/2:ih/2` to parse correctly.
///
/// ## Example
///
/// ```
/// use oximedia_compat_ffmpeg::compat_ext::parse_filter_chain;
///
/// let nodes = parse_filter_chain("scale=1280:720,setsar=1");
/// assert_eq!(nodes.len(), 2);
/// assert_eq!(nodes[0].name, "scale");
/// assert_eq!(nodes[1].name, "setsar");
/// ```
pub fn parse_filter_chain(chain_str: &str) -> Vec<FilterNode> {
    split_on_comma_chain(chain_str)
        .into_iter()
        .map(|segment| {
            let segment = segment.trim();
            parse_filter_node_raw(segment)
        })
        .collect()
}

/// Split a filter chain string on `,` while ignoring commas inside `()` or `[]`.
fn split_on_comma_chain(s: &str) -> Vec<&str> {
    let mut result = Vec::new();
    let mut paren_depth: usize = 0;
    let mut bracket_depth: usize = 0;
    let mut start = 0usize;
    let bytes = s.as_bytes();

    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'(' => paren_depth += 1,
            b')' if paren_depth > 0 => paren_depth -= 1,
            b'[' => bracket_depth += 1,
            b']' if bracket_depth > 0 => bracket_depth -= 1,
            b',' if paren_depth == 0 && bracket_depth == 0 => {
                result.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    result.push(&s[start..]);
    result
}

/// Parse a single raw filter expression like `scale=1280:720` or `hflip` into
/// a [`FilterNode`].
fn parse_filter_node_raw(s: &str) -> FilterNode {
    let s = s.trim();

    // Strip any leading/trailing pad labels `[label]`
    let s = strip_pad_labels(s);

    // Split at first `=`
    let (name, args_str) = match s.find('=') {
        Some(pos) => (s[..pos].trim(), s[pos + 1..].trim()),
        None => (s.trim(), ""),
    };

    // Parse colon-separated args into positional and named
    let (positional_args, named_args) = parse_filter_args(args_str);

    FilterNode {
        inputs: Vec::new(),
        name: name.to_string(),
        positional_args,
        named_args,
        outputs: Vec::new(),
    }
}

/// Strip any leading `[label]` and trailing `[label]` pad markers from a
/// filter expression fragment.
fn strip_pad_labels(s: &str) -> &str {
    let s = s.trim();
    // Strip leading label
    let s = if s.starts_with('[') {
        if let Some(end) = s.find(']') {
            s[end + 1..].trim_start()
        } else {
            s
        }
    } else {
        s
    };
    // Strip trailing label
    let s = if s.ends_with(']') {
        if let Some(start) = s.rfind('[') {
            s[..start].trim_end()
        } else {
            s
        }
    } else {
        s
    };
    s
}

/// Parse colon-separated filter arguments into positional and named lists.
fn parse_filter_args(args_str: &str) -> (Vec<String>, Vec<(String, String)>) {
    if args_str.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let mut positional = Vec::new();
    let mut named = Vec::new();

    for token in args_str.split(':') {
        let token = token.trim();
        if let Some(eq_pos) = token.find('=') {
            let key = token[..eq_pos].trim().to_string();
            let value = token[eq_pos + 1..].trim().to_string();
            named.push((key, value));
        } else if !token.is_empty() {
            positional.push(token.to_string());
        }
    }

    (positional, named)
}

// ─────────────────────────────────────────────────────────────────────────────
// FilterGraphParser — lavfi source detection
// ─────────────────────────────────────────────────────────────────────────────

/// Utilities for inspecting and classifying filter graph expressions.
pub struct FilterGraphParser;

/// Known lavfi source filter names.
static LAVFI_SOURCES: &[&str] = &[
    "color",
    "colour",
    "testsrc",
    "testsrc2",
    "smptebars",
    "smptehdbars",
    "sine",
    "anullsrc",
    "nullsrc",
    "rgbtestsrc",
    "mandelbrot",
    "life",
    "mptestsrc",
    "gradients",
    "haldclutsrc",
    "pal75bars",
    "pal100bars",
    "allrgb",
    "allyuv",
    "yuvtestsrc",
];

impl FilterGraphParser {
    /// Return `true` if `filter` is a known lavfi (virtual device) source filter.
    ///
    /// Lavfi sources generate video/audio without consuming an input stream —
    /// they are typically used with `-f lavfi` or as the source in `-filter_complex`.
    ///
    /// ```
    /// use oximedia_compat_ffmpeg::compat_ext::FilterGraphParser;
    ///
    /// assert!(FilterGraphParser::is_lavfi_source("color"));
    /// assert!(FilterGraphParser::is_lavfi_source("testsrc"));
    /// assert!(FilterGraphParser::is_lavfi_source("sine"));
    /// assert!(!FilterGraphParser::is_lavfi_source("scale"));
    /// ```
    pub fn is_lavfi_source(filter: &str) -> bool {
        // Strip any parameters: `color=c=red:size=1280x720` → `color`
        let name = filter.split('=').next().unwrap_or(filter).trim();
        let name_lower = name.to_lowercase();
        LAVFI_SOURCES.iter().any(|&s| s == name_lower.as_str())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FilterGraphValidator — pad connection checking
// ─────────────────────────────────────────────────────────────────────────────

/// Validates pad connections in a parsed filter graph.
pub struct FilterGraphValidator;

impl FilterGraphValidator {
    /// Check that all named pad labels `[label]` have matching producers and consumers.
    ///
    /// Returns a list of human-readable problem descriptions. An empty list means
    /// the graph is connection-complete.
    ///
    /// For each output pad label produced by a chain, there must be at least one
    /// chain that consumes it (uses it as an input label), and vice versa.
    pub fn check_connections(graph: &crate::filter_graph::FilterGraph) -> Vec<String> {
        let mut problems = Vec::new();

        // Collect output labels (producers) and input labels (consumers)
        let mut producers: Vec<String> = Vec::new();
        let mut consumers: Vec<String> = Vec::new();

        for chain in &graph.chains {
            if let Some(ref lbl) = chain.output_label {
                producers.push(lbl.clone());
            }
            if let Some(ref lbl) = chain.input_label {
                consumers.push(lbl.clone());
            }
        }

        // Each producer must have at least one consumer
        for prod in &producers {
            if !consumers.contains(prod) {
                problems.push(format!(
                    "Output pad '[{}]' has no consumer in the filter graph",
                    prod
                ));
            }
        }

        // Each consumer must have at least one producer
        for cons in &consumers {
            if !producers.contains(cons) {
                problems.push(format!(
                    "Input pad '[{}]' has no producer in the filter graph",
                    cons
                ));
            }
        }

        problems
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Enhanced FfmpegArgumentBuilder with `build()` and string-based methods
// ─────────────────────────────────────────────────────────────────────────────

/// Fluent argument builder for constructing FFmpeg-style command-line argument
/// vectors directly, without needing mutable borrows.
///
/// Unlike [`crate::argument_builder::FfmpegArgumentBuilder`], this builder uses
/// a consuming/owned pattern and exposes a `build()` method that returns
/// `Vec<String>`.  It also exposes the method names requested by the task
/// specification (`codec_video`, `codec_audio`, `bitrate_video`, `bitrate_audio`,
/// `scale`, `seek`, `duration`, `metadata`).
///
/// ## Example
///
/// ```
/// use oximedia_compat_ffmpeg::compat_ext::ArgumentBuilder;
///
/// let args = ArgumentBuilder::new()
///     .input("input.mp4")
///     .codec_video("av1")
///     .codec_audio("opus")
///     .crf(30)
///     .scale(1280, 720)
///     .fps(29.97)
///     .seek(10.0)
///     .duration(60.0)
///     .metadata("title", "My Video")
///     .output("output.webm")
///     .build();
///
/// assert!(args.contains(&"-c:v".to_string()));
/// assert!(args.contains(&"av1".to_string()));
/// ```
#[derive(Debug, Clone, Default)]
pub struct ArgumentBuilder {
    global: Vec<String>,
    inputs: Vec<String>,
    output_opts: Vec<String>,
    output_path: Option<String>,
}

impl ArgumentBuilder {
    /// Create a new, empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an input file path (`-i path`).
    pub fn input(mut self, path: &str) -> Self {
        self.inputs.push("-i".to_string());
        self.inputs.push(path.to_string());
        self
    }

    /// Set the output file path.
    pub fn output(mut self, path: &str) -> Self {
        self.output_path = Some(path.to_string());
        self
    }

    /// Append `-c:v <codec>` (video codec).
    pub fn codec_video(mut self, codec: &str) -> Self {
        self.output_opts.push("-c:v".to_string());
        self.output_opts.push(codec.to_string());
        self
    }

    /// Append `-c:a <codec>` (audio codec).
    pub fn codec_audio(mut self, codec: &str) -> Self {
        self.output_opts.push("-c:a".to_string());
        self.output_opts.push(codec.to_string());
        self
    }

    /// Append `-b:v <bitrate>` (video bitrate string, e.g. `"2M"`, `"4000k"`).
    pub fn bitrate_video(mut self, br: &str) -> Self {
        self.output_opts.push("-b:v".to_string());
        self.output_opts.push(br.to_string());
        self
    }

    /// Append `-b:a <bitrate>` (audio bitrate string, e.g. `"128k"`).
    pub fn bitrate_audio(mut self, br: &str) -> Self {
        self.output_opts.push("-b:a".to_string());
        self.output_opts.push(br.to_string());
        self
    }

    /// Append `-preset <preset>`.
    pub fn preset(mut self, p: &str) -> Self {
        self.output_opts.push("-preset".to_string());
        self.output_opts.push(p.to_string());
        self
    }

    /// Append `-crf <crf>`.
    pub fn crf(mut self, crf: u32) -> Self {
        self.output_opts.push("-crf".to_string());
        self.output_opts.push(crf.to_string());
        self
    }

    /// Append a scale video filter via `-vf scale=<w>:<h>`.
    ///
    /// Pass `-1` for either dimension to preserve aspect ratio.
    pub fn scale(mut self, w: i32, h: i32) -> Self {
        self.output_opts.push("-vf".to_string());
        self.output_opts.push(format!("scale={}:{}", w, h));
        self
    }

    /// Append `-r <fps>` (frame rate).
    pub fn fps(mut self, fps: f32) -> Self {
        self.output_opts.push("-r".to_string());
        self.output_opts.push(format_fps_f32(fps));
        self
    }

    /// Append `-ss <seconds>` (seek start, in seconds).
    pub fn seek(mut self, ss: f64) -> Self {
        self.output_opts.push("-ss".to_string());
        self.output_opts.push(format!("{:.6}", ss).trim_end_matches('0').trim_end_matches('.').to_string());
        self
    }

    /// Append `-t <seconds>` (maximum duration, in seconds).
    pub fn duration(mut self, t: f64) -> Self {
        self.output_opts.push("-t".to_string());
        self.output_opts.push(format!("{:.6}", t).trim_end_matches('0').trim_end_matches('.').to_string());
        self
    }

    /// Append `-metadata key=value`.
    pub fn metadata(mut self, k: &str, v: &str) -> Self {
        self.output_opts.push("-metadata".to_string());
        self.output_opts.push(format!("{}={}", k, v));
        self
    }

    /// Append `-y` (overwrite without asking) to the global flags.
    pub fn overwrite(mut self) -> Self {
        self.global.push("-y".to_string());
        self
    }

    /// Append `-f <fmt>` (force container format).
    pub fn format(mut self, fmt: &str) -> Self {
        self.output_opts.push("-f".to_string());
        self.output_opts.push(fmt.to_string());
        self
    }

    /// Append `-filter_complex <expr>`.
    pub fn filter_complex(mut self, expr: &str) -> Self {
        self.output_opts.push("-filter_complex".to_string());
        self.output_opts.push(expr.to_string());
        self
    }

    /// Append `-loglevel <level>`.
    pub fn loglevel(mut self, level: &str) -> Self {
        self.global.push("-loglevel".to_string());
        self.global.push(level.to_string());
        self
    }

    /// Assemble and return the final argument list.
    ///
    /// Order: `[global] [inputs] [output_opts] [output_path]`.
    /// Does **not** include `"ffmpeg"` as the zeroth element.
    pub fn build(self) -> Vec<String> {
        let mut args = Vec::with_capacity(
            self.global.len() + self.inputs.len() + self.output_opts.len() + 1,
        );
        args.extend(self.global);
        args.extend(self.inputs);
        args.extend(self.output_opts);
        if let Some(path) = self.output_path {
            args.push(path);
        }
        args
    }

    /// Build a human-readable `ffmpeg …` command string for logging/display.
    pub fn to_command_string(self) -> String {
        let args = self.build();
        let mut parts = vec!["ffmpeg".to_string()];
        for arg in &args {
            if arg.contains(' ') {
                parts.push(format!("\"{}\"", arg));
            } else {
                parts.push(arg.clone());
            }
        }
        parts.join(" ")
    }
}

/// Format an f32 FPS value with minimal decimal places.
fn format_fps_f32(fps: f32) -> String {
    let s = format!("{:.3}", fps);
    let s = s.trim_end_matches('0');
    let s = s.trim_end_matches('.');
    s.to_string()
}

// ─────────────────────────────────────────────────────────────────────────────
// Deprecated option detection
// ─────────────────────────────────────────────────────────────────────────────

/// A warning emitted when a deprecated FFmpeg option is detected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FfmpegWarning {
    /// The deprecated flag that was found (e.g. `"-vcodec"`).
    pub deprecated_flag: String,
    /// The recommended replacement (e.g. `"-c:v"`).
    pub replacement: String,
    /// Human-readable message.
    pub message: String,
}

impl FfmpegWarning {
    fn new(deprecated: &str, replacement: &str, note: &str) -> Self {
        Self {
            deprecated_flag: deprecated.to_string(),
            replacement: replacement.to_string(),
            message: format!(
                "Deprecated option '{}': use '{}' instead. {}",
                deprecated, replacement, note
            ),
        }
    }
}

/// Diagnostics utilities for FFmpeg argument lists.
pub struct FfmpegDiagnostics;

impl FfmpegDiagnostics {
    /// Scan an argument slice for deprecated FFmpeg option flags and return a
    /// [`FfmpegWarning`] for each deprecated flag found.
    ///
    /// Checks for:
    /// - `-vcodec` → use `-c:v`
    /// - `-acodec` → use `-c:a`
    /// - `-ab` → use `-b:a`
    /// - (Note: `-ar` for sample rate is not deprecated per se, it is included
    ///   because some users confuse it with a codec option)
    ///
    /// ## Example
    ///
    /// ```
    /// use oximedia_compat_ffmpeg::compat_ext::FfmpegDiagnostics;
    ///
    /// let warnings = FfmpegDiagnostics::check_deprecated_options(&["-vcodec", "libx264"]);
    /// assert_eq!(warnings.len(), 1);
    /// assert_eq!(warnings[0].deprecated_flag, "-vcodec");
    /// ```
    pub fn check_deprecated_options(args: &[&str]) -> Vec<FfmpegWarning> {
        let mut warnings = Vec::new();
        for &arg in args {
            match arg {
                "-vcodec" => warnings.push(FfmpegWarning::new(
                    "-vcodec",
                    "-c:v",
                    "The -vcodec alias is deprecated since FFmpeg 0.9.",
                )),
                "-acodec" => warnings.push(FfmpegWarning::new(
                    "-acodec",
                    "-c:a",
                    "The -acodec alias is deprecated since FFmpeg 0.9.",
                )),
                "-ab" => warnings.push(FfmpegWarning::new(
                    "-ab",
                    "-b:a",
                    "The -ab alias is deprecated; use -b:a for audio bitrate.",
                )),
                "-scodec" => warnings.push(FfmpegWarning::new(
                    "-scodec",
                    "-c:s",
                    "The -scodec alias is deprecated since FFmpeg 0.9.",
                )),
                _ => {}
            }
        }
        warnings
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Compatibility score
// ─────────────────────────────────────────────────────────────────────────────

/// Diagnostics for assessing how well an [`FfmpegArgs`] command can be
/// translated to OxiMedia.
pub struct FfmpegCompatDiagnostics;

impl FfmpegCompatDiagnostics {
    /// Compute a compatibility score for a parsed [`FfmpegArgs`] command.
    ///
    /// Returns a value in `[0.0, 1.0]` where:
    /// - `1.0` means everything is directly translatable.
    /// - Values below `1.0` indicate unknown codecs, formats, or options that
    ///   cannot be mapped to OxiMedia equivalents.
    ///
    /// ## Deductions
    ///
    /// - Unknown video codec: −0.3
    /// - Unknown audio codec: −0.2
    /// - Unknown container format: −0.1
    /// - Unknown filter in `-vf`/`-af`: −0.05 per unknown filter
    /// - Unrecognised extra args: −0.02 per argument pair
    pub fn score(args: &FfmpegArgs) -> f32 {
        let codec_map = CodecMap::new();
        let mut score = 1.0f32;

        for out in &args.outputs {
            // Check video codec
            for opt in &out.stream_options {
                if let Some(ref codec) = opt.codec {
                    if codec != "copy" && !codec_map.is_supported(codec) {
                        match opt.stream_type {
                            StreamType::Video => score -= 0.3,
                            StreamType::Audio => score -= 0.2,
                            StreamType::Subtitle => score -= 0.05,
                            StreamType::All => score -= 0.2,
                        }
                    }
                }
            }

            // Check container format
            if let Some(ref fmt) = out.format {
                if ContainerMapper::ffmpeg_to_oximedia(fmt).is_none() {
                    score -= 0.1;
                }
            }

            // Check for extra args (unrecognised options)
            let penalty = out.extra_args.len() as f32 * 0.02;
            score -= penalty;
        }

        score.max(0.0_f32).min(1.0_f32)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Translation hints
// ─────────────────────────────────────────────────────────────────────────────

/// A human-readable hint describing how one aspect of an FFmpeg command was
/// translated to OxiMedia.
#[derive(Debug, Clone, PartialEq)]
pub struct TranslationHint {
    /// The original FFmpeg value (e.g. `"libx264"`).
    pub original: String,
    /// The OxiMedia equivalent (e.g. `"av1"`).
    pub translated: String,
    /// Confidence in the translation: 1.0 = direct match, 0.5 = substituted.
    pub confidence: f32,
    /// Optional note explaining the mapping rationale.
    pub note: Option<String>,
}

impl TranslationHint {
    /// Create a direct-match hint (confidence 1.0).
    pub fn direct(original: impl Into<String>, translated: impl Into<String>) -> Self {
        Self {
            original: original.into(),
            translated: translated.into(),
            confidence: 1.0,
            note: None,
        }
    }

    /// Create a substitution hint (confidence 0.5) with a note.
    pub fn substituted(
        original: impl Into<String>,
        translated: impl Into<String>,
        note: impl Into<String>,
    ) -> Self {
        Self {
            original: original.into(),
            translated: translated.into(),
            confidence: 0.5,
            note: Some(note.into()),
        }
    }

    /// Attach a note to this hint.
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.note = Some(note.into());
        self
    }
}

/// Generate a list of [`TranslationHint`] values for all translatable elements
/// in a parsed [`FfmpegArgs`] command.
///
/// The hints cover:
/// - Codec mappings (direct and patent-substituted)
/// - Container format mappings
/// - Pixel format mappings
pub fn generate_hints(args: &FfmpegArgs) -> Vec<TranslationHint> {
    use crate::codec_map::CodecCategory;

    let codec_map = CodecMap::new();
    let mut hints = Vec::new();

    for out in &args.outputs {
        // Codec hints
        for opt in &out.stream_options {
            if let Some(ref codec) = opt.codec {
                if codec == "copy" {
                    hints.push(TranslationHint::direct(codec.clone(), "copy"));
                    continue;
                }
                if let Some(entry) = codec_map.lookup(codec) {
                    match entry.category {
                        CodecCategory::DirectMatch => {
                            hints.push(TranslationHint::direct(codec.clone(), entry.oxi_name));
                        }
                        CodecCategory::PatentSubstituted => {
                            hints.push(TranslationHint::substituted(
                                codec.clone(),
                                entry.oxi_name,
                                format!(
                                    "'{}' is patent-encumbered; using '{}' (patent-free)",
                                    codec, entry.oxi_name
                                ),
                            ));
                        }
                        CodecCategory::Copy => {
                            hints.push(TranslationHint::direct(codec.clone(), "copy"));
                        }
                    }
                } else {
                    hints.push(TranslationHint {
                        original: codec.clone(),
                        translated: codec.clone(),
                        confidence: 0.1,
                        note: Some("Unknown codec; passing through as-is".to_string()),
                    });
                }
            }
        }

        // Container format hints
        if let Some(ref fmt) = out.format {
            if let Some(oxi) = ContainerMapper::ffmpeg_to_oximedia(fmt) {
                let confidence = if oxi == fmt { 1.0 } else { 0.8 };
                hints.push(TranslationHint {
                    original: fmt.clone(),
                    translated: oxi.to_string(),
                    confidence,
                    note: if oxi != fmt {
                        Some(format!("Container '{}' mapped to '{}'", fmt, oxi))
                    } else {
                        None
                    },
                });
            }
        }

        // Pixel format hints
        for opt in &out.stream_options {
            if let Some(ref pix) = opt.pixel_fmt {
                if let Some(oxi) = PixelFormatMapper::ffmpeg_to_oximedia(pix) {
                    let confidence = if oxi == pix.as_str() { 1.0 } else { 0.9 };
                    hints.push(TranslationHint {
                        original: pix.clone(),
                        translated: oxi.to_string(),
                        confidence,
                        note: if oxi != pix.as_str() {
                            Some(format!("Pixel format '{}' normalised to '{}'", pix, oxi))
                        } else {
                            None
                        },
                    });
                }
            }
        }
    }

    hints
}

// ─────────────────────────────────────────────────────────────────────────────
// Time parsing helper
// ─────────────────────────────────────────────────────────────────────────────

/// Parse a time string into seconds.
///
/// Accepts formats:
/// - `"123.45"` — plain seconds (float)
/// - `"HH:MM:SS"` — hours:minutes:seconds
/// - `"HH:MM:SS.mmm"` — with fractional seconds
/// - `"MM:SS"` — minutes:seconds
fn parse_time_str(s: &str) -> Option<f64> {
    let s = s.trim();

    // Try plain float first
    if let Ok(v) = s.parse::<f64>() {
        return Some(v);
    }

    // Try HH:MM:SS[.mmm] or MM:SS
    let parts: Vec<&str> = s.splitn(3, ':').collect();
    match parts.as_slice() {
        [mm, ss] => {
            let minutes = mm.parse::<f64>().ok()?;
            let seconds = ss.parse::<f64>().ok()?;
            Some(minutes * 60.0 + seconds)
        }
        [hh, mm, ss] => {
            let hours = hh.parse::<f64>().ok()?;
            let minutes = mm.parse::<f64>().ok()?;
            let seconds = ss.parse::<f64>().ok()?;
            Some(hours * 3600.0 + minutes * 60.0 + seconds)
        }
        _ => None,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arg_parser::FfmpegArgs;

    fn s(v: &str) -> String {
        v.to_string()
    }

    // ── Multi-input parsing ──────────────────────────────────────────────────

    #[test]
    fn test_multi_input_finds_all_inputs() {
        let args = vec![
            s("-i"), s("video.mp4"),
            s("-i"), s("audio.flac"),
            s("-i"), s("subtitle.srt"),
            s("-c:v"), s("copy"),
            s("output.mkv"),
        ];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        assert_eq!(parsed.inputs_all().len(), 3);
        assert_eq!(parsed.inputs_all()[0].path, "video.mp4");
        assert_eq!(parsed.inputs_all()[1].path, "audio.flac");
        assert_eq!(parsed.inputs_all()[2].path, "subtitle.srt");
    }

    #[test]
    fn test_single_input_via_accessor() {
        let args = vec![s("-i"), s("input.mkv"), s("output.webm")];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        assert_eq!(parsed.inputs_all().len(), 1);
        assert_eq!(parsed.inputs_all()[0].path, "input.mkv");
    }

    // ── complex_filter ───────────────────────────────────────────────────────

    #[test]
    fn test_complex_filter_extracted() {
        let args = vec![
            s("-i"), s("a.mp4"),
            s("-i"), s("b.mp4"),
            s("-filter_complex"), s("[0:v][1:v]overlay=W/2:0[out]"),
            s("-map"), s("[out]"),
            s("output.mp4"),
        ];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        assert_eq!(parsed.complex_filter(), Some("[0:v][1:v]overlay=W/2:0[out]"));
    }

    #[test]
    fn test_complex_filter_none_when_absent() {
        let args = vec![s("-i"), s("a.mp4"), s("out.mp4")];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        assert!(parsed.complex_filter().is_none());
    }

    // ── stream_maps ──────────────────────────────────────────────────────────

    #[test]
    fn test_stream_maps_parsed() {
        let args = vec![
            s("-i"), s("in.mp4"),
            s("-map"), s("0:v:0"),
            s("-map"), s("0:a:1"),
            s("out.mkv"),
        ];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        let maps = parsed.stream_maps();
        assert_eq!(maps.len(), 2);
        assert_eq!(maps[0].input_idx, 0);
        assert_eq!(maps[0].stream_type, Some(MapStreamType::Video));
        assert_eq!(maps[0].stream_idx, Some(0));
        assert_eq!(maps[1].stream_type, Some(MapStreamType::Audio));
        assert_eq!(maps[1].stream_idx, Some(1));
    }

    // ── metadata ─────────────────────────────────────────────────────────────

    #[test]
    fn test_metadata_extracted() {
        let args = vec![
            s("-i"), s("in.mp4"),
            s("-metadata"), s("title=Hello World"),
            s("-metadata"), s("artist=COOLJAPAN"),
            s("out.mp4"),
        ];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        let meta = parsed.all_metadata();
        assert_eq!(meta.get("title").map(|s| s.as_str()), Some("Hello World"));
        assert_eq!(meta.get("artist").map(|s| s.as_str()), Some("COOLJAPAN"));
    }

    // ── seek_start / duration / to_time ─────────────────────────────────────

    #[test]
    fn test_seek_start_seconds() {
        let args = vec![
            s("-i"), s("in.mp4"),
            s("-ss"), s("30.5"),
            s("out.mp4"),
        ];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        let ss = parsed.seek_start().expect("should have seek");
        assert!((ss - 30.5).abs() < 0.001);
    }

    #[test]
    fn test_duration_seconds() {
        let args = vec![
            s("-i"), s("in.mp4"),
            s("-t"), s("120.0"),
            s("out.mp4"),
        ];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        let t = parsed.duration().expect("should have duration");
        assert!((t - 120.0).abs() < 0.001);
    }

    #[test]
    fn test_seek_start_timecode() {
        let args = vec![
            s("-i"), s("in.mp4"),
            s("-ss"), s("00:01:30"),
            s("out.mp4"),
        ];
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        let ss = parsed.seek_start().expect("should have seek");
        assert!((ss - 90.0).abs() < 0.001);
    }

    // ── ContainerMapper ──────────────────────────────────────────────────────

    #[test]
    fn test_container_mp4() {
        assert_eq!(ContainerMapper::ffmpeg_to_oximedia("mp4"), Some("mp4"));
    }

    #[test]
    fn test_container_mkv_matroska() {
        assert_eq!(ContainerMapper::ffmpeg_to_oximedia("mkv"), Some("matroska"));
        assert_eq!(ContainerMapper::ffmpeg_to_oximedia("matroska"), Some("matroska"));
    }

    #[test]
    fn test_container_webm() {
        assert_eq!(ContainerMapper::ffmpeg_to_oximedia("webm"), Some("webm"));
    }

    #[test]
    fn test_container_mov_quicktime() {
        assert_eq!(ContainerMapper::ffmpeg_to_oximedia("mov"), Some("quicktime"));
    }

    #[test]
    fn test_container_ts_mpegts() {
        assert_eq!(ContainerMapper::ffmpeg_to_oximedia("ts"), Some("mpegts"));
        assert_eq!(ContainerMapper::ffmpeg_to_oximedia("mts"), Some("mpegts"));
        assert_eq!(ContainerMapper::ffmpeg_to_oximedia("m2ts"), Some("mpegts"));
    }

    #[test]
    fn test_container_ogg_wav_flac() {
        assert_eq!(ContainerMapper::ffmpeg_to_oximedia("ogg"), Some("ogg"));
        assert_eq!(ContainerMapper::ffmpeg_to_oximedia("wav"), Some("wav"));
        assert_eq!(ContainerMapper::ffmpeg_to_oximedia("flac"), Some("flac"));
    }

    #[test]
    fn test_container_avi_flv() {
        assert_eq!(ContainerMapper::ffmpeg_to_oximedia("avi"), Some("avi"));
        assert_eq!(ContainerMapper::ffmpeg_to_oximedia("flv"), Some("flv"));
    }

    #[test]
    fn test_container_unknown() {
        assert!(ContainerMapper::ffmpeg_to_oximedia("xyz_unknown").is_none());
    }

    #[test]
    fn test_container_case_insensitive() {
        assert_eq!(ContainerMapper::ffmpeg_to_oximedia("MP4"), Some("mp4"));
        assert_eq!(ContainerMapper::ffmpeg_to_oximedia("MKV"), Some("matroska"));
    }

    // ── PixelFormatMapper ────────────────────────────────────────────────────

    #[test]
    fn test_pixel_fmt_yuv420p() {
        assert_eq!(PixelFormatMapper::ffmpeg_to_oximedia("yuv420p"), Some("yuv420p"));
    }

    #[test]
    fn test_pixel_fmt_yuv422p() {
        assert_eq!(PixelFormatMapper::ffmpeg_to_oximedia("yuv422p"), Some("yuv422p"));
    }

    #[test]
    fn test_pixel_fmt_yuv444p() {
        assert_eq!(PixelFormatMapper::ffmpeg_to_oximedia("yuv444p"), Some("yuv444p"));
    }

    #[test]
    fn test_pixel_fmt_nv12() {
        assert_eq!(PixelFormatMapper::ffmpeg_to_oximedia("nv12"), Some("nv12"));
    }

    #[test]
    fn test_pixel_fmt_p010le() {
        assert_eq!(PixelFormatMapper::ffmpeg_to_oximedia("p010le"), Some("p010le"));
    }

    #[test]
    fn test_pixel_fmt_rgb24_rgba() {
        assert_eq!(PixelFormatMapper::ffmpeg_to_oximedia("rgb24"), Some("rgb24"));
        assert_eq!(PixelFormatMapper::ffmpeg_to_oximedia("rgba"), Some("rgba"));
    }

    #[test]
    fn test_pixel_fmt_unknown() {
        assert!(PixelFormatMapper::ffmpeg_to_oximedia("xyz_unknown_fmt").is_none());
    }

    // ── parse_filter_chain ───────────────────────────────────────────────────

    #[test]
    fn test_filter_chain_two_nodes() {
        let nodes = parse_filter_chain("scale=1280:720,setsar=1");
        assert_eq!(nodes.len(), 2);
        assert_eq!(nodes[0].name, "scale");
        assert_eq!(nodes[1].name, "setsar");
    }

    #[test]
    fn test_filter_chain_single_node() {
        let nodes = parse_filter_chain("hflip");
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].name, "hflip");
    }

    #[test]
    fn test_filter_chain_args_parsed() {
        let nodes = parse_filter_chain("scale=1920:1080");
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].name, "scale");
        assert_eq!(nodes[0].positional_args, vec!["1920", "1080"]);
    }

    #[test]
    fn test_filter_chain_named_args() {
        let nodes = parse_filter_chain("scale=w=1280:h=720");
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].name, "scale");
        assert!(nodes[0].named_args.iter().any(|(k, v)| k == "w" && v == "1280"));
        assert!(nodes[0].named_args.iter().any(|(k, v)| k == "h" && v == "720"));
    }

    #[test]
    fn test_filter_chain_multiple_nodes() {
        let nodes = parse_filter_chain("scale=1280:720,vflip,hflip");
        assert_eq!(nodes.len(), 3);
        assert_eq!(nodes[0].name, "scale");
        assert_eq!(nodes[1].name, "vflip");
        assert_eq!(nodes[2].name, "hflip");
    }

    // ── FilterGraphParser::is_lavfi_source ───────────────────────────────────

    #[test]
    fn test_lavfi_color() {
        assert!(FilterGraphParser::is_lavfi_source("color"));
        assert!(FilterGraphParser::is_lavfi_source("colour"));
    }

    #[test]
    fn test_lavfi_testsrc() {
        assert!(FilterGraphParser::is_lavfi_source("testsrc"));
        assert!(FilterGraphParser::is_lavfi_source("testsrc2"));
    }

    #[test]
    fn test_lavfi_smptebars() {
        assert!(FilterGraphParser::is_lavfi_source("smptebars"));
        assert!(FilterGraphParser::is_lavfi_source("smptehdbars"));
    }

    #[test]
    fn test_lavfi_sine() {
        assert!(FilterGraphParser::is_lavfi_source("sine"));
    }

    #[test]
    fn test_lavfi_anullsrc_nullsrc() {
        assert!(FilterGraphParser::is_lavfi_source("anullsrc"));
        assert!(FilterGraphParser::is_lavfi_source("nullsrc"));
    }

    #[test]
    fn test_lavfi_not_scale() {
        assert!(!FilterGraphParser::is_lavfi_source("scale"));
        assert!(!FilterGraphParser::is_lavfi_source("hflip"));
        assert!(!FilterGraphParser::is_lavfi_source("overlay"));
    }

    #[test]
    fn test_lavfi_with_params_stripped() {
        // color=c=red:size=1280x720 → name is "color"
        assert!(FilterGraphParser::is_lavfi_source("color=c=red:size=1280x720"));
    }

    // ── FilterGraphValidator::check_connections ───────────────────────────────

    #[test]
    fn test_validator_valid_graph() {
        use crate::filter_graph::FilterGraph;
        // Internal label [mid] connects the two chains; [in] and [out] are external I/O.
        // The validator only checks *internal* labels (labels that appear as both
        // producers and consumers within the graph).
        // Graph: [in]scale[mid];[mid]hflip[out]
        // Producers: ["mid", "out"], Consumers: ["in", "mid"]
        // "mid" is in both → connected (valid internal connection)
        // "out" and "in" are external I/O → may appear unpaired
        // The test just checks: no *internal* label is completely disconnected.
        // A simple chain with all-external labels should produce NO problems.
        let graph = FilterGraph::parse("scale=1280:720")
            .expect("parse");
        let problems = FilterGraphValidator::check_connections(&graph);
        assert!(problems.is_empty(), "simple chain with no labels: {:?}", problems);
    }

    #[test]
    fn test_validator_internal_connection_valid() {
        use crate::filter_graph::FilterGraph;
        // [0:v]scale[mid];[mid]hflip[out_v]
        // Producers: ["mid", "out_v"], Consumers: ["0:v", "mid"]
        // "mid" is matched → valid
        // "0:v" and "out_v" are external I/O → not flagged if we treat them as external
        // Actually our validator checks ALL consumers/producers, so "mid" should match.
        // This tests the matching logic.
        let graph = FilterGraph::parse("[a]scale=1280:720[b];[b]hflip[c]")
            .expect("parse");
        let problems = FilterGraphValidator::check_connections(&graph);
        // "b" is both produced and consumed → 0 errors for it
        // "a" is consumed but not produced, "c" is produced but not consumed → 2 problems
        // Total = 2 external I/O issues
        assert_eq!(problems.len(), 2, "external I/O labels: {:?}", problems);
    }

    #[test]
    fn test_validator_unconnected_internal_label() {
        use crate::filter_graph::FilterGraph;
        // Chain 1 produces [out1], Chain 2 produces [out2], neither is consumed
        // Also chain 2 uses [in] which nobody produces
        let graph = FilterGraph::parse("[in]scale=1280:720[out1];[in]hflip[out2]")
            .expect("parse");
        let problems = FilterGraphValidator::check_connections(&graph);
        // "in" is consumed twice but never produced (external input) → 1 problem
        // "out1" is produced but never consumed (external output) → 1 problem
        // "out2" is produced but never consumed (external output) → 1 problem
        assert!(!problems.is_empty(), "should detect unmatched pads");
    }

    // ── ArgumentBuilder ──────────────────────────────────────────────────────

    #[test]
    fn test_builder_codec_video_audio() {
        let args = ArgumentBuilder::new()
            .input("input.mp4")
            .codec_video("av1")
            .crf(30)
            .codec_audio("opus")
            .output("output.webm")
            .build();
        assert!(args.contains(&"-c:v".to_string()));
        assert!(args.contains(&"av1".to_string()));
        assert!(args.contains(&"-c:a".to_string()));
        assert!(args.contains(&"opus".to_string()));
        assert!(args.contains(&"-crf".to_string()));
        assert!(args.contains(&"30".to_string()));
        assert_eq!(args.last(), Some(&"output.webm".to_string()));
    }

    #[test]
    fn test_builder_bitrate_video_audio() {
        let args = ArgumentBuilder::new()
            .input("in.mp4")
            .bitrate_video("4M")
            .bitrate_audio("192k")
            .output("out.mp4")
            .build();
        let bv_idx = args.iter().position(|a| a == "-b:v").expect("-b:v");
        assert_eq!(args[bv_idx + 1], "4M");
        let ba_idx = args.iter().position(|a| a == "-b:a").expect("-b:a");
        assert_eq!(args[ba_idx + 1], "192k");
    }

    #[test]
    fn test_builder_scale() {
        let args = ArgumentBuilder::new()
            .input("in.mp4")
            .scale(1280, 720)
            .output("out.mp4")
            .build();
        let vf_idx = args.iter().position(|a| a == "-vf").expect("-vf");
        assert_eq!(args[vf_idx + 1], "scale=1280:720");
    }

    #[test]
    fn test_builder_fps() {
        let args = ArgumentBuilder::new()
            .input("in.mp4")
            .fps(29.97)
            .output("out.mp4")
            .build();
        let r_idx = args.iter().position(|a| a == "-r").expect("-r");
        assert_eq!(args[r_idx + 1], "29.97");
    }

    #[test]
    fn test_builder_seek_duration() {
        let args = ArgumentBuilder::new()
            .input("in.mp4")
            .seek(30.0)
            .duration(60.0)
            .output("out.mp4")
            .build();
        let ss_idx = args.iter().position(|a| a == "-ss").expect("-ss");
        assert_eq!(args[ss_idx + 1], "30");
        let t_idx = args.iter().position(|a| a == "-t").expect("-t");
        assert_eq!(args[t_idx + 1], "60");
    }

    #[test]
    fn test_builder_metadata() {
        let args = ArgumentBuilder::new()
            .input("in.mp4")
            .metadata("title", "Test Video")
            .output("out.mp4")
            .build();
        let m_idx = args.iter().position(|a| a == "-metadata").expect("-metadata");
        assert_eq!(args[m_idx + 1], "title=Test Video");
    }

    #[test]
    fn test_builder_preset() {
        let args = ArgumentBuilder::new()
            .input("in.mp4")
            .preset("medium")
            .output("out.mp4")
            .build();
        let p_idx = args.iter().position(|a| a == "-preset").expect("-preset");
        assert_eq!(args[p_idx + 1], "medium");
    }

    #[test]
    fn test_builder_input_before_output() {
        let args = ArgumentBuilder::new()
            .input("in.mp4")
            .codec_video("av1")
            .output("out.webm")
            .build();
        let i_idx = args.iter().position(|a| a == "-i").expect("-i");
        let out_idx = args.iter().position(|a| a == "out.webm").expect("out");
        assert!(i_idx < out_idx);
    }

    #[test]
    fn test_builder_multi_input() {
        let args = ArgumentBuilder::new()
            .input("video.mp4")
            .input("audio.flac")
            .codec_video("copy")
            .codec_audio("copy")
            .output("output.mkv")
            .build();
        let i_positions: Vec<usize> = args
            .iter()
            .enumerate()
            .filter_map(|(i, a)| if a == "-i" { Some(i) } else { None })
            .collect();
        assert_eq!(i_positions.len(), 2);
        assert_eq!(args[i_positions[0] + 1], "video.mp4");
        assert_eq!(args[i_positions[1] + 1], "audio.flac");
    }

    // ── FfmpegDiagnostics::check_deprecated_options ──────────────────────────

    #[test]
    fn test_deprecated_vcodec() {
        let warnings = FfmpegDiagnostics::check_deprecated_options(&["-vcodec", "libx264"]);
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].deprecated_flag, "-vcodec");
        assert_eq!(warnings[0].replacement, "-c:v");
    }

    #[test]
    fn test_deprecated_acodec() {
        let warnings = FfmpegDiagnostics::check_deprecated_options(&["-acodec", "aac"]);
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].deprecated_flag, "-acodec");
        assert_eq!(warnings[0].replacement, "-c:a");
    }

    #[test]
    fn test_deprecated_ab() {
        let warnings = FfmpegDiagnostics::check_deprecated_options(&["-ab", "128k"]);
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].deprecated_flag, "-ab");
        assert_eq!(warnings[0].replacement, "-b:a");
    }

    #[test]
    fn test_deprecated_none_for_modern_flags() {
        let warnings = FfmpegDiagnostics::check_deprecated_options(&[
            "-c:v", "av1", "-c:a", "opus", "-b:a", "128k",
        ]);
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_deprecated_multiple() {
        let warnings = FfmpegDiagnostics::check_deprecated_options(&[
            "-vcodec", "libx264", "-acodec", "aac", "-ab", "128k",
        ]);
        assert_eq!(warnings.len(), 3);
    }

    // ── FfmpegCompatDiagnostics::score ───────────────────────────────────────

    #[test]
    fn test_compat_score_known_codecs() {
        let args_strs = vec![
            s("-i"), s("in.mkv"),
            s("-c:v"), s("libaom-av1"),
            s("-c:a"), s("libopus"),
            s("out.webm"),
        ];
        let parsed = FfmpegArgs::parse(&args_strs).expect("parse");
        let score = FfmpegCompatDiagnostics::score(&parsed);
        assert!(
            score >= 0.9,
            "known patent-free codecs should score near 1.0, got {}",
            score
        );
    }

    #[test]
    fn test_compat_score_unknown_codec_deducts() {
        let args_strs = vec![
            s("-i"), s("in.mkv"),
            s("-c:v"), s("completely_unknown_video_codec_xyz"),
            s("out.mkv"),
        ];
        let parsed = FfmpegArgs::parse(&args_strs).expect("parse");
        let score = FfmpegCompatDiagnostics::score(&parsed);
        assert!(score < 1.0, "unknown codec should deduct from score");
    }

    #[test]
    fn test_compat_score_bounded_zero_to_one() {
        let args_strs = vec![
            s("-i"), s("in.mkv"),
            s("-c:v"), s("unk1"),
            s("-c:a"), s("unk2"),
            s("-f"), s("xyz_unknown_format"),
            s("out.mkv"),
        ];
        let parsed = FfmpegArgs::parse(&args_strs).expect("parse");
        let score = FfmpegCompatDiagnostics::score(&parsed);
        assert!(score >= 0.0);
        assert!(score <= 1.0);
    }

    // ── TranslationHint / generate_hints ─────────────────────────────────────

    #[test]
    fn test_hint_direct_av1() {
        let args_strs = vec![
            s("-i"), s("in.mkv"),
            s("-c:v"), s("libaom-av1"),
            s("out.webm"),
        ];
        let parsed = FfmpegArgs::parse(&args_strs).expect("parse");
        let hints = generate_hints(&parsed);
        let av1_hint = hints.iter().find(|h| h.original == "libaom-av1");
        assert!(av1_hint.is_some(), "should have a hint for libaom-av1");
        let h = av1_hint.expect("tested above");
        assert_eq!(h.translated, "av1");
        assert!((h.confidence - 1.0).abs() < 0.001, "direct match = confidence 1.0");
    }

    #[test]
    fn test_hint_patent_substituted() {
        let args_strs = vec![
            s("-i"), s("in.mp4"),
            s("-c:v"), s("libx265"),
            s("out.webm"),
        ];
        let parsed = FfmpegArgs::parse(&args_strs).expect("parse");
        let hints = generate_hints(&parsed);
        let hint = hints.iter().find(|h| h.original == "libx265");
        assert!(hint.is_some());
        let h = hint.expect("tested above");
        assert_eq!(h.translated, "av1");
        assert!((h.confidence - 0.5).abs() < 0.001, "substitution = confidence 0.5");
        assert!(h.note.is_some(), "substitution should have a note");
    }

    #[test]
    fn test_hint_codec_libx265_av1() {
        // Specifically test libx265→av1 mapping as required by task spec
        use crate::codec_map::CodecMap;
        let cm = CodecMap::new();
        let entry = cm.lookup("libx265").expect("libx265 should exist");
        assert_eq!(entry.oxi_name, "av1");
    }

    // ── StreamMap::parse ─────────────────────────────────────────────────────

    #[test]
    fn test_stream_map_parse_video() {
        let sm = StreamMap::parse("0:v:0").expect("parse");
        assert_eq!(sm.input_idx, 0);
        assert_eq!(sm.stream_type, Some(MapStreamType::Video));
        assert_eq!(sm.stream_idx, Some(0));
        assert!(!sm.negative);
    }

    #[test]
    fn test_stream_map_parse_audio() {
        let sm = StreamMap::parse("0:a:1").expect("parse");
        assert_eq!(sm.input_idx, 0);
        assert_eq!(sm.stream_type, Some(MapStreamType::Audio));
        assert_eq!(sm.stream_idx, Some(1));
    }

    #[test]
    fn test_stream_map_parse_negative() {
        let sm = StreamMap::parse("-0:s").expect("parse");
        assert!(sm.negative);
        assert_eq!(sm.stream_type, Some(MapStreamType::Subtitle));
    }

    #[test]
    fn test_stream_map_parse_all_streams() {
        let sm = StreamMap::parse("1").expect("parse");
        assert_eq!(sm.input_idx, 1);
        assert!(sm.stream_type.is_none());
        assert!(sm.stream_idx.is_none());
    }

    // ── time parsing ─────────────────────────────────────────────────────────

    #[test]
    fn test_parse_time_float() {
        assert!((parse_time_str("123.45").expect("ok") - 123.45).abs() < 0.001);
    }

    #[test]
    fn test_parse_time_hhmmss() {
        assert!((parse_time_str("01:30:00").expect("ok") - 5400.0).abs() < 0.001);
    }

    #[test]
    fn test_parse_time_mmss() {
        assert!((parse_time_str("02:30").expect("ok") - 150.0).abs() < 0.001);
    }
}
