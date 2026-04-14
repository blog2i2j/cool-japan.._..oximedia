//! libavfilter (lavfi) filter graph compatibility for OxiMedia.
//!
//! This module translates FFmpeg/lavfi filtergraph strings into OxiMedia
//! pipeline stage descriptions. It supports:
//!
//! - Simple filter chains (`-vf` / `-af`) with named parameters
//! - `filter_complex` multi-input/output graphs with pad labels
//! - Filter parameter type coercion (int, float, string, rational)
//! - OxiMedia pipeline stage mapping for known lavfi filters
//!
//! ## Example
//!
//! ```rust
//! use oximedia_compat_ffmpeg::lavfi_compat::{translate_lavfi_graph, LavfiGraph};
//!
//! let graph = translate_lavfi_graph("[0:v]scale=1280:720[out]").unwrap();
//! assert_eq!(graph.nodes.len(), 1);
//! assert_eq!(graph.nodes[0].filter_name, "scale");
//! ```

use std::collections::HashMap;

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Error returned when a lavfi filtergraph string cannot be parsed.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum LavfiError {
    /// The filtergraph string is empty.
    #[error("empty filtergraph string")]
    EmptyGraph,

    /// A filter node has a malformed name.
    #[error("malformed filter name in segment: '{0}'")]
    MalformedFilterName(String),

    /// A filter parameter value could not be parsed.
    #[error("failed to parse parameter '{key}' = '{value}': {reason}")]
    ParameterParseError {
        /// Parameter key.
        key: String,
        /// Raw value string.
        value: String,
        /// Reason for failure.
        reason: String,
    },

    /// A pad label is malformed (not enclosed in `[…]`).
    #[error("malformed pad label: '{0}'")]
    MalformedPadLabel(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// Parameter value types
// ─────────────────────────────────────────────────────────────────────────────

/// A typed parameter value decoded from a lavfi filter argument.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamValue {
    /// An integer value, e.g. `scale=1280:720` → `1280`.
    Integer(i64),
    /// A floating-point value, e.g. `fps=29.97`.
    Float(f64),
    /// A rational number represented as numerator/denominator, e.g. `30000/1001`.
    Rational(i64, i64),
    /// A plain string value, e.g. `format=yuv420p`.
    Text(String),
    /// A boolean `true`/`false` / `1`/`0` value.
    Bool(bool),
}

impl ParamValue {
    /// Parse a raw string into the most specific `ParamValue` type.
    ///
    /// Precedence: bool → rational → integer → float → text.
    pub fn parse(raw: &str) -> Self {
        // Bool
        match raw.to_lowercase().as_str() {
            "true" | "1" | "yes" | "on" => return Self::Bool(true),
            "false" | "0" | "no" | "off" => return Self::Bool(false),
            _ => {}
        }

        // Rational (numerator/denominator)
        if let Some((num_str, den_str)) = raw.split_once('/') {
            if let (Ok(n), Ok(d)) = (num_str.trim().parse::<i64>(), den_str.trim().parse::<i64>()) {
                if d != 0 {
                    return Self::Rational(n, d);
                }
            }
        }

        // Integer
        if let Ok(i) = raw.parse::<i64>() {
            return Self::Integer(i);
        }

        // Float
        if let Ok(f) = raw.parse::<f64>() {
            return Self::Float(f);
        }

        // Text fallback
        Self::Text(raw.to_string())
    }

    /// Return the value as an `i64` if it is an `Integer` or truncated `Float`.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Integer(i) => Some(*i),
            Self::Float(f) => Some(*f as i64),
            Self::Bool(b) => Some(if *b { 1 } else { 0 }),
            _ => None,
        }
    }

    /// Return the value as an `f64` if it is numeric.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Integer(i) => Some(*i as f64),
            Self::Float(f) => Some(*f),
            Self::Rational(n, d) => {
                if *d != 0 {
                    Some(*n as f64 / *d as f64)
                } else {
                    None
                }
            }
            Self::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            _ => None,
        }
    }

    /// Return the value as a `&str` if it is a `Text` variant.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Text(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

impl std::fmt::Display for ParamValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Integer(i) => write!(f, "{}", i),
            Self::Float(v) => write!(f, "{}", v),
            Self::Rational(n, d) => write!(f, "{}/{}", n, d),
            Self::Text(s) => write!(f, "{}", s),
            Self::Bool(b) => write!(f, "{}", b),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OxiMedia pipeline stage mapping
// ─────────────────────────────────────────────────────────────────────────────

/// The OxiMedia pipeline stage equivalent for a known lavfi filter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OxiStage {
    /// Video scale operation.
    VideoScale,
    /// Video crop operation.
    VideoCrop,
    /// Video frame-rate conversion.
    VideoFps,
    /// Video color format conversion.
    VideoFormat,
    /// Horizontal flip.
    VideoHflip,
    /// Vertical flip.
    VideoVflip,
    /// Deinterlacing (yadif / bwdif).
    VideoDeinterlace,
    /// Overlay / compositing.
    VideoOverlay,
    /// Split (tee) a video stream.
    VideoSplit,
    /// Pad with borders.
    VideoPad,
    /// 3D LUT application.
    VideoLut3d,
    /// EBU R128 loudness normalization.
    AudioLoudnorm,
    /// Audio volume adjustment.
    AudioVolume,
    /// Audio resampling.
    AudioResample,
    /// Audio channel map.
    AudioChannelMap,
    /// Audio compressor.
    AudioCompress,
    /// Audio mix (amix / amerge).
    AudioMix,
    /// Concat streams.
    Concat,
    /// Null (passthrough) — no-op.
    Null,
    /// A filter that has no direct OxiMedia equivalent.
    Unsupported {
        /// The original lavfi filter name.
        name: String,
    },
}

impl OxiStage {
    /// Return `true` if this stage has no direct OxiMedia implementation.
    pub fn is_unsupported(&self) -> bool {
        matches!(self, Self::Unsupported { .. })
    }
}

/// Map a lavfi filter name to the corresponding [`OxiStage`].
fn map_filter_to_stage(name: &str) -> OxiStage {
    match name {
        "scale" | "scale2ref" => OxiStage::VideoScale,
        "crop" => OxiStage::VideoCrop,
        "fps" | "framerate" => OxiStage::VideoFps,
        "format" | "vformat" => OxiStage::VideoFormat,
        "hflip" => OxiStage::VideoHflip,
        "vflip" => OxiStage::VideoVflip,
        "yadif" | "bwdif" | "w3fdif" => OxiStage::VideoDeinterlace,
        "overlay" => OxiStage::VideoOverlay,
        "split" | "asplit" => OxiStage::VideoSplit,
        "pad" => OxiStage::VideoPad,
        "lut3d" | "haldclut" => OxiStage::VideoLut3d,
        "loudnorm" | "ebur128" => OxiStage::AudioLoudnorm,
        "volume" => OxiStage::AudioVolume,
        "aresample" | "resample" => OxiStage::AudioResample,
        "channelmap" | "pan" => OxiStage::AudioChannelMap,
        "acompressor" | "compand" => OxiStage::AudioCompress,
        "amix" | "amerge" => OxiStage::AudioMix,
        "concat" => OxiStage::Concat,
        "null" | "anull" | "setpts" | "asetpts" | "pts" => OxiStage::Null,
        other => OxiStage::Unsupported {
            name: other.to_string(),
        },
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Parsed graph structures
// ─────────────────────────────────────────────────────────────────────────────

/// A single parsed filter node in a lavfi filtergraph.
#[derive(Debug, Clone)]
pub struct LavfiNode {
    /// The lavfi filter name (e.g. `"scale"`, `"loudnorm"`).
    pub filter_name: String,
    /// Named parameters (key → typed value).
    pub params: HashMap<String, ParamValue>,
    /// Positional parameters (index → typed value), for filters like `scale=W:H`.
    pub positional: Vec<ParamValue>,
    /// Input pad labels (`[in]`, `[0:v]`, …).
    pub input_pads: Vec<String>,
    /// Output pad labels (`[out]`, `[v]`, …).
    pub output_pads: Vec<String>,
    /// The OxiMedia stage this node maps to.
    pub oxi_stage: OxiStage,
}

impl LavfiNode {
    /// Return the value of a named parameter, or `None` if absent.
    pub fn param(&self, key: &str) -> Option<&ParamValue> {
        self.params.get(key)
    }

    /// Return the value of a positional parameter by index, or `None`.
    pub fn positional_param(&self, index: usize) -> Option<&ParamValue> {
        self.positional.get(index)
    }
}

/// A chain of lavfi filter nodes connected in sequence (separated by `,`).
#[derive(Debug, Clone)]
pub struct LavfiChain {
    /// The sequential filter nodes in this chain.
    pub nodes: Vec<LavfiNode>,
}

/// A complete parsed lavfi filtergraph (chains separated by `;`).
#[derive(Debug, Clone)]
pub struct LavfiGraph {
    /// All filter nodes from all chains (flattened for easy iteration).
    pub nodes: Vec<LavfiNode>,
    /// Parallel filter chains (each chain is a `,`-separated sequence).
    pub chains: Vec<LavfiChain>,
    /// Whether this graph uses pad labels (i.e., is a `filter_complex`-style graph).
    pub is_complex: bool,
}

impl LavfiGraph {
    /// Return all nodes that map to an unsupported OxiStage.
    pub fn unsupported_nodes(&self) -> Vec<&LavfiNode> {
        self.nodes
            .iter()
            .filter(|n| n.oxi_stage.is_unsupported())
            .collect()
    }

    /// Return all unique OxiStage variants referenced in this graph.
    pub fn stages(&self) -> Vec<&OxiStage> {
        self.nodes.iter().map(|n| &n.oxi_stage).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Parser implementation
// ─────────────────────────────────────────────────────────────────────────────

/// Parse pad labels (`[label1][label2]`) from the start of `s`, consuming them.
///
/// Returns (labels, remainder).
fn parse_pad_labels(s: &str) -> (Vec<String>, &str) {
    let mut labels = Vec::new();
    let mut rest = s.trim_start();

    while rest.starts_with('[') {
        if let Some(close) = rest.find(']') {
            let label = rest[1..close].to_string();
            labels.push(label);
            rest = rest[close + 1..].trim_start();
        } else {
            break;
        }
    }

    (labels, rest)
}

/// Parse the parameter list from a filter specification string like
/// `"scale=1280:720"` or `"scale=w=1280:h=720"`.
///
/// Returns `(params, positional)`.
fn parse_filter_params(
    params_str: &str,
) -> (HashMap<String, ParamValue>, Vec<ParamValue>) {
    let mut named: HashMap<String, ParamValue> = HashMap::new();
    let mut positional: Vec<ParamValue> = Vec::new();

    if params_str.is_empty() {
        return (named, positional);
    }

    // Split by `:` for positional / named parameters.
    // Named parameters use `key=value` syntax within a segment.
    for segment in params_str.split(':') {
        let segment = segment.trim();
        if segment.is_empty() {
            continue;
        }
        if let Some((key, value)) = segment.split_once('=') {
            named.insert(key.trim().to_string(), ParamValue::parse(value.trim()));
        } else {
            positional.push(ParamValue::parse(segment));
        }
    }

    (named, positional)
}

/// Parse a single filter node from a string like `"scale=1280:720"` or
/// `"scale=w=1280:h=720"` (without pad labels).
fn parse_single_node(s: &str) -> Result<LavfiNode, LavfiError> {
    let s = s.trim();
    if s.is_empty() {
        return Err(LavfiError::MalformedFilterName(s.to_string()));
    }

    let (filter_name, params_raw) = if let Some((name, rest)) = s.split_once('=') {
        (name.trim().to_string(), rest)
    } else {
        (s.to_string(), "")
    };

    if filter_name.is_empty()
        || filter_name
            .chars()
            .any(|c| !c.is_alphanumeric() && c != '_')
    {
        return Err(LavfiError::MalformedFilterName(filter_name));
    }

    let (params, positional) = parse_filter_params(params_raw);
    let oxi_stage = map_filter_to_stage(&filter_name);

    Ok(LavfiNode {
        filter_name,
        params,
        positional,
        input_pads: Vec::new(),
        output_pads: Vec::new(),
        oxi_stage,
    })
}

/// Parse a single chain segment (potentially with leading/trailing pad labels).
///
/// A chain segment looks like: `[in_pad]filter1=params,filter2=params[out_pad]`
fn parse_chain(chain_str: &str) -> Result<(LavfiChain, bool), LavfiError> {
    let chain_str = chain_str.trim();
    if chain_str.is_empty() {
        return Ok((LavfiChain { nodes: Vec::new() }, false));
    }

    // Check for pad-label usage (complex graph indicator).
    let has_labels = chain_str.contains('[');

    // Split filters in this chain by `,` (avoiding label brackets).
    let filter_segments = split_chain_filters(chain_str);

    let mut nodes = Vec::new();

    for (idx, segment) in filter_segments.iter().enumerate() {
        let segment = segment.trim();

        // Parse leading pad labels (only on the first segment for the chain's input).
        let (input_pads, after_input) = if idx == 0 {
            parse_pad_labels(segment)
        } else {
            (Vec::new(), segment)
        };

        // Parse trailing pad labels (only on the last segment for the chain's output).
        let (filter_str, output_pads) = if idx == filter_segments.len() - 1 {
            strip_trailing_pad_labels(after_input)
        } else {
            (after_input, Vec::new())
        };

        if filter_str.trim().is_empty() {
            continue;
        }

        let mut node = parse_single_node(filter_str)?;
        node.input_pads = input_pads;
        node.output_pads = output_pads;
        nodes.push(node);
    }

    Ok((LavfiChain { nodes }, has_labels))
}

/// Split a chain string by `,` while respecting bracket nesting.
fn split_chain_filters(chain: &str) -> Vec<&str> {
    let mut segments = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;

    for (i, ch) in chain.char_indices() {
        match ch {
            '[' => depth += 1,
            ']' => {
                if depth > 0 {
                    depth -= 1;
                }
            }
            ',' if depth == 0 => {
                segments.push(&chain[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    segments.push(&chain[start..]);
    segments
}

/// Strip trailing pad labels from a filter string, returning `(filter_str, labels)`.
fn strip_trailing_pad_labels(s: &str) -> (&str, Vec<String>) {
    let s = s.trim_end();
    if !s.ends_with(']') {
        return (s, Vec::new());
    }

    // Walk backwards to find all trailing `[…]` sequences.
    let mut labels = Vec::new();
    let mut rest = s;

    loop {
        if !rest.trim_end().ends_with(']') {
            break;
        }
        let trimmed = rest.trim_end();
        if let Some(open_pos) = trimmed.rfind('[') {
            let label = trimmed[open_pos + 1..trimmed.len() - 1].to_string();
            labels.insert(0, label);
            rest = &trimmed[..open_pos];
        } else {
            break;
        }
    }

    (rest.trim_end(), labels)
}

/// Translate a lavfi filtergraph string into a structured [`LavfiGraph`].
///
/// The input may be a simple filter chain (e.g. `"scale=1280:720,fps=30"`)
/// or a complex graph with pad labels
/// (e.g. `"[0:v]scale=1280:720[v];[0:a]loudnorm[a]"`).
///
/// # Errors
///
/// Returns [`LavfiError`] if the graph string is empty or contains
/// syntactically invalid filter names or parameters.
pub fn translate_lavfi_graph(graph_str: &str) -> Result<LavfiGraph, LavfiError> {
    let trimmed = graph_str.trim();
    if trimmed.is_empty() {
        return Err(LavfiError::EmptyGraph);
    }

    // Split into chains by `;`.
    let chain_strs: Vec<&str> = trimmed.split(';').collect();
    let mut all_nodes = Vec::new();
    let mut chains = Vec::new();
    let mut is_complex = false;

    for chain_str in &chain_strs {
        let (chain, chain_has_labels) = parse_chain(chain_str)?;
        if chain_has_labels {
            is_complex = true;
        }
        all_nodes.extend(chain.nodes.iter().cloned());
        chains.push(chain);
    }

    Ok(LavfiGraph {
        nodes: all_nodes,
        chains,
        is_complex,
    })
}

/// Translate a simple `-vf` / `-af` filter chain (no pad labels) into a
/// [`LavfiGraph`]. This is a convenience wrapper around
/// [`translate_lavfi_graph`] for simple filter chains.
pub fn translate_simple_filter_chain(chain_str: &str) -> Result<LavfiGraph, LavfiError> {
    translate_lavfi_graph(chain_str)
}

/// Describe all filter nodes in a [`LavfiGraph`] as human-readable strings
/// suitable for diagnostics.
pub fn describe_graph(graph: &LavfiGraph) -> Vec<String> {
    graph
        .nodes
        .iter()
        .map(|node| {
            let param_count = node.params.len() + node.positional.len();
            format!(
                "{} → {:?} ({} params)",
                node.filter_name, node.oxi_stage, param_count
            )
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_graph_returns_error() {
        assert_eq!(translate_lavfi_graph("").unwrap_err(), LavfiError::EmptyGraph);
        assert_eq!(
            translate_lavfi_graph("   ").unwrap_err(),
            LavfiError::EmptyGraph
        );
    }

    #[test]
    fn test_simple_scale_filter() {
        let g = translate_lavfi_graph("scale=1280:720").expect("parse failed");
        assert_eq!(g.nodes.len(), 1);
        let n = &g.nodes[0];
        assert_eq!(n.filter_name, "scale");
        assert!(matches!(n.oxi_stage, OxiStage::VideoScale));
        assert_eq!(n.positional.len(), 2);
        assert_eq!(n.positional[0].as_i64(), Some(1280));
        assert_eq!(n.positional[1].as_i64(), Some(720));
    }

    #[test]
    fn test_named_params_scale() {
        let g = translate_lavfi_graph("scale=w=1920:h=1080").expect("parse failed");
        assert_eq!(g.nodes.len(), 1);
        let n = &g.nodes[0];
        assert_eq!(n.params["w"].as_i64(), Some(1920));
        assert_eq!(n.params["h"].as_i64(), Some(1080));
    }

    #[test]
    fn test_chain_multiple_filters() {
        let g = translate_lavfi_graph("scale=1280:720,fps=30").expect("parse failed");
        assert_eq!(g.nodes.len(), 2);
        assert_eq!(g.nodes[0].filter_name, "scale");
        assert_eq!(g.nodes[1].filter_name, "fps");
    }

    #[test]
    fn test_complex_graph_with_labels() {
        let g = translate_lavfi_graph("[0:v]scale=1280:720[out_v]").expect("parse failed");
        assert!(g.is_complex);
        assert_eq!(g.nodes.len(), 1);
        let n = &g.nodes[0];
        assert_eq!(n.input_pads, vec!["0:v".to_string()]);
        assert_eq!(n.output_pads, vec!["out_v".to_string()]);
    }

    #[test]
    fn test_multi_chain_graph() {
        let g = translate_lavfi_graph("[0:v]scale=1280:720[v];[0:a]loudnorm[a]")
            .expect("parse failed");
        assert_eq!(g.chains.len(), 2);
        assert_eq!(g.nodes.len(), 2);
        assert!(g.is_complex);
    }

    #[test]
    fn test_oxi_stage_mapping() {
        let cases = [
            ("scale", OxiStage::VideoScale),
            ("fps", OxiStage::VideoFps),
            ("hflip", OxiStage::VideoHflip),
            ("vflip", OxiStage::VideoVflip),
            ("yadif", OxiStage::VideoDeinterlace),
            ("loudnorm", OxiStage::AudioLoudnorm),
            ("volume", OxiStage::AudioVolume),
            ("null", OxiStage::Null),
        ];
        for (name, expected) in &cases {
            let g = translate_lavfi_graph(name).expect(name);
            assert_eq!(&g.nodes[0].oxi_stage, expected, "for filter {}", name);
        }
    }

    #[test]
    fn test_unsupported_filter_stage() {
        let g = translate_lavfi_graph("drawtext=text=Hello").expect("parse failed");
        assert_eq!(g.nodes.len(), 1);
        assert!(g.nodes[0].oxi_stage.is_unsupported());
        let unsupported = g.unsupported_nodes();
        assert_eq!(unsupported.len(), 1);
    }

    #[test]
    fn test_param_value_parse_integer() {
        assert_eq!(ParamValue::parse("42"), ParamValue::Integer(42));
        assert_eq!(ParamValue::parse("-100"), ParamValue::Integer(-100));
    }

    #[test]
    fn test_param_value_parse_float() {
        assert_eq!(ParamValue::parse("29.97"), ParamValue::Float(29.97));
    }

    #[test]
    fn test_param_value_parse_rational() {
        assert_eq!(ParamValue::parse("30000/1001"), ParamValue::Rational(30000, 1001));
        let r = ParamValue::Rational(30000, 1001);
        let f = r.as_f64().expect("rational as f64");
        assert!((f - 29.97).abs() < 0.01);
    }

    #[test]
    fn test_param_value_parse_bool() {
        assert_eq!(ParamValue::parse("true"), ParamValue::Bool(true));
        assert_eq!(ParamValue::parse("false"), ParamValue::Bool(false));
    }

    #[test]
    fn test_param_value_parse_text() {
        assert_eq!(
            ParamValue::parse("yuv420p"),
            ParamValue::Text("yuv420p".to_string())
        );
    }

    #[test]
    fn test_describe_graph() {
        let g = translate_lavfi_graph("scale=1280:720,fps=30").expect("parse failed");
        let desc = describe_graph(&g);
        assert_eq!(desc.len(), 2);
        assert!(desc[0].contains("scale"));
        assert!(desc[1].contains("fps"));
    }

    #[test]
    fn test_simple_filter_chain_convenience() {
        let g = translate_simple_filter_chain("hflip,vflip").expect("parse failed");
        assert_eq!(g.nodes.len(), 2);
        assert!(!g.is_complex);
    }

    #[test]
    fn test_stages_list() {
        let g = translate_lavfi_graph("scale=640:480,hflip").expect("parse failed");
        let stages = g.stages();
        assert_eq!(stages.len(), 2);
    }
}
