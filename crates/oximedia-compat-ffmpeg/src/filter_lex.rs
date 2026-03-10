//! Lexer and parser for FFmpeg filtergraph strings.
//!
//! FFmpeg's `-vf`/`-af`/`-filter_complex` strings use a mini-language:
//!
//! ```text
//! [in]scale=1280:720[s];[s]crop=640:480:0:0[out]
//! ```
//!
//! This module provides two layers:
//!
//! 1. **Low-level**: [`FilterNode`] / [`FilterGraph`] — raw AST of filter
//!    expressions, preserving pad labels.
//! 2. **High-level**: [`ParsedFilter`] — a semantic enum that the translator
//!    uses to map FFmpeg filter semantics onto OxiMedia operations.

// ─────────────────────────────────────────────────────────────────────────────
// High-level semantic filter enum
// ─────────────────────────────────────────────────────────────────────────────

/// A semantically parsed filter from an FFmpeg filtergraph string.
///
/// Each variant represents a filter that has meaning in OxiMedia's API.
/// Unknown or unsupported filters are captured as [`ParsedFilter::Unknown`].
#[derive(Debug, Clone, PartialEq)]
pub enum ParsedFilter {
    /// `scale=W:H` — resize video.
    Scale { w: i32, h: i32 },
    /// `crop=w:h:x:y` — crop video.
    Crop { w: i32, h: i32, x: i32, y: i32 },
    /// `fps=N` — change frame rate.
    Fps { rate: f64 },
    /// `hflip` — horizontal mirror.
    HFlip,
    /// `vflip` — vertical flip.
    VFlip,
    /// `rotate=ANGLE` — rotate by angle in radians.
    Rotate { angle: f64 },
    /// `yadif` / `bwdif` — deinterlace.
    Deinterlace,
    /// `eq=brightness=B:contrast=C:saturation=S` — colour correction.
    ColorCorrect {
        brightness: f64,
        contrast: f64,
        saturation: f64,
    },
    /// `lut3d=file=x.cube` — apply a 3D LUT.
    Lut3d { file: String },
    /// `subtitles=x.srt` — burn in subtitles.
    SubtitleBurnIn { file: String },
    /// `loudnorm=I=L:TP=T:LRA=R` — EBU R128 loudness normalisation.
    LoudNorm {
        integrated: f64,
        true_peak: f64,
        lra: f64,
    },
    /// `volume=N` or `volume=NdB` — adjust audio volume.
    Volume { factor: f64 },
    /// `aresample=N` — resample audio.
    Resample { sample_rate: u32 },
    /// `acompressor=threshold=T:ratio=R` — dynamic range compression.
    Compressor { threshold: f64, ratio: f64 },
    /// `null`, `anull`, `setpts`, `format`, `colorspace` — passthroughs.
    Passthrough,
    /// Filters not supported by OxiMedia (stored for diagnostic purposes).
    Unknown { name: String, args: String },
}

// ─────────────────────────────────────────────────────────────────────────────
// Low-level AST
// ─────────────────────────────────────────────────────────────────────────────

/// A single filter node parsed from a filtergraph string.
#[derive(Debug, Clone, PartialEq)]
pub struct FilterNode {
    /// Input pad labels (e.g. `["in"]`).
    pub inputs: Vec<String>,
    /// The filter name (e.g. `"scale"`, `"crop"`, `"volume"`).
    pub name: String,
    /// Positional arguments (colon-separated, no `=` sign).
    pub positional_args: Vec<String>,
    /// Named arguments (`key=value` pairs).
    pub named_args: Vec<(String, String)>,
    /// Output pad labels.
    pub outputs: Vec<String>,
}

/// A parsed filtergraph: an ordered sequence of filter nodes.
#[derive(Debug, Clone, Default)]
pub struct FilterGraph {
    /// All filter nodes in the graph, in parse order.
    pub nodes: Vec<FilterNode>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Parse an FFmpeg filter string into a sequence of [`ParsedFilter`] values.
///
/// For `-filter_complex`, splits on `;` first. Each chain is then split on `,`.
/// Pad labels (`[in]`, `[out_v]`) are stripped before semantic parsing.
pub fn parse_filters(input: &str) -> Vec<ParsedFilter> {
    let graph = match parse_filter_graph(input) {
        Ok(g) => g,
        Err(_) => return Vec::new(),
    };
    graph.nodes.iter().map(node_to_parsed_filter).collect()
}

/// Parse an FFmpeg filter string into the low-level [`FilterGraph`] AST.
pub fn parse_filter_graph(input: &str) -> anyhow::Result<FilterGraph> {
    let mut graph = FilterGraph::default();

    for chain in input.split(';') {
        let chain = chain.trim();
        if chain.is_empty() {
            continue;
        }
        for filter_expr in split_filter_chain(chain) {
            let node = parse_filter_node(&filter_expr)?;
            graph.nodes.push(node);
        }
    }

    Ok(graph)
}

/// Legacy entry point — kept for compatibility; delegates to [`parse_filter_graph`].
pub fn parse_filter_string(input: &str) -> anyhow::Result<FilterGraph> {
    parse_filter_graph(input)
}

// ─────────────────────────────────────────────────────────────────────────────
// Semantic mapping
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a low-level [`FilterNode`] to a [`ParsedFilter`].
fn node_to_parsed_filter(node: &FilterNode) -> ParsedFilter {
    match node.name.as_str() {
        // ── Video geometry ────────────────────────────────────────────────────
        "scale" => parse_scale(node),
        "crop" => parse_crop(node),
        "hflip" => ParsedFilter::HFlip,
        "vflip" => ParsedFilter::VFlip,
        "rotate" => {
            let angle = get_pos_or_named(&node.positional_args, &node.named_args, "angle", 0)
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(0.0);
            ParsedFilter::Rotate { angle }
        }

        // ── Frame rate ────────────────────────────────────────────────────────
        "fps" => {
            // `fps=N` or `fps=fps=N`
            let rate = get_pos_or_named(&node.positional_args, &node.named_args, "fps", 0)
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(25.0);
            ParsedFilter::Fps { rate }
        }

        // ── Deinterlace ───────────────────────────────────────────────────────
        "yadif" | "bwdif" | "w3fdif" | "estdif" => ParsedFilter::Deinterlace,

        // ── Colour ────────────────────────────────────────────────────────────
        "eq" => parse_eq(node),

        // ── LUT ───────────────────────────────────────────────────────────────
        "lut3d" => {
            let file = get_named(&node.named_args, "file")
                .or_else(|| node.positional_args.first().map(String::as_str))
                .unwrap_or("")
                .to_string();
            ParsedFilter::Lut3d { file }
        }

        // ── Subtitles ─────────────────────────────────────────────────────────
        "subtitles" | "ass" => {
            let file = get_named(&node.named_args, "filename")
                .or_else(|| get_named(&node.named_args, "file"))
                .or_else(|| node.positional_args.first().map(String::as_str))
                .unwrap_or("")
                .to_string();
            ParsedFilter::SubtitleBurnIn { file }
        }

        // ── Audio loudness ────────────────────────────────────────────────────
        "loudnorm" => parse_loudnorm(node),

        // ── Audio volume ──────────────────────────────────────────────────────
        "volume" => {
            let raw = get_pos_or_named(&node.positional_args, &node.named_args, "volume", 0)
                .unwrap_or("1.0");
            let factor = parse_volume_factor(raw);
            ParsedFilter::Volume { factor }
        }

        // ── Audio resample ────────────────────────────────────────────────────
        "aresample" | "resample" => {
            let sr = get_pos_or_named(&node.positional_args, &node.named_args, "sample_rate", 0)
                .and_then(|s| s.parse::<u32>().ok())
                .unwrap_or(44100);
            ParsedFilter::Resample { sample_rate: sr }
        }

        // ── Compressor ────────────────────────────────────────────────────────
        "acompressor" | "compressor" => {
            let threshold = get_named(&node.named_args, "threshold")
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(-20.0);
            let ratio = get_named(&node.named_args, "ratio")
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(2.0);
            ParsedFilter::Compressor { threshold, ratio }
        }

        // ── Passthroughs (no-ops) ─────────────────────────────────────────────
        "null" | "anull" | "setpts" | "asetpts" | "format" | "colorspace" | "colormatrix"
        | "colorrange" | "setdar" | "setsar" | "setfield" | "fieldorder" | "tpad" | "apad" => {
            ParsedFilter::Passthrough
        }

        // ── Stored-but-unsupported ────────────────────────────────────────────
        name => {
            let args = reconstruct_args(&node.positional_args, &node.named_args);
            ParsedFilter::Unknown {
                name: name.to_string(),
                args,
            }
        }
    }
}

fn parse_scale(node: &FilterNode) -> ParsedFilter {
    // scale=W:H  or  scale=w=W:h=H  or  scale=W:H:flags=…
    let w = get_pos_or_named(&node.positional_args, &node.named_args, "w", 0)
        .and_then(parse_dim_str)
        .unwrap_or(-1);
    let h = get_pos_or_named(&node.positional_args, &node.named_args, "h", 1)
        .and_then(parse_dim_str)
        .unwrap_or(-1);
    ParsedFilter::Scale { w, h }
}

fn parse_crop(node: &FilterNode) -> ParsedFilter {
    let w = get_pos_or_named(&node.positional_args, &node.named_args, "w", 0)
        .and_then(parse_dim_str)
        .unwrap_or(0);
    let h = get_pos_or_named(&node.positional_args, &node.named_args, "h", 1)
        .and_then(parse_dim_str)
        .unwrap_or(0);
    let x = get_pos_or_named(&node.positional_args, &node.named_args, "x", 2)
        .and_then(parse_dim_str)
        .unwrap_or(0);
    let y = get_pos_or_named(&node.positional_args, &node.named_args, "y", 3)
        .and_then(parse_dim_str)
        .unwrap_or(0);
    ParsedFilter::Crop { w, h, x, y }
}

fn parse_eq(node: &FilterNode) -> ParsedFilter {
    let brightness = get_named(&node.named_args, "brightness")
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0);
    let contrast = get_named(&node.named_args, "contrast")
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(1.0);
    let saturation = get_named(&node.named_args, "saturation")
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(1.0);
    ParsedFilter::ColorCorrect {
        brightness,
        contrast,
        saturation,
    }
}

fn parse_loudnorm(node: &FilterNode) -> ParsedFilter {
    let integrated = get_named(&node.named_args, "I")
        .or_else(|| get_named(&node.named_args, "i"))
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(-23.0);
    let true_peak = get_named(&node.named_args, "TP")
        .or_else(|| get_named(&node.named_args, "tp"))
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(-2.0);
    let lra = get_named(&node.named_args, "LRA")
        .or_else(|| get_named(&node.named_args, "lra"))
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(7.0);
    ParsedFilter::LoudNorm {
        integrated,
        true_peak,
        lra,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Get a named arg by key.
fn get_named<'a>(named: &'a [(String, String)], key: &str) -> Option<&'a str> {
    named
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(key))
        .map(|(_, v)| v.as_str())
}

/// Get a value from positional args by index, falling back to a named arg.
fn get_pos_or_named<'a>(
    positional: &'a [String],
    named: &'a [(String, String)],
    named_key: &str,
    pos_idx: usize,
) -> Option<&'a str> {
    get_named(named, named_key).or_else(|| positional.get(pos_idx).map(String::as_str))
}

/// Parse dimension strings like `"1280"`, `"-1"`, `"iw/2"`.
/// Returns `None` only for expressions (which become `Some(-1)` for keep-aspect).
fn parse_dim_str(s: &str) -> Option<i32> {
    s.trim().parse::<i32>().ok()
}

/// Parse a volume string: `"0.5"`, `"6dB"`, `"-3dB"`.
fn parse_volume_factor(s: &str) -> f64 {
    if let Some(db_str) = s.strip_suffix("dB").or_else(|| s.strip_suffix("db")) {
        db_str
            .trim()
            .parse::<f64>()
            .map(|db| 10f64.powf(db / 20.0))
            .unwrap_or(1.0)
    } else {
        s.trim().parse::<f64>().unwrap_or(1.0)
    }
}

/// Reconstruct a compact args string for Unknown filters.
fn reconstruct_args(positional: &[String], named: &[(String, String)]) -> String {
    let mut parts: Vec<String> = positional.to_vec();
    for (k, v) in named {
        parts.push(format!("{}={}", k, v));
    }
    parts.join(":")
}

// ─────────────────────────────────────────────────────────────────────────────
// Low-level parser implementation
// ─────────────────────────────────────────────────────────────────────────────

/// Split a linear filter chain on `,` while respecting bracket and quote depth.
fn split_filter_chain(chain: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = String::new();
    let mut depth: usize = 0;
    let mut in_single = false;
    let mut in_double = false;

    for ch in chain.chars() {
        match ch {
            '\'' if !in_double => {
                in_single = !in_single;
                current.push(ch);
            }
            '"' if !in_single => {
                in_double = !in_double;
                current.push(ch);
            }
            '[' if !in_single && !in_double => {
                depth += 1;
                current.push(ch);
            }
            ']' if !in_single && !in_double && depth > 0 => {
                depth -= 1;
                current.push(ch);
            }
            ',' if !in_single && !in_double && depth == 0 => {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    result.push(trimmed);
                }
                current.clear();
            }
            _ => current.push(ch),
        }
    }

    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        result.push(trimmed);
    }

    result
}

/// Parse a single filter expression string into a [`FilterNode`].
fn parse_filter_node(expr: &str) -> anyhow::Result<FilterNode> {
    let expr = expr.trim();
    let (inputs, rest) = extract_labels(expr);
    let (filter_part, outputs) = split_trailing_labels(rest);

    let (name, args_str) = if let Some(eq) = filter_part.find('=') {
        let name = filter_part[..eq].trim().to_string();
        let args = filter_part[eq + 1..].trim();
        (name, args)
    } else {
        (filter_part.trim().to_string(), "")
    };

    if name.is_empty() {
        anyhow::bail!("empty filter name in: {:?}", expr);
    }

    let (positional_args, named_args) = parse_args(args_str);

    Ok(FilterNode {
        inputs,
        name,
        positional_args,
        named_args,
        outputs,
    })
}

/// Extract leading `[label]` tokens.
fn extract_labels(s: &str) -> (Vec<String>, &str) {
    let mut labels = Vec::new();
    let mut rest = s;

    while rest.starts_with('[') {
        if let Some(end) = rest.find(']') {
            labels.push(rest[1..end].to_string());
            rest = rest[end + 1..].trim_start();
        } else {
            break;
        }
    }

    (labels, rest)
}

/// Separate trailing `[label]` tokens.
fn split_trailing_labels(s: &str) -> (&str, Vec<String>) {
    let mut labels: Vec<String> = Vec::new();
    let mut end = s.len();

    loop {
        let sub = s[..end].trim_end();
        if !sub.ends_with(']') {
            break;
        }
        if let Some(open) = sub.rfind('[') {
            let label = sub[open + 1..sub.len() - 1].to_string();
            labels.push(label);
            end = open;
        } else {
            break;
        }
    }

    labels.reverse();
    (s[..end].trim_end(), labels)
}

/// Parse the argument portion of a filter expression.
fn parse_args(args_str: &str) -> (Vec<String>, Vec<(String, String)>) {
    if args_str.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let mut positional = Vec::new();
    let mut named = Vec::new();

    for part in args_str.split(':') {
        let part = part.trim();
        if let Some(eq) = part.find('=') {
            let key = part[..eq].trim().to_string();
            let val = part[eq + 1..].trim().to_string();
            named.push((key, val));
        } else if !part.is_empty() {
            positional.push(part.to_string());
        }
    }

    (positional, named)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_positional() {
        let filters = parse_filters("scale=1280:720");
        assert_eq!(filters.len(), 1);
        assert!(matches!(
            filters[0],
            ParsedFilter::Scale { w: 1280, h: 720 }
        ));
    }

    #[test]
    fn test_scale_named() {
        let filters = parse_filters("scale=w=640:h=480");
        assert!(matches!(filters[0], ParsedFilter::Scale { w: 640, h: 480 }));
    }

    #[test]
    fn test_crop() {
        let filters = parse_filters("crop=640:480:10:20");
        assert!(matches!(
            filters[0],
            ParsedFilter::Crop {
                w: 640,
                h: 480,
                x: 10,
                y: 20
            }
        ));
    }

    #[test]
    fn test_fps() {
        let filters = parse_filters("fps=30");
        assert!(matches!(filters[0], ParsedFilter::Fps { rate } if (rate - 30.0).abs() < 0.01));
    }

    #[test]
    fn test_hflip_vflip() {
        let filters = parse_filters("hflip,vflip");
        assert_eq!(filters.len(), 2);
        assert_eq!(filters[0], ParsedFilter::HFlip);
        assert_eq!(filters[1], ParsedFilter::VFlip);
    }

    #[test]
    fn test_yadif_deinterlace() {
        let filters = parse_filters("yadif");
        assert_eq!(filters[0], ParsedFilter::Deinterlace);
    }

    #[test]
    fn test_eq_color_correct() {
        let filters = parse_filters("eq=brightness=0.1:contrast=1.2:saturation=0.8");
        assert!(matches!(
            &filters[0],
            ParsedFilter::ColorCorrect { brightness, contrast, saturation }
            if (*brightness - 0.1).abs() < 0.01
                && (*contrast - 1.2).abs() < 0.01
                && (*saturation - 0.8).abs() < 0.01
        ));
    }

    #[test]
    fn test_lut3d_named() {
        let filters = parse_filters("lut3d=file=my_lut.cube");
        assert!(matches!(&filters[0], ParsedFilter::Lut3d { file } if file == "my_lut.cube"));
    }

    #[test]
    fn test_subtitles() {
        let filters = parse_filters("subtitles=filename=subs.srt");
        assert!(matches!(&filters[0], ParsedFilter::SubtitleBurnIn { file } if file == "subs.srt"));
    }

    #[test]
    fn test_loudnorm() {
        let filters = parse_filters("loudnorm=I=-23:TP=-2:LRA=7");
        assert!(matches!(
            &filters[0],
            ParsedFilter::LoudNorm { integrated, true_peak, lra }
            if (*integrated - -23.0).abs() < 0.01
                && (*true_peak - -2.0).abs() < 0.01
                && (*lra - 7.0).abs() < 0.01
        ));
    }

    #[test]
    fn test_volume_linear() {
        let filters = parse_filters("volume=0.5");
        assert!(
            matches!(&filters[0], ParsedFilter::Volume { factor } if (*factor - 0.5).abs() < 0.01)
        );
    }

    #[test]
    fn test_volume_db() {
        let filters = parse_filters("volume=6dB");
        // 6 dB ≈ 2.0 factor
        assert!(
            matches!(&filters[0], ParsedFilter::Volume { factor } if (*factor - 1.995).abs() < 0.01)
        );
    }

    #[test]
    fn test_aresample() {
        let filters = parse_filters("aresample=48000");
        assert!(matches!(
            &filters[0],
            ParsedFilter::Resample { sample_rate: 48000 }
        ));
    }

    #[test]
    fn test_compressor() {
        let filters = parse_filters("acompressor=threshold=-20:ratio=4");
        assert!(matches!(
            &filters[0],
            ParsedFilter::Compressor { threshold, ratio }
            if (*threshold - -20.0).abs() < 0.01 && (*ratio - 4.0).abs() < 0.01
        ));
    }

    #[test]
    fn test_null_passthrough() {
        let filters = parse_filters("null");
        assert_eq!(filters[0], ParsedFilter::Passthrough);
    }

    #[test]
    fn test_anull_passthrough() {
        let filters = parse_filters("anull");
        assert_eq!(filters[0], ParsedFilter::Passthrough);
    }

    #[test]
    fn test_unknown_filter() {
        let filters = parse_filters("overlay=10:20");
        assert!(matches!(&filters[0], ParsedFilter::Unknown { name, .. } if name == "overlay"));
    }

    #[test]
    fn test_filter_complex_semicolon() {
        let filters = parse_filters("[0:v]scale=1280:720[out];[0:a]volume=0.5[aout]");
        assert_eq!(filters.len(), 2);
        assert!(matches!(
            filters[0],
            ParsedFilter::Scale { w: 1280, h: 720 }
        ));
        assert!(matches!(filters[1], ParsedFilter::Volume { .. }));
    }

    #[test]
    fn test_labelled_graph_low_level() {
        let g = parse_filter_graph("[in]scale=1280:720[out]").expect("parse");
        assert_eq!(g.nodes[0].inputs, ["in"]);
        assert_eq!(g.nodes[0].outputs, ["out"]);
    }

    #[test]
    fn test_bwdif_deinterlace() {
        let filters = parse_filters("bwdif");
        assert_eq!(filters.len(), 1);
        assert!(
            matches!(&filters[0], ParsedFilter::Deinterlace),
            "bwdif should be Deinterlace"
        );
    }

    #[test]
    fn test_multiple_filters_chain_count() {
        let filters = parse_filters("scale=1280:720,fps=30,hflip");
        assert_eq!(filters.len(), 3);
        assert!(matches!(
            &filters[0],
            ParsedFilter::Scale { w: 1280, h: 720 }
        ));
        assert!(matches!(&filters[1], ParsedFilter::Fps { .. }));
        assert!(matches!(&filters[2], ParsedFilter::HFlip));
    }

    #[test]
    fn test_loudnorm_with_lra() {
        let filters = parse_filters("loudnorm=I=-23:TP=-1.5:LRA=11");
        assert_eq!(filters.len(), 1);
        match &filters[0] {
            ParsedFilter::LoudNorm {
                integrated,
                true_peak,
                lra,
            } => {
                assert!(
                    (*integrated - -23.0).abs() < 0.01,
                    "integrated should be -23"
                );
                assert!((*true_peak - -1.5).abs() < 0.01, "true_peak should be -1.5");
                assert!((*lra - 11.0).abs() < 0.01, "lra should be 11");
            }
            _ => panic!("expected LoudNorm"),
        }
    }

    #[test]
    fn test_volume_linear_factor() {
        let filters = parse_filters("volume=2.0");
        assert_eq!(filters.len(), 1);
        match &filters[0] {
            ParsedFilter::Volume { factor } => {
                assert!((*factor - 2.0).abs() < 0.01, "factor should be 2.0");
            }
            _ => panic!("expected Volume"),
        }
    }

    #[test]
    fn test_crop_with_offset() {
        let filters = parse_filters("crop=640:480:0:0");
        assert_eq!(filters.len(), 1);
        match &filters[0] {
            ParsedFilter::Crop { w, h, x, y } => {
                assert_eq!(*w, 640);
                assert_eq!(*h, 480);
                assert_eq!(*x, 0);
                assert_eq!(*y, 0);
            }
            _ => panic!("expected Crop"),
        }
    }

    #[test]
    fn test_aresample_filter() {
        let filters = parse_filters("aresample=48000");
        assert_eq!(filters.len(), 1);
        match &filters[0] {
            ParsedFilter::Resample { sample_rate } => {
                assert_eq!(*sample_rate, 48000);
            }
            _ => panic!("expected Resample"),
        }
    }

    #[test]
    fn test_subtitle_burnin_filename_key() {
        let filters = parse_filters("subtitles=filename=subs.srt");
        assert_eq!(filters.len(), 1);
        match &filters[0] {
            ParsedFilter::SubtitleBurnIn { file } => {
                assert_eq!(file, "subs.srt");
            }
            _ => panic!("expected SubtitleBurnIn"),
        }
    }

    #[test]
    fn test_lut3d_file_key() {
        let filters = parse_filters("lut3d=file=my_lut.cube");
        assert_eq!(filters.len(), 1);
        match &filters[0] {
            ParsedFilter::Lut3d { file } => {
                assert_eq!(file, "my_lut.cube");
            }
            _ => panic!("expected Lut3d"),
        }
    }

    #[test]
    fn test_unknown_filter_preserved_with_args() {
        let filters = parse_filters("someunknownfilter=x=1:y=2");
        assert_eq!(filters.len(), 1);
        match &filters[0] {
            ParsedFilter::Unknown { name, .. } => {
                assert_eq!(name, "someunknownfilter");
            }
            _ => panic!("expected Unknown"),
        }
    }

    #[test]
    fn test_rotate_filter() {
        let filters = parse_filters("rotate=angle=1.5708");
        assert_eq!(filters.len(), 1);
        match &filters[0] {
            ParsedFilter::Rotate { angle } => {
                assert!((*angle - 1.5708).abs() < 0.001, "angle should be ~pi/2");
            }
            _ => panic!("expected Rotate"),
        }
    }

    #[test]
    fn test_eq_color_correct_defaults() {
        // eq with no args should use defaults
        let filters = parse_filters("eq");
        assert_eq!(filters.len(), 1);
        match &filters[0] {
            ParsedFilter::ColorCorrect {
                brightness,
                contrast,
                saturation,
            } => {
                assert!(
                    (*brightness - 0.0).abs() < 0.01,
                    "default brightness should be 0"
                );
                assert!(
                    (*contrast - 1.0).abs() < 0.01,
                    "default contrast should be 1"
                );
                assert!(
                    (*saturation - 1.0).abs() < 0.01,
                    "default saturation should be 1"
                );
            }
            _ => panic!("expected ColorCorrect"),
        }
    }

    #[test]
    fn test_compressor_with_named_args() {
        let filters = parse_filters("acompressor=threshold=-18:ratio=3");
        assert_eq!(filters.len(), 1);
        match &filters[0] {
            ParsedFilter::Compressor { threshold, ratio } => {
                assert!((*threshold - -18.0).abs() < 0.01);
                assert!((*ratio - 3.0).abs() < 0.01);
            }
            _ => panic!("expected Compressor"),
        }
    }

    #[test]
    fn test_setpts_is_passthrough() {
        let filters = parse_filters("setpts=PTS-STARTPTS");
        assert_eq!(filters.len(), 1);
        // setpts is a known passthrough
        assert!(matches!(&filters[0], ParsedFilter::Passthrough));
    }

    #[test]
    fn test_empty_filter_string_gives_empty_vec() {
        let filters = parse_filters("");
        assert!(filters.is_empty(), "empty string should yield no filters");
    }

    #[test]
    fn test_scale_named_args() {
        let filters = parse_filters("scale=w=1920:h=1080");
        assert_eq!(filters.len(), 1);
        assert!(matches!(
            &filters[0],
            ParsedFilter::Scale { w: 1920, h: 1080 }
        ));
    }

    #[test]
    fn test_filter_complex_multi_chain() {
        // Two chains separated by semicolon
        let filters = parse_filters("[0:v]scale=1280:720[s];[s]fps=24[out]");
        assert_eq!(filters.len(), 2);
        assert!(matches!(
            &filters[0],
            ParsedFilter::Scale { w: 1280, h: 720 }
        ));
        assert!(matches!(&filters[1], ParsedFilter::Fps { rate } if (*rate - 24.0).abs() < 0.01));
    }

    #[test]
    fn test_low_level_parse_filter_string_compat() {
        // parse_filter_string is a legacy alias for parse_filter_graph
        let g = parse_filter_string("scale=640:480").expect("legacy parse");
        assert_eq!(g.nodes.len(), 1);
        assert_eq!(g.nodes[0].name, "scale");
    }
}
