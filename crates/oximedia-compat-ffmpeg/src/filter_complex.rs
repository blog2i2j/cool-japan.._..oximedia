//! Recursive-descent parser for the FFmpeg `-filter_complex` syntax.
//!
//! ## Grammar (BNF)
//!
//! ```text
//! filter_graph  := filter_chain (';' filter_chain)*
//! filter_chain  := input_pad* filter_node (filter_node)* output_pad*
//! input_pad     := '[' label ']'
//! output_pad    := '[' label ']'
//! filter_node   := name ('=' option_list)?
//! option_list   := option (',' option)*  |  option (':'  option)*
//! option        := key '=' value  |  value
//! name          := IDENTIFIER
//! label         := [^'\]' '\[' ';']+
//! ```
//!
//! ## Examples
//!
//! ```
//! use oximedia_compat_ffmpeg::filter_complex::FilterGraph;
//!
//! // Simple scale filter
//! let g = FilterGraph::parse("[in]scale=1280:720[out]").unwrap();
//! assert_eq!(g.chains.len(), 1);
//!
//! // Multi-chain with overlay
//! let g = FilterGraph::parse(
//!     "[0:v]scale=1920:1080[bg];[bg][1:v]overlay=x=10:y=10[out]"
//! ).unwrap();
//! assert_eq!(g.chains.len(), 2);
//! ```

/// Error type for `-filter_complex` parsing.
#[derive(Debug, thiserror::Error)]
pub enum FilterComplexError {
    /// The filter graph string contained an unexpected character or structure.
    #[error("unexpected character '{0}' at position {1}")]
    UnexpectedChar(char, usize),

    /// An unclosed `[` bracket was detected.
    #[error("unclosed '[' in pad label starting at position {0}")]
    UnclosedPad(usize),

    /// A filter name was expected but was not found.
    #[error("expected filter name at position {0}")]
    ExpectedFilterName(usize),

    /// A filter option had invalid key=value syntax.
    #[error("malformed filter option '{0}'")]
    MalformedOption(String),

    /// Empty filter graph string.
    #[error("empty filter graph string")]
    Empty,
}

// ─────────────────────────────────────────────────────────────────────────────
// AST types
// ─────────────────────────────────────────────────────────────────────────────

/// A single key=value option (or positional value) inside a filter.
///
/// Examples:
/// - `scale=1280:720`  → `{key: None, value: "1280:720"}` (positional)
/// - `scale=w=1280:h=720` → `{key: Some("w"), value: "1280"}`, `{key: Some("h"), value: "720"}`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilterOption {
    /// Option key (`None` for positional values).
    pub key: Option<String>,
    /// Option value.
    pub value: String,
}

impl FilterOption {
    /// Create a positional option (no key).
    pub fn positional(value: impl Into<String>) -> Self {
        Self {
            key: None,
            value: value.into(),
        }
    }

    /// Create a named key=value option.
    pub fn named(key: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            key: Some(key.into()),
            value: value.into(),
        }
    }
}

impl std::fmt::Display for FilterOption {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.key {
            Some(k) => write!(f, "{}={}", k, self.value),
            None => write!(f, "{}", self.value),
        }
    }
}

/// A single filter node within a filter chain.
///
/// Corresponds to one filter in a chain such as `scale=1280:720` or
/// `overlay=x=10:y=10`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Filter {
    /// Filter name (e.g. `"scale"`, `"overlay"`, `"amix"`).
    pub name: String,
    /// Ordered list of options for this filter.
    pub options: Vec<FilterOption>,
}

impl Filter {
    /// Create a new filter node with no options.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            options: Vec::new(),
        }
    }

    /// Create a filter node with a positional options string (unparsed).
    pub fn with_options(name: impl Into<String>, options: Vec<FilterOption>) -> Self {
        Self {
            name: name.into(),
            options,
        }
    }
}

impl std::fmt::Display for Filter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)?;
        if !self.options.is_empty() {
            write!(f, "=")?;
            let opts: Vec<String> = self.options.iter().map(|o| o.to_string()).collect();
            write!(f, "{}", opts.join(":"))?;
        }
        Ok(())
    }
}

/// A filter chain — a linear sequence of filters with input and output pad labels.
///
/// In `-filter_complex`, chains are separated by `;`.
/// Within a chain, filters are applied left to right; the output of each
/// feeds the input of the next.
///
/// Example: `[0:v]scale=1280:720,unsharp=5:5:1.0[out]`
/// - `input_pads`: `["0:v"]`
/// - `filters`: `[scale=1280:720, unsharp=5:5:1.0]`
/// - `output_pads`: `["out"]`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilterChain {
    /// Named input pads feeding this chain (labels inside `[…]`).
    pub input_pads: Vec<String>,
    /// Ordered sequence of filter nodes.
    pub filters: Vec<Filter>,
    /// Named output pads produced by this chain.
    pub output_pads: Vec<String>,
}

impl FilterChain {
    /// Return `true` if the chain contains at least one filter node.
    pub fn is_empty(&self) -> bool {
        self.filters.is_empty()
    }
}

impl std::fmt::Display for FilterChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for pad in &self.input_pads {
            write!(f, "[{}]", pad)?;
        }
        let filters: Vec<String> = self.filters.iter().map(|fi| fi.to_string()).collect();
        write!(f, "{}", filters.join(","))?;
        for pad in &self.output_pads {
            write!(f, "[{}]", pad)?;
        }
        Ok(())
    }
}

/// A complete filter graph (the value of a `-filter_complex` argument).
///
/// A filter graph is a semicolon-separated list of [`FilterChain`]s.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilterGraph {
    /// Ordered list of filter chains.
    pub chains: Vec<FilterChain>,
}

impl FilterGraph {
    /// Parse a `-filter_complex` string into a [`FilterGraph`] AST.
    ///
    /// Returns [`FilterComplexError`] if the string is malformed.
    pub fn parse(input: &str) -> Result<Self, FilterComplexError> {
        let input = input.trim();
        if input.is_empty() {
            return Err(FilterComplexError::Empty);
        }

        let mut parser = Parser::new(input);
        parser.parse_graph()
    }

    /// Return the total number of filter nodes across all chains.
    pub fn filter_count(&self) -> usize {
        self.chains.iter().map(|c| c.filters.len()).sum()
    }

    /// Return all unique output pad labels across the graph.
    pub fn output_labels(&self) -> Vec<&str> {
        self.chains
            .iter()
            .flat_map(|c| c.output_pads.iter().map(|p| p.as_str()))
            .collect()
    }

    /// Return all unique input pad labels across the graph.
    pub fn input_labels(&self) -> Vec<&str> {
        self.chains
            .iter()
            .flat_map(|c| c.input_pads.iter().map(|p| p.as_str()))
            .collect()
    }
}

impl std::fmt::Display for FilterGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let chains: Vec<String> = self.chains.iter().map(|c| c.to_string()).collect();
        write!(f, "{}", chains.join(";"))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Recursive-descent parser
// ─────────────────────────────────────────────────────────────────────────────

struct Parser<'a> {
    input: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(s: &'a str) -> Self {
        Self {
            input: s.as_bytes(),
            pos: 0,
        }
    }

    // ── Primitives ────────────────────────────────────────────────────────

    fn peek(&self) -> Option<char> {
        self.input.get(self.pos).map(|&b| b as char)
    }

    fn advance(&mut self) {
        if self.pos < self.input.len() {
            self.pos += 1;
        }
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_ascii_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn expect_char(&mut self, expected: char) -> Result<(), FilterComplexError> {
        self.skip_whitespace();
        match self.peek() {
            Some(c) if c == expected => {
                self.advance();
                Ok(())
            }
            Some(c) => Err(FilterComplexError::UnexpectedChar(c, self.pos)),
            None => Err(FilterComplexError::UnexpectedChar('\0', self.pos)),
        }
    }

    // ── Top-level graph ───────────────────────────────────────────────────

    fn parse_graph(&mut self) -> Result<FilterGraph, FilterComplexError> {
        let mut chains = Vec::new();

        loop {
            self.skip_whitespace();
            if self.pos >= self.input.len() {
                break;
            }

            let chain = self.parse_chain()?;
            chains.push(chain);

            self.skip_whitespace();
            match self.peek() {
                Some(';') => {
                    self.advance();
                }
                None => break,
                Some(c) => {
                    return Err(FilterComplexError::UnexpectedChar(c, self.pos));
                }
            }
        }

        if chains.is_empty() {
            return Err(FilterComplexError::Empty);
        }

        Ok(FilterGraph { chains })
    }

    // ── Chain ─────────────────────────────────────────────────────────────

    fn parse_chain(&mut self) -> Result<FilterChain, FilterComplexError> {
        self.skip_whitespace();

        // Collect leading pad labels
        let input_pads = self.parse_pad_labels()?;

        // Collect filter nodes (comma-separated within a chain)
        let mut filters = Vec::new();
        loop {
            self.skip_whitespace();
            match self.peek() {
                None | Some(';') => break,
                Some('[') => break, // start of output pads
                _ => {}
            }

            let filter = self.parse_filter_node()?;
            filters.push(filter);

            self.skip_whitespace();
            match self.peek() {
                Some(',') => {
                    self.advance();
                }
                _ => break,
            }
        }

        if filters.is_empty() {
            return Err(FilterComplexError::ExpectedFilterName(self.pos));
        }

        // Collect trailing pad labels (output pads)
        let output_pads = self.parse_pad_labels()?;

        Ok(FilterChain {
            input_pads,
            filters,
            output_pads,
        })
    }

    // ── Pad label list: '[label1][label2]…' ──────────────────────────────

    fn parse_pad_labels(&mut self) -> Result<Vec<String>, FilterComplexError> {
        let mut pads = Vec::new();
        loop {
            self.skip_whitespace();
            if self.peek() != Some('[') {
                break;
            }
            let pad = self.parse_one_pad()?;
            pads.push(pad);
        }
        Ok(pads)
    }

    fn parse_one_pad(&mut self) -> Result<String, FilterComplexError> {
        let start = self.pos;
        self.expect_char('[')?;
        let mut label = String::new();
        loop {
            match self.peek() {
                Some(']') => {
                    self.advance();
                    return Ok(label);
                }
                Some('[') | Some(';') | None => {
                    return Err(FilterComplexError::UnclosedPad(start));
                }
                Some(c) => {
                    label.push(c);
                    self.advance();
                }
            }
        }
    }

    // ── Single filter node: 'name' or 'name=options' ─────────────────────

    fn parse_filter_node(&mut self) -> Result<Filter, FilterComplexError> {
        self.skip_whitespace();

        // Filter name: alphanumeric + '_' + '-'
        let name_start = self.pos;
        let mut name = String::new();
        while let Some(c) = self.peek() {
            if c.is_alphanumeric() || c == '_' || c == '-' {
                name.push(c);
                self.advance();
            } else {
                break;
            }
        }

        if name.is_empty() {
            return Err(FilterComplexError::ExpectedFilterName(name_start));
        }

        // Optional options block
        self.skip_whitespace();
        if self.peek() != Some('=') {
            return Ok(Filter::new(name));
        }
        self.advance(); // consume '='

        // Parse the option string — up to ',' (next filter), ';' (next chain),
        // '[' (output pad), or end.
        let options = self.parse_filter_options()?;
        Ok(Filter::with_options(name, options))
    }

    // ── Filter options ────────────────────────────────────────────────────
    //
    // FFmpeg supports two separators: `:` (most common) and `,` (less common,
    // but `,` also separates filter nodes in a chain, so we must be careful).
    //
    // Strategy: collect the entire option substring up to a chain separator
    // (`','` at depth 0 when not in a `'=`' value, `';'`, `'['`, or EOF), then
    // split by `:` to get individual option tokens.
    //
    // For option tokens that contain `=` they are key=value; otherwise positional.

    fn parse_filter_options(&mut self) -> Result<Vec<FilterOption>, FilterComplexError> {
        // Collect the raw options string, respecting that ',' terminates the
        // current filter within a chain, but may appear inside a quoted string.
        let raw = self.collect_option_string();
        parse_options_string(&raw)
    }

    /// Collect everything up to the next chain-level separator.
    fn collect_option_string(&mut self) -> String {
        let mut s = String::new();
        let mut depth = 0i32; // track nested parens/brackets

        while let Some(c) = self.peek() {
            match c {
                '(' | '{' => {
                    depth += 1;
                    s.push(c);
                    self.advance();
                }
                ')' | '}' => {
                    if depth > 0 {
                        depth -= 1;
                    }
                    s.push(c);
                    self.advance();
                }
                '[' | ']' | ';' if depth == 0 => break,
                ',' if depth == 0 => break,
                _ => {
                    s.push(c);
                    self.advance();
                }
            }
        }
        s
    }
}

/// Split a raw options string (e.g. `"w=1280:h=720"`) into individual [`FilterOption`]s.
fn parse_options_string(raw: &str) -> Result<Vec<FilterOption>, FilterComplexError> {
    if raw.is_empty() {
        return Ok(Vec::new());
    }

    // Determine the separator: use ':' as primary separator. A ',' appearing
    // here means the FFmpeg comma-sep style was used at the option level; we
    // do NOT support mixed separators, but we allow a single positional value
    // that may itself contain ':' (e.g. video size "1280:720").
    //
    // Simple heuristic: if any token looks like 'key=value', treat ':' as separator.
    // Otherwise treat the whole string as a single positional option.

    let has_kv = raw.contains('=');
    let tokens: Vec<&str> = if has_kv {
        // Split by ':' but be careful: a value may contain ':' after the '='.
        // We do a greedy split and try to reassemble ambiguous segments.
        split_options_by_colon(raw)
    } else {
        // Pure positional: the whole thing is one option (e.g. "1280:720").
        vec![raw]
    };

    let mut options = Vec::with_capacity(tokens.len());
    for token in tokens {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        if let Some(eq_pos) = token.find('=') {
            let key = token[..eq_pos].trim().to_string();
            let value = token[eq_pos + 1..].trim().to_string();
            if key.is_empty() {
                return Err(FilterComplexError::MalformedOption(token.to_string()));
            }
            options.push(FilterOption::named(key, value));
        } else {
            options.push(FilterOption::positional(token));
        }
    }

    Ok(options)
}

/// Split an options string at `:`, being careful not to break inside
/// `key=value` pairs where the value itself contains `:`.
fn split_options_by_colon(raw: &str) -> Vec<&str> {
    let bytes = raw.as_bytes();
    let mut result: Vec<&str> = Vec::new();
    let mut start = 0usize;
    let mut in_value = false; // true once we've seen `=` in the current token

    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'=' => {
                in_value = true;
                i += 1;
            }
            b':' => {
                if !in_value {
                    result.push(&raw[start..i]);
                    start = i + 1;
                    in_value = false;
                } else {
                    // We are inside a value — this ':' might be a sub-separator
                    // within the value (e.g. h264's `profile=high:level=4.1`).
                    // Peek ahead: if the substring after ':' looks like 'key=',
                    // start a new token.
                    let rest = &bytes[i + 1..];
                    let looks_like_key = rest.iter().position(|&b| b == b'=').map_or(false, |eq| {
                        rest[..eq]
                            .iter()
                            .all(|&b| b.is_ascii_alphanumeric() || b == b'_')
                    });
                    if looks_like_key {
                        result.push(&raw[start..i]);
                        start = i + 1;
                        in_value = false;
                    }
                    i += 1;
                }
            }
            _ => {
                i += 1;
            }
        }
    }
    if start <= raw.len() {
        result.push(&raw[start..]);
    }
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_scale() {
        let g = FilterGraph::parse("[in]scale=1280:720[out]").expect("parse");
        assert_eq!(g.chains.len(), 1);
        let chain = &g.chains[0];
        assert_eq!(chain.input_pads, vec!["in"]);
        assert_eq!(chain.output_pads, vec!["out"]);
        assert_eq!(chain.filters.len(), 1);
        assert_eq!(chain.filters[0].name, "scale");
    }

    #[test]
    fn test_two_chain_overlay() {
        let g = FilterGraph::parse("[0:v]scale=1920:1080[bg];[bg][1:v]overlay=x=10:y=10[out]")
            .expect("parse");
        assert_eq!(g.chains.len(), 2);
        assert_eq!(g.chains[1].input_pads, vec!["bg", "1:v"]);
    }

    #[test]
    fn test_filter_chain_no_pads() {
        let g = FilterGraph::parse("scale=1280:720").expect("parse");
        assert_eq!(g.chains.len(), 1);
        assert!(g.chains[0].input_pads.is_empty());
        assert!(g.chains[0].output_pads.is_empty());
    }

    #[test]
    fn test_filter_with_named_options() {
        let g = FilterGraph::parse("overlay=x=10:y=20").expect("parse");
        let filter = &g.chains[0].filters[0];
        assert_eq!(filter.name, "overlay");
        // options should contain x=10 and y=20
        assert!(
            filter
                .options
                .iter()
                .any(|o| o.key.as_deref() == Some("x") && o.value == "10"),
            "x option missing"
        );
        assert!(
            filter
                .options
                .iter()
                .any(|o| o.key.as_deref() == Some("y") && o.value == "20"),
            "y option missing"
        );
    }

    #[test]
    fn test_multi_filter_chain() {
        let g = FilterGraph::parse("[in]scale=1280:720,unsharp=5:5:1.0[out]").expect("parse");
        assert_eq!(g.chains[0].filters.len(), 2);
        assert_eq!(g.chains[0].filters[0].name, "scale");
        assert_eq!(g.chains[0].filters[1].name, "unsharp");
    }

    #[test]
    fn test_amix() {
        let g = FilterGraph::parse("[0:a][1:a]amix=inputs=2:duration=first[aout]").expect("parse");
        assert_eq!(g.chains[0].input_pads, vec!["0:a", "1:a"]);
        assert_eq!(g.chains[0].output_pads, vec!["aout"]);
        let filter = &g.chains[0].filters[0];
        assert_eq!(filter.name, "amix");
    }

    #[test]
    fn test_filter_count() {
        let g = FilterGraph::parse("[0:v]scale=640:360[s1];[s1]unsharp[out]").expect("parse");
        assert_eq!(g.filter_count(), 2);
    }

    #[test]
    fn test_output_labels() {
        let g = FilterGraph::parse("[0:v]scale=640:360[v1];[0:a]aformat=fltp[a1]").expect("parse");
        let labels = g.output_labels();
        assert!(labels.contains(&"v1"));
        assert!(labels.contains(&"a1"));
    }

    #[test]
    fn test_empty_error() {
        assert!(matches!(
            FilterGraph::parse(""),
            Err(FilterComplexError::Empty)
        ));
    }

    #[test]
    fn test_display_roundtrip() {
        let input = "[in]scale=1280:720[out]";
        let g = FilterGraph::parse(input).expect("parse");
        // The roundtrip should produce the same string.
        let s = g.to_string();
        // Re-parse the output to verify it's still valid.
        let g2 = FilterGraph::parse(&s).expect("re-parse");
        assert_eq!(g.chains.len(), g2.chains.len());
    }
}
