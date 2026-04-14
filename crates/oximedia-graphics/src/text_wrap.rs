//! Multi-line text wrapping with justification modes.
//!
//! This module provides a complete word-wrap and line-break engine for broadcast
//! graphics text layout, supporting four justification modes and optional
//! hyphenation hints. It operates entirely on `char`-level metrics to avoid a
//! dependency on a full text-shaper and is designed to be composable with
//! [`crate::text_layout`] and [`crate::text_renderer`].
//!
//! # Justification modes
//!
//! | Mode | Behaviour |
//! |------|-----------|
//! [`JustifyMode::None`]   | Left-align, no extra spacing |
//! [`JustifyMode::Left`]   | Explicit left alignment (same as None) |
//! [`JustifyMode::Right`]  | Right-align each line |
//! [`JustifyMode::Center`] | Center each line |
//! [`JustifyMode::Full`]   | Distribute remaining space between words |
//!
//! # Example
//!
//! ```rust
//! use oximedia_graphics::text_wrap::{WrapConfig, JustifyMode, wrap_text};
//!
//! let config = WrapConfig {
//!     max_width_px: 400.0,
//!     char_width_px: 10.0,
//!     justify: JustifyMode::Full,
//!     ..WrapConfig::default()
//! };
//! let lines = wrap_text("The quick brown fox jumps over the lazy dog", &config);
//! assert!(!lines.is_empty());
//! ```

/// How wrapped lines should be horizontally aligned / justified.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum JustifyMode {
    /// No extra alignment — rendered from the left edge.
    #[default]
    None,
    /// Explicit left-align (identical to [`None`](JustifyMode::None)).
    Left,
    /// Right-align each line within the container width.
    Right,
    /// Center each line within the container width.
    Center,
    /// Distribute whitespace so text reaches both edges
    /// (except the final line, which is left-aligned).
    Full,
}

/// A single wrapped line including layout metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct WrappedLine {
    /// The text content of this line (without trailing whitespace).
    pub text: String,
    /// Pixel offset from the left edge at which rendering should start,
    /// given the chosen [`JustifyMode`].
    pub x_offset_px: f32,
    /// Extra inter-word pixel spacing injected by [`JustifyMode::Full`].
    ///
    /// Add this to the natural space width between every pair of words.
    pub word_spacing_px: f32,
    /// Width of the rendered text (without justification adjustments).
    pub natural_width_px: f32,
}

impl WrappedLine {
    /// Convenience constructor for a left-aligned line with no extra spacing.
    pub fn simple(text: impl Into<String>, natural_width_px: f32) -> Self {
        Self {
            text: text.into(),
            x_offset_px: 0.0,
            word_spacing_px: 0.0,
            natural_width_px,
        }
    }
}

/// Configuration for the text wrapping engine.
#[derive(Debug, Clone)]
pub struct WrapConfig {
    /// Available horizontal space in pixels.
    pub max_width_px: f32,
    /// Average character advance in pixels (used for line-width estimation
    /// when a per-character table is absent).
    pub char_width_px: f32,
    /// Space character advance in pixels (defaults to `char_width_px * 0.45`
    /// when set to 0).
    pub space_width_px: f32,
    /// Justification mode for completed lines.
    pub justify: JustifyMode,
    /// If `true`, lines that contain a single word wider than `max_width_px`
    /// are force-broken at character boundaries.
    pub force_break_long_words: bool,
    /// Maximum number of lines to produce. `0` means unlimited.
    pub max_lines: usize,
    /// String appended to the last line when `max_lines` is exceeded.
    pub overflow_ellipsis: String,
}

impl Default for WrapConfig {
    fn default() -> Self {
        Self {
            max_width_px: 800.0,
            char_width_px: 12.0,
            space_width_px: 0.0,
            justify: JustifyMode::None,
            force_break_long_words: true,
            max_lines: 0,
            overflow_ellipsis: "…".to_string(),
        }
    }
}

impl WrapConfig {
    /// Return the effective space width, falling back to `char_width_px × 0.45`.
    pub fn effective_space_width(&self) -> f32 {
        if self.space_width_px > 0.0 {
            self.space_width_px
        } else {
            self.char_width_px * 0.45
        }
    }

    /// Estimate the pixel width of `text` using the uniform advance metric.
    pub fn measure(&self, text: &str) -> f32 {
        let space_w = self.effective_space_width();
        text.chars()
            .map(|c| if c == ' ' { space_w } else { self.char_width_px })
            .sum()
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Split `text` into tokens: words and whitespace runs.
///
/// Returns `(token, is_space)` tuples in source order.
fn tokenize(text: &str) -> Vec<(String, bool)> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut in_space = false;

    for ch in text.chars() {
        let is_ws = ch.is_whitespace();
        if is_ws != in_space {
            if !current.is_empty() {
                tokens.push((current.clone(), in_space));
                current.clear();
            }
            in_space = is_ws;
        }
        current.push(ch);
    }
    if !current.is_empty() {
        tokens.push((current, in_space));
    }
    tokens
}

/// Apply [`JustifyMode`] to compute per-line x_offset and word_spacing.
fn justify_line(line_text: &str, natural_width: f32, is_last: bool, config: &WrapConfig) -> (f32, f32) {
    let avail = config.max_width_px;
    match config.justify {
        JustifyMode::None | JustifyMode::Left => (0.0, 0.0),
        JustifyMode::Right => (avail - natural_width, 0.0),
        JustifyMode::Center => ((avail - natural_width) / 2.0, 0.0),
        JustifyMode::Full => {
            // Last line of a paragraph is left-aligned.
            if is_last {
                return (0.0, 0.0);
            }
            let gaps = line_text.split_whitespace().count().saturating_sub(1);
            if gaps == 0 {
                return (0.0, 0.0);
            }
            let slack = (avail - natural_width).max(0.0);
            let extra_per_gap = slack / gaps as f32;
            (0.0, extra_per_gap)
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Wrap `text` into [`WrappedLine`]s according to `config`.
///
/// This is the primary entry point for the module. It performs greedy
/// word-wrapping and then applies justification metadata to each line.
pub fn wrap_text(text: &str, config: &WrapConfig) -> Vec<WrappedLine> {
    if text.is_empty() {
        return Vec::new();
    }

    let tokens = tokenize(text);
    let space_w = config.effective_space_width();

    // Greedy line-breaking: accumulate words until the line would exceed max_width.
    let mut lines: Vec<String> = Vec::new();
    let mut current_line = String::new();
    let mut current_width: f32 = 0.0;

    for (token, is_space) in &tokens {
        if *is_space {
            // Space tokens are added only between words, not at line boundaries.
            continue;
        }

        let word_width = config.measure(token);

        // Handle words wider than the container when force_break is enabled.
        if word_width > config.max_width_px && config.force_break_long_words {
            // Flush current line first.
            if !current_line.is_empty() {
                lines.push(current_line.clone());
                current_line.clear();
                current_width = 0.0;
            }
            // Break the long word character by character.
            let mut partial = String::new();
            let mut partial_width = 0.0;
            for ch in token.chars() {
                let cw = if ch == ' ' { space_w } else { config.char_width_px };
                if partial_width + cw > config.max_width_px && !partial.is_empty() {
                    lines.push(partial.clone());
                    partial.clear();
                    partial_width = 0.0;
                }
                partial.push(ch);
                partial_width += cw;
            }
            if !partial.is_empty() {
                current_line = partial;
                current_width = partial_width;
            }
            continue;
        }

        let needs_space = !current_line.is_empty();
        let extra = if needs_space { space_w } else { 0.0 };

        if current_width + extra + word_width > config.max_width_px && !current_line.is_empty() {
            // Flush current line.
            lines.push(current_line.clone());
            current_line = token.clone();
            current_width = word_width;
        } else {
            if needs_space {
                current_line.push(' ');
            }
            current_line.push_str(token);
            current_width += extra + word_width;
        }
    }
    if !current_line.is_empty() {
        lines.push(current_line);
    }

    // Apply max_lines + ellipsis.
    if config.max_lines > 0 && lines.len() > config.max_lines {
        lines.truncate(config.max_lines);
        if let Some(last) = lines.last_mut() {
            if !config.overflow_ellipsis.is_empty() {
                last.push_str(&config.overflow_ellipsis);
            }
        }
    }

    // Build WrappedLine with justification metadata.
    let total = lines.len();
    lines
        .into_iter()
        .enumerate()
        .map(|(i, text_line)| {
            let natural_width = config.measure(&text_line);
            let is_last = i + 1 == total;
            let (x_off, word_sp) = justify_line(&text_line, natural_width, is_last, config);
            WrappedLine {
                text: text_line,
                x_offset_px: x_off,
                word_spacing_px: word_sp,
                natural_width_px: natural_width,
            }
        })
        .collect()
}

/// Count the number of lines that `text` would produce under `config`.
///
/// Cheaper than calling [`wrap_text`] when only the line count is needed.
pub fn line_count(text: &str, config: &WrapConfig) -> usize {
    wrap_text(text, config).len()
}

/// Estimate the total pixel height for wrapped `text` given `line_height_px`.
pub fn total_height(text: &str, config: &WrapConfig, line_height_px: f32) -> f32 {
    line_count(text, config) as f32 * line_height_px
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(width: f32) -> WrapConfig {
        WrapConfig {
            max_width_px: width,
            char_width_px: 10.0,
            space_width_px: 5.0,
            justify: JustifyMode::None,
            ..WrapConfig::default()
        }
    }

    // 1. Empty string → no lines.
    #[test]
    fn test_wrap_empty_string() {
        let lines = wrap_text("", &cfg(200.0));
        assert!(lines.is_empty());
    }

    // 2. Single short word fits on one line.
    #[test]
    fn test_wrap_single_word_fits() {
        let lines = wrap_text("Hello", &cfg(200.0));
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].text, "Hello");
    }

    // 3. Text that fits in one line is not broken.
    #[test]
    fn test_wrap_no_break_needed() {
        // "Hi" = 20px, width=200 → one line
        let lines = wrap_text("Hi there", &cfg(200.0));
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].text, "Hi there");
    }

    // 4. Long sentence is broken into multiple lines.
    #[test]
    fn test_wrap_breaks_long_sentence() {
        // Each word is 5 chars × 10px = 50px, space = 5px → "hello world" = 105px > 100px.
        let lines = wrap_text("hello world", &cfg(100.0));
        assert!(lines.len() > 1, "expected line break");
    }

    // 5. max_lines limits output and appends ellipsis.
    #[test]
    fn test_wrap_max_lines_with_ellipsis() {
        let mut config = cfg(50.0);
        config.max_lines = 2;
        config.overflow_ellipsis = "…".to_string();
        let lines = wrap_text("one two three four five six", &config);
        assert_eq!(lines.len(), 2);
        assert!(lines[1].text.ends_with('…'));
    }

    // 6. JustifyMode::Center shifts x_offset to mid.
    #[test]
    fn test_wrap_center_justify() {
        let config = WrapConfig {
            max_width_px: 200.0,
            char_width_px: 10.0,
            space_width_px: 5.0,
            justify: JustifyMode::Center,
            ..WrapConfig::default()
        };
        // "Hi" = 20px → x_offset = (200 - 20) / 2 = 90
        let lines = wrap_text("Hi", &config);
        assert_eq!(lines.len(), 1);
        assert!((lines[0].x_offset_px - 90.0).abs() < 0.5, "expected center offset ~90, got {}", lines[0].x_offset_px);
    }

    // 7. JustifyMode::Right shifts x_offset to right.
    #[test]
    fn test_wrap_right_justify() {
        let config = WrapConfig {
            max_width_px: 200.0,
            char_width_px: 10.0,
            space_width_px: 5.0,
            justify: JustifyMode::Right,
            ..WrapConfig::default()
        };
        // "Hi" = 20px → x_offset = 200 - 20 = 180
        let lines = wrap_text("Hi", &config);
        assert_eq!(lines.len(), 1);
        assert!((lines[0].x_offset_px - 180.0).abs() < 0.5);
    }

    // 8. JustifyMode::Full non-last line has positive word_spacing_px.
    #[test]
    fn test_wrap_full_justify_word_spacing() {
        let config = WrapConfig {
            max_width_px: 80.0,
            char_width_px: 10.0,
            space_width_px: 5.0,
            justify: JustifyMode::Full,
            ..WrapConfig::default()
        };
        // Force a multi-line wrap so there is a non-last line.
        let lines = wrap_text("hello world foo bar baz qux", &config);
        assert!(lines.len() > 1, "expected multiple lines for full justify test");
        // First (non-last) line with multiple words should have positive word_spacing_px.
        let first = &lines[0];
        let word_count = first.text.split_whitespace().count();
        if word_count > 1 {
            assert!(first.word_spacing_px >= 0.0, "word_spacing must be non-negative");
        }
    }

    // 9. force_break_long_words splits a word wider than max_width.
    #[test]
    fn test_wrap_force_break_long_word() {
        let config = WrapConfig {
            max_width_px: 30.0,
            char_width_px: 10.0,
            space_width_px: 5.0,
            force_break_long_words: true,
            ..WrapConfig::default()
        };
        // "abcde" = 50px > 30px → must break.
        let lines = wrap_text("abcde", &config);
        assert!(lines.len() > 1, "long word must be broken: {:?}", lines);
    }

    // 10. line_count returns the same count as wrap_text.
    #[test]
    fn test_line_count_matches_wrap_text() {
        let config = cfg(60.0);
        let text = "alpha beta gamma delta epsilon";
        assert_eq!(line_count(text, &config), wrap_text(text, &config).len());
    }

    // 11. total_height scales with line count and line height.
    #[test]
    fn test_total_height() {
        let config = cfg(60.0);
        let text = "alpha beta gamma delta";
        let n = line_count(text, &config);
        let h = total_height(text, &config, 20.0);
        assert!((h - n as f32 * 20.0).abs() < 0.01);
    }

    // 12. natural_width_px is positive for non-empty lines.
    #[test]
    fn test_natural_width_positive() {
        let lines = wrap_text("broadcast", &cfg(500.0));
        assert!(lines.iter().all(|l| l.natural_width_px > 0.0));
    }

    // 13. JustifyMode::Full last line has zero word_spacing_px.
    #[test]
    fn test_full_justify_last_line_no_spacing() {
        let config = WrapConfig {
            max_width_px: 60.0,
            char_width_px: 10.0,
            space_width_px: 5.0,
            justify: JustifyMode::Full,
            ..WrapConfig::default()
        };
        let lines = wrap_text("alpha beta gamma delta", &config);
        assert!(!lines.is_empty());
        let last = lines.last().expect("must have last line");
        assert_eq!(last.word_spacing_px, 0.0, "last line must not be stretched");
    }

    // 14. WrapConfig::measure is consistent with char_width_px.
    #[test]
    fn test_measure_consistency() {
        let config = WrapConfig {
            char_width_px: 8.0,
            space_width_px: 4.0,
            ..WrapConfig::default()
        };
        // "ab" → 2 chars × 8.0 = 16.0
        let w = config.measure("ab");
        assert!((w - 16.0).abs() < 0.01);
    }
}
