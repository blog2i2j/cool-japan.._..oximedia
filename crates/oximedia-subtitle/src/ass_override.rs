//! Full ASS/SSA override tag parser.
//!
//! Parses inline override blocks `{...}` found in ASS dialogue lines into
//! structured `Vec<OverrideTag>` representations. Supports the full range of
//! ASS override tags including positioning, colors, transforms, and text styling.
//!
//! # Example
//!
//! ```
//! use oximedia_subtitle::ass_override::{OverrideTagParser, OverrideTag};
//!
//! let tags = OverrideTagParser::parse(r"{\pos(320,240)\fs48\c&H00FF00&}Hello");
//! assert!(tags.is_ok());
//! let tags = tags.expect("test");
//! assert!(tags.iter().any(|t| matches!(t, OverrideTag::FontSize(_))));
//! ```

use crate::error::{SubtitleError, SubtitleResult};

/// A parsed ASS override tag.
#[derive(Clone, Debug, PartialEq)]
pub enum OverrideTag {
    /// `\pos(x, y)` — subtitle position.
    Pos {
        /// X coordinate.
        x: f64,
        /// Y coordinate.
        y: f64,
    },
    /// `\move(x1, y1, x2, y2[, t1, t2])` — animated move.
    Move {
        /// Start X.
        x1: f64,
        /// Start Y.
        y1: f64,
        /// End X.
        x2: f64,
        /// End Y.
        y2: f64,
        /// Optional start time in ms.
        t1: Option<i64>,
        /// Optional end time in ms.
        t2: Option<i64>,
    },
    /// `\org(x, y)` — rotation origin.
    Org {
        /// X coordinate.
        x: f64,
        /// Y coordinate.
        y: f64,
    },
    /// `\fad(fadein, fadeout)` — simple fade.
    Fad {
        /// Fade-in duration in ms.
        fade_in: i64,
        /// Fade-out duration in ms.
        fade_out: i64,
    },
    /// `\clip(x1, y1, x2, y2)` — rectangular clip.
    Clip {
        /// Left.
        x1: f64,
        /// Top.
        y1: f64,
        /// Right.
        x2: f64,
        /// Bottom.
        y2: f64,
    },
    /// `\iclip(x1, y1, x2, y2)` — inverse rectangular clip.
    InverseClip {
        /// Left.
        x1: f64,
        /// Top.
        y1: f64,
        /// Right.
        x2: f64,
        /// Bottom.
        y2: f64,
    },
    /// `\an<n>` — numpad alignment (1-9).
    Alignment(u8),
    /// `\fn<name>` — font name.
    FontName(String),
    /// `\fs<size>` — font size.
    FontSize(f64),
    /// `\c&H<BBGGRR>&` or `\1c` — primary color.
    PrimaryColor(AssColor),
    /// `\2c` — secondary color.
    SecondaryColor(AssColor),
    /// `\3c` — outline color.
    OutlineColor(AssColor),
    /// `\4c` — shadow color.
    ShadowColor(AssColor),
    /// `\alpha&H<AA>&` — global alpha.
    Alpha(u8),
    /// `\1a&H<AA>&` — primary alpha.
    PrimaryAlpha(u8),
    /// `\2a` — secondary alpha.
    SecondaryAlpha(u8),
    /// `\3a` — outline alpha.
    OutlineAlpha(u8),
    /// `\4a` — shadow alpha.
    ShadowAlpha(u8),
    /// `\blur<strength>` — Gaussian blur.
    Blur(f64),
    /// `\be<strength>` — edge blur (integer).
    BlurEdges(i32),
    /// `\bord<size>` — border/outline width.
    Border(f64),
    /// `\shad<depth>` — shadow depth.
    Shadow(f64),
    /// `\frx<degrees>` — X-axis rotation.
    RotationX(f64),
    /// `\fry<degrees>` — Y-axis rotation.
    RotationY(f64),
    /// `\frz<degrees>` or `\fr<degrees>` — Z-axis rotation.
    RotationZ(f64),
    /// `\fscx<percent>` — X scale.
    ScaleX(f64),
    /// `\fscy<percent>` — Y scale.
    ScaleY(f64),
    /// `\b<weight>` — bold (0 or 1, or weight like 700).
    Bold(i32),
    /// `\i<0|1>` — italic.
    Italic(bool),
    /// `\u<0|1>` — underline.
    Underline(bool),
    /// `\s<0|1>` — strikeout.
    Strikeout(bool),
    /// `\r[style]` — reset to style or default.
    Reset(Option<String>),
}

/// An ASS color in BGR order (as stored in ASS format).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AssColor {
    /// Blue component.
    pub b: u8,
    /// Green component.
    pub g: u8,
    /// Red component.
    pub r: u8,
}

impl AssColor {
    /// Create a new ASS color.
    #[must_use]
    pub const fn new(b: u8, g: u8, r: u8) -> Self {
        Self { b, g, r }
    }

    /// Parse from `&HBBGGRR&` or `&HBBGGRR` format.
    fn parse(s: &str) -> SubtitleResult<Self> {
        let s = s.trim_start_matches('&').trim_start_matches('h').trim_start_matches('H');
        let s = s.trim_end_matches('&');

        // Pad with leading zeros if necessary
        let hex = if s.len() < 6 {
            format!("{s:0>6}")
        } else {
            // Take last 6 chars (skip alpha if present, e.g. &H00FFFFFF)
            let start = if s.len() > 6 { s.len() - 6 } else { 0 };
            s[start..].to_string()
        };

        if hex.len() < 6 {
            return Err(SubtitleError::InvalidColor(format!("invalid ASS color: {s}")));
        }

        let b = u8::from_str_radix(&hex[0..2], 16)
            .map_err(|_| SubtitleError::InvalidColor(format!("invalid ASS color: {s}")))?;
        let g = u8::from_str_radix(&hex[2..4], 16)
            .map_err(|_| SubtitleError::InvalidColor(format!("invalid ASS color: {s}")))?;
        let r = u8::from_str_radix(&hex[4..6], 16)
            .map_err(|_| SubtitleError::InvalidColor(format!("invalid ASS color: {s}")))?;

        Ok(Self { b, g, r })
    }
}

/// Parser for ASS override tag blocks.
pub struct OverrideTagParser;

impl OverrideTagParser {
    /// Parse a dialogue line that may contain override blocks `{...}`.
    ///
    /// Returns all tags found in all override blocks. Text outside blocks is ignored.
    ///
    /// # Errors
    ///
    /// Returns an error only for structurally broken input (e.g., unbalanced braces
    /// in critical positions). Individual malformed tags are silently skipped for
    /// resilience against real-world subtitle files.
    pub fn parse(input: &str) -> SubtitleResult<Vec<OverrideTag>> {
        let mut tags = Vec::new();

        // Extract all override blocks
        let mut remaining = input;
        while let Some(start) = remaining.find('{') {
            if let Some(end) = remaining[start..].find('}') {
                let block = &remaining[start + 1..start + end];
                Self::parse_block(block, &mut tags);
                remaining = &remaining[start + end + 1..];
            } else {
                // Unmatched brace — skip it gracefully
                break;
            }
        }

        Ok(tags)
    }

    /// Parse a single override block (content between `{` and `}`).
    fn parse_block(block: &str, tags: &mut Vec<OverrideTag>) {
        let mut pos = 0;
        let bytes = block.as_bytes();
        let len = bytes.len();

        while pos < len {
            if bytes[pos] != b'\\' {
                pos += 1;
                continue;
            }
            pos += 1; // skip backslash
            if pos >= len {
                break;
            }

            // Find the tag name and value
            let tag_start = pos;
            let parsed = Self::try_parse_tag(&block[tag_start..]);
            if let Some((tag, consumed)) = parsed {
                tags.push(tag);
                pos = tag_start + consumed;
            } else {
                // Skip to next backslash for recovery
                pos = tag_start;
                while pos < len && bytes[pos] != b'\\' {
                    pos += 1;
                }
            }
        }
    }

    /// Try to parse a single tag starting at the given position.
    /// Returns the tag and number of characters consumed, or None if unrecognized.
    fn try_parse_tag(s: &str) -> Option<(OverrideTag, usize)> {
        // Ordered by specificity (longer prefixes first)
        if let Some(r) = Self::try_parse_clip(s, "iclip", true) {
            return Some(r);
        }
        if let Some(r) = Self::try_parse_clip(s, "clip", false) {
            return Some(r);
        }
        if let Some(r) = Self::try_parse_move(s) {
            return Some(r);
        }
        if let Some(r) = Self::try_parse_pos(s) {
            return Some(r);
        }
        if let Some(r) = Self::try_parse_org(s) {
            return Some(r);
        }
        if let Some(r) = Self::try_parse_fad(s) {
            return Some(r);
        }
        if let Some(r) = Self::try_parse_fscx(s) {
            return Some(r);
        }
        if let Some(r) = Self::try_parse_fscy(s) {
            return Some(r);
        }
        if let Some(r) = Self::try_parse_frx(s) {
            return Some(r);
        }
        if let Some(r) = Self::try_parse_fry(s) {
            return Some(r);
        }
        if let Some(r) = Self::try_parse_frz(s) {
            return Some(r);
        }
        if let Some(r) = Self::try_parse_fn(s) {
            return Some(r);
        }
        if let Some(r) = Self::try_parse_fs(s) {
            return Some(r);
        }
        if let Some(r) = Self::try_parse_an(s) {
            return Some(r);
        }
        if let Some(r) = Self::try_parse_numbered_color(s) {
            return Some(r);
        }
        if let Some(r) = Self::try_parse_color_c(s) {
            return Some(r);
        }
        if let Some(r) = Self::try_parse_numbered_alpha(s) {
            return Some(r);
        }
        if let Some(r) = Self::try_parse_alpha(s) {
            return Some(r);
        }
        if let Some(r) = Self::try_parse_blur(s) {
            return Some(r);
        }
        if let Some(r) = Self::try_parse_be(s) {
            return Some(r);
        }
        if let Some(r) = Self::try_parse_bord(s) {
            return Some(r);
        }
        if let Some(r) = Self::try_parse_shad(s) {
            return Some(r);
        }
        if let Some(r) = Self::try_parse_bold(s) {
            return Some(r);
        }
        if let Some(r) = Self::try_parse_italic(s) {
            return Some(r);
        }
        if let Some(r) = Self::try_parse_underline(s) {
            return Some(r);
        }
        if let Some(r) = Self::try_parse_strikeout(s) {
            return Some(r);
        }
        if let Some(r) = Self::try_parse_reset(s) {
            return Some(r);
        }

        None
    }

    // ---- Parenthesized tag helpers ----

    /// Parse `(n1, n2, ...)` returning floats and total consumed chars.
    fn parse_paren_floats(s: &str) -> Option<(Vec<f64>, usize)> {
        if !s.starts_with('(') {
            return None;
        }
        let close = s.find(')')?;
        let inner = &s[1..close];
        let parts: Vec<f64> = inner
            .split(',')
            .filter_map(|p| p.trim().parse::<f64>().ok())
            .collect();
        if parts.is_empty() {
            return None;
        }
        Some((parts, close + 1))
    }

    /// Parse `(n1, n2, ...)` returning i64 values.
    fn parse_paren_ints(s: &str) -> Option<(Vec<i64>, usize)> {
        if !s.starts_with('(') {
            return None;
        }
        let close = s.find(')')?;
        let inner = &s[1..close];
        let parts: Vec<i64> = inner
            .split(',')
            .filter_map(|p| p.trim().parse::<i64>().ok())
            .collect();
        if parts.is_empty() {
            return None;
        }
        Some((parts, close + 1))
    }

    fn try_parse_pos(s: &str) -> Option<(OverrideTag, usize)> {
        let rest = s.strip_prefix("pos")?;
        let (vals, consumed) = Self::parse_paren_floats(rest)?;
        if vals.len() >= 2 {
            Some((OverrideTag::Pos { x: vals[0], y: vals[1] }, 3 + consumed))
        } else {
            None
        }
    }

    fn try_parse_move(s: &str) -> Option<(OverrideTag, usize)> {
        let rest = s.strip_prefix("move")?;
        let (vals, consumed) = Self::parse_paren_floats(rest)?;
        if vals.len() >= 4 {
            let t1 = vals.get(4).map(|v| *v as i64);
            let t2 = vals.get(5).map(|v| *v as i64);
            Some((
                OverrideTag::Move {
                    x1: vals[0],
                    y1: vals[1],
                    x2: vals[2],
                    y2: vals[3],
                    t1,
                    t2,
                },
                4 + consumed,
            ))
        } else {
            None
        }
    }

    fn try_parse_org(s: &str) -> Option<(OverrideTag, usize)> {
        let rest = s.strip_prefix("org")?;
        let (vals, consumed) = Self::parse_paren_floats(rest)?;
        if vals.len() >= 2 {
            Some((OverrideTag::Org { x: vals[0], y: vals[1] }, 3 + consumed))
        } else {
            None
        }
    }

    fn try_parse_fad(s: &str) -> Option<(OverrideTag, usize)> {
        let rest = s.strip_prefix("fad")?;
        let (vals, consumed) = Self::parse_paren_ints(rest)?;
        if vals.len() >= 2 {
            Some((
                OverrideTag::Fad {
                    fade_in: vals[0],
                    fade_out: vals[1],
                },
                3 + consumed,
            ))
        } else {
            None
        }
    }

    fn try_parse_clip(s: &str, prefix: &str, inverse: bool) -> Option<(OverrideTag, usize)> {
        let rest = s.strip_prefix(prefix)?;
        let (vals, consumed) = Self::parse_paren_floats(rest)?;
        if vals.len() >= 4 {
            let tag = if inverse {
                OverrideTag::InverseClip {
                    x1: vals[0],
                    y1: vals[1],
                    x2: vals[2],
                    y2: vals[3],
                }
            } else {
                OverrideTag::Clip {
                    x1: vals[0],
                    y1: vals[1],
                    x2: vals[2],
                    y2: vals[3],
                }
            };
            Some((tag, prefix.len() + consumed))
        } else {
            None
        }
    }

    // ---- Simple value tags ----

    /// Consume a numeric value (int or float) right after the prefix.
    fn consume_number(s: &str) -> Option<(f64, usize)> {
        let mut end = 0;
        let bytes = s.as_bytes();
        // Allow leading minus
        if end < bytes.len() && bytes[end] == b'-' {
            end += 1;
        }
        while end < bytes.len() && (bytes[end].is_ascii_digit() || bytes[end] == b'.') {
            end += 1;
        }
        if end == 0 || (end == 1 && bytes[0] == b'-') {
            return None;
        }
        let val = s[..end].parse::<f64>().ok()?;
        Some((val, end))
    }

    fn consume_int(s: &str) -> Option<(i32, usize)> {
        let mut end = 0;
        let bytes = s.as_bytes();
        if end < bytes.len() && bytes[end] == b'-' {
            end += 1;
        }
        while end < bytes.len() && bytes[end].is_ascii_digit() {
            end += 1;
        }
        if end == 0 || (end == 1 && bytes[0] == b'-') {
            return None;
        }
        let val = s[..end].parse::<i32>().ok()?;
        Some((val, end))
    }

    fn try_parse_an(s: &str) -> Option<(OverrideTag, usize)> {
        let rest = s.strip_prefix("an")?;
        let (val, consumed) = Self::consume_int(rest)?;
        if (1..=9).contains(&val) {
            Some((OverrideTag::Alignment(val as u8), 2 + consumed))
        } else {
            None
        }
    }

    fn try_parse_fn(s: &str) -> Option<(OverrideTag, usize)> {
        let rest = s.strip_prefix("fn")?;
        // Font name goes until next `\` or end of block
        let end = rest.find('\\').unwrap_or(rest.len());
        let name = rest[..end].trim().to_string();
        if name.is_empty() {
            return None;
        }
        Some((OverrideTag::FontName(name), 2 + end))
    }

    fn try_parse_fs(s: &str) -> Option<(OverrideTag, usize)> {
        // Must not match fscx/fscy
        if s.starts_with("fscx") || s.starts_with("fscy") {
            return None;
        }
        let rest = s.strip_prefix("fs")?;
        let (val, consumed) = Self::consume_number(rest)?;
        Some((OverrideTag::FontSize(val), 2 + consumed))
    }

    fn try_parse_fscx(s: &str) -> Option<(OverrideTag, usize)> {
        let rest = s.strip_prefix("fscx")?;
        let (val, consumed) = Self::consume_number(rest)?;
        Some((OverrideTag::ScaleX(val), 4 + consumed))
    }

    fn try_parse_fscy(s: &str) -> Option<(OverrideTag, usize)> {
        let rest = s.strip_prefix("fscy")?;
        let (val, consumed) = Self::consume_number(rest)?;
        Some((OverrideTag::ScaleY(val), 4 + consumed))
    }

    fn try_parse_frx(s: &str) -> Option<(OverrideTag, usize)> {
        let rest = s.strip_prefix("frx")?;
        let (val, consumed) = Self::consume_number(rest)?;
        Some((OverrideTag::RotationX(val), 3 + consumed))
    }

    fn try_parse_fry(s: &str) -> Option<(OverrideTag, usize)> {
        let rest = s.strip_prefix("fry")?;
        let (val, consumed) = Self::consume_number(rest)?;
        Some((OverrideTag::RotationY(val), 3 + consumed))
    }

    fn try_parse_frz(s: &str) -> Option<(OverrideTag, usize)> {
        // Also matches bare `\fr`
        let (prefix_len, rest) = if s.starts_with("frz") {
            (3, &s[3..])
        } else if s.starts_with("fr") {
            // bare \fr is alias for \frz — but must not match frx/fry
            if s.starts_with("frx") || s.starts_with("fry") {
                return None;
            }
            (2, &s[2..])
        } else {
            return None;
        };
        let (val, consumed) = Self::consume_number(rest)?;
        Some((OverrideTag::RotationZ(val), prefix_len + consumed))
    }

    // ---- Color tags ----

    /// Consume a color value like `&HBBGGRR&` or `&HBBGGRR`.
    fn consume_color_value(s: &str) -> Option<(AssColor, usize)> {
        // Find the extent: starts with & or hex digits
        let mut end = 0;
        let bytes = s.as_bytes();
        while end < bytes.len()
            && (bytes[end] == b'&'
                || bytes[end] == b'H'
                || bytes[end] == b'h'
                || bytes[end].is_ascii_hexdigit())
        {
            end += 1;
        }
        if end == 0 {
            return None;
        }
        let color = AssColor::parse(&s[..end]).ok()?;
        Some((color, end))
    }

    fn try_parse_color_c(s: &str) -> Option<(OverrideTag, usize)> {
        let rest = s.strip_prefix('c')?;
        // Must not be a digit after 'c' (that would be \clip or numbered color handled elsewhere)
        if rest.starts_with('l') {
            return None;
        }
        let (color, consumed) = Self::consume_color_value(rest)?;
        Some((OverrideTag::PrimaryColor(color), 1 + consumed))
    }

    fn try_parse_numbered_color(s: &str) -> Option<(OverrideTag, usize)> {
        if s.len() < 2 {
            return None;
        }
        let n = s.as_bytes()[0];
        if !(b'1'..=b'4').contains(&n) {
            return None;
        }
        let rest = s[1..].strip_prefix('c')?;
        let (color, consumed) = Self::consume_color_value(rest)?;
        let tag = match n {
            b'1' => OverrideTag::PrimaryColor(color),
            b'2' => OverrideTag::SecondaryColor(color),
            b'3' => OverrideTag::OutlineColor(color),
            b'4' => OverrideTag::ShadowColor(color),
            _ => return None,
        };
        Some((tag, 2 + consumed))
    }

    // ---- Alpha tags ----

    fn consume_alpha_value(s: &str) -> Option<(u8, usize)> {
        let mut end = 0;
        let bytes = s.as_bytes();
        while end < bytes.len()
            && (bytes[end] == b'&' || bytes[end] == b'H' || bytes[end] == b'h' || bytes[end].is_ascii_hexdigit())
        {
            end += 1;
        }
        if end == 0 {
            return None;
        }
        let raw = s[..end]
            .trim_start_matches('&')
            .trim_start_matches('H')
            .trim_start_matches('h')
            .trim_end_matches('&');
        let val = u8::from_str_radix(raw, 16).ok()?;
        Some((val, end))
    }

    fn try_parse_alpha(s: &str) -> Option<(OverrideTag, usize)> {
        let rest = s.strip_prefix("alpha")?;
        let (val, consumed) = Self::consume_alpha_value(rest)?;
        Some((OverrideTag::Alpha(val), 5 + consumed))
    }

    fn try_parse_numbered_alpha(s: &str) -> Option<(OverrideTag, usize)> {
        if s.len() < 2 {
            return None;
        }
        let n = s.as_bytes()[0];
        if !(b'1'..=b'4').contains(&n) {
            return None;
        }
        let rest = s[1..].strip_prefix('a')?;
        let (val, consumed) = Self::consume_alpha_value(rest)?;
        let tag = match n {
            b'1' => OverrideTag::PrimaryAlpha(val),
            b'2' => OverrideTag::SecondaryAlpha(val),
            b'3' => OverrideTag::OutlineAlpha(val),
            b'4' => OverrideTag::ShadowAlpha(val),
            _ => return None,
        };
        Some((tag, 2 + consumed))
    }

    // ---- Blur / Border / Shadow ----

    fn try_parse_blur(s: &str) -> Option<(OverrideTag, usize)> {
        let rest = s.strip_prefix("blur")?;
        let (val, consumed) = Self::consume_number(rest)?;
        Some((OverrideTag::Blur(val), 4 + consumed))
    }

    fn try_parse_be(s: &str) -> Option<(OverrideTag, usize)> {
        let rest = s.strip_prefix("be")?;
        let (val, consumed) = Self::consume_int(rest)?;
        Some((OverrideTag::BlurEdges(val), 2 + consumed))
    }

    fn try_parse_bord(s: &str) -> Option<(OverrideTag, usize)> {
        // Must not match bold (\b)
        let rest = s.strip_prefix("bord")?;
        let (val, consumed) = Self::consume_number(rest)?;
        Some((OverrideTag::Border(val), 4 + consumed))
    }

    fn try_parse_shad(s: &str) -> Option<(OverrideTag, usize)> {
        let rest = s.strip_prefix("shad")?;
        let (val, consumed) = Self::consume_number(rest)?;
        Some((OverrideTag::Shadow(val), 4 + consumed))
    }

    // ---- Text style toggles ----

    fn try_parse_bold(s: &str) -> Option<(OverrideTag, usize)> {
        // \b must not match \blur, \bord, \be
        if s.starts_with("blur") || s.starts_with("bord") || s.starts_with("be") {
            return None;
        }
        let rest = s.strip_prefix('b')?;
        let (val, consumed) = Self::consume_int(rest)?;
        Some((OverrideTag::Bold(val), 1 + consumed))
    }

    fn try_parse_italic(s: &str) -> Option<(OverrideTag, usize)> {
        if s.starts_with("iclip") {
            return None;
        }
        let rest = s.strip_prefix('i')?;
        let (val, consumed) = Self::consume_int(rest)?;
        Some((OverrideTag::Italic(val != 0), 1 + consumed))
    }

    fn try_parse_underline(s: &str) -> Option<(OverrideTag, usize)> {
        let rest = s.strip_prefix('u')?;
        let (val, consumed) = Self::consume_int(rest)?;
        Some((OverrideTag::Underline(val != 0), 1 + consumed))
    }

    fn try_parse_strikeout(s: &str) -> Option<(OverrideTag, usize)> {
        if s.starts_with("shad") {
            return None;
        }
        let rest = s.strip_prefix('s')?;
        let (val, consumed) = Self::consume_int(rest)?;
        Some((OverrideTag::Strikeout(val != 0), 1 + consumed))
    }

    fn try_parse_reset(s: &str) -> Option<(OverrideTag, usize)> {
        let rest = s.strip_prefix('r')?;
        // Style name goes until next `\` or end
        let end = rest.find('\\').unwrap_or(rest.len());
        let name = rest[..end].trim();
        let style = if name.is_empty() {
            None
        } else {
            Some(name.to_string())
        };
        Some((OverrideTag::Reset(style), 1 + end))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_pos() {
        let tags = OverrideTagParser::parse(r"{\pos(320,240)}Hello")
            .expect("should parse");
        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0], OverrideTag::Pos { x: 320.0, y: 240.0 });
    }

    #[test]
    fn test_parse_move_basic() {
        let tags = OverrideTagParser::parse(r"{\move(0,0,100,200)}")
            .expect("should parse");
        assert_eq!(
            tags[0],
            OverrideTag::Move {
                x1: 0.0,
                y1: 0.0,
                x2: 100.0,
                y2: 200.0,
                t1: None,
                t2: None,
            }
        );
    }

    #[test]
    fn test_parse_move_with_timing() {
        let tags = OverrideTagParser::parse(r"{\move(0,0,100,200,500,1500)}")
            .expect("should parse");
        assert_eq!(
            tags[0],
            OverrideTag::Move {
                x1: 0.0,
                y1: 0.0,
                x2: 100.0,
                y2: 200.0,
                t1: Some(500),
                t2: Some(1500),
            }
        );
    }

    #[test]
    fn test_parse_org() {
        let tags = OverrideTagParser::parse(r"{\org(640,360)}")
            .expect("should parse");
        assert_eq!(tags[0], OverrideTag::Org { x: 640.0, y: 360.0 });
    }

    #[test]
    fn test_parse_fad() {
        let tags = OverrideTagParser::parse(r"{\fad(500,800)}")
            .expect("should parse");
        assert_eq!(
            tags[0],
            OverrideTag::Fad {
                fade_in: 500,
                fade_out: 800,
            }
        );
    }

    #[test]
    fn test_parse_clip_and_iclip() {
        let tags = OverrideTagParser::parse(r"{\clip(10,20,300,400)\iclip(50,60,200,300)}")
            .expect("should parse");
        assert_eq!(tags.len(), 2);
        assert_eq!(
            tags[0],
            OverrideTag::Clip {
                x1: 10.0,
                y1: 20.0,
                x2: 300.0,
                y2: 400.0,
            }
        );
        assert_eq!(
            tags[1],
            OverrideTag::InverseClip {
                x1: 50.0,
                y1: 60.0,
                x2: 200.0,
                y2: 300.0,
            }
        );
    }

    #[test]
    fn test_parse_font_name_and_size() {
        let tags = OverrideTagParser::parse(r"{\fnArial\fs48}")
            .expect("should parse");
        assert_eq!(tags.len(), 2);
        assert_eq!(tags[0], OverrideTag::FontName("Arial".to_string()));
        assert_eq!(tags[1], OverrideTag::FontSize(48.0));
    }

    #[test]
    fn test_parse_alignment() {
        let tags = OverrideTagParser::parse(r"{\an8}")
            .expect("should parse");
        assert_eq!(tags[0], OverrideTag::Alignment(8));
    }

    #[test]
    fn test_parse_colors() {
        let tags = OverrideTagParser::parse(r"{\c&H00FF00&\3c&HFF0000&}")
            .expect("should parse");
        assert_eq!(tags.len(), 2);
        assert_eq!(
            tags[0],
            OverrideTag::PrimaryColor(AssColor::new(0x00, 0xFF, 0x00))
        );
        assert_eq!(
            tags[1],
            OverrideTag::OutlineColor(AssColor::new(0xFF, 0x00, 0x00))
        );
    }

    #[test]
    fn test_parse_alpha() {
        let tags = OverrideTagParser::parse(r"{\alpha&H80&\1a&HFF&\4a&H40&}")
            .expect("should parse");
        assert_eq!(tags.len(), 3);
        assert_eq!(tags[0], OverrideTag::Alpha(0x80));
        assert_eq!(tags[1], OverrideTag::PrimaryAlpha(0xFF));
        assert_eq!(tags[2], OverrideTag::ShadowAlpha(0x40));
    }

    #[test]
    fn test_parse_blur_border_shadow() {
        let tags = OverrideTagParser::parse(r"{\blur2.5\be1\bord3\shad1.5}")
            .expect("should parse");
        assert_eq!(tags.len(), 4);
        assert_eq!(tags[0], OverrideTag::Blur(2.5));
        assert_eq!(tags[1], OverrideTag::BlurEdges(1));
        assert_eq!(tags[2], OverrideTag::Border(3.0));
        assert_eq!(tags[3], OverrideTag::Shadow(1.5));
    }

    #[test]
    fn test_parse_rotation_tags() {
        let tags = OverrideTagParser::parse(r"{\frx30\fry45\frz90}")
            .expect("should parse");
        assert_eq!(tags.len(), 3);
        assert_eq!(tags[0], OverrideTag::RotationX(30.0));
        assert_eq!(tags[1], OverrideTag::RotationY(45.0));
        assert_eq!(tags[2], OverrideTag::RotationZ(90.0));
    }

    #[test]
    fn test_parse_scale() {
        let tags = OverrideTagParser::parse(r"{\fscx150\fscy200}")
            .expect("should parse");
        assert_eq!(tags.len(), 2);
        assert_eq!(tags[0], OverrideTag::ScaleX(150.0));
        assert_eq!(tags[1], OverrideTag::ScaleY(200.0));
    }

    #[test]
    fn test_parse_text_style_toggles() {
        let tags = OverrideTagParser::parse(r"{\b1\i1\u1\s1}")
            .expect("should parse");
        assert_eq!(tags.len(), 4);
        assert_eq!(tags[0], OverrideTag::Bold(1));
        assert_eq!(tags[1], OverrideTag::Italic(true));
        assert_eq!(tags[2], OverrideTag::Underline(true));
        assert_eq!(tags[3], OverrideTag::Strikeout(true));
    }

    #[test]
    fn test_parse_reset() {
        let tags = OverrideTagParser::parse(r"{\r}")
            .expect("should parse");
        assert_eq!(tags[0], OverrideTag::Reset(None));

        let tags = OverrideTagParser::parse(r"{\rAlternate}")
            .expect("should parse");
        assert_eq!(
            tags[0],
            OverrideTag::Reset(Some("Alternate".to_string()))
        );
    }

    #[test]
    fn test_multiple_blocks() {
        let tags = OverrideTagParser::parse(r"{\b1}Bold{\b0} Normal{\i1}Italic")
            .expect("should parse");
        assert_eq!(tags.len(), 3);
        assert_eq!(tags[0], OverrideTag::Bold(1));
        assert_eq!(tags[1], OverrideTag::Bold(0));
        assert_eq!(tags[2], OverrideTag::Italic(true));
    }

    #[test]
    fn test_malformed_recovery() {
        // Unknown tags should be skipped
        let tags = OverrideTagParser::parse(r"{\xyzgarbage\fs24\b1}")
            .expect("should parse");
        assert!(tags.len() >= 2);
        assert!(tags.iter().any(|t| matches!(t, OverrideTag::FontSize(v) if (*v - 24.0).abs() < f64::EPSILON)));
        assert!(tags.iter().any(|t| matches!(t, OverrideTag::Bold(1))));
    }

    #[test]
    fn test_empty_and_no_override() {
        let tags = OverrideTagParser::parse("No overrides here")
            .expect("should parse");
        assert!(tags.is_empty());

        let tags = OverrideTagParser::parse(r"{}")
            .expect("should parse");
        assert!(tags.is_empty());
    }

    #[test]
    fn test_bold_weight_700() {
        let tags = OverrideTagParser::parse(r"{\b700}")
            .expect("should parse");
        assert_eq!(tags[0], OverrideTag::Bold(700));
    }

    #[test]
    fn test_negative_rotation() {
        let tags = OverrideTagParser::parse(r"{\frz-45}")
            .expect("should parse");
        assert_eq!(tags[0], OverrideTag::RotationZ(-45.0));
    }
}
