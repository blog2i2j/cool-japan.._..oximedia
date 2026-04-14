//! WebVTT region definitions and extended cue settings with builder API.
//!
//! Provides [`WebVttRegion`] and [`WebVttCueSettings`] with full builder patterns,
//! serialization, and parsing for the WebVTT specification.
//!
//! ## Example
//! ```
//! use oximedia_subtitle::formats::webvtt_regions::{WebVttRegion, WebVttCueSettings};
//!
//! let region = WebVttRegion::new("bottom")
//!     .with_width(40.0)
//!     .with_lines(3)
//!     .with_scroll_up();
//! assert_eq!(region.scroll, oximedia_subtitle::formats::webvtt_regions::WebVttScroll::Up);
//!
//! let settings = WebVttCueSettings::parse("line:5 align:start");
//! assert_eq!(settings.align, oximedia_subtitle::formats::webvtt_regions::WebVttAlign::Start);
//! ```

// ============================================================================
// Region types
// ============================================================================

/// Scroll direction for a WebVTT region.
#[derive(Debug, Clone, PartialEq)]
pub enum WebVttScroll {
    /// No scrolling — region is static.
    None,
    /// Lines scroll upward as new lines arrive.
    Up,
}

impl Default for WebVttScroll {
    fn default() -> Self {
        Self::None
    }
}

/// A WebVTT region definition (W3C WebVTT spec §3.5).
///
/// Regions allow cues to be placed in named areas of the viewport.
#[derive(Debug, Clone)]
pub struct WebVttRegion {
    /// Unique identifier for this region.
    pub id: String,
    /// Width of the region as a percentage of the viewport (0.0–100.0).
    pub width_pct: f32,
    /// Maximum number of lines visible at once.
    pub lines: u32,
    /// Region anchor point (x%, y%) within the region box.
    pub region_anchor: (f32, f32),
    /// Viewport anchor point (x%, y%) in the viewport coordinate space.
    pub viewport_anchor: (f32, f32),
    /// Scroll direction.
    pub scroll: WebVttScroll,
}

impl WebVttRegion {
    /// Create a new region with the given id and sensible defaults.
    ///
    /// Defaults: width=100%, lines=3, region_anchor=(0,100), viewport_anchor=(0,100), scroll=None.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            width_pct: 100.0,
            lines: 3,
            region_anchor: (0.0, 100.0),
            viewport_anchor: (0.0, 100.0),
            scroll: WebVttScroll::None,
        }
    }

    /// Set the width percentage of this region.
    #[must_use]
    pub fn with_width(mut self, pct: f32) -> Self {
        self.width_pct = pct.clamp(0.0, 100.0);
        self
    }

    /// Set the number of visible lines.
    #[must_use]
    pub fn with_lines(mut self, lines: u32) -> Self {
        self.lines = lines;
        self
    }

    /// Set the region anchor (x%, y%).
    #[must_use]
    pub fn with_region_anchor(mut self, x: f32, y: f32) -> Self {
        self.region_anchor = (x, y);
        self
    }

    /// Set the viewport anchor (x%, y%).
    #[must_use]
    pub fn with_viewport_anchor(mut self, x: f32, y: f32) -> Self {
        self.viewport_anchor = (x, y);
        self
    }

    /// Enable scroll-up behaviour.
    #[must_use]
    pub fn with_scroll_up(mut self) -> Self {
        self.scroll = WebVttScroll::Up;
        self
    }

    /// Serialise this region to WebVTT format.
    #[must_use]
    pub fn to_webvtt(&self) -> String {
        let mut out = String::from("REGION\n");
        out.push_str(&format!("id:{}\n", self.id));
        out.push_str(&format!("width:{:.1}%\n", self.width_pct));
        out.push_str(&format!("lines:{}\n", self.lines));
        out.push_str(&format!(
            "regionanchor:{:.1}%,{:.1}%\n",
            self.region_anchor.0, self.region_anchor.1
        ));
        out.push_str(&format!(
            "viewportanchor:{:.1}%,{:.1}%\n",
            self.viewport_anchor.0, self.viewport_anchor.1
        ));
        if self.scroll == WebVttScroll::Up {
            out.push_str("scroll:up\n");
        }
        out
    }
}

// ============================================================================
// Cue settings types
// ============================================================================

/// Vertical text writing direction for WebVTT cues.
#[derive(Debug, Clone, PartialEq)]
pub enum WebVttVertical {
    /// Normal horizontal text (default).
    Horizontal,
    /// Vertical columns flow left-to-right ("lr").
    LeftToRight,
    /// Vertical columns flow right-to-left ("rl").
    RightToLeft,
}

impl Default for WebVttVertical {
    fn default() -> Self {
        Self::Horizontal
    }
}

/// Line positioning for a WebVTT cue.
#[derive(Debug, Clone, PartialEq)]
pub enum WebVttLine {
    /// Browser determines line placement automatically.
    Auto,
    /// Snap-to-lines integer (positive = from top, negative = from bottom).
    Number(i32),
    /// Percentage distance from the top of the video.
    Percentage(f32),
}

impl Default for WebVttLine {
    fn default() -> Self {
        Self::Auto
    }
}

/// Text alignment within a WebVTT cue box.
#[derive(Debug, Clone, PartialEq)]
pub enum WebVttAlign {
    /// Align to the start of the text direction.
    Start,
    /// Center alignment.
    Center,
    /// Align to the end of the text direction.
    End,
    /// Left alignment (absolute).
    Left,
    /// Right alignment (absolute).
    Right,
}

impl Default for WebVttAlign {
    fn default() -> Self {
        Self::Center
    }
}

/// Extended cue settings parsed from a WebVTT timing line.
///
/// Corresponds to the settings block after `-->` on a WebVTT cue timing line.
#[derive(Debug, Clone)]
pub struct WebVttCueSettings {
    /// Region to anchor this cue into.
    pub region_id: Option<String>,
    /// Vertical text writing direction.
    pub vertical: WebVttVertical,
    /// Line placement.
    pub line: WebVttLine,
    /// Horizontal position percentage.
    pub position_pct: Option<f32>,
    /// Cue box size as a percentage of the video width (default 100%).
    pub size_pct: f32,
    /// Text alignment within the cue box.
    pub align: WebVttAlign,
}

impl Default for WebVttCueSettings {
    fn default() -> Self {
        Self {
            region_id: None,
            vertical: WebVttVertical::Horizontal,
            line: WebVttLine::Auto,
            position_pct: None,
            size_pct: 100.0,
            align: WebVttAlign::Center,
        }
    }
}

impl WebVttCueSettings {
    /// Parse cue settings from a settings string such as `"line:5 align:start position:10%"`.
    ///
    /// Unknown tokens are silently ignored.
    #[must_use]
    pub fn parse(s: &str) -> Self {
        let mut settings = Self::default();

        for token in s.split_whitespace() {
            if let Some(val) = token.strip_prefix("region:") {
                settings.region_id = Some(val.to_string());
            } else if let Some(val) = token.strip_prefix("vertical:") {
                settings.vertical = match val {
                    "lr" => WebVttVertical::LeftToRight,
                    "rl" => WebVttVertical::RightToLeft,
                    _ => WebVttVertical::Horizontal,
                };
            } else if let Some(val) = token.strip_prefix("line:") {
                settings.line = parse_line_value(val);
            } else if let Some(val) = token.strip_prefix("position:") {
                settings.position_pct = parse_percent(val);
            } else if let Some(val) = token.strip_prefix("size:") {
                if let Some(pct) = parse_percent(val) {
                    settings.size_pct = pct;
                }
            } else if let Some(val) = token.strip_prefix("align:") {
                settings.align = match val {
                    "start" => WebVttAlign::Start,
                    "center" | "middle" => WebVttAlign::Center,
                    "end" => WebVttAlign::End,
                    "left" => WebVttAlign::Left,
                    "right" => WebVttAlign::Right,
                    _ => WebVttAlign::Center,
                };
            }
        }

        settings
    }

    /// Serialise the settings back to a settings string (only non-default values are emitted).
    #[must_use]
    pub fn to_settings_string(&self) -> String {
        let mut parts: Vec<String> = Vec::new();

        if let Some(ref rid) = self.region_id {
            parts.push(format!("region:{rid}"));
        }

        match self.vertical {
            WebVttVertical::LeftToRight => parts.push("vertical:lr".to_string()),
            WebVttVertical::RightToLeft => parts.push("vertical:rl".to_string()),
            WebVttVertical::Horizontal => {}
        }

        match &self.line {
            WebVttLine::Auto => {}
            WebVttLine::Number(n) => parts.push(format!("line:{n}")),
            WebVttLine::Percentage(p) => parts.push(format!("line:{p:.1}%")),
        }

        if let Some(pos) = self.position_pct {
            parts.push(format!("position:{pos:.1}%"));
        }

        if (self.size_pct - 100.0).abs() > f32::EPSILON {
            parts.push(format!("size:{:.1}%", self.size_pct));
        }

        if self.align != WebVttAlign::Center {
            let a = match self.align {
                WebVttAlign::Start => "start",
                WebVttAlign::End => "end",
                WebVttAlign::Left => "left",
                WebVttAlign::Right => "right",
                WebVttAlign::Center => "center",
            };
            parts.push(format!("align:{a}"));
        }

        parts.join(" ")
    }
}

// ============================================================================
// Private helpers
// ============================================================================

/// Parse a percentage string like "50%" → 50.0.
fn parse_percent(s: &str) -> Option<f32> {
    s.trim_end_matches('%').parse::<f32>().ok()
}

/// Parse a line value: "5" (integer), "80%" (percentage), or "auto".
fn parse_line_value(s: &str) -> WebVttLine {
    if s == "auto" {
        return WebVttLine::Auto;
    }
    if s.ends_with('%') {
        if let Some(pct) = parse_percent(s) {
            return WebVttLine::Percentage(pct);
        }
    }
    if let Ok(n) = s.parse::<i32>() {
        return WebVttLine::Number(n);
    }
    WebVttLine::Auto
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_region_default_values() {
        let r = WebVttRegion::new("test");
        assert_eq!(r.id, "test");
        assert!((r.width_pct - 100.0).abs() < f32::EPSILON);
        assert_eq!(r.lines, 3);
        assert_eq!(r.region_anchor, (0.0, 100.0));
        assert_eq!(r.viewport_anchor, (0.0, 100.0));
        assert_eq!(r.scroll, WebVttScroll::None);
    }

    #[test]
    fn test_region_with_scroll_up() {
        let r = WebVttRegion::new("r1").with_scroll_up();
        assert_eq!(r.scroll, WebVttScroll::Up);
    }

    #[test]
    fn test_region_builder_chain() {
        let r = WebVttRegion::new("bottom")
            .with_width(40.0)
            .with_lines(5)
            .with_region_anchor(0.0, 100.0)
            .with_viewport_anchor(10.0, 90.0)
            .with_scroll_up();

        assert!((r.width_pct - 40.0).abs() < f32::EPSILON);
        assert_eq!(r.lines, 5);
        assert_eq!(r.viewport_anchor, (10.0, 90.0));
        assert_eq!(r.scroll, WebVttScroll::Up);
    }

    #[test]
    fn test_region_to_webvtt_contains_required_fields() {
        let r = WebVttRegion::new("sub").with_width(60.0).with_scroll_up();
        let s = r.to_webvtt();
        assert!(s.contains("REGION"));
        assert!(s.contains("id:sub"));
        assert!(s.contains("width:60.0%"));
        assert!(s.contains("scroll:up"));
    }

    #[test]
    fn test_parse_settings_line_and_align() {
        let settings = WebVttCueSettings::parse("line:5 align:start");
        assert_eq!(settings.line, WebVttLine::Number(5));
        assert_eq!(settings.align, WebVttAlign::Start);
    }

    #[test]
    fn test_parse_settings_vertical_rl() {
        let settings = WebVttCueSettings::parse("vertical:rl");
        assert_eq!(settings.vertical, WebVttVertical::RightToLeft);
    }

    #[test]
    fn test_parse_settings_vertical_lr() {
        let settings = WebVttCueSettings::parse("vertical:lr");
        assert_eq!(settings.vertical, WebVttVertical::LeftToRight);
    }

    #[test]
    fn test_parse_settings_position_percent() {
        let settings = WebVttCueSettings::parse("position:10%");
        assert!(settings.position_pct.is_some());
        let p = settings.position_pct.expect("position_pct should be set");
        assert!((p - 10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_parse_settings_region_id() {
        let settings = WebVttCueSettings::parse("region:bottom line:0");
        assert_eq!(settings.region_id, Some("bottom".to_string()));
        assert_eq!(settings.line, WebVttLine::Number(0));
    }

    #[test]
    fn test_settings_to_string_round_trip() {
        // Build settings, serialise, then re-parse and compare
        let mut orig = WebVttCueSettings::default();
        orig.vertical = WebVttVertical::RightToLeft;
        orig.line = WebVttLine::Number(-1);
        orig.align = WebVttAlign::Start;
        orig.position_pct = Some(25.0);
        orig.size_pct = 80.0;

        let s = orig.to_settings_string();
        let parsed = WebVttCueSettings::parse(&s);

        assert_eq!(parsed.vertical, WebVttVertical::RightToLeft);
        assert_eq!(parsed.line, WebVttLine::Number(-1));
        assert_eq!(parsed.align, WebVttAlign::Start);
        let pos = parsed.position_pct.expect("position_pct present");
        assert!((pos - 25.0).abs() < 0.01);
        assert!((parsed.size_pct - 80.0).abs() < 0.01);
    }

    #[test]
    fn test_settings_default_no_output() {
        // Default settings should produce empty string (all defaults suppressed)
        let settings = WebVttCueSettings::default();
        let s = settings.to_settings_string();
        assert!(s.is_empty());
    }

    #[test]
    fn test_parse_settings_line_percentage() {
        let settings = WebVttCueSettings::parse("line:80%");
        assert_eq!(settings.line, WebVttLine::Percentage(80.0));
    }

    #[test]
    fn test_parse_settings_size() {
        let settings = WebVttCueSettings::parse("size:50%");
        assert!((settings.size_pct - 50.0).abs() < f32::EPSILON);
    }
}
