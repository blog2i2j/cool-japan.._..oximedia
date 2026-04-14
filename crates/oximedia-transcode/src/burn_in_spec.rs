//! Subtitle burn-in pipeline specification.
//!
//! This module defines the *specification* layer for burning subtitles into
//! video frames during a transcode operation.  It is deliberately separate from
//! the lower-level font/positioning primitives in [`crate::burn_subs`]: those
//! primitives describe *how* a single glyph is rendered, while this module
//! describes *what* the transcode pipeline should render and *when*.
//!
//! # Pipeline overview
//!
//! ```text
//! SubtitleSource ──► BurnInSpec (global style)
//!                         │
//!                         ├─ SubtitleRenderCue × N  (per-cue overrides)
//!                         │
//!                         └─ BurnInPlan (validated, ready for pipeline)
//! ```
//!
//! A [`BurnInPlan`] is the resolved, validated artefact that a transcode stage
//! receives.  It bundles the global style with all per-cue render instructions
//! and exposes helper methods for querying which cues are active at a given
//! presentation timestamp.

#![allow(clippy::cast_precision_loss)]

use std::fmt;

use crate::{Result, TranscodeError};

// ─────────────────────────────────────────────────────────────────────────────
// SafeArea
// ─────────────────────────────────────────────────────────────────────────────

/// Safe-area margins as a fraction of the frame dimension (0.0–1.0).
///
/// A value of `0.05` means "5 % of the frame width/height is reserved as a
/// margin on each side".
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SafeArea {
    /// Left margin fraction.
    pub left: f32,
    /// Right margin fraction.
    pub right: f32,
    /// Top margin fraction.
    pub top: f32,
    /// Bottom margin fraction.
    pub bottom: f32,
}

impl SafeArea {
    /// Creates a uniform safe-area with the same fraction on all four sides.
    #[must_use]
    pub fn uniform(fraction: f32) -> Self {
        Self {
            left: fraction,
            right: fraction,
            top: fraction,
            bottom: fraction,
        }
    }

    /// Returns the standard broadcast 10 % title-safe area.
    #[must_use]
    pub fn title_safe() -> Self {
        Self::uniform(0.10)
    }

    /// Returns the standard broadcast 5 % action-safe area.
    #[must_use]
    pub fn action_safe() -> Self {
        Self::uniform(0.05)
    }

    /// Computes the usable pixel rectangle for a frame of the given dimensions.
    ///
    /// Returns `(x, y, width, height)` in pixels.
    #[must_use]
    pub fn usable_rect(&self, frame_width: u32, frame_height: u32) -> (u32, u32, u32, u32) {
        let fw = frame_width as f32;
        let fh = frame_height as f32;
        let x = (fw * self.left) as u32;
        let y = (fh * self.top) as u32;
        let w = (fw * (1.0 - self.left - self.right)) as u32;
        let h = (fh * (1.0 - self.top - self.bottom)) as u32;
        (x, y, w, h)
    }

    /// Returns `true` if all fractions are in the valid range `[0.0, 0.5)`.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        let ok = |v: f32| (0.0..0.5).contains(&v);
        ok(self.left) && ok(self.right) && ok(self.top) && ok(self.bottom)
    }
}

impl Default for SafeArea {
    fn default() -> Self {
        Self::uniform(0.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SubtitleAlignment
// ─────────────────────────────────────────────────────────────────────────────

/// Horizontal alignment of subtitle text within the safe area.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubtitleAlignment {
    /// Align text to the left edge.
    Left,
    /// Center text horizontally.
    Center,
    /// Align text to the right edge.
    Right,
}

/// Vertical placement of subtitles within the safe area.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubtitleVertical {
    /// Place subtitles at the bottom (most common).
    Bottom,
    /// Place subtitles at the top.
    Top,
    /// Place subtitles in the middle (used for on-screen graphics).
    Middle,
}

// ─────────────────────────────────────────────────────────────────────────────
// BurnInStyle
// ─────────────────────────────────────────────────────────────────────────────

/// Global style parameters for the subtitle burn-in pipeline.
///
/// These apply to all cues unless overridden by a per-cue [`CueStyleOverride`].
#[derive(Debug, Clone)]
pub struct BurnInStyle {
    /// Font family name (e.g. `"Arial"`, `"Helvetica Neue"`).
    pub font_family: String,
    /// Font size in points (rendered at the output resolution).
    pub font_size_pt: f32,
    /// Whether to use bold weight.
    pub bold: bool,
    /// Whether to use italic rendering.
    pub italic: bool,
    /// Text fill colour as (R, G, B, A).
    pub color: (u8, u8, u8, u8),
    /// Outline / shadow colour as (R, G, B, A).
    pub outline_color: (u8, u8, u8, u8),
    /// Outline thickness in pixels.
    pub outline_px: u32,
    /// Background box colour (drawn behind each text line).  `None` = no box.
    pub background_color: Option<(u8, u8, u8, u8)>,
    /// Safe-area margins used to position text.
    pub safe_area: SafeArea,
    /// Horizontal alignment.
    pub alignment: SubtitleAlignment,
    /// Vertical placement.
    pub vertical: SubtitleVertical,
    /// Line spacing multiplier (1.0 = normal).
    pub line_spacing: f32,
    /// Maximum width of the text box as a fraction of the safe-area width (0.0–1.0).
    pub max_width_fraction: f32,
}

impl BurnInStyle {
    /// Creates a style suitable for most broadcast/streaming use cases.
    #[must_use]
    pub fn broadcast_default() -> Self {
        Self {
            font_family: "Arial".to_string(),
            font_size_pt: 38.0,
            bold: false,
            italic: false,
            color: (255, 255, 255, 255),
            outline_color: (0, 0, 0, 210),
            outline_px: 2,
            background_color: None,
            safe_area: SafeArea::title_safe(),
            alignment: SubtitleAlignment::Center,
            vertical: SubtitleVertical::Bottom,
            line_spacing: 1.2,
            max_width_fraction: 0.85,
        }
    }

    /// Creates a style optimised for SDH (Subtitles for the Deaf/Hard-of-hearing)
    /// with a semi-transparent background box.
    #[must_use]
    pub fn sdh_default() -> Self {
        Self {
            font_family: "Courier New".to_string(),
            font_size_pt: 32.0,
            bold: false,
            italic: false,
            color: (255, 255, 255, 255),
            outline_color: (0, 0, 0, 0),
            outline_px: 0,
            background_color: Some((0, 0, 0, 180)),
            safe_area: SafeArea::title_safe(),
            alignment: SubtitleAlignment::Center,
            vertical: SubtitleVertical::Bottom,
            line_spacing: 1.3,
            max_width_fraction: 0.80,
        }
    }

    /// Validates the style parameters.
    ///
    /// # Errors
    ///
    /// Returns [`TranscodeError::InvalidInput`] if any parameter is out of range.
    pub fn validate(&self) -> Result<()> {
        if self.font_family.trim().is_empty() {
            return Err(TranscodeError::InvalidInput(
                "font_family must not be empty".to_string(),
            ));
        }
        if self.font_size_pt <= 0.0 {
            return Err(TranscodeError::InvalidInput(
                "font_size_pt must be positive".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.max_width_fraction) {
            return Err(TranscodeError::InvalidInput(format!(
                "max_width_fraction {} is out of range [0, 1]",
                self.max_width_fraction
            )));
        }
        if !self.safe_area.is_valid() {
            return Err(TranscodeError::InvalidInput(
                "safe_area margins must each be in [0, 0.5)".to_string(),
            ));
        }
        if self.line_spacing <= 0.0 {
            return Err(TranscodeError::InvalidInput(
                "line_spacing must be positive".to_string(),
            ));
        }
        Ok(())
    }
}

impl Default for BurnInStyle {
    fn default() -> Self {
        Self::broadcast_default()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CueStyleOverride
// ─────────────────────────────────────────────────────────────────────────────

/// Per-cue style overrides.  Any `Some` field replaces the global value.
#[derive(Debug, Clone, Default)]
pub struct CueStyleOverride {
    /// Override font size in points.
    pub font_size_pt: Option<f32>,
    /// Override text colour.
    pub color: Option<(u8, u8, u8, u8)>,
    /// Override italic flag.
    pub italic: Option<bool>,
    /// Override bold flag.
    pub bold: Option<bool>,
    /// Override vertical placement.
    pub vertical: Option<SubtitleVertical>,
    /// Override horizontal alignment.
    pub alignment: Option<SubtitleAlignment>,
}

// ─────────────────────────────────────────────────────────────────────────────
// SubtitleRenderCue
// ─────────────────────────────────────────────────────────────────────────────

/// A single subtitle cue with timing and optional per-cue style overrides.
#[derive(Debug, Clone)]
pub struct SubtitleRenderCue {
    /// Display text (may contain newlines for multi-line cues).
    pub text: String,
    /// Start presentation time in milliseconds.
    pub start_ms: u64,
    /// End presentation time in milliseconds (exclusive).
    pub end_ms: u64,
    /// Optional per-cue style overrides.
    pub style_override: Option<CueStyleOverride>,
    /// Optional speaker label (used for SDH / multi-speaker workflows).
    pub speaker: Option<String>,
}

impl SubtitleRenderCue {
    /// Creates a new cue with the given text and timing.
    #[must_use]
    pub fn new(text: impl Into<String>, start_ms: u64, end_ms: u64) -> Self {
        Self {
            text: text.into(),
            start_ms,
            end_ms,
            style_override: None,
            speaker: None,
        }
    }

    /// Attaches a per-cue style override.
    #[must_use]
    pub fn with_style(mut self, ovr: CueStyleOverride) -> Self {
        self.style_override = Some(ovr);
        self
    }

    /// Attaches a speaker label.
    #[must_use]
    pub fn with_speaker(mut self, speaker: impl Into<String>) -> Self {
        self.speaker = Some(speaker.into());
        self
    }

    /// Duration in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// Returns `true` if this cue is active at `pts_ms`.
    #[must_use]
    pub fn is_active_at(&self, pts_ms: u64) -> bool {
        pts_ms >= self.start_ms && pts_ms < self.end_ms
    }

    /// Returns `true` if the cue parameters are valid.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.end_ms > self.start_ms && !self.text.is_empty()
    }
}

impl fmt::Display for SubtitleRenderCue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{:>8}–{:>8} ms] {}",
            self.start_ms,
            self.end_ms,
            self.text.replace('\n', " / ")
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BurnInSpec
// ─────────────────────────────────────────────────────────────────────────────

/// High-level specification for a burn-in operation in the transcode pipeline.
///
/// A [`BurnInSpec`] is built with the builder API and then converted into a
/// [`BurnInPlan`] via [`BurnInSpec::into_plan`].
#[derive(Debug, Clone)]
pub struct BurnInSpec {
    /// Global style applied to all cues.
    pub style: BurnInStyle,
    /// All subtitle cues to be rendered.
    pub cues: Vec<SubtitleRenderCue>,
    /// Frame width of the output video in pixels.
    pub frame_width: u32,
    /// Frame height of the output video in pixels.
    pub frame_height: u32,
    /// Whether to embed the language label as a visible tag in each cue.
    pub show_language_tag: bool,
    /// Maximum number of simultaneously visible lines.
    pub max_lines: u8,
}

impl BurnInSpec {
    /// Creates a new spec with default broadcast style.
    #[must_use]
    pub fn new(frame_width: u32, frame_height: u32) -> Self {
        Self {
            style: BurnInStyle::broadcast_default(),
            cues: Vec::new(),
            frame_width,
            frame_height,
            show_language_tag: false,
            max_lines: 2,
        }
    }

    /// Overrides the global style.
    #[must_use]
    pub fn with_style(mut self, style: BurnInStyle) -> Self {
        self.style = style;
        self
    }

    /// Adds a single render cue.
    #[must_use]
    pub fn add_cue(mut self, cue: SubtitleRenderCue) -> Self {
        self.cues.push(cue);
        self
    }

    /// Adds multiple render cues from a slice.
    #[must_use]
    pub fn add_cues(mut self, cues: impl IntoIterator<Item = SubtitleRenderCue>) -> Self {
        self.cues.extend(cues);
        self
    }

    /// Enables the language tag display.
    #[must_use]
    pub fn show_language_tag(mut self) -> Self {
        self.show_language_tag = true;
        self
    }

    /// Sets the maximum number of visible lines.
    #[must_use]
    pub fn max_lines(mut self, n: u8) -> Self {
        self.max_lines = n;
        self
    }

    /// Validates and converts this spec into a [`BurnInPlan`].
    ///
    /// # Errors
    ///
    /// Returns [`TranscodeError::InvalidInput`] if:
    /// - The frame dimensions are zero.
    /// - The style is invalid (see [`BurnInStyle::validate`]).
    /// - Any cue has zero duration or empty text.
    pub fn into_plan(self) -> Result<BurnInPlan> {
        if self.frame_width == 0 || self.frame_height == 0 {
            return Err(TranscodeError::InvalidInput(
                "frame dimensions must be non-zero".to_string(),
            ));
        }
        self.style.validate()?;

        for (idx, cue) in self.cues.iter().enumerate() {
            if !cue.is_valid() {
                return Err(TranscodeError::InvalidInput(format!(
                    "cue {idx} is invalid (empty text or zero duration)"
                )));
            }
        }

        // Sort cues by start time.
        let mut sorted_cues = self.cues;
        sorted_cues.sort_by_key(|c| c.start_ms);

        Ok(BurnInPlan {
            style: self.style,
            cues: sorted_cues,
            frame_width: self.frame_width,
            frame_height: self.frame_height,
            show_language_tag: self.show_language_tag,
            max_lines: self.max_lines,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BurnInPlan
// ─────────────────────────────────────────────────────────────────────────────

/// A validated, immutable burn-in plan ready to be consumed by the transcode
/// pipeline.
#[derive(Debug, Clone)]
pub struct BurnInPlan {
    /// Resolved global style.
    pub style: BurnInStyle,
    /// Cues sorted by `start_ms`.
    pub cues: Vec<SubtitleRenderCue>,
    /// Output frame width.
    pub frame_width: u32,
    /// Output frame height.
    pub frame_height: u32,
    /// Whether to show language tags.
    pub show_language_tag: bool,
    /// Maximum simultaneous visible lines.
    pub max_lines: u8,
}

impl BurnInPlan {
    /// Returns the cues that are active at the given presentation timestamp.
    #[must_use]
    pub fn active_cues_at(&self, pts_ms: u64) -> Vec<&SubtitleRenderCue> {
        self.cues
            .iter()
            .filter(|c| c.is_active_at(pts_ms))
            .collect()
    }

    /// Returns the usable pixel rectangle respecting the safe-area margins.
    ///
    /// Result is `(x, y, width, height)`.
    #[must_use]
    pub fn usable_rect(&self) -> (u32, u32, u32, u32) {
        self.style
            .safe_area
            .usable_rect(self.frame_width, self.frame_height)
    }

    /// Returns the total number of cues in the plan.
    #[must_use]
    pub fn cue_count(&self) -> usize {
        self.cues.len()
    }

    /// Returns `true` if there are no cues.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cues.is_empty()
    }

    /// Iterates over all cues.
    pub fn iter_cues(&self) -> impl Iterator<Item = &SubtitleRenderCue> {
        self.cues.iter()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_plan() -> BurnInPlan {
        BurnInSpec::new(1920, 1080)
            .add_cue(SubtitleRenderCue::new("Hello world", 1_000, 4_000))
            .add_cue(SubtitleRenderCue::new("Second cue", 5_000, 9_000))
            .into_plan()
            .unwrap()
    }

    #[test]
    fn test_safe_area_uniform() {
        let sa = SafeArea::uniform(0.10);
        assert_eq!(sa.left, 0.10);
        assert_eq!(sa.right, 0.10);
        assert_eq!(sa.top, 0.10);
        assert_eq!(sa.bottom, 0.10);
    }

    #[test]
    fn test_safe_area_usable_rect() {
        let sa = SafeArea::uniform(0.10);
        let (x, y, w, h) = sa.usable_rect(1920, 1080);
        assert_eq!(x, 192);
        assert_eq!(y, 108);
        // 1920 * 0.80 as f32 may round to 1535 or 1536 depending on FP precision.
        assert!(w == 1535 || w == 1536, "expected w≈1536, got {w}");
        assert!(h == 863 || h == 864, "expected h≈864, got {h}");
    }

    #[test]
    fn test_safe_area_is_valid() {
        assert!(SafeArea::uniform(0.05).is_valid());
        assert!(!SafeArea::uniform(0.5).is_valid()); // 0.5 is NOT in [0, 0.5)
        assert!(!SafeArea::uniform(-0.01).is_valid());
    }

    #[test]
    fn test_burn_in_style_validate_ok() {
        assert!(BurnInStyle::broadcast_default().validate().is_ok());
    }

    #[test]
    fn test_burn_in_style_validate_empty_font() {
        let mut s = BurnInStyle::broadcast_default();
        s.font_family = "   ".to_string();
        assert!(s.validate().is_err());
    }

    #[test]
    fn test_burn_in_style_validate_zero_size() {
        let mut s = BurnInStyle::broadcast_default();
        s.font_size_pt = 0.0;
        assert!(s.validate().is_err());
    }

    #[test]
    fn test_subtitle_render_cue_basics() {
        let cue = SubtitleRenderCue::new("Hello", 2_000, 5_000);
        assert_eq!(cue.duration_ms(), 3_000);
        assert!(cue.is_active_at(2_000));
        assert!(cue.is_active_at(4_999));
        assert!(!cue.is_active_at(5_000));
        assert!(!cue.is_active_at(1_999));
    }

    #[test]
    fn test_cue_invalid_zero_duration() {
        let cue = SubtitleRenderCue::new("X", 5_000, 5_000);
        assert!(!cue.is_valid());
    }

    #[test]
    fn test_cue_invalid_empty_text() {
        let cue = SubtitleRenderCue::new("", 1_000, 2_000);
        assert!(!cue.is_valid());
    }

    #[test]
    fn test_burn_in_plan_active_cues() {
        let plan = sample_plan();

        // At 2 s — first cue is active.
        let active = plan.active_cues_at(2_000);
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].text, "Hello world");

        // At 6 s — second cue is active.
        let active = plan.active_cues_at(6_000);
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].text, "Second cue");

        // At 4.5 s — no cue is active.
        let active = plan.active_cues_at(4_500);
        assert!(active.is_empty());
    }

    #[test]
    fn test_burn_in_spec_zero_frame_dimensions() {
        let err = BurnInSpec::new(0, 1080).into_plan();
        assert!(err.is_err());
    }

    #[test]
    fn test_burn_in_spec_invalid_cue_rejected() {
        let err = BurnInSpec::new(1920, 1080)
            .add_cue(SubtitleRenderCue::new("", 0, 1_000))
            .into_plan();
        assert!(err.is_err());
    }

    #[test]
    fn test_burn_in_plan_usable_rect() {
        let plan = sample_plan();
        let (x, y, w, h) = plan.usable_rect();
        // Default style uses 10 % title-safe margins.
        assert_eq!(x, 192);   // 1920 * 0.10
        assert_eq!(y, 108);   // 1080 * 0.10
        // 1920 * 0.80 as f32 may round to 1535 or 1536 depending on FP precision.
        assert!(w == 1535 || w == 1536, "expected w≈1536, got {w}");
        assert!(h == 863 || h == 864, "expected h≈864, got {h}");
    }

    #[test]
    fn test_burn_in_plan_cue_count() {
        let plan = sample_plan();
        assert_eq!(plan.cue_count(), 2);
        assert!(!plan.is_empty());
    }

    #[test]
    fn test_cues_sorted_by_start_time() {
        // Add cues out of order.
        let plan = BurnInSpec::new(1280, 720)
            .add_cue(SubtitleRenderCue::new("Late", 10_000, 12_000))
            .add_cue(SubtitleRenderCue::new("Early", 1_000, 3_000))
            .into_plan()
            .unwrap();

        assert_eq!(plan.cues[0].text, "Early");
        assert_eq!(plan.cues[1].text, "Late");
    }

    #[test]
    fn test_per_cue_style_override() {
        let ovr = CueStyleOverride {
            italic: Some(true),
            color: Some((255, 0, 0, 255)),
            ..Default::default()
        };
        let cue = SubtitleRenderCue::new("Red italic", 0, 2_000).with_style(ovr.clone());
        assert!(cue.style_override.is_some());
        let s = cue.style_override.as_ref().unwrap();
        assert_eq!(s.italic, Some(true));
        assert_eq!(s.color, Some((255, 0, 0, 255)));
    }

    #[test]
    fn test_sdh_style_has_background() {
        let style = BurnInStyle::sdh_default();
        assert!(style.background_color.is_some());
        assert!(style.validate().is_ok());
    }

    #[test]
    fn test_cue_display() {
        let cue = SubtitleRenderCue::new("Hello\nworld", 1_000, 4_000);
        let s = cue.to_string();
        assert!(s.contains("Hello"));
        assert!(s.contains('/'));
    }

    #[test]
    fn test_plan_iter_cues() {
        let plan = sample_plan();
        let texts: Vec<&str> = plan.iter_cues().map(|c| c.text.as_str()).collect();
        assert_eq!(texts, ["Hello world", "Second cue"]);
    }
}
