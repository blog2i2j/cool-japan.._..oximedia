//! Automatic title card generation for video sequences.
//!
//! This module generates title card metadata — text layout, fade timing,
//! style templates, and duration calculation — without performing any
//! pixel-level rendering.  A downstream renderer (e.g. `oximedia-graphics`)
//! consumes the [`TitleCard`] structs to produce actual frames.
//!
//! # Design
//!
//! * [`TitleCardGenerator`] is the main entry-point.  Feed it a slice of
//!   [`TitleCardRequest`] structs and it returns ready-to-render [`TitleCard`]s.
//! * [`StyleTemplate`] captures the visual style (font size, colours, position).
//! * [`FadeTiming`] describes the fade-in / hold / fade-out envelope.
//! * Duration is calculated from the character count of the text with a
//!   configurable reading-speed parameter (characters per second).
//!
//! # Example
//!
//! ```
//! use oximedia_auto::title_card_generator::{
//!     TitleCardGenerator, TitleCardGeneratorConfig, TitleCardRequest, StyleTemplate,
//! };
//!
//! let config = TitleCardGeneratorConfig::default();
//! let generator = TitleCardGenerator::new(config);
//!
//! let requests = vec![
//!     TitleCardRequest::new("Introduction", 0, StyleTemplate::default()),
//!     TitleCardRequest::new("Chapter 1", 5_000, StyleTemplate::lower_third()),
//! ];
//!
//! let cards = generator.generate(&requests).unwrap();
//! assert_eq!(cards.len(), 2);
//! ```

#![allow(dead_code)]

use crate::error::{AutoError, AutoResult};
use oximedia_core::{types::Rational, Timestamp};

// ─── Colour ───────────────────────────────────────────────────────────────────

/// An RGBA colour with components in 0–255.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rgba {
    /// Red channel.
    pub r: u8,
    /// Green channel.
    pub g: u8,
    /// Blue channel.
    pub b: u8,
    /// Alpha channel (0 = transparent, 255 = opaque).
    pub a: u8,
}

impl Rgba {
    /// Construct an opaque colour.
    #[must_use]
    pub const fn opaque(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }

    /// Fully transparent black.
    #[must_use]
    pub const fn transparent() -> Self {
        Self {
            r: 0,
            g: 0,
            b: 0,
            a: 0,
        }
    }

    /// White.
    #[must_use]
    pub const fn white() -> Self {
        Self::opaque(255, 255, 255)
    }

    /// Black.
    #[must_use]
    pub const fn black() -> Self {
        Self::opaque(0, 0, 0)
    }
}

impl Default for Rgba {
    fn default() -> Self {
        Self::white()
    }
}

// ─── Alignment ────────────────────────────────────────────────────────────────

/// Horizontal text alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HAlign {
    /// Left-aligned.
    Left,
    /// Centre-aligned.
    #[default]
    Center,
    /// Right-aligned.
    Right,
}

/// Vertical position of the card on the frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VPosition {
    /// Near the top (10 % from top edge).
    Top,
    /// Vertically centred.
    #[default]
    Middle,
    /// Lower-third region (25 % from bottom edge).
    LowerThird,
    /// Near the bottom (5 % from bottom edge).
    Bottom,
}

impl VPosition {
    /// Normalised Y coordinate (0.0 = top, 1.0 = bottom) of the anchor point.
    #[must_use]
    pub fn y_anchor(self) -> f32 {
        match self {
            Self::Top => 0.10,
            Self::Middle => 0.50,
            Self::LowerThird => 0.75,
            Self::Bottom => 0.92,
        }
    }
}

// ─── Style template ───────────────────────────────────────────────────────────

/// Visual styling for a title card.
#[derive(Debug, Clone, PartialEq)]
pub struct StyleTemplate {
    /// Name / identifier for this template.
    pub name: String,
    /// Foreground (text) colour.
    pub text_color: Rgba,
    /// Background fill colour (transparent by default for full-screen cards).
    pub background_color: Rgba,
    /// Accent / highlight colour used for decorative elements.
    pub accent_color: Rgba,
    /// Logical font size in points (renderer scales to resolution).
    pub font_size_pt: f32,
    /// Horizontal alignment.
    pub h_align: HAlign,
    /// Vertical position on the frame.
    pub v_position: VPosition,
    /// Whether a semi-transparent background band should be drawn behind text.
    pub show_band: bool,
    /// Relative opacity of the band (0.0–1.0).
    pub band_opacity: f32,
    /// Padding in logical pixels around the text.
    pub padding_px: u32,
}

impl Default for StyleTemplate {
    fn default() -> Self {
        Self {
            name: "default".into(),
            text_color: Rgba::white(),
            background_color: Rgba::transparent(),
            accent_color: Rgba::opaque(255, 200, 0),
            font_size_pt: 48.0,
            h_align: HAlign::Center,
            v_position: VPosition::Middle,
            show_band: true,
            band_opacity: 0.6,
            padding_px: 24,
        }
    }
}

impl StyleTemplate {
    /// Lower-third style (band at the bottom of the frame).
    #[must_use]
    pub fn lower_third() -> Self {
        Self {
            name: "lower_third".into(),
            font_size_pt: 32.0,
            v_position: VPosition::LowerThird,
            h_align: HAlign::Left,
            show_band: true,
            band_opacity: 0.75,
            padding_px: 16,
            ..Self::default()
        }
    }

    /// Minimal style — text only, no band.
    #[must_use]
    pub fn minimal() -> Self {
        Self {
            name: "minimal".into(),
            show_band: false,
            band_opacity: 0.0,
            ..Self::default()
        }
    }

    /// Bold cinematic style — large centred text with a strong band.
    #[must_use]
    pub fn cinematic() -> Self {
        Self {
            name: "cinematic".into(),
            font_size_pt: 72.0,
            show_band: true,
            band_opacity: 0.85,
            v_position: VPosition::Middle,
            ..Self::default()
        }
    }

    /// Validate that the style parameters are in range.
    ///
    /// # Errors
    ///
    /// Returns [`AutoError::InvalidParameter`] if any parameter is out of range.
    pub fn validate(&self) -> AutoResult<()> {
        if self.font_size_pt <= 0.0 {
            return Err(AutoError::invalid_parameter(
                "font_size_pt",
                format!("{} (must be > 0)", self.font_size_pt),
            ));
        }
        if !(0.0..=1.0).contains(&self.band_opacity) {
            return Err(AutoError::invalid_parameter(
                "band_opacity",
                format!("{} (must be 0.0–1.0)", self.band_opacity),
            ));
        }
        Ok(())
    }
}

// ─── Fade timing ──────────────────────────────────────────────────────────────

/// Fade-in / hold / fade-out envelope for a title card.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FadeTiming {
    /// Duration of the fade-in in milliseconds.
    pub fade_in_ms: u32,
    /// Duration during which the card is fully visible.
    pub hold_ms: u32,
    /// Duration of the fade-out in milliseconds.
    pub fade_out_ms: u32,
}

impl FadeTiming {
    /// Total on-screen duration (fade-in + hold + fade-out).
    #[must_use]
    pub fn total_ms(&self) -> u32 {
        self.fade_in_ms
            .saturating_add(self.hold_ms)
            .saturating_add(self.fade_out_ms)
    }

    /// Compute the opacity (0.0–1.0) at `elapsed_ms` milliseconds after card appears.
    #[must_use]
    pub fn opacity_at(&self, elapsed_ms: u32) -> f32 {
        if elapsed_ms < self.fade_in_ms {
            if self.fade_in_ms == 0 {
                1.0
            } else {
                elapsed_ms as f32 / self.fade_in_ms as f32
            }
        } else {
            let hold_end = self.fade_in_ms.saturating_add(self.hold_ms);
            if elapsed_ms <= hold_end {
                1.0
            } else {
                let fade_elapsed = elapsed_ms.saturating_sub(hold_end);
                if self.fade_out_ms == 0 {
                    0.0
                } else {
                    1.0 - (fade_elapsed as f32 / self.fade_out_ms as f32).min(1.0)
                }
            }
        }
    }
}

impl Default for FadeTiming {
    fn default() -> Self {
        Self {
            fade_in_ms: 250,
            hold_ms: 2_000,
            fade_out_ms: 500,
        }
    }
}

// ─── Request / result types ───────────────────────────────────────────────────

/// A request to generate a single title card.
#[derive(Debug, Clone)]
pub struct TitleCardRequest {
    /// Primary text (title line).
    pub title: String,
    /// Optional subtitle / second line.
    pub subtitle: Option<String>,
    /// Timestamp (ms PTS) at which the card should appear.
    pub start_pts_ms: i64,
    /// Visual style for this card.
    pub style: StyleTemplate,
    /// Override automatically-calculated duration (ms).
    /// `None` → duration is computed from text length and reading speed.
    pub duration_override_ms: Option<u32>,
}

impl TitleCardRequest {
    /// Create a minimal request with just a title.
    #[must_use]
    pub fn new(title: impl Into<String>, start_pts_ms: i64, style: StyleTemplate) -> Self {
        Self {
            title: title.into(),
            subtitle: None,
            start_pts_ms,
            style,
            duration_override_ms: None,
        }
    }

    /// Attach a subtitle.
    #[must_use]
    pub fn with_subtitle(mut self, subtitle: impl Into<String>) -> Self {
        self.subtitle = Some(subtitle.into());
        self
    }

    /// Override the auto-calculated duration.
    #[must_use]
    pub const fn with_duration_ms(mut self, ms: u32) -> Self {
        self.duration_override_ms = Some(ms);
        self
    }

    /// Total character count (title + subtitle if present).
    #[must_use]
    fn char_count(&self) -> usize {
        let sub_len = self.subtitle.as_deref().map_or(0, str::len);
        self.title.len() + sub_len
    }
}

/// A fully resolved title card ready for rendering.
#[derive(Debug, Clone)]
pub struct TitleCard {
    /// Primary text line.
    pub title: String,
    /// Optional subtitle.
    pub subtitle: Option<String>,
    /// Presentation timestamp (ms) at which the card starts.
    pub start: Timestamp,
    /// Presentation timestamp (ms) at which the card ends.
    pub end: Timestamp,
    /// Fade timing envelope.
    pub fade: FadeTiming,
    /// Visual style.
    pub style: StyleTemplate,
}

impl TitleCard {
    /// Total on-screen duration in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> i64 {
        (self.end.pts - self.start.pts).max(0)
    }

    /// Opacity at a given absolute PTS (milliseconds).
    #[must_use]
    pub fn opacity_at_pts(&self, pts_ms: i64) -> f32 {
        let elapsed = (pts_ms - self.start.pts).max(0) as u32;
        self.fade.opacity_at(elapsed)
    }
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for [`TitleCardGenerator`].
#[derive(Debug, Clone)]
pub struct TitleCardGeneratorConfig {
    /// Characters per second used for reading-speed duration calculation.
    /// Typical values: 10–25 (10 = slow / accessible, 20 = fast).
    pub reading_speed_cps: f32,
    /// Minimum card duration in milliseconds regardless of text length.
    pub min_duration_ms: u32,
    /// Maximum card duration in milliseconds.
    pub max_duration_ms: u32,
    /// Default fade timing applied when no per-card timing is supplied.
    pub default_fade: FadeTiming,
    /// Gap between consecutive cards in milliseconds (used only when
    /// [`TitleCardGenerator::layout_sequence`] auto-positions cards).
    pub inter_card_gap_ms: u32,
}

impl Default for TitleCardGeneratorConfig {
    fn default() -> Self {
        Self {
            reading_speed_cps: 15.0,
            min_duration_ms: 1_500,
            max_duration_ms: 8_000,
            default_fade: FadeTiming::default(),
            inter_card_gap_ms: 500,
        }
    }
}

impl TitleCardGeneratorConfig {
    /// Create a new config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the reading speed in characters per second.
    #[must_use]
    pub fn with_reading_speed(mut self, cps: f32) -> Self {
        self.reading_speed_cps = cps;
        self
    }

    /// Set the minimum card duration.
    #[must_use]
    pub const fn with_min_duration_ms(mut self, ms: u32) -> Self {
        self.min_duration_ms = ms;
        self
    }

    /// Set the maximum card duration.
    #[must_use]
    pub const fn with_max_duration_ms(mut self, ms: u32) -> Self {
        self.max_duration_ms = ms;
        self
    }

    /// Validate configuration values.
    ///
    /// # Errors
    ///
    /// Returns an error if any parameter is invalid.
    pub fn validate(&self) -> AutoResult<()> {
        if self.reading_speed_cps <= 0.0 {
            return Err(AutoError::invalid_parameter(
                "reading_speed_cps",
                format!("{} (must be > 0)", self.reading_speed_cps),
            ));
        }
        if self.min_duration_ms > self.max_duration_ms {
            return Err(AutoError::invalid_parameter(
                "min_duration_ms",
                format!(
                    "{} exceeds max_duration_ms {}",
                    self.min_duration_ms, self.max_duration_ms
                ),
            ));
        }
        Ok(())
    }
}

// ─── Generator ────────────────────────────────────────────────────────────────

/// Generates [`TitleCard`] metadata from [`TitleCardRequest`]s.
#[derive(Debug, Clone)]
pub struct TitleCardGenerator {
    config: TitleCardGeneratorConfig,
}

impl TitleCardGenerator {
    /// Create a new generator with the given configuration.
    #[must_use]
    pub fn new(config: TitleCardGeneratorConfig) -> Self {
        Self { config }
    }

    /// Compute the display duration (ms) for a card from its character count.
    #[must_use]
    pub fn compute_duration_ms(&self, char_count: usize) -> u32 {
        let base_ms = if self.config.reading_speed_cps > 0.0 {
            ((char_count as f32 / self.config.reading_speed_cps) * 1_000.0) as u32
        } else {
            self.config.min_duration_ms
        };
        // Add fade times on top of the reading window
        let with_fades = base_ms
            .saturating_add(self.config.default_fade.fade_in_ms)
            .saturating_add(self.config.default_fade.fade_out_ms);
        with_fades
            .max(self.config.min_duration_ms)
            .min(self.config.max_duration_ms)
    }

    /// Generate title cards from a slice of requests.
    ///
    /// Each card's `start` PTS is taken from `TitleCardRequest::start_pts_ms`
    /// and the `end` PTS is derived from the computed or overridden duration.
    ///
    /// # Errors
    ///
    /// Returns an error if any request has an invalid style or the config
    /// fails validation.
    pub fn generate(&self, requests: &[TitleCardRequest]) -> AutoResult<Vec<TitleCard>> {
        self.config.validate()?;
        let mut cards = Vec::with_capacity(requests.len());
        for req in requests {
            req.style.validate()?;
            let duration_ms = req
                .duration_override_ms
                .unwrap_or_else(|| self.compute_duration_ms(req.char_count()));
            let start = Timestamp::new(req.start_pts_ms, Rational::new(1, 1000));
            let end = Timestamp::new(
                req.start_pts_ms + i64::from(duration_ms),
                Rational::new(1, 1000),
            );
            cards.push(TitleCard {
                title: req.title.clone(),
                subtitle: req.subtitle.clone(),
                start,
                end,
                fade: self.config.default_fade,
                style: req.style.clone(),
            });
        }
        Ok(cards)
    }

    /// Auto-layout a sequence of titles: positions cards sequentially with
    /// `inter_card_gap_ms` between them, starting at `start_pts_ms`.
    ///
    /// The caller supplies only titles (and optional subtitles), and the
    /// generator chooses styles from the provided `templates` list (cycling
    /// if there are more titles than templates).  If `templates` is empty the
    /// default template is used for all cards.
    ///
    /// # Errors
    ///
    /// Returns an error if configuration is invalid or a derived style is
    /// invalid.
    pub fn layout_sequence(
        &self,
        titles: &[(&str, Option<&str>)],
        templates: &[StyleTemplate],
        start_pts_ms: i64,
    ) -> AutoResult<Vec<TitleCard>> {
        self.config.validate()?;
        let default_template = StyleTemplate::default();
        let mut cards = Vec::with_capacity(titles.len());
        let mut cursor_ms = start_pts_ms;

        for (idx, &(title, subtitle)) in titles.iter().enumerate() {
            let style = if templates.is_empty() {
                default_template.clone()
            } else {
                templates[idx % templates.len()].clone()
            };
            style.validate()?;

            let char_count = title.len() + subtitle.map_or(0, str::len);
            let duration_ms = self.compute_duration_ms(char_count);
            let start = Timestamp::new(cursor_ms, Rational::new(1, 1000));
            let end = Timestamp::new(cursor_ms + i64::from(duration_ms), Rational::new(1, 1000));

            cards.push(TitleCard {
                title: title.to_string(),
                subtitle: subtitle.map(str::to_string),
                start,
                end,
                fade: self.config.default_fade,
                style,
            });

            cursor_ms =
                cursor_ms + i64::from(duration_ms) + i64::from(self.config.inter_card_gap_ms);
        }
        Ok(cards)
    }
}

impl Default for TitleCardGenerator {
    fn default() -> Self {
        Self::new(TitleCardGeneratorConfig::default())
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_generator() -> TitleCardGenerator {
        TitleCardGenerator::default()
    }

    #[test]
    fn test_generate_single_card() {
        let gen = make_generator();
        let requests = vec![TitleCardRequest::new(
            "Hello World",
            0,
            StyleTemplate::default(),
        )];
        let cards = gen.generate(&requests).unwrap();
        assert_eq!(cards.len(), 1);
        assert_eq!(cards[0].title, "Hello World");
        assert!(cards[0].end.pts > cards[0].start.pts);
    }

    #[test]
    fn test_duration_clamped_to_min() {
        let gen = make_generator();
        // Single character → well below min_duration_ms
        let dur = gen.compute_duration_ms(1);
        assert!(dur >= gen.config.min_duration_ms);
    }

    #[test]
    fn test_duration_clamped_to_max() {
        let gen = make_generator();
        // 10 000 chars → would exceed max
        let dur = gen.compute_duration_ms(10_000);
        assert!(dur <= gen.config.max_duration_ms);
    }

    #[test]
    fn test_duration_override() {
        let gen = make_generator();
        let req =
            TitleCardRequest::new("Title", 0, StyleTemplate::default()).with_duration_ms(3_000);
        let cards = gen.generate(&[req]).unwrap();
        assert_eq!(cards[0].duration_ms(), 3_000);
    }

    #[test]
    fn test_subtitle_attached() {
        let gen = make_generator();
        let req = TitleCardRequest::new("Title", 0, StyleTemplate::default()).with_subtitle("Sub");
        let cards = gen.generate(&[req]).unwrap();
        assert_eq!(cards[0].subtitle.as_deref(), Some("Sub"));
    }

    #[test]
    fn test_fade_opacity_envelope() {
        let fade = FadeTiming {
            fade_in_ms: 500,
            hold_ms: 1_000,
            fade_out_ms: 500,
        };
        // At t=0 opacity should be 0
        assert!((fade.opacity_at(0) - 0.0).abs() < 1e-6);
        // At t=250 (mid fade-in) opacity should be ~0.5
        assert!((fade.opacity_at(250) - 0.5).abs() < 1e-4);
        // At t=500 (start of hold) opacity should be 1.0
        assert!((fade.opacity_at(500) - 1.0).abs() < 1e-6);
        // At t=1500 (end of hold) opacity should be 1.0
        assert!((fade.opacity_at(1_500) - 1.0).abs() < 1e-6);
        // At t=2000 (end of fade-out) opacity should be 0.0
        assert!(fade.opacity_at(2_000) < 1e-4);
    }

    #[test]
    fn test_layout_sequence() {
        let gen = make_generator();
        let titles = vec![
            ("Intro", None),
            ("Chapter 1", Some("The Beginning")),
            ("Chapter 2", None),
        ];
        let cards = gen.layout_sequence(&titles, &[], 0).unwrap();
        assert_eq!(cards.len(), 3);
        // Cards should be laid out in order
        assert!(cards[1].start.pts >= cards[0].end.pts);
        assert!(cards[2].start.pts >= cards[1].end.pts);
    }

    #[test]
    fn test_layout_sequence_with_templates() {
        let gen = make_generator();
        let templates = vec![StyleTemplate::cinematic(), StyleTemplate::lower_third()];
        let titles = vec![("A", None), ("B", None), ("C", None)];
        let cards = gen.layout_sequence(&titles, &templates, 1_000).unwrap();
        assert_eq!(cards.len(), 3);
        // Templates cycle
        assert_eq!(cards[0].style.name, "cinematic");
        assert_eq!(cards[1].style.name, "lower_third");
        assert_eq!(cards[2].style.name, "cinematic");
    }

    #[test]
    fn test_invalid_config_rejected() {
        let config = TitleCardGeneratorConfig {
            reading_speed_cps: -1.0,
            ..Default::default()
        };
        let gen = TitleCardGenerator::new(config);
        let result = gen.generate(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_style_rejected() {
        let gen = make_generator();
        let bad_style = StyleTemplate {
            font_size_pt: -10.0,
            ..StyleTemplate::default()
        };
        let req = TitleCardRequest::new("T", 0, bad_style);
        assert!(gen.generate(&[req]).is_err());
    }

    #[test]
    fn test_v_position_anchors_in_range() {
        for pos in [
            VPosition::Top,
            VPosition::Middle,
            VPosition::LowerThird,
            VPosition::Bottom,
        ] {
            let y = pos.y_anchor();
            assert!((0.0..=1.0).contains(&y));
        }
    }

    #[test]
    fn test_multiple_cards_ordering() {
        let gen = make_generator();
        let requests = vec![
            TitleCardRequest::new("First", 0, StyleTemplate::default()),
            TitleCardRequest::new("Second", 10_000, StyleTemplate::default()),
            TitleCardRequest::new("Third", 20_000, StyleTemplate::default()),
        ];
        let cards = gen.generate(&requests).unwrap();
        assert!(cards[0].start.pts < cards[1].start.pts);
        assert!(cards[1].start.pts < cards[2].start.pts);
    }
}
