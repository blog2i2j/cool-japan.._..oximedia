//! Social media clip formatter.
//!
//! Prepares video clips for specific social media platforms by computing:
//!
//! - **Aspect ratio conversion**: letter-box/pillar-box geometry for 9:16, 1:1,
//!   4:5, and 16:9 targets.
//! - **Caption burn-in specification**: font size, safe zone, line wrapping, and
//!   position anchoring following each platform's visual standards.
//! - **Platform-specific duration enforcement**: trim or pad clips to the
//!   official limits for TikTok, Instagram Reels, YouTube Shorts, Twitter/X,
//!   LinkedIn, and Facebook Reels.
//! - **Thumbnail frame suggestion**: percentage offset into the clip to find the
//!   most representative frame.
//! - **Output spec generation**: a self-contained `SocialClipSpec` that a
//!   downstream renderer can execute without additional configuration.
//!
//! All logic is pure arithmetic — no I/O or network dependencies.
//!
//! # Example
//!
//! ```
//! use oximedia_auto::social_clip_formatter::{
//!     SocialClipFormatter, FormatterConfig, Platform, AspectRatioTarget,
//! };
//! use oximedia_core::{Rational, Timestamp};
//!
//! let config = FormatterConfig::for_platform(Platform::InstagramReels);
//! let formatter = SocialClipFormatter::new(config);
//!
//! let tb = Rational::new(1, 1000);
//! let start = Timestamp::new(0, tb);
//! let end = Timestamp::new(45_000, tb); // 45-second clip
//!
//! let spec = formatter.format(start, end, 1920, 1080).expect("format clip");
//! assert!(spec.output_duration_ms() <= 60_000);
//! ```

#![allow(dead_code)]

use crate::error::{AutoError, AutoResult};
use oximedia_core::Timestamp;

// ─── Platform ─────────────────────────────────────────────────────────────────

/// Supported social media platforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Platform {
    /// TikTok — vertical, 9:16, up to 60 s (standard) / 3 min (extended).
    TikTok,
    /// Instagram Reels — vertical, 9:16, up to 90 s.
    InstagramReels,
    /// Instagram Feed — square or landscape/portrait, up to 60 s.
    InstagramFeed,
    /// YouTube Shorts — vertical, 9:16, up to 60 s.
    YouTubeShorts,
    /// Twitter / X — landscape or square, up to 140 s.
    TwitterX,
    /// LinkedIn — landscape or square, up to 10 min.
    LinkedIn,
    /// Facebook Reels — vertical, 9:16, up to 90 s.
    FacebookReels,
}

impl Platform {
    /// Recommended aspect ratio for this platform.
    #[must_use]
    pub const fn recommended_aspect_ratio(&self) -> AspectRatioTarget {
        match self {
            Self::TikTok | Self::InstagramReels | Self::YouTubeShorts | Self::FacebookReels => {
                AspectRatioTarget::Vertical9x16
            }
            Self::InstagramFeed => AspectRatioTarget::Square1x1,
            Self::TwitterX | Self::LinkedIn => AspectRatioTarget::Landscape16x9,
        }
    }

    /// Maximum clip duration in milliseconds (None = no platform-imposed limit).
    #[must_use]
    pub const fn max_duration_ms(&self) -> Option<i64> {
        match self {
            Self::TikTok => Some(60_000),
            Self::InstagramReels | Self::FacebookReels => Some(90_000),
            Self::YouTubeShorts => Some(60_000),
            Self::TwitterX => Some(140_000),
            Self::LinkedIn => Some(600_000),
            Self::InstagramFeed => Some(60_000),
        }
    }

    /// Minimum clip duration in milliseconds.
    #[must_use]
    pub const fn min_duration_ms(&self) -> i64 {
        match self {
            Self::TikTok | Self::YouTubeShorts | Self::FacebookReels => 3_000,
            Self::InstagramReels | Self::InstagramFeed => 3_000,
            Self::TwitterX | Self::LinkedIn => 1_000,
        }
    }

    /// Human-readable display name.
    #[must_use]
    pub const fn display_name(&self) -> &'static str {
        match self {
            Self::TikTok => "TikTok",
            Self::InstagramReels => "Instagram Reels",
            Self::InstagramFeed => "Instagram Feed",
            Self::YouTubeShorts => "YouTube Shorts",
            Self::TwitterX => "Twitter/X",
            Self::LinkedIn => "LinkedIn",
            Self::FacebookReels => "Facebook Reels",
        }
    }
}

// ─── AspectRatioTarget ────────────────────────────────────────────────────────

/// Target aspect ratio for the output clip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AspectRatioTarget {
    /// 9:16 — vertical (portrait) for short-form video.
    Vertical9x16,
    /// 1:1 — square.
    Square1x1,
    /// 4:5 — portrait, commonly used for Instagram Feed.
    Portrait4x5,
    /// 16:9 — landscape (standard HD).
    Landscape16x9,
}

impl AspectRatioTarget {
    /// Rational width/height as (numerator, denominator).
    #[must_use]
    pub const fn ratio(&self) -> (u32, u32) {
        match self {
            Self::Vertical9x16 => (9, 16),
            Self::Square1x1 => (1, 1),
            Self::Portrait4x5 => (4, 5),
            Self::Landscape16x9 => (16, 9),
        }
    }

    /// Compute output canvas dimensions that fit `source_w × source_h` while
    /// maintaining the target ratio, using the supplied `base_long_side` as
    /// the longer dimension of the output canvas.
    ///
    /// Returns `(canvas_width, canvas_height)`.
    #[must_use]
    pub fn canvas_dimensions(&self, base_long_side: u32) -> (u32, u32) {
        let (rw, rh) = self.ratio();
        if rw >= rh {
            // Landscape or square: width is the long side
            let w = base_long_side;
            let h = (base_long_side as u64 * rh as u64 / rw as u64) as u32;
            (w, h)
        } else {
            // Portrait: height is the long side
            let h = base_long_side;
            let w = (base_long_side as u64 * rw as u64 / rh as u64) as u32;
            (w, h)
        }
    }

    /// Compute the crop/letterbox rectangle needed to fit `src_w × src_h` into
    /// this aspect ratio.
    ///
    /// Returns `CropGeometry`: what region to sample from the source and where
    /// to place it on the canvas.
    #[must_use]
    pub fn crop_geometry(
        &self,
        src_w: u32,
        src_h: u32,
        canvas_w: u32,
        canvas_h: u32,
    ) -> CropGeometry {
        if src_w == 0 || src_h == 0 || canvas_w == 0 || canvas_h == 0 {
            return CropGeometry::identity(src_w, src_h);
        }

        let src_ratio = src_w as f64 / src_h as f64;
        let dst_ratio = canvas_w as f64 / canvas_h as f64;

        if (src_ratio - dst_ratio).abs() < 1e-4 {
            // Same ratio — no crop needed
            return CropGeometry {
                src_x: 0,
                src_y: 0,
                src_w,
                src_h,
                dst_x: 0,
                dst_y: 0,
                dst_w: canvas_w,
                dst_h: canvas_h,
                letterbox_color: [0u8; 3],
            };
        }

        if src_ratio > dst_ratio {
            // Source is wider → crop horizontally (pillar-box)
            let crop_h = src_h;
            let crop_w = (src_h as f64 * dst_ratio).round() as u32;
            let crop_x = (src_w - crop_w) / 2;
            CropGeometry {
                src_x: crop_x,
                src_y: 0,
                src_w: crop_w,
                src_h: crop_h,
                dst_x: 0,
                dst_y: 0,
                dst_w: canvas_w,
                dst_h: canvas_h,
                letterbox_color: [0u8; 3],
            }
        } else {
            // Source is taller → crop vertically (letter-box)
            let crop_w = src_w;
            let crop_h = (src_w as f64 / dst_ratio).round() as u32;
            let crop_y = (src_h - crop_h.min(src_h)) / 2;
            CropGeometry {
                src_x: 0,
                src_y: crop_y,
                src_w: crop_w,
                src_h: crop_h.min(src_h),
                dst_x: 0,
                dst_y: 0,
                dst_w: canvas_w,
                dst_h: canvas_h,
                letterbox_color: [0u8; 3],
            }
        }
    }
}

// ─── CropGeometry ─────────────────────────────────────────────────────────────

/// Describes the crop and placement geometry for aspect ratio conversion.
#[derive(Debug, Clone, Copy)]
pub struct CropGeometry {
    /// X offset in the source frame to begin cropping.
    pub src_x: u32,
    /// Y offset in the source frame to begin cropping.
    pub src_y: u32,
    /// Width of the crop region in the source.
    pub src_w: u32,
    /// Height of the crop region in the source.
    pub src_h: u32,
    /// X offset on the destination canvas where the crop is placed.
    pub dst_x: u32,
    /// Y offset on the destination canvas where the crop is placed.
    pub dst_y: u32,
    /// Width of the region on the destination canvas.
    pub dst_w: u32,
    /// Height of the region on the destination canvas.
    pub dst_h: u32,
    /// RGB color to fill letterbox/pillarbox bars.
    pub letterbox_color: [u8; 3],
}

impl CropGeometry {
    fn identity(w: u32, h: u32) -> Self {
        Self {
            src_x: 0,
            src_y: 0,
            src_w: w,
            src_h: h,
            dst_x: 0,
            dst_y: 0,
            dst_w: w,
            dst_h: h,
            letterbox_color: [0u8; 3],
        }
    }

    /// Whether the conversion requires any cropping.
    #[must_use]
    pub fn requires_crop(&self) -> bool {
        self.src_x != 0 || self.src_y != 0 || self.src_w != self.dst_w || self.src_h != self.dst_h
    }
}

// ─── CaptionSpec ──────────────────────────────────────────────────────────────

/// Vertical anchor for caption placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptionAnchor {
    /// Captions near the bottom of the frame (most platforms).
    Bottom,
    /// Captions centered (accessibility / karaoke style).
    Center,
    /// Captions near the top (avoids UI overlays on some platforms).
    Top,
}

/// Specification for burning captions into a social media clip.
#[derive(Debug, Clone)]
pub struct CaptionSpec {
    /// Recommended font size in points relative to canvas height.
    pub font_size_pt: f32,
    /// Maximum characters per line before wrapping.
    pub max_chars_per_line: usize,
    /// Vertical anchor for caption text.
    pub anchor: CaptionAnchor,
    /// Bottom / top margin as a fraction of canvas height.
    pub margin_fraction: f32,
    /// Safe-zone inset as a fraction of canvas width (left/right).
    pub safe_zone_h_fraction: f32,
    /// Whether to render a semi-transparent background behind the text.
    pub background_box: bool,
    /// Background opacity [0.0, 1.0].
    pub background_opacity: f32,
    /// Whether to render drop-shadows on the text.
    pub drop_shadow: bool,
    /// Maximum number of simultaneously visible caption lines.
    pub max_lines: usize,
}

impl CaptionSpec {
    /// Standard specification for vertical (9:16) short-form platforms.
    #[must_use]
    pub fn vertical_short_form() -> Self {
        Self {
            font_size_pt: 28.0,
            max_chars_per_line: 32,
            anchor: CaptionAnchor::Bottom,
            margin_fraction: 0.12,
            safe_zone_h_fraction: 0.08,
            background_box: true,
            background_opacity: 0.65,
            drop_shadow: false,
            max_lines: 2,
        }
    }

    /// Standard specification for landscape / square platforms.
    #[must_use]
    pub fn landscape_standard() -> Self {
        Self {
            font_size_pt: 22.0,
            max_chars_per_line: 48,
            anchor: CaptionAnchor::Bottom,
            margin_fraction: 0.08,
            safe_zone_h_fraction: 0.05,
            background_box: true,
            background_opacity: 0.70,
            drop_shadow: true,
            max_lines: 2,
        }
    }

    /// Compute absolute pixel values for a given canvas size.
    ///
    /// Returns `(font_size_px, margin_px, safe_zone_px_left_right)`.
    #[must_use]
    pub fn resolve_pixels(&self, canvas_w: u32, canvas_h: u32) -> (u32, u32, u32) {
        let font_px = (canvas_h as f32 * self.font_size_pt / 100.0).round() as u32;
        let margin_px = (canvas_h as f32 * self.margin_fraction).round() as u32;
        let safe_px = (canvas_w as f32 * self.safe_zone_h_fraction).round() as u32;
        (font_px, margin_px, safe_px)
    }
}

// ─── DurationSpec ─────────────────────────────────────────────────────────────

/// How to handle a clip whose duration falls outside the platform's limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DurationStrategy {
    /// Trim the end of the clip to meet the maximum.
    TrimEnd,
    /// Trim from the highest-score point (center by default when score unknown).
    TrimBestSegment,
    /// Reject the clip with an error if it's out of range.
    Reject,
}

// ─── FormatterConfig ──────────────────────────────────────────────────────────

/// Configuration for the social clip formatter.
#[derive(Debug, Clone)]
pub struct FormatterConfig {
    /// Target platform.
    pub platform: Platform,
    /// Target aspect ratio (defaults to platform recommendation).
    pub aspect_ratio: AspectRatioTarget,
    /// Long side of the output canvas in pixels.
    pub canvas_long_side: u32,
    /// Strategy when clip duration exceeds platform maximum.
    pub duration_strategy: DurationStrategy,
    /// Caption specification (None = no captions).
    pub caption_spec: Option<CaptionSpec>,
    /// Fraction of the clip duration at which to suggest the thumbnail frame.
    pub thumbnail_offset_fraction: f64,
}

impl FormatterConfig {
    /// Create a configuration for a specific platform with sensible defaults.
    #[must_use]
    pub fn for_platform(platform: Platform) -> Self {
        let aspect_ratio = platform.recommended_aspect_ratio();
        let caption_spec = match aspect_ratio {
            AspectRatioTarget::Vertical9x16 | AspectRatioTarget::Portrait4x5 => {
                Some(CaptionSpec::vertical_short_form())
            }
            _ => Some(CaptionSpec::landscape_standard()),
        };
        Self {
            platform,
            aspect_ratio,
            canvas_long_side: 1920,
            duration_strategy: DurationStrategy::TrimEnd,
            caption_spec,
            thumbnail_offset_fraction: 0.33,
        }
    }

    /// Validate configuration.
    pub fn validate(&self) -> AutoResult<()> {
        if self.canvas_long_side == 0 {
            return Err(AutoError::invalid_parameter(
                "canvas_long_side",
                "must be > 0",
            ));
        }
        if !(0.0..=1.0).contains(&self.thumbnail_offset_fraction) {
            return Err(AutoError::InvalidThreshold {
                threshold: self.thumbnail_offset_fraction,
                min: 0.0,
                max: 1.0,
            });
        }
        Ok(())
    }
}

// ─── SocialClipSpec ───────────────────────────────────────────────────────────

/// A fully-resolved specification for exporting a clip to a social platform.
#[derive(Debug, Clone)]
pub struct SocialClipSpec {
    /// Target platform.
    pub platform: Platform,
    /// Source clip in-point (ms).
    pub source_start: Timestamp,
    /// Source clip out-point (ms), possibly trimmed.
    pub source_end: Timestamp,
    /// Output canvas width in pixels.
    pub canvas_width: u32,
    /// Output canvas height in pixels.
    pub canvas_height: u32,
    /// Crop / letterbox geometry to convert from source to canvas.
    pub crop: CropGeometry,
    /// Caption burn-in specification (None if disabled).
    pub caption_spec: Option<CaptionSpec>,
    /// Suggested thumbnail frame timestamp.
    pub thumbnail_timestamp: Timestamp,
    /// Whether the clip was trimmed to fit the platform limit.
    pub was_trimmed: bool,
    /// Any informational warnings generated during formatting.
    pub warnings: Vec<String>,
}

impl SocialClipSpec {
    /// Duration of the output clip in milliseconds.
    #[must_use]
    pub fn output_duration_ms(&self) -> i64 {
        (self.source_end.pts - self.source_start.pts).max(0)
    }

    /// Whether captions are enabled for this clip.
    #[must_use]
    pub fn has_captions(&self) -> bool {
        self.caption_spec.is_some()
    }
}

// ─── SocialClipFormatter ──────────────────────────────────────────────────────

/// Formats clip parameters for social media export.
pub struct SocialClipFormatter {
    config: FormatterConfig,
}

impl SocialClipFormatter {
    /// Create a new formatter with the given configuration.
    #[must_use]
    pub fn new(config: FormatterConfig) -> Self {
        Self { config }
    }

    /// Format a clip for social media export.
    ///
    /// `src_start` and `src_end` describe the clip in the source timeline.
    /// `src_w` × `src_h` are the source frame dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid, the clip is too
    /// short for the platform, or `DurationStrategy::Reject` is set and the
    /// clip exceeds the platform maximum.
    pub fn format(
        &self,
        src_start: Timestamp,
        src_end: Timestamp,
        src_w: u32,
        src_h: u32,
    ) -> AutoResult<SocialClipSpec> {
        self.config.validate()?;

        let platform = self.config.platform;
        let raw_duration_ms = (src_end.pts - src_start.pts).max(0);
        let mut warnings: Vec<String> = Vec::new();

        // --- Duration enforcement ---
        if raw_duration_ms < platform.min_duration_ms() {
            return Err(AutoError::InvalidDuration {
                duration_ms: raw_duration_ms,
            });
        }

        let (effective_end, was_trimmed) = if let Some(max_ms) = platform.max_duration_ms() {
            if raw_duration_ms > max_ms {
                match self.config.duration_strategy {
                    DurationStrategy::Reject => {
                        return Err(AutoError::InvalidDuration {
                            duration_ms: raw_duration_ms,
                        });
                    }
                    DurationStrategy::TrimEnd | DurationStrategy::TrimBestSegment => {
                        let trimmed_end =
                            Timestamp::new(src_start.pts + max_ms, src_start.timebase);
                        warnings.push(format!(
                            "Clip trimmed from {}ms to {}ms for {}",
                            raw_duration_ms,
                            max_ms,
                            platform.display_name()
                        ));
                        (trimmed_end, true)
                    }
                }
            } else {
                (src_end, false)
            }
        } else {
            (src_end, false)
        };

        // Warn if clip is on the short side but still valid
        let output_duration = (effective_end.pts - src_start.pts).max(0);
        if output_duration < platform.min_duration_ms() * 2 {
            warnings.push(format!(
                "Clip is short ({}ms); may underperform on {}",
                output_duration,
                platform.display_name()
            ));
        }

        // --- Canvas geometry ---
        let (canvas_w, canvas_h) = self
            .config
            .aspect_ratio
            .canvas_dimensions(self.config.canvas_long_side);

        // --- Crop geometry ---
        let crop = self
            .config
            .aspect_ratio
            .crop_geometry(src_w, src_h, canvas_w, canvas_h);

        // --- Thumbnail frame ---
        let thumbnail_ts = Timestamp::new(
            src_start.pts + (output_duration as f64 * self.config.thumbnail_offset_fraction) as i64,
            src_start.timebase,
        );

        Ok(SocialClipSpec {
            platform,
            source_start: src_start,
            source_end: effective_end,
            canvas_width: canvas_w,
            canvas_height: canvas_h,
            crop,
            caption_spec: self.config.caption_spec.clone(),
            thumbnail_timestamp: thumbnail_ts,
            was_trimmed,
            warnings,
        })
    }

    /// Batch-format multiple clips for the same platform.
    ///
    /// Errors in individual clips are collected and returned together rather than
    /// stopping on the first failure.  Returns `(specs, errors)`.
    #[must_use]
    pub fn format_batch(
        &self,
        clips: &[(Timestamp, Timestamp, u32, u32)],
    ) -> (Vec<SocialClipSpec>, Vec<(usize, AutoError)>) {
        let mut specs = Vec::with_capacity(clips.len());
        let mut errors = Vec::new();

        for (i, &(start, end, w, h)) in clips.iter().enumerate() {
            match self.format(start, end, w, h) {
                Ok(spec) => specs.push(spec),
                Err(e) => errors.push((i, e)),
            }
        }

        (specs, errors)
    }

    /// Get the current configuration.
    #[must_use]
    pub const fn config(&self) -> &FormatterConfig {
        &self.config
    }

    /// Return platform display name.
    #[must_use]
    pub fn platform_name(&self) -> &'static str {
        self.config.platform.display_name()
    }
}

// ─── Convenience constructors ─────────────────────────────────────────────────

/// Create a formatter preset for TikTok.
#[must_use]
pub fn tiktok_formatter() -> SocialClipFormatter {
    SocialClipFormatter::new(FormatterConfig::for_platform(Platform::TikTok))
}

/// Create a formatter preset for Instagram Reels.
#[must_use]
pub fn instagram_reels_formatter() -> SocialClipFormatter {
    SocialClipFormatter::new(FormatterConfig::for_platform(Platform::InstagramReels))
}

/// Create a formatter preset for YouTube Shorts.
#[must_use]
pub fn youtube_shorts_formatter() -> SocialClipFormatter {
    SocialClipFormatter::new(FormatterConfig::for_platform(Platform::YouTubeShorts))
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::Rational;

    fn tb() -> Rational {
        Rational::new(1, 1000)
    }

    fn ts(ms: i64) -> Timestamp {
        Timestamp::new(ms, tb())
    }

    fn landscape_clip(duration_ms: i64) -> (Timestamp, Timestamp, u32, u32) {
        (ts(0), ts(duration_ms), 1920, 1080)
    }

    #[test]
    fn test_platform_limits_tiktok() {
        assert_eq!(Platform::TikTok.max_duration_ms(), Some(60_000));
        assert_eq!(Platform::TikTok.min_duration_ms(), 3_000);
    }

    #[test]
    fn test_platform_display_names() {
        assert_eq!(Platform::TikTok.display_name(), "TikTok");
        assert_eq!(Platform::InstagramReels.display_name(), "Instagram Reels");
        assert_eq!(Platform::YouTubeShorts.display_name(), "YouTube Shorts");
    }

    #[test]
    fn test_aspect_ratio_canvas_vertical() {
        let (w, h) = AspectRatioTarget::Vertical9x16.canvas_dimensions(1920);
        assert_eq!(h, 1920);
        assert_eq!(w, 1080);
    }

    #[test]
    fn test_aspect_ratio_canvas_square() {
        let (w, h) = AspectRatioTarget::Square1x1.canvas_dimensions(1080);
        assert_eq!(w, h);
        assert_eq!(w, 1080);
    }

    #[test]
    fn test_crop_geometry_landscape_to_vertical() {
        // Convert 1920×1080 (16:9) to 9:16 canvas
        let crop = AspectRatioTarget::Vertical9x16.crop_geometry(1920, 1080, 1080, 1920);
        // Source is wider → should crop horizontally
        assert!(crop.requires_crop());
        assert_eq!(crop.src_y, 0); // no vertical crop
        assert_eq!(crop.src_h, 1080); // full height used
                                      // Cropped width = 1080 * (9/16) = 607 (approx)
                                      // centre-crop: src_x = (1920 - 607) / 2
        assert!(crop.src_x > 0, "should have horizontal offset");
    }

    #[test]
    fn test_crop_geometry_same_ratio_no_crop() {
        let crop = AspectRatioTarget::Landscape16x9.crop_geometry(1920, 1080, 1920, 1080);
        assert!(!crop.requires_crop());
    }

    #[test]
    fn test_format_basic_tiktok() {
        let formatter = tiktok_formatter();
        let (start, end, w, h) = landscape_clip(30_000);
        let spec = formatter.format(start, end, w, h).expect("should succeed");
        assert_eq!(spec.platform, Platform::TikTok);
        assert_eq!(spec.output_duration_ms(), 30_000);
        assert!(!spec.was_trimmed);
    }

    #[test]
    fn test_format_trims_to_platform_max() {
        let formatter = tiktok_formatter(); // max = 60 s
        let (start, end, w, h) = landscape_clip(90_000); // 90 s
        let spec = formatter.format(start, end, w, h).expect("should succeed");
        assert!(spec.was_trimmed);
        assert_eq!(spec.output_duration_ms(), 60_000);
        assert!(!spec.warnings.is_empty());
    }

    #[test]
    fn test_format_rejects_too_short() {
        let formatter = tiktok_formatter(); // min = 3 s
        let (start, end, w, h) = landscape_clip(1_000); // 1 s
        assert!(formatter.format(start, end, w, h).is_err());
    }

    #[test]
    fn test_format_reject_strategy_on_long_clip() {
        let mut config = FormatterConfig::for_platform(Platform::TikTok);
        config.duration_strategy = DurationStrategy::Reject;
        let formatter = SocialClipFormatter::new(config);
        let (start, end, w, h) = landscape_clip(90_000);
        assert!(formatter.format(start, end, w, h).is_err());
    }

    #[test]
    fn test_format_canvas_dimensions_vertical() {
        let formatter = tiktok_formatter();
        let (start, end, w, h) = landscape_clip(30_000);
        let spec = formatter.format(start, end, w, h).expect("should succeed");
        // TikTok is vertical 9:16 → canvas height > canvas width
        assert!(spec.canvas_height > spec.canvas_width);
    }

    #[test]
    fn test_thumbnail_timestamp_within_clip() {
        let formatter = tiktok_formatter();
        let (start, end, w, h) = landscape_clip(30_000);
        let spec = formatter.format(start, end, w, h).expect("should succeed");
        assert!(spec.thumbnail_timestamp.pts >= start.pts);
        assert!(spec.thumbnail_timestamp.pts <= end.pts);
    }

    #[test]
    fn test_has_captions_enabled_by_default() {
        let formatter = tiktok_formatter();
        let (start, end, w, h) = landscape_clip(30_000);
        let spec = formatter.format(start, end, w, h).expect("should succeed");
        assert!(spec.has_captions());
    }

    #[test]
    fn test_format_batch_mixed_results() {
        let formatter = tiktok_formatter();
        let clips = vec![
            (ts(0), ts(30_000), 1920u32, 1080u32), // valid
            (ts(0), ts(500), 1920u32, 1080u32),    // too short → error
            (ts(0), ts(15_000), 1920u32, 1080u32), // valid
        ];
        let (specs, errors) = formatter.format_batch(&clips);
        assert_eq!(specs.len(), 2);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].0, 1); // index 1 failed
    }

    #[test]
    fn test_caption_spec_resolve_pixels() {
        let spec = CaptionSpec::vertical_short_form();
        let (font_px, margin_px, safe_px) = spec.resolve_pixels(1080, 1920);
        assert!(font_px > 0);
        assert!(margin_px > 0);
        assert!(safe_px > 0);
    }

    #[test]
    fn test_instagram_reels_formatter() {
        let formatter = instagram_reels_formatter();
        let (start, end, w, h) = landscape_clip(60_000);
        let spec = formatter.format(start, end, w, h).expect("should succeed");
        assert_eq!(spec.platform, Platform::InstagramReels);
        assert!(!spec.was_trimmed); // 60 s ≤ 90 s limit
    }

    #[test]
    fn test_youtube_shorts_formatter() {
        let formatter = youtube_shorts_formatter();
        assert_eq!(formatter.platform_name(), "YouTube Shorts");
    }

    #[test]
    fn test_linkedin_no_max_duration_limit() {
        let formatter = SocialClipFormatter::new(FormatterConfig::for_platform(Platform::LinkedIn));
        // 5-minute clip
        let (start, end, w, h) = landscape_clip(300_000);
        let spec = formatter.format(start, end, w, h).expect("should succeed");
        assert!(!spec.was_trimmed);
        assert_eq!(spec.output_duration_ms(), 300_000);
    }

    #[test]
    fn test_config_invalid_canvas_long_side() {
        let mut config = FormatterConfig::for_platform(Platform::TikTok);
        config.canvas_long_side = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_invalid_thumbnail_offset() {
        let mut config = FormatterConfig::for_platform(Platform::TikTok);
        config.thumbnail_offset_fraction = 1.5;
        assert!(config.validate().is_err());
    }
}
