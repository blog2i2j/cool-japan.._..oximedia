//! Configuration types for the timecode burn-in filter.
//!
//! This module contains all enums, structs, and builder patterns that
//! define the visual appearance and behaviour of the timecode overlay.

#![allow(clippy::derivable_impls)]

use oximedia_core::Rational;

// ── TimecodeFormat ────────────────────────────────────────────────────────────

/// SMPTE timecode format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimecodeFormat {
    /// SMPTE timecode at 23.976 fps (non-drop-frame).
    Smpte23976,
    /// SMPTE timecode at 24 fps.
    Smpte24,
    /// SMPTE timecode at 25 fps (PAL).
    Smpte25,
    /// SMPTE timecode at 29.97 fps drop-frame.
    Smpte2997Df,
    /// SMPTE timecode at 29.97 fps non-drop-frame.
    Smpte2997Ndf,
    /// SMPTE timecode at 30 fps.
    Smpte30,
    /// SMPTE timecode at 60 fps.
    Smpte60,
    /// Frame count (0, 1, 2, ...).
    FrameCount,
    /// Milliseconds (0, 16, 33, ...).
    Milliseconds,
    /// Seconds (0.000, 0.016, 0.033, ...).
    Seconds,
    /// HH:MM:SS.mmm format.
    HhMmSsMmm,
}

impl TimecodeFormat {
    /// Get the frame rate for this timecode format.
    #[must_use]
    pub fn framerate(&self) -> Rational {
        match self {
            Self::Smpte23976 => Rational::new(24000, 1001),
            Self::Smpte24 => Rational::new(24, 1),
            Self::Smpte25 => Rational::new(25, 1),
            Self::Smpte2997Df | Self::Smpte2997Ndf => Rational::new(30000, 1001),
            Self::Smpte30 => Rational::new(30, 1),
            Self::Smpte60 => Rational::new(60, 1),
            Self::FrameCount | Self::Milliseconds | Self::Seconds | Self::HhMmSsMmm => {
                Rational::new(1, 1)
            }
        }
    }

    /// Check if this format uses drop-frame timecode.
    #[must_use]
    pub fn is_drop_frame(&self) -> bool {
        matches!(self, Self::Smpte2997Df)
    }

    /// Format a frame number as timecode.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn format_timecode(&self, frame_number: u64, fps: &Rational) -> String {
        match self {
            Self::Smpte23976
            | Self::Smpte24
            | Self::Smpte25
            | Self::Smpte2997Ndf
            | Self::Smpte30
            | Self::Smpte60 => {
                let fps_val = fps.to_f64();
                let total_frames = frame_number;
                let hours = total_frames / (fps_val * 3600.0) as u64;
                let minutes = (total_frames / (fps_val * 60.0) as u64) % 60;
                let seconds = (total_frames / fps_val as u64) % 60;
                let frames = total_frames % fps_val as u64;
                format!("{hours:02}:{minutes:02}:{seconds:02}:{frames:02}")
            }
            Self::Smpte2997Df => {
                let fps_val = 30.0;
                let drop_frames = 2;
                let frames_per_minute = (fps_val * 60.0) as u64 - drop_frames;
                let frames_per_10_minutes = frames_per_minute * 10 + drop_frames;

                let mut total_frames = frame_number;
                let tens_of_minutes = total_frames / frames_per_10_minutes;
                total_frames %= frames_per_10_minutes;

                let mut minutes = if total_frames < drop_frames {
                    0
                } else {
                    (total_frames - drop_frames) / frames_per_minute + 1
                };
                let mut frames_in_minute = if total_frames < drop_frames {
                    total_frames
                } else {
                    (total_frames - drop_frames) % frames_per_minute + drop_frames
                };

                let hours = (tens_of_minutes * 10 + minutes) / 60;
                minutes = (tens_of_minutes * 10 + minutes) % 60;
                let seconds = frames_in_minute / fps_val as u64;
                frames_in_minute %= fps_val as u64;

                format!(
                    "{:02}:{:02}:{:02};{:02}",
                    hours, minutes, seconds, frames_in_minute
                )
            }
            Self::FrameCount => format!("{frame_number}"),
            Self::Milliseconds => {
                let ms = (frame_number as f64 / fps.to_f64() * 1000.0) as u64;
                format!("{ms}")
            }
            Self::Seconds => {
                let seconds = frame_number as f64 / fps.to_f64();
                format!("{seconds:.3}")
            }
            Self::HhMmSsMmm => {
                let total_ms = (frame_number as f64 / fps.to_f64() * 1000.0) as u64;
                let hours = total_ms / 3_600_000;
                let minutes = (total_ms / 60_000) % 60;
                let seconds = (total_ms / 1_000) % 60;
                let milliseconds = total_ms % 1_000;
                format!("{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}")
            }
        }
    }
}

impl Default for TimecodeFormat {
    fn default() -> Self {
        Self::Smpte25
    }
}

// ── Position ──────────────────────────────────────────────────────────────────

/// Position for overlay elements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Position {
    /// Top-left corner.
    TopLeft,
    /// Top-center.
    TopCenter,
    /// Top-right corner.
    TopRight,
    /// Center-left.
    CenterLeft,
    /// Center.
    Center,
    /// Center-right.
    CenterRight,
    /// Bottom-left corner.
    BottomLeft,
    /// Bottom-center.
    BottomCenter,
    /// Bottom-right corner.
    BottomRight,
    /// Custom X/Y coordinates (absolute pixels).
    Custom(i32, i32),
}

impl Position {
    /// Calculate the actual position in pixels.
    #[must_use]
    pub fn calculate(
        &self,
        frame_width: u32,
        frame_height: u32,
        element_width: u32,
        element_height: u32,
        margin: u32,
    ) -> (i32, i32) {
        match self {
            Self::TopLeft => (margin as i32, margin as i32),
            Self::TopCenter => (
                (frame_width.saturating_sub(element_width) / 2) as i32,
                margin as i32,
            ),
            Self::TopRight => (
                (frame_width
                    .saturating_sub(element_width)
                    .saturating_sub(margin)) as i32,
                margin as i32,
            ),
            Self::CenterLeft => (
                margin as i32,
                (frame_height.saturating_sub(element_height) / 2) as i32,
            ),
            Self::Center => (
                (frame_width.saturating_sub(element_width) / 2) as i32,
                (frame_height.saturating_sub(element_height) / 2) as i32,
            ),
            Self::CenterRight => (
                (frame_width
                    .saturating_sub(element_width)
                    .saturating_sub(margin)) as i32,
                (frame_height.saturating_sub(element_height) / 2) as i32,
            ),
            Self::BottomLeft => (
                margin as i32,
                (frame_height
                    .saturating_sub(element_height)
                    .saturating_sub(margin)) as i32,
            ),
            Self::BottomCenter => (
                (frame_width.saturating_sub(element_width) / 2) as i32,
                (frame_height
                    .saturating_sub(element_height)
                    .saturating_sub(margin)) as i32,
            ),
            Self::BottomRight => (
                (frame_width
                    .saturating_sub(element_width)
                    .saturating_sub(margin)) as i32,
                (frame_height
                    .saturating_sub(element_height)
                    .saturating_sub(margin)) as i32,
            ),
            Self::Custom(x, y) => (*x, *y),
        }
    }
}

impl Default for Position {
    fn default() -> Self {
        Self::TopLeft
    }
}

// ── Color ─────────────────────────────────────────────────────────────────────

/// Color representation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Color {
    /// Red component (0-255).
    pub r: u8,
    /// Green component (0-255).
    pub g: u8,
    /// Blue component (0-255).
    pub b: u8,
    /// Alpha component (0-255, 255 = opaque).
    pub a: u8,
}

impl Color {
    /// Create a new color.
    #[must_use]
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Create a fully opaque color.
    #[must_use]
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self::new(r, g, b, 255)
    }

    /// White color.
    #[must_use]
    pub const fn white() -> Self {
        Self::rgb(255, 255, 255)
    }

    /// Black color.
    #[must_use]
    pub const fn black() -> Self {
        Self::rgb(0, 0, 0)
    }

    /// Transparent color.
    #[must_use]
    pub const fn transparent() -> Self {
        Self::new(0, 0, 0, 0)
    }

    /// Yellow color.
    #[must_use]
    pub const fn yellow() -> Self {
        Self::rgb(255, 255, 0)
    }

    /// Red color.
    #[must_use]
    pub const fn red() -> Self {
        Self::rgb(255, 0, 0)
    }

    /// Green color.
    #[must_use]
    pub const fn green() -> Self {
        Self::rgb(0, 255, 0)
    }

    /// Blue color.
    #[must_use]
    pub const fn blue() -> Self {
        Self::rgb(0, 0, 255)
    }
}

// ── TextStyle ─────────────────────────────────────────────────────────────────

/// Text styling options.
#[derive(Clone, Debug)]
pub struct TextStyle {
    /// Font size in points.
    pub font_size: f32,
    /// Foreground color.
    pub foreground: Color,
    /// Background color.
    pub background: Color,
    /// Outline color.
    pub outline: Color,
    /// Outline width in pixels.
    pub outline_width: f32,
    /// Background box padding in pixels.
    pub padding: u32,
    /// Drop shadow offset (x, y) in pixels.
    pub shadow_offset: (i32, i32),
    /// Drop shadow color.
    pub shadow_color: Color,
    /// Enable background box.
    pub draw_background: bool,
    /// Enable outline.
    pub draw_outline: bool,
    /// Enable drop shadow.
    pub draw_shadow: bool,
}

impl Default for TextStyle {
    fn default() -> Self {
        Self {
            font_size: 24.0,
            foreground: Color::white(),
            background: Color::new(0, 0, 0, 192),
            outline: Color::black(),
            outline_width: 1.0,
            padding: 4,
            shadow_offset: (2, 2),
            shadow_color: Color::new(0, 0, 0, 128),
            draw_background: true,
            draw_outline: false,
            draw_shadow: true,
        }
    }
}

// ── MetadataField ─────────────────────────────────────────────────────────────

/// Metadata field type.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MetadataField {
    /// Timecode.
    Timecode,
    /// Frame number.
    FrameNumber,
    /// Filename.
    Filename,
    /// Resolution (e.g., "1920x1080").
    Resolution,
    /// Framerate (e.g., "25.00 fps").
    Framerate,
    /// Codec information.
    Codec,
    /// Current bitrate.
    Bitrate,
    /// Current date.
    Date,
    /// Current time.
    Time,
    /// Custom text field.
    Custom(String),
}

impl MetadataField {
    /// Get the display value for this field.
    #[must_use]
    pub fn value(&self, context: &FrameContext) -> String {
        match self {
            Self::Timecode => context.timecode.clone(),
            Self::FrameNumber => format!("Frame: {}", context.frame_number),
            Self::Filename => context.filename.clone(),
            Self::Resolution => format!("{}x{}", context.width, context.height),
            Self::Framerate => format!("{:.2} fps", context.framerate.to_f64()),
            Self::Codec => context.codec.clone(),
            Self::Bitrate => {
                if context.bitrate > 0 {
                    format!("{:.2} Mbps", context.bitrate as f64 / 1_000_000.0)
                } else {
                    "N/A".to_string()
                }
            }
            Self::Date => format_current_date(),
            Self::Time => format_current_time(),
            Self::Custom(text) => text.clone(),
        }
    }
}

/// Format the current date as YYYY-MM-DD.
#[allow(clippy::cast_possible_truncation)]
fn format_current_date() -> String {
    // Simple date calculation (Gregorian calendar approximation)
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let days_since_epoch = now / 86400;

    let mut year = 1970u64;
    let mut remaining_days = days_since_epoch;

    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if remaining_days < days_in_year {
            break;
        }
        remaining_days -= days_in_year;
        year += 1;
    }

    let mut month = 1u64;
    for m in 1u64..=12 {
        let days_in_month = days_in_month_gregorian(m, year);
        if remaining_days < days_in_month {
            month = m;
            break;
        }
        remaining_days -= days_in_month;
    }

    let day = remaining_days + 1;
    format!("{year:04}-{month:02}-{day:02}")
}

/// Format the current wall-clock time as HH:MM:SS.
fn format_current_time() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let hours = (now / 3600) % 24;
    let minutes = (now / 60) % 60;
    let seconds = now % 60;
    format!("{hours:02}:{minutes:02}:{seconds:02}")
}

/// Check if a year is a leap year.
pub(crate) fn is_leap_year(year: u64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

/// Get the number of days in a month for a given year.
pub(crate) fn days_in_month_gregorian(month: u64, year: u64) -> u64 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap_year(year) {
                29
            } else {
                28
            }
        }
        _ => 30, // Invalid month, return default
    }
}

// ── FrameContext ──────────────────────────────────────────────────────────────

/// Frame context for metadata templating.
#[derive(Clone, Debug)]
pub struct FrameContext {
    /// Current timecode string.
    pub timecode: String,
    /// Frame number.
    pub frame_number: u64,
    /// Filename.
    pub filename: String,
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// Framerate.
    pub framerate: Rational,
    /// Codec name.
    pub codec: String,
    /// Current bitrate in bits per second.
    pub bitrate: u64,
}

impl Default for FrameContext {
    fn default() -> Self {
        Self {
            timecode: "00:00:00:00".to_string(),
            frame_number: 0,
            filename: "unknown.mp4".to_string(),
            width: 1920,
            height: 1080,
            framerate: Rational::new(25, 1),
            codec: "Unknown".to_string(),
            bitrate: 0,
        }
    }
}

// ── OverlayElement ────────────────────────────────────────────────────────────

/// A single overlay element.
#[derive(Clone, Debug)]
pub struct OverlayElement {
    /// Metadata field to display.
    pub field: MetadataField,
    /// Position on the frame.
    pub position: Position,
    /// Text styling.
    pub style: TextStyle,
    /// Element is enabled.
    pub enabled: bool,
}

impl OverlayElement {
    /// Create a new overlay element.
    #[must_use]
    pub fn new(field: MetadataField, position: Position) -> Self {
        Self {
            field,
            position,
            style: TextStyle::default(),
            enabled: true,
        }
    }

    /// Set the text style.
    #[must_use]
    pub fn with_style(mut self, style: TextStyle) -> Self {
        self.style = style;
        self
    }

    /// Enable or disable the element.
    #[must_use]
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }
}

// ── ProgressBar ───────────────────────────────────────────────────────────────

/// Progress bar visualization.
#[derive(Clone, Debug)]
pub struct ProgressBar {
    /// Position on the frame.
    pub position: Position,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Foreground color.
    pub foreground: Color,
    /// Background color.
    pub background: Color,
    /// Total duration in frames.
    pub total_frames: u64,
    /// Show percentage text.
    pub show_percentage: bool,
    /// Enabled.
    pub enabled: bool,
}

impl Default for ProgressBar {
    fn default() -> Self {
        Self {
            position: Position::BottomCenter,
            width: 400,
            height: 8,
            foreground: Color::green(),
            background: Color::new(64, 64, 64, 192),
            total_frames: 1000,
            show_percentage: true,
            enabled: false,
        }
    }
}

// ── TimecodeConfig ────────────────────────────────────────────────────────────

/// Configuration for the timecode filter.
#[derive(Clone, Debug)]
pub struct TimecodeConfig {
    /// Timecode format.
    pub timecode_format: TimecodeFormat,
    /// Overlay elements.
    pub elements: Vec<OverlayElement>,
    /// Safe area margin in pixels.
    pub safe_margin: u32,
    /// Progress bar configuration.
    pub progress_bar: ProgressBar,
    /// Frame context for metadata.
    pub context: FrameContext,
}

impl Default for TimecodeConfig {
    fn default() -> Self {
        Self {
            timecode_format: TimecodeFormat::default(),
            elements: vec![
                OverlayElement::new(MetadataField::Timecode, Position::TopLeft),
                OverlayElement::new(MetadataField::FrameNumber, Position::TopRight),
            ],
            safe_margin: 10,
            progress_bar: ProgressBar::default(),
            context: FrameContext::default(),
        }
    }
}

impl TimecodeConfig {
    /// Create a new timecode configuration.
    #[must_use]
    pub fn new(format: TimecodeFormat) -> Self {
        Self {
            timecode_format: format,
            ..Default::default()
        }
    }

    /// Add an overlay element.
    #[must_use]
    pub fn with_element(mut self, element: OverlayElement) -> Self {
        self.elements.push(element);
        self
    }

    /// Set the safe area margin.
    #[must_use]
    pub fn with_safe_margin(mut self, margin: u32) -> Self {
        self.safe_margin = margin;
        self
    }

    /// Set the progress bar.
    #[must_use]
    pub fn with_progress_bar(mut self, progress_bar: ProgressBar) -> Self {
        self.progress_bar = progress_bar;
        self
    }

    /// Set the frame context.
    #[must_use]
    pub fn with_context(mut self, context: FrameContext) -> Self {
        self.context = context;
        self
    }
}
