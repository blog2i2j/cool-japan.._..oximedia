//! Preset timecode configurations for common use cases.

use super::config::{
    Color, FrameContext, MetadataField, OverlayElement, Position, ProgressBar, TextStyle,
    TimecodeConfig, TimecodeFormat,
};

/// Simple timecode in top-left corner.
#[must_use]
pub fn simple_timecode(format: TimecodeFormat) -> TimecodeConfig {
    TimecodeConfig {
        timecode_format: format,
        elements: vec![OverlayElement::new(
            MetadataField::Timecode,
            Position::TopLeft,
        )],
        safe_margin: 10,
        progress_bar: ProgressBar::default(),
        context: FrameContext::default(),
    }
}

/// Full metadata overlay.
#[must_use]
pub fn full_metadata(format: TimecodeFormat) -> TimecodeConfig {
    TimecodeConfig {
        timecode_format: format,
        elements: vec![
            OverlayElement::new(MetadataField::Timecode, Position::TopLeft),
            OverlayElement::new(MetadataField::FrameNumber, Position::TopRight),
            OverlayElement::new(MetadataField::Resolution, Position::BottomLeft),
            OverlayElement::new(MetadataField::Framerate, Position::BottomRight),
        ],
        safe_margin: 10,
        progress_bar: ProgressBar::default(),
        context: FrameContext::default(),
    }
}

/// Broadcast-style timecode.
#[must_use]
pub fn broadcast_timecode(format: TimecodeFormat) -> TimecodeConfig {
    let style = TextStyle {
        font_size: 32.0,
        foreground: Color::yellow(),
        background: Color::new(0, 0, 0, 224),
        padding: 8,
        ..TextStyle::default()
    };

    TimecodeConfig {
        timecode_format: format,
        elements: vec![
            OverlayElement::new(MetadataField::Timecode, Position::TopCenter).with_style(style),
        ],
        safe_margin: 20,
        progress_bar: ProgressBar::default(),
        context: FrameContext::default(),
    }
}

/// Production overlay with comprehensive metadata.
#[must_use]
pub fn production_overlay(format: TimecodeFormat) -> TimecodeConfig {
    let tc_style = TextStyle {
        font_size: 28.0,
        foreground: Color::yellow(),
        background: Color::new(0, 0, 0, 200),
        padding: 6,
        ..TextStyle::default()
    };

    let meta_style = TextStyle {
        font_size: 18.0,
        foreground: Color::white(),
        background: Color::new(0, 0, 0, 180),
        padding: 4,
        ..TextStyle::default()
    };

    TimecodeConfig {
        timecode_format: format,
        elements: vec![
            OverlayElement::new(MetadataField::Timecode, Position::TopLeft)
                .with_style(tc_style.clone()),
            OverlayElement::new(MetadataField::FrameNumber, Position::TopRight)
                .with_style(meta_style.clone()),
            OverlayElement::new(MetadataField::Resolution, Position::BottomLeft)
                .with_style(meta_style.clone()),
            OverlayElement::new(MetadataField::Framerate, Position::BottomRight)
                .with_style(meta_style.clone()),
            OverlayElement::new(MetadataField::Codec, Position::Custom(10, 40))
                .with_style(meta_style.clone()),
            OverlayElement::new(MetadataField::Bitrate, Position::Custom(10, 65))
                .with_style(meta_style),
        ],
        safe_margin: 10,
        progress_bar: ProgressBar {
            enabled: true,
            ..ProgressBar::default()
        },
        context: FrameContext::default(),
    }
}

/// Minimal corner timecode (small and unobtrusive).
#[must_use]
pub fn minimal_corner(format: TimecodeFormat) -> TimecodeConfig {
    let style = TextStyle {
        font_size: 14.0,
        foreground: Color::new(255, 255, 255, 200),
        background: Color::new(0, 0, 0, 100),
        padding: 2,
        draw_shadow: false,
        ..TextStyle::default()
    };

    TimecodeConfig {
        timecode_format: format,
        elements: vec![
            OverlayElement::new(MetadataField::Timecode, Position::BottomRight).with_style(style),
        ],
        safe_margin: 5,
        progress_bar: ProgressBar::default(),
        context: FrameContext::default(),
    }
}

/// QC (Quality Control) overlay for review.
#[must_use]
pub fn qc_overlay(format: TimecodeFormat) -> TimecodeConfig {
    let tc_style = TextStyle {
        font_size: 36.0,
        foreground: Color::new(0, 255, 0, 255),
        background: Color::new(0, 0, 0, 220),
        padding: 10,
        draw_outline: true,
        outline_width: 2.0,
        ..TextStyle::default()
    };

    let meta_style = TextStyle {
        font_size: 20.0,
        foreground: Color::white(),
        background: Color::new(0, 0, 0, 200),
        padding: 5,
        ..TextStyle::default()
    };

    TimecodeConfig {
        timecode_format: format,
        elements: vec![
            OverlayElement::new(MetadataField::Timecode, Position::TopCenter).with_style(tc_style),
            OverlayElement::new(MetadataField::FrameNumber, Position::TopLeft)
                .with_style(meta_style.clone()),
            OverlayElement::new(MetadataField::Filename, Position::TopRight)
                .with_style(meta_style.clone()),
            OverlayElement::new(MetadataField::Resolution, Position::BottomLeft)
                .with_style(meta_style.clone()),
            OverlayElement::new(MetadataField::Date, Position::BottomRight).with_style(meta_style),
        ],
        safe_margin: 15,
        progress_bar: ProgressBar {
            enabled: true,
            width: 600,
            height: 10,
            show_percentage: true,
            ..ProgressBar::default()
        },
        context: FrameContext::default(),
    }
}

/// Streaming overlay optimized for live streams.
#[must_use]
pub fn streaming_overlay(format: TimecodeFormat) -> TimecodeConfig {
    let time_style = TextStyle {
        font_size: 24.0,
        foreground: Color::new(255, 100, 100, 255),
        background: Color::new(0, 0, 0, 200),
        padding: 6,
        draw_shadow: true,
        ..TextStyle::default()
    };

    let info_style = TextStyle {
        font_size: 16.0,
        foreground: Color::new(200, 200, 255, 255),
        background: Color::new(0, 0, 0, 180),
        padding: 4,
        ..TextStyle::default()
    };

    TimecodeConfig {
        timecode_format: format,
        elements: vec![
            OverlayElement::new(MetadataField::Time, Position::TopRight).with_style(time_style),
            OverlayElement::new(MetadataField::Bitrate, Position::BottomRight)
                .with_style(info_style.clone()),
            OverlayElement::new(MetadataField::Framerate, Position::BottomLeft)
                .with_style(info_style),
        ],
        safe_margin: 10,
        progress_bar: ProgressBar::default(),
        context: FrameContext::default(),
    }
}
