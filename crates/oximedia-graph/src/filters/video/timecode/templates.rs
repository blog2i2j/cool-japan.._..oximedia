//! Template presets for common metadata layout use cases.

use super::config::{Color, MetadataField, OverlayElement, Position, TextStyle};
use super::renderer::MetadataTemplate;

/// Create a basic four-corner metadata template.
#[must_use]
pub fn four_corner_metadata() -> MetadataTemplate {
    let meta_style = TextStyle::default();

    MetadataTemplate::new("FourCorner")
        .with_element(
            OverlayElement::new(MetadataField::Timecode, Position::TopLeft)
                .with_style(meta_style.clone()),
        )
        .with_element(
            OverlayElement::new(MetadataField::FrameNumber, Position::TopRight)
                .with_style(meta_style.clone()),
        )
        .with_element(
            OverlayElement::new(MetadataField::Resolution, Position::BottomLeft)
                .with_style(meta_style.clone()),
        )
        .with_element(
            OverlayElement::new(MetadataField::Framerate, Position::BottomRight)
                .with_style(meta_style),
        )
        .with_safe_margin(10)
}

/// Create a center-focused template.
#[must_use]
pub fn center_focused() -> MetadataTemplate {
    let large_style = TextStyle {
        font_size: 48.0,
        foreground: Color::white(),
        background: Color::new(0, 0, 0, 200),
        padding: 12,
        ..TextStyle::default()
    };

    MetadataTemplate::new("CenterFocused")
        .with_element(
            OverlayElement::new(MetadataField::Timecode, Position::Center).with_style(large_style),
        )
        .with_safe_margin(20)
}

/// Create a top bar template with multiple fields.
#[must_use]
pub fn top_bar() -> MetadataTemplate {
    let bar_style = TextStyle {
        font_size: 20.0,
        foreground: Color::white(),
        background: Color::new(0, 0, 0, 220),
        padding: 6,
        ..TextStyle::default()
    };

    MetadataTemplate::new("TopBar")
        .with_element(
            OverlayElement::new(MetadataField::Timecode, Position::TopLeft)
                .with_style(bar_style.clone()),
        )
        .with_element(
            OverlayElement::new(MetadataField::FrameNumber, Position::TopCenter)
                .with_style(bar_style.clone()),
        )
        .with_element(
            OverlayElement::new(MetadataField::Resolution, Position::TopRight)
                .with_style(bar_style),
        )
        .with_safe_margin(5)
}
