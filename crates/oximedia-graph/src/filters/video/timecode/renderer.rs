//! Rendering helper types for the timecode burn-in filter.
//!
//! This module contains the rich overlay types — `TextStyle` is in `config`,
//! while the higher-level composite helpers live here:
//! `MetadataTemplate`, `MultiLineText`, `SafeAreaOverlay`, and `TextAlignment`.

use super::config::{Color, OverlayElement, Position, TextStyle, TimecodeConfig};

// ── TextAlignment ─────────────────────────────────────────────────────────────

/// Text alignment within a box.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextAlignment {
    /// Left-aligned.
    Left,
    /// Center-aligned.
    Center,
    /// Right-aligned.
    Right,
}

// ── MultiLineText ─────────────────────────────────────────────────────────────

/// Multi-line text overlay for complex layouts.
#[derive(Clone, Debug)]
pub struct MultiLineText {
    /// Lines of text.
    pub lines: Vec<String>,
    /// Position on the frame.
    pub position: Position,
    /// Text styling.
    pub style: TextStyle,
    /// Line spacing in pixels.
    pub line_spacing: f32,
    /// Alignment of text within the box.
    pub alignment: TextAlignment,
}

impl Default for MultiLineText {
    fn default() -> Self {
        Self {
            lines: Vec::new(),
            position: Position::TopLeft,
            style: TextStyle::default(),
            line_spacing: 4.0,
            alignment: TextAlignment::Left,
        }
    }
}

impl MultiLineText {
    /// Create a new multi-line text overlay.
    #[must_use]
    pub fn new(lines: Vec<String>, position: Position) -> Self {
        Self {
            lines,
            position,
            ..Default::default()
        }
    }

    /// Set the text style.
    #[must_use]
    pub fn with_style(mut self, style: TextStyle) -> Self {
        self.style = style;
        self
    }

    /// Set line spacing.
    #[must_use]
    pub fn with_line_spacing(mut self, spacing: f32) -> Self {
        self.line_spacing = spacing;
        self
    }

    /// Set text alignment.
    #[must_use]
    pub fn with_alignment(mut self, alignment: TextAlignment) -> Self {
        self.alignment = alignment;
        self
    }
}

// ── SafeAreaOverlay ───────────────────────────────────────────────────────────

/// Safe area visualization (for broadcast).
#[derive(Clone, Debug)]
pub struct SafeAreaOverlay {
    /// Action safe area percentage (typically 90%).
    pub action_safe: f32,
    /// Title safe area percentage (typically 80%).
    pub title_safe: f32,
    /// Line color.
    pub color: Color,
    /// Line width.
    pub line_width: u32,
    /// Enabled.
    pub enabled: bool,
}

impl Default for SafeAreaOverlay {
    fn default() -> Self {
        Self {
            action_safe: 0.9,
            title_safe: 0.8,
            color: Color::new(255, 255, 0, 128),
            line_width: 2,
            enabled: false,
        }
    }
}

// ── MetadataTemplate ──────────────────────────────────────────────────────────

/// Template for complex metadata layouts.
#[derive(Clone, Debug)]
pub struct MetadataTemplate {
    /// Template name.
    pub name: String,
    /// Template elements.
    pub elements: Vec<OverlayElement>,
    /// Safe area margin.
    pub safe_margin: u32,
    /// Enable progress bar.
    pub enable_progress: bool,
}

impl MetadataTemplate {
    /// Create a new template.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            elements: Vec::new(),
            safe_margin: 10,
            enable_progress: false,
        }
    }

    /// Add an element to the template.
    #[must_use]
    pub fn with_element(mut self, element: OverlayElement) -> Self {
        self.elements.push(element);
        self
    }

    /// Set safe margin.
    #[must_use]
    pub fn with_safe_margin(mut self, margin: u32) -> Self {
        self.safe_margin = margin;
        self
    }

    /// Enable progress bar.
    #[must_use]
    pub fn with_progress(mut self, enabled: bool) -> Self {
        self.enable_progress = enabled;
        self
    }

    /// Apply this template to a config.
    #[must_use]
    pub fn apply_to_config(&self, mut config: TimecodeConfig) -> TimecodeConfig {
        config.elements = self.elements.clone();
        config.safe_margin = self.safe_margin;
        config.progress_bar.enabled = self.enable_progress;
        config
    }
}
