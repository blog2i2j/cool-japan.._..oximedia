//! Timecode and metadata burn-in filter.
//!
//! This filter overlays timecode and metadata information onto video frames
//! with support for multiple formats, positions, and styling options.
//!
//! # Overview
//!
//! The timecode filter provides professional-grade burn-in capabilities for video
//! post-production, broadcast, and quality control workflows. It supports:
//!
//! - Multiple SMPTE timecode formats (23.976, 24, 25, 29.97 DF/NDF, 30, 60)
//! - Alternative time representations (frame count, milliseconds, HH:MM:SS)
//! - Flexible positioning with 9 preset positions plus custom coordinates
//! - Rich text styling (fonts, colors, backgrounds, outlines, shadows)
//! - Multiple simultaneous overlays with independent styling
//! - Progress bar visualization
//! - Safe area margins for broadcast compliance
//! - Template system for reusable configurations
//!
//! # Basic Usage
//!
//! ```ignore
//! use oximedia_graph::filters::video::{TimecodeFilter, TimecodeFormat, presets};
//! use oximedia_graph::node::NodeId;
//!
//! // Load a font (e.g., from system fonts)
//! let font_data = std::fs::read("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")?;
//!
//! // Create a simple timecode overlay using a preset
//! let config = presets::simple_timecode(TimecodeFormat::Smpte25);
//! let filter = TimecodeFilter::new(NodeId(0), "timecode", config, font_data)?;
//! ```
//!
//! # Advanced Usage
//!
//! ```ignore
//! use oximedia_graph::filters::video::{
//!     TimecodeFilter, TimecodeConfig, TimecodeFormat,
//!     MetadataField, OverlayElement, Position, TextStyle, Color,
//! };
//! use oximedia_graph::node::NodeId;
//!
//! // Create a custom configuration
//! let mut config = TimecodeConfig::new(TimecodeFormat::Smpte2997Df);
//!
//! // Customize text style
//! let mut style = TextStyle::default();
//! style.font_size = 32.0;
//! style.foreground = Color::yellow();
//! style.background = Color::new(0, 0, 0, 200);
//! style.draw_outline = true;
//! style.outline_width = 2.0;
//!
//! // Add multiple elements
//! config = config
//!     .with_element(
//!         OverlayElement::new(MetadataField::Timecode, Position::TopLeft)
//!             .with_style(style.clone())
//!     )
//!     .with_element(
//!         OverlayElement::new(MetadataField::FrameNumber, Position::TopRight)
//!             .with_style(style)
//!     );
//!
//! let font_data = std::fs::read("path/to/font.ttf")?;
//! let filter = TimecodeFilter::new(NodeId(0), "timecode", config, font_data)?;
//! ```
//!
//! # Presets
//!
//! The [`presets`] module provides ready-to-use configurations:
//!
//! - [`presets::simple_timecode`] - Basic timecode in top-left corner
//! - [`presets::full_metadata`] - Four-corner metadata display
//! - [`presets::broadcast_timecode`] - Broadcast-standard timecode
//! - [`presets::production_overlay`] - Comprehensive production metadata
//! - [`presets::qc_overlay`] - Quality control review overlay
//! - [`presets::streaming_overlay`] - Live streaming information
//! - [`presets::minimal_corner`] - Small, unobtrusive corner timecode
//!
//! # Templates
//!
//! The [`templates`] module provides reusable layout templates:
//!
//! - [`templates::four_corner_metadata`] - Four-corner layout
//! - [`templates::center_focused`] - Center-focused display
//! - [`templates::top_bar`] - Top bar with multiple fields

#![forbid(unsafe_code)]

pub mod config;
pub mod presets;
pub mod processor;
pub mod renderer;
pub mod templates;

// ── Public re-exports — preserve the exact API surface of the old timecode.rs ─

pub use config::{
    Color, FrameContext, MetadataField, OverlayElement, Position, ProgressBar, TextStyle,
    TimecodeConfig, TimecodeFormat,
};

pub use renderer::{MetadataTemplate, MultiLineText, SafeAreaOverlay, TextAlignment};

pub use processor::TimecodeFilter;
