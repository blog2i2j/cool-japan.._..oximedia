//! Broadcast Graphics Engine for `OxiMedia`
//!
//! This crate provides a comprehensive broadcast graphics engine with support for:
//! - 2D vector graphics rendering
//! - Advanced text layout and typography
//! - Broadcast graphics elements (lower thirds, tickers, bugs, scoreboards)
//! - Keyframe animation system
//! - Template-based graphics
//! - Real-time video overlay
//! - GPU acceleration
//! - Professional broadcast features

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::unused_self,
    clippy::float_cmp,
    clippy::match_same_arms,
    clippy::field_reassign_with_default,
    clippy::unnecessary_wraps,
    clippy::must_use_candidate,
    clippy::too_many_arguments,
    clippy::many_single_char_names,
    clippy::struct_field_names,
    clippy::trivially_copy_pass_by_ref,
    clippy::unreadable_literal,
    clippy::similar_names,
    clippy::needless_borrow,
    clippy::map_unwrap_or,
    clippy::return_self_not_must_use,
    dead_code,
    missing_docs
)]

pub mod animation;
pub mod animation_curve;
pub mod bitmap_font;
pub mod clock_widget;
pub mod color;
pub mod countdown_timer;
pub mod effects;
pub mod elements;
pub mod error;
pub mod examples;
pub mod graphic_template;
pub mod keyframe;
pub mod lower_third;
pub mod overlay;
pub mod particles;
pub mod presets;
pub mod primitives;
pub mod professional;
pub mod render;
pub mod scoreboard;
pub mod template;
pub mod text;
pub mod text_renderer;
pub mod ticker;
pub mod transitions;
pub mod virtual_set;
pub mod weather_widget;

// Wave 10 modules
pub mod color_picker;
pub mod layout_engine;
pub mod svg_renderer;

// Wave 11 modules
pub mod font_metrics;
pub mod sprite_sheet;
pub mod transition_wipe;

// Wave 12 modules
pub mod gradient_fill;
pub mod shape_render;
pub mod text_layout;
pub mod text_layout_ext;

// Wave 15 modules
pub mod color_blend;
pub mod mask_layer;
pub mod path_builder;

// Wave 16 modules
pub mod color_grade;
pub mod hdr_composite;
pub mod lut_apply;

// Wave 17 modules
pub mod chroma_key;
pub mod data_visualization;
pub mod picture_in_picture;

// Wave 18 modules — broadcast overlay components
pub mod audio_waveform;
pub mod crawl;
pub mod logo_bug;
pub mod stinger_transition;
pub mod template_variables;

// Text-on-path rendering (fontdue integration)
pub mod bezier_path;
pub mod text_on_path;

#[cfg(all(feature = "server", not(target_arch = "wasm32")))]
pub mod control;

#[cfg(feature = "gpu")]
pub mod gpu;

pub use error::{GraphicsError, Result};

/// Graphics engine version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default target framerate for animations
pub const DEFAULT_FPS: f32 = 60.0;

/// Maximum supported resolution width
pub const MAX_WIDTH: u32 = 7680; // 8K

/// Maximum supported resolution height
pub const MAX_HEIGHT: u32 = 4320; // 8K

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_constants() {
        assert_eq!(DEFAULT_FPS, 60.0);
        assert_eq!(MAX_WIDTH, 7680);
        assert_eq!(MAX_HEIGHT, 4320);
    }
}
