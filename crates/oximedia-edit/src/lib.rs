//! Video timeline editor for `OxiMedia`.
//!
//! `oximedia-edit` provides a comprehensive video editing system with:
//!
//! - **Multi-track timeline**: Manage video, audio, and subtitle tracks
//! - **Clip operations**: Add, remove, move, trim, split clips
//! - **Advanced editing**: Ripple, roll, slip, and slide edits
//! - **Effects system**: Apply effects with keyframe animation
//! - **Transitions**: Cross-fades, dissolves, wipes, and more
//! - **Rendering**: Real-time preview and high-quality export
//!
//! # Example
//!
//! ```
//! use oximedia_edit::{Timeline, TimelineEditor, Clip, ClipType};
//! use oximedia_core::Rational;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a timeline
//! let mut timeline = Timeline::new(
//!     Rational::new(1, 1000),  // 1ms timebase
//!     Rational::new(30, 1),     // 30 fps
//! );
//!
//! // Add video track
//! let video_track = timeline.add_track(oximedia_edit::TrackType::Video);
//!
//! // Create and add a clip
//! let clip = Clip::new(1, ClipType::Video, 0, 5000); // 5 seconds
//! timeline.add_clip(video_track, clip)?;
//!
//! // Edit operations
//! let mut editor = TimelineEditor::new();
//! timeline.set_playhead(2500); // Seek to 2.5 seconds
//! editor.split_at_playhead(&mut timeline)?;
//! # Ok(())
//! # }
//! ```
//!
//! # Architecture
//!
//! The editing system is built around these core components:
//!
//! ## Timeline
//!
//! The [`Timeline`] is the central structure containing multiple [`Track`]s.
//! Each track holds [`Clip`]s that represent media segments.
//!
//! ## Clips
//!
//! [`Clip`]s are segments of media with timing information:
//! - Timeline position and duration
//! - Source media and in/out points
//! - Speed and direction (normal/reverse)
//! - Effects and opacity
//!
//! ## Effects
//!
//! The [`effect`] module provides an effects system with:
//! - Common effects (brightness, blur, color correction)
//! - Keyframe animation with interpolation
//! - Effect stacks per clip
//!
//! ## Transitions
//!
//! [`Transition`]s provide smooth blending between adjacent clips:
//! - Video transitions (dissolve, wipe, zoom)
//! - Audio cross-fades
//! - Easing functions
//!
//! ## Editing Operations
//!
//! The [`TimelineEditor`] provides editing operations:
//! - Cut, copy, paste
//! - Split and trim
//! - Ripple, roll, slip, slide edits
//! - Speed changes and reverse
//!
//! ## Rendering
//!
//! The [`render`] module handles timeline rendering:
//! - [`TimelineRenderer`]: Render individual frames
//! - [`PreviewRenderer`]: Real-time playback preview
//! - [`ExportRenderer`]: High-quality final export
//! - [`BackgroundRenderer`]: Non-blocking background rendering
//!
//! # Green List Only
//!
//! Like all `OxiMedia` components, `oximedia-edit` only supports patent-free
//! codecs (AV1, VP9, VP8, Opus, Vorbis, FLAC). Attempting to use patent-
//! encumbered codecs will result in errors.

#![warn(missing_docs)]

pub mod auto_edit;
pub mod blade_tool;
pub mod clip;
pub mod clip_arrange;
pub mod clip_speed;
pub mod collab_edit;
pub mod color_grade_edit;
pub mod color_label;
pub mod edit;
pub mod edit_context;
pub mod edit_macro;
pub mod edit_preset;
pub mod edl;
pub mod edl_import;
pub mod effect;
pub mod error;
pub mod export;
pub mod frame_prefetch;
pub mod fx_strip;
pub mod group;
pub mod group_edit;
pub mod history;
pub mod history_tree;
pub mod incremental_render;
pub mod insert_mode;
pub mod interval_tree;
pub mod magnetic_snap;
pub mod marker;
pub mod marker_edit;
pub mod multi_export;
pub mod multicam;
pub mod multitrack;
pub mod nested_sequence;
pub mod parallel_render;
pub mod picture_in_picture;
pub mod pip;
pub mod proxy;
pub mod render;
pub mod render_queue;
pub mod ripple;
pub mod selection;
pub mod slip_slide;
pub mod smart_trim;
pub mod timeline;
pub mod timeline_export;
pub mod timeline_validator;
pub mod title_overlay;
pub mod track_lock;
pub mod transition;
pub mod trim_mode;
pub mod trim_selection;
pub mod waveform;

// Re-export commonly used items
pub use clip::{Clip, ClipId, ClipRef, ClipSelection, ClipType, Clipboard};
pub use edit::{EditMode, TimelineEditor};
pub use effect::{
    Effect, EffectInstance, EffectPreset, EffectStack, EffectType, InterpolationMode, Parameter,
    ParameterValue,
};
pub use error::{EditError, EditResult};
pub use group::{
    ClipGroup, ClipLink, CompoundClip, CompoundClipManager, GroupManager, LinkManager, LinkType,
};
pub use marker::{InOutPoints, Marker, MarkerId, MarkerManager, MarkerType, Region, RegionManager};
#[cfg(not(target_arch = "wasm32"))]
pub use render::BackgroundRenderer;
pub use render::{
    ExportRenderer, ExportSettings, PreviewRenderer, RawFrameCache, RenderConfig, RenderFrame,
    RenderQuality, TimelineRenderer, RAW_FRAME_CACHE_CAPACITY,
};
pub use timeline::{PlaybackState, Timeline, TimelineConfig, Track, TrackType};
pub use transition::{
    EasingFunction, Transition, TransitionBuilder, TransitionManager, TransitionParameters,
    TransitionPresets, TransitionType,
};

/// Version information.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Prelude module for convenient imports.
pub mod prelude {

    pub use crate::clip::{Clip, ClipId, ClipType};
    pub use crate::edit::{EditMode, TimelineEditor};
    pub use crate::effect::{Effect, EffectStack, EffectType};
    pub use crate::error::{EditError, EditResult};
    pub use crate::group::{GroupManager, LinkManager};
    pub use crate::marker::{Marker, MarkerManager, Region, RegionManager};
    pub use crate::render::{PreviewRenderer, RenderConfig, TimelineRenderer};
    pub use crate::timeline::{Timeline, Track, TrackType};
    pub use crate::transition::{Transition, TransitionType};
}
