//! MPEG-DASH manifest emitter.
//!
//! This module provides utilities for generating MPEG-DASH MPD (Media
//! Presentation Description) manifests.
//!
//! # Supported profile
//!
//! - DASH-IF IOP on-demand: `urn:mpeg:dash:profile:isoff-on-demand:2011`
//!
//! # Example
//!
//! ```
//! use oximedia_container::dash::{
//!     DashManifestConfig, DashAdaptationSet, DashRepresentation,
//!     DashSegmentTemplate, emit_mpd,
//! };
//!
//! let config = DashManifestConfig {
//!     media_presentation_duration: "PT60S".to_string(),
//!     min_buffer_time: "PT2S".to_string(),
//!     base_url: None,
//!     adaptation_sets: vec![
//!         DashAdaptationSet {
//!             id: 1,
//!             content_type: "video".to_string(),
//!             mime_type: "video/mp4".to_string(),
//!             codecs: "av01.0.04M.08".to_string(),
//!             representations: vec![
//!                 DashRepresentation {
//!                     id: "1080p".to_string(),
//!                     bandwidth: 4_000_000,
//!                     width: Some(1920),
//!                     height: Some(1080),
//!                     frame_rate: Some("25".to_string()),
//!                     audio_sampling_rate: None,
//!                     segment_template: DashSegmentTemplate {
//!                         timescale: 90000,
//!                         duration: Some(270000),
//!                         initialization: "1080p/init.mp4".to_string(),
//!                         media: "1080p/seg$Number$.m4s".to_string(),
//!                         start_number: 1,
//!                         segment_timeline: None,
//!                     },
//!                 },
//!             ],
//!         },
//!     ],
//! };
//!
//! let xml = emit_mpd(&config);
//! assert!(xml.contains("SegmentTemplate"));
//! ```

pub mod manifest;

pub use manifest::{
    emit_mpd, DashAdaptationSet, DashManifestConfig, DashRepresentation, DashSegmentTemplate,
    DashSegmentTimeline, DashSegmentTimelineEntry,
};
