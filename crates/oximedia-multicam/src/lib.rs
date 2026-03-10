//! Multi-camera synchronization and switching for `OxiMedia`.
//!
//! This crate provides comprehensive multi-camera production capabilities:
//!
//! # Synchronization
//!
//! The [`sync`] module provides temporal synchronization:
//!
//! - **Temporal Sync** - Frame-accurate synchronization across cameras
//! - **Audio Sync** - Cross-correlation based audio synchronization
//! - **Timecode Sync** - LTC/VITC/SMPTE timecode-based sync
//! - **Visual Sync** - Flash/clapper-based synchronization
//! - **Genlock Simulation** - Virtual genlock for post-production
//! - **Drift Correction** - Detect and correct sync drift over time
//!
//! # Multi-angle Editing
//!
//! The [`edit`] module provides multi-angle editing capabilities:
//!
//! - **Timeline** - Multi-angle timeline with angle switching
//! - **Switching** - Automatic and manual camera angle switching
//! - **Transitions** - Smooth transitions between camera angles
//!
//! # Automatic Switching
//!
//! The [`auto`] module provides AI-based camera selection:
//!
//! - **Camera Selection** - Intelligent angle selection
//! - **Rules Engine** - Speaker detection, action following
//! - **Scoring** - Score camera angles based on multiple criteria
//!
//! # Manual Control
//!
//! The [`manual`] module provides manual switching control:
//!
//! - **Control** - Manual switching interface
//! - **Preview** - Preview all camera angles simultaneously
//!
//! # Composition
//!
//! The [`composite`] module provides multi-view layouts:
//!
//! - **Picture-in-Picture** - PIP composition with corner insets
//! - **Split-screen** - Side-by-side, quad-split layouts
//! - **Grid** - 2x2, 3x3, 4x4 grid layouts
//!
//! # Color Matching
//!
//! The [`color`] module matches colors across cameras:
//!
//! - **Color Match** - Match color appearance across angles
//! - **White Balance** - Normalize white balance across cameras
//!
//! # Spatial Alignment
//!
//! The [`spatial`] module provides spatial alignment:
//!
//! - **Alignment** - Align overlapping camera views
//! - **Stitching** - Stitch overlapping views into panoramas
//!
//! # Metadata
//!
//! The [`metadata`] module tracks per-angle metadata:
//!
//! - **Track** - Per-angle metadata and properties
//! - **Markers** - Sync markers and cue points
//!
//! # Example: Multi-camera Timeline
//!
//! ```
//! use oximedia_multicam::edit::MultiCamTimeline;
//! use oximedia_multicam::sync::SyncMethod;
//! use oximedia_multicam::composite::Layout;
//!
//! # fn example() -> oximedia_multicam::Result<()> {
//! // Create a multi-camera timeline with 3 angles
//! let mut timeline = MultiCamTimeline::new(3);
//!
//! // Add camera angles
//! // timeline.add_angle(0, video_track_0, audio_track_0)?;
//! // timeline.add_angle(1, video_track_1, audio_track_1)?;
//! // timeline.add_angle(2, video_track_2, audio_track_2)?;
//!
//! // Synchronize using audio cross-correlation
//! // timeline.synchronize(SyncMethod::Audio)?;
//!
//! // Set composition layout
//! // timeline.set_layout(Layout::PictureInPicture { main: 0, inset: 1 })?;
//!
//! // Switch to different angle at specific time
//! // timeline.add_switch(time, 2)?;
//! # Ok(())
//! # }
//! ```
//!
//! # Example: Automatic Switching
//!
//! ```
//! use oximedia_multicam::auto::{AutoSwitcher, SwitchingRule};
//! use oximedia_multicam::auto::SelectionCriteria;
//!
//! # fn example() -> oximedia_multicam::Result<()> {
//! // Create auto-switcher with rules
//! let mut switcher = AutoSwitcher::new();
//!
//! // Add switching rules
//! switcher.add_rule(SwitchingRule::SpeakerDetection { sensitivity: 0.8 });
//! switcher.add_rule(SwitchingRule::ActionFollowing { smoothness: 0.7 });
//! switcher.add_rule(SwitchingRule::ShotVariety { min_duration_ms: 2000 });
//!
//! // Configure selection criteria
//! let criteria = SelectionCriteria {
//!     face_detection: true,
//!     composition_quality: true,
//!     audio_activity: true,
//!     motion_detection: true,
//!     speaker_detection: true,
//!     min_confidence: 0.7,
//! };
//!
//! // Analyze frames and get recommended angle
//! // let angle = switcher.select_angle(&frames, &criteria)?;
//! # Ok(())
//! # }
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod angle;
pub mod angle_group;
pub mod angle_priority;
pub mod angle_score;
pub mod angle_sync;
pub mod angle_sync_ext;
pub mod auto;
pub mod bank_ctrl;
pub mod bank_system;
pub mod cam_label;
pub mod cam_metadata;
pub mod clip_split;
pub mod color;
pub mod composite;
pub mod coverage_map;
pub mod cut_analysis;
pub mod cut_point;
pub mod edit;
pub mod edit_decision;
pub mod error;
pub mod genlock_master;
pub mod iso_file_sync;
pub mod iso_record;
pub mod iso_recording;
pub mod iso_sync;
pub mod manual;
pub mod metadata;
pub mod multicam_export;
pub mod replay_buffer;
pub mod spatial;
pub mod switch_list;
pub mod switcher;
pub mod sync;
pub mod sync_report;
pub mod sync_verify;
pub mod tally_system;
pub mod timecode_sync;

// Re-exports
pub use error::{MultiCamError, Result};

/// Camera angle identifier
pub type AngleId = usize;

/// Frame number in timeline
pub type FrameNumber = u64;

/// Camera information
#[derive(Debug, Clone)]
pub struct CameraInfo {
    /// Camera identifier
    pub id: AngleId,
    /// Camera name/label
    pub name: String,
    /// Camera position in space
    pub position: Option<CameraPosition>,
    /// Camera sensor information
    pub sensor: Option<SensorInfo>,
    /// Lens information
    pub lens: Option<LensInfo>,
}

/// Camera position in 3D space
#[derive(Debug, Clone, Copy)]
pub struct CameraPosition {
    /// X coordinate (meters)
    pub x: f64,
    /// Y coordinate (meters)
    pub y: f64,
    /// Z coordinate (meters)
    pub z: f64,
    /// Pan angle (degrees)
    pub pan: f64,
    /// Tilt angle (degrees)
    pub tilt: f64,
    /// Roll angle (degrees)
    pub roll: f64,
}

/// Camera sensor information
#[derive(Debug, Clone)]
pub struct SensorInfo {
    /// Sensor width (mm)
    pub width_mm: f64,
    /// Sensor height (mm)
    pub height_mm: f64,
    /// Sensor type (e.g., "Super 35", "Full Frame")
    pub sensor_type: String,
}

/// Lens information
#[derive(Debug, Clone)]
pub struct LensInfo {
    /// Focal length (mm)
    pub focal_length: f64,
    /// Maximum aperture (f-stop)
    pub max_aperture: f64,
    /// Lens model
    pub model: String,
}

/// Synchronization status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncStatus {
    /// Not synchronized
    NotSynced,
    /// Synchronization in progress
    Syncing,
    /// Synchronized
    Synced,
    /// Sync lost/drifted
    Drifted,
}

/// Multi-camera session configuration
#[derive(Debug, Clone)]
pub struct MultiCamConfig {
    /// Number of camera angles
    pub angle_count: usize,
    /// Frame rate (fps)
    pub frame_rate: f64,
    /// Target frame rate for output
    pub output_frame_rate: f64,
    /// Enable audio synchronization
    pub enable_audio_sync: bool,
    /// Enable timecode synchronization
    pub enable_timecode_sync: bool,
    /// Enable visual marker synchronization
    pub enable_visual_sync: bool,
    /// Sync drift tolerance (frames)
    pub drift_tolerance: u32,
    /// Automatic color matching
    pub auto_color_match: bool,
}

impl Default for MultiCamConfig {
    fn default() -> Self {
        Self {
            angle_count: 2,
            frame_rate: 25.0,
            output_frame_rate: 25.0,
            enable_audio_sync: true,
            enable_timecode_sync: false,
            enable_visual_sync: false,
            drift_tolerance: 2,
            auto_color_match: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = MultiCamConfig::default();
        assert_eq!(config.angle_count, 2);
        assert_eq!(config.frame_rate, 25.0);
        assert!(config.enable_audio_sync);
        assert!(config.auto_color_match);
    }

    #[test]
    fn test_camera_position() {
        let pos = CameraPosition {
            x: 0.0,
            y: 1.5,
            z: -5.0,
            pan: 0.0,
            tilt: -10.0,
            roll: 0.0,
        };
        assert_eq!(pos.y, 1.5);
        assert_eq!(pos.tilt, -10.0);
    }
}
