//! Virtual Production and LED Wall Tools for `OxiMedia`
//!
//! This crate provides comprehensive virtual production capabilities including:
//! - Camera tracking and calibration
//! - LED wall rendering with perspective correction
//! - In-camera VFX compositing
//! - Color pipeline management
//! - Genlock synchronization
//! - Motion capture integration
//! - Real-time keying and green screen alternatives
//! - Unreal Engine integration
//! - Multi-camera support
//!
//! # Examples
//!
//! ```
//! use oximedia_virtual::{VirtualProduction, VirtualProductionConfig, WorkflowType};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = VirtualProductionConfig::default()
//!     .with_workflow(WorkflowType::LedWall)
//!     .with_target_fps(60.0)
//!     .with_sync_accuracy_ms(0.5);
//!
//! let mut vp = VirtualProduction::new(config)?;
//! # Ok(())
//! # }
//! ```

#![forbid(unsafe_code)]

pub mod math;

pub mod background_plate;
pub mod camera_frustum;
pub mod camera_tracking;
pub mod color;
pub mod constants;
pub mod examples;
pub mod frustum;
pub mod genlock;
pub mod greenscreen;
pub mod icvfx;
pub mod keying;
pub mod led;
pub mod led_volume;
pub mod led_wall;
pub mod lens;
pub mod metrics;
pub mod mocap;
pub mod motion_path;
pub mod multicam;
pub mod ndi_bridge;
pub mod panel_topology;
pub mod pixel_mapping;
pub mod preview;
pub mod projection_map;
pub mod render_layer;
pub mod render_output;
pub mod scene;
pub mod scene_setup;
pub mod stage;
pub mod stage_layout;
pub mod stage_manager;
pub mod sync;
pub mod talent_keying;
pub mod tracking;
pub mod tracking_data;
pub mod tracking_session;
pub mod unreal;
pub mod utils;
pub mod virtual_set;
pub mod virtual_studio;
pub mod volume_calibration;
pub mod workflows;

use std::time::Duration;
use thiserror::Error;

/// Virtual production errors
#[derive(Debug, Error)]
pub enum VirtualProductionError {
    #[error("Camera tracking error: {0}")]
    CameraTracking(String),

    #[error("LED wall error: {0}")]
    LedWall(String),

    #[error("Calibration error: {0}")]
    Calibration(String),

    #[error("Synchronization error: {0}")]
    Sync(String),

    #[error("Color pipeline error: {0}")]
    Color(String),

    #[error("Motion capture error: {0}")]
    MotionCapture(String),

    #[error("Compositing error: {0}")]
    Compositing(String),

    #[error("Unreal integration error: {0}")]
    UnrealIntegration(String),

    #[error("Multi-camera error: {0}")]
    MultiCamera(String),

    #[error("Frame timing error: expected {expected:?}, got {actual:?}")]
    FrameTiming {
        expected: Duration,
        actual: Duration,
    },

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

pub type Result<T> = std::result::Result<T, VirtualProductionError>;

/// Workflow types for virtual production
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum WorkflowType {
    /// Full LED volume with camera tracking
    LedWall,
    /// Mix LED wall and green screen
    Hybrid,
    /// Traditional green screen with real-time compositing
    GreenScreen,
    /// Augmented reality overlay
    AugmentedReality,
}

/// Quality mode for real-time processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum QualityMode {
    /// Draft quality for setup and rehearsal
    Draft,
    /// Preview quality for monitoring
    Preview,
    /// Final quality for recording
    Final,
}

/// Virtual production configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VirtualProductionConfig {
    /// Workflow type
    pub workflow: WorkflowType,
    /// Target frames per second (minimum 60fps recommended)
    pub target_fps: f64,
    /// Synchronization accuracy in milliseconds (target <1ms)
    pub sync_accuracy_ms: f64,
    /// Quality mode
    pub quality: QualityMode,
    /// Enable color calibration
    pub color_calibration: bool,
    /// Enable lens distortion correction
    pub lens_correction: bool,
    /// Number of cameras to track
    pub num_cameras: usize,
    /// Enable motion capture integration
    pub motion_capture: bool,
    /// Enable Unreal Engine integration
    pub unreal_integration: bool,
}

impl Default for VirtualProductionConfig {
    fn default() -> Self {
        Self {
            workflow: WorkflowType::LedWall,
            target_fps: 60.0,
            sync_accuracy_ms: 0.5,
            quality: QualityMode::Preview,
            color_calibration: true,
            lens_correction: true,
            num_cameras: 1,
            motion_capture: false,
            unreal_integration: false,
        }
    }
}

impl VirtualProductionConfig {
    /// Set workflow type
    #[must_use]
    pub fn with_workflow(mut self, workflow: WorkflowType) -> Self {
        self.workflow = workflow;
        self
    }

    /// Set target FPS
    #[must_use]
    pub fn with_target_fps(mut self, fps: f64) -> Self {
        self.target_fps = fps;
        self
    }

    /// Set sync accuracy in milliseconds
    #[must_use]
    pub fn with_sync_accuracy_ms(mut self, accuracy: f64) -> Self {
        self.sync_accuracy_ms = accuracy;
        self
    }

    /// Set quality mode
    #[must_use]
    pub fn with_quality(mut self, quality: QualityMode) -> Self {
        self.quality = quality;
        self
    }

    /// Set number of cameras
    #[must_use]
    pub fn with_num_cameras(mut self, num: usize) -> Self {
        self.num_cameras = num;
        self
    }
}

/// Main virtual production system
pub struct VirtualProduction {
    config: VirtualProductionConfig,
    camera_tracker: tracking::camera::CameraTracker,
    led_renderer: led::render::LedRenderer,
    compositor: icvfx::composite::IcvfxCompositor,
    color_pipeline: color::pipeline::ColorPipeline,
    genlock: sync::genlock::GenlockSync,
    multicam_manager: Option<multicam::manager::MultiCameraManager>,
}

impl VirtualProduction {
    /// Create new virtual production system
    pub fn new(config: VirtualProductionConfig) -> Result<Self> {
        let camera_tracker =
            tracking::camera::CameraTracker::new(tracking::camera::CameraTrackerConfig::default())?;

        let led_renderer =
            led::render::LedRenderer::new(led::render::LedRendererConfig::default())?;

        let compositor =
            icvfx::composite::IcvfxCompositor::new(icvfx::composite::CompositorConfig::default())?;

        let color_pipeline =
            color::pipeline::ColorPipeline::new(color::pipeline::ColorPipelineConfig::default())?;

        let genlock = sync::genlock::GenlockSync::new(sync::genlock::GenlockConfig::default())?;

        let multicam_manager = if config.num_cameras > 1 {
            Some(multicam::manager::MultiCameraManager::new(
                multicam::manager::MultiCameraConfig {
                    num_cameras: config.num_cameras,
                    auto_switch: false,
                },
            )?)
        } else {
            None
        };

        Ok(Self {
            config,
            camera_tracker,
            led_renderer,
            compositor,
            color_pipeline,
            genlock,
            multicam_manager,
        })
    }

    /// Get current configuration
    #[must_use]
    pub fn config(&self) -> &VirtualProductionConfig {
        &self.config
    }

    /// Get camera tracker
    #[must_use]
    pub fn camera_tracker(&self) -> &tracking::camera::CameraTracker {
        &self.camera_tracker
    }

    /// Get mutable camera tracker
    pub fn camera_tracker_mut(&mut self) -> &mut tracking::camera::CameraTracker {
        &mut self.camera_tracker
    }

    /// Get LED renderer
    #[must_use]
    pub fn led_renderer(&self) -> &led::render::LedRenderer {
        &self.led_renderer
    }

    /// Get mutable LED renderer
    pub fn led_renderer_mut(&mut self) -> &mut led::render::LedRenderer {
        &mut self.led_renderer
    }

    /// Get compositor
    #[must_use]
    pub fn compositor(&self) -> &icvfx::composite::IcvfxCompositor {
        &self.compositor
    }

    /// Get mutable compositor
    pub fn compositor_mut(&mut self) -> &mut icvfx::composite::IcvfxCompositor {
        &mut self.compositor
    }

    /// Reconfigure the compositor with a new resolution.
    ///
    /// Useful in tests to use a small resolution (e.g. 64×64) for speed.
    pub fn set_compositor_resolution(&mut self, width: usize, height: usize) -> Result<()> {
        let config = icvfx::composite::CompositorConfig {
            resolution: (width, height),
            ..self.compositor.config().clone()
        };
        self.compositor = icvfx::composite::IcvfxCompositor::new(config)?;
        Ok(())
    }

    /// Get color pipeline
    #[must_use]
    pub fn color_pipeline(&self) -> &color::pipeline::ColorPipeline {
        &self.color_pipeline
    }

    /// Get mutable reference to color pipeline
    pub fn color_pipeline_mut(&mut self) -> &mut color::pipeline::ColorPipeline {
        &mut self.color_pipeline
    }

    /// Get genlock sync
    #[must_use]
    pub fn genlock(&self) -> &sync::genlock::GenlockSync {
        &self.genlock
    }

    /// Get multi-camera manager
    #[must_use]
    pub fn multicam_manager(&self) -> Option<&multicam::manager::MultiCameraManager> {
        self.multicam_manager.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = VirtualProductionConfig::default();
        assert_eq!(config.workflow, WorkflowType::LedWall);
        assert_eq!(config.target_fps, 60.0);
        assert_eq!(config.sync_accuracy_ms, 0.5);
        assert_eq!(config.quality, QualityMode::Preview);
    }

    #[test]
    fn test_config_builder() {
        let config = VirtualProductionConfig::default()
            .with_workflow(WorkflowType::Hybrid)
            .with_target_fps(120.0)
            .with_quality(QualityMode::Final)
            .with_num_cameras(4);

        assert_eq!(config.workflow, WorkflowType::Hybrid);
        assert_eq!(config.target_fps, 120.0);
        assert_eq!(config.quality, QualityMode::Final);
        assert_eq!(config.num_cameras, 4);
    }

    #[test]
    fn test_virtual_production_creation() {
        let config = VirtualProductionConfig::default();
        let vp = VirtualProduction::new(config);
        assert!(vp.is_ok());
    }

    #[test]
    fn test_multicam_manager_creation() {
        let config = VirtualProductionConfig::default().with_num_cameras(4);
        let vp = VirtualProduction::new(config).expect("should succeed in test");
        assert!(vp.multicam_manager().is_some());
    }
}
