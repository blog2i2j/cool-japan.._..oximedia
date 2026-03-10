//! Virtual production workflow management
//!
//! Provides high-level workflow management for common virtual production scenarios.

use crate::{
    icvfx::composite::CompositeFrame, led::LedWall, tracking::CameraPose, QualityMode, Result,
    VirtualProduction, VirtualProductionConfig, WorkflowType,
};

use serde::{Deserialize, Serialize};

/// Workflow state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkflowState {
    /// Workflow is idle
    Idle,
    /// Calibrating
    Calibrating,
    /// Ready to record
    Ready,
    /// Recording
    Recording,
    /// Playing back
    Playback,
    /// Error state
    Error,
}

/// Workflow session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowSession {
    /// Session ID
    pub id: String,
    /// Workflow type
    pub workflow_type: WorkflowType,
    /// Current state
    pub state: WorkflowState,
    /// Start time
    pub start_time: Option<u64>,
    /// Frame count
    pub frame_count: u64,
}

impl WorkflowSession {
    /// Create new workflow session
    #[must_use]
    pub fn new(id: String, workflow_type: WorkflowType) -> Self {
        Self {
            id,
            workflow_type,
            state: WorkflowState::Idle,
            start_time: None,
            frame_count: 0,
        }
    }

    /// Start session
    pub fn start(&mut self, timestamp_ns: u64) {
        self.start_time = Some(timestamp_ns);
        self.state = WorkflowState::Recording;
        self.frame_count = 0;
    }

    /// Stop session
    pub fn stop(&mut self) {
        self.state = WorkflowState::Idle;
    }

    /// Increment frame count
    pub fn next_frame(&mut self) {
        self.frame_count += 1;
    }
}

/// LED wall workflow
pub struct LedWallWorkflow {
    vp: VirtualProduction,
    session: Option<WorkflowSession>,
    #[allow(dead_code)]
    led_wall: LedWall,
}

impl LedWallWorkflow {
    /// Create new LED wall workflow with production-sized panel (1920x1080)
    pub fn new() -> Result<Self> {
        Self::with_panel_resolution((1920, 1080))
    }

    /// Create LED wall workflow with a specified panel resolution.
    ///
    /// Prefer small resolutions (e.g. 64×64) in unit tests to keep them fast.
    pub fn with_panel_resolution(panel_res: (usize, usize)) -> Result<Self> {
        let config = VirtualProductionConfig::default()
            .with_workflow(WorkflowType::LedWall)
            .with_quality(QualityMode::Final);

        let mut vp = VirtualProduction::new(config)?;
        let mut led_wall = LedWall::new("Main LED Wall".to_string());

        // Add a default panel so the renderer can work
        use crate::led::LedPanel;
        use crate::math::Point3;
        led_wall.add_panel(LedPanel::new(
            Point3::new(0.0, 0.0, 0.0),
            5.0,
            3.0,
            panel_res,
            2.5,
        ));
        vp.led_renderer_mut().set_led_wall(led_wall.clone());

        Ok(Self {
            vp,
            session: None,
            led_wall,
        })
    }

    /// Start recording session
    pub fn start_recording(&mut self, session_id: String, timestamp_ns: u64) -> Result<()> {
        let mut session = WorkflowSession::new(session_id, WorkflowType::LedWall);
        session.start(timestamp_ns);
        self.session = Some(session);
        Ok(())
    }

    /// Process frame
    pub fn process_frame(
        &mut self,
        camera_pose: &CameraPose,
        source_frame: &[u8],
        source_width: usize,
        source_height: usize,
        timestamp_ns: u64,
    ) -> Result<Vec<u8>> {
        // Update session
        if let Some(session) = &mut self.session {
            if session.state == WorkflowState::Recording {
                session.next_frame();
            }
        }

        // Render to LED wall
        let led_output = self.vp.led_renderer_mut().render(
            camera_pose,
            source_frame,
            source_width,
            source_height,
            timestamp_ns,
        )?;

        Ok(led_output.pixels)
    }

    /// Stop recording
    pub fn stop_recording(&mut self) {
        if let Some(session) = &mut self.session {
            session.stop();
        }
    }

    /// Get current session
    #[must_use]
    pub fn session(&self) -> Option<&WorkflowSession> {
        self.session.as_ref()
    }
}

/// Hybrid workflow (LED wall + green screen)
pub struct HybridWorkflow {
    vp: VirtualProduction,
    session: Option<WorkflowSession>,
    #[allow(dead_code)]
    led_wall: LedWall,
}

impl HybridWorkflow {
    /// Create new hybrid workflow with the default production resolution (1920×1080).
    pub fn new() -> Result<Self> {
        Self::with_resolution(1920, 1080)
    }

    /// Create a hybrid workflow with a custom compositor resolution.
    ///
    /// Use small values (e.g. 64×64) in unit tests to keep them fast.
    pub fn with_resolution(width: usize, height: usize) -> Result<Self> {
        let config = VirtualProductionConfig::default()
            .with_workflow(WorkflowType::Hybrid)
            .with_quality(QualityMode::Final);

        let mut vp = VirtualProduction::new(config)?;
        vp.set_compositor_resolution(width, height)?;
        let led_wall = LedWall::new("Hybrid Wall".to_string());

        Ok(Self {
            vp,
            session: None,
            led_wall,
        })
    }

    /// Start session
    pub fn start_session(&mut self, session_id: String, timestamp_ns: u64) -> Result<()> {
        let mut session = WorkflowSession::new(session_id, WorkflowType::Hybrid);
        session.start(timestamp_ns);
        self.session = Some(session);
        Ok(())
    }

    /// Composite frame
    pub fn composite_frame(
        &mut self,
        foreground: &[u8],
        background: &[u8],
        timestamp_ns: u64,
    ) -> Result<CompositeFrame> {
        self.vp
            .compositor_mut()
            .composite(foreground, background, None, timestamp_ns)
    }
}

/// AR workflow
pub struct ArWorkflow {
    vp: VirtualProduction,
    session: Option<WorkflowSession>,
}

impl ArWorkflow {
    /// Create new AR workflow with the default production resolution (1920×1080).
    pub fn new() -> Result<Self> {
        Self::with_resolution(1920, 1080)
    }

    /// Create an AR workflow with a custom compositor resolution.
    ///
    /// Use small values (e.g. 64×64) in unit tests to keep them fast.
    pub fn with_resolution(width: usize, height: usize) -> Result<Self> {
        let config = VirtualProductionConfig::default()
            .with_workflow(WorkflowType::AugmentedReality)
            .with_quality(QualityMode::Preview);

        let mut vp = VirtualProduction::new(config)?;
        vp.set_compositor_resolution(width, height)?;

        Ok(Self { vp, session: None })
    }

    /// Start AR session
    pub fn start(&mut self, session_id: String, timestamp_ns: u64) -> Result<()> {
        let mut session = WorkflowSession::new(session_id, WorkflowType::AugmentedReality);
        session.start(timestamp_ns);
        self.session = Some(session);
        Ok(())
    }

    /// Overlay AR content
    pub fn overlay(
        &mut self,
        camera_feed: &[u8],
        virtual_content: &[u8],
        timestamp_ns: u64,
    ) -> Result<CompositeFrame> {
        self.vp
            .compositor_mut()
            .composite(camera_feed, virtual_content, None, timestamp_ns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workflow_session() {
        let mut session = WorkflowSession::new("test-1".to_string(), WorkflowType::LedWall);
        assert_eq!(session.state, WorkflowState::Idle);

        session.start(0);
        assert_eq!(session.state, WorkflowState::Recording);

        session.next_frame();
        assert_eq!(session.frame_count, 1);

        session.stop();
        assert_eq!(session.state, WorkflowState::Idle);
    }

    #[test]
    fn test_led_wall_workflow() {
        let workflow = LedWallWorkflow::new();
        assert!(workflow.is_ok());
    }

    #[test]
    fn test_led_wall_workflow_session() {
        let mut workflow = LedWallWorkflow::new().expect("should succeed in test");

        workflow
            .start_recording("session-1".to_string(), 0)
            .expect("should succeed in test");
        assert!(workflow.session().is_some());

        workflow.stop_recording();
        assert_eq!(
            workflow.session().expect("should succeed in test").state,
            WorkflowState::Idle
        );
    }

    #[test]
    fn test_hybrid_workflow() {
        let workflow = HybridWorkflow::new();
        assert!(workflow.is_ok());
    }

    #[test]
    fn test_ar_workflow() {
        let workflow = ArWorkflow::new();
        assert!(workflow.is_ok());
    }

    #[test]
    fn test_ar_workflow_start() {
        let mut workflow = ArWorkflow::new().expect("should succeed in test");
        workflow
            .start("ar-session-1".to_string(), 0)
            .expect("should succeed in test");
        assert!(workflow.session.is_some());
    }
}
