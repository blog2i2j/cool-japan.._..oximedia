//! Manual control for camera switching.

use super::ControlMode;
use crate::edit::EditDecision;
use crate::{AngleId, FrameNumber, Result};

/// Manual controller
#[derive(Debug)]
pub struct ManualController {
    /// Control mode
    mode: ControlMode,
    /// Program angle (current output)
    program_angle: AngleId,
    /// Preview angle (next to be switched)
    preview_angle: Option<AngleId>,
    /// Control history
    history: Vec<ControlEvent>,
}

/// Control event
#[derive(Debug, Clone, Copy)]
pub struct ControlEvent {
    /// Frame number
    pub frame: FrameNumber,
    /// Event type
    pub event_type: ControlEventType,
    /// Involved angle
    pub angle: AngleId,
}

/// Control event type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlEventType {
    /// Direct switch to angle
    DirectSwitch,
    /// Preview angle selected
    PreviewSelected,
    /// Preview cut to program
    PreviewCut,
    /// Transition initiated
    TransitionStart,
}

impl ManualController {
    /// Create a new manual controller
    #[must_use]
    pub fn new(mode: ControlMode, initial_angle: AngleId) -> Self {
        Self {
            mode,
            program_angle: initial_angle,
            preview_angle: None,
            history: Vec::new(),
        }
    }

    /// Get control mode
    #[must_use]
    pub fn mode(&self) -> ControlMode {
        self.mode
    }

    /// Set control mode
    pub fn set_mode(&mut self, mode: ControlMode) {
        self.mode = mode;
    }

    /// Get program angle
    #[must_use]
    pub fn program_angle(&self) -> AngleId {
        self.program_angle
    }

    /// Get preview angle
    #[must_use]
    pub fn preview_angle(&self) -> Option<AngleId> {
        self.preview_angle
    }

    /// Direct switch to angle
    ///
    /// # Errors
    ///
    /// Returns an error if operation is not allowed
    pub fn switch_to(&mut self, angle: AngleId, frame: FrameNumber) -> Result<EditDecision> {
        if self.mode == ControlMode::PreviewProgram {
            return Err(crate::MultiCamError::InvalidOperation(
                "Use preview/program workflow in PreviewProgram mode".to_string(),
            ));
        }

        self.program_angle = angle;
        self.history.push(ControlEvent {
            frame,
            event_type: ControlEventType::DirectSwitch,
            angle,
        });

        Ok(EditDecision::cut(frame, angle))
    }

    /// Select angle for preview
    ///
    /// # Errors
    ///
    /// Returns an error if operation is not allowed
    pub fn select_preview(&mut self, angle: AngleId, frame: FrameNumber) -> Result<()> {
        if self.mode == ControlMode::Direct {
            return Err(crate::MultiCamError::InvalidOperation(
                "Preview not available in Direct mode".to_string(),
            ));
        }

        self.preview_angle = Some(angle);
        self.history.push(ControlEvent {
            frame,
            event_type: ControlEventType::PreviewSelected,
            angle,
        });

        Ok(())
    }

    /// Cut preview to program
    ///
    /// # Errors
    ///
    /// Returns an error if no preview is selected
    pub fn cut_preview(&mut self, frame: FrameNumber) -> Result<EditDecision> {
        if let Some(preview) = self.preview_angle {
            self.program_angle = preview;
            self.preview_angle = None;

            self.history.push(ControlEvent {
                frame,
                event_type: ControlEventType::PreviewCut,
                angle: preview,
            });

            Ok(EditDecision::cut(frame, preview))
        } else {
            Err(crate::MultiCamError::InvalidOperation(
                "No preview angle selected".to_string(),
            ))
        }
    }

    /// Transition preview to program
    ///
    /// # Errors
    ///
    /// Returns an error if no preview is selected
    pub fn transition_preview(
        &mut self,
        frame: FrameNumber,
        duration: u32,
    ) -> Result<EditDecision> {
        if let Some(preview) = self.preview_angle {
            let decision = EditDecision::dissolve(frame, preview, duration);

            self.program_angle = preview;
            self.preview_angle = None;

            self.history.push(ControlEvent {
                frame,
                event_type: ControlEventType::TransitionStart,
                angle: preview,
            });

            Ok(decision)
        } else {
            Err(crate::MultiCamError::InvalidOperation(
                "No preview angle selected".to_string(),
            ))
        }
    }

    /// Get control history
    #[must_use]
    pub fn history(&self) -> &[ControlEvent] {
        &self.history
    }

    /// Clear history
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Get event count
    #[must_use]
    pub fn event_count(&self) -> usize {
        self.history.len()
    }

    /// Get events by type
    #[must_use]
    pub fn events_by_type(&self, event_type: ControlEventType) -> Vec<&ControlEvent> {
        self.history
            .iter()
            .filter(|e| e.event_type == event_type)
            .collect()
    }

    /// Reset controller
    pub fn reset(&mut self, initial_angle: AngleId) {
        self.program_angle = initial_angle;
        self.preview_angle = None;
        self.history.clear();
    }
}

/// Keyboard shortcuts for manual control
#[derive(Debug, Clone)]
pub struct KeyboardShortcuts {
    /// Angle selection keys (e.g., 1-9)
    pub angle_keys: Vec<String>,
    /// Cut key
    pub cut_key: String,
    /// Transition key
    pub transition_key: String,
    /// Preview toggle key
    pub preview_key: String,
}

impl Default for KeyboardShortcuts {
    fn default() -> Self {
        Self {
            angle_keys: (1..=9).map(|n| n.to_string()).collect(),
            cut_key: "Space".to_string(),
            transition_key: "T".to_string(),
            preview_key: "P".to_string(),
        }
    }
}

/// T-bar controller (like video switchers)
#[derive(Debug)]
pub struct TBarController {
    /// Current position (0.0 to 1.0)
    position: f32,
    /// Source angle (at position 0.0)
    source_angle: AngleId,
    /// Target angle (at position 1.0)
    target_angle: AngleId,
}

impl TBarController {
    /// Create a new T-bar controller
    #[must_use]
    pub fn new(source: AngleId, target: AngleId) -> Self {
        Self {
            position: 0.0,
            source_angle: source,
            target_angle: target,
        }
    }

    /// Set T-bar position
    pub fn set_position(&mut self, position: f32) {
        self.position = position.clamp(0.0, 1.0);
    }

    /// Get T-bar position
    #[must_use]
    pub fn position(&self) -> f32 {
        self.position
    }

    /// Get mix level for source angle
    #[must_use]
    pub fn source_mix(&self) -> f32 {
        1.0 - self.position
    }

    /// Get mix level for target angle
    #[must_use]
    pub fn target_mix(&self) -> f32 {
        self.position
    }

    /// Set source angle
    pub fn set_source(&mut self, angle: AngleId) {
        self.source_angle = angle;
    }

    /// Set target angle
    pub fn set_target(&mut self, angle: AngleId) {
        self.target_angle = angle;
    }

    /// Get source angle
    #[must_use]
    pub fn source(&self) -> AngleId {
        self.source_angle
    }

    /// Get target angle
    #[must_use]
    pub fn target(&self) -> AngleId {
        self.target_angle
    }

    /// Reset to start position
    pub fn reset_to_source(&mut self) {
        self.position = 0.0;
    }

    /// Reset to end position
    pub fn reset_to_target(&mut self) {
        self.position = 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manual_controller_creation() {
        let controller = ManualController::new(ControlMode::Direct, 0);
        assert_eq!(controller.mode(), ControlMode::Direct);
        assert_eq!(controller.program_angle(), 0);
    }

    #[test]
    fn test_direct_switch() {
        let mut controller = ManualController::new(ControlMode::Direct, 0);
        let result = controller.switch_to(1, 100);
        assert!(result.is_ok());
        assert_eq!(controller.program_angle(), 1);
    }

    #[test]
    fn test_preview_program() {
        let mut controller = ManualController::new(ControlMode::PreviewProgram, 0);
        assert!(controller.select_preview(1, 100).is_ok());
        assert_eq!(controller.preview_angle(), Some(1));

        assert!(controller.cut_preview(105).is_ok());
        assert_eq!(controller.program_angle(), 1);
        assert_eq!(controller.preview_angle(), None);
    }

    #[test]
    fn test_transition_preview() {
        let mut controller = ManualController::new(ControlMode::PreviewProgram, 0);
        controller
            .select_preview(1, 100)
            .expect("multicam test operation should succeed");

        let result = controller.transition_preview(105, 10);
        assert!(result.is_ok());
        assert_eq!(controller.program_angle(), 1);
    }

    #[test]
    fn test_events_by_type() {
        let mut controller = ManualController::new(ControlMode::Direct, 0);
        controller
            .switch_to(1, 100)
            .expect("multicam test operation should succeed");
        controller
            .switch_to(2, 200)
            .expect("multicam test operation should succeed");

        let direct_switches = controller.events_by_type(ControlEventType::DirectSwitch);
        assert_eq!(direct_switches.len(), 2);
    }

    #[test]
    fn test_tbar_controller() {
        let mut tbar = TBarController::new(0, 1);
        assert_eq!(tbar.position(), 0.0);

        tbar.set_position(0.5);
        assert_eq!(tbar.source_mix(), 0.5);
        assert_eq!(tbar.target_mix(), 0.5);

        tbar.reset_to_target();
        assert_eq!(tbar.position(), 1.0);
    }

    #[test]
    fn test_keyboard_shortcuts() {
        let shortcuts = KeyboardShortcuts::default();
        assert_eq!(shortcuts.angle_keys.len(), 9);
        assert_eq!(shortcuts.cut_key, "Space");
    }
}
