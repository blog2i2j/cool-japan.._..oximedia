//! Camera angle switching engine.

use super::EditDecision;
use crate::{AngleId, FrameNumber, Result};

/// Camera angle switcher
#[derive(Debug)]
pub struct AngleSwitcher {
    /// Current active angle
    current_angle: AngleId,
    /// Minimum hold time before switching (frames)
    min_hold_time: u32,
    /// Last switch frame
    last_switch_frame: FrameNumber,
    /// Switch history
    history: Vec<SwitchEvent>,
}

/// Switch event record
#[derive(Debug, Clone, Copy)]
pub struct SwitchEvent {
    /// Frame number of switch
    pub frame: FrameNumber,
    /// Source angle
    pub from_angle: AngleId,
    /// Target angle
    pub to_angle: AngleId,
    /// Whether switch was manual or automatic
    pub manual: bool,
}

impl AngleSwitcher {
    /// Create a new angle switcher
    #[must_use]
    pub fn new(initial_angle: AngleId, min_hold_time: u32) -> Self {
        Self {
            current_angle: initial_angle,
            min_hold_time,
            last_switch_frame: 0,
            history: Vec::new(),
        }
    }

    /// Get current active angle
    #[must_use]
    pub fn current_angle(&self) -> AngleId {
        self.current_angle
    }

    /// Check if switching is allowed at frame
    #[must_use]
    pub fn can_switch(&self, current_frame: FrameNumber) -> bool {
        current_frame >= self.last_switch_frame + u64::from(self.min_hold_time)
    }

    /// Switch to new angle
    ///
    /// # Errors
    ///
    /// Returns an error if switching is not allowed
    pub fn switch_to(
        &mut self,
        angle: AngleId,
        current_frame: FrameNumber,
        manual: bool,
    ) -> Result<EditDecision> {
        if !self.can_switch(current_frame) {
            return Err(crate::MultiCamError::SwitchingError(format!(
                "Cannot switch before frame {}",
                self.last_switch_frame + u64::from(self.min_hold_time)
            )));
        }

        let from_angle = self.current_angle;
        self.current_angle = angle;
        self.last_switch_frame = current_frame;

        // Record event
        self.history.push(SwitchEvent {
            frame: current_frame,
            from_angle,
            to_angle: angle,
            manual,
        });

        Ok(EditDecision::cut(current_frame, angle))
    }

    /// Force switch regardless of timing constraints
    pub fn force_switch(&mut self, angle: AngleId, current_frame: FrameNumber) -> EditDecision {
        let from_angle = self.current_angle;
        self.current_angle = angle;
        self.last_switch_frame = current_frame;

        self.history.push(SwitchEvent {
            frame: current_frame,
            from_angle,
            to_angle: angle,
            manual: true,
        });

        EditDecision::cut(current_frame, angle)
    }

    /// Set minimum hold time
    pub fn set_min_hold_time(&mut self, frames: u32) {
        self.min_hold_time = frames;
    }

    /// Get minimum hold time
    #[must_use]
    pub fn min_hold_time(&self) -> u32 {
        self.min_hold_time
    }

    /// Get switch history
    #[must_use]
    pub fn history(&self) -> &[SwitchEvent] {
        &self.history
    }

    /// Clear switch history
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Get last switch event
    #[must_use]
    pub fn last_switch(&self) -> Option<&SwitchEvent> {
        self.history.last()
    }

    /// Count switches
    #[must_use]
    pub fn switch_count(&self) -> usize {
        self.history.len()
    }

    /// Count manual switches
    #[must_use]
    pub fn manual_switch_count(&self) -> usize {
        self.history.iter().filter(|e| e.manual).count()
    }

    /// Count automatic switches
    #[must_use]
    pub fn auto_switch_count(&self) -> usize {
        self.history.iter().filter(|e| !e.manual).count()
    }

    /// Get average time between switches
    #[must_use]
    pub fn average_switch_interval(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }

        let mut sum = 0u64;
        for i in 1..self.history.len() {
            sum += self.history[i].frame - self.history[i - 1].frame;
        }

        sum as f64 / (self.history.len() - 1) as f64
    }

    /// Get angle usage statistics
    #[must_use]
    pub fn angle_usage(&self) -> Vec<(AngleId, usize)> {
        let mut usage: std::collections::HashMap<AngleId, usize> = std::collections::HashMap::new();

        for event in &self.history {
            *usage.entry(event.to_angle).or_insert(0) += 1;
        }

        let mut result: Vec<_> = usage.into_iter().collect();
        result.sort_by_key(|&(_, count)| std::cmp::Reverse(count));
        result
    }

    /// Reset switcher to initial state
    pub fn reset(&mut self, initial_angle: AngleId) {
        self.current_angle = initial_angle;
        self.last_switch_frame = 0;
        self.history.clear();
    }
}

/// Switch mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwitchMode {
    /// Manual switching only
    Manual,
    /// Automatic switching only
    Automatic,
    /// Hybrid (automatic with manual override)
    Hybrid,
}

/// Advanced switcher with multiple modes
#[derive(Debug)]
pub struct AdvancedSwitcher {
    /// Base switcher
    switcher: AngleSwitcher,
    /// Switch mode
    mode: SwitchMode,
    /// Auto-switch enabled
    auto_enabled: bool,
}

impl AdvancedSwitcher {
    /// Create a new advanced switcher
    #[must_use]
    pub fn new(initial_angle: AngleId, min_hold_time: u32, mode: SwitchMode) -> Self {
        Self {
            switcher: AngleSwitcher::new(initial_angle, min_hold_time),
            mode,
            auto_enabled: mode != SwitchMode::Manual,
        }
    }

    /// Get current angle
    #[must_use]
    pub fn current_angle(&self) -> AngleId {
        self.switcher.current_angle()
    }

    /// Manual switch
    ///
    /// # Errors
    ///
    /// Returns an error if manual switching is not allowed
    pub fn manual_switch(&mut self, angle: AngleId, frame: FrameNumber) -> Result<EditDecision> {
        if self.mode == SwitchMode::Automatic {
            return Err(crate::MultiCamError::SwitchingError(
                "Manual switching disabled in automatic mode".to_string(),
            ));
        }

        self.switcher.switch_to(angle, frame, true)
    }

    /// Automatic switch
    ///
    /// # Errors
    ///
    /// Returns an error if automatic switching is not allowed
    pub fn auto_switch(&mut self, angle: AngleId, frame: FrameNumber) -> Result<EditDecision> {
        if !self.auto_enabled {
            return Err(crate::MultiCamError::SwitchingError(
                "Automatic switching disabled".to_string(),
            ));
        }

        self.switcher.switch_to(angle, frame, false)
    }

    /// Enable/disable automatic switching
    pub fn set_auto_enabled(&mut self, enabled: bool) {
        self.auto_enabled = enabled && self.mode != SwitchMode::Manual;
    }

    /// Check if automatic switching is enabled
    #[must_use]
    pub fn is_auto_enabled(&self) -> bool {
        self.auto_enabled
    }

    /// Get switch mode
    #[must_use]
    pub fn mode(&self) -> SwitchMode {
        self.mode
    }

    /// Set switch mode
    pub fn set_mode(&mut self, mode: SwitchMode) {
        self.mode = mode;
        if mode == SwitchMode::Manual {
            self.auto_enabled = false;
        }
    }

    /// Get inner switcher
    #[must_use]
    pub fn switcher(&self) -> &AngleSwitcher {
        &self.switcher
    }

    /// Get mutable inner switcher
    pub fn switcher_mut(&mut self) -> &mut AngleSwitcher {
        &mut self.switcher
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_switcher_creation() {
        let switcher = AngleSwitcher::new(0, 25);
        assert_eq!(switcher.current_angle(), 0);
        assert_eq!(switcher.min_hold_time(), 25);
    }

    #[test]
    fn test_switch_to() {
        let mut switcher = AngleSwitcher::new(0, 25);
        let result = switcher.switch_to(1, 30, false);
        assert!(result.is_ok());
        assert_eq!(switcher.current_angle(), 1);
        assert_eq!(switcher.switch_count(), 1);
    }

    #[test]
    fn test_min_hold_time() {
        let mut switcher = AngleSwitcher::new(0, 25);
        switcher
            .switch_to(1, 30, false)
            .expect("multicam test operation should succeed");

        // Try to switch too soon
        let result = switcher.switch_to(2, 40, false);
        assert!(result.is_err());

        // Wait until hold time elapsed
        let result = switcher.switch_to(2, 55, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_force_switch() {
        let mut switcher = AngleSwitcher::new(0, 25);
        switcher
            .switch_to(1, 30, false)
            .expect("multicam test operation should succeed");

        // Force switch ignores hold time
        let decision = switcher.force_switch(2, 40);
        assert_eq!(decision.angle, 2);
        assert_eq!(switcher.current_angle(), 2);
    }

    #[test]
    fn test_switch_statistics() {
        let mut switcher = AngleSwitcher::new(0, 1);
        switcher
            .switch_to(1, 10, true)
            .expect("multicam test operation should succeed");
        switcher
            .switch_to(2, 20, false)
            .expect("multicam test operation should succeed");
        switcher
            .switch_to(1, 30, false)
            .expect("multicam test operation should succeed");

        assert_eq!(switcher.switch_count(), 3);
        assert_eq!(switcher.manual_switch_count(), 1);
        assert_eq!(switcher.auto_switch_count(), 2);

        let usage = switcher.angle_usage();
        assert_eq!(usage[0].0, 1); // Angle 1 used most
        assert_eq!(usage[0].1, 2);
    }

    #[test]
    fn test_average_interval() {
        let mut switcher = AngleSwitcher::new(0, 1);
        switcher
            .switch_to(1, 10, false)
            .expect("multicam test operation should succeed");
        switcher
            .switch_to(2, 20, false)
            .expect("multicam test operation should succeed");
        switcher
            .switch_to(1, 30, false)
            .expect("multicam test operation should succeed");

        let avg = switcher.average_switch_interval();
        assert!((avg - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_advanced_switcher_modes() {
        let mut switcher = AdvancedSwitcher::new(0, 25, SwitchMode::Manual);
        assert_eq!(switcher.mode(), SwitchMode::Manual);
        assert!(!switcher.is_auto_enabled());

        let result = switcher.auto_switch(1, 30);
        assert!(result.is_err());

        switcher.set_mode(SwitchMode::Hybrid);
        switcher.set_auto_enabled(true);
        let result = switcher.auto_switch(1, 30);
        assert!(result.is_ok());
    }
}
