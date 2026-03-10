//! Live production switcher automation.

use crate::{AutomationError, Result};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

/// Live switcher automation configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveSwitcherConfig {
    /// Number of inputs
    pub num_inputs: usize,
    /// Auto-cut interval in seconds (0 = manual)
    pub auto_cut_interval: u64,
    /// Enable audio follow video
    pub audio_follow_video: bool,
}

impl Default for LiveSwitcherConfig {
    fn default() -> Self {
        Self {
            num_inputs: 8,
            auto_cut_interval: 0,
            audio_follow_video: true,
        }
    }
}

/// Live switcher automation.
pub struct LiveSwitcherAutomation {
    config: LiveSwitcherConfig,
    current_source: Option<usize>,
    next_source: Option<usize>,
}

impl LiveSwitcherAutomation {
    /// Create a new live switcher automation.
    pub fn new(config: LiveSwitcherConfig) -> Self {
        info!(
            "Creating live switcher automation with {} inputs",
            config.num_inputs
        );

        Self {
            config,
            current_source: None,
            next_source: None,
        }
    }

    /// Set next source for preview.
    pub fn set_preview(&mut self, source: usize) -> Result<()> {
        if source >= self.config.num_inputs {
            return Err(AutomationError::LiveSwitching(format!(
                "Invalid source: {} (max: {})",
                source,
                self.config.num_inputs - 1
            )));
        }

        debug!("Setting preview to source: {}", source);
        self.next_source = Some(source);

        Ok(())
    }

    /// Perform auto-cut to preview source.
    pub fn auto_cut(&mut self) -> Result<()> {
        if let Some(next) = self.next_source {
            info!("Performing auto-cut to source: {}", next);
            self.current_source = Some(next);
            self.next_source = None;
            Ok(())
        } else {
            Err(AutomationError::LiveSwitching(
                "No preview source set".to_string(),
            ))
        }
    }

    /// Get current program source.
    pub fn current_source(&self) -> Option<usize> {
        self.current_source
    }

    /// Get next preview source.
    pub fn preview_source(&self) -> Option<usize> {
        self.next_source
    }

    /// Perform automated sequence (cycle through sources).
    pub fn auto_sequence(&mut self) -> Result<()> {
        let current = self.current_source.unwrap_or(0);
        let next = (current + 1) % self.config.num_inputs;

        info!("Auto-sequence: {} -> {}", current, next);
        self.current_source = Some(next);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_live_switcher_creation() {
        let config = LiveSwitcherConfig::default();
        let switcher = LiveSwitcherAutomation::new(config);
        assert_eq!(switcher.current_source(), None);
    }

    #[test]
    fn test_set_preview() {
        let config = LiveSwitcherConfig::default();
        let mut switcher = LiveSwitcherAutomation::new(config);

        assert!(switcher.set_preview(2).is_ok());
        assert_eq!(switcher.preview_source(), Some(2));
    }

    #[test]
    fn test_auto_cut() {
        let config = LiveSwitcherConfig::default();
        let mut switcher = LiveSwitcherAutomation::new(config);

        switcher.set_preview(2).expect("set_preview should succeed");
        switcher.auto_cut().expect("auto_cut should succeed");

        assert_eq!(switcher.current_source(), Some(2));
        assert_eq!(switcher.preview_source(), None);
    }

    #[test]
    fn test_auto_sequence() {
        let config = LiveSwitcherConfig {
            num_inputs: 4,
            ..Default::default()
        };
        let mut switcher = LiveSwitcherAutomation::new(config);

        switcher
            .auto_sequence()
            .expect("auto_sequence should succeed");
        assert_eq!(switcher.current_source(), Some(1));

        switcher
            .auto_sequence()
            .expect("auto_sequence should succeed");
        assert_eq!(switcher.current_source(), Some(2));
    }
}
