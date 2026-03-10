//! Automated switcher control for live production.

use crate::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

/// Switcher automation state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SwitcherState {
    /// Switcher is stopped
    Stopped,
    /// Switcher is running
    Running,
    /// Switcher is in transition
    InTransition,
}

/// Automated switcher for live production.
pub struct AutomatedSwitcher {
    channel_id: usize,
    state: Arc<RwLock<SwitcherState>>,
    current_program: Arc<RwLock<Option<usize>>>,
    current_preview: Arc<RwLock<Option<usize>>>,
}

impl AutomatedSwitcher {
    /// Create a new automated switcher.
    pub async fn new(channel_id: usize) -> Result<Self> {
        info!("Creating automated switcher for channel {}", channel_id);

        Ok(Self {
            channel_id,
            state: Arc::new(RwLock::new(SwitcherState::Stopped)),
            current_program: Arc::new(RwLock::new(None)),
            current_preview: Arc::new(RwLock::new(None)),
        })
    }

    /// Start switcher automation.
    pub async fn start(&mut self) -> Result<()> {
        info!(
            "Starting switcher automation for channel {}",
            self.channel_id
        );

        let mut state = self.state.write().await;
        *state = SwitcherState::Running;

        Ok(())
    }

    /// Stop switcher automation.
    pub async fn stop(&mut self) -> Result<()> {
        info!(
            "Stopping switcher automation for channel {}",
            self.channel_id
        );

        let mut state = self.state.write().await;
        *state = SwitcherState::Stopped;

        Ok(())
    }

    /// Perform automated cut.
    pub async fn auto_cut(&mut self) -> Result<()> {
        info!("Performing automated cut on channel {}", self.channel_id);

        let mut state = self.state.write().await;
        *state = SwitcherState::InTransition;

        // Swap program and preview
        let preview = *self.current_preview.read().await;
        let mut program = self.current_program.write().await;
        *program = preview;

        *state = SwitcherState::Running;

        Ok(())
    }

    /// Set program source.
    pub async fn set_program(&mut self, source: usize) -> Result<()> {
        let mut program = self.current_program.write().await;
        *program = Some(source);
        Ok(())
    }

    /// Set preview source.
    pub async fn set_preview(&mut self, source: usize) -> Result<()> {
        let mut preview = self.current_preview.write().await;
        *preview = Some(source);
        Ok(())
    }

    /// Get current program source.
    pub async fn get_program(&self) -> Option<usize> {
        *self.current_program.read().await
    }

    /// Get current preview source.
    pub async fn get_preview(&self) -> Option<usize> {
        *self.current_preview.read().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_switcher_creation() {
        let switcher = AutomatedSwitcher::new(0).await;
        assert!(switcher.is_ok());
    }

    #[tokio::test]
    async fn test_switcher_program_preview() {
        let mut switcher = AutomatedSwitcher::new(0).await.expect("new should succeed");

        switcher
            .set_program(1)
            .await
            .expect("operation should succeed");
        switcher
            .set_preview(2)
            .await
            .expect("operation should succeed");

        assert_eq!(switcher.get_program().await, Some(1));
        assert_eq!(switcher.get_preview().await, Some(2));
    }

    #[tokio::test]
    async fn test_auto_cut() {
        let mut switcher = AutomatedSwitcher::new(0).await.expect("new should succeed");

        switcher
            .set_program(1)
            .await
            .expect("operation should succeed");
        switcher
            .set_preview(2)
            .await
            .expect("operation should succeed");
        switcher.auto_cut().await.expect("operation should succeed");

        assert_eq!(switcher.get_program().await, Some(2));
    }
}
