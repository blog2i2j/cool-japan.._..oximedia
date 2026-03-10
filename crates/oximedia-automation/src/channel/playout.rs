//! Playlist playout engine.

use crate::Result;
use oximedia_timecode::Timecode;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

/// Playout state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlayoutState {
    /// Playout is stopped
    Stopped,
    /// Playout is playing
    Playing,
    /// Playout is paused
    Paused,
    /// Pre-rolling next item
    PreRolling,
    /// Waiting for scheduled time
    Waiting,
}

/// Current playout item information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayoutItem {
    /// Item ID
    pub id: String,
    /// Item title
    pub title: String,
    /// Duration in frames
    pub duration_frames: u64,
    /// Current position in frames
    pub position_frames: u64,
    /// Scheduled start time
    pub scheduled_start: Option<Timecode>,
}

/// Playlist playout engine.
///
/// Executes broadcast playlists with frame-accurate timing.
pub struct PlayoutEngine {
    channel_id: usize,
    state: Arc<RwLock<PlayoutState>>,
    current_item: Arc<RwLock<Option<PlayoutItem>>>,
    frame_count: Arc<RwLock<u64>>,
}

impl PlayoutEngine {
    /// Create a new playout engine.
    pub async fn new(channel_id: usize) -> Result<Self> {
        info!("Creating playout engine for channel {}", channel_id);

        Ok(Self {
            channel_id,
            state: Arc::new(RwLock::new(PlayoutState::Stopped)),
            current_item: Arc::new(RwLock::new(None)),
            frame_count: Arc::new(RwLock::new(0)),
        })
    }

    /// Start playout.
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting playout for channel {}", self.channel_id);

        let mut state = self.state.write().await;
        *state = PlayoutState::Playing;

        Ok(())
    }

    /// Stop playout.
    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping playout for channel {}", self.channel_id);

        let mut state = self.state.write().await;
        *state = PlayoutState::Stopped;

        let mut current_item = self.current_item.write().await;
        *current_item = None;

        Ok(())
    }

    /// Pause playout.
    pub async fn pause(&mut self) -> Result<()> {
        let mut state = self.state.write().await;
        if *state == PlayoutState::Playing {
            *state = PlayoutState::Paused;
            info!("Playout paused for channel {}", self.channel_id);
        } else {
            warn!("Cannot pause playout in state: {:?}", *state);
        }

        Ok(())
    }

    /// Resume playout.
    pub async fn resume(&mut self) -> Result<()> {
        let mut state = self.state.write().await;
        if *state == PlayoutState::Paused {
            *state = PlayoutState::Playing;
            info!("Playout resumed for channel {}", self.channel_id);
        } else {
            warn!("Cannot resume playout in state: {:?}", *state);
        }

        Ok(())
    }

    /// Get current playout state.
    pub async fn get_state(&self) -> Result<PlayoutState> {
        Ok(*self.state.read().await)
    }

    /// Get current item.
    pub async fn current_item(&self) -> Option<PlayoutItem> {
        self.current_item.read().await.clone()
    }

    /// Load next item.
    pub async fn load_next_item(&mut self, item: PlayoutItem) -> Result<()> {
        info!("Loading next item: {}", item.title);

        let mut current_item = self.current_item.write().await;
        *current_item = Some(item);

        Ok(())
    }

    /// Advance to next frame.
    pub async fn advance_frame(&mut self) -> Result<()> {
        let mut frame_count = self.frame_count.write().await;
        *frame_count += 1;

        if let Some(ref mut item) = *self.current_item.write().await {
            item.position_frames += 1;

            // Check if item is complete
            if item.position_frames >= item.duration_frames {
                info!("Item complete: {}", item.title);
                return Ok(());
            }
        }

        Ok(())
    }

    /// Get total frame count.
    pub async fn frame_count(&self) -> u64 {
        *self.frame_count.read().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_playout_creation() {
        let playout = PlayoutEngine::new(0).await;
        assert!(playout.is_ok());
    }

    #[tokio::test]
    async fn test_playout_lifecycle() {
        let mut playout = PlayoutEngine::new(0).await.expect("new should succeed");

        assert_eq!(
            playout.get_state().await.expect("value should be valid"),
            PlayoutState::Stopped
        );
        playout.start().await.expect("operation should succeed");
        assert_eq!(
            playout.get_state().await.expect("value should be valid"),
            PlayoutState::Playing
        );
        playout.pause().await.expect("operation should succeed");
        assert_eq!(
            playout.get_state().await.expect("value should be valid"),
            PlayoutState::Paused
        );
        playout.resume().await.expect("operation should succeed");
        assert_eq!(
            playout.get_state().await.expect("value should be valid"),
            PlayoutState::Playing
        );
        playout.stop().await.expect("operation should succeed");
        assert_eq!(
            playout.get_state().await.expect("value should be valid"),
            PlayoutState::Stopped
        );
    }

    #[tokio::test]
    async fn test_load_item() {
        let mut playout = PlayoutEngine::new(0).await.expect("new should succeed");

        let item = PlayoutItem {
            id: "test1".to_string(),
            title: "Test Item".to_string(),
            duration_frames: 1000,
            position_frames: 0,
            scheduled_start: None,
        };

        playout
            .load_next_item(item.clone())
            .await
            .expect("operation should succeed");
        let current = playout.current_item().await;
        assert!(current.is_some());
        assert_eq!(current.expect("value should be valid").id, "test1");
    }
}
