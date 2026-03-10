//! Playlist execution engine with frame-accurate timing.

use crate::playlist::event::{EventType, PlaylistEvent};
use crate::playlist::preroll::PrerollManager;
use crate::Result;
use oximedia_timecode::Timecode;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Playlist item for execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutableItem {
    /// Item ID
    pub id: String,
    /// Media file path
    pub file_path: String,
    /// Duration in frames
    pub duration_frames: u64,
    /// Scheduled start time
    pub scheduled_start: Option<Timecode>,
    /// Pre-roll frames required
    pub preroll_frames: u64,
    /// Metadata
    pub metadata: std::collections::HashMap<String, String>,
}

/// Playlist executor state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutorState {
    /// Executor is idle
    Idle,
    /// Executor is running
    Running,
    /// Pre-rolling next item
    PreRolling,
    /// Paused
    Paused,
}

/// Playlist execution engine.
#[allow(dead_code)]
pub struct PlaylistExecutor {
    state: Arc<RwLock<ExecutorState>>,
    queue: Arc<RwLock<VecDeque<ExecutableItem>>>,
    current_item: Arc<RwLock<Option<ExecutableItem>>>,
    preroll_manager: PrerollManager,
    frame_position: Arc<RwLock<u64>>,
}

impl PlaylistExecutor {
    /// Create a new playlist executor.
    pub async fn new() -> Result<Self> {
        info!("Creating playlist executor");

        Ok(Self {
            state: Arc::new(RwLock::new(ExecutorState::Idle)),
            queue: Arc::new(RwLock::new(VecDeque::new())),
            current_item: Arc::new(RwLock::new(None)),
            preroll_manager: PrerollManager::new(),
            frame_position: Arc::new(RwLock::new(0)),
        })
    }

    /// Add item to execution queue.
    pub async fn enqueue(&self, item: ExecutableItem) -> Result<()> {
        debug!("Enqueueing item: {}", item.id);

        let mut queue = self.queue.write().await;
        queue.push_back(item);

        Ok(())
    }

    /// Start playlist execution.
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting playlist executor");

        let mut state = self.state.write().await;
        *state = ExecutorState::Running;

        Ok(())
    }

    /// Stop playlist execution.
    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping playlist executor");

        let mut state = self.state.write().await;
        *state = ExecutorState::Idle;

        Ok(())
    }

    /// Pause playlist execution.
    pub async fn pause(&mut self) -> Result<()> {
        let mut state = self.state.write().await;
        *state = ExecutorState::Paused;

        Ok(())
    }

    /// Resume playlist execution.
    pub async fn resume(&mut self) -> Result<()> {
        let mut state = self.state.write().await;
        *state = ExecutorState::Running;

        Ok(())
    }

    /// Process next frame.
    pub async fn process_frame(&mut self) -> Result<Option<PlaylistEvent>> {
        let state = *self.state.read().await;

        if state != ExecutorState::Running {
            return Ok(None);
        }

        let mut frame_pos = self.frame_position.write().await;
        *frame_pos += 1;

        // Check if current item is complete
        let current_complete = {
            let current = self.current_item.read().await;
            if let Some(ref item) = *current {
                *frame_pos >= item.duration_frames
            } else {
                true
            }
        };

        // Load next item if current is complete
        if current_complete {
            let next_item = {
                let mut queue = self.queue.write().await;
                queue.pop_front()
            };

            if let Some(item) = next_item {
                info!("Loading next item: {}", item.id);

                let mut current = self.current_item.write().await;
                *current = Some(item.clone());

                *frame_pos = 0;

                return Ok(Some(PlaylistEvent {
                    event_type: EventType::ItemStart,
                    item_id: item.id,
                    timestamp: std::time::SystemTime::now(),
                }));
            }
        }

        Ok(None)
    }

    /// Get current item.
    pub async fn current_item(&self) -> Option<ExecutableItem> {
        self.current_item.read().await.clone()
    }

    /// Get queue length.
    pub async fn queue_length(&self) -> usize {
        self.queue.read().await.len()
    }

    /// Get current frame position.
    pub async fn frame_position(&self) -> u64 {
        *self.frame_position.read().await
    }
}

impl Default for PlaylistExecutor {
    fn default() -> Self {
        Self {
            state: Arc::new(RwLock::new(ExecutorState::Idle)),
            queue: Arc::new(RwLock::new(VecDeque::new())),
            current_item: Arc::new(RwLock::new(None)),
            preroll_manager: PrerollManager::new(),
            frame_position: Arc::new(RwLock::new(0)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_executor_creation() {
        let executor = PlaylistExecutor::new().await;
        assert!(executor.is_ok());
    }

    #[tokio::test]
    async fn test_enqueue_item() {
        let executor = PlaylistExecutor::new().await.expect("new should succeed");

        let item = ExecutableItem {
            id: "test1".to_string(),
            file_path: "/path/to/file.mxf".to_string(),
            duration_frames: 1000,
            scheduled_start: None,
            preroll_frames: 150,
            metadata: std::collections::HashMap::new(),
        };

        executor
            .enqueue(item)
            .await
            .expect("operation should succeed");
        assert_eq!(executor.queue_length().await, 1);
    }
}
