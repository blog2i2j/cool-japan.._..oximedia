//! Automatic failover switching.

use crate::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

/// Failover switch controller.
#[derive(Clone)]
pub struct FailoverSwitch {
    delay_ms: u64,
    active_channel: Arc<RwLock<HashMap<usize, bool>>>,
}

impl FailoverSwitch {
    /// Create a new failover switch.
    pub fn new(delay_ms: u64) -> Self {
        info!("Creating failover switch with {}ms delay", delay_ms);

        Self {
            delay_ms,
            active_channel: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Trigger failover for a channel (switch to secondary).
    pub async fn trigger(&self, channel_id: usize) -> Result<()> {
        info!("Triggering failover switch for channel {}", channel_id);

        // Apply switch delay
        if self.delay_ms > 0 {
            tokio::time::sleep(tokio::time::Duration::from_millis(self.delay_ms)).await;
        }

        // In a real implementation, this would:
        // 1. Send control signals to physical switchers/routers
        // 2. Redirect network streams
        // 3. Update monitoring systems
        // 4. Log the failover event

        let mut active = self.active_channel.write().await;
        active.insert(channel_id, false); // false = secondary

        info!("Failover switch completed for channel {}", channel_id);

        Ok(())
    }

    /// Restore to primary for a channel.
    pub async fn restore(&self, channel_id: usize) -> Result<()> {
        info!("Restoring channel {} to primary", channel_id);

        // Apply switch delay
        if self.delay_ms > 0 {
            tokio::time::sleep(tokio::time::Duration::from_millis(self.delay_ms)).await;
        }

        let mut active = self.active_channel.write().await;
        active.insert(channel_id, true); // true = primary

        info!("Restore to primary completed for channel {}", channel_id);

        Ok(())
    }

    /// Check if a channel is on primary (true) or secondary (false).
    pub async fn is_primary(&self, channel_id: usize) -> bool {
        let active = self.active_channel.read().await;
        active.get(&channel_id).copied().unwrap_or(true)
    }

    /// Perform synchronized failover for multiple channels.
    pub async fn synchronized_failover(&self, channel_ids: &[usize]) -> Result<()> {
        info!(
            "Performing synchronized failover for {} channels",
            channel_ids.len()
        );

        for &channel_id in channel_ids {
            self.trigger(channel_id).await?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_failover_switch_creation() {
        let switch = FailoverSwitch::new(100);
        assert_eq!(switch.delay_ms, 100);
    }

    #[tokio::test]
    async fn test_trigger_failover() {
        let switch = FailoverSwitch::new(0);
        assert!(switch.is_primary(0).await);

        switch.trigger(0).await.expect("operation should succeed");
        assert!(!switch.is_primary(0).await);
    }

    #[tokio::test]
    async fn test_restore_primary() {
        let switch = FailoverSwitch::new(0);

        switch.trigger(0).await.expect("operation should succeed");
        assert!(!switch.is_primary(0).await);

        switch.restore(0).await.expect("operation should succeed");
        assert!(switch.is_primary(0).await);
    }

    #[tokio::test]
    async fn test_synchronized_failover() {
        let switch = FailoverSwitch::new(0);

        let channels = vec![0, 1, 2];
        switch
            .synchronized_failover(&channels)
            .await
            .expect("operation should succeed");

        for channel in channels {
            assert!(!switch.is_primary(channel).await);
        }
    }
}
