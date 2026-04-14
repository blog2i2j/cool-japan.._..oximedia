//! Failover management and orchestration.

use crate::failover::health::{HealthMonitor, HealthStatus};
use crate::failover::switch::FailoverSwitch;
use crate::{AutomationError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

/// Redundancy mode for the failover manager.
///
/// Controls how standby channels are selected when the primary fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum RedundancyMode {
    /// Classic 1+1 hot-standby — a single dedicated standby is always ready
    /// and takes over immediately when the primary fails.
    #[default]
    HotStandby,
    /// N+1 redundancy — a shared pool of standby channels covers N primary
    /// channels.  When any primary fails the manager picks the first healthy
    /// standby from the pool, optimising hardware utilisation.
    NplusOne,
}

impl std::fmt::Display for RedundancyMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HotStandby => write!(f, "HotStandby (1+1)"),
            Self::NplusOne => write!(f, "N+1"),
        }
    }
}

/// Failover configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailoverConfig {
    /// Enable automatic failover
    pub auto_failover: bool,
    /// Health check interval in seconds
    pub health_check_interval: u64,
    /// Number of failed checks before failover
    pub failure_threshold: u32,
    /// Failover switch delay in milliseconds
    pub switch_delay_ms: u64,
    /// Redundancy mode.
    pub redundancy_mode: RedundancyMode,
    /// Standby channel IDs available for N+1 replacement.
    ///
    /// Used only when `redundancy_mode` is [`RedundancyMode::NplusOne`].
    /// Each entry is a channel index that is currently in standby and may be
    /// assigned to replace a failing primary.
    pub standby_pool: Vec<usize>,
}

impl Default for FailoverConfig {
    fn default() -> Self {
        Self {
            auto_failover: true,
            health_check_interval: 5,
            failure_threshold: 3,
            switch_delay_ms: 100,
            redundancy_mode: RedundancyMode::HotStandby,
            standby_pool: Vec::new(),
        }
    }
}

/// Failover state for a channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FailoverState {
    /// Primary is active
    Primary,
    /// Secondary is active (failover)
    Secondary,
}

/// Assignment of a standby channel to replace a failed primary in N+1 mode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandbyAssignment {
    /// The primary channel that failed.
    pub primary_channel_id: usize,
    /// The standby channel assigned to replace it.
    pub standby_channel_id: usize,
}

/// Failover manager.
pub struct FailoverManager {
    config: FailoverConfig,
    health_monitor: HealthMonitor,
    failover_switch: FailoverSwitch,
    channel_states: Arc<RwLock<HashMap<usize, FailoverState>>>,
    running: Arc<RwLock<bool>>,
    /// N+1 standby assignments: primary → assigned standby.
    standby_assignments: Arc<RwLock<HashMap<usize, StandbyAssignment>>>,
    /// Available standby channels (not yet assigned).
    available_standbys: Arc<RwLock<Vec<usize>>>,
}

impl FailoverManager {
    /// Create a new failover manager.
    pub async fn new(config: FailoverConfig) -> Result<Self> {
        info!(
            "Creating failover manager (redundancy_mode={})",
            config.redundancy_mode
        );

        let available_standbys = config.standby_pool.clone();

        Ok(Self {
            config: config.clone(),
            health_monitor: HealthMonitor::new(config.health_check_interval),
            failover_switch: FailoverSwitch::new(config.switch_delay_ms),
            channel_states: Arc::new(RwLock::new(HashMap::new())),
            running: Arc::new(RwLock::new(false)),
            standby_assignments: Arc::new(RwLock::new(HashMap::new())),
            available_standbys: Arc::new(RwLock::new(available_standbys)),
        })
    }

    /// Select and assign a standby channel for the given primary in N+1 mode.
    ///
    /// Removes the first available standby from the pool and records the
    /// assignment so it can be released when the primary is restored.
    ///
    /// # Errors
    ///
    /// Returns [`AutomationError::Failover`] if no standby channels are
    /// available.
    pub async fn assign_nplusone_standby(
        &self,
        primary_channel_id: usize,
    ) -> Result<StandbyAssignment> {
        let mut pool = self.available_standbys.write().await;
        if pool.is_empty() {
            return Err(AutomationError::Failover(format!(
                "No standby channels available in N+1 pool for primary channel {}",
                primary_channel_id
            )));
        }
        // Take the first available standby.
        let standby_channel_id = pool.remove(0);
        let assignment = StandbyAssignment {
            primary_channel_id,
            standby_channel_id,
        };
        drop(pool);

        let mut assignments = self.standby_assignments.write().await;
        assignments.insert(primary_channel_id, assignment.clone());
        drop(assignments);

        info!(
            "N+1 assigned standby channel {} to replace primary channel {}",
            standby_channel_id, primary_channel_id
        );
        Ok(assignment)
    }

    /// Release an N+1 standby assignment, returning the standby channel to the
    /// pool when the primary is restored.
    pub async fn release_nplusone_standby(&self, primary_channel_id: usize) -> Option<usize> {
        let mut assignments = self.standby_assignments.write().await;
        if let Some(assignment) = assignments.remove(&primary_channel_id) {
            let standby = assignment.standby_channel_id;
            drop(assignments);
            let mut pool = self.available_standbys.write().await;
            pool.push(standby);
            info!(
                "N+1 released standby channel {} back to pool (primary {} restored)",
                standby, primary_channel_id
            );
            Some(standby)
        } else {
            None
        }
    }

    /// Get the current N+1 assignment for a primary channel, if any.
    pub async fn get_nplusone_assignment(
        &self,
        primary_channel_id: usize,
    ) -> Option<StandbyAssignment> {
        let assignments = self.standby_assignments.read().await;
        assignments.get(&primary_channel_id).cloned()
    }

    /// Returns the number of standby channels currently available in the pool.
    pub async fn available_standby_count(&self) -> usize {
        self.available_standbys.read().await.len()
    }

    /// Returns the current redundancy mode.
    pub fn redundancy_mode(&self) -> RedundancyMode {
        self.config.redundancy_mode
    }

    /// Start the failover manager.
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting failover manager");

        {
            let mut running = self.running.write().await;
            *running = true;
        }

        self.health_monitor.start().await?;

        // Spawn monitoring task if auto-failover is enabled
        if self.config.auto_failover {
            let health_monitor = self.health_monitor.clone();
            let failover_switch = self.failover_switch.clone();
            let channel_states = Arc::clone(&self.channel_states);
            let running = Arc::clone(&self.running);
            let failure_threshold = self.config.failure_threshold;

            tokio::spawn(async move {
                while *running.read().await {
                    // Check health and trigger failover if needed
                    let channels = health_monitor.get_all_statuses().await;

                    for (channel_id, status) in channels {
                        if status.consecutive_failures >= failure_threshold {
                            warn!(
                                "Channel {} health check failed {} times, triggering failover",
                                channel_id, status.consecutive_failures
                            );

                            if let Err(e) = failover_switch.trigger(channel_id).await {
                                error!("Failover switch failed for channel {}: {}", channel_id, e);
                            } else {
                                let mut states = channel_states.write().await;
                                states.insert(channel_id, FailoverState::Secondary);
                            }
                        }
                    }

                    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                }
            });
        }

        Ok(())
    }

    /// Stop the failover manager.
    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping failover manager");

        {
            let mut running = self.running.write().await;
            *running = false;
        }

        self.health_monitor.stop().await?;

        Ok(())
    }

    /// Manually trigger failover for a channel.
    ///
    /// In `HotStandby` mode this behaves as before.  In `NplusOne` mode it
    /// additionally selects a replacement from the standby pool.
    pub async fn trigger_failover(&mut self, channel_id: usize) -> Result<()> {
        info!(
            "Manually triggering failover for channel {} (mode={})",
            channel_id, self.config.redundancy_mode
        );

        self.failover_switch.trigger(channel_id).await?;

        if self.config.redundancy_mode == RedundancyMode::NplusOne {
            if let Err(e) = self.assign_nplusone_standby(channel_id).await {
                warn!(
                    "N+1 standby assignment failed for channel {}: {}",
                    channel_id, e
                );
            }
        }

        let mut states = self.channel_states.write().await;
        states.insert(channel_id, FailoverState::Secondary);

        Ok(())
    }

    /// Restore to primary for a channel.
    ///
    /// In `NplusOne` mode this also releases the standby back to the pool.
    pub async fn restore_primary(&mut self, channel_id: usize) -> Result<()> {
        info!("Restoring channel {} to primary", channel_id);

        self.failover_switch.restore(channel_id).await?;

        if self.config.redundancy_mode == RedundancyMode::NplusOne {
            self.release_nplusone_standby(channel_id).await;
        }

        let mut states = self.channel_states.write().await;
        states.insert(channel_id, FailoverState::Primary);

        Ok(())
    }

    /// Get failover state for a channel.
    pub async fn get_state(&self, channel_id: usize) -> FailoverState {
        let states = self.channel_states.read().await;
        states
            .get(&channel_id)
            .copied()
            .unwrap_or(FailoverState::Primary)
    }

    /// Get health status for a channel.
    pub async fn get_health(&self, channel_id: usize) -> Option<HealthStatus> {
        self.health_monitor.get_status(channel_id).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_failover_manager_creation() {
        let config = FailoverConfig::default();
        let manager = FailoverManager::new(config).await;
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_failover_state() {
        let config = FailoverConfig::default();
        let mut manager = FailoverManager::new(config)
            .await
            .expect("new should succeed");

        assert_eq!(manager.get_state(0).await, FailoverState::Primary);

        manager
            .trigger_failover(0)
            .await
            .expect("operation should succeed");
        assert_eq!(manager.get_state(0).await, FailoverState::Secondary);

        manager
            .restore_primary(0)
            .await
            .expect("operation should succeed");
        assert_eq!(manager.get_state(0).await, FailoverState::Primary);
    }

    #[test]
    fn test_redundancy_mode_display() {
        assert_eq!(RedundancyMode::HotStandby.to_string(), "HotStandby (1+1)");
        assert_eq!(RedundancyMode::NplusOne.to_string(), "N+1");
    }

    #[tokio::test]
    async fn test_nplusone_assign_and_release() {
        let config = FailoverConfig {
            redundancy_mode: RedundancyMode::NplusOne,
            standby_pool: vec![10, 11, 12],
            ..FailoverConfig::default()
        };
        let manager = FailoverManager::new(config)
            .await
            .expect("new should succeed");

        assert_eq!(manager.available_standby_count().await, 3);

        // Assign a standby for primary channel 0
        let assignment = manager
            .assign_nplusone_standby(0)
            .await
            .expect("should assign standby");
        assert_eq!(assignment.primary_channel_id, 0);
        assert_eq!(assignment.standby_channel_id, 10); // first from pool

        assert_eq!(
            manager.available_standby_count().await,
            2,
            "pool should shrink by 1"
        );

        // Assign another
        let assignment2 = manager
            .assign_nplusone_standby(1)
            .await
            .expect("should assign second standby");
        assert_eq!(assignment2.standby_channel_id, 11);
        assert_eq!(manager.available_standby_count().await, 1);

        // Release channel 0's standby
        let released = manager.release_nplusone_standby(0).await;
        assert_eq!(released, Some(10));
        assert_eq!(
            manager.available_standby_count().await,
            2,
            "pool should grow back by 1 after release"
        );
    }

    #[tokio::test]
    async fn test_nplusone_no_standby_available() {
        let config = FailoverConfig {
            redundancy_mode: RedundancyMode::NplusOne,
            standby_pool: vec![], // empty pool
            ..FailoverConfig::default()
        };
        let manager = FailoverManager::new(config)
            .await
            .expect("new should succeed");
        let result = manager.assign_nplusone_standby(0).await;
        assert!(result.is_err(), "should error when pool is empty");
    }

    #[tokio::test]
    async fn test_nplusone_trigger_failover_assigns_standby() {
        let config = FailoverConfig {
            redundancy_mode: RedundancyMode::NplusOne,
            standby_pool: vec![99],
            ..FailoverConfig::default()
        };
        let mut manager = FailoverManager::new(config)
            .await
            .expect("new should succeed");

        // trigger_failover should auto-assign in N+1 mode
        manager
            .trigger_failover(5)
            .await
            .expect("trigger_failover should succeed");

        assert_eq!(manager.get_state(5).await, FailoverState::Secondary);

        let assignment = manager.get_nplusone_assignment(5).await;
        assert!(
            assignment.is_some(),
            "N+1 assignment should exist after trigger_failover"
        );
        assert_eq!(
            assignment
                .expect("assignment should exist")
                .standby_channel_id,
            99
        );

        // Restore should release the standby
        manager
            .restore_primary(5)
            .await
            .expect("restore_primary should succeed");
        assert_eq!(manager.get_state(5).await, FailoverState::Primary);
        assert_eq!(
            manager.available_standby_count().await,
            1,
            "standby should return to pool after restore"
        );
    }

    // ── Timing-based failover tests ───────────────────────────────────────────

    /// Simulates a primary signal loss and verifies the backup activates within
    /// the configured switchover timeout using deterministic time control.
    #[tokio::test(start_paused = true)]
    async fn test_failover_activates_within_timeout_after_primary_loss() {
        // Zero delay so the switch is effectively instant in wall-clock terms
        // even when tokio::time is paused.
        let config = FailoverConfig {
            auto_failover: false, // manual control for determinism
            switch_delay_ms: 0,
            ..FailoverConfig::default()
        };
        let mut manager = FailoverManager::new(config)
            .await
            .expect("new should succeed");

        // Channel 0 starts as Primary.
        assert_eq!(manager.get_state(0).await, FailoverState::Primary);

        // Simulate signal loss after ~100 ms by advancing tokio time and
        // then calling trigger_failover.
        tokio::time::advance(tokio::time::Duration::from_millis(100)).await;

        // The switchover must complete within 500 ms.
        let result = tokio::time::timeout(
            tokio::time::Duration::from_millis(500),
            manager.trigger_failover(0),
        )
        .await;

        assert!(
            result.is_ok(),
            "failover must complete within 500 ms timeout"
        );
        result
            .expect("timeout should not fire")
            .expect("trigger_failover should succeed");

        assert_eq!(
            manager.get_state(0).await,
            FailoverState::Secondary,
            "backup channel should be active after failover"
        );
    }

    /// N+1 redundancy: 3 channels active, primary fails → secondary takes over,
    /// tertiary remains on standby.
    #[tokio::test(start_paused = true)]
    async fn test_nplusone_three_channels_primary_fails_secondary_activates() {
        let config = FailoverConfig {
            redundancy_mode: RedundancyMode::NplusOne,
            // Channels 0 and 1 are primaries; channel 2 is a standby.
            standby_pool: vec![2],
            switch_delay_ms: 0,
            auto_failover: false,
            ..FailoverConfig::default()
        };
        let mut manager = FailoverManager::new(config)
            .await
            .expect("new should succeed");

        // Initial state: all channels are "Primary" by default.
        assert_eq!(manager.get_state(0).await, FailoverState::Primary);
        assert_eq!(manager.get_state(1).await, FailoverState::Primary);
        assert_eq!(manager.get_state(2).await, FailoverState::Primary);
        assert_eq!(manager.available_standby_count().await, 1);

        // Primary channel 0 fails — should trigger failover and assign standby 2.
        manager
            .trigger_failover(0)
            .await
            .expect("trigger should succeed");

        assert_eq!(manager.get_state(0).await, FailoverState::Secondary);

        let assignment = manager.get_nplusone_assignment(0).await;
        assert!(
            assignment.is_some(),
            "a standby must be assigned for failed primary 0"
        );
        assert_eq!(
            assignment.expect("assignment").standby_channel_id,
            2,
            "standby channel 2 should be assigned"
        );

        // Channel 1 (another primary) and channel 2 (assigned standby) are
        // still in their expected states — verify pool is now empty.
        assert_eq!(
            manager.available_standby_count().await,
            0,
            "standby pool should be exhausted after one assignment"
        );

        // Channel 1 is unaffected (still Primary).
        assert_eq!(manager.get_state(1).await, FailoverState::Primary);
    }

    /// Verify that restoring a primary in N+1 mode returns the standby to the
    /// pool, making it available for future failures.
    #[tokio::test(start_paused = true)]
    async fn test_nplusone_restore_returns_standby_to_pool() {
        let config = FailoverConfig {
            redundancy_mode: RedundancyMode::NplusOne,
            standby_pool: vec![5],
            switch_delay_ms: 0,
            auto_failover: false,
            ..FailoverConfig::default()
        };
        let mut manager = FailoverManager::new(config)
            .await
            .expect("new should succeed");

        // Fail primary 0 — standby 5 assigned.
        manager
            .trigger_failover(0)
            .await
            .expect("trigger should succeed");
        assert_eq!(manager.available_standby_count().await, 0);

        // Advance time to simulate repair window then restore.
        tokio::time::advance(tokio::time::Duration::from_secs(5)).await;

        manager
            .restore_primary(0)
            .await
            .expect("restore should succeed");

        assert_eq!(manager.get_state(0).await, FailoverState::Primary);
        assert_eq!(
            manager.available_standby_count().await,
            1,
            "standby 5 must be returned to pool after primary restore"
        );
    }

    /// Verify a second failover can be handled once the first primary is restored.
    #[tokio::test(start_paused = true)]
    async fn test_hot_standby_successive_failover_restore_cycles() {
        let config = FailoverConfig {
            auto_failover: false,
            switch_delay_ms: 0,
            ..FailoverConfig::default()
        };
        let mut manager = FailoverManager::new(config)
            .await
            .expect("new should succeed");

        for cycle in 0..3usize {
            manager
                .trigger_failover(0)
                .await
                .unwrap_or_else(|e| panic!("trigger cycle {cycle}: {e}"));
            assert_eq!(manager.get_state(0).await, FailoverState::Secondary);

            manager
                .restore_primary(0)
                .await
                .unwrap_or_else(|e| panic!("restore cycle {cycle}: {e}"));
            assert_eq!(manager.get_state(0).await, FailoverState::Primary);
        }
    }
}
