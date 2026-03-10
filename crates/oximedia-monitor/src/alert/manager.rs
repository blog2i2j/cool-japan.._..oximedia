//! Alert manager for handling alerts.

use crate::alert::{Alert, AlertRule};
use crate::config::AlertConfig;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// Alert manager.
pub struct AlertManager {
    config: AlertConfig,
    rules: Arc<RwLock<Vec<AlertRule>>>,
    active_alerts: Arc<RwLock<HashMap<String, Alert>>>,
    running: Arc<tokio::sync::RwLock<bool>>,
}

impl AlertManager {
    /// Create a new alert manager.
    #[must_use]
    pub fn new(config: AlertConfig) -> Self {
        Self {
            config,
            rules: Arc::new(RwLock::new(Vec::new())),
            active_alerts: Arc::new(RwLock::new(HashMap::new())),
            running: Arc::new(tokio::sync::RwLock::new(false)),
        }
    }

    /// Start the alert manager.
    pub async fn start(&self) -> crate::error::MonitorResult<()> {
        let mut running = self.running.write().await;
        *running = true;
        Ok(())
    }

    /// Stop the alert manager.
    pub async fn stop(&self) {
        let mut running = self.running.write().await;
        *running = false;
    }

    /// Add a rule.
    pub fn add_rule(&self, rule: AlertRule) {
        self.rules.write().push(rule);
    }

    /// Fire an alert.
    pub fn fire(&self, alert: Alert) {
        self.active_alerts.write().insert(alert.id.clone(), alert);
    }

    /// Get active alerts.
    #[must_use]
    pub fn active_alerts(&self) -> Vec<Alert> {
        self.active_alerts.read().values().cloned().collect()
    }
}

/// Alert deduplicator.
pub struct AlertDeduplicator {
    seen_alerts: Arc<RwLock<HashMap<String, chrono::DateTime<chrono::Utc>>>>,
}

impl AlertDeduplicator {
    /// Create a new deduplicator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            seen_alerts: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Check if an alert should be deduplicated.
    #[must_use]
    pub fn should_deduplicate(&self, alert_id: &str, window_secs: i64) -> bool {
        let seen = self.seen_alerts.read();
        if let Some(last_seen) = seen.get(alert_id) {
            let now = chrono::Utc::now();
            let elapsed = now.signed_duration_since(*last_seen);
            elapsed.num_seconds() < window_secs
        } else {
            false
        }
    }

    /// Mark an alert as seen.
    pub fn mark_seen(&self, alert_id: String) {
        self.seen_alerts
            .write()
            .insert(alert_id, chrono::Utc::now());
    }
}

impl Default for AlertDeduplicator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_alert_manager() {
        let config = AlertConfig::default();
        let manager = AlertManager::new(config);

        manager.start().await.expect("await should be valid");

        let alert = Alert::new(
            "test",
            crate::alert::AlertSeverity::Warning,
            "Test alert",
            "test.metric",
            100.0,
        );

        manager.fire(alert);

        assert_eq!(manager.active_alerts().len(), 1);

        manager.stop().await;
    }
}
