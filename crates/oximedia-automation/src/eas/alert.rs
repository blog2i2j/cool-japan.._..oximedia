//! Emergency Alert System (EAS) alert handling.

use crate::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tracing::info;

/// EAS alert type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EasAlertType {
    /// Emergency Action Notification
    EmergencyActionNotification,
    /// Required Weekly Test
    RequiredWeeklyTest,
    /// Required Monthly Test
    RequiredMonthlyTest,
    /// Tornado Warning
    TornadoWarning,
    /// Severe Thunderstorm Warning
    SevereThunderstormWarning,
    /// Flash Flood Warning
    FlashFloodWarning,
    /// Earthquake Warning
    EarthquakeWarning,
    /// Civil Emergency Message
    CivilEmergencyMessage,
    /// National Emergency Message
    NationalEmergencyMessage,
}

/// EAS alert priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AlertPriority {
    /// Low priority (tests)
    Low = 1,
    /// Medium priority (weather alerts)
    Medium = 2,
    /// High priority (severe weather)
    High = 3,
    /// Critical priority (national emergencies)
    Critical = 4,
}

/// EAS alert.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EasAlert {
    /// Alert type
    pub alert_type: EasAlertType,
    /// Alert priority
    pub priority: AlertPriority,
    /// Alert message
    pub message: String,
    /// Originator code
    pub originator: String,
    /// Event code
    pub event_code: String,
    /// Location codes
    pub locations: Vec<String>,
    /// Valid until
    pub valid_until: SystemTime,
    /// Issue time
    pub issued_at: SystemTime,
}

impl EasAlert {
    /// Create a new EAS alert.
    pub fn new(alert_type: EasAlertType, message: String, duration: Duration) -> Self {
        let priority = Self::determine_priority(alert_type);
        let now = SystemTime::now();

        Self {
            alert_type,
            priority,
            message,
            originator: "EAS".to_string(),
            event_code: Self::event_code_for_type(alert_type),
            locations: Vec::new(),
            valid_until: now + duration,
            issued_at: now,
        }
    }

    /// Determine priority from alert type.
    fn determine_priority(alert_type: EasAlertType) -> AlertPriority {
        match alert_type {
            EasAlertType::RequiredWeeklyTest | EasAlertType::RequiredMonthlyTest => {
                AlertPriority::Low
            }
            EasAlertType::SevereThunderstormWarning => AlertPriority::Medium,
            EasAlertType::TornadoWarning
            | EasAlertType::FlashFloodWarning
            | EasAlertType::EarthquakeWarning => AlertPriority::High,
            EasAlertType::EmergencyActionNotification
            | EasAlertType::CivilEmergencyMessage
            | EasAlertType::NationalEmergencyMessage => AlertPriority::Critical,
        }
    }

    /// Get event code for alert type.
    fn event_code_for_type(alert_type: EasAlertType) -> String {
        match alert_type {
            EasAlertType::EmergencyActionNotification => "EAN",
            EasAlertType::RequiredWeeklyTest => "RWT",
            EasAlertType::RequiredMonthlyTest => "RMT",
            EasAlertType::TornadoWarning => "TOR",
            EasAlertType::SevereThunderstormWarning => "SVR",
            EasAlertType::FlashFloodWarning => "FFW",
            EasAlertType::EarthquakeWarning => "EQW",
            EasAlertType::CivilEmergencyMessage => "CEM",
            EasAlertType::NationalEmergencyMessage => "NEM",
        }
        .to_string()
    }

    /// Check if alert is still valid.
    pub fn is_valid(&self) -> bool {
        SystemTime::now() < self.valid_until
    }

    /// Add location code.
    pub fn add_location(&mut self, location: String) {
        self.locations.push(location);
    }
}

/// EAS manager.
pub struct EasManager {
    active_alerts: Arc<RwLock<Vec<EasAlert>>>,
    running: Arc<RwLock<bool>>,
}

impl EasManager {
    /// Create a new EAS manager.
    pub async fn new() -> Result<Self> {
        info!("Creating EAS manager");

        Ok(Self {
            active_alerts: Arc::new(RwLock::new(Vec::new())),
            running: Arc::new(RwLock::new(false)),
        })
    }

    /// Start EAS manager.
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting EAS manager");

        {
            let mut running = self.running.write().await;
            *running = true;
        }

        // Spawn cleanup task to remove expired alerts
        let active_alerts = Arc::clone(&self.active_alerts);
        let running = Arc::clone(&self.running);

        tokio::spawn(async move {
            while *running.read().await {
                let mut alerts = active_alerts.write().await;
                let before = alerts.len();
                alerts.retain(EasAlert::is_valid);
                let after = alerts.len();

                if before != after {
                    info!("Removed {} expired EAS alerts", before - after);
                }

                tokio::time::sleep(Duration::from_secs(60)).await;
            }
        });

        Ok(())
    }

    /// Stop EAS manager.
    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping EAS manager");

        let mut running = self.running.write().await;
        *running = false;

        Ok(())
    }

    /// Handle incoming EAS alert.
    pub async fn handle_alert(&mut self, alert: EasAlert) -> Result<()> {
        info!(
            "Handling EAS alert: {:?} - {}",
            alert.alert_type, alert.message
        );

        // Add to active alerts
        let mut alerts = self.active_alerts.write().await;
        alerts.push(alert.clone());

        // Sort by priority (highest first)
        alerts.sort_by(|a, b| b.priority.cmp(&a.priority));

        Ok(())
    }

    /// Get all active alerts.
    pub async fn get_active_alerts(&self) -> Vec<EasAlert> {
        self.active_alerts.read().await.clone()
    }

    /// Get highest priority active alert.
    pub async fn get_highest_priority_alert(&self) -> Option<EasAlert> {
        let alerts = self.active_alerts.read().await;
        alerts.first().cloned()
    }

    /// Clear all alerts.
    pub async fn clear_alerts(&mut self) {
        info!("Clearing all EAS alerts");

        let mut alerts = self.active_alerts.write().await;
        alerts.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alert_creation() {
        let alert = EasAlert::new(
            EasAlertType::RequiredWeeklyTest,
            "This is a test".to_string(),
            Duration::from_secs(60),
        );

        assert_eq!(alert.event_code, "RWT");
        assert_eq!(alert.priority, AlertPriority::Low);
        assert!(alert.is_valid());
    }

    #[test]
    fn test_alert_priority() {
        let test_alert = EasAlert::new(
            EasAlertType::RequiredWeeklyTest,
            "Test".to_string(),
            Duration::from_secs(60),
        );

        let tornado_alert = EasAlert::new(
            EasAlertType::TornadoWarning,
            "Tornado".to_string(),
            Duration::from_secs(60),
        );

        assert!(tornado_alert.priority > test_alert.priority);
    }

    #[tokio::test]
    async fn test_eas_manager() {
        let mut manager = EasManager::new().await.expect("new should succeed");

        let alert = EasAlert::new(
            EasAlertType::RequiredWeeklyTest,
            "Test".to_string(),
            Duration::from_secs(60),
        );

        manager
            .handle_alert(alert)
            .await
            .expect("operation should succeed");

        let active = manager.get_active_alerts().await;
        assert_eq!(active.len(), 1);
    }
}
