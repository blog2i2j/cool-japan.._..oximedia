//! Emergency Alert System (EAS) alert handling.

use crate::{AutomationError, Result};
use quick_xml::events::Event;
use quick_xml::Reader as XmlReader;
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

// ─────────────────────────────────────────────────────────────────────────────
// EAS playout controller — tracks audio/video interruption and restoration
// ─────────────────────────────────────────────────────────────────────────────

/// Describes which content type is currently on-air.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputType {
    /// Regular programme content.
    Normal,
    /// EAS alert is interrupting normal content.
    Eas,
}

/// Lightweight controller that tracks whether an EAS alert is currently
/// interrupting normal playout.
///
/// This struct uses simulated elapsed time to determine whether the active
/// alert has expired, modelling the audio/video restoration that occurs when
/// an EAS break ends.  It is deliberately synchronous so tests can exercise
/// it without a Tokio runtime.
pub struct EasPlayoutController {
    /// Queued alerts: (alert, duration after which it expires).
    alerts: Vec<(EasAlert, Duration)>,
    /// Content identifier for the background programme.
    background: String,
    /// Simulated elapsed time (advanced by `advance_time`).
    elapsed: Duration,
}

impl EasPlayoutController {
    /// Create a new playout controller.
    pub fn new() -> Self {
        Self {
            alerts: Vec::new(),
            background: String::new(),
            elapsed: Duration::ZERO,
        }
    }

    /// Set the background programme content identifier.
    pub fn set_background_content(&mut self, content: &str) {
        self.background = content.to_string();
    }

    /// Insert an EAS alert.  Higher-priority alerts pre-empt lower-priority
    /// ones.  The alert is valid for `duration` of simulated time from the
    /// moment it is inserted, measured from the current simulated clock.
    pub fn insert_alert(&mut self, alert: EasAlert, duration: Duration) {
        // Store the absolute simulated expiry time.
        let expires_at = self.elapsed + duration;
        self.alerts.push((alert, expires_at));
        // Sort by priority descending (highest first).
        self.alerts.sort_by(|a, b| b.0.priority.cmp(&a.0.priority));
    }

    /// Advance the simulated clock by `delta`.
    ///
    /// Any alerts whose simulated expiry time has passed are removed.
    pub fn advance_time(&mut self, delta: Duration) {
        self.elapsed += delta;
        self.alerts
            .retain(|(_, expires_at)| self.elapsed < *expires_at);
    }

    /// Return the current output type based on whether any alert is active.
    pub fn current_output(&self) -> OutputType {
        if self.alerts.is_empty() {
            OutputType::Normal
        } else {
            OutputType::Eas
        }
    }

    /// Return the highest-priority active alert, if any.
    pub fn highest_priority_alert(&self) -> Option<&EasAlert> {
        self.alerts.first().map(|(a, _)| a)
    }

    /// Return the current background content identifier.
    pub fn background_content(&self) -> &str {
        &self.background
    }

    /// Return the number of active alerts.
    pub fn active_alert_count(&self) -> usize {
        self.alerts.len()
    }
}

impl Default for EasPlayoutController {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CAP (Common Alerting Protocol) XML parsing
// ─────────────────────────────────────────────────────────────────────────────

/// A parsed CAP (Common Alerting Protocol) alert message.
///
/// Follows the OASIS CAP 1.2 standard structure.  Fields correspond to the
/// top-level `<alert>` element and the first `<info>` sub-element.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapAlert {
    /// Unique identifier of this alert message.
    pub identifier: String,
    /// Identifier of the originator of this alert.
    pub sender: String,
    /// Time and date of the origination (ISO 8601 string as-received).
    pub sent: String,
    /// The appropriate handling of the alert (e.g. "Actual", "Test").
    pub status: String,
    /// The nature of the alert (e.g. "Alert", "Cancel", "Update").
    pub msg_type: String,
    /// Urgency of the subject event (e.g. "Immediate", "Expected").
    pub urgency: String,
    /// Severity of the subject event (e.g. "Extreme", "Severe").
    pub severity: String,
    /// Certainty of the subject event (e.g. "Observed", "Likely").
    pub certainty: String,
    /// Human-readable description of the event.
    pub description: String,
}

impl CapAlert {
    /// Create a new blank `CapAlert` with all fields empty.
    pub fn empty() -> Self {
        Self {
            identifier: String::new(),
            sender: String::new(),
            sent: String::new(),
            status: String::new(),
            msg_type: String::new(),
            urgency: String::new(),
            severity: String::new(),
            certainty: String::new(),
            description: String::new(),
        }
    }
}

/// Parse a CAP 1.2 XML document and return a [`CapAlert`].
///
/// Extracts the first `<info>` block's `urgency`, `severity`, `certainty`, and
/// `description` fields together with the top-level alert metadata.
///
/// # Errors
///
/// Returns [`AutomationError::Eas`] if the XML is malformed or a required field
/// is absent.
pub fn parse_cap_xml(xml: &str) -> Result<CapAlert> {
    let mut reader = XmlReader::from_str(xml);
    reader.config_mut().trim_text(true);

    let mut alert = CapAlert::empty();
    // Track which element we are currently collecting text for.
    let mut current_tag: Option<String> = None;

    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => {
                let tag = std::str::from_utf8(e.local_name().as_ref())
                    .map_err(|err| AutomationError::Eas(format!("Invalid XML tag UTF-8: {err}")))?
                    .to_owned();
                current_tag = Some(tag);
            }
            Ok(Event::Text(ref e)) => {
                let text = reader
                    .decoder()
                    .decode(e.as_ref())
                    .map_err(|err| AutomationError::Eas(format!("XML decode error: {err}")))?
                    .into_owned();
                if let Some(ref tag) = current_tag {
                    match tag.as_str() {
                        "identifier" => alert.identifier = text,
                        "sender" => alert.sender = text,
                        "sent" => alert.sent = text,
                        "status" => alert.status = text,
                        "msgType" => alert.msg_type = text,
                        "urgency" => {
                            if alert.urgency.is_empty() {
                                alert.urgency = text;
                            }
                        }
                        "severity" => {
                            if alert.severity.is_empty() {
                                alert.severity = text;
                            }
                        }
                        "certainty" => {
                            if alert.certainty.is_empty() {
                                alert.certainty = text;
                            }
                        }
                        "description" => {
                            if alert.description.is_empty() {
                                alert.description = text;
                            }
                        }
                        _ => {}
                    }
                }
            }
            Ok(Event::End(_)) => {
                current_tag = None;
            }
            Ok(Event::Eof) => break,
            Err(err) => {
                return Err(AutomationError::Eas(format!("CAP XML parse error: {err}")));
            }
            _ => {}
        }
        buf.clear();
    }

    if alert.identifier.is_empty() {
        return Err(AutomationError::Eas(
            "CAP XML missing required <identifier> element".to_string(),
        ));
    }

    Ok(alert)
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

    // ─── EasPlayoutController tests ─────────────────────────────────────────

    #[test]
    fn eas_playout_controller_starts_in_normal_mode() {
        let ctrl = EasPlayoutController::new();
        assert_eq!(ctrl.current_output(), OutputType::Normal);
        assert_eq!(ctrl.active_alert_count(), 0);
    }

    #[test]
    fn eas_alert_interrupts_playout() {
        let mut ctrl = EasPlayoutController::new();
        ctrl.set_background_content("normal_show");
        assert_eq!(ctrl.background_content(), "normal_show");
        assert_eq!(ctrl.current_output(), OutputType::Normal);

        // Insert a Critical EAS alert lasting 60 simulated seconds.
        let alert = EasAlert::new(
            EasAlertType::NationalEmergencyMessage,
            "National emergency!".to_string(),
            Duration::from_secs(60),
        );
        ctrl.insert_alert(alert, Duration::from_secs(60));

        assert_eq!(
            ctrl.current_output(),
            OutputType::Eas,
            "output must switch to EAS after alert insertion"
        );
    }

    #[test]
    fn eas_alert_restoration_after_duration() {
        let mut ctrl = EasPlayoutController::new();
        ctrl.set_background_content("normal_show");

        let alert = EasAlert::new(
            EasAlertType::TornadoWarning,
            "Tornado warning!".to_string(),
            Duration::from_secs(30),
        );
        ctrl.insert_alert(alert, Duration::from_secs(30));
        assert_eq!(ctrl.current_output(), OutputType::Eas);

        // Advance past the alert's duration.
        ctrl.advance_time(Duration::from_secs(31));
        assert_eq!(
            ctrl.current_output(),
            OutputType::Normal,
            "output must restore to Normal after alert expires"
        );
    }

    #[test]
    fn eas_higher_priority_alert_preempts_lower() {
        let mut ctrl = EasPlayoutController::new();

        let low = EasAlert::new(
            EasAlertType::RequiredWeeklyTest,
            "Weekly test".to_string(),
            Duration::from_secs(60),
        );
        let high = EasAlert::new(
            EasAlertType::EmergencyActionNotification,
            "Emergency!".to_string(),
            Duration::from_secs(60),
        );

        ctrl.insert_alert(low, Duration::from_secs(60));
        ctrl.insert_alert(high.clone(), Duration::from_secs(60));

        let top = ctrl
            .highest_priority_alert()
            .expect("at least one alert active");
        assert_eq!(
            top.priority,
            AlertPriority::Critical,
            "Critical alert should be first in queue"
        );
        assert_eq!(top.event_code, high.event_code);
    }

    #[test]
    fn eas_priority_ordering_extreme_to_low() {
        // AlertPriority uses integer discriminants: Low=1, Medium=2, High=3, Critical=4
        assert!(AlertPriority::Critical > AlertPriority::High);
        assert!(AlertPriority::High > AlertPriority::Medium);
        assert!(AlertPriority::Medium > AlertPriority::Low);
    }

    #[test]
    fn eas_multiple_alerts_queued_highest_shown_first() {
        let mut ctrl = EasPlayoutController::new();

        for alert_type in [
            EasAlertType::RequiredMonthlyTest,
            EasAlertType::SevereThunderstormWarning,
            EasAlertType::TornadoWarning,
            EasAlertType::CivilEmergencyMessage,
        ] {
            let a = EasAlert::new(alert_type, "msg".to_string(), Duration::from_secs(60));
            ctrl.insert_alert(a, Duration::from_secs(60));
        }

        assert_eq!(ctrl.active_alert_count(), 4);
        let top = ctrl.highest_priority_alert().expect("alerts present");
        assert_eq!(top.priority, AlertPriority::Critical);
    }

    #[test]
    fn eas_lower_priority_expires_higher_remains() {
        let mut ctrl = EasPlayoutController::new();

        // Low-priority alert expires after 10 s.
        let low = EasAlert::new(
            EasAlertType::RequiredWeeklyTest,
            "test".to_string(),
            Duration::from_secs(10),
        );
        // High-priority alert lasts 60 s.
        let high = EasAlert::new(
            EasAlertType::FlashFloodWarning,
            "flood".to_string(),
            Duration::from_secs(60),
        );

        ctrl.insert_alert(low, Duration::from_secs(10));
        ctrl.insert_alert(high, Duration::from_secs(60));
        assert_eq!(ctrl.active_alert_count(), 2);

        // Advance past low-priority expiry but before high-priority.
        ctrl.advance_time(Duration::from_secs(11));
        assert_eq!(
            ctrl.active_alert_count(),
            1,
            "low-priority alert should have expired"
        );
        assert_eq!(
            ctrl.current_output(),
            OutputType::Eas,
            "high-priority still active"
        );
    }

    #[test]
    fn eas_all_alerts_expire_restores_to_normal() {
        let mut ctrl = EasPlayoutController::new();
        ctrl.set_background_content("show");

        let a = EasAlert::new(
            EasAlertType::TornadoWarning,
            "Tornado".to_string(),
            Duration::from_secs(5),
        );
        ctrl.insert_alert(a, Duration::from_secs(5));
        assert_eq!(ctrl.current_output(), OutputType::Eas);

        ctrl.advance_time(Duration::from_secs(6));
        assert_eq!(ctrl.current_output(), OutputType::Normal);
        assert_eq!(ctrl.background_content(), "show");
    }

    #[tokio::test]
    async fn eas_manager_priority_sort_on_handle() {
        let mut manager = EasManager::new().await.expect("new should succeed");

        let low = EasAlert::new(
            EasAlertType::RequiredWeeklyTest,
            "Test".to_string(),
            Duration::from_secs(120),
        );
        let critical = EasAlert::new(
            EasAlertType::NationalEmergencyMessage,
            "National emergency".to_string(),
            Duration::from_secs(120),
        );

        manager.handle_alert(low).await.expect("handle low");
        manager
            .handle_alert(critical)
            .await
            .expect("handle critical");

        let top = manager
            .get_highest_priority_alert()
            .await
            .expect("should have an alert");
        assert_eq!(top.priority, AlertPriority::Critical);
    }

    #[test]
    fn test_cap_xml_parse() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
    <identifier>WXUS55-KPHI-202403011200</identifier>
    <sender>nws-alerts@noaa.gov</sender>
    <sent>2024-03-01T12:00:00-05:00</sent>
    <status>Actual</status>
    <msgType>Alert</msgType>
    <info>
        <urgency>Immediate</urgency>
        <severity>Extreme</severity>
        <certainty>Observed</certainty>
        <description>A tornado has been spotted in the area. Take shelter immediately.</description>
    </info>
</alert>"#;

        let cap = parse_cap_xml(xml).expect("CAP XML should parse successfully");
        assert_eq!(cap.identifier, "WXUS55-KPHI-202403011200");
        assert_eq!(cap.sender, "nws-alerts@noaa.gov");
        assert_eq!(cap.sent, "2024-03-01T12:00:00-05:00");
        assert_eq!(cap.status, "Actual");
        assert_eq!(cap.msg_type, "Alert");
        assert_eq!(cap.urgency, "Immediate");
        assert_eq!(cap.severity, "Extreme");
        assert_eq!(cap.certainty, "Observed");
        assert!(!cap.description.is_empty());
    }

    #[test]
    fn test_cap_xml_parse_missing_identifier() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
    <sender>test@example.com</sender>
    <sent>2024-01-01T00:00:00Z</sent>
    <status>Test</status>
    <msgType>Alert</msgType>
    <info>
        <urgency>Minor</urgency>
        <severity>Minor</severity>
        <certainty>Unlikely</certainty>
        <description>Test alert with no identifier</description>
    </info>
</alert>"#;

        let result = parse_cap_xml(xml);
        assert!(result.is_err(), "Should fail when identifier is missing");
    }

    #[test]
    fn test_cap_xml_parse_malformed() {
        let xml = "this is not xml at all <<<";
        // quick-xml may or may not fail on this as an error; either way,
        // the identifier will be absent and we should get an error.
        let result = parse_cap_xml(xml);
        assert!(result.is_err(), "Malformed XML should produce an error");
    }
}
