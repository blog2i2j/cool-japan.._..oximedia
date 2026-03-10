//! Risk alert system

use super::{FormatRisk, RiskLevel};
use serde::{Deserialize, Serialize};

/// Alert level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskAlertLevel {
    /// Informational
    Info,
    /// Warning
    Warning,
    /// Error - action required
    Error,
}

/// Risk alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAlert {
    /// Alert level
    pub level: RiskAlertLevel,
    /// Format
    pub format: String,
    /// Message
    pub message: String,
    /// Recommended action
    pub action: String,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl RiskAlert {
    /// Create alert from format risk assessment
    #[must_use]
    pub fn from_risk(risk: &FormatRisk) -> Option<Self> {
        let (level, message) = match risk.risk_level {
            RiskLevel::Critical => (
                RiskAlertLevel::Error,
                format!(
                    "CRITICAL: Format {} requires immediate migration",
                    risk.format
                ),
            ),
            RiskLevel::High => (
                RiskAlertLevel::Error,
                format!("HIGH RISK: Format {} should be migrated soon", risk.format),
            ),
            RiskLevel::Medium => (
                RiskAlertLevel::Warning,
                format!("MEDIUM RISK: Consider migrating format {}", risk.format),
            ),
            RiskLevel::Low => (
                RiskAlertLevel::Info,
                format!("Low risk for format {}", risk.format),
            ),
            RiskLevel::None => return None,
        };

        Some(Self {
            level,
            format: risk.format.clone(),
            message,
            action: risk.recommendation.clone(),
            timestamp: chrono::Utc::now(),
        })
    }

    /// Generate multiple alerts from risks
    #[must_use]
    pub fn from_risks(risks: &[FormatRisk]) -> Vec<Self> {
        risks.iter().filter_map(Self::from_risk).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_alert_from_critical_risk() {
        let risk = FormatRisk {
            format: "wmv".to_string(),
            risk_level: RiskLevel::Critical,
            factors: Vec::new(),
            recommendation: "Migrate now".to_string(),
            timestamp: chrono::Utc::now(),
        };

        let alert = RiskAlert::from_risk(&risk).expect("operation should succeed");
        assert_eq!(alert.level, RiskAlertLevel::Error);
        assert!(alert.message.contains("CRITICAL"));
    }

    #[test]
    fn test_no_alert_for_safe_format() {
        let risk = FormatRisk {
            format: "mkv".to_string(),
            risk_level: RiskLevel::None,
            factors: Vec::new(),
            recommendation: String::new(),
            timestamp: chrono::Utc::now(),
        };

        assert!(RiskAlert::from_risk(&risk).is_none());
    }
}
