//! Format obsolescence risk assessment

use super::RiskLevel;
use crate::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Format risk assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormatRisk {
    /// Format name
    pub format: String,
    /// Risk level
    pub risk_level: RiskLevel,
    /// Risk factors
    pub factors: Vec<String>,
    /// Recommended action
    pub recommendation: String,
    /// Assessment timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Risk assessor
pub struct RiskAssessor;

impl Default for RiskAssessor {
    fn default() -> Self {
        Self::new()
    }
}

impl RiskAssessor {
    /// Create a new risk assessor
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Assess format risk
    #[must_use]
    pub fn assess_format(&self, format: &str) -> FormatRisk {
        let (risk_level, factors, recommendation) = self.evaluate_format(format);

        FormatRisk {
            format: format.to_string(),
            risk_level,
            factors,
            recommendation,
            timestamp: chrono::Utc::now(),
        }
    }

    /// Assess risk for a file
    ///
    /// # Errors
    ///
    /// Returns an error if format cannot be determined
    pub fn assess_file(&self, path: &Path) -> Result<FormatRisk> {
        let format = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("unknown");

        Ok(self.assess_format(format))
    }

    fn evaluate_format(&self, format: &str) -> (RiskLevel, Vec<String>, String) {
        let format_lower = format.to_lowercase();

        match format_lower.as_str() {
            // Critical risk - proprietary/obsolete
            "wmv" | "asf" | "rm" | "flv" | "3gp" => (
                RiskLevel::Critical,
                vec![
                    "Proprietary format".to_string(),
                    "Limited decoder support".to_string(),
                    "Patent encumbered".to_string(),
                ],
                "Migrate immediately to preservation format (FFV1/Matroska)".to_string(),
            ),

            // High risk
            "avi" | "mov" => (
                RiskLevel::High,
                vec![
                    "Container may contain patent-encumbered codecs".to_string(),
                    "Limited long-term support".to_string(),
                ],
                "Verify codec and consider migration to MKV container".to_string(),
            ),

            // Medium risk
            "mp4" | "m4v" | "h264" | "h265" => (
                RiskLevel::Medium,
                vec![
                    "Patent encumbered codecs".to_string(),
                    "Licensing requirements".to_string(),
                ],
                "Plan migration to patent-free formats (AV1, VP9)".to_string(),
            ),

            // Low risk - open but not ideal for preservation
            "webm" | "vp8" | "vp9" => (
                RiskLevel::Low,
                vec!["Open format but lossy compression".to_string()],
                "Consider lossless preservation master (FFV1)".to_string(),
            ),

            // No risk - preservation formats
            "mkv" | "ffv1" | "flac" | "wav" | "tiff" | "png" => (
                RiskLevel::None,
                vec!["Suitable for long-term preservation".to_string()],
                "Continue current preservation strategy".to_string(),
            ),

            // Unknown format
            _ => (
                RiskLevel::Medium,
                vec![format!("Unknown format: {}", format)],
                "Investigate format specifications and viability".to_string(),
            ),
        }
    }

    /// Assess risk for multiple files
    #[must_use]
    pub fn assess_batch(&self, paths: &[std::path::PathBuf]) -> Vec<FormatRisk> {
        paths
            .iter()
            .filter_map(|p| self.assess_file(p).ok())
            .collect()
    }

    /// Get high-risk assessments
    #[must_use]
    pub fn filter_high_risk(assessments: &[FormatRisk]) -> Vec<&FormatRisk> {
        assessments
            .iter()
            .filter(|a| a.risk_level >= RiskLevel::High)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_assess_critical_format() {
        let assessor = RiskAssessor::new();
        let risk = assessor.assess_format("wmv");

        assert_eq!(risk.risk_level, RiskLevel::Critical);
        assert!(!risk.factors.is_empty());
        assert!(risk.recommendation.contains("Migrate"));
    }

    #[test]
    fn test_assess_preservation_format() {
        let assessor = RiskAssessor::new();
        let risk = assessor.assess_format("mkv");

        assert_eq!(risk.risk_level, RiskLevel::None);
    }

    #[test]
    fn test_assess_file() {
        let assessor = RiskAssessor::new();
        let risk = assessor
            .assess_file(&PathBuf::from("test.flv"))
            .expect("operation should succeed");

        assert_eq!(risk.risk_level, RiskLevel::Critical);
    }

    #[test]
    fn test_filter_high_risk() {
        let assessments = vec![
            FormatRisk {
                format: "mkv".to_string(),
                risk_level: RiskLevel::None,
                factors: Vec::new(),
                recommendation: String::new(),
                timestamp: chrono::Utc::now(),
            },
            FormatRisk {
                format: "wmv".to_string(),
                risk_level: RiskLevel::Critical,
                factors: Vec::new(),
                recommendation: String::new(),
                timestamp: chrono::Utc::now(),
            },
        ];

        let high_risk = RiskAssessor::filter_high_risk(&assessments);
        assert_eq!(high_risk.len(), 1);
        assert_eq!(high_risk[0].format, "wmv");
    }
}
