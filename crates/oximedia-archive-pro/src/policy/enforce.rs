//! Policy enforcement

use super::PreservationPolicy;
use crate::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Policy violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyViolation {
    /// File path
    pub path: PathBuf,
    /// Violation type
    pub violation_type: String,
    /// Description
    pub description: String,
    /// Severity (0-10)
    pub severity: u8,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Policy enforcer
pub struct PolicyEnforcer {
    policy: PreservationPolicy,
}

impl PolicyEnforcer {
    /// Create a new policy enforcer
    #[must_use]
    pub fn new(policy: PreservationPolicy) -> Self {
        Self { policy }
    }

    /// Check if a file complies with policy
    ///
    /// # Errors
    ///
    /// Returns an error if check fails
    pub fn check_compliance(&self, file_path: &Path) -> Result<Vec<PolicyViolation>> {
        let mut violations = Vec::new();

        // Check format compliance
        if let Some(ext) = file_path.extension().and_then(|e| e.to_str()) {
            if !self.is_format_allowed(ext) {
                violations.push(PolicyViolation {
                    path: file_path.to_path_buf(),
                    violation_type: "Format".to_string(),
                    description: format!("Format '{ext}' not in allowed list"),
                    severity: 8,
                    timestamp: chrono::Utc::now(),
                });
            }
        }

        // Check metadata requirement
        if self.policy.require_metadata {
            let metadata_file = file_path.with_extension("json");
            if !metadata_file.exists() {
                violations.push(PolicyViolation {
                    path: file_path.to_path_buf(),
                    violation_type: "Metadata".to_string(),
                    description: "Required metadata file not found".to_string(),
                    severity: 6,
                    timestamp: chrono::Utc::now(),
                });
            }
        }

        Ok(violations)
    }

    fn is_format_allowed(&self, extension: &str) -> bool {
        self.policy
            .allowed_formats
            .iter()
            .any(|f| f.extension().eq_ignore_ascii_case(extension))
    }

    /// Check compliance for multiple files
    ///
    /// # Errors
    ///
    /// Returns an error if batch check fails
    pub fn check_batch(&self, files: &[PathBuf]) -> Result<Vec<PolicyViolation>> {
        let mut all_violations = Vec::new();

        for file in files {
            let violations = self.check_compliance(file)?;
            all_violations.extend(violations);
        }

        Ok(all_violations)
    }

    /// Get high-severity violations
    #[must_use]
    pub fn get_high_severity_violations(violations: &[PolicyViolation]) -> Vec<&PolicyViolation> {
        violations.iter().filter(|v| v.severity >= 7).collect()
    }

    /// Generate compliance report
    #[must_use]
    pub fn generate_report(violations: &[PolicyViolation]) -> String {
        let mut report = String::from("Policy Compliance Report\n");
        report.push_str("========================\n\n");

        if violations.is_empty() {
            report.push_str("All files comply with policy.\n");
        } else {
            report.push_str(&format!("Total violations: {}\n\n", violations.len()));

            let high_severity = Self::get_high_severity_violations(violations);
            report.push_str(&format!(
                "High severity violations: {}\n\n",
                high_severity.len()
            ));

            report.push_str("Violations:\n");
            for (i, violation) in violations.iter().enumerate() {
                report.push_str(&format!(
                    "\n{}. {} [Severity: {}]\n",
                    i + 1,
                    violation.path.display(),
                    violation.severity
                ));
                report.push_str(&format!("   Type: {}\n", violation.violation_type));
                report.push_str(&format!("   Description: {}\n", violation.description));
            }
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PreservationFormat;

    #[test]
    fn test_check_compliance() {
        let policy = PreservationPolicy {
            name: "Test".to_string(),
            allowed_formats: vec![PreservationFormat::VideoFfv1Mkv],
            required_checksums: Vec::new(),
            min_versions: 1,
            fixity_check_frequency: 30,
            max_risk_level: 0.5,
            require_metadata: false,
            description: None,
        };

        let enforcer = PolicyEnforcer::new(policy);

        // Compliant file
        let violations = enforcer
            .check_compliance(&PathBuf::from("test.mkv"))
            .expect("operation should succeed");
        assert_eq!(violations.len(), 0);

        // Non-compliant file
        let violations = enforcer
            .check_compliance(&PathBuf::from("test.wmv"))
            .expect("operation should succeed");
        assert!(violations.len() > 0);
    }

    #[test]
    fn test_metadata_requirement() {
        let policy = PreservationPolicy {
            name: "Test".to_string(),
            allowed_formats: vec![PreservationFormat::VideoFfv1Mkv],
            required_checksums: Vec::new(),
            min_versions: 1,
            fixity_check_frequency: 30,
            max_risk_level: 0.5,
            require_metadata: true,
            description: None,
        };

        let enforcer = PolicyEnforcer::new(policy);
        let violations = enforcer
            .check_compliance(&PathBuf::from("test.mkv"))
            .expect("operation should succeed");

        // Should have violation for missing metadata
        assert!(violations.iter().any(|v| v.violation_type == "Metadata"));
    }

    #[test]
    fn test_generate_report() {
        let violations = vec![PolicyViolation {
            path: PathBuf::from("test.wmv"),
            violation_type: "Format".to_string(),
            description: "Not allowed".to_string(),
            severity: 8,
            timestamp: chrono::Utc::now(),
        }];

        let report = PolicyEnforcer::generate_report(&violations);
        assert!(report.contains("Total violations: 1"));
        assert!(report.contains("test.wmv"));
    }
}
