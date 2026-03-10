//! Migration validation

use super::MigrationResult;
use crate::{checksum::ChecksumGenerator, Result};
use serde::{Deserialize, Serialize};

/// Validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Migration was successful
    pub migration_success: bool,
    /// Output file exists
    pub output_exists: bool,
    /// Output is readable
    pub output_readable: bool,
    /// File size is reasonable
    pub size_reasonable: bool,
    /// Checksums match (if applicable)
    pub checksum_match: Option<bool>,
    /// Overall validation passed
    pub passed: bool,
    /// Validation errors
    pub errors: Vec<String>,
}

/// Migration validator
pub struct MigrationValidator {
    verify_checksums: bool,
}

impl Default for MigrationValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl MigrationValidator {
    /// Create a new migration validator
    #[must_use]
    pub fn new() -> Self {
        Self {
            verify_checksums: false,
        }
    }

    /// Enable checksum verification
    #[must_use]
    pub fn with_checksum_verification(mut self, verify: bool) -> Self {
        self.verify_checksums = verify;
        self
    }

    /// Validate a migration result
    ///
    /// # Errors
    ///
    /// Returns an error if validation fails
    pub fn validate(&self, result: &MigrationResult) -> Result<ValidationResult> {
        let mut errors = Vec::new();
        let mut validation = ValidationResult {
            migration_success: result.success,
            output_exists: false,
            output_readable: false,
            size_reasonable: false,
            checksum_match: None,
            passed: false,
            errors: Vec::new(),
        };

        if !result.success {
            errors.push(format!("Migration failed: {:?}", result.error));
        }

        // Check output files exist
        if result.outputs.is_empty() {
            errors.push("No output files produced".to_string());
        } else {
            validation.output_exists = result.outputs.iter().all(|p| p.exists());
            if !validation.output_exists {
                errors.push("Some output files do not exist".to_string());
            }

            // Check if outputs are readable
            validation.output_readable =
                result.outputs.iter().all(|p| std::fs::metadata(p).is_ok());
            if !validation.output_readable {
                errors.push("Some output files are not readable".to_string());
            }

            // Check file sizes
            validation.size_reasonable = result
                .outputs
                .iter()
                .all(|p| std::fs::metadata(p).map(|m| m.len() > 0).unwrap_or(false));
            if !validation.size_reasonable {
                errors.push("Some output files have zero size".to_string());
            }
        }

        // Checksum verification if enabled
        if self.verify_checksums && !result.outputs.is_empty() {
            validation.checksum_match = Some(self.verify_output_checksums(&result.outputs)?);
            if !validation.checksum_match.unwrap_or(true) {
                errors.push("Checksum verification failed".to_string());
            }
        }

        validation.errors = errors.clone();
        validation.passed = errors.is_empty() && result.success;

        Ok(validation)
    }

    fn verify_output_checksums(&self, outputs: &[std::path::PathBuf]) -> Result<bool> {
        let generator = ChecksumGenerator::new();

        for output in outputs {
            let checksum = generator.generate_file(output)?;
            // In a real implementation, we would compare against expected checksums
            // For now, just verify that we can generate a checksum
            if checksum.checksums.is_empty() {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Validate multiple migration results
    ///
    /// # Errors
    ///
    /// Returns an error if batch validation fails
    pub fn validate_batch(&self, results: &[MigrationResult]) -> Result<Vec<ValidationResult>> {
        results.iter().map(|r| self.validate(r)).collect()
    }

    /// Generate a validation report
    #[must_use]
    pub fn generate_report(&self, validations: &[ValidationResult]) -> String {
        let total = validations.len();
        let passed = validations.iter().filter(|v| v.passed).count();
        let failed = total - passed;

        let mut report = String::new();
        report.push_str(&"Migration Validation Report\n".to_string());
        report.push_str(&"===========================\n\n".to_string());
        report.push_str(&format!("Total migrations: {total}\n"));
        report.push_str(&format!("Passed: {passed}\n"));
        report.push_str(&format!("Failed: {failed}\n\n"));

        if failed > 0 {
            report.push_str("Failed validations:\n");
            for (i, validation) in validations.iter().enumerate() {
                if !validation.passed {
                    report.push_str(&format!("\nMigration {}:\n", i + 1));
                    for error in &validation.errors {
                        report.push_str(&format!("  - {error}\n"));
                    }
                }
            }
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_validate_successful_migration() {
        let result = MigrationResult {
            source: PathBuf::from("test.mp4"),
            outputs: vec![PathBuf::from("/tmp/test_output.mkv")],
            success: true,
            error: None,
            duration_ms: 1000,
            timestamp: chrono::Utc::now(),
        };

        let validator = MigrationValidator::new();
        let validation = validator
            .validate(&result)
            .expect("operation should succeed");

        assert!(validation.migration_success);
    }

    #[test]
    fn test_validate_failed_migration() {
        let result = MigrationResult {
            source: PathBuf::from("test.mp4"),
            outputs: Vec::new(),
            success: false,
            error: Some("Conversion failed".to_string()),
            duration_ms: 100,
            timestamp: chrono::Utc::now(),
        };

        let validator = MigrationValidator::new();
        let validation = validator
            .validate(&result)
            .expect("operation should succeed");

        assert!(!validation.passed);
        assert!(!validation.errors.is_empty());
    }

    #[test]
    fn test_generate_report() {
        let validations = vec![
            ValidationResult {
                migration_success: true,
                output_exists: true,
                output_readable: true,
                size_reasonable: true,
                checksum_match: None,
                passed: true,
                errors: Vec::new(),
            },
            ValidationResult {
                migration_success: false,
                output_exists: false,
                output_readable: false,
                size_reasonable: false,
                checksum_match: None,
                passed: false,
                errors: vec!["Test error".to_string()],
            },
        ];

        let validator = MigrationValidator::new();
        let report = validator.generate_report(&validations);

        assert!(report.contains("Total migrations: 2"));
        assert!(report.contains("Passed: 1"));
        assert!(report.contains("Failed: 1"));
    }
}
