//! Preservation policy definition

use crate::{checksum::ChecksumAlgorithm, PreservationFormat};
use serde::{Deserialize, Serialize};

/// Preservation policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreservationPolicy {
    /// Policy name
    pub name: String,
    /// Allowed preservation formats
    pub allowed_formats: Vec<PreservationFormat>,
    /// Required checksum algorithms
    pub required_checksums: Vec<ChecksumAlgorithm>,
    /// Minimum versions to keep
    pub min_versions: usize,
    /// Fixity check frequency (in days)
    pub fixity_check_frequency: u32,
    /// Maximum acceptable risk level (0.0-1.0)
    pub max_risk_level: f32,
    /// Require metadata
    pub require_metadata: bool,
    /// Description
    pub description: Option<String>,
}

impl Default for PreservationPolicy {
    fn default() -> Self {
        Self {
            name: "Default Preservation Policy".to_string(),
            allowed_formats: vec![
                PreservationFormat::VideoFfv1Mkv,
                PreservationFormat::AudioFlac,
                PreservationFormat::ImageTiff,
            ],
            required_checksums: vec![ChecksumAlgorithm::Sha256],
            min_versions: 3,
            fixity_check_frequency: 90, // Quarterly
            max_risk_level: 0.3,
            require_metadata: true,
            description: Some("Default policy for media preservation".to_string()),
        }
    }
}

/// Policy builder
pub struct PolicyBuilder {
    policy: PreservationPolicy,
}

impl Default for PolicyBuilder {
    fn default() -> Self {
        Self::new("Custom Policy")
    }
}

impl PolicyBuilder {
    /// Create a new policy builder
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            policy: PreservationPolicy {
                name: name.to_string(),
                allowed_formats: Vec::new(),
                required_checksums: Vec::new(),
                min_versions: 1,
                fixity_check_frequency: 30,
                max_risk_level: 0.5,
                require_metadata: false,
                description: None,
            },
        }
    }

    /// Add allowed format
    #[must_use]
    pub fn with_format(mut self, format: PreservationFormat) -> Self {
        self.policy.allowed_formats.push(format);
        self
    }

    /// Add required checksum algorithm
    #[must_use]
    pub fn with_checksum(mut self, algorithm: ChecksumAlgorithm) -> Self {
        self.policy.required_checksums.push(algorithm);
        self
    }

    /// Set minimum versions to keep
    #[must_use]
    pub fn with_min_versions(mut self, count: usize) -> Self {
        self.policy.min_versions = count;
        self
    }

    /// Set fixity check frequency in days
    #[must_use]
    pub fn with_fixity_frequency(mut self, days: u32) -> Self {
        self.policy.fixity_check_frequency = days;
        self
    }

    /// Set maximum acceptable risk level
    #[must_use]
    pub fn with_max_risk(mut self, risk: f32) -> Self {
        self.policy.max_risk_level = risk;
        self
    }

    /// Set whether metadata is required
    #[must_use]
    pub fn with_metadata_required(mut self, required: bool) -> Self {
        self.policy.require_metadata = required;
        self
    }

    /// Set description
    #[must_use]
    pub fn with_description(mut self, description: &str) -> Self {
        self.policy.description = Some(description.to_string());
        self
    }

    /// Build the policy
    #[must_use]
    pub fn build(self) -> PreservationPolicy {
        self.policy
    }
}

impl PreservationPolicy {
    /// Save policy to JSON file
    ///
    /// # Errors
    ///
    /// Returns an error if save fails
    pub fn save(&self, path: &std::path::Path) -> crate::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| crate::Error::Metadata(format!("JSON serialization failed: {e}")))?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load policy from JSON file
    ///
    /// # Errors
    ///
    /// Returns an error if load fails
    pub fn load(path: &std::path::Path) -> crate::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let policy = serde_json::from_str(&content)
            .map_err(|e| crate::Error::Metadata(format!("JSON parse failed: {e}")))?;
        Ok(policy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_policy() {
        let policy = PreservationPolicy::default();
        assert!(!policy.allowed_formats.is_empty());
        assert!(!policy.required_checksums.is_empty());
        assert!(policy.min_versions > 0);
    }

    #[test]
    fn test_policy_builder() {
        let policy = PolicyBuilder::new("Test Policy")
            .with_format(PreservationFormat::VideoFfv1Mkv)
            .with_checksum(ChecksumAlgorithm::Sha256)
            .with_min_versions(5)
            .with_fixity_frequency(30)
            .with_metadata_required(true)
            .build();

        assert_eq!(policy.name, "Test Policy");
        assert_eq!(policy.allowed_formats.len(), 1);
        assert_eq!(policy.min_versions, 5);
        assert!(policy.require_metadata);
    }

    #[test]
    fn test_save_and_load_policy() {
        use tempfile::NamedTempFile;

        let policy = PreservationPolicy::default();
        let file = NamedTempFile::new().expect("operation should succeed");

        policy.save(file.path()).expect("operation should succeed");
        let loaded = PreservationPolicy::load(file.path()).expect("operation should succeed");

        assert_eq!(loaded.name, policy.name);
        assert_eq!(loaded.min_versions, policy.min_versions);
    }
}
