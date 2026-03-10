//! Job presets and templates

use crate::job::{BatchJob, BatchOperation};
use crate::operations::{AnalysisType, FileOperation};
use crate::types::{Priority, RetryPolicy};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Job preset definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobPreset {
    /// Preset name
    pub name: String,
    /// Description
    pub description: String,
    /// Operation
    pub operation: BatchOperation,
    /// Default priority
    pub priority: Priority,
    /// Retry policy
    pub retry: RetryPolicy,
    /// Parameters
    pub parameters: HashMap<String, String>,
}

impl JobPreset {
    /// Create a job from this preset
    #[must_use]
    pub fn create_job(&self, name: String) -> BatchJob {
        let mut job = BatchJob::new(name, self.operation.clone());
        job.set_priority(self.priority);
        job.set_retry_policy(self.retry.clone());
        job
    }
}

/// Preset manager
pub struct PresetManager {
    presets: HashMap<String, JobPreset>,
}

impl PresetManager {
    /// Create a new preset manager
    #[must_use]
    pub fn new() -> Self {
        let mut manager = Self {
            presets: HashMap::new(),
        };

        // Load built-in presets
        manager.load_builtin_presets();

        manager
    }

    #[allow(clippy::too_many_lines)]
    fn load_builtin_presets(&mut self) {
        // Web transcoding
        self.presets.insert(
            "transcode-web".to_string(),
            JobPreset {
                name: "transcode-web".to_string(),
                description: "Transcode for web delivery".to_string(),
                operation: BatchOperation::Transcode {
                    preset: "web".to_string(),
                },
                priority: Priority::Normal,
                retry: RetryPolicy::default(),
                parameters: HashMap::new(),
            },
        );

        // Mobile transcoding
        self.presets.insert(
            "transcode-mobile".to_string(),
            JobPreset {
                name: "transcode-mobile".to_string(),
                description: "Transcode for mobile devices".to_string(),
                operation: BatchOperation::Transcode {
                    preset: "mobile".to_string(),
                },
                priority: Priority::Normal,
                retry: RetryPolicy::default(),
                parameters: HashMap::new(),
            },
        );

        // Broadcast transcoding
        self.presets.insert(
            "transcode-broadcast".to_string(),
            JobPreset {
                name: "transcode-broadcast".to_string(),
                description: "Transcode for broadcast delivery".to_string(),
                operation: BatchOperation::Transcode {
                    preset: "broadcast".to_string(),
                },
                priority: Priority::High,
                retry: RetryPolicy::default(),
                parameters: HashMap::new(),
            },
        );

        // Quality control
        self.presets.insert(
            "qc-full".to_string(),
            JobPreset {
                name: "qc-full".to_string(),
                description: "Full quality control check".to_string(),
                operation: BatchOperation::QualityCheck {
                    profile: "full".to_string(),
                },
                priority: Priority::Normal,
                retry: RetryPolicy::default(),
                parameters: HashMap::new(),
            },
        );

        // QC quick
        self.presets.insert(
            "qc-quick".to_string(),
            JobPreset {
                name: "qc-quick".to_string(),
                description: "Quick quality control check".to_string(),
                operation: BatchOperation::QualityCheck {
                    profile: "quick".to_string(),
                },
                priority: Priority::Normal,
                retry: RetryPolicy::default(),
                parameters: HashMap::new(),
            },
        );

        // Video analysis
        self.presets.insert(
            "analyze-video".to_string(),
            JobPreset {
                name: "analyze-video".to_string(),
                description: "Analyze video quality".to_string(),
                operation: BatchOperation::Analyze {
                    analysis_type: AnalysisType::VideoQuality,
                },
                priority: Priority::Low,
                retry: RetryPolicy::default(),
                parameters: HashMap::new(),
            },
        );

        // Audio analysis
        self.presets.insert(
            "analyze-audio".to_string(),
            JobPreset {
                name: "analyze-audio".to_string(),
                description: "Analyze audio levels".to_string(),
                operation: BatchOperation::Analyze {
                    analysis_type: AnalysisType::AudioLevel,
                },
                priority: Priority::Low,
                retry: RetryPolicy::default(),
                parameters: HashMap::new(),
            },
        );

        // Scene detection
        self.presets.insert(
            "detect-scenes".to_string(),
            JobPreset {
                name: "detect-scenes".to_string(),
                description: "Detect scene changes".to_string(),
                operation: BatchOperation::Analyze {
                    analysis_type: AnalysisType::SceneDetection,
                },
                priority: Priority::Low,
                retry: RetryPolicy::default(),
                parameters: HashMap::new(),
            },
        );

        // Black frame detection
        self.presets.insert(
            "detect-black".to_string(),
            JobPreset {
                name: "detect-black".to_string(),
                description: "Detect black frames".to_string(),
                operation: BatchOperation::Analyze {
                    analysis_type: AnalysisType::BlackFrameDetection,
                },
                priority: Priority::Low,
                retry: RetryPolicy::default(),
                parameters: HashMap::new(),
            },
        );

        // Silence detection
        self.presets.insert(
            "detect-silence".to_string(),
            JobPreset {
                name: "detect-silence".to_string(),
                description: "Detect audio silence".to_string(),
                operation: BatchOperation::Analyze {
                    analysis_type: AnalysisType::SilenceDetection,
                },
                priority: Priority::Low,
                retry: RetryPolicy::default(),
                parameters: HashMap::new(),
            },
        );

        // Loudness measurement
        self.presets.insert(
            "measure-loudness".to_string(),
            JobPreset {
                name: "measure-loudness".to_string(),
                description: "Measure audio loudness".to_string(),
                operation: BatchOperation::Analyze {
                    analysis_type: AnalysisType::LoudnessMeasurement,
                },
                priority: Priority::Normal,
                retry: RetryPolicy::default(),
                parameters: HashMap::new(),
            },
        );

        // Copy files
        self.presets.insert(
            "copy-files".to_string(),
            JobPreset {
                name: "copy-files".to_string(),
                description: "Copy files".to_string(),
                operation: BatchOperation::FileOp {
                    operation: FileOperation::Copy { overwrite: false },
                },
                priority: Priority::Normal,
                retry: RetryPolicy::default(),
                parameters: HashMap::new(),
            },
        );

        // Move files
        self.presets.insert(
            "move-files".to_string(),
            JobPreset {
                name: "move-files".to_string(),
                description: "Move files".to_string(),
                operation: BatchOperation::FileOp {
                    operation: FileOperation::Move { overwrite: false },
                },
                priority: Priority::Normal,
                retry: RetryPolicy::default(),
                parameters: HashMap::new(),
            },
        );

        // Create archive
        self.presets.insert(
            "create-archive".to_string(),
            JobPreset {
                name: "create-archive".to_string(),
                description: "Create archive".to_string(),
                operation: BatchOperation::FileOp {
                    operation: FileOperation::Archive {
                        format: crate::operations::file_ops::ArchiveFormat::Zip,
                        compression: 6,
                    },
                },
                priority: Priority::Normal,
                retry: RetryPolicy::default(),
                parameters: HashMap::new(),
            },
        );

        // Calculate checksums
        self.presets.insert(
            "checksum".to_string(),
            JobPreset {
                name: "checksum".to_string(),
                description: "Calculate file checksums".to_string(),
                operation: BatchOperation::FileOp {
                    operation: FileOperation::Checksum {
                        algorithm: crate::operations::file_ops::HashAlgorithm::Sha256,
                    },
                },
                priority: Priority::Low,
                retry: RetryPolicy::default(),
                parameters: HashMap::new(),
            },
        );
    }

    /// Get a preset by name
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&JobPreset> {
        self.presets.get(name)
    }

    /// List all preset names
    #[must_use]
    pub fn list(&self) -> Vec<String> {
        self.presets.keys().cloned().collect()
    }

    /// Add a custom preset
    pub fn add(&mut self, preset: JobPreset) {
        self.presets.insert(preset.name.clone(), preset);
    }

    /// Remove a preset
    pub fn remove(&mut self, name: &str) -> Option<JobPreset> {
        self.presets.remove(name)
    }
}

impl Default for PresetManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preset_manager_creation() {
        let manager = PresetManager::new();
        assert!(!manager.presets.is_empty());
    }

    #[test]
    fn test_get_preset() {
        let manager = PresetManager::new();
        let preset = manager.get("transcode-web");
        assert!(preset.is_some());
    }

    #[test]
    fn test_list_presets() {
        let manager = PresetManager::new();
        let presets = manager.list();
        assert!(presets.contains(&"transcode-web".to_string()));
    }

    #[test]
    fn test_add_custom_preset() {
        let mut manager = PresetManager::new();

        let preset = JobPreset {
            name: "custom".to_string(),
            description: "Custom preset".to_string(),
            operation: BatchOperation::FileOp {
                operation: FileOperation::Copy { overwrite: false },
            },
            priority: Priority::Normal,
            retry: RetryPolicy::default(),
            parameters: HashMap::new(),
        };

        manager.add(preset);
        assert!(manager.get("custom").is_some());
    }

    #[test]
    fn test_remove_preset() {
        let mut manager = PresetManager::new();
        let preset = manager.remove("transcode-web");
        assert!(preset.is_some());
        assert!(manager.get("transcode-web").is_none());
    }

    #[test]
    fn test_create_job_from_preset() {
        let manager = PresetManager::new();
        let preset = manager.get("transcode-web").expect("failed to get value");

        let job = preset.create_job("test-job".to_string());
        assert_eq!(job.name, "test-job");
        assert_eq!(job.priority, Priority::Normal);
    }
}
