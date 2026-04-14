//! Codec configuration validation.
//!
//! Provides rules for rejecting invalid preset, bitrate, and CQ-level
//! combinations on a per-codec basis so that benchmarks do not waste time
//! on nonsensical parameter sets.
//!
//! # Supported codecs
//!
//! | Codec   | Valid presets                               | Bitrate range (kbps) | CQ range |
//! |---------|--------------------------------------------|----------------------|----------|
//! | AV1     | ultrafast, fast, medium, slow, veryslow    | 100 .. 50 000        | 0 .. 63  |
//! | VP9     | realtime, good, best                       | 100 .. 50 000        | 0 .. 63  |
//! | VP8     | realtime, good, best                       | 100 .. 20 000        | 0 .. 63  |
//! | Theora  | *(none)*                                   | 100 .. 16 000        | 0 .. 63  |
//!
//! # Example
//!
//! ```
//! use oximedia_bench::codec_validation::{CodecValidator, ValidationResult};
//! use oximedia_bench::CodecConfig;
//! use oximedia_core::types::CodecId;
//!
//! let config = CodecConfig::new(CodecId::Av1)
//!     .with_preset("medium")
//!     .with_bitrate(2000)
//!     .with_cq_level(30);
//!
//! let result = CodecValidator::validate(&config);
//! assert!(result.is_valid());
//! ```

use crate::{BenchError, BenchResult, CodecConfig};
use oximedia_core::types::CodecId;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Validation result
// ---------------------------------------------------------------------------

/// A single validation issue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationIssue {
    /// Short machine-readable code (e.g. `"invalid_preset"`).
    pub code: String,
    /// Human-readable description.
    pub message: String,
    /// Severity level.
    pub severity: IssueSeverity,
}

/// Severity of a validation issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IssueSeverity {
    /// The configuration is invalid and must be fixed before benchmarking.
    Error,
    /// The configuration is unusual but technically allowed.
    Warning,
}

/// The result of validating a [`CodecConfig`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ValidationResult {
    /// All issues found during validation.
    pub issues: Vec<ValidationIssue>,
}

impl ValidationResult {
    /// Create an empty (valid) result.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `true` when there are no error-level issues.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        !self.issues.iter().any(|i| i.severity == IssueSeverity::Error)
    }

    /// Returns `true` when there are any warning-level issues.
    #[must_use]
    pub fn has_warnings(&self) -> bool {
        self.issues
            .iter()
            .any(|i| i.severity == IssueSeverity::Warning)
    }

    /// Returns a count of error-level issues.
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Error)
            .count()
    }

    /// Returns a count of warning-level issues.
    #[must_use]
    pub fn warning_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Warning)
            .count()
    }

    /// Convert errors into a [`BenchError`] if any are present.
    ///
    /// # Errors
    ///
    /// Returns `BenchError::InvalidConfig` when at least one error exists.
    pub fn into_result(self) -> BenchResult<()> {
        if self.is_valid() {
            Ok(())
        } else {
            let msgs: Vec<String> = self
                .issues
                .iter()
                .filter(|i| i.severity == IssueSeverity::Error)
                .map(|i| i.message.clone())
                .collect();
            Err(BenchError::InvalidConfig(msgs.join("; ")))
        }
    }

    fn push_error(&mut self, code: impl Into<String>, message: impl Into<String>) {
        self.issues.push(ValidationIssue {
            code: code.into(),
            message: message.into(),
            severity: IssueSeverity::Error,
        });
    }

    fn push_warning(&mut self, code: impl Into<String>, message: impl Into<String>) {
        self.issues.push(ValidationIssue {
            code: code.into(),
            message: message.into(),
            severity: IssueSeverity::Warning,
        });
    }
}

// ---------------------------------------------------------------------------
// Codec-specific rules
// ---------------------------------------------------------------------------

/// Known valid presets per codec.
struct CodecRules {
    valid_presets: &'static [&'static str],
    min_bitrate_kbps: u32,
    max_bitrate_kbps: u32,
    min_cq: u32,
    max_cq: u32,
}

fn rules_for_codec(codec_id: CodecId) -> Option<CodecRules> {
    match codec_id {
        CodecId::Av1 => Some(CodecRules {
            valid_presets: &["ultrafast", "fast", "medium", "slow", "veryslow"],
            min_bitrate_kbps: 100,
            max_bitrate_kbps: 50_000,
            min_cq: 0,
            max_cq: 63,
        }),
        CodecId::Vp9 => Some(CodecRules {
            valid_presets: &["realtime", "good", "best"],
            min_bitrate_kbps: 100,
            max_bitrate_kbps: 50_000,
            min_cq: 0,
            max_cq: 63,
        }),
        CodecId::Vp8 => Some(CodecRules {
            valid_presets: &["realtime", "good", "best"],
            min_bitrate_kbps: 100,
            max_bitrate_kbps: 20_000,
            min_cq: 0,
            max_cq: 63,
        }),
        CodecId::Theora => Some(CodecRules {
            valid_presets: &[],
            min_bitrate_kbps: 100,
            max_bitrate_kbps: 16_000,
            min_cq: 0,
            max_cq: 63,
        }),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Validator
// ---------------------------------------------------------------------------

/// Stateless validator for [`CodecConfig`] instances.
pub struct CodecValidator;

impl CodecValidator {
    /// Validate a single [`CodecConfig`].
    ///
    /// Returns a [`ValidationResult`] containing all issues found.
    #[must_use]
    pub fn validate(config: &CodecConfig) -> ValidationResult {
        let mut result = ValidationResult::new();

        let rules = match rules_for_codec(config.codec_id) {
            Some(r) => r,
            None => {
                result.push_error(
                    "unsupported_codec",
                    format!("No validation rules for codec {:?}", config.codec_id),
                );
                return result;
            }
        };

        // --- Preset ---
        if let Some(ref preset) = config.preset {
            if rules.valid_presets.is_empty() {
                result.push_warning(
                    "preset_ignored",
                    format!(
                        "Codec {:?} does not use presets; preset \"{}\" will be ignored",
                        config.codec_id, preset
                    ),
                );
            } else if !rules.valid_presets.contains(&preset.as_str()) {
                result.push_error(
                    "invalid_preset",
                    format!(
                        "Preset \"{}\" is not valid for {:?}; valid presets: {:?}",
                        preset, config.codec_id, rules.valid_presets
                    ),
                );
            }
        }

        // --- Bitrate ---
        if let Some(bitrate) = config.bitrate_kbps {
            if bitrate < rules.min_bitrate_kbps {
                result.push_error(
                    "bitrate_too_low",
                    format!(
                        "Bitrate {} kbps is below minimum {} kbps for {:?}",
                        bitrate, rules.min_bitrate_kbps, config.codec_id
                    ),
                );
            }
            if bitrate > rules.max_bitrate_kbps {
                result.push_error(
                    "bitrate_too_high",
                    format!(
                        "Bitrate {} kbps exceeds maximum {} kbps for {:?}",
                        bitrate, rules.max_bitrate_kbps, config.codec_id
                    ),
                );
            }
        }

        // --- CQ level ---
        if let Some(cq) = config.cq_level {
            if cq < rules.min_cq {
                result.push_error(
                    "cq_too_low",
                    format!(
                        "CQ level {} is below minimum {} for {:?}",
                        cq, rules.min_cq, config.codec_id
                    ),
                );
            }
            if cq > rules.max_cq {
                result.push_error(
                    "cq_too_high",
                    format!(
                        "CQ level {} exceeds maximum {} for {:?}",
                        cq, rules.max_cq, config.codec_id
                    ),
                );
            }
        }

        // --- Mutual exclusion: bitrate + CQ together is a warning ---
        if config.bitrate_kbps.is_some() && config.cq_level.is_some() {
            result.push_warning(
                "bitrate_cq_conflict",
                format!(
                    "Both bitrate ({} kbps) and CQ level ({}) set for {:?}; \
                     most codecs will ignore one of them",
                    config.bitrate_kbps.unwrap_or(0),
                    config.cq_level.unwrap_or(0),
                    config.codec_id
                ),
            );
        }

        // --- Passes ---
        if config.passes == 0 {
            result.push_error("zero_passes", "Number of encoding passes must be >= 1");
        }
        if config.passes > 3 {
            result.push_warning(
                "high_passes",
                format!(
                    "{} encoding passes is unusually high; most codecs support 1 or 2",
                    config.passes
                ),
            );
        }

        result
    }

    /// Validate a slice of configs, returning a map from index to result.
    ///
    /// Only entries with at least one issue are included.
    #[must_use]
    pub fn validate_all(configs: &[CodecConfig]) -> Vec<(usize, ValidationResult)> {
        configs
            .iter()
            .enumerate()
            .map(|(i, cfg)| (i, Self::validate(cfg)))
            .filter(|(_, r)| !r.issues.is_empty())
            .collect()
    }

    /// Validate and convert to a `BenchResult`.
    ///
    /// # Errors
    ///
    /// Returns `BenchError::InvalidConfig` if there are any error-level issues.
    pub fn validate_strict(config: &CodecConfig) -> BenchResult<()> {
        Self::validate(config).into_result()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_av1_config() {
        let cfg = CodecConfig::new(CodecId::Av1)
            .with_preset("medium")
            .with_bitrate(2000);
        let result = CodecValidator::validate(&cfg);
        assert!(result.is_valid());
        assert_eq!(result.error_count(), 0);
    }

    #[test]
    fn test_invalid_preset_av1() {
        let cfg = CodecConfig::new(CodecId::Av1).with_preset("turbo");
        let result = CodecValidator::validate(&cfg);
        assert!(!result.is_valid());
        assert!(result
            .issues
            .iter()
            .any(|i| i.code == "invalid_preset"));
    }

    #[test]
    fn test_bitrate_too_low() {
        let cfg = CodecConfig::new(CodecId::Vp9).with_bitrate(10);
        let result = CodecValidator::validate(&cfg);
        assert!(!result.is_valid());
        assert!(result
            .issues
            .iter()
            .any(|i| i.code == "bitrate_too_low"));
    }

    #[test]
    fn test_bitrate_too_high_vp8() {
        let cfg = CodecConfig::new(CodecId::Vp8).with_bitrate(30_000);
        let result = CodecValidator::validate(&cfg);
        assert!(!result.is_valid());
        assert!(result
            .issues
            .iter()
            .any(|i| i.code == "bitrate_too_high"));
    }

    #[test]
    fn test_cq_too_high() {
        let cfg = CodecConfig::new(CodecId::Av1).with_cq_level(100);
        let result = CodecValidator::validate(&cfg);
        assert!(!result.is_valid());
        assert!(result
            .issues
            .iter()
            .any(|i| i.code == "cq_too_high"));
    }

    #[test]
    fn test_bitrate_cq_conflict_warning() {
        let cfg = CodecConfig::new(CodecId::Av1)
            .with_bitrate(2000)
            .with_cq_level(30);
        let result = CodecValidator::validate(&cfg);
        // Warning, not error
        assert!(result.is_valid());
        assert!(result.has_warnings());
        assert!(result
            .issues
            .iter()
            .any(|i| i.code == "bitrate_cq_conflict"));
    }

    #[test]
    fn test_theora_preset_warning() {
        let cfg = CodecConfig::new(CodecId::Theora).with_preset("fast");
        let result = CodecValidator::validate(&cfg);
        assert!(result.is_valid()); // warning, not error
        assert!(result.has_warnings());
        assert!(result
            .issues
            .iter()
            .any(|i| i.code == "preset_ignored"));
    }

    #[test]
    fn test_zero_passes() {
        let cfg = CodecConfig::new(CodecId::Av1).with_passes(0);
        let result = CodecValidator::validate(&cfg);
        assert!(!result.is_valid());
        assert!(result
            .issues
            .iter()
            .any(|i| i.code == "zero_passes"));
    }

    #[test]
    fn test_high_passes_warning() {
        let cfg = CodecConfig::new(CodecId::Vp9).with_passes(5);
        let result = CodecValidator::validate(&cfg);
        assert!(result.is_valid());
        assert!(result.has_warnings());
        assert!(result
            .issues
            .iter()
            .any(|i| i.code == "high_passes"));
    }

    #[test]
    fn test_validate_strict_ok() {
        let cfg = CodecConfig::new(CodecId::Vp9)
            .with_preset("good")
            .with_bitrate(3000);
        assert!(CodecValidator::validate_strict(&cfg).is_ok());
    }

    #[test]
    fn test_validate_strict_err() {
        let cfg = CodecConfig::new(CodecId::Vp9).with_preset("nonexistent");
        assert!(CodecValidator::validate_strict(&cfg).is_err());
    }

    #[test]
    fn test_validate_all() {
        let configs = vec![
            CodecConfig::new(CodecId::Av1).with_preset("medium"),
            CodecConfig::new(CodecId::Vp9).with_preset("wrong"),
            CodecConfig::new(CodecId::Vp8).with_bitrate(5),
        ];
        let issues = CodecValidator::validate_all(&configs);
        // First config is clean, so only indices 1 and 2 should appear.
        assert_eq!(issues.len(), 2);
        assert_eq!(issues[0].0, 1);
        assert_eq!(issues[1].0, 2);
    }

    #[test]
    fn test_into_result_ok() {
        let r = ValidationResult::new();
        assert!(r.into_result().is_ok());
    }

    #[test]
    fn test_validation_result_counts() {
        let mut r = ValidationResult::new();
        r.push_error("e1", "error one");
        r.push_warning("w1", "warning one");
        r.push_error("e2", "error two");
        assert_eq!(r.error_count(), 2);
        assert_eq!(r.warning_count(), 1);
        assert!(!r.is_valid());
        assert!(r.has_warnings());
    }

    #[test]
    fn test_valid_vp9_all_presets() {
        for preset in &["realtime", "good", "best"] {
            let cfg = CodecConfig::new(CodecId::Vp9).with_preset(*preset);
            let result = CodecValidator::validate(&cfg);
            assert!(result.is_valid(), "preset {} should be valid", preset);
        }
    }
}
