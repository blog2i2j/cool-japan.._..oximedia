//! Multi-pass encoding controller for optimal quality encoding.

use crate::{Result, TranscodeError};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Multi-pass encoding mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MultiPassMode {
    /// Single-pass encoding (fastest, lower quality).
    #[default]
    SinglePass,
    /// Two-pass encoding (good balance of speed and quality).
    TwoPass,
    /// Three-pass encoding (best quality, slowest).
    ThreePass,
}

/// Multi-pass encoding configuration.
#[derive(Debug, Clone)]
pub struct MultiPassConfig {
    /// Encoding mode.
    pub mode: MultiPassMode,
    /// Statistics file path for pass data.
    pub stats_file: PathBuf,
    /// Current pass number.
    pub current_pass: u32,
    /// Whether to keep statistics files after encoding.
    pub keep_stats: bool,
    /// Target bitrate for multi-pass encoding.
    pub target_bitrate: Option<u64>,
    /// Maximum bitrate for constrained encoding.
    pub max_bitrate: Option<u64>,
}

impl MultiPassMode {
    /// Returns the total number of passes for this mode.
    #[must_use]
    pub fn pass_count(self) -> u32 {
        match self {
            Self::SinglePass => 1,
            Self::TwoPass => 2,
            Self::ThreePass => 3,
        }
    }

    /// Checks if this mode requires statistics files.
    #[must_use]
    pub fn requires_stats(self) -> bool {
        !matches!(self, Self::SinglePass)
    }

    /// Gets a human-readable description of the mode.
    #[must_use]
    pub fn description(self) -> &'static str {
        match self {
            Self::SinglePass => "Single-pass encoding (fast, good quality)",
            Self::TwoPass => "Two-pass encoding (balanced speed and quality)",
            Self::ThreePass => "Three-pass encoding (slow, best quality)",
        }
    }
}

impl MultiPassConfig {
    /// Creates a new multi-pass configuration.
    ///
    /// # Arguments
    ///
    /// * `mode` - The multi-pass mode to use
    /// * `stats_file` - Path where statistics will be stored
    pub fn new(mode: MultiPassMode, stats_file: impl Into<PathBuf>) -> Self {
        Self {
            mode,
            stats_file: stats_file.into(),
            current_pass: 1,
            keep_stats: false,
            target_bitrate: None,
            max_bitrate: None,
        }
    }

    /// Sets the target bitrate for multi-pass encoding.
    #[must_use]
    pub fn with_target_bitrate(mut self, bitrate: u64) -> Self {
        self.target_bitrate = Some(bitrate);
        self
    }

    /// Sets the maximum bitrate for constrained encoding.
    #[must_use]
    pub fn with_max_bitrate(mut self, bitrate: u64) -> Self {
        self.max_bitrate = Some(bitrate);
        self
    }

    /// Sets whether to keep statistics files after encoding.
    #[must_use]
    pub fn keep_stats(mut self, keep: bool) -> Self {
        self.keep_stats = keep;
        self
    }

    /// Gets the statistics file path for a specific pass.
    #[must_use]
    pub fn stats_file_for_pass(&self, pass: u32) -> PathBuf {
        if self.mode.pass_count() == 1 {
            return self.stats_file.clone();
        }

        let mut path = self.stats_file.clone();
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("stats");
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("log");

        path.set_file_name(format!("{stem}_pass{pass}.{ext}"));
        path
    }

    /// Checks if a pass is an analysis pass (no output).
    #[must_use]
    pub fn is_analysis_pass(&self, pass: u32) -> bool {
        if self.mode == MultiPassMode::SinglePass {
            return false;
        }

        // In multi-pass encoding, all passes except the last are analysis
        pass < self.mode.pass_count()
    }

    /// Gets the encoder flags for a specific pass.
    #[must_use]
    pub fn encoder_flags_for_pass(&self, pass: u32) -> Vec<String> {
        let mut flags = Vec::new();

        match self.mode {
            MultiPassMode::SinglePass => {
                // No special flags needed
            }
            MultiPassMode::TwoPass => {
                if pass == 1 {
                    flags.push("pass=1".to_string());
                    flags.push(format!(
                        "stats={}",
                        self.stats_file_for_pass(pass).display()
                    ));
                } else {
                    flags.push("pass=2".to_string());
                    flags.push(format!("stats={}", self.stats_file_for_pass(1).display()));
                }
            }
            MultiPassMode::ThreePass => match pass {
                1 => {
                    flags.push("pass=1".to_string());
                    flags.push(format!(
                        "stats={}",
                        self.stats_file_for_pass(pass).display()
                    ));
                }
                2 => {
                    flags.push("pass=2".to_string());
                    flags.push(format!(
                        "stats={}",
                        self.stats_file_for_pass(pass).display()
                    ));
                }
                3 => {
                    flags.push("pass=3".to_string());
                    flags.push(format!("stats={}", self.stats_file_for_pass(1).display()));
                    flags.push(format!("stats2={}", self.stats_file_for_pass(2).display()));
                }
                _ => {}
            },
        }

        flags
    }

    /// Validates the multi-pass configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn validate(&self) -> Result<()> {
        if self.mode.requires_stats() {
            let parent = self.stats_file.parent().ok_or_else(|| {
                TranscodeError::MultiPassError("Invalid stats file path".to_string())
            })?;

            if !parent.exists() {
                return Err(TranscodeError::MultiPassError(format!(
                    "Stats directory does not exist: {}",
                    parent.display()
                )));
            }
        }

        if let (Some(target), Some(max)) = (self.target_bitrate, self.max_bitrate) {
            if target > max {
                return Err(TranscodeError::MultiPassError(
                    "Target bitrate cannot exceed max bitrate".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Cleans up statistics files.
    pub fn cleanup(&self) -> Result<()> {
        if !self.keep_stats && self.mode.requires_stats() {
            for pass in 1..=self.mode.pass_count() {
                let stats_file = self.stats_file_for_pass(pass);
                if stats_file.exists() {
                    std::fs::remove_file(&stats_file)?;
                }
            }
        }
        Ok(())
    }
}

/// Multi-pass encoder controller.
pub struct MultiPassEncoder {
    config: MultiPassConfig,
}

impl MultiPassEncoder {
    /// Creates a new multi-pass encoder controller.
    #[must_use]
    pub fn new(config: MultiPassConfig) -> Self {
        Self { config }
    }

    /// Gets the total number of passes.
    #[must_use]
    pub fn pass_count(&self) -> u32 {
        self.config.mode.pass_count()
    }

    /// Checks if more passes are needed.
    #[must_use]
    pub fn has_more_passes(&self) -> bool {
        self.config.current_pass < self.pass_count()
    }

    /// Advances to the next pass.
    pub fn next_pass(&mut self) {
        if self.has_more_passes() {
            self.config.current_pass += 1;
        }
    }

    /// Gets the current pass number.
    #[must_use]
    pub fn current_pass(&self) -> u32 {
        self.config.current_pass
    }

    /// Gets the encoder flags for the current pass.
    #[must_use]
    pub fn current_encoder_flags(&self) -> Vec<String> {
        self.config.encoder_flags_for_pass(self.config.current_pass)
    }

    /// Checks if the current pass is an analysis pass.
    #[must_use]
    pub fn is_current_analysis_pass(&self) -> bool {
        self.config.is_analysis_pass(self.config.current_pass)
    }

    /// Resets the encoder to the first pass.
    pub fn reset(&mut self) {
        self.config.current_pass = 1;
    }

    /// Cleans up statistics files.
    pub fn cleanup(&self) -> Result<()> {
        self.config.cleanup()
    }
}

/// Builder for multi-pass configuration.
#[allow(dead_code)]
pub struct MultiPassConfigBuilder {
    mode: MultiPassMode,
    stats_file: Option<PathBuf>,
    keep_stats: bool,
    target_bitrate: Option<u64>,
    max_bitrate: Option<u64>,
}

#[allow(dead_code)]
impl MultiPassConfigBuilder {
    /// Creates a new builder with the specified mode.
    #[must_use]
    pub fn new(mode: MultiPassMode) -> Self {
        Self {
            mode,
            stats_file: None,
            keep_stats: false,
            target_bitrate: None,
            max_bitrate: None,
        }
    }

    /// Sets the statistics file path.
    #[must_use]
    pub fn stats_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.stats_file = Some(path.into());
        self
    }

    /// Sets whether to keep statistics files.
    #[must_use]
    pub fn keep_stats(mut self, keep: bool) -> Self {
        self.keep_stats = keep;
        self
    }

    /// Sets the target bitrate.
    #[must_use]
    pub fn target_bitrate(mut self, bitrate: u64) -> Self {
        self.target_bitrate = Some(bitrate);
        self
    }

    /// Sets the maximum bitrate.
    #[must_use]
    pub fn max_bitrate(mut self, bitrate: u64) -> Self {
        self.max_bitrate = Some(bitrate);
        self
    }

    /// Builds the multi-pass configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the stats file is required but not set.
    pub fn build(self) -> Result<MultiPassConfig> {
        let stats_file = if self.mode.requires_stats() {
            self.stats_file.ok_or_else(|| {
                TranscodeError::MultiPassError(
                    "Stats file required for multi-pass encoding".to_string(),
                )
            })?
        } else {
            self.stats_file
                .unwrap_or_else(|| std::env::temp_dir().join("oximedia-stats.log"))
        };

        let mut config = MultiPassConfig::new(self.mode, stats_file);
        config.keep_stats = self.keep_stats;
        config.target_bitrate = self.target_bitrate;
        config.max_bitrate = self.max_bitrate;

        config.validate()?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multipass_mode_pass_count() {
        assert_eq!(MultiPassMode::SinglePass.pass_count(), 1);
        assert_eq!(MultiPassMode::TwoPass.pass_count(), 2);
        assert_eq!(MultiPassMode::ThreePass.pass_count(), 3);
    }

    #[test]
    fn test_multipass_mode_requires_stats() {
        assert!(!MultiPassMode::SinglePass.requires_stats());
        assert!(MultiPassMode::TwoPass.requires_stats());
        assert!(MultiPassMode::ThreePass.requires_stats());
    }

    #[test]
    fn test_multipass_config_stats_file() {
        let tmp_stats = std::env::temp_dir().join("oximedia-multipass-stats.log");
        let config = MultiPassConfig::new(MultiPassMode::TwoPass, tmp_stats.clone());
        let expected_pass1 = std::env::temp_dir().join("oximedia-multipass-stats_pass1.log");
        let expected_pass2 = std::env::temp_dir().join("oximedia-multipass-stats_pass2.log");
        assert_eq!(config.stats_file_for_pass(1), expected_pass1);
        assert_eq!(config.stats_file_for_pass(2), expected_pass2);
    }

    #[test]
    fn test_multipass_config_is_analysis_pass() {
        let tmp_stats = std::env::temp_dir().join("oximedia-multipass-stats.log");
        let config = MultiPassConfig::new(MultiPassMode::TwoPass, tmp_stats);
        assert!(config.is_analysis_pass(1));
        assert!(!config.is_analysis_pass(2));
    }

    #[test]
    fn test_multipass_encoder_flow() {
        let tmp_stats = std::env::temp_dir().join("oximedia-multipass-stats.log");
        let config = MultiPassConfig::new(MultiPassMode::TwoPass, tmp_stats);
        let mut encoder = MultiPassEncoder::new(config);

        assert_eq!(encoder.current_pass(), 1);
        assert!(encoder.has_more_passes());
        assert!(encoder.is_current_analysis_pass());

        encoder.next_pass();

        assert_eq!(encoder.current_pass(), 2);
        assert!(!encoder.has_more_passes());
        assert!(!encoder.is_current_analysis_pass());
    }

    #[test]
    fn test_multipass_encoder_reset() {
        let tmp_stats = std::env::temp_dir().join("oximedia-multipass-stats.log");
        let config = MultiPassConfig::new(MultiPassMode::TwoPass, tmp_stats);
        let mut encoder = MultiPassEncoder::new(config);

        encoder.next_pass();
        assert_eq!(encoder.current_pass(), 2);

        encoder.reset();
        assert_eq!(encoder.current_pass(), 1);
    }

    #[test]
    fn test_multipass_config_builder() {
        let tmp_stats = std::env::temp_dir().join("oximedia-multipass-test-stats.log");
        let config = MultiPassConfigBuilder::new(MultiPassMode::TwoPass)
            .stats_file(tmp_stats)
            .target_bitrate(5_000_000)
            .max_bitrate(8_000_000)
            .keep_stats(true)
            .build()
            .expect("should succeed in test");

        assert_eq!(config.mode, MultiPassMode::TwoPass);
        assert_eq!(config.target_bitrate, Some(5_000_000));
        assert_eq!(config.max_bitrate, Some(8_000_000));
        assert!(config.keep_stats);
    }

    #[test]
    fn test_encoder_flags_two_pass() {
        let tmp_stats = std::env::temp_dir().join("oximedia-multipass-stats.log");
        let config = MultiPassConfig::new(MultiPassMode::TwoPass, tmp_stats);

        let flags1 = config.encoder_flags_for_pass(1);
        assert!(flags1.contains(&"pass=1".to_string()));

        let flags2 = config.encoder_flags_for_pass(2);
        assert!(flags2.contains(&"pass=2".to_string()));
    }

    #[test]
    fn test_single_pass_no_stats() {
        let tmp_stats = std::env::temp_dir().join("oximedia-multipass-stats.log");
        let config = MultiPassConfig::new(MultiPassMode::SinglePass, tmp_stats);
        assert!(!config.is_analysis_pass(1));
        assert_eq!(config.encoder_flags_for_pass(1).len(), 0);
    }
}
