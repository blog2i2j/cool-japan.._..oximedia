//! Usage examples and common patterns.
//!
//! This module provides practical examples of using the optimization suite.

#![allow(dead_code)]

use crate::{
    LookaheadAnalyzer, OptimizationLevel, OptimizationPresets, Optimizer, OptimizerConfig,
};
use oximedia_core::OxiResult;

/// Example: Basic optimizer setup.
///
/// # Example
///
/// ```ignore
/// use oximedia_optimize::examples::basic_optimizer;
///
/// let optimizer = basic_optimizer()?;
/// ```
pub fn basic_optimizer() -> OxiResult<Optimizer> {
    let config = OptimizerConfig::default();
    Optimizer::new(config)
}

/// Example: High-quality encoding configuration.
///
/// # Example
///
/// ```ignore
/// use oximedia_optimize::examples::high_quality_config;
///
/// let config = high_quality_config();
/// ```
#[must_use]
pub fn high_quality_config() -> OptimizerConfig {
    OptimizationPresets::slow()
}

/// Example: Fast encoding configuration.
///
/// # Example
///
/// ```ignore
/// use oximedia_optimize::examples::fast_encoding_config;
///
/// let config = fast_encoding_config();
/// ```
#[must_use]
pub fn fast_encoding_config() -> OptimizerConfig {
    OptimizationPresets::veryfast()
}

/// Example: Animation-optimized configuration.
///
/// # Example
///
/// ```ignore
/// use oximedia_optimize::examples::animation_config;
///
/// let config = animation_config();
/// ```
#[must_use]
pub fn animation_config() -> OptimizerConfig {
    OptimizationPresets::animation()
}

/// Example: Live streaming configuration.
///
/// # Example
///
/// ```ignore
/// use oximedia_optimize::examples::live_streaming_config;
///
/// let config = live_streaming_config();
/// ```
#[must_use]
pub fn live_streaming_config() -> OptimizerConfig {
    let mut config = OptimizationPresets::faster();
    config.lookahead_frames = 0; // Zero latency
    config
}

/// Example: Archive/storage configuration.
///
/// # Example
///
/// ```ignore
/// use oximedia_optimize::examples::archive_config;
///
/// let config = archive_config();
/// ```
#[must_use]
pub fn archive_config() -> OptimizerConfig {
    OptimizationPresets::placebo()
}

/// Example: Custom configuration builder.
///
/// # Example
///
/// ```ignore
/// use oximedia_optimize::examples::ConfigBuilder;
///
/// let config = ConfigBuilder::new()
///     .level(OptimizationLevel::Slow)
///     .enable_psychovisual(true)
///     .lookahead(40)
///     .build();
/// ```
pub struct ConfigBuilder {
    config: OptimizerConfig,
}

impl Default for ConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfigBuilder {
    /// Creates a new config builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: OptimizerConfig::default(),
        }
    }

    /// Sets optimization level.
    #[must_use]
    pub fn level(mut self, level: OptimizationLevel) -> Self {
        self.config.level = level;
        self
    }

    /// Enables/disables psychovisual optimization.
    #[must_use]
    pub fn enable_psychovisual(mut self, enable: bool) -> Self {
        self.config.enable_psychovisual = enable;
        self
    }

    /// Enables/disables adaptive quantization.
    #[must_use]
    pub fn enable_aq(mut self, enable: bool) -> Self {
        self.config.enable_aq = enable;
        self
    }

    /// Sets lookahead frames.
    #[must_use]
    pub fn lookahead(mut self, frames: usize) -> Self {
        self.config.lookahead_frames = frames;
        self
    }

    /// Sets lambda multiplier.
    #[must_use]
    pub fn lambda_multiplier(mut self, multiplier: f64) -> Self {
        self.config.lambda_multiplier = multiplier;
        self
    }

    /// Enables/disables parallel RDO.
    #[must_use]
    pub fn parallel_rdo(mut self, enable: bool) -> Self {
        self.config.parallel_rdo = enable;
        self
    }

    /// Builds the configuration.
    #[must_use]
    pub fn build(self) -> OptimizerConfig {
        self.config
    }
}

/// Example: Progressive optimization workflow.
///
/// Shows how to start with fast encoding and progressively improve quality.
pub struct ProgressiveWorkflow {
    current_level: OptimizationLevel,
    max_level: OptimizationLevel,
}

impl ProgressiveWorkflow {
    /// Creates a new progressive workflow.
    #[must_use]
    pub fn new() -> Self {
        Self {
            current_level: OptimizationLevel::Fast,
            max_level: OptimizationLevel::Placebo,
        }
    }

    /// Gets current configuration.
    #[must_use]
    pub fn current_config(&self) -> OptimizerConfig {
        let mut config = OptimizerConfig::default();
        config.level = self.current_level;
        config
    }

    /// Advances to next level.
    pub fn advance(&mut self) -> bool {
        self.current_level = match self.current_level {
            OptimizationLevel::Fast => OptimizationLevel::Medium,
            OptimizationLevel::Medium => OptimizationLevel::Slow,
            OptimizationLevel::Slow => OptimizationLevel::Placebo,
            OptimizationLevel::Placebo => return false,
        };
        true
    }

    /// Checks if workflow is complete.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.current_level == self.max_level
    }
}

impl Default for ProgressiveWorkflow {
    fn default() -> Self {
        Self::new()
    }
}

/// Example: Quality ladder for adaptive streaming.
///
/// Generates multiple quality levels for ABR (Adaptive Bitrate) streaming.
pub struct QualityLadder {
    rungs: Vec<QualityRung>,
}

/// Quality rung for a single quality level in adaptive streaming.
#[derive(Debug, Clone)]
pub struct QualityRung {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Target bitrate.
    pub bitrate: u64,
    /// Optimization config.
    pub config: OptimizerConfig,
}

impl QualityLadder {
    /// Creates a standard quality ladder.
    #[must_use]
    pub fn standard() -> Self {
        let rungs = vec![
            QualityRung {
                width: 426,
                height: 240,
                bitrate: 400_000,
                config: OptimizationPresets::faster(),
            },
            QualityRung {
                width: 640,
                height: 360,
                bitrate: 800_000,
                config: OptimizationPresets::fast(),
            },
            QualityRung {
                width: 854,
                height: 480,
                bitrate: 1_400_000,
                config: OptimizationPresets::medium(),
            },
            QualityRung {
                width: 1280,
                height: 720,
                bitrate: 2_800_000,
                config: OptimizationPresets::medium(),
            },
            QualityRung {
                width: 1920,
                height: 1080,
                bitrate: 5_000_000,
                config: OptimizationPresets::slow(),
            },
            QualityRung {
                width: 3840,
                height: 2160,
                bitrate: 15_000_000,
                config: OptimizationPresets::slow(),
            },
        ];

        Self { rungs }
    }

    /// Gets all rungs.
    #[must_use]
    pub fn rungs(&self) -> &[QualityRung] {
        &self.rungs
    }

    /// Finds rung for target resolution.
    #[must_use]
    pub fn find_rung(&self, width: usize, height: usize) -> Option<&QualityRung> {
        self.rungs
            .iter()
            .find(|r| r.width == width && r.height == height)
    }
}

impl Default for QualityLadder {
    fn default() -> Self {
        Self::standard()
    }
}

/// Example: Encoding pipeline with optimization.
pub struct EncodingPipeline {
    optimizer: Optimizer,
    lookahead_analyzer: LookaheadAnalyzer,
}

impl EncodingPipeline {
    /// Creates a new encoding pipeline.
    pub fn new(config: OptimizerConfig) -> OxiResult<Self> {
        let optimizer = Optimizer::new(config.clone())?;
        let lookahead_analyzer = LookaheadAnalyzer::new(&config)?;

        Ok(Self {
            optimizer,
            lookahead_analyzer,
        })
    }

    /// Process a frame.
    #[allow(unused_variables)]
    pub fn process_frame(&self, frame_data: &[u8], width: usize, height: usize) -> OxiResult<()> {
        // Example: Get RDO engine
        let _rdo_engine = self.optimizer.rdo_engine();

        // Example: Get psycho analyzer
        let _psycho = self.optimizer.psycho_analyzer();

        // Example: Get AQ engine
        let _aq = self.optimizer.aq_engine();

        Ok(())
    }
}

/// Example: Batch encoding with different presets.
pub struct BatchEncoder {
    presets: Vec<(&'static str, OptimizerConfig)>,
}

impl BatchEncoder {
    /// Creates a new batch encoder with standard presets.
    #[must_use]
    pub fn new() -> Self {
        let presets = vec![
            ("ultrafast", OptimizationPresets::ultrafast()),
            ("fast", OptimizationPresets::fast()),
            ("medium", OptimizationPresets::medium()),
            ("slow", OptimizationPresets::slow()),
            ("placebo", OptimizationPresets::placebo()),
        ];

        Self { presets }
    }

    /// Gets all preset names.
    #[must_use]
    pub fn preset_names(&self) -> Vec<&str> {
        self.presets.iter().map(|(name, _)| *name).collect()
    }

    /// Gets config for a preset.
    #[must_use]
    pub fn get_config(&self, name: &str) -> Option<&OptimizerConfig> {
        self.presets
            .iter()
            .find(|(preset_name, _)| *preset_name == name)
            .map(|(_, config)| config)
    }
}

impl Default for BatchEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Example: A/B testing framework.
pub struct AbTestFramework {
    config_a: OptimizerConfig,
    config_b: OptimizerConfig,
}

impl AbTestFramework {
    /// Creates a new A/B test framework.
    #[must_use]
    pub fn new(config_a: OptimizerConfig, config_b: OptimizerConfig) -> Self {
        Self { config_a, config_b }
    }

    /// Runs A/B test.
    #[allow(unused_variables)]
    #[must_use]
    pub fn run_test(&self, frames: &[&[u8]]) -> TestResults {
        // Simplified test results
        TestResults {
            config_a_time: std::time::Duration::from_secs(10),
            config_b_time: std::time::Duration::from_secs(15),
            config_a_quality: 42.0,
            config_b_quality: 44.0,
            config_a_size: 1_000_000,
            config_b_size: 900_000,
        }
    }
}

/// A/B test results.
#[derive(Debug, Clone)]
pub struct TestResults {
    /// Encoding time for config A.
    pub config_a_time: std::time::Duration,
    /// Encoding time for config B.
    pub config_b_time: std::time::Duration,
    /// Quality for config A.
    pub config_a_quality: f64,
    /// Quality for config B.
    pub config_b_quality: f64,
    /// Output size for config A.
    pub config_a_size: u64,
    /// Output size for config B.
    pub config_b_size: u64,
}

impl TestResults {
    /// Determines which config is better.
    #[must_use]
    pub fn better_config(&self) -> &str {
        // Simple heuristic: better quality at similar or smaller size
        if self.config_a_quality > self.config_b_quality && self.config_a_size <= self.config_b_size
        {
            "A"
        } else if self.config_b_quality > self.config_a_quality
            && self.config_b_size <= self.config_a_size
        {
            "B"
        } else {
            "Inconclusive"
        }
    }

    /// Prints results.
    pub fn print(&self) {
        println!("A/B Test Results:");
        println!("Config A:");
        println!("  Time: {:?}", self.config_a_time);
        println!("  Quality: {:.2} dB", self.config_a_quality);
        println!("  Size: {} bytes", self.config_a_size);
        println!("Config B:");
        println!("  Time: {:?}", self.config_b_time);
        println!("  Quality: {:.2} dB", self.config_b_quality);
        println!("  Size: {} bytes", self.config_b_size);
        println!("Better: {}", self.better_config());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_optimizer() {
        let optimizer = basic_optimizer();
        assert!(optimizer.is_ok());
    }

    #[test]
    fn test_high_quality_config() {
        let config = high_quality_config();
        assert_eq!(config.level, OptimizationLevel::Slow);
    }

    #[test]
    fn test_fast_encoding_config() {
        let config = fast_encoding_config();
        assert_eq!(config.level, OptimizationLevel::Fast);
    }

    #[test]
    fn test_live_streaming_config() {
        let config = live_streaming_config();
        assert_eq!(config.lookahead_frames, 0);
    }

    #[test]
    fn test_config_builder() {
        let config = ConfigBuilder::new()
            .level(OptimizationLevel::Slow)
            .enable_psychovisual(true)
            .lookahead(40)
            .lambda_multiplier(1.2)
            .build();

        assert_eq!(config.level, OptimizationLevel::Slow);
        assert!(config.enable_psychovisual);
        assert_eq!(config.lookahead_frames, 40);
        assert_eq!(config.lambda_multiplier, 1.2);
    }

    #[test]
    fn test_progressive_workflow() {
        let mut workflow = ProgressiveWorkflow::new();
        assert!(!workflow.is_complete());

        assert!(workflow.advance()); // Fast -> Medium
        assert!(workflow.advance()); // Medium -> Slow
        assert!(workflow.advance()); // Slow -> Placebo
        assert!(workflow.is_complete());
        assert!(!workflow.advance()); // Can't advance further
    }

    #[test]
    fn test_quality_ladder() {
        let ladder = QualityLadder::standard();
        assert_eq!(ladder.rungs().len(), 6);

        let hd_rung = ladder.find_rung(1280, 720);
        assert!(hd_rung.is_some());
        assert_eq!(hd_rung.expect("rung should be found").bitrate, 2_800_000);
    }

    #[test]
    fn test_batch_encoder() {
        let encoder = BatchEncoder::new();
        let names = encoder.preset_names();
        assert!(names.contains(&"fast"));
        assert!(names.contains(&"slow"));

        let config = encoder.get_config("medium");
        assert!(config.is_some());
    }

    #[test]
    fn test_ab_test_framework() {
        let config_a = OptimizationPresets::fast();
        let config_b = OptimizationPresets::slow();
        let framework = AbTestFramework::new(config_a, config_b);

        let frames = vec![];
        let results = framework.run_test(&frames);
        assert!(results.config_a_time < results.config_b_time);
    }

    #[test]
    fn test_test_results_better_config() {
        let results = TestResults {
            config_a_time: std::time::Duration::from_secs(10),
            config_b_time: std::time::Duration::from_secs(15),
            config_a_quality: 44.0,
            config_b_quality: 42.0,
            config_a_size: 900_000,
            config_b_size: 1_000_000,
        };

        assert_eq!(results.better_config(), "A");
    }
}
