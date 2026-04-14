//! Codec optimization and tuning suite for `OxiMedia`.
//!
//! `oximedia-optimize` provides advanced optimization techniques for video encoders:
//!
//! - **Rate-Distortion Optimization (RDO)** - Advanced mode decision based on rate-distortion curves
//! - **Psychovisual Optimization** - Perceptual quality tuning using visual masking models
//! - **Motion Search Tuning** - Advanced algorithms (`TZSearch`, EPZS, UMH) for motion estimation
//! - **Intra Prediction Optimization** - RDO-based mode selection for intra frames
//! - **Transform Optimization** - Adaptive transform selection (DCT/ADST) and quantization
//! - **Loop Filter Tuning** - Deblocking and Sample Adaptive Offset (SAO) optimization
//! - **Partition Selection** - Complexity-based block size selection
//! - **Reference Frame Management** - Optimal reference frame selection and DPB management
//! - **Adaptive Quantization** - Variance and psychovisual-based AQ modes
//! - **Entropy Coding Optimization** - Context modeling for CABAC/CAVLC
//! - **ROI Encoding** - Region-of-interest based quality allocation via [`roi_encode`]
//! - **Temporal AQ** - Frame-level QP adaptation based on temporal complexity via [`temporal_aq`]
//! - **VMAF Prediction** - Lightweight VMAF score estimation from pixel features via [`vmaf_predict`]
//! - **Content-Adaptive GOP** - GOP structure selection based on content type via [`gop_optimizer`]
//! - **Scene-Aware QP** - Lookahead-based scene-cut-aware QP adjustment via [`scene_encode`]
//!
//! # Architecture
//!
//! The optimization suite is organized into several modules:
//!
//! - [`rdo`] - Rate-distortion optimization engine and cost functions
//! - [`psycho`] - Psychovisual optimization and masking models
//! - [`motion`] - Advanced motion search algorithms
//! - [`intra`] - Intra mode selection and directional prediction
//! - [`transform`] - Transform type selection and quantization
//! - [`filter`] - Loop filter strength tuning
//! - [`partition`] - Partition decision trees
//! - [`mod@reference`] - Reference frame management
//! - [`aq`] - Adaptive quantization strategies
//! - [`entropy`] - Entropy coding context optimization
//! - [`roi_encode`] - Region of interest encoding with per-CTU QP maps
//! - [`temporal_aq`] - Temporal adaptive quantization with AQ bridge
//! - [`vmaf_predict`] - VMAF score prediction from spatial/temporal features
//! - [`gop_optimizer`] - Content-adaptive GOP structure selection
//! - [`scene_encode`] - Lookahead-based scene-aware QP adjustment
//!
//! # Optimization Levels
//!
//! Different preset levels balance encoding speed vs. quality:
//!
//! - **Fast**: Simple SAD-based decisions, limited search patterns
//! - **Medium**: SATD-based with moderate RDO
//! - **Slow**: Full RDO with extended search patterns
//! - **Placebo**: Exhaustive search for maximum quality
//!
//! # Example
//!
//! ```ignore
//! use oximedia_optimize::{OptimizerConfig, OptimizationLevel, Optimizer};
//!
//! let config = OptimizerConfig {
//!     level: OptimizationLevel::Slow,
//!     enable_psychovisual: true,
//!     enable_aq: true,
//!     lookahead_frames: 40,
//!     ..Default::default()
//! };
//!
//! let optimizer = Optimizer::new(config)?;
//! let decision = optimizer.optimize_block(&frame_data, block_info)?;
//! ```

#![deny(unsafe_code)]
#![warn(missing_docs)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::module_name_repetitions)]

pub mod adaptive_ladder;
pub mod aq;
pub mod aq_map;
pub mod av1_tile_opt;
pub mod benchmark;
pub mod bitrate_controller;
pub mod bitrate_optimizer;
pub mod cache_opt;
pub mod cache_optimizer;
pub mod cache_strategy;
pub mod complexity_analysis;
pub mod crf_sweep;
pub mod decision;
pub mod denoise_aware;
pub mod encode_preset;
pub mod encode_stats;
pub mod entropy;
pub mod examples;
pub mod filter;
pub mod frame_budget;
pub mod gop_optimizer;
pub mod intra;
pub mod ladder_opt;
pub mod lookahead;
pub mod media_optimize;
pub mod motion;
pub mod multi_pass;
pub mod parallel_strategy;
pub mod partition;
pub mod perceptual_optimization;
pub mod prefetch;
pub mod presets;
pub mod psycho;
pub mod quality_ladder;
pub mod quality_metric;
pub mod quantizer_curve;
pub mod rd_model;
pub mod rdo;
pub mod reference;
pub mod roi_encode;
pub mod scene_encode;
pub mod scene_params;
pub mod strategies;
pub mod temporal_aq;
pub mod transcode_optimizer;
pub mod transform;
pub mod two_pass;
pub mod utils;
pub mod vbr_budget;
pub mod viterbi_alloc;
pub mod vmaf_predict;

/// SIMD-accelerated block difference metrics (SAD, SATD).
#[allow(unsafe_code)]
pub mod simd_metrics;

use oximedia_core::OxiResult;

/// Optimization level presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OptimizationLevel {
    /// Fast encoding with simple SAD-based decisions.
    Fast,
    /// Medium quality with SATD and moderate RDO.
    #[default]
    Medium,
    /// Slow encoding with full RDO.
    Slow,
    /// Exhaustive search for maximum quality.
    Placebo,
}

/// Content type hints for adaptive optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ContentType {
    /// Animation content (sharp edges, flat areas).
    Animation,
    /// Film/camera content (grain, natural textures).
    Film,
    /// Screen content (text, graphics).
    Screen,
    /// Generic mixed content.
    #[default]
    Generic,
}

/// Main optimizer configuration.
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Optimization level preset.
    pub level: OptimizationLevel,
    /// Enable psychovisual optimizations.
    pub enable_psychovisual: bool,
    /// Enable adaptive quantization.
    pub enable_aq: bool,
    /// Number of lookahead frames for temporal optimization.
    pub lookahead_frames: usize,
    /// Content type hint.
    pub content_type: ContentType,
    /// Enable parallel RDO evaluation.
    pub parallel_rdo: bool,
    /// Lambda multiplier for rate-distortion tradeoff.
    pub lambda_multiplier: f64,
    /// Enable ROI-based encoding optimization.
    pub enable_roi: bool,
    /// Enable temporal AQ (frame-level QP adaptation).
    pub enable_temporal_aq: bool,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            level: OptimizationLevel::default(),
            enable_psychovisual: true,
            enable_aq: true,
            lookahead_frames: 20,
            content_type: ContentType::default(),
            parallel_rdo: true,
            lambda_multiplier: 1.0,
            enable_roi: false,
            enable_temporal_aq: false,
        }
    }
}

/// Main optimization engine.
///
/// Integrates RDO, psychovisual analysis, motion optimization, adaptive quantization,
/// ROI encoding, and temporal AQ into a unified pipeline.
pub struct Optimizer {
    config: OptimizerConfig,
    rdo_engine: rdo::RdoEngine,
    psycho_analyzer: psycho::PsychoAnalyzer,
    motion_optimizer: motion::MotionOptimizer,
    aq_engine: aq::AqEngine,
    roi_encoder: Option<roi_encode::RoiEncoder>,
    temporal_aq_bridge: Option<temporal_aq::TemporalAqBridge>,
}

impl Optimizer {
    /// Creates a new optimizer with the given configuration.
    pub fn new(config: OptimizerConfig) -> OxiResult<Self> {
        let rdo_engine = rdo::RdoEngine::new(&config)?;
        let psycho_analyzer = psycho::PsychoAnalyzer::new(&config)?;
        let motion_optimizer = motion::MotionOptimizer::new(&config)?;
        let aq_engine = aq::AqEngine::new(&config)?;

        let roi_encoder = if config.enable_roi {
            Some(roi_encode::RoiEncoder::new(
                roi_encode::RoiEncoderConfig::default(),
            ))
        } else {
            None
        };

        let temporal_aq_bridge = if config.enable_temporal_aq {
            Some(temporal_aq::TemporalAqBridge::with_defaults())
        } else {
            None
        };

        Ok(Self {
            config,
            rdo_engine,
            psycho_analyzer,
            motion_optimizer,
            aq_engine,
            roi_encoder,
            temporal_aq_bridge,
        })
    }

    /// Gets the optimizer configuration.
    #[must_use]
    pub fn config(&self) -> &OptimizerConfig {
        &self.config
    }

    /// Gets the RDO engine.
    #[must_use]
    pub fn rdo_engine(&self) -> &rdo::RdoEngine {
        &self.rdo_engine
    }

    /// Gets the psychovisual analyzer.
    #[must_use]
    pub fn psycho_analyzer(&self) -> &psycho::PsychoAnalyzer {
        &self.psycho_analyzer
    }

    /// Gets the motion optimizer.
    #[must_use]
    pub fn motion_optimizer(&self) -> &motion::MotionOptimizer {
        &self.motion_optimizer
    }

    /// Gets the adaptive quantization engine.
    #[must_use]
    pub fn aq_engine(&self) -> &aq::AqEngine {
        &self.aq_engine
    }

    /// Gets a mutable reference to the adaptive quantization engine.
    pub fn aq_engine_mut(&mut self) -> &mut aq::AqEngine {
        &mut self.aq_engine
    }

    /// Gets the ROI encoder, if enabled.
    #[must_use]
    pub fn roi_encoder(&self) -> Option<&roi_encode::RoiEncoder> {
        self.roi_encoder.as_ref()
    }

    /// Gets a mutable reference to the ROI encoder, if enabled.
    pub fn roi_encoder_mut(&mut self) -> Option<&mut roi_encode::RoiEncoder> {
        self.roi_encoder.as_mut()
    }

    /// Gets the temporal AQ bridge, if enabled.
    #[must_use]
    pub fn temporal_aq_bridge(&self) -> Option<&temporal_aq::TemporalAqBridge> {
        self.temporal_aq_bridge.as_ref()
    }

    /// Gets a mutable reference to the temporal AQ bridge, if enabled.
    pub fn temporal_aq_bridge_mut(&mut self) -> Option<&mut temporal_aq::TemporalAqBridge> {
        self.temporal_aq_bridge.as_mut()
    }

    /// Processes a frame's temporal activity through the temporal AQ bridge
    /// and returns the combined AQ result for a given spatial QP offset.
    ///
    /// This is the main integration point between temporal and spatial AQ.
    pub fn process_temporal_aq(
        &mut self,
        activity: temporal_aq::TemporalActivity,
        spatial_qp_offset: i8,
    ) -> Option<temporal_aq::CombinedAqResult> {
        if let Some(ref mut bridge) = self.temporal_aq_bridge {
            bridge.update_temporal(activity);
            Some(bridge.combine_with_spatial(spatial_qp_offset))
        } else {
            None
        }
    }

    /// Generates the ROI QP delta map for the current frame.
    ///
    /// Returns `None` if ROI encoding is not enabled or no regions are set.
    pub fn generate_roi_map(&self) -> Option<roi_encode::RoiOptimizeResult> {
        self.roi_encoder.as_ref().map(|enc| enc.optimize_frame())
    }

    /// Adds an ROI region to the encoder (if enabled).
    ///
    /// Returns `true` if the region was added, `false` if ROI is not enabled.
    pub fn add_roi_region(&mut self, region: roi_encode::RoiRegion) -> bool {
        if let Some(ref mut enc) = self.roi_encoder {
            enc.add_region(region);
            true
        } else {
            false
        }
    }

    /// Clears all ROI regions (if ROI is enabled).
    pub fn clear_roi_regions(&mut self) {
        if let Some(ref mut enc) = self.roi_encoder {
            enc.clear_regions();
        }
    }
}

// Re-export commonly used types
pub use aq::{AqEngine, AqMode, AqResult};
pub use benchmark::{BenchmarkConfig, BenchmarkResult, BenchmarkRunner, Profiler};
pub use decision::{
    DecisionContext, DecisionStrategy, ModeDecision, ReferenceDecision, SplitDecision,
};
pub use entropy::{ContextModel, ContextOptimizer, EntropyStats};
pub use filter::{DeblockOptimizer, FilterDecision, SaoOptimizer};
pub use gop_optimizer::{
    ContentAdaptiveGop, ContentGopDecision, GopOptimizer, GopPattern, GopPlan,
};
pub use intra::{AngleOptimizer, IntraModeDecision, ModeOptimizer};
pub use lookahead::{GopStructure, LookaheadAnalyzer, LookaheadFrame};
pub use motion::{
    BidirectionalOptimizer, MotionOptimizer, MotionSearchResult, MotionVector, MvPredictor,
    SubpelOptimizer,
};
pub use partition::{ComplexityAnalyzer, PartitionDecision, SplitOptimizer};
pub use presets::{OptimizationPresets, TunePresets};
pub use psycho::{ContrastSensitivity, PsychoAnalyzer, VisualMasking};
pub use rdo::{CostEstimate, LambdaCalculator, RdoEngine, RdoResult, RdoqOptimizer};
pub use reference::{DpbOptimizer, ReferenceSelection};
pub use roi_encode::{QpDeltaMap, RoiEncoder, RoiEncoderConfig, RoiOptimizeResult, RoiRegion};
pub use scene_encode::{LookaheadSceneQp, SceneEncodeParams, SceneEncoder, SceneMetrics};
pub use strategies::{
    BitrateAllocator, ContentAdaptiveOptimizer, OptimizationStrategy, StrategySelector,
    TemporalOptimizer,
};
pub use temporal_aq::{CombinedAqResult, TemporalActivity, TemporalAqBridge, TemporalAqEngine};
pub use transform::{QuantizationOptimizer, TransformSelection};
pub use utils::{BlockMetrics, FrameMetrics, OptimizationStats};
pub use vmaf_predict::{
    BatchVmafPredictor, VmafFeatures, VmafPrediction, VmafPredictor, VmafPredictorConfig,
};

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_optimizer_with_roi() {
        let config = OptimizerConfig {
            enable_roi: true,
            ..Default::default()
        };
        let mut optimizer = Optimizer::new(config).expect("Optimizer creation should succeed");

        assert!(optimizer.roi_encoder().is_some());
        assert!(
            optimizer.add_roi_region(roi_encode::RoiRegion::with_priority(0, 0, 128, 128, 2.0),)
        );

        let map = optimizer.generate_roi_map();
        assert!(map.is_some());
        let result = map.expect("ROI map should exist");
        assert!(result.has_active_regions);
    }

    #[test]
    fn test_optimizer_without_roi() {
        let config = OptimizerConfig::default();
        let mut optimizer = Optimizer::new(config).expect("Optimizer creation should succeed");

        assert!(optimizer.roi_encoder().is_none());
        assert!(!optimizer.add_roi_region(roi_encode::RoiRegion::new(0, 0, 64, 64),));
        assert!(optimizer.generate_roi_map().is_none());
    }

    #[test]
    fn test_optimizer_with_temporal_aq() {
        let config = OptimizerConfig {
            enable_temporal_aq: true,
            ..Default::default()
        };
        let mut optimizer = Optimizer::new(config).expect("Optimizer creation should succeed");

        assert!(optimizer.temporal_aq_bridge().is_some());

        let mut activity = temporal_aq::TemporalActivity::new(0);
        activity.avg_motion_magnitude = 30.0;
        activity.motion_coverage = 0.6;

        let result = optimizer.process_temporal_aq(activity, -2);
        assert!(result.is_some());
        let combined = result.expect("Temporal AQ result should exist");
        assert_ne!(combined.final_qp_delta, 0);
    }

    #[test]
    fn test_optimizer_without_temporal_aq() {
        let config = OptimizerConfig::default();
        let mut optimizer = Optimizer::new(config).expect("Optimizer creation should succeed");

        assert!(optimizer.temporal_aq_bridge().is_none());
        let result = optimizer.process_temporal_aq(temporal_aq::TemporalActivity::new(0), 0);
        assert!(result.is_none());
    }

    #[test]
    fn test_optimizer_full_pipeline() {
        let config = OptimizerConfig {
            enable_roi: true,
            enable_temporal_aq: true,
            enable_aq: true,
            enable_psychovisual: true,
            content_type: ContentType::Film,
            ..Default::default()
        };
        let mut optimizer = Optimizer::new(config).expect("Optimizer creation should succeed");

        // Add ROI region
        optimizer.add_roi_region(roi_encode::RoiRegion::with_priority(
            100, 100, 200, 200, 1.5,
        ));

        // Process temporal AQ
        let mut activity = temporal_aq::TemporalActivity::new(0);
        activity.avg_motion_magnitude = 15.0;
        activity.motion_coverage = 0.4;
        let temporal_result = optimizer.process_temporal_aq(activity, -1);
        assert!(temporal_result.is_some());

        // Generate ROI map
        let roi_result = optimizer.generate_roi_map();
        assert!(roi_result.is_some());

        // Verify AQ engine is accessible
        assert_eq!(optimizer.aq_engine().mode(), aq::AqMode::Combined);
    }

    #[test]
    fn test_clear_roi_regions() {
        let config = OptimizerConfig {
            enable_roi: true,
            ..Default::default()
        };
        let mut optimizer = Optimizer::new(config).expect("Optimizer creation should succeed");

        optimizer.add_roi_region(roi_encode::RoiRegion::new(0, 0, 64, 64));
        optimizer.clear_roi_regions();
        let result = optimizer
            .generate_roi_map()
            .expect("ROI map should exist even without regions");
        assert!(!result.has_active_regions);
    }
}
