//! Decoder pipeline for coordinating reconstruction stages.
//!
//! The `DecoderPipeline` coordinates all stages of video frame reconstruction,
//! from parsing through final output formatting.

#![forbid(unsafe_code)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::identity_op)]
#![allow(clippy::range_plus_one)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::useless_conversion)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::single_match_else)]
#![allow(dead_code)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::unused_self)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::similar_names)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::needless_pass_by_value)]

use super::{
    BufferPool, CdefApplicator, ChromaSubsampling, DeblockFilter, FilmGrainSynthesizer,
    FrameBuffer, LoopFilterPipeline, OutputFormatter, ReconstructResult, ReconstructionError,
    ResidualBuffer, SuperResUpscaler, MAX_BIT_DEPTH, MAX_FRAME_HEIGHT, MAX_FRAME_WIDTH,
    MIN_BIT_DEPTH, NUM_REF_FRAMES,
};

// =============================================================================
// Pipeline Stage
// =============================================================================

/// Pipeline processing stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PipelineStage {
    /// Bitstream parsing.
    Parse,
    /// Entropy decoding.
    Entropy,
    /// Intra/inter prediction.
    Predict,
    /// Inverse transform.
    Transform,
    /// Deblocking filter.
    Deblock,
    /// Loop filter (edge filtering).
    LoopFilter,
    /// CDEF (Constrained Directional Enhancement Filter).
    Cdef,
    /// Super-resolution upscaling.
    SuperRes,
    /// Film grain synthesis.
    FilmGrain,
    /// Output formatting.
    Output,
}

impl PipelineStage {
    /// Get the stage name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Parse => "Parse",
            Self::Entropy => "Entropy",
            Self::Predict => "Predict",
            Self::Transform => "Transform",
            Self::Deblock => "Deblock",
            Self::LoopFilter => "LoopFilter",
            Self::Cdef => "CDEF",
            Self::SuperRes => "SuperRes",
            Self::FilmGrain => "FilmGrain",
            Self::Output => "Output",
        }
    }

    /// Get all stages in processing order.
    #[must_use]
    pub const fn all_ordered() -> &'static [Self] {
        &[
            Self::Parse,
            Self::Entropy,
            Self::Predict,
            Self::Transform,
            Self::Deblock,
            Self::LoopFilter,
            Self::Cdef,
            Self::SuperRes,
            Self::FilmGrain,
            Self::Output,
        ]
    }

    /// Check if this stage is a filter stage.
    #[must_use]
    pub const fn is_filter(self) -> bool {
        matches!(
            self,
            Self::Deblock | Self::LoopFilter | Self::Cdef | Self::FilmGrain
        )
    }

    /// Check if this stage can be skipped.
    #[must_use]
    pub const fn is_optional(self) -> bool {
        matches!(
            self,
            Self::Deblock | Self::LoopFilter | Self::Cdef | Self::SuperRes | Self::FilmGrain
        )
    }
}

// =============================================================================
// Stage Result
// =============================================================================

/// Result from a pipeline stage.
#[derive(Clone, Debug)]
pub struct StageResult {
    /// The stage that produced this result.
    pub stage: PipelineStage,
    /// Processing time in microseconds.
    pub processing_time_us: u64,
    /// Whether the stage was skipped.
    pub skipped: bool,
    /// Additional metrics.
    pub metrics: StageMetrics,
}

impl StageResult {
    /// Create a new stage result.
    #[must_use]
    pub fn new(stage: PipelineStage) -> Self {
        Self {
            stage,
            processing_time_us: 0,
            skipped: false,
            metrics: StageMetrics::default(),
        }
    }

    /// Create a skipped stage result.
    #[must_use]
    pub fn skipped(stage: PipelineStage) -> Self {
        Self {
            stage,
            processing_time_us: 0,
            skipped: true,
            metrics: StageMetrics::default(),
        }
    }

    /// Set processing time.
    #[must_use]
    pub const fn with_time(mut self, time_us: u64) -> Self {
        self.processing_time_us = time_us;
        self
    }
}

/// Metrics from a pipeline stage.
#[derive(Clone, Debug, Default)]
pub struct StageMetrics {
    /// Number of blocks processed.
    pub blocks_processed: u64,
    /// Number of pixels processed.
    pub pixels_processed: u64,
    /// Number of operations performed.
    pub operations: u64,
}

// =============================================================================
// Pipeline Configuration
// =============================================================================

/// Configuration for the decoder pipeline.
#[derive(Clone, Debug)]
pub struct PipelineConfig {
    /// Enable deblocking filter.
    pub enable_deblock: bool,
    /// Enable loop filter.
    pub enable_loop_filter: bool,
    /// Enable CDEF.
    pub enable_cdef: bool,
    /// Enable super-resolution.
    pub enable_super_res: bool,
    /// Enable film grain synthesis.
    pub enable_film_grain: bool,
    /// Frame dimensions.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// Bit depth.
    pub bit_depth: u8,
    /// Chroma subsampling.
    pub subsampling: ChromaSubsampling,
    /// Number of threads for parallel processing.
    pub threads: usize,
    /// Buffer pool size.
    pub buffer_pool_size: usize,
    /// Enable performance metrics collection.
    pub collect_metrics: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            enable_deblock: true,
            enable_loop_filter: true,
            enable_cdef: true,
            enable_super_res: false,
            enable_film_grain: false,
            width: 1920,
            height: 1080,
            bit_depth: 8,
            subsampling: ChromaSubsampling::Cs420,
            threads: 1,
            buffer_pool_size: 4,
            collect_metrics: false,
        }
    }
}

impl PipelineConfig {
    /// Create a new pipeline configuration.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            ..Default::default()
        }
    }

    /// Set bit depth.
    #[must_use]
    pub const fn with_bit_depth(mut self, bit_depth: u8) -> Self {
        self.bit_depth = bit_depth;
        self
    }

    /// Set chroma subsampling.
    #[must_use]
    pub const fn with_subsampling(mut self, subsampling: ChromaSubsampling) -> Self {
        self.subsampling = subsampling;
        self
    }

    /// Enable all filters.
    #[must_use]
    pub const fn with_all_filters(mut self) -> Self {
        self.enable_deblock = true;
        self.enable_loop_filter = true;
        self.enable_cdef = true;
        self
    }

    /// Disable all filters.
    #[must_use]
    pub const fn without_filters(mut self) -> Self {
        self.enable_deblock = false;
        self.enable_loop_filter = false;
        self.enable_cdef = false;
        self.enable_film_grain = false;
        self
    }

    /// Set number of threads.
    #[must_use]
    pub const fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> ReconstructResult<()> {
        if self.width == 0 || self.width > MAX_FRAME_WIDTH {
            return Err(ReconstructionError::InvalidDimensions {
                width: self.width,
                height: self.height,
            });
        }

        if self.height == 0 || self.height > MAX_FRAME_HEIGHT {
            return Err(ReconstructionError::InvalidDimensions {
                width: self.width,
                height: self.height,
            });
        }

        if self.bit_depth < MIN_BIT_DEPTH || self.bit_depth > MAX_BIT_DEPTH {
            return Err(ReconstructionError::UnsupportedBitDepth(self.bit_depth));
        }

        Ok(())
    }
}

// =============================================================================
// Frame Context
// =============================================================================

/// Context for processing a single frame.
#[derive(Clone, Debug)]
pub struct FrameContext {
    /// Frame number in decode order.
    pub decode_order: u64,
    /// Frame number in display order.
    pub display_order: u64,
    /// Frame width (may differ from config for super-res).
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// Bit depth for this frame.
    pub bit_depth: u8,
    /// Is this a keyframe?
    pub is_keyframe: bool,
    /// Is this a show frame?
    pub show_frame: bool,
    /// Reference frame indices.
    pub ref_frame_indices: [Option<usize>; NUM_REF_FRAMES],
    /// Super-resolution scale factor (1.0 = no scaling).
    pub super_res_scale: f32,
    /// Film grain parameters present.
    pub has_film_grain: bool,
}

impl Default for FrameContext {
    fn default() -> Self {
        Self {
            decode_order: 0,
            display_order: 0,
            width: 0,
            height: 0,
            bit_depth: 8,
            is_keyframe: true,
            show_frame: true,
            ref_frame_indices: [None; NUM_REF_FRAMES],
            super_res_scale: 1.0,
            has_film_grain: false,
        }
    }
}

impl FrameContext {
    /// Create a new frame context.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            ..Default::default()
        }
    }

    /// Check if super-resolution is needed.
    #[must_use]
    pub fn needs_super_res(&self) -> bool {
        (self.super_res_scale - 1.0).abs() > f32::EPSILON
    }
}

// =============================================================================
// Decoder Pipeline
// =============================================================================

/// Main decoder pipeline coordinating all reconstruction stages.
#[derive(Debug)]
pub struct DecoderPipeline {
    /// Pipeline configuration.
    config: PipelineConfig,
    /// Buffer pool for frame allocation.
    buffer_pool: BufferPool,
    /// Reference frame manager.
    reference_frames: Vec<Option<FrameBuffer>>,
    /// Deblocking filter.
    deblock_filter: DeblockFilter,
    /// Loop filter pipeline.
    loop_filter: LoopFilterPipeline,
    /// CDEF applicator.
    cdef: CdefApplicator,
    /// Super-resolution upscaler.
    super_res: SuperResUpscaler,
    /// Film grain synthesizer.
    film_grain: FilmGrainSynthesizer,
    /// Output formatter.
    output: OutputFormatter,
    /// Current working buffer.
    work_buffer: Option<FrameBuffer>,
    /// Residual buffer.
    residual_buffer: ResidualBuffer,
    /// Frame counter.
    frame_count: u64,
    /// Stage results for last frame.
    last_stage_results: Vec<StageResult>,
}

impl DecoderPipeline {
    /// Create a new decoder pipeline.
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid.
    pub fn new(config: PipelineConfig) -> ReconstructResult<Self> {
        config.validate()?;

        let buffer_pool = BufferPool::new(
            config.width,
            config.height,
            config.bit_depth,
            config.subsampling,
            config.buffer_pool_size,
        );

        let reference_frames = vec![None; NUM_REF_FRAMES];
        let residual_buffer = ResidualBuffer::new(config.width, config.height, config.subsampling);

        Ok(Self {
            config: config.clone(),
            buffer_pool,
            reference_frames,
            deblock_filter: DeblockFilter::new(),
            loop_filter: LoopFilterPipeline::new(),
            cdef: CdefApplicator::new(config.width, config.height, config.bit_depth),
            super_res: SuperResUpscaler::new(),
            film_grain: FilmGrainSynthesizer::new(config.bit_depth),
            output: OutputFormatter::new(),
            work_buffer: None,
            residual_buffer,
            frame_count: 0,
            last_stage_results: Vec::new(),
        })
    }

    /// Process a complete frame through the pipeline.
    ///
    /// # Errors
    ///
    /// Returns error if any pipeline stage fails.
    pub fn process_frame(
        &mut self,
        data: &[u8],
        context: &FrameContext,
    ) -> ReconstructResult<FrameBuffer> {
        self.last_stage_results.clear();

        // Allocate work buffer
        self.work_buffer = Some(self.buffer_pool.acquire()?);

        // Process each stage
        self.stage_parse(data, context)?;
        self.stage_entropy(context)?;
        self.stage_predict(context)?;
        self.stage_transform(context)?;
        self.stage_deblock(context)?;
        self.stage_loop_filter(context)?;
        self.stage_cdef(context)?;
        self.stage_super_res(context)?;
        self.stage_film_grain(context)?;

        // Get the final buffer
        let output = self.work_buffer.take().ok_or_else(|| {
            ReconstructionError::Internal("Work buffer not available".to_string())
        })?;

        // Update reference frames if this is a reference frame
        if context.show_frame || context.is_keyframe {
            self.update_references(&output, context);
        }

        self.frame_count += 1;

        Ok(output)
    }

    /// Parse stage - parse bitstream data.
    fn stage_parse(&mut self, _data: &[u8], _context: &FrameContext) -> ReconstructResult<()> {
        let result = StageResult::new(PipelineStage::Parse);
        self.last_stage_results.push(result);
        Ok(())
    }

    /// Entropy decoding stage.
    fn stage_entropy(&mut self, _context: &FrameContext) -> ReconstructResult<()> {
        let result = StageResult::new(PipelineStage::Entropy);
        self.last_stage_results.push(result);
        Ok(())
    }

    /// Prediction stage - intra and inter prediction.
    fn stage_predict(&mut self, _context: &FrameContext) -> ReconstructResult<()> {
        let result = StageResult::new(PipelineStage::Predict);
        self.last_stage_results.push(result);
        Ok(())
    }

    /// Transform stage - inverse transform and residual addition.
    fn stage_transform(&mut self, _context: &FrameContext) -> ReconstructResult<()> {
        let result = StageResult::new(PipelineStage::Transform);
        self.last_stage_results.push(result);
        Ok(())
    }

    /// Deblocking filter stage.
    fn stage_deblock(&mut self, context: &FrameContext) -> ReconstructResult<()> {
        if !self.config.enable_deblock {
            self.last_stage_results
                .push(StageResult::skipped(PipelineStage::Deblock));
            return Ok(());
        }

        if let Some(ref mut buffer) = self.work_buffer {
            self.deblock_filter.apply(buffer, context)?;
        }

        let result = StageResult::new(PipelineStage::Deblock);
        self.last_stage_results.push(result);
        Ok(())
    }

    /// Loop filter stage.
    fn stage_loop_filter(&mut self, context: &FrameContext) -> ReconstructResult<()> {
        if !self.config.enable_loop_filter {
            self.last_stage_results
                .push(StageResult::skipped(PipelineStage::LoopFilter));
            return Ok(());
        }

        if let Some(ref mut buffer) = self.work_buffer {
            self.loop_filter.apply(buffer, context)?;
        }

        let result = StageResult::new(PipelineStage::LoopFilter);
        self.last_stage_results.push(result);
        Ok(())
    }

    /// CDEF stage.
    fn stage_cdef(&mut self, context: &FrameContext) -> ReconstructResult<()> {
        if !self.config.enable_cdef {
            self.last_stage_results
                .push(StageResult::skipped(PipelineStage::Cdef));
            return Ok(());
        }

        if let Some(ref mut buffer) = self.work_buffer {
            self.cdef.apply(buffer, context)?;
        }

        let result = StageResult::new(PipelineStage::Cdef);
        self.last_stage_results.push(result);
        Ok(())
    }

    /// Super-resolution stage.
    fn stage_super_res(&mut self, context: &FrameContext) -> ReconstructResult<()> {
        if !self.config.enable_super_res || !context.needs_super_res() {
            self.last_stage_results
                .push(StageResult::skipped(PipelineStage::SuperRes));
            return Ok(());
        }

        if let Some(ref mut buffer) = self.work_buffer {
            self.super_res.apply(buffer, context)?;
        }

        let result = StageResult::new(PipelineStage::SuperRes);
        self.last_stage_results.push(result);
        Ok(())
    }

    /// Film grain synthesis stage.
    fn stage_film_grain(&mut self, context: &FrameContext) -> ReconstructResult<()> {
        if !self.config.enable_film_grain || !context.has_film_grain {
            self.last_stage_results
                .push(StageResult::skipped(PipelineStage::FilmGrain));
            return Ok(());
        }

        if let Some(ref mut buffer) = self.work_buffer {
            self.film_grain.apply(buffer, context)?;
        }

        let result = StageResult::new(PipelineStage::FilmGrain);
        self.last_stage_results.push(result);
        Ok(())
    }

    /// Update reference frame storage.
    fn update_references(&mut self, frame: &FrameBuffer, context: &FrameContext) {
        // For keyframes, clear all references
        if context.is_keyframe {
            for ref_frame in &mut self.reference_frames {
                *ref_frame = None;
            }
        }

        // Store in the first available slot or overwrite oldest
        // In a full implementation, this would use ref_frame_indices from context
        let slot = (self.frame_count as usize) % NUM_REF_FRAMES;
        self.reference_frames[slot] = Some(frame.clone());
    }

    /// Get a reference frame by index.
    pub fn get_reference(&self, index: usize) -> ReconstructResult<&FrameBuffer> {
        self.reference_frames
            .get(index)
            .and_then(|r| r.as_ref())
            .ok_or(ReconstructionError::ReferenceNotAvailable(index))
    }

    /// Get the pipeline configuration.
    #[must_use]
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// Get the number of frames processed.
    #[must_use]
    pub const fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Get results from the last frame's stages.
    #[must_use]
    pub fn last_stage_results(&self) -> &[StageResult] {
        &self.last_stage_results
    }

    /// Reset the pipeline state.
    pub fn reset(&mut self) {
        self.frame_count = 0;
        self.work_buffer = None;
        self.last_stage_results.clear();
        for ref_frame in &mut self.reference_frames {
            *ref_frame = None;
        }
        self.buffer_pool.reset();
        self.residual_buffer.clear();
    }

    /// Reconfigure the pipeline with new settings.
    ///
    /// # Errors
    ///
    /// Returns error if the new configuration is invalid.
    pub fn reconfigure(&mut self, config: PipelineConfig) -> ReconstructResult<()> {
        config.validate()?;

        // Check if dimensions changed
        let dimensions_changed =
            self.config.width != config.width || self.config.height != config.height;

        self.config = config.clone();

        if dimensions_changed {
            self.buffer_pool = BufferPool::new(
                config.width,
                config.height,
                config.bit_depth,
                config.subsampling,
                config.buffer_pool_size,
            );
            self.residual_buffer =
                ResidualBuffer::new(config.width, config.height, config.subsampling);
            self.cdef = CdefApplicator::new(config.width, config.height, config.bit_depth);
        }

        Ok(())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_stage_name() {
        assert_eq!(PipelineStage::Parse.name(), "Parse");
        assert_eq!(PipelineStage::Cdef.name(), "CDEF");
        assert_eq!(PipelineStage::SuperRes.name(), "SuperRes");
    }

    #[test]
    fn test_pipeline_stage_is_filter() {
        assert!(!PipelineStage::Parse.is_filter());
        assert!(PipelineStage::Deblock.is_filter());
        assert!(PipelineStage::LoopFilter.is_filter());
        assert!(PipelineStage::Cdef.is_filter());
    }

    #[test]
    fn test_pipeline_stage_is_optional() {
        assert!(!PipelineStage::Parse.is_optional());
        assert!(!PipelineStage::Entropy.is_optional());
        assert!(PipelineStage::Deblock.is_optional());
        assert!(PipelineStage::SuperRes.is_optional());
        assert!(PipelineStage::FilmGrain.is_optional());
    }

    #[test]
    fn test_stage_result() {
        let result = StageResult::new(PipelineStage::Parse);
        assert_eq!(result.stage, PipelineStage::Parse);
        assert!(!result.skipped);

        let skipped = StageResult::skipped(PipelineStage::Cdef);
        assert!(skipped.skipped);
    }

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.bit_depth, 8);
        assert!(config.enable_deblock);
        assert!(config.enable_loop_filter);
        assert!(config.enable_cdef);
    }

    #[test]
    fn test_pipeline_config_builder() {
        let config = PipelineConfig::new(1280, 720)
            .with_bit_depth(10)
            .with_subsampling(ChromaSubsampling::Cs422)
            .with_threads(4);

        assert_eq!(config.width, 1280);
        assert_eq!(config.height, 720);
        assert_eq!(config.bit_depth, 10);
        assert_eq!(config.subsampling, ChromaSubsampling::Cs422);
        assert_eq!(config.threads, 4);
    }

    #[test]
    fn test_pipeline_config_validation() {
        let valid = PipelineConfig::new(1920, 1080);
        assert!(valid.validate().is_ok());

        let invalid_width = PipelineConfig::new(0, 1080);
        assert!(invalid_width.validate().is_err());

        let invalid_height = PipelineConfig::new(1920, 0);
        assert!(invalid_height.validate().is_err());

        let invalid_bit_depth = PipelineConfig::new(1920, 1080).with_bit_depth(7);
        assert!(invalid_bit_depth.validate().is_err());
    }

    #[test]
    fn test_frame_context_default() {
        let ctx = FrameContext::default();
        assert_eq!(ctx.width, 0);
        assert_eq!(ctx.height, 0);
        assert!(ctx.is_keyframe);
        assert!(ctx.show_frame);
    }

    #[test]
    fn test_frame_context_super_res() {
        let mut ctx = FrameContext::new(1920, 1080);
        assert!(!ctx.needs_super_res());

        ctx.super_res_scale = 1.5;
        assert!(ctx.needs_super_res());
    }

    #[test]
    fn test_decoder_pipeline_creation() {
        let config = PipelineConfig::new(1920, 1080);
        let pipeline = DecoderPipeline::new(config);
        assert!(pipeline.is_ok());
    }

    #[test]
    fn test_decoder_pipeline_process_frame() {
        let config = PipelineConfig::new(64, 64).without_filters();
        let mut pipeline = DecoderPipeline::new(config).expect("should succeed");

        let context = FrameContext::new(64, 64);
        let result = pipeline.process_frame(&[], &context);
        assert!(result.is_ok());

        assert_eq!(pipeline.frame_count(), 1);
    }

    #[test]
    fn test_decoder_pipeline_reset() {
        let config = PipelineConfig::new(64, 64).without_filters();
        let mut pipeline = DecoderPipeline::new(config).expect("should succeed");

        let context = FrameContext::new(64, 64);
        let _ = pipeline.process_frame(&[], &context);

        pipeline.reset();
        assert_eq!(pipeline.frame_count(), 0);
    }

    #[test]
    fn test_decoder_pipeline_reconfigure() {
        let config = PipelineConfig::new(64, 64);
        let mut pipeline = DecoderPipeline::new(config).expect("should succeed");

        let new_config = PipelineConfig::new(128, 128);
        assert!(pipeline.reconfigure(new_config).is_ok());
        assert_eq!(pipeline.config().width, 128);
        assert_eq!(pipeline.config().height, 128);
    }

    #[test]
    fn test_stage_results() {
        let config = PipelineConfig::new(64, 64).without_filters();
        let mut pipeline = DecoderPipeline::new(config).expect("should succeed");

        let context = FrameContext::new(64, 64);
        let _ = pipeline.process_frame(&[], &context);

        let results = pipeline.last_stage_results();
        assert!(!results.is_empty());

        // Check that filter stages are skipped
        for result in results {
            if result.stage.is_filter() {
                assert!(result.skipped);
            }
        }
    }
}
