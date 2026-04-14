//! Frame-level decode -> filter -> encode pipeline execution.
//!
//! This module provides `PipelineExecutor`, which pulls frames from a decoder
//! abstraction, passes them through a sequence of `PipelineStageProcessor`
//! nodes, and pushes them to an encoder. It handles timestamp management,
//! frame rate conversion boundaries, and collects execution statistics.
//!
//! # Architecture
//!
//! ```text
//!   Decoder -> [Stage 0] -> [Stage 1] -> ... -> [Stage N] -> Encoder
//! ```
//!
//! Each stage implements `PipelineStageProcessor` and can transform, drop,
//! or duplicate frames. The executor manages the data flow and collects
//! per-stage timing statistics.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

use std::time::{Duration, Instant};

use crate::Result;

// ─── Frame representation ────────────────────────────────────────────────────

/// A frame flowing through the pipeline executor.
///
/// Carries raw media data with presentation timestamp and frame metadata.
#[derive(Debug, Clone)]
pub struct PipelineFrame {
    /// Raw data (planar YUV, interleaved PCM, RGBA, etc.).
    pub data: Vec<u8>,
    /// Presentation timestamp in microseconds from stream start.
    pub pts_us: i64,
    /// Duration of this frame in microseconds (0 if unknown).
    pub duration_us: i64,
    /// Frame width in pixels (0 for audio).
    pub width: u32,
    /// Frame height in pixels (0 for audio).
    pub height: u32,
    /// Whether this is an audio frame.
    pub is_audio: bool,
    /// Frame sequence number (monotonically increasing per stream).
    pub sequence: u64,
    /// Whether this is a keyframe / sync point.
    pub is_keyframe: bool,
}

impl PipelineFrame {
    /// Creates a new video frame.
    #[must_use]
    pub fn video(data: Vec<u8>, pts_us: i64, width: u32, height: u32) -> Self {
        Self {
            data,
            pts_us,
            duration_us: 0,
            width,
            height,
            is_audio: false,
            sequence: 0,
            is_keyframe: false,
        }
    }

    /// Creates a new audio frame.
    #[must_use]
    pub fn audio(data: Vec<u8>, pts_us: i64) -> Self {
        Self {
            data,
            pts_us,
            duration_us: 0,
            width: 0,
            height: 0,
            is_audio: true,
            sequence: 0,
            is_keyframe: false,
        }
    }

    /// Sets the duration (builder-style).
    #[must_use]
    pub fn with_duration(mut self, duration_us: i64) -> Self {
        self.duration_us = duration_us;
        self
    }

    /// Sets the sequence number (builder-style).
    #[must_use]
    pub fn with_sequence(mut self, seq: u64) -> Self {
        self.sequence = seq;
        self
    }

    /// Sets the keyframe flag (builder-style).
    #[must_use]
    pub fn with_keyframe(mut self, kf: bool) -> Self {
        self.is_keyframe = kf;
        self
    }

    /// Returns the PTS in seconds.
    #[must_use]
    pub fn pts_secs(&self) -> f64 {
        self.pts_us as f64 / 1_000_000.0
    }
}

// ─── Pipeline stage trait ────────────────────────────────────────────────────

/// Outcome of processing a frame through a pipeline stage.
#[derive(Debug)]
pub enum StageOutput {
    /// Pass the (possibly modified) frame to the next stage.
    Pass(PipelineFrame),
    /// The stage consumed the frame; nothing is forwarded.
    Drop,
    /// The stage produced multiple output frames from a single input.
    Multiple(Vec<PipelineFrame>),
}

/// A single processing stage in the pipeline.
///
/// Implementations receive one frame at a time and return a `StageOutput`
/// indicating how to proceed.
pub trait PipelineStageProcessor: Send {
    /// Human-readable name of this stage (used in stats).
    fn name(&self) -> &str;

    /// Process a single frame.
    ///
    /// # Errors
    ///
    /// Returns an error if the processing fails irrecoverably.
    fn process(&mut self, frame: PipelineFrame) -> Result<StageOutput>;

    /// Called when the input stream is exhausted so the stage can flush
    /// any internally-buffered frames.
    ///
    /// The default implementation returns an empty vec.
    fn flush(&mut self) -> Result<Vec<PipelineFrame>> {
        Ok(Vec::new())
    }
}

// ─── Decoder / Encoder abstractions ──────────────────────────────────────────

/// Abstraction over a frame decoder feeding the pipeline.
pub trait PipelineDecoder: Send {
    /// Decode and return the next frame, or `None` at end-of-stream.
    fn next_frame(&mut self) -> Option<PipelineFrame>;

    /// Returns true when the stream is fully consumed.
    fn eof(&self) -> bool;
}

/// Abstraction over a frame encoder consuming the pipeline output.
pub trait PipelineEncoder: Send {
    /// Encode a single frame.
    ///
    /// # Errors
    ///
    /// Returns an error if encoding fails.
    fn encode(&mut self, frame: &PipelineFrame) -> Result<Vec<u8>>;

    /// Flush any internally-buffered data at end-of-stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the flush fails.
    fn flush(&mut self) -> Result<Vec<u8>>;
}

// ─── Execution statistics ────────────────────────────────────────────────────

/// Per-stage timing and frame statistics.
#[derive(Debug, Clone)]
pub struct StageStats {
    /// Stage name.
    pub name: String,
    /// Total wall-clock time spent inside this stage.
    pub total_time: Duration,
    /// Number of frames that entered this stage.
    pub frames_in: u64,
    /// Number of frames that left this stage.
    pub frames_out: u64,
}

/// Aggregate execution statistics for the full pipeline run.
#[derive(Debug, Clone, Default)]
pub struct ExecutionStats {
    /// Per-stage statistics (in pipeline order).
    pub stages: Vec<StageStats>,
    /// Total frames decoded from the source.
    pub total_decoded: u64,
    /// Total frames passed to the encoder.
    pub total_encoded: u64,
    /// Total encoded bytes produced.
    pub total_bytes: u64,
    /// Total wall-clock time for the full pipeline.
    pub wall_time: Duration,
    /// Number of frames dropped by stages.
    pub frames_dropped: u64,
}

impl ExecutionStats {
    /// Returns throughput in frames per second.
    #[must_use]
    pub fn fps(&self) -> f64 {
        let secs = self.wall_time.as_secs_f64();
        if secs > 0.0 {
            self.total_encoded as f64 / secs
        } else {
            0.0
        }
    }

    /// Returns the speed factor given the content duration.
    #[must_use]
    pub fn speed_factor(&self, content_duration_secs: f64) -> f64 {
        let secs = self.wall_time.as_secs_f64();
        if secs > 0.0 && content_duration_secs > 0.0 {
            content_duration_secs / secs
        } else {
            1.0
        }
    }
}

// ─── Timestamp manager ───────────────────────────────────────────────────────

/// Manages timestamp re-mapping when the output frame rate differs from input.
#[derive(Debug, Clone)]
pub struct TimestampManager {
    /// Input frame rate as (numerator, denominator).
    input_fps: (u32, u32),
    /// Output frame rate as (numerator, denominator).
    output_fps: (u32, u32),
    /// Accumulated output frame count for PTS generation.
    output_frame_count: u64,
    /// Last input PTS seen (for monotonicity checks).
    last_input_pts: i64,
}

impl TimestampManager {
    /// Creates a new timestamp manager.
    ///
    /// `input_fps` and `output_fps` are `(numerator, denominator)` pairs.
    #[must_use]
    pub fn new(input_fps: (u32, u32), output_fps: (u32, u32)) -> Self {
        Self {
            input_fps,
            output_fps,
            output_frame_count: 0,
            last_input_pts: i64::MIN,
        }
    }

    /// Creates a passthrough manager (no frame rate conversion).
    #[must_use]
    pub fn passthrough() -> Self {
        Self::new((30, 1), (30, 1))
    }

    /// Returns whether frame rate conversion is needed.
    #[must_use]
    pub fn needs_conversion(&self) -> bool {
        let in_rate = self.input_rate();
        let out_rate = self.output_rate();
        (in_rate - out_rate).abs() > 0.001
    }

    /// Returns the input frame rate as f64.
    #[must_use]
    pub fn input_rate(&self) -> f64 {
        if self.input_fps.1 == 0 {
            return 0.0;
        }
        f64::from(self.input_fps.0) / f64::from(self.input_fps.1)
    }

    /// Returns the output frame rate as f64.
    #[must_use]
    pub fn output_rate(&self) -> f64 {
        if self.output_fps.1 == 0 {
            return 0.0;
        }
        f64::from(self.output_fps.0) / f64::from(self.output_fps.1)
    }

    /// Computes the output PTS for the next frame and advances the counter.
    ///
    /// If frame rate conversion is disabled, returns the input PTS unchanged.
    pub fn map_pts(&mut self, input_pts_us: i64) -> i64 {
        self.last_input_pts = input_pts_us;

        if !self.needs_conversion() {
            self.output_frame_count += 1;
            return input_pts_us;
        }

        let out_rate = self.output_rate();
        if out_rate <= 0.0 {
            self.output_frame_count += 1;
            return input_pts_us;
        }

        let pts = (self.output_frame_count as f64 / out_rate * 1_000_000.0) as i64;
        self.output_frame_count += 1;
        pts
    }

    /// Returns the duration of one output frame in microseconds.
    #[must_use]
    pub fn output_frame_duration_us(&self) -> i64 {
        let rate = self.output_rate();
        if rate <= 0.0 {
            return 33_333; // default ~30fps
        }
        (1_000_000.0 / rate) as i64
    }

    /// Determines how many output frames should be produced for a given
    /// input frame boundary.
    ///
    /// For up-conversion (e.g. 24fps -> 60fps) this may return > 1.
    /// For down-conversion (e.g. 60fps -> 24fps) this may return 0 for
    /// some input frames.
    #[must_use]
    pub fn frames_at_boundary(&self, input_frame_index: u64) -> u32 {
        if !self.needs_conversion() {
            return 1;
        }

        let in_rate = self.input_rate();
        let out_rate = self.output_rate();
        if in_rate <= 0.0 || out_rate <= 0.0 {
            return 1;
        }

        let ratio = out_rate / in_rate;

        if ratio >= 1.0 {
            // Up-conversion: how many output frames cover this input frame?
            let start = (input_frame_index as f64 * ratio).floor() as u64;
            let end = ((input_frame_index + 1) as f64 * ratio).floor() as u64;
            let count = end.saturating_sub(start);
            count.min(u64::from(u32::MAX)) as u32
        } else {
            // Down-conversion: does this input frame produce an output frame?
            let out_idx = (input_frame_index as f64 * ratio).floor() as u64;
            let prev_out_idx = if input_frame_index > 0 {
                ((input_frame_index - 1) as f64 * ratio).floor() as u64
            } else {
                u64::MAX // ensure first frame always produces output
            };
            if out_idx != prev_out_idx { 1 } else { 0 }
        }
    }

    /// Returns the number of output frames produced so far.
    #[must_use]
    pub fn output_frame_count(&self) -> u64 {
        self.output_frame_count
    }
}

// ─── Frame rate conversion stage ─────────────────────────────────────────────

/// A pipeline stage that handles frame rate conversion.
///
/// For up-conversion, duplicates frames. For down-conversion, drops frames.
/// For matching rates, passes through unchanged.
pub struct FrameRateConverter {
    ts_manager: TimestampManager,
    input_frame_index: u64,
}

impl FrameRateConverter {
    /// Creates a new frame rate converter.
    #[must_use]
    pub fn new(input_fps: (u32, u32), output_fps: (u32, u32)) -> Self {
        Self {
            ts_manager: TimestampManager::new(input_fps, output_fps),
            input_frame_index: 0,
        }
    }
}

impl PipelineStageProcessor for FrameRateConverter {
    fn name(&self) -> &str {
        "frame_rate_converter"
    }

    fn process(&mut self, frame: PipelineFrame) -> Result<StageOutput> {
        // Audio frames pass through without rate conversion.
        if frame.is_audio {
            return Ok(StageOutput::Pass(frame));
        }

        let count = self.ts_manager.frames_at_boundary(self.input_frame_index);
        self.input_frame_index += 1;

        match count {
            0 => Ok(StageOutput::Drop),
            1 => {
                let mut out = frame;
                out.pts_us = self.ts_manager.map_pts(out.pts_us);
                out.duration_us = self.ts_manager.output_frame_duration_us();
                Ok(StageOutput::Pass(out))
            }
            n => {
                let mut frames = Vec::with_capacity(n as usize);
                for _ in 0..n {
                    let mut dup = frame.clone();
                    dup.pts_us = self.ts_manager.map_pts(dup.pts_us);
                    dup.duration_us = self.ts_manager.output_frame_duration_us();
                    frames.push(dup);
                }
                Ok(StageOutput::Multiple(frames))
            }
        }
    }
}

// ─── Passthrough stage ───────────────────────────────────────────────────────

/// A no-op stage that passes frames through unchanged (useful for testing).
pub struct PassthroughStage;

impl PipelineStageProcessor for PassthroughStage {
    fn name(&self) -> &str {
        "passthrough"
    }

    fn process(&mut self, frame: PipelineFrame) -> Result<StageOutput> {
        Ok(StageOutput::Pass(frame))
    }
}

// ─── Pipeline executor ──────────────────────────────────────────────────────

/// Configuration for the pipeline executor.
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    /// Maximum number of frames to process (0 = unlimited).
    pub max_frames: u64,
    /// Whether to collect per-stage timing statistics.
    pub collect_stage_stats: bool,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            max_frames: 0,
            collect_stage_stats: true,
        }
    }
}

/// Frame-level decode -> filter -> encode pipeline executor.
///
/// Pulls frames from a `PipelineDecoder`, passes them through a chain of
/// `PipelineStageProcessor` nodes, then pushes to a `PipelineEncoder`.
pub struct PipelineExecutor {
    decoder: Box<dyn PipelineDecoder>,
    stages: Vec<Box<dyn PipelineStageProcessor>>,
    encoder: Box<dyn PipelineEncoder>,
    config: ExecutorConfig,
}

impl PipelineExecutor {
    /// Creates a new pipeline executor.
    pub fn new(
        decoder: Box<dyn PipelineDecoder>,
        encoder: Box<dyn PipelineEncoder>,
    ) -> Self {
        Self {
            decoder,
            stages: Vec::new(),
            encoder,
            config: ExecutorConfig::default(),
        }
    }

    /// Sets the executor configuration.
    #[must_use]
    pub fn with_config(mut self, config: ExecutorConfig) -> Self {
        self.config = config;
        self
    }

    /// Adds a processing stage to the pipeline.
    pub fn add_stage(&mut self, stage: Box<dyn PipelineStageProcessor>) {
        self.stages.push(stage);
    }

    /// Adds a processing stage (builder-style).
    #[must_use]
    pub fn with_stage(mut self, stage: Box<dyn PipelineStageProcessor>) -> Self {
        self.stages.push(stage);
        self
    }

    /// Returns the number of stages.
    #[must_use]
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Executes the full pipeline: decode -> stages -> encode.
    ///
    /// # Errors
    ///
    /// Returns an error if decoding, any stage processing, or encoding fails.
    pub fn execute(&mut self) -> Result<ExecutionStats> {
        let start = Instant::now();

        let n_stages = self.stages.len();
        let mut stage_times = vec![Duration::ZERO; n_stages];
        let mut stage_in = vec![0u64; n_stages];
        let mut stage_out = vec![0u64; n_stages];

        let mut total_decoded: u64 = 0;
        let mut total_encoded: u64 = 0;
        let mut total_bytes: u64 = 0;
        let mut frames_dropped: u64 = 0;

        // Main decode loop.
        while let Some(frame) = self.decoder.next_frame() {
            if self.config.max_frames > 0 && total_decoded >= self.config.max_frames {
                break;
            }

            total_decoded += 1;

            // Run through stages.
            let mut current_frames = vec![frame];

            for (i, stage) in self.stages.iter_mut().enumerate() {
                let mut next_frames = Vec::new();

                for f in current_frames {
                    stage_in[i] += 1;
                    let t0 = Instant::now();
                    let output = stage.process(f)?;
                    if self.config.collect_stage_stats {
                        stage_times[i] += t0.elapsed();
                    }

                    match output {
                        StageOutput::Pass(out) => {
                            stage_out[i] += 1;
                            next_frames.push(out);
                        }
                        StageOutput::Drop => {
                            frames_dropped += 1;
                        }
                        StageOutput::Multiple(multi) => {
                            stage_out[i] += multi.len() as u64;
                            next_frames.extend(multi);
                        }
                    }
                }

                current_frames = next_frames;
            }

            // Encode surviving frames.
            for f in &current_frames {
                let encoded = self.encoder.encode(f)?;
                total_bytes += encoded.len() as u64;
                total_encoded += 1;
            }
        }

        // Flush stages.
        for (i, stage) in self.stages.iter_mut().enumerate() {
            let flushed = stage.flush()?;
            for f in &flushed {
                stage_out[i] += 1;
                let encoded = self.encoder.encode(f)?;
                total_bytes += encoded.len() as u64;
                total_encoded += 1;
            }
        }

        // Flush encoder.
        let tail = self.encoder.flush()?;
        total_bytes += tail.len() as u64;

        // Build stats.
        let mut stage_stats = Vec::with_capacity(n_stages);
        for i in 0..n_stages {
            stage_stats.push(StageStats {
                name: self.stages[i].name().to_string(),
                total_time: stage_times[i],
                frames_in: stage_in[i],
                frames_out: stage_out[i],
            });
        }

        Ok(ExecutionStats {
            stages: stage_stats,
            total_decoded,
            total_encoded,
            total_bytes,
            wall_time: start.elapsed(),
            frames_dropped,
        })
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Test decoder / encoder ───────────────────────────────────────────

    struct MockDecoder {
        frames: Vec<PipelineFrame>,
        index: usize,
    }

    impl MockDecoder {
        fn new(frames: Vec<PipelineFrame>) -> Self {
            Self { frames, index: 0 }
        }
    }

    impl PipelineDecoder for MockDecoder {
        fn next_frame(&mut self) -> Option<PipelineFrame> {
            if self.index < self.frames.len() {
                let f = self.frames[self.index].clone();
                self.index += 1;
                Some(f)
            } else {
                None
            }
        }

        fn eof(&self) -> bool {
            self.index >= self.frames.len()
        }
    }

    struct MockEncoder {
        encoded: Vec<Vec<u8>>,
    }

    impl MockEncoder {
        fn new() -> Self {
            Self {
                encoded: Vec::new(),
            }
        }
    }

    impl PipelineEncoder for MockEncoder {
        fn encode(&mut self, frame: &PipelineFrame) -> Result<Vec<u8>> {
            let data = frame.data.clone();
            self.encoded.push(data.clone());
            Ok(data)
        }

        fn flush(&mut self) -> Result<Vec<u8>> {
            Ok(Vec::new())
        }
    }

    /// A stage that doubles the pixel values (for testing).
    struct DoublerStage;

    impl PipelineStageProcessor for DoublerStage {
        fn name(&self) -> &str {
            "doubler"
        }

        fn process(&mut self, mut frame: PipelineFrame) -> Result<StageOutput> {
            for byte in &mut frame.data {
                *byte = byte.saturating_mul(2);
            }
            Ok(StageOutput::Pass(frame))
        }
    }

    /// A stage that drops every other frame.
    struct DropEveryOtherStage {
        count: u64,
    }

    impl DropEveryOtherStage {
        fn new() -> Self {
            Self { count: 0 }
        }
    }

    impl PipelineStageProcessor for DropEveryOtherStage {
        fn name(&self) -> &str {
            "drop_every_other"
        }

        fn process(&mut self, frame: PipelineFrame) -> Result<StageOutput> {
            self.count += 1;
            if self.count % 2 == 0 {
                Ok(StageOutput::Drop)
            } else {
                Ok(StageOutput::Pass(frame))
            }
        }
    }

    fn make_test_frames(n: usize) -> Vec<PipelineFrame> {
        (0..n)
            .map(|i| {
                PipelineFrame::video(vec![10, 20, 30, 40], (i as i64) * 33_333, 2, 2)
                    .with_sequence(i as u64)
            })
            .collect()
    }

    // ── Tests ────────────────────────────────────────────────────────────

    #[test]
    fn test_passthrough_pipeline() {
        let frames = make_test_frames(5);
        let decoder = Box::new(MockDecoder::new(frames));
        let encoder = Box::new(MockEncoder::new());

        let mut executor = PipelineExecutor::new(decoder, encoder);
        let stats = executor.execute().expect("pipeline should succeed");

        assert_eq!(stats.total_decoded, 5);
        assert_eq!(stats.total_encoded, 5);
        assert_eq!(stats.frames_dropped, 0);
        assert!(stats.stages.is_empty());
    }

    #[test]
    fn test_pipeline_with_stage() {
        let frames = make_test_frames(4);
        let decoder = Box::new(MockDecoder::new(frames));
        let encoder = Box::new(MockEncoder::new());

        let mut executor = PipelineExecutor::new(decoder, encoder)
            .with_stage(Box::new(PassthroughStage));

        let stats = executor.execute().expect("pipeline should succeed");

        assert_eq!(stats.total_decoded, 4);
        assert_eq!(stats.total_encoded, 4);
        assert_eq!(stats.stages.len(), 1);
        assert_eq!(stats.stages[0].name, "passthrough");
        assert_eq!(stats.stages[0].frames_in, 4);
        assert_eq!(stats.stages[0].frames_out, 4);
    }

    #[test]
    fn test_pipeline_drop_stage() {
        let frames = make_test_frames(6);
        let decoder = Box::new(MockDecoder::new(frames));
        let encoder = Box::new(MockEncoder::new());

        let mut executor = PipelineExecutor::new(decoder, encoder)
            .with_stage(Box::new(DropEveryOtherStage::new()));

        let stats = executor.execute().expect("pipeline should succeed");

        assert_eq!(stats.total_decoded, 6);
        assert_eq!(stats.total_encoded, 3);
        assert_eq!(stats.frames_dropped, 3);
    }

    #[test]
    fn test_pipeline_multiple_stages() {
        let frames = make_test_frames(4);
        let decoder = Box::new(MockDecoder::new(frames));
        let encoder = Box::new(MockEncoder::new());

        let mut executor = PipelineExecutor::new(decoder, encoder)
            .with_stage(Box::new(DoublerStage))
            .with_stage(Box::new(PassthroughStage));

        let stats = executor.execute().expect("pipeline should succeed");

        assert_eq!(stats.total_decoded, 4);
        assert_eq!(stats.total_encoded, 4);
        assert_eq!(stats.stages.len(), 2);
        assert_eq!(stats.stages[0].name, "doubler");
        assert_eq!(stats.stages[1].name, "passthrough");
    }

    #[test]
    fn test_execution_stats_fps() {
        let stats = ExecutionStats {
            total_encoded: 100,
            wall_time: Duration::from_secs(2),
            ..ExecutionStats::default()
        };
        let fps = stats.fps();
        assert!((fps - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_execution_stats_speed_factor() {
        let stats = ExecutionStats {
            wall_time: Duration::from_secs(5),
            ..ExecutionStats::default()
        };
        let sf = stats.speed_factor(10.0);
        assert!((sf - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_timestamp_manager_passthrough() {
        let mut mgr = TimestampManager::passthrough();
        assert!(!mgr.needs_conversion());
        let pts = mgr.map_pts(100_000);
        assert_eq!(pts, 100_000);
    }

    #[test]
    fn test_timestamp_manager_conversion() {
        let mgr = TimestampManager::new((24, 1), (60, 1));
        assert!(mgr.needs_conversion());
        assert!((mgr.input_rate() - 24.0).abs() < 0.01);
        assert!((mgr.output_rate() - 60.0).abs() < 0.01);
    }

    #[test]
    fn test_frame_rate_converter_passthrough() {
        let mut conv = FrameRateConverter::new((30, 1), (30, 1));
        let frame = PipelineFrame::video(vec![1, 2, 3, 4], 0, 2, 2);
        let result = conv.process(frame).expect("should succeed");
        match result {
            StageOutput::Pass(_) => {}
            other => panic!("expected Pass, got {:?}", other),
        }
    }

    #[test]
    fn test_frame_rate_converter_audio_passthrough() {
        let mut conv = FrameRateConverter::new((24, 1), (60, 1));
        let frame = PipelineFrame::audio(vec![0; 1024], 0);
        let result = conv.process(frame).expect("should succeed");
        match result {
            StageOutput::Pass(f) => assert!(f.is_audio),
            other => panic!("expected Pass for audio, got {:?}", other),
        }
    }

    #[test]
    fn test_pipeline_frame_constructors() {
        let vf = PipelineFrame::video(vec![1, 2], 1000, 320, 240)
            .with_duration(33_333)
            .with_sequence(5)
            .with_keyframe(true);

        assert!(!vf.is_audio);
        assert_eq!(vf.width, 320);
        assert_eq!(vf.height, 240);
        assert_eq!(vf.duration_us, 33_333);
        assert_eq!(vf.sequence, 5);
        assert!(vf.is_keyframe);
        assert!((vf.pts_secs() - 0.001).abs() < 0.0001);

        let af = PipelineFrame::audio(vec![0; 100], 500_000);
        assert!(af.is_audio);
        assert_eq!(af.width, 0);
    }

    #[test]
    fn test_max_frames_limit() {
        let frames = make_test_frames(20);
        let decoder = Box::new(MockDecoder::new(frames));
        let encoder = Box::new(MockEncoder::new());

        let config = ExecutorConfig {
            max_frames: 5,
            collect_stage_stats: true,
        };

        let mut executor = PipelineExecutor::new(decoder, encoder)
            .with_config(config);

        let stats = executor.execute().expect("should succeed");
        assert_eq!(stats.total_decoded, 5);
        assert_eq!(stats.total_encoded, 5);
    }

    #[test]
    fn test_timestamp_manager_boundary_up_conversion() {
        let mgr = TimestampManager::new((24, 1), (48, 1));
        // 2:1 ratio — each input frame should produce 2 output frames.
        assert_eq!(mgr.frames_at_boundary(0), 2);
        assert_eq!(mgr.frames_at_boundary(1), 2);
    }

    #[test]
    fn test_timestamp_manager_boundary_down_conversion() {
        let mgr = TimestampManager::new((60, 1), (30, 1));
        // 0.5:1 ratio — only every other input frame produces output.
        let total: u32 = (0..10).map(|i| mgr.frames_at_boundary(i)).sum();
        assert_eq!(total, 5);
    }

    #[test]
    fn test_empty_pipeline() {
        let decoder = Box::new(MockDecoder::new(Vec::new()));
        let encoder = Box::new(MockEncoder::new());

        let mut executor = PipelineExecutor::new(decoder, encoder);
        let stats = executor.execute().expect("should succeed with 0 frames");
        assert_eq!(stats.total_decoded, 0);
        assert_eq!(stats.total_encoded, 0);
    }
}
