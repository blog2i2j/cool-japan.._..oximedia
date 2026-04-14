//! Offline bounce / render engine.
//!
//! Unlike real-time processing, an offline bounce can run faster-than-realtime
//! because it processes an arbitrary number of samples per iteration with no
//! deadline constraints.  It accepts a user-supplied [`AudioSource`] trait
//! object for input and accumulates the rendered output into an interleaved
//! `f32` buffer.
//!
//! # Features
//!
//! * **Variable speed** — configurable block size; larger blocks amortise
//!   per-call overhead and increase throughput.
//! * **Progress callbacks** — optional callback receives `(samples_done,
//!   total_samples)` on each block so callers can drive a progress bar.
//! * **Per-channel gain, pan, and mute** — simple DSP applied inline without
//!   a full mixer instantiation.
//! * **Peak / RMS metering** — accumulated during the render for a post-bounce
//!   loudness report.
//! * **Silence detection** — optional trimming of trailing silence at the end
//!   of the bounce region.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by the offline bounce engine.
#[derive(Debug, thiserror::Error)]
pub enum BounceError {
    /// Bounce region is zero-length.
    #[error("Bounce region is zero-length")]
    ZeroLengthRegion,

    /// Audio source reported an I/O error.
    #[error("Audio source error: {0}")]
    SourceError(String),

    /// Requested channel count is zero.
    #[error("Output channel count must be at least 1")]
    InvalidChannelCount,

    /// Block size is zero.
    #[error("Block size must be at least 1")]
    InvalidBlockSize,

    /// Per-channel gain is out of range.
    #[error("Gain {0} is out of range (0.0 ..= 4.0)")]
    GainOutOfRange(f32),
}

/// Result alias.
pub type BounceResult<T> = Result<T, BounceError>;

// ---------------------------------------------------------------------------
// AudioSource trait
// ---------------------------------------------------------------------------

/// An audio source that can supply interleaved `f32` samples on demand.
///
/// Implementations must be deterministic for a given `start_sample` offset.
pub trait AudioSource: Send {
    /// Fill `output` with `output.len() / channels` frames of interleaved
    /// `f32` audio starting at `sample_offset`.
    ///
    /// If the source has fewer remaining samples it writes what it can and
    /// returns the number of **frames** actually written.  Returning `0`
    /// signals end-of-source.
    fn fill(
        &mut self,
        sample_offset: u64,
        channels: u32,
        output: &mut [f32],
    ) -> Result<usize, String>;

    /// Total length of the source in frames (if known).
    fn total_frames(&self) -> Option<u64>;
}

// ---------------------------------------------------------------------------
// Channel strip configuration
// ---------------------------------------------------------------------------

/// Simple per-channel parameter set applied during the bounce.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BounceChannelParams {
    /// Linear gain (0.0 = silence, 1.0 = unity).
    pub gain: f32,
    /// Stereo pan (-1.0 = hard-left, 0.0 = centre, +1.0 = hard-right).
    pub pan: f32,
    /// If `true` this channel contributes silence.
    pub muted: bool,
}

impl Default for BounceChannelParams {
    fn default() -> Self {
        Self {
            gain: 1.0,
            pan: 0.0,
            muted: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Bounce configuration
// ---------------------------------------------------------------------------

/// Configuration for an offline bounce operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BounceConfig {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of output channels (1 = mono, 2 = stereo).
    pub output_channels: u32,
    /// Number of output frames to render (i.e. bounce length in samples).
    pub total_frames: u64,
    /// Number of frames to process per engine block.
    pub block_size: u32,
    /// Optional silence threshold: trailing frames whose peak is below this
    /// linear level will be trimmed from the output.
    pub silence_trim_threshold: Option<f32>,
    /// Per-input-channel parameters (keyed by 0-based input channel index).
    pub channel_params: HashMap<u32, BounceChannelParams>,
}

impl BounceConfig {
    /// Create a stereo bounce config with sensible defaults.
    #[must_use]
    pub fn stereo(sample_rate: u32, total_frames: u64) -> Self {
        Self {
            sample_rate,
            output_channels: 2,
            total_frames,
            block_size: 1024,
            silence_trim_threshold: None,
            channel_params: HashMap::new(),
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> BounceResult<()> {
        if self.total_frames == 0 {
            return Err(BounceError::ZeroLengthRegion);
        }
        if self.output_channels == 0 {
            return Err(BounceError::InvalidChannelCount);
        }
        if self.block_size == 0 {
            return Err(BounceError::InvalidBlockSize);
        }
        for (_, params) in &self.channel_params {
            if !(0.0..=4.0).contains(&params.gain) {
                return Err(BounceError::GainOutOfRange(params.gain));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Bounce metrics
// ---------------------------------------------------------------------------

/// Post-bounce measurement data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BounceMetrics {
    /// True peak (linear) across all output channels.
    pub true_peak: f32,
    /// RMS level (linear) across all output channels.
    pub rms: f32,
    /// Number of frames actually written (may be less than `total_frames` when
    /// silence trimming is applied).
    pub frames_rendered: u64,
    /// Number of frames trimmed from the tail.
    pub frames_trimmed: u64,
    /// Time taken to complete the render in milliseconds (wall-clock).
    pub render_time_ms: u64,
}

impl BounceMetrics {
    /// Convert true peak to dBFS.
    #[must_use]
    pub fn true_peak_dbfs(&self) -> f32 {
        if self.true_peak <= 0.0 {
            return f32::NEG_INFINITY;
        }
        20.0 * self.true_peak.log10()
    }

    /// Convert RMS to dBFS.
    #[must_use]
    pub fn rms_dbfs(&self) -> f32 {
        if self.rms <= 0.0 {
            return f32::NEG_INFINITY;
        }
        20.0 * self.rms.log10()
    }
}

// ---------------------------------------------------------------------------
// BounceEngine
// ---------------------------------------------------------------------------

/// Progress callback signature: `(frames_done, total_frames)`.
pub type ProgressCallback = Box<dyn FnMut(u64, u64) + Send>;

/// Offline bounce / render engine.
pub struct BounceEngine {
    config: BounceConfig,
    progress_callback: Option<ProgressCallback>,
}

impl BounceEngine {
    /// Create a new engine from the given configuration.
    ///
    /// Returns `Err` if the configuration is invalid.
    pub fn new(config: BounceConfig) -> BounceResult<Self> {
        config.validate()?;
        Ok(Self {
            config,
            progress_callback: None,
        })
    }

    /// Set a progress callback that is called after each block.
    pub fn set_progress_callback(&mut self, cb: ProgressCallback) {
        self.progress_callback = Some(cb);
    }

    /// Run the bounce, reading from `source` and returning the rendered audio
    /// together with a [`BounceMetrics`] report.
    ///
    /// The returned `Vec<f32>` is interleaved with `config.output_channels`
    /// channels.
    pub fn bounce(
        &mut self,
        source: &mut dyn AudioSource,
    ) -> BounceResult<(Vec<f32>, BounceMetrics)> {
        let total = self.config.total_frames;
        let out_ch = self.config.output_channels as usize;
        let block_size = self.config.block_size as usize;

        // Allocate output buffer.
        let capacity = total as usize * out_ch;
        let mut output = vec![0.0_f32; capacity];

        // Per-block source buffer.
        // Assume mono source (1 input channel) and upmix/pan to stereo.
        let in_ch: u32 = 1; // simplification: single mono input summed to output
        let mut src_buf = vec![0.0_f32; block_size * in_ch as usize];

        let mut frames_done: u64 = 0;
        let mut peak: f32 = 0.0;
        let mut sum_sq: f64 = 0.0;

        let start_instant = std::time::Instant::now();

        while frames_done < total {
            let block_frames = ((total - frames_done) as usize).min(block_size);
            let src_slice = &mut src_buf[..block_frames * in_ch as usize];

            let filled = source
                .fill(frames_done, in_ch, src_slice)
                .map_err(BounceError::SourceError)?;

            if filled == 0 {
                break; // Source exhausted before total_frames.
            }

            // Apply per-channel DSP and write to output.
            self.process_block(
                src_slice,
                filled,
                in_ch,
                frames_done,
                &mut output,
                out_ch,
                &mut peak,
                &mut sum_sq,
            );

            frames_done += filled as u64;

            if let Some(cb) = self.progress_callback.as_mut() {
                cb(frames_done, total);
            }
        }

        let render_time_ms = start_instant.elapsed().as_millis() as u64;

        // Silence trimming.
        let (frames_rendered, frames_trimmed) =
            self.trim_silence(&mut output, out_ch, frames_done);

        let rms = if frames_rendered > 0 {
            (sum_sq / (frames_rendered as f64 * out_ch as f64)).sqrt() as f32
        } else {
            0.0
        };

        let metrics = BounceMetrics {
            true_peak: peak,
            rms,
            frames_rendered,
            frames_trimmed,
            render_time_ms,
        };

        // Truncate buffer to actual rendered length.
        output.truncate(frames_rendered as usize * out_ch);

        Ok((output, metrics))
    }

    // ------------------------------------------------------------------
    // DSP helpers
    // ------------------------------------------------------------------

    fn process_block(
        &self,
        src: &[f32],
        frames: usize,
        in_ch: u32,
        frame_offset: u64,
        output: &mut [f32],
        out_ch: usize,
        peak: &mut f32,
        sum_sq: &mut f64,
    ) {
        let default_params = BounceChannelParams::default();
        // We iterate over each output frame.
        for frame_idx in 0..frames {
            // Accumulate all input channels into L/R.
            let mut acc_l = 0.0_f32;
            let mut acc_r = 0.0_f32;
            for in_c in 0..in_ch as usize {
                let params = self
                    .config
                    .channel_params
                    .get(&(in_c as u32))
                    .unwrap_or(&default_params);
                if params.muted {
                    continue;
                }
                let sample = src[frame_idx * in_ch as usize + in_c] * params.gain;
                // -3 dB pan law
                let pan = params.pan.clamp(-1.0, 1.0);
                let angle = (pan + 1.0) * std::f32::consts::FRAC_PI_4; // 0..π/2
                let l_gain = angle.cos();
                let r_gain = angle.sin();
                acc_l += sample * l_gain;
                acc_r += sample * r_gain;
            }

            let out_base = (frame_offset as usize + frame_idx) * out_ch;
            if out_ch >= 2 {
                output[out_base] = acc_l;
                output[out_base + 1] = acc_r;
                *peak = peak.max(acc_l.abs()).max(acc_r.abs());
                *sum_sq += (acc_l * acc_l + acc_r * acc_r) as f64;
            } else {
                let mono = (acc_l + acc_r) * 0.5;
                output[out_base] = mono;
                *peak = peak.max(mono.abs());
                *sum_sq += (mono * mono) as f64;
            }
        }
    }

    fn trim_silence(
        &self,
        output: &mut Vec<f32>,
        out_ch: usize,
        frames_rendered: u64,
    ) -> (u64, u64) {
        let threshold = match self.config.silence_trim_threshold {
            Some(t) => t,
            None => return (frames_rendered, 0),
        };

        let total_samples = frames_rendered as usize * out_ch;
        let mut last_non_silent_sample = 0usize;
        for i in (0..total_samples).rev() {
            if output[i].abs() > threshold {
                last_non_silent_sample = i + 1;
                break;
            }
        }

        // Round up to frame boundary.
        let trimmed_samples =
            ((total_samples - last_non_silent_sample + out_ch - 1) / out_ch) * out_ch;
        let kept_samples = total_samples - trimmed_samples;
        let kept_frames = (kept_samples / out_ch) as u64;
        let trimmed_frames = frames_rendered - kept_frames;

        (kept_frames, trimmed_frames)
    }
}

// ---------------------------------------------------------------------------
// Convenience builder
// ---------------------------------------------------------------------------

/// Builder for [`BounceEngine`].
pub struct BounceEngineBuilder {
    config: BounceConfig,
    progress: Option<ProgressCallback>,
}

impl BounceEngineBuilder {
    /// Start building from a [`BounceConfig`].
    #[must_use]
    pub fn new(config: BounceConfig) -> Self {
        Self {
            config,
            progress: None,
        }
    }

    /// Attach a progress callback.
    #[must_use]
    pub fn with_progress(mut self, cb: ProgressCallback) -> Self {
        self.progress = Some(cb);
        self
    }

    /// Set silence trim threshold.
    #[must_use]
    pub fn with_silence_trim(mut self, threshold: f32) -> Self {
        self.config.silence_trim_threshold = Some(threshold);
        self
    }

    /// Set the processing block size.
    #[must_use]
    pub fn with_block_size(mut self, block_size: u32) -> Self {
        self.config.block_size = block_size;
        self
    }

    /// Set per-channel gain.
    #[must_use]
    pub fn with_channel_gain(mut self, channel: u32, gain: f32) -> Self {
        self.config
            .channel_params
            .entry(channel)
            .or_default()
            .gain = gain;
        self
    }

    /// Set per-channel pan.
    #[must_use]
    pub fn with_channel_pan(mut self, channel: u32, pan: f32) -> Self {
        self.config
            .channel_params
            .entry(channel)
            .or_default()
            .pan = pan;
        self
    }

    /// Build the engine, validating configuration.
    pub fn build(self) -> BounceResult<BounceEngine> {
        let mut engine = BounceEngine::new(self.config)?;
        if let Some(cb) = self.progress {
            engine.set_progress_callback(cb);
        }
        Ok(engine)
    }
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// A simple in-memory audio source that yields a sine wave.
pub struct SineSource {
    frequency: f32,
    sample_rate: f32,
    amplitude: f32,
    total_frames: u64,
}

impl SineSource {
    /// Create a sine source.
    #[must_use]
    pub fn new(frequency: f32, sample_rate: f32, total_frames: u64) -> Self {
        Self {
            frequency,
            sample_rate,
            amplitude: 0.5,
            total_frames,
        }
    }

    /// Set amplitude.
    #[must_use]
    pub fn with_amplitude(mut self, amplitude: f32) -> Self {
        self.amplitude = amplitude;
        self
    }
}

impl AudioSource for SineSource {
    fn fill(&mut self, sample_offset: u64, channels: u32, output: &mut [f32]) -> Result<usize, String> {
        let frames = output.len() / channels as usize;
        let remaining = (self.total_frames.saturating_sub(sample_offset)) as usize;
        let write_frames = frames.min(remaining);
        for f in 0..write_frames {
            let t = (sample_offset + f as u64) as f32 / self.sample_rate;
            let sample = self.amplitude * (2.0 * std::f32::consts::PI * self.frequency * t).sin();
            for c in 0..channels as usize {
                output[f * channels as usize + c] = sample;
            }
        }
        Ok(write_frames)
    }

    fn total_frames(&self) -> Option<u64> {
        Some(self.total_frames)
    }
}

/// A silent audio source (all zeros).
pub struct SilentSource {
    total_frames: u64,
}

impl SilentSource {
    /// Create a silent source.
    #[must_use]
    pub fn new(total_frames: u64) -> Self {
        Self { total_frames }
    }
}

impl AudioSource for SilentSource {
    fn fill(&mut self, sample_offset: u64, channels: u32, output: &mut [f32]) -> Result<usize, String> {
        let frames = output.len() / channels as usize;
        let remaining = (self.total_frames.saturating_sub(sample_offset)) as usize;
        let write_frames = frames.min(remaining);
        for s in output[..write_frames * channels as usize].iter_mut() {
            *s = 0.0;
        }
        Ok(write_frames)
    }

    fn total_frames(&self) -> Option<u64> {
        Some(self.total_frames)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounce_produces_correct_length() {
        let cfg = BounceConfig::stereo(48000, 4800);
        let mut engine = BounceEngine::new(cfg).unwrap();
        let mut source = SineSource::new(440.0, 48000.0, 4800);
        let (output, metrics) = engine.bounce(&mut source).unwrap();
        assert_eq!(output.len(), 4800 * 2); // 2 channels
        assert_eq!(metrics.frames_rendered, 4800);
    }

    #[test]
    fn test_bounce_peak_within_unit() {
        let cfg = BounceConfig::stereo(48000, 9600);
        let mut engine = BounceEngine::new(cfg).unwrap();
        let mut source = SineSource::new(1000.0, 48000.0, 9600);
        let (_, metrics) = engine.bounce(&mut source).unwrap();
        assert!(
            metrics.true_peak <= 1.0 + 1e-5,
            "peak={} exceeded 1.0",
            metrics.true_peak
        );
    }

    #[test]
    fn test_zero_length_region_rejected() {
        let cfg = BounceConfig::stereo(48000, 0);
        assert!(BounceEngine::new(cfg).is_err());
    }

    #[test]
    fn test_invalid_block_size_rejected() {
        let mut cfg = BounceConfig::stereo(48000, 1000);
        cfg.block_size = 0;
        assert!(BounceEngine::new(cfg).is_err());
    }

    #[test]
    fn test_muted_channel_produces_silence() {
        let mut cfg = BounceConfig::stereo(48000, 4800);
        cfg.channel_params.insert(
            0,
            BounceChannelParams {
                gain: 1.0,
                pan: 0.0,
                muted: true,
            },
        );
        let mut engine = BounceEngine::new(cfg).unwrap();
        let mut source = SineSource::new(440.0, 48000.0, 4800);
        let (output, metrics) = engine.bounce(&mut source).unwrap();
        assert!(metrics.true_peak < 1e-6);
        assert!(output.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn test_gain_reduction_lowers_peak() {
        let mut cfg_half = BounceConfig::stereo(48000, 4800);
        cfg_half.channel_params.insert(
            0,
            BounceChannelParams {
                gain: 0.5,
                pan: 0.0,
                muted: false,
            },
        );
        let mut engine_full = BounceEngine::new(BounceConfig::stereo(48000, 4800)).unwrap();
        let mut engine_half = BounceEngine::new(cfg_half).unwrap();
        let mut src1 = SineSource::new(440.0, 48000.0, 4800);
        let mut src2 = SineSource::new(440.0, 48000.0, 4800);
        let (_, m_full) = engine_full.bounce(&mut src1).unwrap();
        let (_, m_half) = engine_half.bounce(&mut src2).unwrap();
        assert!(m_half.true_peak < m_full.true_peak);
    }

    #[test]
    fn test_silence_trim() {
        // 1 second of sine followed by silence.
        let total = 48000u64;
        let mut cfg = BounceConfig::stereo(48000, total);
        cfg.silence_trim_threshold = Some(1e-4);
        let mut engine = BounceEngine::new(cfg).unwrap();
        // Source: half sine, half silent.
        let mut source = SilentSource::new(total);
        let (_, metrics) = engine.bounce(&mut source).unwrap();
        // All silence → should trim to 0 rendered frames.
        assert_eq!(metrics.frames_rendered, 0);
    }

    #[test]
    fn test_progress_callback_called() {
        use std::sync::{Arc, Mutex};
        let calls = Arc::new(Mutex::new(0u32));
        let calls_clone = Arc::clone(&calls);
        let mut cfg = BounceConfig::stereo(48000, 2048);
        cfg.block_size = 256;
        let mut engine = BounceEngine::new(cfg).unwrap();
        engine.set_progress_callback(Box::new(move |_, _| {
            *calls_clone.lock().unwrap() += 1;
        }));
        let mut source = SineSource::new(440.0, 48000.0, 2048);
        engine.bounce(&mut source).unwrap();
        assert!(*calls.lock().unwrap() >= 1);
    }

    #[test]
    fn test_builder_api() {
        let cfg = BounceConfig::stereo(48000, 1024);
        let engine = BounceEngineBuilder::new(cfg)
            .with_block_size(512)
            .with_channel_gain(0, 0.8)
            .with_channel_pan(0, -0.2)
            .build();
        assert!(engine.is_ok());
    }

    #[test]
    fn test_metrics_dbfs() {
        let metrics = BounceMetrics {
            true_peak: 1.0,
            rms: 0.5,
            frames_rendered: 100,
            frames_trimmed: 0,
            render_time_ms: 1,
        };
        assert!((metrics.true_peak_dbfs() - 0.0).abs() < 1e-5);
        assert!(metrics.rms_dbfs() < 0.0);
    }
}
