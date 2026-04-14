// #![allow(dead_code)]
//! NDI frame-rate conversion for `oximedia-ndi`.
//!
//! Converts a stream of video frames from one frame rate to another using one
//! of three configurable strategies:
//!
//! - **Drop** — discard surplus frames when going to a lower rate.
//! - **Repeat** — duplicate the most recent frame when going to a higher rate.
//! - **Blend** — linearly blend adjacent frames for smoother motion.
//!
//! The converter works with raw byte buffers and is independent of any specific
//! NDI frame type, making it easy to integrate at any point in the pipeline.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::module_name_repetitions)]

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// FrameRatio
// ---------------------------------------------------------------------------

/// A rational frame rate expressed as numerator / denominator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FrameRatio {
    /// Numerator (frames per second × denominator).
    pub num: u32,
    /// Denominator (usually 1 or 1001 for drop-frame rates).
    pub den: u32,
}

impl FrameRatio {
    /// Create a new frame ratio, normalising to avoid zero denominators.
    pub fn new(num: u32, den: u32) -> Self {
        let den = den.max(1);
        let g = gcd(num, den);
        Self {
            num: num / g,
            den: den / g,
        }
    }

    /// Standard 24 fps.
    pub fn fps_24() -> Self {
        Self::new(24, 1)
    }

    /// NTSC 23.976 fps.
    pub fn fps_23_976() -> Self {
        Self::new(24000, 1001)
    }

    /// Standard 25 fps (PAL).
    pub fn fps_25() -> Self {
        Self::new(25, 1)
    }

    /// Standard 29.97 fps (NTSC drop-frame).
    pub fn fps_29_97() -> Self {
        Self::new(30000, 1001)
    }

    /// Standard 30 fps.
    pub fn fps_30() -> Self {
        Self::new(30, 1)
    }

    /// Standard 50 fps.
    pub fn fps_50() -> Self {
        Self::new(50, 1)
    }

    /// Standard 59.94 fps (NTSC HD).
    pub fn fps_59_94() -> Self {
        Self::new(60000, 1001)
    }

    /// Standard 60 fps.
    pub fn fps_60() -> Self {
        Self::new(60, 1)
    }

    /// Return the frame duration in microseconds.
    pub fn frame_duration_us(&self) -> u64 {
        if self.num == 0 {
            return 0;
        }
        (1_000_000u64 * self.den as u64) / self.num as u64
    }

    /// Return the frame rate as a 64-bit float.
    pub fn as_f64(&self) -> f64 {
        self.num as f64 / self.den as f64
    }
}

impl Default for FrameRatio {
    fn default() -> Self {
        Self::fps_30()
    }
}

// ---------------------------------------------------------------------------
// ConversionStrategy
// ---------------------------------------------------------------------------

/// Strategy used when the output frame rate differs from the input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConversionStrategy {
    /// Drop surplus frames (useful for down-conversion).
    Drop,
    /// Repeat the last frame to fill gaps (useful for up-conversion).
    Repeat,
    /// Linearly blend adjacent frames for smoother motion.
    Blend,
}

impl Default for ConversionStrategy {
    fn default() -> Self {
        Self::Blend
    }
}

// ---------------------------------------------------------------------------
// ConverterConfig
// ---------------------------------------------------------------------------

/// Configuration for the frame-rate converter.
#[derive(Debug, Clone)]
pub struct ConverterConfig {
    /// Frame rate of the input stream.
    pub input_fps: FrameRatio,
    /// Target frame rate for the output stream.
    pub output_fps: FrameRatio,
    /// Conversion strategy.
    pub strategy: ConversionStrategy,
    /// Maximum number of input frames to buffer internally.
    pub input_buffer_size: usize,
}

impl Default for ConverterConfig {
    fn default() -> Self {
        Self {
            input_fps: FrameRatio::fps_30(),
            output_fps: FrameRatio::fps_60(),
            strategy: ConversionStrategy::default(),
            input_buffer_size: 8,
        }
    }
}

// ---------------------------------------------------------------------------
// FrameRateConverter
// ---------------------------------------------------------------------------

/// Stateful frame-rate converter.
///
/// Call [`push`] to feed input frames and [`pull`] / [`pull_all`] to retrieve
/// output frames at the target rate.
#[derive(Debug)]
pub struct FrameRateConverter {
    config: ConverterConfig,
    /// Buffered input frames (raw pixel data + original timestamp in µs).
    input_buffer: VecDeque<(Vec<u8>, u64)>,
    /// Accumulator for the rational-counter algorithm.
    acc: i64,
    /// Input clock (incremented per input frame).
    input_clock: u64,
    /// Output clock (incremented per output frame emitted).
    output_clock: u64,
    /// Total input frames ever pushed.
    total_input: u64,
    /// Total output frames ever emitted.
    total_output: u64,
    /// Total frames dropped.
    total_dropped: u64,
    /// Total frames repeated.
    total_repeated: u64,
    /// Total frames blended.
    total_blended: u64,
    /// Most recently emitted frame data (for repeat mode).
    last_frame: Option<Vec<u8>>,
}

impl FrameRateConverter {
    /// Create a new converter with the given configuration.
    pub fn new(config: ConverterConfig) -> Self {
        Self {
            config,
            input_buffer: VecDeque::new(),
            acc: 0,
            input_clock: 0,
            output_clock: 0,
            total_input: 0,
            total_output: 0,
            total_dropped: 0,
            total_repeated: 0,
            total_blended: 0,
            last_frame: None,
        }
    }

    /// Return the converter configuration.
    pub fn config(&self) -> &ConverterConfig {
        &self.config
    }

    /// Push an input frame (raw pixel bytes, timestamp in µs) into the buffer.
    /// Returns `true` if the frame was accepted or `false` if the buffer is full.
    pub fn push(&mut self, data: Vec<u8>, timestamp_us: u64) -> bool {
        if self.input_buffer.len() >= self.config.input_buffer_size {
            self.total_dropped += 1;
            return false;
        }
        self.input_buffer.push_back((data, timestamp_us));
        self.total_input += 1;
        true
    }

    /// Pull the next output frame. Returns `None` when no output is ready.
    ///
    /// Internally this uses a rational counter to correctly map input frame
    /// boundaries to output frame boundaries.
    pub fn pull(&mut self) -> Option<Vec<u8>> {
        // Rational counter: the output should produce
        //   out_num / (out_den)  frames per second
        // for every
        //   in_num / (in_den)  input frames per second.
        //
        // We emit one output frame for each output period that passes.
        // acc represents: how many output ticks have accumulated?
        //   acc += out_num * in_den
        //   if acc >= in_num * out_den  →  emit, acc -= in_num * out_den

        let in_n = self.config.input_fps.num as i64;
        let in_d = self.config.input_fps.den as i64;
        let out_n = self.config.output_fps.num as i64;
        let out_d = self.config.output_fps.den as i64;

        // Step the output accumulator by one input frame worth.
        self.acc += out_n * in_d;
        let threshold = in_n * out_d;

        if self.acc < threshold {
            // Not yet time for an output frame; drop or hold.
            if let Some((_, _ts)) = self.input_buffer.pop_front() {
                self.total_dropped += 1;
                self.input_clock += 1;
            }
            return None;
        }

        // Emit one (or possibly more) output frames.
        self.acc -= threshold;

        match self.config.strategy {
            ConversionStrategy::Drop => self.emit_drop(),
            ConversionStrategy::Repeat => self.emit_repeat(),
            ConversionStrategy::Blend => self.emit_blend(),
        }
    }

    /// Pull all available output frames.
    pub fn pull_all(&mut self) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        while let Some(frame) = self.pull() {
            out.push(frame);
            // Safety valve: never return more frames than there are input frames
            // in the buffer plus one (for repeat).
            if out.len() > self.total_input as usize + 1 {
                break;
            }
        }
        out
    }

    fn emit_drop(&mut self) -> Option<Vec<u8>> {
        let (data, _) = self.input_buffer.pop_front()?;
        self.input_clock += 1;
        self.output_clock += 1;
        self.total_output += 1;
        self.last_frame = Some(data.clone());
        Some(data)
    }

    fn emit_repeat(&mut self) -> Option<Vec<u8>> {
        // If we have input, use it; otherwise repeat the last frame.
        if let Some((data, _)) = self.input_buffer.pop_front() {
            self.input_clock += 1;
            self.output_clock += 1;
            self.total_output += 1;
            self.last_frame = Some(data.clone());
            Some(data)
        } else if let Some(last) = self.last_frame.clone() {
            self.output_clock += 1;
            self.total_output += 1;
            self.total_repeated += 1;
            Some(last)
        } else {
            None
        }
    }

    fn emit_blend(&mut self) -> Option<Vec<u8>> {
        // Need two frames to blend; fall back to repeat if only one available.
        if self.input_buffer.len() >= 2 {
            let (a, _) = self.input_buffer.pop_front()?;
            let (b, _) = self.input_buffer.front()?;
            let blended = blend_frames(&a, b, 0.5);
            self.input_clock += 1;
            self.output_clock += 1;
            self.total_output += 1;
            self.total_blended += 1;
            self.last_frame = Some(blended.clone());
            Some(blended)
        } else {
            // Fall back to repeat
            self.emit_repeat()
        }
    }

    /// Return how many input frames are buffered.
    pub fn input_buffer_len(&self) -> usize {
        self.input_buffer.len()
    }

    /// Return the total number of input frames pushed.
    pub fn total_input(&self) -> u64 {
        self.total_input
    }

    /// Return the total number of output frames emitted.
    pub fn total_output(&self) -> u64 {
        self.total_output
    }

    /// Return the total number of frames dropped.
    pub fn total_dropped(&self) -> u64 {
        self.total_dropped
    }

    /// Return the total number of frames that were repeated.
    pub fn total_repeated(&self) -> u64 {
        self.total_repeated
    }

    /// Return the total number of frames that were blended.
    pub fn total_blended(&self) -> u64 {
        self.total_blended
    }

    /// Whether the output rate is higher than the input rate.
    pub fn is_upconversion(&self) -> bool {
        self.config.output_fps.as_f64() > self.config.input_fps.as_f64()
    }

    /// Whether the output rate is lower than the input rate.
    pub fn is_downconversion(&self) -> bool {
        self.config.output_fps.as_f64() < self.config.input_fps.as_f64()
    }

    /// Return the conversion ratio (output fps / input fps).
    pub fn conversion_ratio(&self) -> f64 {
        let in_fps = self.config.input_fps.as_f64();
        if in_fps == 0.0 {
            return 0.0;
        }
        self.config.output_fps.as_f64() / in_fps
    }

    /// Flush any remaining buffered frames without conversion logic.
    pub fn flush(&mut self) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        while let Some((data, _)) = self.input_buffer.pop_front() {
            out.push(data);
        }
        out
    }

    /// Reset internal state (accumulators and buffers) but keep config.
    pub fn reset(&mut self) {
        self.input_buffer.clear();
        self.acc = 0;
        self.input_clock = 0;
        self.output_clock = 0;
        self.total_input = 0;
        self.total_output = 0;
        self.total_dropped = 0;
        self.total_repeated = 0;
        self.total_blended = 0;
        self.last_frame = None;
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Linear blend of two same-size byte buffers at weight `alpha` (0.0 = all a,
/// 1.0 = all b).  Buffers of different lengths are handled by truncating to
/// the shorter.
fn blend_frames(a: &[u8], b: &[u8], alpha: f64) -> Vec<u8> {
    let len = a.len().min(b.len());
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let va = a[i] as f64;
        let vb = b[i] as f64;
        let blended = va * (1.0 - alpha) + vb * alpha;
        out.push(blended.round().clamp(0.0, 255.0) as u8);
    }
    out
}

/// Euclidean GCD.
fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a.max(1)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(value: u8, size: usize) -> Vec<u8> {
        vec![value; size]
    }

    #[test]
    fn test_frame_ratio_new_normalises() {
        let r = FrameRatio::new(60, 2);
        assert_eq!(r.num, 30);
        assert_eq!(r.den, 1);
    }

    #[test]
    fn test_frame_ratio_fps_30_duration() {
        let r = FrameRatio::fps_30();
        // 1_000_000 / 30 = 33333 µs
        assert_eq!(r.frame_duration_us(), 33333);
    }

    #[test]
    fn test_frame_ratio_fps_29_97() {
        let r = FrameRatio::fps_29_97();
        let fps = r.as_f64();
        assert!((fps - 29.97).abs() < 0.01);
    }

    #[test]
    fn test_conversion_ratio_2x() {
        let config = ConverterConfig {
            input_fps: FrameRatio::fps_30(),
            output_fps: FrameRatio::fps_60(),
            strategy: ConversionStrategy::Repeat,
            input_buffer_size: 16,
        };
        let converter = FrameRateConverter::new(config);
        assert!((converter.conversion_ratio() - 2.0).abs() < 0.001);
        assert!(converter.is_upconversion());
        assert!(!converter.is_downconversion());
    }

    #[test]
    fn test_push_and_pull_drop_strategy() {
        // 60fps → 30fps: half the frames should pass through
        let config = ConverterConfig {
            input_fps: FrameRatio::fps_60(),
            output_fps: FrameRatio::fps_30(),
            strategy: ConversionStrategy::Drop,
            input_buffer_size: 64,
        };
        let mut c = FrameRateConverter::new(config);
        for i in 0..60u8 {
            c.push(make_frame(i, 4), i as u64 * 16666);
        }
        let mut outputs = Vec::new();
        for _ in 0..60 {
            if let Some(f) = c.pull() {
                outputs.push(f);
            }
        }
        // At 2:1 ratio, roughly 30 frames should come out.
        assert!(
            outputs.len() >= 28 && outputs.len() <= 32,
            "expected ~30 output frames, got {}",
            outputs.len()
        );
    }

    #[test]
    fn test_repeat_strategy_fills_gap() {
        // 30fps → 60fps: each input frame should produce ~2 outputs
        let config = ConverterConfig {
            input_fps: FrameRatio::fps_30(),
            output_fps: FrameRatio::fps_60(),
            strategy: ConversionStrategy::Repeat,
            input_buffer_size: 64,
        };
        let mut c = FrameRateConverter::new(config);
        for i in 0..30u8 {
            c.push(make_frame(i, 4), i as u64 * 33333);
        }
        let mut count = 0usize;
        for _ in 0..120 {
            if c.pull().is_some() {
                count += 1;
            }
        }
        // Should be close to 60 (2× input)
        assert!(count >= 28, "expected at least 28, got {count}");
    }

    #[test]
    fn test_blend_strategy() {
        let config = ConverterConfig {
            input_fps: FrameRatio::fps_30(),
            output_fps: FrameRatio::fps_30(),
            strategy: ConversionStrategy::Blend,
            input_buffer_size: 16,
        };
        let mut c = FrameRateConverter::new(config);
        c.push(make_frame(0, 4), 0);
        c.push(make_frame(100, 4), 33333);
        let out = c.pull();
        assert!(out.is_some(), "expected a blended output frame");
    }

    #[test]
    fn test_blend_helper() {
        let a = vec![0u8, 0, 0, 0];
        let b = vec![200u8, 200, 200, 200];
        let blended = blend_frames(&a, &b, 0.5);
        for &v in &blended {
            assert!((v as i32 - 100).abs() <= 1, "expected ~100, got {v}");
        }
    }

    #[test]
    fn test_flush_returns_remaining() {
        let config = ConverterConfig {
            input_fps: FrameRatio::fps_30(),
            output_fps: FrameRatio::fps_30(),
            strategy: ConversionStrategy::Drop,
            input_buffer_size: 16,
        };
        let mut c = FrameRateConverter::new(config);
        c.push(make_frame(1, 4), 0);
        c.push(make_frame(2, 4), 33333);
        let flushed = c.flush();
        assert_eq!(flushed.len(), 2);
        assert_eq!(c.input_buffer_len(), 0);
    }

    #[test]
    fn test_reset_clears_state() {
        let config = ConverterConfig::default();
        let mut c = FrameRateConverter::new(config);
        c.push(make_frame(1, 4), 0);
        c.reset();
        assert_eq!(c.input_buffer_len(), 0);
        assert_eq!(c.total_input(), 0);
        assert_eq!(c.total_output(), 0);
    }

    #[test]
    fn test_buffer_full_drops() {
        let config = ConverterConfig {
            input_fps: FrameRatio::fps_30(),
            output_fps: FrameRatio::fps_30(),
            strategy: ConversionStrategy::Drop,
            input_buffer_size: 2,
        };
        let mut c = FrameRateConverter::new(config);
        assert!(c.push(make_frame(1, 4), 0));
        assert!(c.push(make_frame(2, 4), 33333));
        // Third push should fail (buffer full)
        assert!(!c.push(make_frame(3, 4), 66666));
        assert!(c.total_dropped() > 0);
    }

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(12, 8), 4);
        assert_eq!(gcd(7, 1), 1);
        // gcd(0, 5): our implementation loops until b==0 giving a=5, then max(5,1)=5
        assert_eq!(gcd(0, 5), 5);
        assert_eq!(gcd(0, 0), 1); // edge case: both zero → returns 1 via .max(1)
    }

    #[test]
    fn test_frame_ratio_identity_conversion() {
        let config = ConverterConfig {
            input_fps: FrameRatio::fps_30(),
            output_fps: FrameRatio::fps_30(),
            strategy: ConversionStrategy::Drop,
            input_buffer_size: 32,
        };
        let mut c = FrameRateConverter::new(config);
        for i in 0..10u8 {
            c.push(make_frame(i, 4), i as u64 * 33333);
        }
        let out = c.pull_all();
        // 1:1 ratio — should get roughly the same count out
        assert!(!out.is_empty());
    }

    #[test]
    fn test_downconversion_flag() {
        let config = ConverterConfig {
            input_fps: FrameRatio::fps_60(),
            output_fps: FrameRatio::fps_30(),
            strategy: ConversionStrategy::Drop,
            ..Default::default()
        };
        let c = FrameRateConverter::new(config);
        assert!(c.is_downconversion());
        assert!(!c.is_upconversion());
    }
}
