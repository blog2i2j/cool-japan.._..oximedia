//! Lock-free parameter smoothing for the audio thread.
//!
//! Audio parameters such as fader gain, pan, and send levels must change
//! smoothly to avoid clicks and zipper noise.  This module provides:
//!
//! - [`LinearSmoother`] — linear ramp from current to target value over a
//!   configurable number of samples.
//! - [`ExponentialSmoother`] — one-pole IIR low-pass filter (exponential
//!   approach), commonly used for gain automation.
//! - [`SharedParam`] — an `Arc<AtomicU32>` wrapper that lets a control thread
//!   write new target values while the audio thread reads them, with no locks.
//! - [`SmoothedChannel`] — combines a `SharedParam` with a `LinearSmoother`
//!   so the audio thread automatically ramps to whatever value the UI thread
//!   sets.
//!
//! # Example
//!
//! ```rust
//! use oximedia_mixer::param_smoother::{LinearSmoother, ExponentialSmoother};
//!
//! // Smooth from 0.0 to 1.0 over 100 samples.
//! let mut smoother = LinearSmoother::new(0.0, 100);
//! smoother.set_target(1.0);
//! let block: Vec<f32> = smoother.next_block(100);
//! assert!((block[99] - 1.0).abs() < 1e-5);
//!
//! // One-pole exponential smoother, tau = 20 ms at 48 kHz.
//! let mut exp = ExponentialSmoother::new(48000, 0.02);
//! exp.set_target(0.8);
//! let _out = exp.next_sample();
//! ```

use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc,
};

// ── helpers ──────────────────────────────────────────────────────────────────

/// Bit-cast `f32` to its `u32` representation.
#[inline]
fn f32_to_bits(v: f32) -> u32 {
    v.to_bits()
}

/// Bit-cast `u32` back to `f32`.
#[inline]
fn bits_to_f32(b: u32) -> f32 {
    f32::from_bits(b)
}

// ── LinearSmoother ──────────────────────────────────────────────────────────

/// Smooths a parameter value with a linear ramp.
///
/// When a new target is set the smoother ramps from the current value to the
/// target over `ramp_samples` samples.  Once the target is reached the smoother
/// outputs a constant value.
#[derive(Debug, Clone)]
pub struct LinearSmoother {
    /// Current instantaneous value.
    current: f32,
    /// Target value.
    target: f32,
    /// Increment per sample (positive or negative).
    increment: f32,
    /// Remaining samples in the current ramp.
    remaining: usize,
    /// Number of samples for a full ramp.
    ramp_samples: usize,
}

impl LinearSmoother {
    /// Create a new `LinearSmoother` with an initial value and a ramp length.
    ///
    /// `ramp_samples` must be > 0; if 0 is passed it is clamped to 1 so the
    /// smoother always transitions in at least one step.
    pub fn new(initial: f32, ramp_samples: usize) -> Self {
        let rs = ramp_samples.max(1);
        Self {
            current: initial,
            target: initial,
            increment: 0.0,
            remaining: 0,
            ramp_samples: rs,
        }
    }

    /// Set a new target value.  The smoother ramps from `current` to `target`
    /// over `ramp_samples` samples.
    pub fn set_target(&mut self, target: f32) {
        if (target - self.target).abs() < f32::EPSILON && self.remaining == 0 {
            return;
        }
        self.target = target;
        self.remaining = self.ramp_samples;
        self.increment = (target - self.current) / self.ramp_samples as f32;
    }

    /// Advance by one sample and return the smoothed value.
    #[inline]
    pub fn next_sample(&mut self) -> f32 {
        if self.remaining > 0 {
            self.current += self.increment;
            self.remaining -= 1;
            if self.remaining == 0 {
                // Snap to target to eliminate accumulated floating-point drift.
                self.current = self.target;
            }
        }
        self.current
    }

    /// Fill `output` with the next `output.len()` smoothed values.
    pub fn fill_block(&mut self, output: &mut [f32]) {
        for sample in output.iter_mut() {
            *sample = self.next_sample();
        }
    }

    /// Return a `Vec<f32>` containing `count` smoothed values.
    pub fn next_block(&mut self, count: usize) -> Vec<f32> {
        let mut out = vec![0.0; count];
        self.fill_block(&mut out);
        out
    }

    /// Returns true if the smoother is still ramping (has not reached target).
    pub fn is_ramping(&self) -> bool {
        self.remaining > 0
    }

    /// Returns the current instantaneous value without advancing.
    pub fn current(&self) -> f32 {
        self.current
    }

    /// Returns the target value.
    pub fn target(&self) -> f32 {
        self.target
    }

    /// Set a new ramp length in samples (takes effect on the next
    /// [`set_target`](Self::set_target) call).
    pub fn set_ramp_samples(&mut self, ramp_samples: usize) {
        self.ramp_samples = ramp_samples.max(1);
    }
}

// ── ExponentialSmoother ──────────────────────────────────────────────────────

/// One-pole IIR low-pass (exponential-approach) parameter smoother.
///
/// The time constant `tau` (seconds) controls how quickly the smoother tracks
/// the target.  Specifically, the output reaches ~63% of the target step in
/// `tau` seconds.
#[derive(Debug, Clone)]
pub struct ExponentialSmoother {
    /// IIR coefficient α = 1 − exp(−1 / (τ · fs)).
    alpha: f32,
    /// Current value (IIR state).
    current: f32,
    /// Target value.
    target: f32,
}

impl ExponentialSmoother {
    /// Create a new `ExponentialSmoother`.
    ///
    /// - `sample_rate`: audio sample rate in Hz.
    /// - `tau_secs`: time constant in seconds (e.g. 0.02 for 20 ms).
    pub fn new(sample_rate: u32, tau_secs: f32) -> Self {
        let alpha = Self::compute_alpha(sample_rate, tau_secs);
        Self {
            alpha,
            current: 0.0,
            target: 0.0,
        }
    }

    /// Create a new smoother with an explicit initial value.
    pub fn with_initial(sample_rate: u32, tau_secs: f32, initial: f32) -> Self {
        let alpha = Self::compute_alpha(sample_rate, tau_secs);
        Self {
            alpha,
            current: initial,
            target: initial,
        }
    }

    fn compute_alpha(sample_rate: u32, tau_secs: f32) -> f32 {
        if tau_secs <= 0.0 || sample_rate == 0 {
            return 1.0; // Instantaneous — no smoothing.
        }
        1.0 - (-1.0 / (tau_secs * sample_rate as f32)).exp()
    }

    /// Update the time constant (takes effect from the next sample).
    pub fn set_tau(&mut self, sample_rate: u32, tau_secs: f32) {
        self.alpha = Self::compute_alpha(sample_rate, tau_secs);
    }

    /// Set a new target value.
    pub fn set_target(&mut self, target: f32) {
        self.target = target;
    }

    /// Advance by one sample and return the smoothed value.
    #[inline]
    pub fn next_sample(&mut self) -> f32 {
        self.current += self.alpha * (self.target - self.current);
        self.current
    }

    /// Fill `output` with the next `output.len()` smoothed values.
    pub fn fill_block(&mut self, output: &mut [f32]) {
        for sample in output.iter_mut() {
            *sample = self.next_sample();
        }
    }

    /// Return a `Vec<f32>` containing `count` smoothed values.
    pub fn next_block(&mut self, count: usize) -> Vec<f32> {
        let mut out = vec![0.0; count];
        self.fill_block(&mut out);
        out
    }

    /// Returns the current instantaneous value without advancing.
    pub fn current(&self) -> f32 {
        self.current
    }

    /// Returns the target value.
    pub fn target(&self) -> f32 {
        self.target
    }

    /// Returns the IIR coefficient α.
    pub fn alpha(&self) -> f32 {
        self.alpha
    }
}

// ── SharedParam ──────────────────────────────────────────────────────────────

/// A lock-free shared parameter: one writer (UI/control thread), one reader
/// (audio thread).
///
/// Internally stores the `f32` bits in an `AtomicU32` using `Relaxed` ordering.
/// This is safe for audio-thread use: we never need sequential consistency,
/// only eventual visibility.
#[derive(Debug, Clone)]
pub struct SharedParam {
    value: Arc<AtomicU32>,
}

impl SharedParam {
    /// Create a new `SharedParam` with an initial value.
    pub fn new(initial: f32) -> Self {
        Self {
            value: Arc::new(AtomicU32::new(f32_to_bits(initial))),
        }
    }

    /// Write a new value from the control thread.
    ///
    /// Uses `Release` ordering so the audio thread (reader) is guaranteed to
    /// see the write after it acquires.
    pub fn set(&self, v: f32) {
        self.value.store(f32_to_bits(v), Ordering::Release);
    }

    /// Read the current value from the audio thread.
    ///
    /// Uses `Acquire` ordering to pair with the `Release` in `set`.
    pub fn get(&self) -> f32 {
        bits_to_f32(self.value.load(Ordering::Acquire))
    }

    /// Create a second handle pointing to the same underlying atomic.
    pub fn handle(&self) -> SharedParamHandle {
        SharedParamHandle {
            value: Arc::clone(&self.value),
        }
    }
}

/// A read-only handle to a [`SharedParam`] for the audio thread.
#[derive(Debug, Clone)]
pub struct SharedParamHandle {
    value: Arc<AtomicU32>,
}

impl SharedParamHandle {
    /// Read the current value.
    pub fn get(&self) -> f32 {
        bits_to_f32(self.value.load(Ordering::Acquire))
    }
}

// ── SmoothedChannel ──────────────────────────────────────────────────────────

/// Combines a [`SharedParam`] with a [`LinearSmoother`].
///
/// The audio thread calls [`SmoothedChannel::next_sample`] each sample.  On
/// each call it checks if the shared value has changed; if so, it starts a new
/// ramp.  This avoids costly per-sample atomic reads by snapshotting the target
/// at the start of each block via [`SmoothedChannel::prepare_block`].
pub struct SmoothedChannel {
    handle: SharedParamHandle,
    smoother: LinearSmoother,
    /// Cached target from the last `prepare_block` call.
    cached_target: f32,
}

impl SmoothedChannel {
    /// Create a new `SmoothedChannel`.
    ///
    /// Returns both the channel (for the audio thread) and the `SharedParam`
    /// (for the control thread).
    pub fn new(initial: f32, ramp_samples: usize) -> (Self, SharedParam) {
        let param = SharedParam::new(initial);
        let handle = param.handle();
        let smoother = LinearSmoother::new(initial, ramp_samples);
        let sc = Self {
            handle,
            smoother,
            cached_target: initial,
        };
        (sc, param)
    }

    /// Snapshot the current shared value and start a ramp if it has changed.
    ///
    /// Call this once at the beginning of each audio buffer callback, before
    /// calling [`next_sample`](Self::next_sample) in the processing loop.
    pub fn prepare_block(&mut self) {
        let new_target = self.handle.get();
        if (new_target - self.cached_target).abs() > f32::EPSILON {
            self.cached_target = new_target;
            self.smoother.set_target(new_target);
        }
    }

    /// Advance by one sample and return the smoothed value.
    #[inline]
    pub fn next_sample(&mut self) -> f32 {
        self.smoother.next_sample()
    }

    /// Fill `output` with smoothed values (calls `prepare_block` first).
    pub fn process_block(&mut self, output: &mut [f32]) {
        self.prepare_block();
        self.smoother.fill_block(output);
    }

    /// Returns the current smoothed value.
    pub fn current(&self) -> f32 {
        self.smoother.current()
    }

    /// Returns whether the smoother is still ramping.
    pub fn is_ramping(&self) -> bool {
        self.smoother.is_ramping()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_smoother_reaches_target() {
        let mut s = LinearSmoother::new(0.0, 100);
        s.set_target(1.0);
        let block = s.next_block(100);
        assert!((block[99] - 1.0).abs() < 1e-5, "last sample = {}", block[99]);
        assert!(!s.is_ramping());
    }

    #[test]
    fn test_linear_smoother_monotone_ramp() {
        let mut s = LinearSmoother::new(0.0, 50);
        s.set_target(1.0);
        let block = s.next_block(50);
        // Each sample should be greater than the previous.
        for w in block.windows(2) {
            assert!(
                w[1] >= w[0],
                "non-monotone: w[0]={} w[1]={}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn test_linear_smoother_no_ramp_when_target_unchanged() {
        let mut s = LinearSmoother::new(0.5, 100);
        // Setting the same target should not start a ramp.
        s.set_target(0.5);
        assert!(!s.is_ramping());
        assert!((s.next_sample() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_linear_smoother_decreasing_ramp() {
        let mut s = LinearSmoother::new(1.0, 50);
        s.set_target(0.0);
        let block = s.next_block(50);
        assert!((block[49]).abs() < 1e-5, "last sample = {}", block[49]);
    }

    #[test]
    fn test_exponential_smoother_approaches_target() {
        let mut s = ExponentialSmoother::new(48000, 0.001); // very fast
        s.set_target(1.0);
        let block = s.next_block(500);
        // After 500 samples with a very fast tau the value should be > 0.99.
        assert!(
            block[499] > 0.99,
            "exponential smoother too slow: {}",
            block[499]
        );
    }

    #[test]
    fn test_exponential_smoother_initial_value() {
        let s = ExponentialSmoother::with_initial(48000, 0.02, 0.75);
        assert!((s.current() - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_shared_param_read_write() {
        let param = SharedParam::new(0.0);
        let handle = param.handle();
        param.set(0.42);
        assert!((handle.get() - 0.42).abs() < 1e-6);
    }

    #[test]
    fn test_smoothed_channel_ramps_on_set() {
        let (mut channel, param) = SmoothedChannel::new(0.0, 128);
        param.set(1.0);
        channel.prepare_block();
        assert!(channel.is_ramping());
        let mut buf = vec![0.0_f32; 128];
        channel.smoother.fill_block(&mut buf);
        assert!((buf[127] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_smoothed_channel_process_block() {
        let (mut channel, param) = SmoothedChannel::new(0.0, 64);
        param.set(1.0);
        let mut buf = vec![0.0_f32; 64];
        channel.process_block(&mut buf);
        assert!((buf[63] - 1.0).abs() < 1e-5, "buf[63]={}", buf[63]);
    }
}
