//! Clip guard — brickwall limiter and clip detection for the `OxiMedia` mixer.
//!
//! Provides:
//!
//! - [`ClipGuard`]: a look-ahead brickwall limiter that prevents digital clipping
//!   by attenuating any signal that would exceed a configurable ceiling.
//! - [`ClipDetector`]: a lightweight clip detector that counts and timestamps
//!   clipping events, supporting both hard (> 0 dBFS) and soft (above a user
//!   threshold) clip detection.
//! - [`OverloadHistory`]: a fixed-size circular buffer of recent overload events
//!   for display on a mixer console.
//!
//! # Design
//!
//! The look-ahead brickwall limiter works by buffering `look_ahead` samples,
//! computing the maximum absolute peak in each buffer segment, and applying a
//! gain reduction so the output never exceeds the ceiling.  Attack/release
//! smoothing prevents audible gain-pumping artefacts.

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// ClipEvent
// ─────────────────────────────────────────────────────────────────────────────

/// A single clip event.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ClipEvent {
    /// Sample index at which the clip occurred (relative to stream start).
    pub sample_index: u64,
    /// Peak linear amplitude that caused the clip.
    pub peak_linear: f32,
    /// Peak level in dBFS.
    pub peak_db: f32,
    /// Whether this was a hard clip (above 0 dBFS).
    pub hard_clip: bool,
}

impl ClipEvent {
    /// Create a new clip event.
    #[must_use]
    pub fn new(sample_index: u64, peak_linear: f32, threshold_linear: f32) -> Self {
        let peak_db = if peak_linear > 0.0 {
            20.0 * peak_linear.log10()
        } else {
            f32::NEG_INFINITY
        };
        Self {
            sample_index,
            peak_linear,
            peak_db,
            hard_clip: peak_linear >= threshold_linear,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OverloadHistory
// ─────────────────────────────────────────────────────────────────────────────

/// Fixed-capacity circular buffer of recent [`ClipEvent`]s.
///
/// When full, the oldest event is silently overwritten.
#[derive(Debug, Clone)]
pub struct OverloadHistory {
    events: Vec<ClipEvent>,
    head: usize,
    len: usize,
    capacity: usize,
}

impl OverloadHistory {
    /// Create a new overload history with the given capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(1);
        Self {
            events: Vec::with_capacity(cap),
            head: 0,
            len: 0,
            capacity: cap,
        }
    }

    /// Push a new clip event into the history.
    pub fn push(&mut self, event: ClipEvent) {
        if self.events.len() < self.capacity {
            self.events.push(event);
            self.len = self.events.len();
        } else {
            self.events[self.head] = event;
            self.head = (self.head + 1) % self.capacity;
        }
    }

    /// Returns the most recent `n` events (or all if fewer than `n` exist).
    #[must_use]
    pub fn recent(&self, n: usize) -> Vec<&ClipEvent> {
        let count = n.min(self.len);
        (0..count)
            .map(|i| {
                let idx = (self.head + self.len - 1 - i) % self.capacity;
                &self.events[idx]
            })
            .collect()
    }

    /// Number of events in the history.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the history is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Clear all events.
    pub fn clear(&mut self) {
        self.events.clear();
        self.head = 0;
        self.len = 0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ClipDetector
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the clip detector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipDetectorConfig {
    /// Threshold above which a *soft* clip is reported (0.0..=1.0, linear).
    /// Hard clips (> 1.0) are always reported regardless.
    pub soft_threshold_linear: f32,
    /// Maximum number of events to retain in history.
    pub history_capacity: usize,
}

impl Default for ClipDetectorConfig {
    fn default() -> Self {
        Self {
            soft_threshold_linear: 0.9,
            history_capacity: 64,
        }
    }
}

/// Clip detector — counts and records clipping events.
#[derive(Debug, Clone)]
pub struct ClipDetector {
    config: ClipDetectorConfig,
    hard_clip_count: u64,
    soft_clip_count: u64,
    sample_cursor: u64,
    history: OverloadHistory,
    /// Peak level seen since last reset.
    peak_linear: f32,
}

impl ClipDetector {
    /// Create a new clip detector.
    #[must_use]
    pub fn new(config: ClipDetectorConfig) -> Self {
        let cap = config.history_capacity;
        Self {
            config,
            hard_clip_count: 0,
            soft_clip_count: 0,
            sample_cursor: 0,
            history: OverloadHistory::new(cap),
            peak_linear: 0.0,
        }
    }

    /// Process a single sample, returning `true` if a clip event was detected.
    pub fn process_sample(&mut self, sample: f32) -> bool {
        let abs = sample.abs();
        if abs > self.peak_linear {
            self.peak_linear = abs;
        }
        let mut clipped = false;
        if abs > 1.0 {
            self.hard_clip_count += 1;
            self.history.push(ClipEvent::new(self.sample_cursor, abs, 1.0));
            clipped = true;
        } else if abs > self.config.soft_threshold_linear {
            self.soft_clip_count += 1;
            self.history.push(ClipEvent::new(
                self.sample_cursor,
                abs,
                self.config.soft_threshold_linear,
            ));
            clipped = true;
        }
        self.sample_cursor += 1;
        clipped
    }

    /// Process a buffer of samples, returning the number of clip events detected.
    pub fn process_buffer(&mut self, buffer: &[f32]) -> u64 {
        let before = self.hard_clip_count + self.soft_clip_count;
        for &s in buffer {
            self.process_sample(s);
        }
        self.hard_clip_count + self.soft_clip_count - before
    }

    /// Total number of hard clips (above 0 dBFS).
    #[must_use]
    pub fn hard_clip_count(&self) -> u64 {
        self.hard_clip_count
    }

    /// Total number of soft clips (above `soft_threshold`).
    #[must_use]
    pub fn soft_clip_count(&self) -> u64 {
        self.soft_clip_count
    }

    /// Peak amplitude seen since last reset.
    #[must_use]
    pub fn peak_linear(&self) -> f32 {
        self.peak_linear
    }

    /// Peak amplitude in dBFS since last reset.
    #[must_use]
    pub fn peak_db(&self) -> f32 {
        if self.peak_linear > 0.0 {
            20.0 * self.peak_linear.log10()
        } else {
            f32::NEG_INFINITY
        }
    }

    /// Access the overload history.
    #[must_use]
    pub fn history(&self) -> &OverloadHistory {
        &self.history
    }

    /// Reset all counters and clear history.
    pub fn reset(&mut self) {
        self.hard_clip_count = 0;
        self.soft_clip_count = 0;
        self.peak_linear = 0.0;
        self.sample_cursor = 0;
        self.history.clear();
    }

    /// Returns `true` if any clip event was detected since last reset.
    #[must_use]
    pub fn has_clipped(&self) -> bool {
        self.hard_clip_count > 0 || self.soft_clip_count > 0
    }
}

impl Default for ClipDetector {
    fn default() -> Self {
        Self::new(ClipDetectorConfig::default())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ClipGuardConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the brickwall limiter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipGuardConfig {
    /// Output ceiling in dBFS (e.g. −0.3 for broadcast safety).
    pub ceiling_db: f32,
    /// Look-ahead window in samples (0 = no look-ahead, just hard clipping).
    pub look_ahead_samples: usize,
    /// Attack time constant in milliseconds.
    pub attack_ms: f32,
    /// Release time constant in milliseconds.
    pub release_ms: f32,
    /// Whether to also count and store clip events.
    pub detect_clips: bool,
}

impl Default for ClipGuardConfig {
    fn default() -> Self {
        Self {
            ceiling_db: -0.3,
            look_ahead_samples: 64,
            attack_ms: 0.1,
            release_ms: 50.0,
            detect_clips: true,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ClipGuard
// ─────────────────────────────────────────────────────────────────────────────

/// Look-ahead brickwall limiter and clip guard.
///
/// Buffers `look_ahead_samples` samples, finds the maximum peak, computes
/// the gain reduction needed to stay below `ceiling_db`, then applies ballistic
/// smoothing and outputs the delayed (look-ahead-compensated) signal.
///
/// When `look_ahead_samples == 0` the limiter degrades gracefully to a single-
/// sample hard clipper (zero latency).
#[derive(Debug)]
pub struct ClipGuard {
    config: ClipGuardConfig,
    /// Linear ceiling.
    ceiling: f32,
    /// Look-ahead delay buffer.
    delay_buf: Vec<f32>,
    /// Write position in the delay buffer.
    write_pos: usize,
    /// Current smoothed gain coefficient (1.0 = no gain reduction).
    gain: f32,
    /// Attack coefficient (IIR one-pole, per sample).
    attack_coeff: f32,
    /// Release coefficient (IIR one-pole, per sample).
    release_coeff: f32,
    /// Optional clip detector.
    detector: Option<ClipDetector>,
}

impl ClipGuard {
    /// Create a new clip guard.
    #[must_use]
    pub fn new(config: ClipGuardConfig, sample_rate: u32) -> Self {
        let ceiling = if config.ceiling_db == 0.0 {
            1.0
        } else {
            10.0_f32.powf(config.ceiling_db / 20.0)
        };
        let sr = sample_rate as f32;
        let attack_coeff = if config.attack_ms <= 0.0 {
            0.0
        } else {
            (-1.0 / (config.attack_ms * 0.001 * sr)).exp()
        };
        let release_coeff = if config.release_ms <= 0.0 {
            0.0
        } else {
            (-1.0 / (config.release_ms * 0.001 * sr)).exp()
        };
        let la = config.look_ahead_samples;
        let detector = if config.detect_clips {
            Some(ClipDetector::default())
        } else {
            None
        };
        Self {
            config,
            ceiling,
            delay_buf: vec![0.0f32; la.max(1)],
            write_pos: 0,
            gain: 1.0,
            attack_coeff,
            release_coeff,
            detector,
        }
    }

    /// Process a single sample through the clip guard.
    ///
    /// Returns the gain-limited (and look-ahead delayed) output sample.
    #[must_use]
    pub fn process_sample(&mut self, input: f32) -> f32 {
        let la = self.config.look_ahead_samples;

        if la == 0 {
            // No look-ahead: just hard clip.
            let output = input.clamp(-self.ceiling, self.ceiling);
            if let Some(det) = &mut self.detector {
                det.process_sample(input);
            }
            return output;
        }

        // Store sample in delay buffer.
        self.delay_buf[self.write_pos] = input;

        // Compute the peak in the upcoming look-ahead window.
        let peak = self
            .delay_buf
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);

        // Target gain needed to stay below ceiling.
        let target_gain = if peak > self.ceiling {
            self.ceiling / peak
        } else {
            1.0
        };

        // Apply ballistic smoothing.
        let coeff = if target_gain < self.gain {
            self.attack_coeff
        } else {
            self.release_coeff
        };
        self.gain = coeff * self.gain + (1.0 - coeff) * target_gain;

        // Read the oldest sample from the delay buffer (the actual output).
        let read_pos = (self.write_pos + 1) % la;
        let delayed = self.delay_buf[read_pos];
        self.write_pos = (self.write_pos + 1) % la;

        let output = delayed * self.gain;

        if let Some(det) = &mut self.detector {
            det.process_sample(output);
        }

        output
    }

    /// Process a buffer of samples in-place.
    pub fn process_buffer(&mut self, buffer: &mut [f32]) {
        for sample in buffer.iter_mut() {
            *sample = self.process_sample(*sample);
        }
    }

    /// Current smoothed gain (1.0 = no reduction).
    #[must_use]
    pub fn current_gain(&self) -> f32 {
        self.gain
    }

    /// Current gain reduction in dB (positive = reduction amount).
    #[must_use]
    pub fn gain_reduction_db(&self) -> f32 {
        if self.gain <= 0.0 {
            f32::INFINITY
        } else {
            -20.0 * self.gain.log10()
        }
    }

    /// Access the clip detector (if enabled).
    #[must_use]
    pub fn detector(&self) -> Option<&ClipDetector> {
        self.detector.as_ref()
    }

    /// Reset internal state.
    pub fn reset(&mut self) {
        self.delay_buf.fill(0.0);
        self.write_pos = 0;
        self.gain = 1.0;
        if let Some(det) = &mut self.detector {
            det.reset();
        }
    }

    /// Get the ceiling in linear amplitude.
    #[must_use]
    pub fn ceiling(&self) -> f32 {
        self.ceiling
    }

    /// Get the look-ahead latency in samples.
    #[must_use]
    pub fn latency_samples(&self) -> usize {
        self.config.look_ahead_samples
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ClipDetector ─────────────────────────────────────────────────────────

    #[test]
    fn test_clip_detector_no_clip_on_normal_signal() {
        let mut det = ClipDetector::default();
        let buf: Vec<f32> = (0..100).map(|_| 0.5f32).collect();
        let events = det.process_buffer(&buf);
        assert_eq!(events, 0);
        assert!(!det.has_clipped());
    }

    #[test]
    fn test_clip_detector_hard_clip_detected() {
        let mut det = ClipDetector::default();
        det.process_sample(1.5f32);
        assert_eq!(det.hard_clip_count(), 1);
        assert!(det.has_clipped());
    }

    #[test]
    fn test_clip_detector_soft_clip_detected() {
        let mut det = ClipDetector::new(ClipDetectorConfig {
            soft_threshold_linear: 0.8,
            ..Default::default()
        });
        det.process_sample(0.85f32);
        assert_eq!(det.soft_clip_count(), 1);
        assert_eq!(det.hard_clip_count(), 0);
    }

    #[test]
    fn test_clip_detector_peak_tracking() {
        let mut det = ClipDetector::default();
        det.process_sample(0.3);
        det.process_sample(0.7);
        det.process_sample(0.5);
        assert!((det.peak_linear() - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_clip_detector_reset_clears_counts() {
        let mut det = ClipDetector::default();
        det.process_sample(2.0);
        det.reset();
        assert_eq!(det.hard_clip_count(), 0);
        assert_eq!(det.soft_clip_count(), 0);
        assert!(!det.has_clipped());
    }

    #[test]
    fn test_clip_detector_peak_db() {
        let mut det = ClipDetector::default();
        det.process_sample(1.0);
        // peak_linear = 1.0 → peak_db = 0 dBFS
        assert!((det.peak_db()).abs() < 0.01);
    }

    #[test]
    fn test_clip_detector_history_stores_events() {
        let mut det = ClipDetector::default();
        det.process_sample(2.0);
        det.process_sample(1.5);
        assert_eq!(det.history().len(), 2);
    }

    // ── OverloadHistory ───────────────────────────────────────────────────────

    #[test]
    fn test_overload_history_capacity() {
        let mut hist = OverloadHistory::new(3);
        for i in 0..5u64 {
            hist.push(ClipEvent::new(i, 1.1, 1.0));
        }
        assert_eq!(hist.len(), 3, "should not exceed capacity");
    }

    #[test]
    fn test_overload_history_recent() {
        let mut hist = OverloadHistory::new(10);
        for i in 0..5u64 {
            hist.push(ClipEvent::new(i, 1.1, 1.0));
        }
        let recent = hist.recent(3);
        assert_eq!(recent.len(), 3);
    }

    #[test]
    fn test_overload_history_clear() {
        let mut hist = OverloadHistory::new(10);
        hist.push(ClipEvent::new(0, 1.2, 1.0));
        hist.clear();
        assert!(hist.is_empty());
    }

    // ── ClipGuard ─────────────────────────────────────────────────────────────

    #[test]
    fn test_clip_guard_passes_signal_below_ceiling() {
        let config = ClipGuardConfig {
            ceiling_db: 0.0,
            look_ahead_samples: 0, // no look-ahead for simpler testing
            detect_clips: false,
            ..Default::default()
        };
        let mut guard = ClipGuard::new(config, 48000);
        let output = guard.process_sample(0.5);
        assert!((output - 0.5).abs() < 1e-4, "should pass signal unchanged: {output}");
    }

    #[test]
    fn test_clip_guard_hard_limits_above_ceiling() {
        let config = ClipGuardConfig {
            ceiling_db: 0.0,
            look_ahead_samples: 0,
            detect_clips: false,
            ..Default::default()
        };
        let mut guard = ClipGuard::new(config, 48000);
        let output = guard.process_sample(2.0);
        assert!(
            output <= 1.0 + 1e-4,
            "signal above ceiling should be limited, got {output}"
        );
    }

    #[test]
    fn test_clip_guard_ceiling_negative_db() {
        let config = ClipGuardConfig {
            ceiling_db: -6.0,
            look_ahead_samples: 0,
            detect_clips: false,
            ..Default::default()
        };
        let mut guard = ClipGuard::new(config, 48000);
        let ceiling = guard.ceiling();
        // −6 dBFS ≈ 0.5012
        assert!((ceiling - 0.5012).abs() < 0.01, "ceiling={ceiling}");
    }

    #[test]
    fn test_clip_guard_reset_clears_state() {
        let config = ClipGuardConfig::default();
        let mut guard = ClipGuard::new(config, 48000);
        // Process a clipping signal.
        let mut buf = vec![2.0f32; 128];
        guard.process_buffer(&mut buf);
        guard.reset();
        assert!((guard.current_gain() - 1.0).abs() < 1e-4, "gain should reset to 1.0");
    }

    #[test]
    fn test_clip_guard_latency_reported() {
        let config = ClipGuardConfig {
            look_ahead_samples: 32,
            ..Default::default()
        };
        let guard = ClipGuard::new(config, 48000);
        assert_eq!(guard.latency_samples(), 32);
    }

    #[test]
    fn test_clip_guard_no_look_ahead_no_latency() {
        let config = ClipGuardConfig {
            look_ahead_samples: 0,
            ..Default::default()
        };
        let guard = ClipGuard::new(config, 48000);
        assert_eq!(guard.latency_samples(), 0);
    }

    #[test]
    fn test_clip_guard_gain_reduction_db_unity() {
        let config = ClipGuardConfig {
            look_ahead_samples: 0,
            ..Default::default()
        };
        let guard = ClipGuard::new(config, 48000);
        // At unity gain, gain_reduction_db should be ~0.
        assert!(
            guard.gain_reduction_db().abs() < 0.01,
            "at unity gain_reduction_db should be 0, got {}",
            guard.gain_reduction_db()
        );
    }

    #[test]
    fn test_clip_guard_detector_present_by_default() {
        let config = ClipGuardConfig::default();
        let guard = ClipGuard::new(config, 48000);
        assert!(guard.detector().is_some());
    }

    #[test]
    fn test_clip_guard_detector_absent_when_disabled() {
        let config = ClipGuardConfig {
            detect_clips: false,
            ..Default::default()
        };
        let guard = ClipGuard::new(config, 48000);
        assert!(guard.detector().is_none());
    }

    #[test]
    fn test_clip_event_hard_clip_flag() {
        let event = ClipEvent::new(0, 1.2, 1.0);
        assert!(event.hard_clip);
        let soft_event = ClipEvent::new(0, 0.95, 1.0);
        assert!(!soft_event.hard_clip);
    }
}
