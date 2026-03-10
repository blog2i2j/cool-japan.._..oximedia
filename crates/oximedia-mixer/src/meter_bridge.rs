//! Meter bridge — multi-channel peak/RMS metering for a mixer console row.
//!
//! The `MeterBridge` tracks peak and RMS levels for every channel in real time,
//! supports configurable hold and clip detection, and provides a clean snapshot
//! API for UI updates.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Configuration for a meter bridge.
#[derive(Debug, Clone)]
pub struct MeterConfig {
    /// Peak-hold duration in seconds (0 = no hold).
    pub peak_hold_secs: f32,
    /// Whether to compute RMS as well as peak.
    pub rms_enabled: bool,
    /// RMS averaging window in frames.
    pub rms_window_frames: usize,
    /// Clip threshold (linear, default 1.0 = 0 dBFS).
    pub clip_threshold: f32,
    /// Sample rate, used for hold timing.
    pub sample_rate: u32,
}

impl Default for MeterConfig {
    fn default() -> Self {
        Self {
            peak_hold_secs: 2.0,
            rms_enabled: true,
            rms_window_frames: 512,
            clip_threshold: 1.0,
            sample_rate: 48000,
        }
    }
}

impl MeterConfig {
    /// Builder: enable RMS and set the window size.
    #[must_use]
    pub fn with_rms(mut self, window_frames: usize) -> Self {
        self.rms_enabled = true;
        self.rms_window_frames = window_frames;
        self
    }

    /// Builder: set peak-hold duration in seconds.
    #[must_use]
    pub fn with_peak_hold(mut self, secs: f32) -> Self {
        self.peak_hold_secs = secs;
        self
    }

    /// Builder: set the clip threshold.
    #[must_use]
    pub fn with_clip_threshold(mut self, threshold: f32) -> Self {
        self.clip_threshold = threshold;
        self
    }
}

/// Per-channel meter state.
#[derive(Debug, Clone)]
pub struct MeterChannel {
    /// Channel name.
    pub name: String,
    /// Current instantaneous peak level (linear).
    pub peak: f32,
    /// Held peak level (linear).
    pub held_peak: f32,
    /// RMS level (linear).  `None` if RMS is disabled.
    pub rms: Option<f32>,
    /// Frames remaining on the current peak hold.
    hold_frames_remaining: usize,
    /// Clip flag — set when the peak has ever exceeded `clip_threshold`.
    clip_detected: bool,
    /// Ring buffer for RMS computation.
    rms_buffer: Vec<f32>,
    /// Write index into the RMS ring buffer.
    rms_write_idx: usize,
}

impl MeterChannel {
    /// Create a new `MeterChannel` with the given name.
    #[must_use]
    pub fn new(name: impl Into<String>, rms_window_frames: usize) -> Self {
        Self {
            name: name.into(),
            peak: 0.0,
            held_peak: 0.0,
            rms: None,
            hold_frames_remaining: 0,
            clip_detected: false,
            rms_buffer: vec![0.0; rms_window_frames.max(1)],
            rms_write_idx: 0,
        }
    }

    /// Returns `true` if the clip indicator is set.
    #[must_use]
    pub fn is_clipping(&self) -> bool {
        self.clip_detected
    }

    /// Reset the clip indicator.
    pub fn reset_clip(&mut self) {
        self.clip_detected = false;
    }

    /// Update this channel with a new peak sample and optional RMS sample.
    ///
    /// `hold_frames` is how many frames the peak should be held for.
    pub fn update(
        &mut self,
        new_peak: f32,
        clip_threshold: f32,
        hold_frames: usize,
        rms_sample: Option<f32>,
    ) {
        self.peak = new_peak;

        if new_peak >= clip_threshold {
            self.clip_detected = true;
        }

        if new_peak >= self.held_peak {
            self.held_peak = new_peak;
            self.hold_frames_remaining = hold_frames;
        } else if self.hold_frames_remaining > 0 {
            self.hold_frames_remaining -= 1;
        } else {
            // Decay held peak
            self.held_peak = (self.held_peak - 0.001).max(new_peak);
        }

        if let Some(s) = rms_sample {
            self.rms_buffer[self.rms_write_idx] = s * s;
            self.rms_write_idx = (self.rms_write_idx + 1) % self.rms_buffer.len();
            let mean_sq: f32 = self.rms_buffer.iter().sum::<f32>() / self.rms_buffer.len() as f32;
            self.rms = Some(mean_sq.sqrt());
        }
    }
}

/// A row of meters — one per channel in the mixer console.
#[derive(Debug, Clone)]
pub struct MeterBridge {
    channels: Vec<MeterChannel>,
    config: MeterConfig,
}

impl MeterBridge {
    /// Create a new `MeterBridge` with `channel_count` channels.
    #[must_use]
    pub fn new(channel_count: usize, config: MeterConfig) -> Self {
        let window = config.rms_window_frames;
        let channels = (0..channel_count)
            .map(|i| MeterChannel::new(format!("Ch{}", i + 1), window))
            .collect();
        Self { channels, config }
    }

    /// Update a single channel with a new peak value.
    ///
    /// The RMS sample (if any) is the same value; real implementations would
    /// pass in the computed RMS from the audio engine.
    pub fn update_channel(&mut self, index: usize, peak: f32) {
        if index >= self.channels.len() {
            return;
        }
        let hold_frames = (self.config.peak_hold_secs * self.config.sample_rate as f32
            / self.config.rms_window_frames as f32) as usize;
        let rms_sample = if self.config.rms_enabled {
            Some(peak)
        } else {
            None
        };
        let threshold = self.config.clip_threshold;
        self.channels[index].update(peak, threshold, hold_frames, rms_sample);
    }

    /// Return the indices of channels currently showing a clip.
    #[must_use]
    pub fn peak_channels(&self) -> Vec<usize> {
        self.channels
            .iter()
            .enumerate()
            .filter(|(_, ch)| ch.is_clipping())
            .map(|(i, _)| i)
            .collect()
    }

    /// Reset all clip indicators.
    pub fn reset_peaks(&mut self) {
        for ch in &mut self.channels {
            ch.reset_clip();
            ch.held_peak = 0.0;
        }
    }

    /// Return a reference to a specific channel meter.
    #[must_use]
    pub fn channel(&self, index: usize) -> Option<&MeterChannel> {
        self.channels.get(index)
    }

    /// Return the number of channels in this bridge.
    #[must_use]
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    /// Return the config.
    #[must_use]
    pub fn config(&self) -> &MeterConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meter_config_default() {
        let cfg = MeterConfig::default();
        assert_eq!(cfg.sample_rate, 48000);
        assert!(cfg.rms_enabled);
    }

    #[test]
    fn test_meter_config_with_rms() {
        let cfg = MeterConfig::default().with_rms(1024);
        assert_eq!(cfg.rms_window_frames, 1024);
        assert!(cfg.rms_enabled);
    }

    #[test]
    fn test_meter_config_with_peak_hold() {
        let cfg = MeterConfig::default().with_peak_hold(3.5);
        assert!((cfg.peak_hold_secs - 3.5).abs() < 1e-6);
    }

    #[test]
    fn test_meter_channel_initial_state() {
        let ch = MeterChannel::new("L", 512);
        assert_eq!(ch.peak, 0.0);
        assert!(!ch.is_clipping());
    }

    #[test]
    fn test_meter_channel_clip_detection() {
        let mut ch = MeterChannel::new("L", 512);
        ch.update(1.1, 1.0, 10, None);
        assert!(ch.is_clipping());
    }

    #[test]
    fn test_meter_channel_no_clip_below_threshold() {
        let mut ch = MeterChannel::new("L", 512);
        ch.update(0.9, 1.0, 10, None);
        assert!(!ch.is_clipping());
    }

    #[test]
    fn test_meter_channel_reset_clip() {
        let mut ch = MeterChannel::new("L", 512);
        ch.update(1.5, 1.0, 10, None);
        assert!(ch.is_clipping());
        ch.reset_clip();
        assert!(!ch.is_clipping());
    }

    #[test]
    fn test_meter_channel_rms_computed() {
        let mut ch = MeterChannel::new("L", 4);
        // Feed unity signal into all window slots
        for _ in 0..4 {
            ch.update(1.0, 2.0, 10, Some(1.0));
        }
        let rms = ch.rms.expect("rms should be valid");
        assert!((rms - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_meter_bridge_creation() {
        let bridge = MeterBridge::new(8, MeterConfig::default());
        assert_eq!(bridge.channel_count(), 8);
    }

    #[test]
    fn test_meter_bridge_update_channel() {
        let mut bridge = MeterBridge::new(4, MeterConfig::default());
        bridge.update_channel(2, 0.8);
        assert!((bridge.channel(2).expect("channel should succeed").peak - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_meter_bridge_peak_channels() {
        let mut bridge = MeterBridge::new(4, MeterConfig::default().with_clip_threshold(0.5));
        bridge.update_channel(1, 0.9); // clips
        bridge.update_channel(3, 0.2); // no clip
        let clipping = bridge.peak_channels();
        assert!(clipping.contains(&1));
        assert!(!clipping.contains(&3));
    }

    #[test]
    fn test_meter_bridge_reset_peaks() {
        let mut bridge = MeterBridge::new(2, MeterConfig::default().with_clip_threshold(0.5));
        bridge.update_channel(0, 1.0);
        bridge.update_channel(1, 1.0);
        bridge.reset_peaks();
        assert!(bridge.peak_channels().is_empty());
    }

    #[test]
    fn test_meter_bridge_out_of_bounds_ignored() {
        let mut bridge = MeterBridge::new(2, MeterConfig::default());
        // Should not panic
        bridge.update_channel(99, 1.0);
    }
}
