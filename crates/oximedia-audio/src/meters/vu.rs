//! VU (Volume Unit) meter implementation.
//!
//! Implements IEC 60268-10 VU meter standard with proper ballistics.

use super::ballistics::{linear_to_db, BallisticsConfig, BallisticsProcessor, RmsWindow};
use crate::frame::AudioFrame;

/// VU meter implementation (IEC 60268-10).
///
/// VU meters measure average audio level with slow ballistics:
/// - 300ms integration time
/// - Scale: -20 dBVU to +3 dBVU
/// - Reference: 0 dBVU = +4 dBu (nominal operating level)
/// - Overload: Readings above 0 dBVU
pub struct VuMeter {
    /// Ballistics processors (one per channel).
    processors: Vec<BallisticsProcessor>,
    /// RMS windows (one per channel).
    rms_windows: Vec<RmsWindow>,
    /// Sample rate in Hz.
    #[allow(dead_code)]
    sample_rate: f64,
    /// Number of channels.
    #[allow(dead_code)]
    channels: usize,
    /// Reference level in dBFS (default: -18 dBFS = 0 dBVU).
    reference_level_dbfs: f64,
    /// Current VU readings per channel (dBVU).
    vu_readings: Vec<f64>,
    /// Peak VU readings per channel (dBVU).
    peak_vu_readings: Vec<f64>,
}

impl VuMeter {
    /// Create a new VU meter.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz
    /// * `channels` - Number of audio channels
    #[must_use]
    pub fn new(sample_rate: f64, channels: usize) -> Self {
        Self::with_reference(sample_rate, channels, -18.0)
    }

    /// Create a VU meter with custom reference level.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz
    /// * `channels` - Number of channels
    /// * `reference_dbfs` - Reference level in dBFS (0 dBVU corresponds to this)
    #[must_use]
    pub fn with_reference(sample_rate: f64, channels: usize, reference_dbfs: f64) -> Self {
        let config = BallisticsConfig::vu_meter(sample_rate);
        let processors = (0..channels)
            .map(|_| BallisticsProcessor::new(config.clone()))
            .collect();

        let rms_windows = (0..channels)
            .map(|_| RmsWindow::new(0.3, sample_rate))
            .collect();

        Self {
            processors,
            rms_windows,
            sample_rate,
            channels,
            reference_level_dbfs: reference_dbfs,
            vu_readings: vec![f64::NEG_INFINITY; channels],
            peak_vu_readings: vec![f64::NEG_INFINITY; channels],
        }
    }

    /// Process an audio frame and update VU readings.
    ///
    /// # Arguments
    ///
    /// * `frame` - Audio frame to process
    pub fn process(&mut self, frame: &AudioFrame) {
        let samples = extract_samples_f64(frame);
        let num_samples = samples.len() / self.channels;

        for i in 0..num_samples {
            for ch in 0..self.channels {
                let idx = i * self.channels + ch;
                if let Some(&sample) = samples.get(idx) {
                    // Compute RMS
                    let rms = self.rms_windows[ch].process(sample);

                    // Convert to dBFS, clamping to a finite floor so that
                    // -inf (from silence / zero-crossings) does not poison
                    // the ballistics IIR filter.
                    let db_fs = linear_to_db(rms).max(-100.0);

                    // Convert to dBVU (relative to reference)
                    let db_vu = db_fs - self.reference_level_dbfs;

                    // Apply ballistics
                    let reading = self.processors[ch].process(db_vu);

                    // Update readings
                    self.vu_readings[ch] = reading;
                    self.peak_vu_readings[ch] = self.peak_vu_readings[ch].max(reading);
                }
            }
        }
    }

    /// Get current VU reading for a channel in dBVU.
    ///
    /// # Arguments
    ///
    /// * `channel` - Channel index
    ///
    /// # Returns
    ///
    /// VU reading in dBVU (range: -20 to +3)
    #[must_use]
    pub fn vu_reading(&self, channel: usize) -> f64 {
        self.vu_readings
            .get(channel)
            .copied()
            .unwrap_or(f64::NEG_INFINITY)
    }

    /// Get peak VU reading for a channel in dBVU.
    #[must_use]
    pub fn peak_vu_reading(&self, channel: usize) -> f64 {
        self.peak_vu_readings
            .get(channel)
            .copied()
            .unwrap_or(f64::NEG_INFINITY)
    }

    /// Get stereo VU reading (average of L/R for stereo, or mono for mono).
    #[must_use]
    pub fn stereo_vu_reading(&self) -> f64 {
        if self.channels == 1 {
            self.vu_reading(0)
        } else if self.channels >= 2 {
            let left = self.vu_reading(0);
            let right = self.vu_reading(1);
            if left.is_finite() && right.is_finite() {
                (left + right) / 2.0
            } else if left.is_finite() {
                left
            } else {
                right
            }
        } else {
            f64::NEG_INFINITY
        }
    }

    /// Get normalized VU reading (0.0 to 1.0) for visualization.
    ///
    /// Maps -20 dBVU to 0.0 and +3 dBVU to 1.0.
    ///
    /// # Arguments
    ///
    /// * `channel` - Channel index
    #[must_use]
    pub fn normalized_reading(&self, channel: usize) -> f64 {
        let db_vu = self.vu_reading(channel);
        normalize_vu(db_vu)
    }

    /// Get peak normalized VU reading.
    #[must_use]
    pub fn normalized_peak(&self, channel: usize) -> f64 {
        let db_vu = self.peak_vu_reading(channel);
        normalize_vu(db_vu)
    }

    /// Check if channel is in overload (> 0 dBVU).
    #[must_use]
    pub fn is_overload(&self, channel: usize) -> bool {
        self.vu_reading(channel) > 0.0
    }

    /// Get visualization data for a channel.
    #[must_use]
    pub fn visualization_data(&self, channel: usize) -> VuVisualization {
        let reading = self.vu_reading(channel);
        let normalized = self.normalized_reading(channel);
        let peak = self.peak_vu_reading(channel);
        let normalized_peak = self.normalized_peak(channel);

        VuVisualization {
            db_vu: reading,
            normalized,
            peak_db_vu: peak,
            normalized_peak,
            overload: reading > 0.0,
            color_zone: get_color_zone(reading),
        }
    }

    /// Get all channels visualization data.
    #[must_use]
    pub fn all_channels_visualization(&self) -> Vec<VuVisualization> {
        (0..self.channels)
            .map(|ch| self.visualization_data(ch))
            .collect()
    }

    /// Reset all VU readings.
    pub fn reset(&mut self) {
        for processor in &mut self.processors {
            processor.reset();
        }
        for window in &mut self.rms_windows {
            window.reset();
        }
        self.vu_readings.fill(f64::NEG_INFINITY);
        self.peak_vu_readings.fill(f64::NEG_INFINITY);
    }

    /// Reset peak VU readings only.
    pub fn reset_peaks(&mut self) {
        for processor in &mut self.processors {
            processor.reset_peak_hold();
        }
        self.peak_vu_readings.fill(f64::NEG_INFINITY);
    }

    /// Set reference level.
    pub fn set_reference_level(&mut self, reference_dbfs: f64) {
        self.reference_level_dbfs = reference_dbfs;
    }

    /// Get reference level.
    #[must_use]
    pub fn reference_level(&self) -> f64 {
        self.reference_level_dbfs
    }
}

/// VU meter visualization data.
#[derive(Clone, Debug)]
pub struct VuVisualization {
    /// VU reading in dBVU.
    pub db_vu: f64,
    /// Normalized value (0.0 to 1.0).
    pub normalized: f64,
    /// Peak VU reading in dBVU.
    pub peak_db_vu: f64,
    /// Normalized peak value (0.0 to 1.0).
    pub normalized_peak: f64,
    /// Overload indicator.
    pub overload: bool,
    /// Color zone for the current reading.
    pub color_zone: ColorZone,
}

impl VuVisualization {
    /// Get scale markings for VU meter display.
    #[must_use]
    pub fn scale_markings() -> Vec<(f64, String)> {
        vec![
            (-20.0, "-20".to_string()),
            (-10.0, "-10".to_string()),
            (-7.0, "-7".to_string()),
            (-5.0, "-5".to_string()),
            (-3.0, "-3".to_string()),
            (0.0, "0".to_string()),
            (3.0, "+3".to_string()),
        ]
    }

    /// Convert dBVU to normalized position.
    #[must_use]
    pub fn db_to_normalized(db_vu: f64) -> f64 {
        normalize_vu(db_vu)
    }
}

/// Color zone for meter display.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorZone {
    /// Green zone (safe operation).
    Green,
    /// Yellow zone (approaching limits).
    Yellow,
    /// Red zone (overload).
    Red,
}

/// Normalize VU reading to 0.0-1.0 range.
///
/// Maps -20 dBVU to 0.0 and +3 dBVU to 1.0.
#[must_use]
fn normalize_vu(db_vu: f64) -> f64 {
    const MIN_DB: f64 = -20.0;
    const MAX_DB: f64 = 3.0;

    if db_vu.is_infinite() && db_vu.is_sign_negative() {
        0.0
    } else {
        ((db_vu - MIN_DB) / (MAX_DB - MIN_DB)).clamp(0.0, 1.0)
    }
}

/// Get color zone for VU reading.
#[must_use]
fn get_color_zone(db_vu: f64) -> ColorZone {
    if db_vu > 0.0 {
        ColorZone::Red
    } else if db_vu > -3.0 {
        ColorZone::Yellow
    } else {
        ColorZone::Green
    }
}

/// Extract samples from audio frame as f64.
#[allow(dead_code)]
fn extract_samples_f64(frame: &AudioFrame) -> Vec<f64> {
    match &frame.samples {
        crate::frame::AudioBuffer::Interleaved(data) => bytes_to_samples_f64(data),
        crate::frame::AudioBuffer::Planar(planes) => {
            if planes.is_empty() {
                return Vec::new();
            }

            let channels = planes.len();
            let sample_size = std::mem::size_of::<f32>();
            let frames = planes[0].len() / sample_size;
            let mut interleaved = Vec::with_capacity(frames * channels);

            for frame_idx in 0..frames {
                for plane in planes {
                    let samples = bytes_to_samples_f64(plane);
                    if let Some(&sample) = samples.get(frame_idx) {
                        interleaved.push(sample);
                    }
                }
            }

            interleaved
        }
    }
}

/// Convert bytes to f64 samples (assumes f32 format).
fn bytes_to_samples_f64(bytes: &bytes::Bytes) -> Vec<f64> {
    let sample_count = bytes.len() / 4;
    let mut samples = Vec::with_capacity(sample_count);

    for i in 0..sample_count {
        let offset = i * 4;
        if offset + 4 <= bytes.len() {
            let bytes_array = [
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ];
            let sample = f32::from_le_bytes(bytes_array);
            samples.push(f64::from(sample));
        }
    }

    samples
}
