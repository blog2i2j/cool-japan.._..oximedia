#![allow(dead_code)]
//! Meter bridge aggregation for broadcast metering.
//!
//! Provides a unified meter bridge that combines multiple meter types
//! (peak, VU, loudness, phase) into a single coordinated view, similar
//! to hardware meter bridges found in professional broadcast consoles.

use std::collections::HashMap;

/// Type of meter in the bridge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MeterType {
    /// Peak program meter (PPM).
    Peak,
    /// VU meter (averaging/ballistic).
    Vu,
    /// Loudness meter (LUFS/LKFS).
    Loudness,
    /// Phase correlation meter.
    Phase,
    /// True peak meter.
    TruePeak,
    /// RMS level meter.
    Rms,
    /// Dynamic range meter.
    DynamicRange,
    /// Crest factor meter.
    CrestFactor,
}

impl std::fmt::Display for MeterType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Peak => write!(f, "Peak"),
            Self::Vu => write!(f, "VU"),
            Self::Loudness => write!(f, "Loudness"),
            Self::Phase => write!(f, "Phase"),
            Self::TruePeak => write!(f, "True Peak"),
            Self::Rms => write!(f, "RMS"),
            Self::DynamicRange => write!(f, "Dynamic Range"),
            Self::CrestFactor => write!(f, "Crest Factor"),
        }
    }
}

/// Status of an individual meter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MeterStatus {
    /// Meter is active and receiving data.
    #[default]
    Active,
    /// Meter is idle (no signal).
    Idle,
    /// Meter is in clip/overload condition.
    Clip,
    /// Meter is in warning range.
    Warning,
    /// Meter has an error.
    Error,
}

/// Reading from a single meter channel.
#[derive(Clone, Debug)]
pub struct MeterReading {
    /// Current level in dB.
    pub level_db: f64,
    /// Peak hold level in dB.
    pub peak_hold_db: f64,
    /// Minimum level seen in dB.
    pub min_db: f64,
    /// Maximum level seen in dB.
    pub max_db: f64,
    /// Status of the meter.
    pub status: MeterStatus,
    /// Channel index.
    pub channel: usize,
}

impl MeterReading {
    /// Create a new meter reading.
    pub fn new(level_db: f64, peak_hold_db: f64, channel: usize) -> Self {
        Self {
            level_db,
            peak_hold_db,
            min_db: level_db,
            max_db: peak_hold_db,
            status: MeterStatus::Active,
            channel,
        }
    }

    /// Create a silent reading.
    pub fn silent(channel: usize) -> Self {
        Self {
            level_db: f64::NEG_INFINITY,
            peak_hold_db: f64::NEG_INFINITY,
            min_db: f64::NEG_INFINITY,
            max_db: f64::NEG_INFINITY,
            status: MeterStatus::Idle,
            channel,
        }
    }

    /// Check if the meter is clipping (above 0 dBFS).
    pub fn is_clipping(&self) -> bool {
        self.peak_hold_db > -0.1
    }

    /// Get the dynamic range (difference between max and min).
    pub fn dynamic_range_db(&self) -> f64 {
        if self.max_db.is_finite() && self.min_db.is_finite() {
            self.max_db - self.min_db
        } else {
            0.0
        }
    }
}

impl Default for MeterReading {
    fn default() -> Self {
        Self::silent(0)
    }
}

/// Configuration for a meter slot in the bridge.
#[derive(Clone, Debug)]
pub struct MeterSlotConfig {
    /// Name/label for this slot.
    pub label: String,
    /// Type of meter.
    pub meter_type: MeterType,
    /// Number of channels.
    pub channels: usize,
    /// Warning threshold in dB.
    pub warning_threshold_db: f64,
    /// Clip threshold in dB.
    pub clip_threshold_db: f64,
    /// Reference level in dB (for VU meters).
    pub reference_level_db: f64,
    /// Peak hold time in seconds.
    pub peak_hold_seconds: f64,
    /// Whether this meter is enabled.
    pub enabled: bool,
}

impl Default for MeterSlotConfig {
    fn default() -> Self {
        Self {
            label: String::from("Meter"),
            meter_type: MeterType::Peak,
            channels: 2,
            warning_threshold_db: -6.0,
            clip_threshold_db: -0.1,
            reference_level_db: -18.0,
            peak_hold_seconds: 2.0,
            enabled: true,
        }
    }
}

impl MeterSlotConfig {
    /// Create a PPM slot config.
    pub fn ppm(label: &str, channels: usize) -> Self {
        Self {
            label: label.to_string(),
            meter_type: MeterType::Peak,
            channels,
            ..Default::default()
        }
    }

    /// Create a VU slot config.
    pub fn vu(label: &str, channels: usize) -> Self {
        Self {
            label: label.to_string(),
            meter_type: MeterType::Vu,
            channels,
            reference_level_db: -18.0,
            ..Default::default()
        }
    }

    /// Create a loudness slot config.
    pub fn loudness(label: &str, channels: usize) -> Self {
        Self {
            label: label.to_string(),
            meter_type: MeterType::Loudness,
            channels,
            warning_threshold_db: -22.0,
            clip_threshold_db: -20.0,
            ..Default::default()
        }
    }
}

/// A meter slot in the bridge with current readings.
#[derive(Clone, Debug)]
pub struct MeterSlot {
    /// Configuration.
    pub config: MeterSlotConfig,
    /// Current readings per channel.
    pub readings: Vec<MeterReading>,
    /// Overall status (worst-case across channels).
    pub overall_status: MeterStatus,
    /// Timestamp of last update (samples processed).
    pub last_update_samples: u64,
}

impl MeterSlot {
    /// Create a new meter slot.
    pub fn new(config: MeterSlotConfig) -> Self {
        let readings = (0..config.channels).map(MeterReading::silent).collect();
        Self {
            config,
            readings,
            overall_status: MeterStatus::Idle,
            last_update_samples: 0,
        }
    }

    /// Update a channel reading.
    pub fn update_channel(&mut self, channel: usize, level_db: f64, peak_db: f64) {
        if channel >= self.readings.len() {
            return;
        }
        let reading = &mut self.readings[channel];
        reading.level_db = level_db;
        reading.peak_hold_db = peak_db.max(reading.peak_hold_db);
        reading.min_db = level_db.min(reading.min_db);
        reading.max_db = peak_db.max(reading.max_db);

        // Update status.
        reading.status = if peak_db > self.config.clip_threshold_db {
            MeterStatus::Clip
        } else if level_db > self.config.warning_threshold_db {
            MeterStatus::Warning
        } else if level_db > -60.0 {
            MeterStatus::Active
        } else {
            MeterStatus::Idle
        };

        // Update overall status.
        self.update_overall_status();
    }

    /// Update all channels from interleaved peak and level data.
    pub fn update_all(&mut self, levels_db: &[f64], peaks_db: &[f64]) {
        for ch in 0..self
            .config
            .channels
            .min(levels_db.len())
            .min(peaks_db.len())
        {
            self.update_channel(ch, levels_db[ch], peaks_db[ch]);
        }
    }

    /// Recalculate overall status from individual channels.
    fn update_overall_status(&mut self) {
        self.overall_status = MeterStatus::Idle;
        for reading in &self.readings {
            match reading.status {
                MeterStatus::Clip => {
                    self.overall_status = MeterStatus::Clip;
                    return;
                }
                MeterStatus::Error => {
                    self.overall_status = MeterStatus::Error;
                    return;
                }
                MeterStatus::Warning => {
                    if self.overall_status != MeterStatus::Error {
                        self.overall_status = MeterStatus::Warning;
                    }
                }
                MeterStatus::Active => {
                    if self.overall_status == MeterStatus::Idle {
                        self.overall_status = MeterStatus::Active;
                    }
                }
                MeterStatus::Idle => {}
            }
        }
    }

    /// Reset all readings.
    pub fn reset(&mut self) {
        for (i, reading) in self.readings.iter_mut().enumerate() {
            *reading = MeterReading::silent(i);
        }
        self.overall_status = MeterStatus::Idle;
        self.last_update_samples = 0;
    }

    /// Check if any channel is clipping.
    pub fn is_clipping(&self) -> bool {
        self.readings.iter().any(MeterReading::is_clipping)
    }

    /// Get the maximum peak across all channels.
    pub fn max_peak_db(&self) -> f64 {
        self.readings
            .iter()
            .map(|r| r.peak_hold_db)
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Get the average level across all channels.
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_level_db(&self) -> f64 {
        if self.readings.is_empty() {
            return f64::NEG_INFINITY;
        }
        let sum: f64 = self.readings.iter().map(|r| r.level_db).sum();
        sum / self.readings.len() as f64
    }
}

/// Meter bridge combining multiple meters.
///
/// Represents a physical or virtual meter bridge that aggregates
/// multiple meter slots for a unified monitoring view.
pub struct MeterBridge {
    /// Name of the meter bridge.
    name: String,
    /// Meter slots indexed by ID.
    slots: Vec<MeterSlot>,
    /// Map of labels to slot indices.
    label_map: HashMap<String, usize>,
    /// Sample rate.
    sample_rate: f64,
    /// Total samples processed.
    total_samples: u64,
    /// Maximum number of slots.
    max_slots: usize,
}

impl MeterBridge {
    /// Create a new meter bridge.
    pub fn new(name: &str, sample_rate: f64) -> Self {
        Self {
            name: name.to_string(),
            slots: Vec::new(),
            label_map: HashMap::new(),
            sample_rate,
            total_samples: 0,
            max_slots: 64,
        }
    }

    /// Get the bridge name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the sample rate.
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// Get the number of slots.
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Add a meter slot and return its index.
    pub fn add_slot(&mut self, config: MeterSlotConfig) -> Option<usize> {
        if self.slots.len() >= self.max_slots {
            return None;
        }
        let idx = self.slots.len();
        self.label_map.insert(config.label.clone(), idx);
        self.slots.push(MeterSlot::new(config));
        Some(idx)
    }

    /// Get a slot by index.
    pub fn slot(&self, index: usize) -> Option<&MeterSlot> {
        self.slots.get(index)
    }

    /// Get a mutable slot by index.
    pub fn slot_mut(&mut self, index: usize) -> Option<&mut MeterSlot> {
        self.slots.get_mut(index)
    }

    /// Find a slot by label.
    pub fn slot_by_label(&self, label: &str) -> Option<&MeterSlot> {
        self.label_map
            .get(label)
            .and_then(|&idx| self.slots.get(idx))
    }

    /// Find a mutable slot by label.
    pub fn slot_by_label_mut(&mut self, label: &str) -> Option<&mut MeterSlot> {
        if let Some(&idx) = self.label_map.get(label) {
            self.slots.get_mut(idx)
        } else {
            None
        }
    }

    /// Check if any slot is clipping.
    pub fn any_clipping(&self) -> bool {
        self.slots.iter().any(MeterSlot::is_clipping)
    }

    /// Get a summary of all slot statuses.
    pub fn status_summary(&self) -> BridgeStatusSummary {
        let mut clip_count = 0;
        let mut warning_count = 0;
        let mut active_count = 0;
        let mut idle_count = 0;
        let mut error_count = 0;

        for slot in &self.slots {
            match slot.overall_status {
                MeterStatus::Clip => clip_count += 1,
                MeterStatus::Warning => warning_count += 1,
                MeterStatus::Active => active_count += 1,
                MeterStatus::Idle => idle_count += 1,
                MeterStatus::Error => error_count += 1,
            }
        }

        BridgeStatusSummary {
            total_slots: self.slots.len(),
            clip_count,
            warning_count,
            active_count,
            idle_count,
            error_count,
        }
    }

    /// Reset all meter slots.
    pub fn reset_all(&mut self) {
        for slot in &mut self.slots {
            slot.reset();
        }
        self.total_samples = 0;
    }

    /// Get the duration of processed audio in seconds.
    #[allow(clippy::cast_precision_loss)]
    pub fn duration_seconds(&self) -> f64 {
        self.total_samples as f64 / self.sample_rate
    }

    /// Advance the sample counter.
    pub fn advance_samples(&mut self, count: u64) {
        self.total_samples += count;
    }

    /// Create a standard broadcast meter bridge (PPM L/R + loudness + phase).
    pub fn broadcast_standard(sample_rate: f64) -> Self {
        let mut bridge = Self::new("Broadcast Standard", sample_rate);
        bridge.add_slot(MeterSlotConfig::ppm("PPM L/R", 2));
        bridge.add_slot(MeterSlotConfig::loudness("Loudness", 2));
        bridge.add_slot(MeterSlotConfig {
            label: "Phase".to_string(),
            meter_type: MeterType::Phase,
            channels: 1,
            ..Default::default()
        });
        bridge
    }

    /// Create a music mastering meter bridge.
    pub fn mastering_bridge(sample_rate: f64) -> Self {
        let mut bridge = Self::new("Mastering", sample_rate);
        bridge.add_slot(MeterSlotConfig::ppm("Peak L/R", 2));
        bridge.add_slot(MeterSlotConfig::vu("VU L/R", 2));
        bridge.add_slot(MeterSlotConfig::loudness("LUFS", 2));
        bridge.add_slot(MeterSlotConfig {
            label: "Crest".to_string(),
            meter_type: MeterType::CrestFactor,
            channels: 2,
            ..Default::default()
        });
        bridge.add_slot(MeterSlotConfig {
            label: "Phase".to_string(),
            meter_type: MeterType::Phase,
            channels: 1,
            ..Default::default()
        });
        bridge
    }
}

/// Summary of bridge meter statuses.
#[derive(Clone, Debug)]
pub struct BridgeStatusSummary {
    /// Total number of meter slots.
    pub total_slots: usize,
    /// Number of slots in clip.
    pub clip_count: usize,
    /// Number of slots in warning.
    pub warning_count: usize,
    /// Number of active slots.
    pub active_count: usize,
    /// Number of idle slots.
    pub idle_count: usize,
    /// Number of slots with errors.
    pub error_count: usize,
}

impl BridgeStatusSummary {
    /// Check if there are any issues (clip or error).
    pub fn has_issues(&self) -> bool {
        self.clip_count > 0 || self.error_count > 0
    }

    /// Check if all slots are healthy (active or idle, no clip/error).
    pub fn is_healthy(&self) -> bool {
        self.clip_count == 0 && self.error_count == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meter_type_display() {
        assert_eq!(format!("{}", MeterType::Peak), "Peak");
        assert_eq!(format!("{}", MeterType::Loudness), "Loudness");
        assert_eq!(format!("{}", MeterType::CrestFactor), "Crest Factor");
    }

    #[test]
    fn test_meter_reading_new() {
        let r = MeterReading::new(-12.0, -6.0, 0);
        assert!((r.level_db - (-12.0)).abs() < f64::EPSILON);
        assert!((r.peak_hold_db - (-6.0)).abs() < f64::EPSILON);
        assert_eq!(r.channel, 0);
    }

    #[test]
    fn test_meter_reading_silent() {
        let r = MeterReading::silent(1);
        assert!(r.level_db.is_infinite());
        assert_eq!(r.status, MeterStatus::Idle);
        assert_eq!(r.channel, 1);
    }

    #[test]
    fn test_meter_reading_clipping() {
        let r = MeterReading::new(0.0, 0.5, 0);
        assert!(r.is_clipping());

        let r2 = MeterReading::new(-12.0, -6.0, 0);
        assert!(!r2.is_clipping());
    }

    #[test]
    fn test_meter_reading_dynamic_range() {
        let mut r = MeterReading::new(-12.0, -6.0, 0);
        r.min_db = -30.0;
        r.max_db = -3.0;
        assert!((r.dynamic_range_db() - 27.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_meter_slot_config_ppm() {
        let cfg = MeterSlotConfig::ppm("Main L/R", 2);
        assert_eq!(cfg.label, "Main L/R");
        assert_eq!(cfg.meter_type, MeterType::Peak);
        assert_eq!(cfg.channels, 2);
    }

    #[test]
    fn test_meter_slot_config_vu() {
        let cfg = MeterSlotConfig::vu("VU L/R", 2);
        assert_eq!(cfg.meter_type, MeterType::Vu);
        assert!((cfg.reference_level_db - (-18.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_meter_slot_creation() {
        let slot = MeterSlot::new(MeterSlotConfig::ppm("Test", 2));
        assert_eq!(slot.readings.len(), 2);
        assert_eq!(slot.overall_status, MeterStatus::Idle);
    }

    #[test]
    fn test_meter_slot_update_channel() {
        let mut slot = MeterSlot::new(MeterSlotConfig::ppm("Test", 2));
        slot.update_channel(0, -12.0, -6.0);
        assert!((slot.readings[0].level_db - (-12.0)).abs() < f64::EPSILON);
        assert_eq!(slot.readings[0].status, MeterStatus::Active);
        assert_eq!(slot.overall_status, MeterStatus::Active);
    }

    #[test]
    fn test_meter_slot_clipping() {
        let mut slot = MeterSlot::new(MeterSlotConfig::ppm("Test", 2));
        slot.update_channel(0, 0.0, 0.5);
        assert!(slot.is_clipping());
        assert_eq!(slot.overall_status, MeterStatus::Clip);
    }

    #[test]
    fn test_meter_slot_max_peak() {
        let mut slot = MeterSlot::new(MeterSlotConfig::ppm("Test", 2));
        slot.update_channel(0, -12.0, -6.0);
        slot.update_channel(1, -18.0, -3.0);
        assert!((slot.max_peak_db() - (-3.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_meter_slot_avg_level() {
        let mut slot = MeterSlot::new(MeterSlotConfig::ppm("Test", 2));
        slot.update_channel(0, -12.0, -6.0);
        slot.update_channel(1, -18.0, -6.0);
        let avg = slot.avg_level_db();
        assert!((avg - (-15.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_meter_slot_reset() {
        let mut slot = MeterSlot::new(MeterSlotConfig::ppm("Test", 2));
        slot.update_channel(0, -12.0, -6.0);
        slot.reset();
        assert_eq!(slot.overall_status, MeterStatus::Idle);
        assert!(slot.readings[0].level_db.is_infinite());
    }

    #[test]
    fn test_meter_bridge_creation() {
        let bridge = MeterBridge::new("Test Bridge", 48000.0);
        assert_eq!(bridge.name(), "Test Bridge");
        assert_eq!(bridge.slot_count(), 0);
    }

    #[test]
    fn test_meter_bridge_add_slot() {
        let mut bridge = MeterBridge::new("Test", 48000.0);
        let idx = bridge.add_slot(MeterSlotConfig::ppm("PPM", 2));
        assert_eq!(idx, Some(0));
        assert_eq!(bridge.slot_count(), 1);
    }

    #[test]
    fn test_meter_bridge_slot_by_label() {
        let mut bridge = MeterBridge::new("Test", 48000.0);
        bridge.add_slot(MeterSlotConfig::ppm("PPM L/R", 2));
        bridge.add_slot(MeterSlotConfig::loudness("LUFS", 2));

        let slot = bridge.slot_by_label("PPM L/R");
        assert!(slot.is_some());
        assert_eq!(
            slot.expect("test expectation failed").config.meter_type,
            MeterType::Peak
        );

        assert!(bridge.slot_by_label("NonExistent").is_none());
    }

    #[test]
    fn test_meter_bridge_status_summary() {
        let mut bridge = MeterBridge::new("Test", 48000.0);
        bridge.add_slot(MeterSlotConfig::ppm("PPM", 2));
        bridge.add_slot(MeterSlotConfig::loudness("LUFS", 2));
        let summary = bridge.status_summary();
        assert_eq!(summary.total_slots, 2);
        assert_eq!(summary.idle_count, 2);
        assert!(summary.is_healthy());
    }

    #[test]
    fn test_meter_bridge_broadcast_standard() {
        let bridge = MeterBridge::broadcast_standard(48000.0);
        assert_eq!(bridge.slot_count(), 3);
        assert_eq!(bridge.name(), "Broadcast Standard");
    }

    #[test]
    fn test_meter_bridge_mastering() {
        let bridge = MeterBridge::mastering_bridge(48000.0);
        assert_eq!(bridge.slot_count(), 5);
    }

    #[test]
    fn test_meter_bridge_reset_all() {
        let mut bridge = MeterBridge::broadcast_standard(48000.0);
        bridge.advance_samples(48000);
        bridge.reset_all();
        assert!((bridge.duration_seconds()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bridge_status_has_issues() {
        let summary = BridgeStatusSummary {
            total_slots: 3,
            clip_count: 1,
            warning_count: 0,
            active_count: 2,
            idle_count: 0,
            error_count: 0,
        };
        assert!(summary.has_issues());
        assert!(!summary.is_healthy());
    }
}
