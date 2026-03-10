//! Meter type enumeration and per-type configuration sets.
//!
//! This module defines [`MeterType`], [`MeterTypeConfig`], and [`MeterConfigSet`]
//! which allow callers to select and validate configuration for multiple
//! concurrent meter types in a single processing chain.

#![allow(dead_code)]

/// Identifies the functional type of a metering unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MeterType {
    /// ITU-R BS.1770 / EBU R128 integrated loudness (LUFS).
    Loudness,
    /// True-peak detection with oversampling.
    TruePeak,
    /// Loudness range (LRA) measurement.
    LoudnessRange,
    /// Peak programme meter (PPM) with ballistics.
    Ppm,
    /// VU meter with analogue-style ballistics.
    Vu,
    /// RMS power level meter.
    Rms,
    /// Stereo phase correlation meter.
    Correlation,
    /// K-system loudness (K-12, K-14, K-20).
    KSystem,
}

impl MeterType {
    /// Return a short human-readable name for this meter type.
    pub fn name(self) -> &'static str {
        match self {
            Self::Loudness => "Loudness",
            Self::TruePeak => "True Peak",
            Self::LoudnessRange => "Loudness Range",
            Self::Ppm => "PPM",
            Self::Vu => "VU",
            Self::Rms => "RMS",
            Self::Correlation => "Correlation",
            Self::KSystem => "K-System",
        }
    }

    /// Return `true` if this meter type requires K-weighted filtering.
    pub fn needs_k_weighting(self) -> bool {
        matches!(self, Self::Loudness | Self::LoudnessRange | Self::KSystem)
    }

    /// Return `true` if this meter type requires oversampling.
    pub fn needs_oversampling(self) -> bool {
        matches!(self, Self::TruePeak)
    }

    /// Return all available meter types.
    pub fn all() -> &'static [MeterType] {
        &[
            Self::Loudness,
            Self::TruePeak,
            Self::LoudnessRange,
            Self::Ppm,
            Self::Vu,
            Self::Rms,
            Self::Correlation,
            Self::KSystem,
        ]
    }
}

impl std::fmt::Display for MeterType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Configuration parameters for a single meter type.
#[derive(Debug, Clone)]
pub struct MeterTypeConfig {
    /// Which meter type this config applies to.
    pub meter_type: MeterType,
    /// Whether this meter is active (enabled).
    pub enabled: bool,
    /// Integration window in milliseconds (0 = use default for type).
    pub window_ms: f64,
    /// Attack time constant in milliseconds.
    pub attack_ms: f64,
    /// Release time constant in milliseconds.
    pub release_ms: f64,
    /// Peak hold duration in seconds (0 = no hold).
    pub peak_hold_s: f64,
}

impl MeterTypeConfig {
    /// Create a config with the defaults appropriate for `meter_type`.
    pub fn default_for(meter_type: MeterType) -> Self {
        let (window_ms, attack_ms, release_ms, peak_hold_s) = match meter_type {
            MeterType::Loudness => (400.0, 0.0, 0.0, 0.0),
            MeterType::TruePeak => (0.0, 0.0, 0.0, 2.0),
            MeterType::LoudnessRange => (3_000.0, 0.0, 0.0, 0.0),
            MeterType::Ppm => (0.0, 10.0, 1_500.0, 2.0),
            MeterType::Vu => (300.0, 300.0, 300.0, 0.0),
            MeterType::Rms => (300.0, 0.0, 0.0, 0.0),
            MeterType::Correlation => (400.0, 0.0, 0.0, 0.0),
            MeterType::KSystem => (300.0, 0.0, 0.0, 2.0),
        };
        Self {
            meter_type,
            enabled: true,
            window_ms,
            attack_ms,
            release_ms,
            peak_hold_s,
        }
    }

    /// Validate the configuration values.
    ///
    /// Returns `Err` with a description if any value is out of range.
    pub fn validate(&self) -> Result<(), String> {
        if self.window_ms < 0.0 {
            return Err(format!(
                "{}: window_ms must be >= 0, got {}",
                self.meter_type, self.window_ms
            ));
        }
        if self.attack_ms < 0.0 {
            return Err(format!(
                "{}: attack_ms must be >= 0, got {}",
                self.meter_type, self.attack_ms
            ));
        }
        if self.release_ms < 0.0 {
            return Err(format!(
                "{}: release_ms must be >= 0, got {}",
                self.meter_type, self.release_ms
            ));
        }
        if self.peak_hold_s < 0.0 {
            return Err(format!(
                "{}: peak_hold_s must be >= 0, got {}",
                self.meter_type, self.peak_hold_s
            ));
        }
        Ok(())
    }
}

/// A set of [`MeterTypeConfig`] entries keyed by [`MeterType`].
///
/// Provides a convenient way to manage configuration for several meter types
/// at once and to look up or update individual entries.
#[derive(Debug, Clone, Default)]
pub struct MeterConfigSet {
    entries: Vec<MeterTypeConfig>,
}

impl MeterConfigSet {
    /// Create an empty set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a set populated with the default config for every meter type.
    pub fn all_defaults() -> Self {
        let entries = MeterType::all()
            .iter()
            .map(|&t| MeterTypeConfig::default_for(t))
            .collect();
        Self { entries }
    }

    /// Add or replace the config for `config.meter_type`.
    pub fn insert(&mut self, config: MeterTypeConfig) {
        if let Some(existing) = self
            .entries
            .iter_mut()
            .find(|e| e.meter_type == config.meter_type)
        {
            *existing = config;
        } else {
            self.entries.push(config);
        }
    }

    /// Return the config for `meter_type`, if present.
    pub fn get(&self, meter_type: MeterType) -> Option<&MeterTypeConfig> {
        self.entries.iter().find(|e| e.meter_type == meter_type)
    }

    /// Return `true` if `meter_type` is present and enabled.
    pub fn is_enabled(&self, meter_type: MeterType) -> bool {
        self.get(meter_type).is_some_and(|c| c.enabled)
    }

    /// Disable `meter_type` (no-op if not present).
    pub fn disable(&mut self, meter_type: MeterType) {
        if let Some(entry) = self.entries.iter_mut().find(|e| e.meter_type == meter_type) {
            entry.enabled = false;
        }
    }

    /// Enable `meter_type` (no-op if not present).
    pub fn enable(&mut self, meter_type: MeterType) {
        if let Some(entry) = self.entries.iter_mut().find(|e| e.meter_type == meter_type) {
            entry.enabled = true;
        }
    }

    /// Validate all entries; returns a list of error strings.
    pub fn validate_all(&self) -> Vec<String> {
        self.entries
            .iter()
            .filter_map(|e| e.validate().err())
            .collect()
    }

    /// Number of entries in the set.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` if the set contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over all entries.
    pub fn iter(&self) -> impl Iterator<Item = &MeterTypeConfig> {
        self.entries.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meter_type_name_loudness() {
        assert_eq!(MeterType::Loudness.name(), "Loudness");
    }

    #[test]
    fn meter_type_name_true_peak() {
        assert_eq!(MeterType::TruePeak.name(), "True Peak");
    }

    #[test]
    fn meter_type_needs_k_weighting() {
        assert!(MeterType::Loudness.needs_k_weighting());
        assert!(MeterType::KSystem.needs_k_weighting());
        assert!(!MeterType::Vu.needs_k_weighting());
        assert!(!MeterType::Ppm.needs_k_weighting());
    }

    #[test]
    fn meter_type_needs_oversampling() {
        assert!(MeterType::TruePeak.needs_oversampling());
        assert!(!MeterType::Loudness.needs_oversampling());
    }

    #[test]
    fn meter_type_all_has_eight_variants() {
        assert_eq!(MeterType::all().len(), 8);
    }

    #[test]
    fn meter_type_display() {
        assert_eq!(format!("{}", MeterType::Rms), "RMS");
        assert_eq!(format!("{}", MeterType::Correlation), "Correlation");
    }

    #[test]
    fn config_default_vu_window() {
        let cfg = MeterTypeConfig::default_for(MeterType::Vu);
        assert_eq!(cfg.meter_type, MeterType::Vu);
        assert!(cfg.window_ms > 0.0);
        assert!(cfg.enabled);
    }

    #[test]
    fn config_validate_ok() {
        let cfg = MeterTypeConfig::default_for(MeterType::Ppm);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn config_validate_negative_window() {
        let mut cfg = MeterTypeConfig::default_for(MeterType::Rms);
        cfg.window_ms = -1.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validate_negative_attack() {
        let mut cfg = MeterTypeConfig::default_for(MeterType::Vu);
        cfg.attack_ms = -5.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_set_all_defaults_count() {
        let set = MeterConfigSet::all_defaults();
        assert_eq!(set.len(), MeterType::all().len());
    }

    #[test]
    fn config_set_insert_and_get() {
        let mut set = MeterConfigSet::new();
        let cfg = MeterTypeConfig::default_for(MeterType::Loudness);
        set.insert(cfg);
        assert!(set.get(MeterType::Loudness).is_some());
        assert!(set.get(MeterType::Vu).is_none());
    }

    #[test]
    fn config_set_is_enabled() {
        let set = MeterConfigSet::all_defaults();
        assert!(set.is_enabled(MeterType::TruePeak));
    }

    #[test]
    fn config_set_disable_enable_roundtrip() {
        let mut set = MeterConfigSet::all_defaults();
        set.disable(MeterType::LoudnessRange);
        assert!(!set.is_enabled(MeterType::LoudnessRange));
        set.enable(MeterType::LoudnessRange);
        assert!(set.is_enabled(MeterType::LoudnessRange));
    }

    #[test]
    fn config_set_validate_all_ok() {
        let set = MeterConfigSet::all_defaults();
        assert!(set.validate_all().is_empty());
    }

    #[test]
    fn config_set_is_empty() {
        let set = MeterConfigSet::new();
        assert!(set.is_empty());
    }

    #[test]
    fn config_set_replace_existing() {
        let mut set = MeterConfigSet::all_defaults();
        let initial_len = set.len();
        let mut new_cfg = MeterTypeConfig::default_for(MeterType::Vu);
        new_cfg.window_ms = 500.0;
        set.insert(new_cfg);
        // Length should not grow — it replaces
        assert_eq!(set.len(), initial_len);
        assert_eq!(
            set.get(MeterType::Vu)
                .expect("get should succeed")
                .window_ms,
            500.0
        );
    }
}
