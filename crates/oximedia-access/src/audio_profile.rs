#![allow(dead_code)]
//! User audio accessibility profiles and preferences.
//!
//! Manages per-user audio settings that adapt media playback to
//! individual hearing needs, including frequency adjustments,
//! dialogue enhancement, and dynamic range compression.

use std::collections::HashMap;
use std::fmt;

/// Hearing loss frequency band.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FrequencyBand {
    /// Low frequencies (below 500 Hz).
    Low,
    /// Low-mid frequencies (500 - 1000 Hz).
    LowMid,
    /// Mid frequencies (1000 - 2000 Hz).
    Mid,
    /// High-mid frequencies (2000 - 4000 Hz).
    HighMid,
    /// High frequencies (above 4000 Hz).
    High,
}

impl fmt::Display for FrequencyBand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Low => "Low (<500 Hz)",
            Self::LowMid => "Low-Mid (500-1000 Hz)",
            Self::Mid => "Mid (1000-2000 Hz)",
            Self::HighMid => "High-Mid (2000-4000 Hz)",
            Self::High => "High (>4000 Hz)",
        };
        write!(f, "{label}")
    }
}

/// Severity of hearing difficulty in a frequency band.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HearingSeverity {
    /// No difficulty.
    None,
    /// Mild difficulty.
    Mild,
    /// Moderate difficulty.
    Moderate,
    /// Severe difficulty.
    Severe,
    /// Profound difficulty.
    Profound,
}

/// A boost amount in decibels for a frequency band.
#[derive(Debug, Clone, Copy)]
pub struct BandBoost {
    /// The frequency band.
    pub band: FrequencyBand,
    /// Boost in dB (can be negative for attenuation).
    pub boost_db: f64,
}

/// Dynamic range compression settings.
#[derive(Debug, Clone, Copy)]
pub struct CompressionSettings {
    /// Threshold in dB below which compression starts.
    pub threshold_db: f64,
    /// Compression ratio (e.g., 2.0 means 2:1).
    pub ratio: f64,
    /// Attack time in milliseconds.
    pub attack_ms: f64,
    /// Release time in milliseconds.
    pub release_ms: f64,
    /// Makeup gain in dB.
    pub makeup_gain_db: f64,
}

impl Default for CompressionSettings {
    fn default() -> Self {
        Self {
            threshold_db: -20.0,
            ratio: 3.0,
            attack_ms: 10.0,
            release_ms: 100.0,
            makeup_gain_db: 6.0,
        }
    }
}

impl CompressionSettings {
    /// Create gentle compression for mild hearing difficulty.
    #[must_use]
    pub fn gentle() -> Self {
        Self {
            threshold_db: -24.0,
            ratio: 2.0,
            attack_ms: 15.0,
            release_ms: 150.0,
            makeup_gain_db: 4.0,
        }
    }

    /// Create moderate compression.
    #[must_use]
    pub fn moderate() -> Self {
        Self::default()
    }

    /// Create aggressive compression for severe hearing difficulty.
    #[must_use]
    pub fn aggressive() -> Self {
        Self {
            threshold_db: -15.0,
            ratio: 5.0,
            attack_ms: 5.0,
            release_ms: 60.0,
            makeup_gain_db: 10.0,
        }
    }

    /// Compute the output level for a given input level.
    #[must_use]
    pub fn compute_output(&self, input_db: f64) -> f64 {
        if input_db <= self.threshold_db {
            input_db + self.makeup_gain_db
        } else {
            let over = input_db - self.threshold_db;
            let compressed = self.threshold_db + over / self.ratio;
            compressed + self.makeup_gain_db
        }
    }
}

/// Dialogue enhancement configuration.
#[derive(Debug, Clone, Copy)]
pub struct DialogueEnhancement {
    /// Boost for dialogue clarity in dB.
    pub clarity_boost_db: f64,
    /// Whether to reduce background music during dialogue.
    pub reduce_background: bool,
    /// Background reduction amount in dB (negative).
    pub background_reduction_db: f64,
    /// Center channel boost for surround mixes.
    pub center_boost_db: f64,
}

impl Default for DialogueEnhancement {
    fn default() -> Self {
        Self {
            clarity_boost_db: 3.0,
            reduce_background: true,
            background_reduction_db: -6.0,
            center_boost_db: 3.0,
        }
    }
}

/// A complete audio accessibility profile for a user.
#[derive(Debug, Clone)]
pub struct AudioAccessProfile {
    /// Profile name.
    pub name: String,
    /// Per-band hearing sensitivity adjustments.
    pub band_adjustments: HashMap<FrequencyBand, BandBoost>,
    /// Dynamic range compression settings.
    pub compression: Option<CompressionSettings>,
    /// Dialogue enhancement settings.
    pub dialogue_enhancement: Option<DialogueEnhancement>,
    /// Overall volume boost in dB.
    pub volume_boost_db: f64,
    /// Whether mono downmix is preferred.
    pub prefer_mono: bool,
    /// Whether to enable audio description track by default.
    pub auto_audio_description: bool,
    /// Whether the profile is active.
    pub enabled: bool,
}

impl Default for AudioAccessProfile {
    fn default() -> Self {
        Self {
            name: "Default".to_string(),
            band_adjustments: HashMap::new(),
            compression: None,
            dialogue_enhancement: None,
            volume_boost_db: 0.0,
            prefer_mono: false,
            auto_audio_description: false,
            enabled: true,
        }
    }
}

impl AudioAccessProfile {
    /// Create a new profile with a given name.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    /// Create a preset for mild hearing loss (high-frequency emphasis).
    #[must_use]
    pub fn mild_hearing_loss() -> Self {
        let mut profile = Self::new("Mild Hearing Loss");
        profile.set_band_boost(FrequencyBand::HighMid, 4.0);
        profile.set_band_boost(FrequencyBand::High, 6.0);
        profile.compression = Some(CompressionSettings::gentle());
        profile.dialogue_enhancement = Some(DialogueEnhancement::default());
        profile
    }

    /// Create a preset for moderate hearing loss.
    #[must_use]
    pub fn moderate_hearing_loss() -> Self {
        let mut profile = Self::new("Moderate Hearing Loss");
        profile.set_band_boost(FrequencyBand::Mid, 3.0);
        profile.set_band_boost(FrequencyBand::HighMid, 6.0);
        profile.set_band_boost(FrequencyBand::High, 9.0);
        profile.compression = Some(CompressionSettings::moderate());
        profile.dialogue_enhancement = Some(DialogueEnhancement {
            clarity_boost_db: 6.0,
            reduce_background: true,
            background_reduction_db: -9.0,
            center_boost_db: 6.0,
        });
        profile.volume_boost_db = 3.0;
        profile
    }

    /// Create a preset for severe hearing loss.
    #[must_use]
    pub fn severe_hearing_loss() -> Self {
        let mut profile = Self::new("Severe Hearing Loss");
        profile.set_band_boost(FrequencyBand::LowMid, 3.0);
        profile.set_band_boost(FrequencyBand::Mid, 6.0);
        profile.set_band_boost(FrequencyBand::HighMid, 10.0);
        profile.set_band_boost(FrequencyBand::High, 12.0);
        profile.compression = Some(CompressionSettings::aggressive());
        profile.dialogue_enhancement = Some(DialogueEnhancement {
            clarity_boost_db: 10.0,
            reduce_background: true,
            background_reduction_db: -12.0,
            center_boost_db: 9.0,
        });
        profile.volume_boost_db = 6.0;
        profile.prefer_mono = true;
        profile.auto_audio_description = true;
        profile
    }

    /// Set a boost for a frequency band.
    pub fn set_band_boost(&mut self, band: FrequencyBand, boost_db: f64) {
        self.band_adjustments.insert(
            band,
            BandBoost { band, boost_db },
        );
    }

    /// Get the boost for a specific band (0.0 if not set).
    #[must_use]
    pub fn get_band_boost(&self, band: FrequencyBand) -> f64 {
        self.band_adjustments
            .get(&band)
            .map_or(0.0, |b| b.boost_db)
    }

    /// Get the total number of band adjustments.
    #[must_use]
    pub fn adjustment_count(&self) -> usize {
        self.band_adjustments.len()
    }

    /// Check if the profile has any active processing.
    #[must_use]
    pub fn has_processing(&self) -> bool {
        !self.band_adjustments.is_empty()
            || self.compression.is_some()
            || self.dialogue_enhancement.is_some()
            || self.volume_boost_db.abs() > f64::EPSILON
    }

    /// Compute the total boost across all bands.
    #[must_use]
    pub fn total_boost(&self) -> f64 {
        self.band_adjustments.values().map(|b| b.boost_db).sum::<f64>() + self.volume_boost_db
    }
}

/// Manager for multiple audio profiles.
#[derive(Debug)]
pub struct AudioProfileManager {
    /// All registered profiles.
    profiles: Vec<AudioAccessProfile>,
    /// Index of the active profile.
    active_index: Option<usize>,
}

impl Default for AudioProfileManager {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioProfileManager {
    /// Create a new empty profile manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            profiles: Vec::new(),
            active_index: None,
        }
    }

    /// Add a profile and return its index.
    pub fn add_profile(&mut self, profile: AudioAccessProfile) -> usize {
        let idx = self.profiles.len();
        self.profiles.push(profile);
        idx
    }

    /// Set the active profile by index.
    pub fn set_active(&mut self, index: usize) -> bool {
        if index < self.profiles.len() {
            self.active_index = Some(index);
            true
        } else {
            false
        }
    }

    /// Get the active profile.
    #[must_use]
    pub fn active_profile(&self) -> Option<&AudioAccessProfile> {
        self.active_index.and_then(|i| self.profiles.get(i))
    }

    /// Get number of profiles.
    #[must_use]
    pub fn profile_count(&self) -> usize {
        self.profiles.len()
    }

    /// Get a profile by name.
    #[must_use]
    pub fn find_by_name(&self, name: &str) -> Option<&AudioAccessProfile> {
        self.profiles.iter().find(|p| p.name == name)
    }

    /// Remove a profile by index.
    pub fn remove_profile(&mut self, index: usize) -> bool {
        if index < self.profiles.len() {
            self.profiles.remove(index);
            // Adjust active index
            match self.active_index {
                Some(ai) if ai == index => self.active_index = None,
                Some(ai) if ai > index => self.active_index = Some(ai - 1),
                _ => {}
            }
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_profile() {
        let profile = AudioAccessProfile::default();
        assert_eq!(profile.name, "Default");
        assert!(!profile.has_processing());
        assert!((profile.volume_boost_db - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mild_hearing_loss_preset() {
        let profile = AudioAccessProfile::mild_hearing_loss();
        assert_eq!(profile.name, "Mild Hearing Loss");
        assert!(profile.has_processing());
        assert!(profile.get_band_boost(FrequencyBand::High) > 0.0);
        assert!(profile.compression.is_some());
    }

    #[test]
    fn test_moderate_hearing_loss_preset() {
        let profile = AudioAccessProfile::moderate_hearing_loss();
        assert!(profile.get_band_boost(FrequencyBand::HighMid) > 0.0);
        assert!(profile.dialogue_enhancement.is_some());
        assert!(profile.volume_boost_db > 0.0);
    }

    #[test]
    fn test_severe_hearing_loss_preset() {
        let profile = AudioAccessProfile::severe_hearing_loss();
        assert!(profile.prefer_mono);
        assert!(profile.auto_audio_description);
        assert!(profile.get_band_boost(FrequencyBand::High) >= 12.0);
    }

    #[test]
    fn test_set_and_get_band_boost() {
        let mut profile = AudioAccessProfile::new("Custom");
        profile.set_band_boost(FrequencyBand::Mid, 5.0);
        assert!((profile.get_band_boost(FrequencyBand::Mid) - 5.0).abs() < f64::EPSILON);
        assert!((profile.get_band_boost(FrequencyBand::Low) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_adjustment_count() {
        let mut profile = AudioAccessProfile::new("Test");
        assert_eq!(profile.adjustment_count(), 0);
        profile.set_band_boost(FrequencyBand::Low, 2.0);
        profile.set_band_boost(FrequencyBand::High, 4.0);
        assert_eq!(profile.adjustment_count(), 2);
    }

    #[test]
    fn test_total_boost() {
        let mut profile = AudioAccessProfile::new("Test");
        profile.set_band_boost(FrequencyBand::Low, 2.0);
        profile.set_band_boost(FrequencyBand::High, 4.0);
        profile.volume_boost_db = 3.0;
        assert!((profile.total_boost() - 9.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compression_compute_output_below_threshold() {
        let comp = CompressionSettings::default();
        let output = comp.compute_output(-30.0);
        // Below threshold: input + makeup gain
        assert!((output - (-30.0 + comp.makeup_gain_db)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compression_compute_output_above_threshold() {
        let comp = CompressionSettings::default();
        let output = comp.compute_output(-10.0);
        // Above threshold: compressed
        let expected = comp.threshold_db + (-10.0 - comp.threshold_db) / comp.ratio + comp.makeup_gain_db;
        assert!((output - expected).abs() < 1e-10);
    }

    #[test]
    fn test_profile_manager_add_and_activate() {
        let mut mgr = AudioProfileManager::new();
        let idx = mgr.add_profile(AudioAccessProfile::new("Test"));
        assert_eq!(mgr.profile_count(), 1);
        assert!(mgr.active_profile().is_none());
        assert!(mgr.set_active(idx));
        assert!(mgr.active_profile().is_some());
        assert_eq!(mgr.active_profile().expect("active_profile should succeed").name, "Test");
    }

    #[test]
    fn test_profile_manager_find_by_name() {
        let mut mgr = AudioProfileManager::new();
        mgr.add_profile(AudioAccessProfile::mild_hearing_loss());
        mgr.add_profile(AudioAccessProfile::severe_hearing_loss());
        let found = mgr.find_by_name("Severe Hearing Loss");
        assert!(found.is_some());
        assert!(mgr.find_by_name("Nonexistent").is_none());
    }

    #[test]
    fn test_profile_manager_remove() {
        let mut mgr = AudioProfileManager::new();
        mgr.add_profile(AudioAccessProfile::new("A"));
        mgr.add_profile(AudioAccessProfile::new("B"));
        mgr.set_active(1);
        assert!(mgr.remove_profile(0));
        assert_eq!(mgr.profile_count(), 1);
        // Active adjusted
        assert_eq!(mgr.active_profile().expect("active_profile should succeed").name, "B");
    }

    #[test]
    fn test_frequency_band_display() {
        assert_eq!(format!("{}", FrequencyBand::Low), "Low (<500 Hz)");
        assert_eq!(format!("{}", FrequencyBand::High), "High (>4000 Hz)");
    }
}
