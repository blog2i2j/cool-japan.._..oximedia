//! Stem management for creating and mixing multiple audio stems.

use crate::error::{AudioPostError, AudioPostResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Stem type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StemType {
    /// Dialogue stem
    Dialogue,
    /// Music stem
    Music,
    /// Effects stem
    Effects,
    /// Foley stem
    Foley,
    /// Ambience stem
    Ambience,
    /// Custom stem
    Custom,
}

impl StemType {
    /// Get all standard stem types
    #[must_use]
    pub fn standard_types() -> Vec<Self> {
        vec![
            Self::Dialogue,
            Self::Music,
            Self::Effects,
            Self::Foley,
            Self::Ambience,
        ]
    }

    /// Get stem type name
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Dialogue => "Dialogue",
            Self::Music => "Music",
            Self::Effects => "Effects",
            Self::Foley => "Foley",
            Self::Ambience => "Ambience",
            Self::Custom => "Custom",
        }
    }
}

/// Audio stem
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stem {
    /// Stem name
    pub name: String,
    /// Stem type
    pub stem_type: StemType,
    /// Audio file path
    pub audio_path: Option<PathBuf>,
    /// Level in dB
    pub level_db: f32,
    /// Pan (-1.0 to 1.0)
    pub pan: f32,
    /// Muted flag
    pub muted: bool,
    /// Solo flag
    pub solo: bool,
}

impl Stem {
    /// Create a new stem
    #[must_use]
    pub fn new(name: &str, stem_type: StemType) -> Self {
        Self {
            name: name.to_string(),
            stem_type,
            audio_path: None,
            level_db: 0.0,
            pan: 0.0,
            muted: false,
            solo: false,
        }
    }

    /// Set audio file path
    pub fn set_audio_path(&mut self, path: PathBuf) {
        self.audio_path = Some(path);
    }

    /// Set level in dB
    ///
    /// # Errors
    ///
    /// Returns an error if level is out of range
    pub fn set_level(&mut self, level_db: f32) -> AudioPostResult<()> {
        if !(-60.0..=12.0).contains(&level_db) {
            return Err(AudioPostError::InvalidGain(level_db));
        }
        self.level_db = level_db;
        Ok(())
    }

    /// Set pan
    ///
    /// # Errors
    ///
    /// Returns an error if pan is out of range
    pub fn set_pan(&mut self, pan: f32) -> AudioPostResult<()> {
        if !(-1.0..=1.0).contains(&pan) {
            return Err(AudioPostError::InvalidPan(pan));
        }
        self.pan = pan;
        Ok(())
    }
}

/// Stem manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StemManager {
    /// Stems indexed by name
    stems: HashMap<String, Stem>,
    /// Sample rate
    pub sample_rate: u32,
}

impl StemManager {
    /// Create a new stem manager
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate is invalid
    pub fn new(sample_rate: u32) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }

        Ok(Self {
            stems: HashMap::new(),
            sample_rate,
        })
    }

    /// Add a stem
    ///
    /// # Errors
    ///
    /// Returns an error if stem already exists
    pub fn add_stem(&mut self, stem: Stem) -> AudioPostResult<()> {
        if self.stems.contains_key(&stem.name) {
            return Err(AudioPostError::Generic(format!(
                "Stem '{}' already exists",
                stem.name
            )));
        }
        self.stems.insert(stem.name.clone(), stem);
        Ok(())
    }

    /// Get a stem
    ///
    /// # Errors
    ///
    /// Returns an error if stem is not found
    pub fn get_stem(&self, name: &str) -> AudioPostResult<&Stem> {
        self.stems
            .get(name)
            .ok_or_else(|| AudioPostError::StemNotFound(name.to_string()))
    }

    /// Get a mutable stem
    ///
    /// # Errors
    ///
    /// Returns an error if stem is not found
    pub fn get_stem_mut(&mut self, name: &str) -> AudioPostResult<&mut Stem> {
        self.stems
            .get_mut(name)
            .ok_or_else(|| AudioPostError::StemNotFound(name.to_string()))
    }

    /// Remove a stem
    ///
    /// # Errors
    ///
    /// Returns an error if stem is not found
    pub fn remove_stem(&mut self, name: &str) -> AudioPostResult<Stem> {
        self.stems
            .remove(name)
            .ok_or_else(|| AudioPostError::StemNotFound(name.to_string()))
    }

    /// Get all stems
    #[must_use]
    pub fn get_all_stems(&self) -> Vec<&Stem> {
        self.stems.values().collect()
    }

    /// Get stems by type
    #[must_use]
    pub fn get_stems_by_type(&self, stem_type: StemType) -> Vec<&Stem> {
        self.stems
            .values()
            .filter(|stem| stem.stem_type == stem_type)
            .collect()
    }

    /// Create standard stems
    pub fn create_standard_stems(&mut self) {
        for stem_type in StemType::standard_types() {
            let stem = Stem::new(stem_type.as_str(), stem_type);
            let _ = self.add_stem(stem);
        }
    }

    /// Get stem count
    #[must_use]
    pub fn stem_count(&self) -> usize {
        self.stems.len()
    }
}

/// Stem export settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StemExportSettings {
    /// Output directory
    pub output_dir: PathBuf,
    /// Sample rate
    pub sample_rate: u32,
    /// Bit depth
    pub bit_depth: u16,
    /// File format
    pub format: StemFormat,
    /// Normalize stems
    pub normalize: bool,
    /// Target level for normalization (dB)
    pub normalize_target: f32,
}

impl StemExportSettings {
    /// Create new export settings
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate or bit depth is invalid
    pub fn new(output_dir: PathBuf, sample_rate: u32, bit_depth: u16) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }
        if bit_depth != 16 && bit_depth != 24 && bit_depth != 32 {
            return Err(AudioPostError::Generic(
                "Bit depth must be 16, 24, or 32".to_string(),
            ));
        }

        Ok(Self {
            output_dir,
            sample_rate,
            bit_depth,
            format: StemFormat::Wav,
            normalize: false,
            normalize_target: -1.0,
        })
    }
}

/// Stem export format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StemFormat {
    /// WAV format
    Wav,
    /// FLAC format
    Flac,
    /// Broadcast Wave Format
    Bwf,
}

impl StemFormat {
    /// Get file extension
    #[must_use]
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Flac => "flac",
            Self::Bwf => "wav",
        }
    }
}

/// DCP/IMF stem package
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DcpStemPackage {
    /// Package name
    pub name: String,
    /// Dialogue stem path
    pub dialogue_stem: Option<PathBuf>,
    /// Music stem path
    pub music_stem: Option<PathBuf>,
    /// Effects stem path
    pub effects_stem: Option<PathBuf>,
    /// Sample rate
    pub sample_rate: u32,
    /// Bit depth
    pub bit_depth: u16,
}

impl DcpStemPackage {
    /// Create a new DCP stem package
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate or bit depth is invalid
    pub fn new(name: &str, sample_rate: u32, bit_depth: u16) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }
        if bit_depth != 24 {
            return Err(AudioPostError::Generic(
                "DCP requires 24-bit audio".to_string(),
            ));
        }

        Ok(Self {
            name: name.to_string(),
            dialogue_stem: None,
            music_stem: None,
            effects_stem: None,
            sample_rate,
            bit_depth,
        })
    }

    /// Validate package
    ///
    /// # Errors
    ///
    /// Returns an error if required stems are missing
    pub fn validate(&self) -> AudioPostResult<()> {
        if self.dialogue_stem.is_none() {
            return Err(AudioPostError::Generic(
                "Dialogue stem is required".to_string(),
            ));
        }
        if self.music_stem.is_none() {
            return Err(AudioPostError::Generic(
                "Music stem is required".to_string(),
            ));
        }
        if self.effects_stem.is_none() {
            return Err(AudioPostError::Generic(
                "Effects stem is required".to_string(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stem_creation() {
        let stem = Stem::new("Dialogue", StemType::Dialogue);
        assert_eq!(stem.name, "Dialogue");
        assert_eq!(stem.stem_type, StemType::Dialogue);
    }

    #[test]
    fn test_stem_set_level() {
        let mut stem = Stem::new("Music", StemType::Music);
        assert!(stem.set_level(-6.0).is_ok());
        assert_eq!(stem.level_db, -6.0);
    }

    #[test]
    fn test_stem_set_pan() {
        let mut stem = Stem::new("Effects", StemType::Effects);
        assert!(stem.set_pan(0.5).is_ok());
        assert_eq!(stem.pan, 0.5);
    }

    #[test]
    fn test_invalid_stem_level() {
        let mut stem = Stem::new("Music", StemType::Music);
        assert!(stem.set_level(-70.0).is_err());
        assert!(stem.set_level(20.0).is_err());
    }

    #[test]
    fn test_stem_manager_creation() {
        let manager = StemManager::new(48000).expect("failed to create");
        assert_eq!(manager.sample_rate, 48000);
        assert_eq!(manager.stem_count(), 0);
    }

    #[test]
    fn test_add_stem() {
        let mut manager = StemManager::new(48000).expect("failed to create");
        let stem = Stem::new("Dialogue", StemType::Dialogue);
        assert!(manager.add_stem(stem).is_ok());
        assert_eq!(manager.stem_count(), 1);
    }

    #[test]
    fn test_get_stem() {
        let mut manager = StemManager::new(48000).expect("failed to create");
        let stem = Stem::new("Dialogue", StemType::Dialogue);
        manager.add_stem(stem).expect("add_stem should succeed");
        assert!(manager.get_stem("Dialogue").is_ok());
    }

    #[test]
    fn test_remove_stem() {
        let mut manager = StemManager::new(48000).expect("failed to create");
        let stem = Stem::new("Dialogue", StemType::Dialogue);
        manager.add_stem(stem).expect("add_stem should succeed");
        assert!(manager.remove_stem("Dialogue").is_ok());
        assert_eq!(manager.stem_count(), 0);
    }

    #[test]
    fn test_create_standard_stems() {
        let mut manager = StemManager::new(48000).expect("failed to create");
        manager.create_standard_stems();
        assert_eq!(manager.stem_count(), 5);
    }

    #[test]
    fn test_get_stems_by_type() {
        let mut manager = StemManager::new(48000).expect("failed to create");
        manager.create_standard_stems();
        let dialogue_stems = manager.get_stems_by_type(StemType::Dialogue);
        assert_eq!(dialogue_stems.len(), 1);
    }

    #[test]
    fn test_stem_export_settings() {
        let settings =
            StemExportSettings::new(std::env::temp_dir(), 48000, 24).expect("failed to create");
        assert_eq!(settings.sample_rate, 48000);
        assert_eq!(settings.bit_depth, 24);
    }

    #[test]
    fn test_invalid_bit_depth() {
        assert!(StemExportSettings::new(std::env::temp_dir(), 48000, 8).is_err());
    }

    #[test]
    fn test_stem_format_extension() {
        assert_eq!(StemFormat::Wav.extension(), "wav");
        assert_eq!(StemFormat::Flac.extension(), "flac");
    }

    #[test]
    fn test_dcp_stem_package() {
        let package = DcpStemPackage::new("Test Package", 48000, 24).expect("failed to create");
        assert_eq!(package.name, "Test Package");
    }

    #[test]
    fn test_dcp_invalid_bit_depth() {
        assert!(DcpStemPackage::new("Test", 48000, 16).is_err());
    }

    #[test]
    fn test_dcp_validation() {
        let package = DcpStemPackage::new("Test", 48000, 24).expect("failed to create");
        assert!(package.validate().is_err());
    }

    #[test]
    fn test_stem_type_as_str() {
        assert_eq!(StemType::Dialogue.as_str(), "Dialogue");
        assert_eq!(StemType::Music.as_str(), "Music");
    }
}
