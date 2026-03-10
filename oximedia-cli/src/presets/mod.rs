//! Transcoding preset system for OxiMedia.
//!
//! Provides a comprehensive preset management system with:
//! - Built-in presets for common use cases
//! - Custom user presets via TOML files
//! - Preset validation and listing
//! - Organized by categories (Web, Devices, Quality, Archival, Streaming)
//!
//! # Examples
//!
//! ```no_run
//! use oximedia_cli::presets::{PresetManager, PresetCategory};
//!
//! let manager = PresetManager::new();
//! let preset = manager.get_preset("youtube-1080p")?;
//! println!("Video codec: {}", preset.video.codec);
//! ```

pub mod builtin;
pub mod custom;
pub mod device;
pub mod streaming;
pub mod validate;
pub mod web;

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Video codec configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VideoConfig {
    /// Video codec (av1, vp9, vp8, theora)
    pub codec: String,

    /// Target bitrate (e.g., "5M", "2.5M", "500k")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bitrate: Option<String>,

    /// Constant Rate Factor (quality-based encoding)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub crf: Option<u32>,

    /// Video width in pixels
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,

    /// Video height in pixels
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,

    /// Frame rate (e.g., 30, 60, 23.976)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fps: Option<f64>,

    /// Encoder preset (ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preset: Option<String>,

    /// Pixel format (yuv420p, yuv444p, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pixel_format: Option<String>,

    /// Enable two-pass encoding
    #[serde(default)]
    pub two_pass: bool,

    /// Maximum bitrate for VBV (Variable Bitrate Video)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_bitrate: Option<String>,

    /// Minimum bitrate for VBV
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_bitrate: Option<String>,

    /// Buffer size for VBV
    #[serde(skip_serializing_if = "Option::is_none")]
    pub buffer_size: Option<String>,

    /// Keyframe interval (GOP size)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keyframe_interval: Option<u32>,

    /// Minimum keyframe interval
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_keyframe_interval: Option<u32>,

    /// Aspect ratio (e.g., "16:9", "4:3")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aspect_ratio: Option<String>,
}

/// Audio codec configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AudioConfig {
    /// Audio codec (opus, vorbis, flac, pcm)
    pub codec: String,

    /// Target bitrate (e.g., "128k", "192k", "256k")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bitrate: Option<String>,

    /// Sample rate in Hz (e.g., 48000, 44100)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<u32>,

    /// Number of audio channels
    #[serde(skip_serializing_if = "Option::is_none")]
    pub channels: Option<u32>,

    /// Audio quality (codec-specific)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<f64>,

    /// Compression level (for FLAC)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compression_level: Option<u32>,
}

/// Filter chain configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FilterConfig {
    /// Video filters (e.g., "scale=1920:1080,fps=30")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub video_filters: Option<Vec<String>>,

    /// Audio filters (e.g., "volume=0.5")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_filters: Option<Vec<String>>,

    /// Deinterlacing method
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deinterlace: Option<String>,

    /// Denoise filter
    #[serde(skip_serializing_if = "Option::is_none")]
    pub denoise: Option<String>,
}

/// Complete transcoding preset.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Preset {
    /// Preset name (unique identifier)
    pub name: String,

    /// Human-readable description
    pub description: String,

    /// Preset category
    pub category: PresetCategory,

    /// Video configuration
    pub video: VideoConfig,

    /// Audio configuration
    pub audio: AudioConfig,

    /// Container format (webm, mkv, ogg, flac, wav)
    pub container: String,

    /// Optional filter chain
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filters: Option<FilterConfig>,

    /// Whether this is a built-in preset (non-modifiable)
    #[serde(default)]
    pub builtin: bool,

    /// Preset tags for searching/filtering
    #[serde(default)]
    pub tags: Vec<String>,
}

/// Preset category for organization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PresetCategory {
    /// Web platform presets (YouTube, Vimeo, Social Media)
    Web,

    /// Device-specific presets (iPhone, Android, TV)
    Device,

    /// Quality tier presets (4K, 1080p, 720p, 480p)
    Quality,

    /// Archival presets (lossless, high quality)
    Archival,

    /// Streaming presets (HLS/DASH variants)
    Streaming,

    /// Custom user presets
    Custom,
}

impl PresetCategory {
    /// Get category name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Web => "Web",
            Self::Device => "Device",
            Self::Quality => "Quality",
            Self::Archival => "Archival",
            Self::Streaming => "Streaming",
            Self::Custom => "Custom",
        }
    }

    /// Get category description.
    pub fn description(&self) -> &'static str {
        match self {
            Self::Web => "Presets optimized for web platforms (YouTube, Vimeo, social media)",
            Self::Device => "Presets optimized for specific devices (iPhone, Android, TV)",
            Self::Quality => "Quality tier presets (4K, 1080p, 720p, 480p)",
            Self::Archival => "Archival presets (lossless, high quality preservation)",
            Self::Streaming => "Streaming presets (HLS/DASH adaptive bitrate variants)",
            Self::Custom => "User-defined custom presets",
        }
    }

    /// Parse category from string.
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "web" => Ok(Self::Web),
            "device" => Ok(Self::Device),
            "quality" => Ok(Self::Quality),
            "archival" => Ok(Self::Archival),
            "streaming" => Ok(Self::Streaming),
            "custom" => Ok(Self::Custom),
            _ => Err(anyhow!("Unknown preset category: {}", s)),
        }
    }
}

/// Preset manager for loading and managing presets.
pub struct PresetManager {
    /// All loaded presets by name
    presets: HashMap<String, Preset>,

    /// Custom preset directory
    custom_dir: Option<PathBuf>,
}

impl PresetManager {
    /// Create a new preset manager with built-in presets.
    pub fn new() -> Self {
        let mut manager = Self {
            presets: HashMap::new(),
            custom_dir: None,
        };

        // Load built-in presets
        manager.load_builtin_presets();

        manager
    }

    /// Create a preset manager with custom preset directory.
    pub fn with_custom_dir<P: AsRef<Path>>(custom_dir: P) -> Result<Self> {
        let mut manager = Self::new();
        manager.custom_dir = Some(custom_dir.as_ref().to_path_buf());
        manager.load_custom_presets()?;
        Ok(manager)
    }

    /// Load all built-in presets.
    fn load_builtin_presets(&mut self) {
        // Load web platform presets
        for preset in web::get_web_presets() {
            self.presets.insert(preset.name.clone(), preset);
        }

        // Load device presets
        for preset in device::get_device_presets() {
            self.presets.insert(preset.name.clone(), preset);
        }

        // Load streaming presets
        for preset in streaming::get_streaming_presets() {
            self.presets.insert(preset.name.clone(), preset);
        }

        // Load quality and archival presets
        for preset in builtin::get_quality_presets() {
            self.presets.insert(preset.name.clone(), preset);
        }

        for preset in builtin::get_archival_presets() {
            self.presets.insert(preset.name.clone(), preset);
        }
    }

    /// Load custom presets from directory.
    fn load_custom_presets(&mut self) -> Result<()> {
        if let Some(ref dir) = self.custom_dir {
            if dir.exists() && dir.is_dir() {
                for entry in std::fs::read_dir(dir)? {
                    let entry = entry?;
                    let path = entry.path();

                    if path.extension().and_then(|s| s.to_str()) == Some("toml") {
                        match custom::load_preset_from_file(&path) {
                            Ok(preset) => {
                                // Don't allow overriding built-in presets
                                if let Some(existing) = self.presets.get(&preset.name) {
                                    if existing.builtin {
                                        eprintln!(
                                            "Warning: Cannot override built-in preset '{}' with custom preset",
                                            preset.name
                                        );
                                        continue;
                                    }
                                }
                                self.presets.insert(preset.name.clone(), preset);
                            }
                            Err(e) => {
                                eprintln!(
                                    "Warning: Failed to load preset from {}: {}",
                                    path.display(),
                                    e
                                );
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Get a preset by name.
    pub fn get_preset(&self, name: &str) -> Result<&Preset> {
        self.presets
            .get(name)
            .ok_or_else(|| anyhow!("Preset '{}' not found", name))
    }

    /// List all presets.
    pub fn list_presets(&self) -> Vec<&Preset> {
        let mut presets: Vec<_> = self.presets.values().collect();
        presets.sort_by(|a, b| a.name.cmp(&b.name));
        presets
    }

    /// List presets by category.
    pub fn list_presets_by_category(&self, category: PresetCategory) -> Vec<&Preset> {
        let mut presets: Vec<_> = self
            .presets
            .values()
            .filter(|p| p.category == category)
            .collect();
        presets.sort_by(|a, b| a.name.cmp(&b.name));
        presets
    }

    /// Get all available preset names.
    pub fn preset_names(&self) -> Vec<String> {
        let mut names: Vec<_> = self.presets.keys().cloned().collect();
        names.sort();
        names
    }

    /// Check if a preset exists.
    #[allow(dead_code)]
    pub fn has_preset(&self, name: &str) -> bool {
        self.presets.contains_key(name)
    }

    /// Add a custom preset.
    #[allow(dead_code)]
    pub fn add_preset(&mut self, preset: Preset) -> Result<()> {
        // Validate preset
        validate::validate_preset(&preset)?;

        // Don't allow overriding built-in presets
        if let Some(existing) = self.presets.get(&preset.name) {
            if existing.builtin {
                return Err(anyhow!("Cannot override built-in preset '{}'", preset.name));
            }
        }

        self.presets.insert(preset.name.clone(), preset);
        Ok(())
    }

    /// Save a custom preset to file.
    #[allow(dead_code)]
    pub fn save_preset(&self, name: &str) -> Result<()> {
        let preset = self.get_preset(name)?;

        if preset.builtin {
            return Err(anyhow!("Cannot save built-in preset '{}'", name));
        }

        let dir = self
            .custom_dir
            .as_ref()
            .ok_or_else(|| anyhow!("No custom preset directory configured"))?;

        if !dir.exists() {
            std::fs::create_dir_all(dir).context("Failed to create custom preset directory")?;
        }

        custom::save_preset_to_file(preset, dir)?;
        Ok(())
    }

    /// Remove a custom preset.
    #[allow(dead_code)]
    pub fn remove_preset(&mut self, name: &str) -> Result<()> {
        let preset = self.get_preset(name)?;

        if preset.builtin {
            return Err(anyhow!("Cannot remove built-in preset '{}'", name));
        }

        self.presets.remove(name);

        // Also remove file if in custom directory
        if let Some(ref dir) = self.custom_dir {
            let path = dir.join(format!("{}.toml", name));
            if path.exists() {
                std::fs::remove_file(&path).context("Failed to remove preset file")?;
            }
        }

        Ok(())
    }

    /// Get default custom preset directory.
    pub fn default_custom_dir() -> Result<PathBuf> {
        let config_dir =
            dirs::config_dir().ok_or_else(|| anyhow!("Could not determine config directory"))?;
        Ok(config_dir.join("oximedia").join("presets"))
    }
}

impl Default for PresetManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper function to parse bitrate string to bits per second.
#[allow(dead_code)]
pub fn parse_bitrate(bitrate: &str) -> Result<u64> {
    let bitrate = bitrate.trim();
    let multiplier = if bitrate.ends_with('M') || bitrate.ends_with('m') {
        1_000_000
    } else if bitrate.ends_with('K') || bitrate.ends_with('k') {
        1_000
    } else {
        1
    };

    let numeric = bitrate.trim_end_matches(|c: char| c.is_alphabetic()).trim();

    let value: f64 = numeric
        .parse()
        .context(format!("Invalid bitrate format: {}", bitrate))?;

    Ok((value * multiplier as f64) as u64)
}

/// Helper function to format bitrate for display.
#[allow(dead_code)]
pub fn format_bitrate(bits_per_second: u64) -> String {
    if bits_per_second >= 1_000_000 {
        format!("{:.1}M", bits_per_second as f64 / 1_000_000.0)
    } else if bits_per_second >= 1_000 {
        format!("{}k", bits_per_second / 1_000)
    } else {
        format!("{}", bits_per_second)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_bitrate() {
        assert_eq!(
            parse_bitrate("5M").expect("parse should succeed"),
            5_000_000
        );
        assert_eq!(
            parse_bitrate("2.5M").expect("parse should succeed"),
            2_500_000
        );
        assert_eq!(
            parse_bitrate("128k").expect("parse should succeed"),
            128_000
        );
        assert_eq!(parse_bitrate("1000").expect("parse should succeed"), 1_000);
    }

    #[test]
    fn test_format_bitrate() {
        assert_eq!(format_bitrate(5_000_000), "5.0M");
        assert_eq!(format_bitrate(128_000), "128k");
        assert_eq!(format_bitrate(500), "500");
    }
}
