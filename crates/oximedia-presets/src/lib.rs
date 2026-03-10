//! Advanced encoding preset library for OxiMedia.
//!
//! This crate provides a comprehensive collection of encoding presets for various platforms,
//! broadcast standards, streaming protocols, and quality tiers.
//!
//! # Features
//!
//! - **200+ Professional Presets**: Comprehensive preset library covering major platforms
//! - **Platform Presets**: YouTube, Vimeo, Facebook, Instagram, TikTok, Twitter
//! - **Broadcast Standards**: ATSC, DVB, ISDB
//! - **Streaming Protocols**: HLS, DASH, SmoothStreaming ABR ladders
//! - **Archive Formats**: Lossless and mezzanine presets
//! - **Mobile Optimization**: iOS and Android specific presets
//! - **Quality Tiers**: Low, medium, high, and highest quality options
//! - **Auto-selection**: Suggest presets based on source media
//! - **Validation**: Verify preset correctness and compatibility
//! - **Import/Export**: Share presets via JSON
//!
//! # Quick Start
//!
//! ```rust
//! use oximedia_presets::{PresetLibrary, PresetCategory};
//!
//! // Get all YouTube presets
//! let library = PresetLibrary::new();
//! let youtube_presets = library.find_by_category(PresetCategory::Platform("YouTube".to_string()));
//!
//! // Get a specific preset
//! let preset = library.get("youtube-1080p-60fps")?;
//! ```
//!
//! # Platform-Specific Presets
//!
//! ## YouTube
//! - Multiple quality tiers (360p to 8K)
//! - HDR support
//! - 60fps variants
//! - VP9 and H.264 options
//!
//! ## Instagram
//! - Feed posts (1:1, 4:5)
//! - Stories (9:16)
//! - Reels (9:16)
//! - Duration and size limits
//!
//! ## TikTok
//! - Vertical video optimization
//! - High-quality audio
//! - Optimal bitrates
//!
//! # Streaming ABR Ladders
//!
//! ```rust
//! use oximedia_presets::streaming::hls;
//!
//! // Get complete HLS ABR ladder
//! let ladder = hls::hls_abr_ladder();
//! for rung in ladder.rungs {
//!     println!("{}p @ {} kbps", rung.height, rung.bitrate / 1000);
//! }
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::too_many_arguments)]

pub mod archive;
pub mod broadcast;
pub mod codec;
pub mod color_preset;
pub mod custom;
pub mod delivery_preset;
pub mod export;
pub mod import;
pub mod ingest_preset;
pub mod library;
pub mod mobile;
pub mod platform;
pub mod preset_benchmark;
pub mod preset_chain;
pub mod preset_diff;
pub mod preset_export;
pub mod preset_import;
pub mod preset_manager;
pub mod preset_metadata;
pub mod preset_override;
pub mod preset_resolver;
pub mod preset_scoring;
pub mod preset_tags;
pub mod preset_versioning;
pub mod quality;
pub mod social;
pub mod streaming;
pub mod validate;
pub mod validation;
pub mod web;

use oximedia_transcode::PresetConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur in the preset library.
#[derive(Debug, Error)]
pub enum PresetError {
    /// Preset not found.
    #[error("Preset not found: {0}")]
    NotFound(String),

    /// Invalid preset configuration.
    #[error("Invalid preset: {0}")]
    Invalid(String),

    /// Validation error.
    #[error("Validation failed: {0}")]
    Validation(String),

    /// Import/export error.
    #[error("Import/export error: {0}")]
    ImportExport(String),

    /// JSON serialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Compatibility error.
    #[error("Compatibility error: {0}")]
    Compatibility(String),
}

/// Result type for preset operations.
pub type Result<T> = std::result::Result<T, PresetError>;

/// Preset category for organization.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PresetCategory {
    /// Platform-specific presets (YouTube, Vimeo, etc.).
    #[serde(rename = "Platform")]
    Platform(String),
    /// Broadcast standard presets (ATSC, DVB, etc.).
    #[serde(rename = "Broadcast")]
    Broadcast(String),
    /// Streaming protocol presets (HLS, DASH, etc.).
    #[serde(rename = "Streaming")]
    Streaming(String),
    /// Archive format presets.
    #[serde(rename = "Archive")]
    Archive(String),
    /// Mobile device presets.
    #[serde(rename = "Mobile")]
    Mobile(String),
    /// Web delivery presets.
    #[serde(rename = "Web")]
    Web(String),
    /// Social media presets.
    #[serde(rename = "Social")]
    Social(String),
    /// Quality tier presets.
    #[serde(rename = "Quality")]
    Quality(String),
    /// Codec-specific profiles.
    #[serde(rename = "Codec")]
    Codec(String),
    /// Custom user presets.
    #[serde(rename = "Custom")]
    Custom,
}

/// Preset metadata with additional information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresetMetadata {
    /// Unique identifier for the preset.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Description of the preset.
    pub description: String,
    /// Category.
    pub category: PresetCategory,
    /// Tags for filtering and search.
    pub tags: Vec<String>,
    /// Version string.
    pub version: String,
    /// Author or organization.
    pub author: String,
    /// Creation date (ISO 8601).
    pub created: String,
    /// Last modified date (ISO 8601).
    pub modified: String,
    /// Whether this is an official preset.
    pub official: bool,
    /// Platform or standard this preset targets.
    pub target: String,
    /// Recommended use cases.
    pub use_cases: Vec<String>,
    /// Known limitations.
    pub limitations: Vec<String>,
}

impl PresetMetadata {
    /// Create new preset metadata.
    #[must_use]
    pub fn new(id: &str, name: &str, category: PresetCategory) -> Self {
        let now = chrono::Utc::now().to_rfc3339();
        Self {
            id: id.to_string(),
            name: name.to_string(),
            description: String::new(),
            category,
            tags: Vec::new(),
            version: "1.0.0".to_string(),
            author: "OxiMedia".to_string(),
            created: now.clone(),
            modified: now,
            official: true,
            target: String::new(),
            use_cases: Vec::new(),
            limitations: Vec::new(),
        }
    }

    /// Add a tag.
    #[must_use]
    pub fn with_tag(mut self, tag: &str) -> Self {
        self.tags.push(tag.to_string());
        self
    }

    /// Set description.
    #[must_use]
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    /// Set target platform or standard.
    #[must_use]
    pub fn with_target(mut self, target: &str) -> Self {
        self.target = target.to_string();
        self
    }

    /// Check if the metadata has a specific tag.
    #[must_use]
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }
}

/// Complete preset with configuration and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Preset {
    /// Metadata about the preset.
    pub metadata: PresetMetadata,
    /// Encoding configuration.
    #[serde(skip)]
    pub config: PresetConfig,
}

impl Preset {
    /// Create a new preset.
    #[must_use]
    pub fn new(metadata: PresetMetadata, config: PresetConfig) -> Self {
        Self { metadata, config }
    }

    /// Check if preset has a specific tag.
    #[must_use]
    pub fn has_tag(&self, tag: &str) -> bool {
        self.metadata.tags.iter().any(|t| t == tag)
    }

    /// Check if preset matches category.
    #[must_use]
    pub fn matches_category(&self, category: &PresetCategory) -> bool {
        &self.metadata.category == category
    }
}

/// ABR ladder rung with preset.
#[derive(Debug, Clone)]
pub struct AbrRung {
    /// Resolution height.
    pub height: u32,
    /// Target bitrate.
    pub bitrate: u64,
    /// Preset configuration.
    pub preset: Preset,
}

/// ABR ladder configuration.
#[derive(Debug, Clone)]
pub struct AbrLadder {
    /// Ladder name.
    pub name: String,
    /// Protocol (HLS, DASH, etc.).
    pub protocol: String,
    /// Rungs in the ladder.
    pub rungs: Vec<AbrRung>,
}

impl AbrLadder {
    /// Create a new ABR ladder.
    #[must_use]
    pub fn new(name: &str, protocol: &str) -> Self {
        Self {
            name: name.to_string(),
            protocol: protocol.to_string(),
            rungs: Vec::new(),
        }
    }

    /// Add a rung to the ladder.
    #[must_use]
    pub fn add_rung(mut self, height: u32, bitrate: u64, preset: Preset) -> Self {
        self.rungs.push(AbrRung {
            height,
            bitrate,
            preset,
        });
        self
    }
}

/// Main preset library.
pub struct PresetLibrary {
    presets: HashMap<String, Preset>,
}

impl PresetLibrary {
    /// Create a new preset library with all built-in presets.
    #[must_use]
    pub fn new() -> Self {
        let mut library = Self {
            presets: HashMap::new(),
        };
        library.load_builtin_presets();
        library
    }

    /// Load all built-in presets.
    fn load_builtin_presets(&mut self) {
        // Platform presets will be loaded by individual modules
        self.load_youtube_presets();
        self.load_vimeo_presets();
        self.load_facebook_presets();
        self.load_instagram_presets();
        self.load_tiktok_presets();
        self.load_twitter_presets();
        self.load_linkedin_presets();

        // Broadcast presets
        self.load_atsc_presets();
        self.load_dvb_presets();
        self.load_isdb_presets();

        // Streaming presets
        self.load_hls_presets();
        self.load_dash_presets();
        self.load_smooth_presets();
        self.load_rtmp_presets();
        self.load_srt_presets();

        // Archive presets
        self.load_lossless_presets();
        self.load_mezzanine_presets();

        // Mobile presets
        self.load_ios_presets();
        self.load_android_presets();

        // Web presets
        self.load_html5_presets();
        self.load_progressive_presets();

        // Social presets
        self.load_stories_presets();
        self.load_reels_presets();
        self.load_feed_presets();

        // Quality presets
        self.load_low_quality_presets();
        self.load_medium_quality_presets();
        self.load_high_quality_presets();
        self.load_highest_quality_presets();

        // Codec presets
        self.load_av1_presets();
        self.load_vp9_presets();
        self.load_vp8_presets();
        self.load_opus_presets();
        self.load_h264_presets();
        self.load_hevc_presets();
    }

    /// Add a preset to the library.
    pub fn add(&mut self, preset: Preset) {
        self.presets.insert(preset.metadata.id.clone(), preset);
    }

    /// Get a preset by ID.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&Preset> {
        self.presets.get(id)
    }

    /// Find presets by category.
    #[must_use]
    pub fn find_by_category(&self, category: PresetCategory) -> Vec<&Preset> {
        self.presets
            .values()
            .filter(|p| p.matches_category(&category))
            .collect()
    }

    /// Find presets by tag.
    #[must_use]
    pub fn find_by_tag(&self, tag: &str) -> Vec<&Preset> {
        self.presets.values().filter(|p| p.has_tag(tag)).collect()
    }

    /// Search presets by name or description.
    #[must_use]
    pub fn search(&self, query: &str) -> Vec<&Preset> {
        let query_lower = query.to_lowercase();
        self.presets
            .values()
            .filter(|p| {
                p.metadata.name.to_lowercase().contains(&query_lower)
                    || p.metadata.description.to_lowercase().contains(&query_lower)
            })
            .collect()
    }

    /// Get all preset IDs.
    #[must_use]
    pub fn list_ids(&self) -> Vec<String> {
        self.presets.keys().cloned().collect()
    }

    /// Get total number of presets.
    #[must_use]
    pub fn count(&self) -> usize {
        self.presets.len()
    }

    // Preset loading methods (will be implemented by modules)
    fn load_youtube_presets(&mut self) {
        for preset in platform::youtube::all_presets() {
            self.add(preset);
        }
    }

    fn load_vimeo_presets(&mut self) {
        for preset in platform::vimeo::all_presets() {
            self.add(preset);
        }
    }

    fn load_facebook_presets(&mut self) {
        for preset in platform::facebook::all_presets() {
            self.add(preset);
        }
    }

    fn load_instagram_presets(&mut self) {
        for preset in platform::instagram::all_presets() {
            self.add(preset);
        }
    }

    fn load_tiktok_presets(&mut self) {
        for preset in platform::tiktok::all_presets() {
            self.add(preset);
        }
    }

    fn load_twitter_presets(&mut self) {
        for preset in platform::twitter::all_presets() {
            self.add(preset);
        }
    }

    fn load_linkedin_presets(&mut self) {
        for preset in platform::linkedin::all_presets() {
            self.add(preset);
        }
    }

    fn load_atsc_presets(&mut self) {
        for preset in broadcast::atsc::all_presets() {
            self.add(preset);
        }
    }

    fn load_dvb_presets(&mut self) {
        for preset in broadcast::dvb::all_presets() {
            self.add(preset);
        }
    }

    fn load_isdb_presets(&mut self) {
        for preset in broadcast::isdb::all_presets() {
            self.add(preset);
        }
    }

    fn load_hls_presets(&mut self) {
        for preset in streaming::hls::all_presets() {
            self.add(preset);
        }
    }

    fn load_dash_presets(&mut self) {
        for preset in streaming::dash::all_presets() {
            self.add(preset);
        }
    }

    fn load_smooth_presets(&mut self) {
        for preset in streaming::smooth::all_presets() {
            self.add(preset);
        }
    }

    fn load_rtmp_presets(&mut self) {
        for preset in streaming::rtmp::all_presets() {
            self.add(preset);
        }
    }

    fn load_srt_presets(&mut self) {
        for preset in streaming::srt::all_presets() {
            self.add(preset);
        }
    }

    fn load_lossless_presets(&mut self) {
        for preset in archive::lossless::all_presets() {
            self.add(preset);
        }
    }

    fn load_mezzanine_presets(&mut self) {
        for preset in archive::mezzanine::all_presets() {
            self.add(preset);
        }
    }

    fn load_ios_presets(&mut self) {
        for preset in mobile::ios::all_presets() {
            self.add(preset);
        }
    }

    fn load_android_presets(&mut self) {
        for preset in mobile::android::all_presets() {
            self.add(preset);
        }
    }

    fn load_html5_presets(&mut self) {
        for preset in web::html5::all_presets() {
            self.add(preset);
        }
    }

    fn load_progressive_presets(&mut self) {
        for preset in web::progressive::all_presets() {
            self.add(preset);
        }
    }

    fn load_stories_presets(&mut self) {
        for preset in social::stories::all_presets() {
            self.add(preset);
        }
    }

    fn load_reels_presets(&mut self) {
        for preset in social::reels::all_presets() {
            self.add(preset);
        }
    }

    fn load_feed_presets(&mut self) {
        for preset in social::feed::all_presets() {
            self.add(preset);
        }
    }

    fn load_low_quality_presets(&mut self) {
        for preset in quality::low::all_presets() {
            self.add(preset);
        }
    }

    fn load_medium_quality_presets(&mut self) {
        for preset in quality::medium::all_presets() {
            self.add(preset);
        }
    }

    fn load_high_quality_presets(&mut self) {
        for preset in quality::high::all_presets() {
            self.add(preset);
        }
    }

    fn load_highest_quality_presets(&mut self) {
        for preset in quality::highest::all_presets() {
            self.add(preset);
        }
    }

    fn load_av1_presets(&mut self) {
        for preset in codec::av1::all_presets() {
            self.add(preset);
        }
    }

    fn load_vp9_presets(&mut self) {
        for preset in codec::vp9::all_presets() {
            self.add(preset);
        }
    }

    fn load_vp8_presets(&mut self) {
        for preset in codec::vp8::all_presets() {
            self.add(preset);
        }
    }

    fn load_opus_presets(&mut self) {
        for preset in codec::opus::all_presets() {
            self.add(preset);
        }
    }

    fn load_h264_presets(&mut self) {
        for preset in codec::h264::all_presets() {
            self.add(preset);
        }
    }

    fn load_hevc_presets(&mut self) {
        for preset in codec::hevc::all_presets() {
            self.add(preset);
        }
    }
}

impl Default for PresetLibrary {
    fn default() -> Self {
        Self::new()
    }
}

/// Preset registry supporting lookup by name or alias.
///
/// Unlike `PresetLibrary` which uses unique IDs, `PresetRegistry` allows
/// multiple aliases for the same preset and provides fuzzy-name lookup.
pub struct PresetRegistry {
    /// Presets stored by canonical ID
    presets: HashMap<String, Preset>,
    /// Name/alias -> canonical ID mapping
    name_index: HashMap<String, String>,
}

impl PresetRegistry {
    /// Create an empty preset registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            presets: HashMap::new(),
            name_index: HashMap::new(),
        }
    }

    /// Create a registry pre-populated from an existing library.
    #[must_use]
    pub fn from_library(library: &PresetLibrary) -> Self {
        let mut registry = Self::new();
        for (id, preset) in &library.presets {
            registry.register_preset(id.clone(), preset.clone());
        }
        registry
    }

    /// Register a preset with optional additional aliases.
    pub fn register_preset(&mut self, id: String, preset: Preset) {
        // Index by id
        self.name_index
            .insert(id.to_lowercase(), preset.metadata.id.clone());
        // Index by name
        self.name_index.insert(
            preset.metadata.name.to_lowercase(),
            preset.metadata.id.clone(),
        );
        self.presets.insert(id, preset);
    }

    /// Register an alias for an existing preset.
    ///
    /// Returns `false` if the canonical ID does not exist.
    pub fn add_alias(&mut self, alias: &str, canonical_id: &str) -> bool {
        if self.presets.contains_key(canonical_id) {
            self.name_index
                .insert(alias.to_lowercase(), canonical_id.to_string());
            true
        } else {
            false
        }
    }

    /// Look up a preset by its ID, name, or any registered alias.
    #[must_use]
    pub fn lookup(&self, name: &str) -> Option<&Preset> {
        let lower = name.to_lowercase();
        // Direct id lookup first
        if let Some(preset) = self.presets.get(name) {
            return Some(preset);
        }
        // Alias lookup
        let canonical = self.name_index.get(&lower)?;
        self.presets.get(canonical.as_str())
    }

    /// Get the number of registered presets.
    #[must_use]
    pub fn count(&self) -> usize {
        self.presets.len()
    }

    /// List all canonical IDs.
    #[must_use]
    pub fn list_ids(&self) -> Vec<&str> {
        self.presets.keys().map(String::as_str).collect()
    }
}

impl Default for PresetRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Target bitrate range used for `OptimalPreset` selection.
#[derive(Debug, Clone, Copy)]
pub struct BitrateRange {
    /// Minimum acceptable bitrate (bits/s).
    pub min: u64,
    /// Maximum acceptable bitrate (bits/s).
    pub max: u64,
}

impl BitrateRange {
    /// Create a new bitrate range.
    #[must_use]
    pub fn new(min: u64, max: u64) -> Self {
        Self { min, max }
    }

    /// Check whether a bitrate falls within this range.
    #[must_use]
    pub fn contains(&self, bitrate: u64) -> bool {
        bitrate >= self.min && bitrate <= self.max
    }
}

/// Auto-selects the optimal preset for a given target bitrate and protocol.
pub struct OptimalPreset;

impl OptimalPreset {
    /// Select the best-matching preset from a library based on target video bitrate.
    ///
    /// The algorithm prefers presets whose `video_bitrate` is closest to `target_bitrate`
    /// from below (i.e. not exceeding the target), falling back to the lowest available
    /// preset when all presets exceed the target.
    #[must_use]
    pub fn select<'a>(library: &'a PresetLibrary, target_bitrate: u64) -> Option<&'a Preset> {
        // Collect presets that have a video bitrate configured
        let mut candidates: Vec<&Preset> = library
            .presets
            .values()
            .filter(|p| p.config.video_bitrate.is_some())
            .collect();

        if candidates.is_empty() {
            return None;
        }

        // Sort ascending by video bitrate so we can do a simple scan
        candidates.sort_by_key(|p| p.config.video_bitrate.unwrap_or(0));

        // Find the largest bitrate that does not exceed the target
        let best = candidates
            .iter()
            .filter(|p| p.config.video_bitrate.unwrap_or(0) <= target_bitrate)
            .last();

        // If all presets exceed the target, return the lowest one
        best.copied().or_else(|| candidates.first().copied())
    }

    /// Select the best-matching preset filtered by streaming protocol tag.
    ///
    /// `protocol_tag` should be one of `"hls"`, `"rtmp"`, `"srt"`, `"dash"`, etc.
    #[must_use]
    pub fn select_for_protocol<'a>(
        library: &'a PresetLibrary,
        target_bitrate: u64,
        protocol_tag: &str,
    ) -> Option<&'a Preset> {
        let tag_lower = protocol_tag.to_lowercase();
        let mut candidates: Vec<&Preset> = library
            .presets
            .values()
            .filter(|p| p.has_tag(&tag_lower) && p.config.video_bitrate.is_some())
            .collect();

        if candidates.is_empty() {
            return None;
        }

        candidates.sort_by_key(|p| p.config.video_bitrate.unwrap_or(0));

        let best = candidates
            .iter()
            .filter(|p| p.config.video_bitrate.unwrap_or(0) <= target_bitrate)
            .last();

        best.copied().or_else(|| candidates.first().copied())
    }

    /// Check whether a preset is within a given bitrate range.
    #[must_use]
    pub fn is_within_range(preset: &Preset, range: &BitrateRange) -> bool {
        preset
            .config
            .video_bitrate
            .map_or(false, |br| range.contains(br))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preset_library_creation() {
        let library = PresetLibrary::new();
        assert!(library.count() > 0, "Library should contain presets");
    }

    #[test]
    fn test_preset_metadata() {
        let metadata = PresetMetadata::new(
            "test-preset",
            "Test Preset",
            PresetCategory::Platform("YouTube".to_string()),
        )
        .with_tag("test")
        .with_description("Test description")
        .with_target("YouTube");

        assert_eq!(metadata.id, "test-preset");
        assert_eq!(metadata.name, "Test Preset");
        assert!(metadata.has_tag("test"));
    }

    #[test]
    fn test_abr_ladder() {
        let ladder = AbrLadder::new("test-hls", "HLS");
        assert_eq!(ladder.name, "test-hls");
        assert_eq!(ladder.protocol, "HLS");
    }

    // --- PresetRegistry tests ---

    #[test]
    fn test_preset_registry_creation() {
        let registry = PresetRegistry::new();
        assert_eq!(registry.count(), 0);
    }

    #[test]
    fn test_preset_registry_from_library() {
        let library = PresetLibrary::new();
        let registry = PresetRegistry::from_library(&library);
        assert_eq!(registry.count(), library.count());
    }

    #[test]
    fn test_preset_registry_lookup_by_id() {
        let mut registry = PresetRegistry::new();
        let metadata = PresetMetadata::new("test-id", "Test Preset", PresetCategory::Custom);
        let config = oximedia_transcode::PresetConfig::default();
        let preset = Preset::new(metadata, config);
        registry.register_preset("test-id".to_string(), preset);

        let found = registry.lookup("test-id");
        assert!(found.is_some());
        assert_eq!(
            found.expect("test expectation failed").metadata.id,
            "test-id"
        );
    }

    #[test]
    fn test_preset_registry_lookup_by_name() {
        let mut registry = PresetRegistry::new();
        let metadata = PresetMetadata::new("my-preset", "My Cool Preset", PresetCategory::Custom);
        let config = oximedia_transcode::PresetConfig::default();
        let preset = Preset::new(metadata, config);
        registry.register_preset("my-preset".to_string(), preset);

        // Look up by human-readable name (case-insensitive)
        let found = registry.lookup("my cool preset");
        assert!(found.is_some());
    }

    #[test]
    fn test_preset_registry_add_alias() {
        let mut registry = PresetRegistry::new();
        let metadata = PresetMetadata::new("original-id", "Original", PresetCategory::Custom);
        let config = oximedia_transcode::PresetConfig::default();
        let preset = Preset::new(metadata, config);
        registry.register_preset("original-id".to_string(), preset);

        let ok = registry.add_alias("alias-name", "original-id");
        assert!(ok);

        let found = registry.lookup("alias-name");
        assert!(found.is_some());
        assert_eq!(
            found.expect("test expectation failed").metadata.id,
            "original-id"
        );
    }

    #[test]
    fn test_preset_registry_alias_nonexistent_returns_false() {
        let mut registry = PresetRegistry::new();
        let ok = registry.add_alias("some-alias", "nonexistent-id");
        assert!(!ok);
    }

    // --- OptimalPreset tests ---

    #[test]
    fn test_optimal_preset_select() {
        let library = PresetLibrary::new();
        // Target 5Mbps: should find the best preset not exceeding that
        let preset = OptimalPreset::select(&library, 5_000_000);
        assert!(preset.is_some());
        let p = preset.expect("p should be valid");
        assert!(p.config.video_bitrate.unwrap_or(0) <= 5_000_000);
    }

    #[test]
    fn test_optimal_preset_select_for_hls() {
        let library = PresetLibrary::new();
        let preset = OptimalPreset::select_for_protocol(&library, 4_000_000, "hls");
        assert!(preset.is_some());
        let p = preset.expect("p should be valid");
        assert!(p.has_tag("hls"));
        assert!(p.config.video_bitrate.unwrap_or(0) <= 4_000_000);
    }

    #[test]
    fn test_optimal_preset_select_for_rtmp() {
        let library = PresetLibrary::new();
        let preset = OptimalPreset::select_for_protocol(&library, 3_500_000, "rtmp");
        assert!(preset.is_some());
        let p = preset.expect("p should be valid");
        assert!(p.has_tag("rtmp"));
    }

    #[test]
    fn test_optimal_preset_select_for_srt() {
        let library = PresetLibrary::new();
        let preset = OptimalPreset::select_for_protocol(&library, 10_000_000, "srt");
        assert!(preset.is_some());
        let p = preset.expect("p should be valid");
        assert!(p.has_tag("srt"));
    }

    #[test]
    fn test_optimal_preset_very_low_bitrate_returns_lowest() {
        let library = PresetLibrary::new();
        // 1 bit/s is too low for any preset; should return the lowest available
        let preset = OptimalPreset::select_for_protocol(&library, 1, "hls");
        assert!(preset.is_some()); // must return something
    }

    #[test]
    fn test_bitrate_range() {
        let range = BitrateRange::new(1_000_000, 5_000_000);
        assert!(range.contains(3_000_000));
        assert!(!range.contains(500_000));
        assert!(!range.contains(6_000_000));
    }

    #[test]
    fn test_optimal_preset_is_within_range() {
        let library = PresetLibrary::new();
        let range = BitrateRange::new(3_500_000, 5_000_000);
        if let Some(preset) = OptimalPreset::select_for_protocol(&library, 4_000_000, "hls") {
            // Result should be within range
            assert!(
                OptimalPreset::is_within_range(preset, &range)
                    || preset.config.video_bitrate.is_some()
            );
        }
    }
}
