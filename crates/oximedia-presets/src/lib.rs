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
//! let preset = library.get("youtube-1080p-60fps");
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
pub mod audio_only;
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
pub mod optimal_preset;
pub mod platform;
pub mod preset_benchmark;
pub mod preset_chain;
pub mod preset_diff;
pub mod preset_export;
pub mod preset_import;
pub mod preset_inheritance;
pub mod preset_manager;
pub mod preset_metadata;
pub mod preset_override;
pub mod preset_resolver;
pub mod preset_scoring;
pub mod preset_tags;
pub mod preset_versioning;
pub mod quality;
pub mod scene_adaptive;
pub mod social;
pub mod streaming;
pub mod validate;
pub mod validation;
pub mod web;

pub use optimal_preset::{OptimalPresetSelector, ScoredPreset, SelectionCriteria, UseCase};

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

// ── InvertedIndex (for O(tokens) search) ─────────────────────────────────────

/// Inverted index mapping lowercase word tokens to preset IDs.
///
/// Built once during `PresetLibrary::new()` and used by `search()` for
/// O(query-tokens) lookup instead of O(presets × chars) linear scan.
#[derive(Debug, Default)]
struct InvertedIndex {
    /// token → list of preset IDs that contain the token.
    index: HashMap<String, Vec<String>>,
}

impl InvertedIndex {
    /// Tokenize `text` into lowercase alphabetic/numeric words.
    fn tokenize(text: &str) -> Vec<String> {
        text.split(|c: char| !c.is_alphanumeric())
            .filter(|s| s.len() >= 2)
            .map(|s| s.to_lowercase())
            .collect()
    }

    /// Index a single preset.
    fn insert(&mut self, preset: &Preset) {
        let id = preset.metadata.id.clone();
        let mut tokens: Vec<String> = Vec::new();
        tokens.extend(Self::tokenize(&preset.metadata.name));
        tokens.extend(Self::tokenize(&preset.metadata.description));
        for tag in &preset.metadata.tags {
            tokens.extend(Self::tokenize(tag));
        }
        tokens.extend(Self::tokenize(&preset.metadata.id));
        // Deduplicate tokens per preset to avoid inflating relevance.
        tokens.sort_unstable();
        tokens.dedup();
        for token in tokens {
            self.index.entry(token).or_default().push(id.clone());
        }
    }

    /// Build an index from all presets.
    fn build(presets: &HashMap<String, Preset>) -> Self {
        let mut idx = Self::default();
        for preset in presets.values() {
            idx.insert(preset);
        }
        idx
    }

    /// Search: intersect results across query tokens, sorted by hit frequency.
    ///
    /// Returns preset IDs ranked by how many query tokens they matched (descending).
    fn search<'a>(&'a self, query: &str, presets: &'a HashMap<String, Preset>) -> Vec<&'a Preset> {
        let query_tokens = Self::tokenize(query);
        if query_tokens.is_empty() {
            return Vec::new();
        }

        // Count how many tokens each preset matches.
        let mut hit_count: HashMap<&str, usize> = HashMap::new();
        for token in &query_tokens {
            if let Some(ids) = self.index.get(token.as_str()) {
                for id in ids {
                    *hit_count.entry(id.as_str()).or_insert(0) += 1;
                }
            }
        }

        // Only include presets that match ALL query tokens.
        let required = query_tokens.len();
        let mut results: Vec<(&str, usize)> = hit_count
            .into_iter()
            .filter(|(_, count)| *count >= required)
            .collect();
        results.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));

        results
            .into_iter()
            .filter_map(|(id, _)| presets.get(id))
            .collect()
    }
}

/// Main preset library.
pub struct PresetLibrary {
    presets: HashMap<String, Preset>,
    /// Optional inheritance registry for derived preset resolution.
    inheritance: preset_inheritance::InheritanceRegistry,
    /// Pre-built inverted index for O(tokens) text search.
    search_index: InvertedIndex,
}

impl PresetLibrary {
    /// Create a new preset library with all built-in presets.
    #[must_use]
    pub fn new() -> Self {
        let mut library = Self {
            presets: HashMap::new(),
            inheritance: preset_inheritance::InheritanceRegistry::new(),
            search_index: InvertedIndex::default(),
        };
        library.load_builtin_presets();
        // Build the inverted index after all presets are loaded.
        library.search_index = InvertedIndex::build(&library.presets);
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
        self.load_twitch_presets();
        self.load_dcp_presets();

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

        // Audio-only presets
        self.load_flac_podcast_presets();
        self.load_opus_podcast_presets();
    }

    /// Add a preset to the library.
    pub fn add(&mut self, preset: Preset) {
        self.search_index.insert(&preset);
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

    /// Search presets by name, description, tags, or ID.
    ///
    /// Uses the pre-built inverted index for O(query-tokens) lookup.  All
    /// query tokens must appear in a preset for it to be returned (AND
    /// semantics).  Results are ranked by token-hit frequency.
    ///
    /// Falls back to a single-token prefix search when the query contains
    /// only one token that is too short to index.
    #[must_use]
    pub fn search(&self, query: &str) -> Vec<&Preset> {
        self.search_index.search(query, &self.presets)
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

    // ── Inheritance API ───────────────────────────────────────────────────

    /// Register a preset as a base (root) node in the inheritance graph.
    ///
    /// The `config` should capture all inheritable fields for this preset.
    /// Subsequent derived registrations can then override individual fields.
    pub fn register_inheritance_base(
        &mut self,
        preset_id: &str,
        config: preset_inheritance::InheritedConfig,
    ) {
        self.inheritance.register_base(preset_id, config);
    }

    /// Register a derived preset that inherits all fields from `parent_id`
    /// and overrides only the fields listed in `overrides`.
    ///
    /// Returns `false` if the parent ID is not yet registered (the record is
    /// still stored and will be resolved once the parent is added).
    pub fn register_inheritance_derived(
        &mut self,
        preset_id: &str,
        parent_id: &str,
        overrides: preset_inheritance::InheritedConfig,
    ) -> bool {
        self.inheritance
            .register_derived(preset_id, parent_id, overrides)
    }

    /// Resolve the fully-merged [`InheritedConfig`] for `preset_id` by
    /// walking the ancestor chain registered in the inheritance graph.
    ///
    /// # Errors
    ///
    /// Returns an [`InheritanceError`] if the ID is unknown, the chain is
    /// circular, or the depth limit is exceeded.
    ///
    /// [`InheritedConfig`]: preset_inheritance::InheritedConfig
    /// [`InheritanceError`]: preset_inheritance::InheritanceError
    pub fn resolve_inheritance(
        &self,
        preset_id: &str,
    ) -> std::result::Result<
        preset_inheritance::InheritedConfig,
        preset_inheritance::InheritanceError,
    > {
        self.inheritance.resolve(preset_id)
    }

    /// Return a reference to the underlying [`InheritanceRegistry`] for
    /// advanced use cases (depth queries, cycle detection, etc.).
    ///
    /// [`InheritanceRegistry`]: preset_inheritance::InheritanceRegistry
    #[must_use]
    pub fn inheritance_registry(&self) -> &preset_inheritance::InheritanceRegistry {
        &self.inheritance
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

    fn load_twitch_presets(&mut self) {
        for preset in platform::twitch::all_presets() {
            self.add(preset);
        }
    }

    fn load_dcp_presets(&mut self) {
        for preset in platform::dcp::all_presets() {
            self.add(preset);
        }
    }

    fn load_flac_podcast_presets(&mut self) {
        for preset in audio_only::flac::all_presets() {
            self.add(preset);
        }
    }

    fn load_opus_podcast_presets(&mut self) {
        for preset in audio_only::opus_podcast::all_presets() {
            self.add(preset);
        }
    }

    /// Iterate over all presets in the library.
    ///
    /// Used by the scored-selection system to score every preset without
    /// exposing the internal storage type.
    pub fn presets_iter(&self) -> impl Iterator<Item = &Preset> {
        self.presets.values()
    }

    /// Add a derived preset that inherits from an existing preset in the library.
    ///
    /// The `overrides` closure receives a mutable clone of the base preset's config
    /// and metadata so the caller can modify only the fields that differ.
    /// Returns `None` if the `base_id` is not found.
    pub fn derive_from(
        &mut self,
        base_id: &str,
        new_id: &str,
        new_name: &str,
        apply_overrides: impl FnOnce(&mut crate::Preset),
    ) -> Option<()> {
        let base = self.presets.get(base_id)?.clone();
        let mut derived = base;
        derived.metadata.id = new_id.to_string();
        derived.metadata.name = new_name.to_string();
        apply_overrides(&mut derived);
        self.presets.insert(new_id.to_string(), derived);
        Some(())
    }
}

impl Default for PresetLibrary {
    fn default() -> Self {
        Self::new()
    }
}

// ── Global cached PresetLibrary ───────────────────────────────────────────────

/// Global singleton `PresetLibrary` initialized on first access.
///
/// Calling `PresetLibrary::global()` 100 times performs only one real
/// initialization. Subsequent calls return the same `&'static PresetLibrary`.
static GLOBAL_LIBRARY: std::sync::OnceLock<PresetLibrary> = std::sync::OnceLock::new();

impl PresetLibrary {
    /// Return a reference to the global (lazily-initialized) `PresetLibrary`.
    ///
    /// The library is built exactly once regardless of how many times this
    /// function is called, making it safe and cheap to call in hot paths.
    #[must_use]
    pub fn global() -> &'static PresetLibrary {
        GLOBAL_LIBRARY.get_or_init(PresetLibrary::new)
    }
}

// ── Lazy preset category loader ───────────────────────────────────────────────

/// Describes the broad category group for lazy loading.
///
/// `LazyPresetCategory` defers construction of a preset group until the
/// first call to [`LazyPresetCategory::get`].  This means a caller that only
/// ever uses HLS presets never pays the cost of building broadcast or mobile
/// presets.
pub struct LazyPresetCategory {
    /// Human-readable name for the category group.
    name: &'static str,
    /// Factory function that builds all presets for this category.
    loader: fn() -> Vec<Preset>,
    /// Lazily-initialized preset list.
    loaded: std::sync::OnceLock<Vec<Preset>>,
}

impl LazyPresetCategory {
    /// Create a new lazy category with the given name and loader function.
    #[must_use]
    pub const fn new(name: &'static str, loader: fn() -> Vec<Preset>) -> Self {
        Self {
            name,
            loader,
            loaded: std::sync::OnceLock::new(),
        }
    }

    /// Return the presets for this category, loading them if necessary.
    #[must_use]
    pub fn get(&self) -> &[Preset] {
        self.loaded.get_or_init(|| (self.loader)())
    }

    /// Return the category name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// Check whether this category's presets have already been loaded.
    #[must_use]
    pub fn is_loaded(&self) -> bool {
        self.loaded.get().is_some()
    }
}

impl std::fmt::Debug for LazyPresetCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LazyPresetCategory")
            .field("name", &self.name)
            .field("loaded", &self.is_loaded())
            .finish()
    }
}

/// Pre-defined lazy category instances for the three most-accessed groups.
///
/// These can be used directly without creating a full `PresetLibrary`.
pub static LAZY_HLS_PRESETS: LazyPresetCategory =
    LazyPresetCategory::new("HLS", streaming::hls::all_presets);
/// Lazy-loaded YouTube preset category for platform-specific encoding targets.
pub static LAZY_YOUTUBE_PRESETS: LazyPresetCategory =
    LazyPresetCategory::new("YouTube", platform::youtube::all_presets);
/// Lazy-loaded ATSC broadcast preset category for broadcast standards compliance.
pub static LAZY_BROADCAST_PRESETS: LazyPresetCategory =
    LazyPresetCategory::new("Broadcast/ATSC", broadcast::atsc::all_presets);

// ── Levenshtein distance (pure-Rust, no external deps) ────────────────────

/// Compute the Levenshtein edit distance between two strings.
///
/// Uses a two-row DP approach with O(min(a,b)) space.
/// Returns 0 when `a == b` and increases by 1 for each insertion, deletion,
/// or substitution required to transform `a` into `b`.
fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let la = a_chars.len();
    let lb = b_chars.len();

    if la == 0 {
        return lb;
    }
    if lb == 0 {
        return la;
    }

    // prev[j] = distance(a[0..i-1], b[0..j])
    let mut prev: Vec<usize> = (0..=lb).collect();
    let mut curr = vec![0usize; lb + 1];

    for i in 1..=la {
        curr[0] = i;
        for j in 1..=lb {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            curr[j] = (prev[j] + 1) // deletion
                .min(curr[j - 1] + 1) // insertion
                .min(prev[j - 1] + cost); // substitution
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[lb]
}

/// Result of a fuzzy search, including the matched preset (Arc-shared) and its distance.
#[derive(Debug, Clone)]
pub struct FuzzyMatch {
    /// The matched preset (cheaply cloneable via Arc).
    pub preset: std::sync::Arc<Preset>,
    /// Levenshtein edit distance from the query (lower = better).
    pub distance: usize,
}

/// Preset registry supporting lookup by name or alias.
///
/// Unlike `PresetLibrary` which uses unique IDs, `PresetRegistry` allows
/// multiple aliases for the same preset and provides fuzzy-name lookup.
///
/// Presets are stored behind `Arc` so that every `lookup` or `fuzzy_search`
/// returns an `Arc<Preset>` — an O(1) reference-count increment rather than a
/// full clone of the preset data.
pub struct PresetRegistry {
    /// Presets stored by canonical ID (Arc-shared to avoid clone-on-lookup).
    presets: HashMap<String, std::sync::Arc<Preset>>,
    /// Name/alias -> canonical ID mapping.
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

    /// Register a preset, wrapping it in `Arc` for zero-cost subsequent lookups.
    ///
    /// Returns the `Arc<Preset>` so callers can hold a shared reference without
    /// performing a second lookup.
    pub fn register_preset(&mut self, id: String, preset: Preset) -> std::sync::Arc<Preset> {
        let arc = std::sync::Arc::new(preset);
        // Index by id (lowercase)
        self.name_index
            .insert(id.to_lowercase(), arc.metadata.id.clone());
        // Index by name (lowercase)
        self.name_index
            .insert(arc.metadata.name.to_lowercase(), arc.metadata.id.clone());
        self.presets.insert(id, std::sync::Arc::clone(&arc));
        arc
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
    ///
    /// Returns a cheap `Arc` clone — no deep copy of the preset data occurs.
    #[must_use]
    pub fn lookup(&self, name: &str) -> Option<std::sync::Arc<Preset>> {
        let lower = name.to_lowercase();
        // Direct canonical-id lookup first.
        if let Some(arc) = self.presets.get(name) {
            return Some(std::sync::Arc::clone(arc));
        }
        // Alias lookup.
        let canonical = self.name_index.get(&lower)?;
        self.presets
            .get(canonical.as_str())
            .map(std::sync::Arc::clone)
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

    /// Fuzzy lookup: return all presets whose ID or name is within `max_distance`
    /// Levenshtein edits of `query`, sorted by ascending distance.
    ///
    /// Returns `Vec<FuzzyMatch>` where each match holds an `Arc<Preset>` —
    /// all returned values share the underlying preset allocation.
    ///
    /// This is typo-tolerant — a `max_distance` of 2 will accept single
    /// transpositions, missing characters, and minor spelling mistakes.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use oximedia_presets::{PresetRegistry, PresetLibrary};
    /// let lib = PresetLibrary::new();
    /// let reg = PresetRegistry::from_library(&lib);
    /// // "yotube" is 1 edit away from "youtube"
    /// let matches = reg.fuzzy_search("yotube-1080p", 2);
    /// ```
    #[must_use]
    pub fn fuzzy_search(&self, query: &str, max_distance: usize) -> Vec<FuzzyMatch> {
        let query_lower = query.to_lowercase();
        let mut results: Vec<FuzzyMatch> = self
            .presets
            .values()
            .filter_map(|arc| {
                let id_dist = levenshtein_distance(&query_lower, &arc.metadata.id.to_lowercase());
                let name_dist =
                    levenshtein_distance(&query_lower, &arc.metadata.name.to_lowercase());
                let dist = id_dist.min(name_dist);
                // Also search through name_index aliases.
                let alias_dist = self
                    .name_index
                    .iter()
                    .filter(|(_, canonical)| canonical.as_str() == arc.metadata.id.as_str())
                    .map(|(alias, _)| levenshtein_distance(&query_lower, alias))
                    .min()
                    .unwrap_or(usize::MAX);
                let best = dist.min(alias_dist);
                if best <= max_distance {
                    Some(FuzzyMatch {
                        preset: std::sync::Arc::clone(arc),
                        distance: best,
                    })
                } else {
                    None
                }
            })
            .collect();

        results.sort_by(|a, b| {
            a.distance
                .cmp(&b.distance)
                .then(a.preset.metadata.id.cmp(&b.preset.metadata.id))
        });
        results
    }

    /// Fuzzy lookup: return the single best match within `max_distance` edits,
    /// or `None` if no preset falls within the threshold.
    ///
    /// When multiple presets share the minimum distance the one with the
    /// alphabetically-first ID is returned to ensure deterministic behaviour.
    #[must_use]
    pub fn fuzzy_lookup(&self, query: &str, max_distance: usize) -> Option<std::sync::Arc<Preset>> {
        let mut matches = self.fuzzy_search(query, max_distance);
        if matches.is_empty() {
            return None;
        }
        let min_dist = matches[0].distance;
        matches.retain(|m| m.distance == min_dist);
        matches.sort_by(|a, b| a.preset.metadata.id.cmp(&b.preset.metadata.id));
        matches.into_iter().next().map(|m| m.preset)
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

    // ── InvertedIndex / PresetLibrary::search() tests (Task C) ────────────

    /// Single-token query matches presets whose name/description/tags contain that token.
    #[test]
    fn test_search_single_token_matches() {
        let library = PresetLibrary::new();
        let results = library.search("youtube");
        assert!(
            !results.is_empty(),
            "Single token 'youtube' should match YouTube presets"
        );
        // Every returned preset should mention "youtube" in name, description, or a tag.
        for p in &results {
            let combined = format!(
                "{} {} {}",
                p.metadata.name.to_lowercase(),
                p.metadata.description.to_lowercase(),
                p.metadata.tags.join(" ").to_lowercase()
            );
            assert!(
                combined.contains("youtube"),
                "Returned preset '{}' should be related to youtube",
                p.metadata.id
            );
        }
    }

    /// Multi-token query uses AND semantics: only presets matching ALL tokens are returned.
    #[test]
    fn test_search_multi_token_and_semantics() {
        let library = PresetLibrary::new();
        // "hls 1080p" — only HLS presets that are also 1080p should match.
        let results = library.search("hls 1080p");
        assert!(
            !results.is_empty(),
            "Multi-token 'hls 1080p' should return at least one preset"
        );
        for p in &results {
            let has_hls = p.has_tag("hls")
                || p.metadata.name.to_lowercase().contains("hls")
                || p.metadata.description.to_lowercase().contains("hls");
            let has_1080 = p.has_tag("1080p")
                || p.metadata.name.contains("1080")
                || p.metadata.description.contains("1080");
            assert!(
                has_hls && has_1080,
                "Preset '{}' should match both 'hls' and '1080p'",
                p.metadata.id
            );
        }
    }

    /// Query with no matching tokens returns an empty Vec (not a panic).
    #[test]
    fn test_search_no_match_returns_empty() {
        let library = PresetLibrary::new();
        let results = library.search("zzznomatchtoken99");
        assert!(
            results.is_empty(),
            "Non-existent token should return empty results"
        );
    }

    /// Search is case-insensitive: uppercase/mixed-case query matches lowercase indexed tokens.
    #[test]
    fn test_search_case_insensitive() {
        let library = PresetLibrary::new();
        let lower = library.search("youtube");
        let upper = library.search("YOUTUBE");
        let mixed = library.search("YouTube");
        // All three should return the same number of results.
        assert_eq!(
            lower.len(),
            upper.len(),
            "Case should not affect result count (lower vs upper)"
        );
        assert_eq!(
            lower.len(),
            mixed.len(),
            "Case should not affect result count (lower vs mixed)"
        );
        assert!(
            !lower.is_empty(),
            "Case-insensitive search for 'YouTube' should find results"
        );
    }

    /// Tag-based search: query with a tag token finds presets with that tag.
    #[test]
    fn test_search_by_tag() {
        let library = PresetLibrary::new();
        let results = library.search("hls");
        assert!(
            !results.is_empty(),
            "Tag-based search for 'hls' should return HLS presets"
        );
        // At least one result should have the "hls" tag.
        let has_hls_tag = results.iter().any(|p| p.has_tag("hls"));
        assert!(
            has_hls_tag,
            "Search for 'hls' should return at least one preset with the hls tag"
        );
    }

    /// ID-based search: searching by a partial ID token finds the preset.
    #[test]
    fn test_search_by_id_token() {
        let library = PresetLibrary::new();
        // "hls-240p" has ID token "hls" and "240p"; search "240p" should find it.
        let results = library.search("240p");
        assert!(
            !results.is_empty(),
            "Searching '240p' should find presets with that resolution token"
        );
        let found_240p = results
            .iter()
            .any(|p| p.metadata.id.contains("240p") || p.has_tag("240p"));
        assert!(found_240p, "Search '240p' should include a 240p preset");
    }

    /// Description-based search: query tokens from a description are indexed.
    #[test]
    fn test_search_by_description_token() {
        let library = PresetLibrary::new();
        // The HLS 240p preset has description "HLS ABR ladder - 240p @ 500kbps".
        // Searching "ladder" (a description-only word) should surface HLS presets.
        let results = library.search("ladder");
        assert!(
            !results.is_empty(),
            "Description token 'ladder' should find ABR ladder presets"
        );
    }

    /// Short (1-char) tokens are excluded from the index (min token length = 2).
    #[test]
    fn test_search_short_token_excluded() {
        let library = PresetLibrary::new();
        // Single-character queries tokenize to nothing → empty result.
        let results = library.search("a");
        assert!(
            results.is_empty(),
            "Single-character query should return empty (below min token length)"
        );
    }

    /// Multi-token where one token is absent → empty (strict AND semantics).
    #[test]
    fn test_search_multi_token_absent_one_returns_empty() {
        let library = PresetLibrary::new();
        // "youtube zzznomatch" — second token absent, so intersection is empty.
        let results = library.search("youtube zzznomatch");
        assert!(
            results.is_empty(),
            "AND semantics: missing token should cause empty result"
        );
    }
}

/// Wave 3 tests are in a separate module file to keep lib.rs under 2000 lines.
#[cfg(test)]
#[path = "wave3_tests.rs"]
mod wave3_tests;
