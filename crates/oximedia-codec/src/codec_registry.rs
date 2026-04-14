//! Codec registry — register and look up codecs by name, FOURCC, or [`CodecId`].
//!
//! The registry stores one [`CodecDescriptor`] per entry and supports queries by:
//! - String name (e.g. `"av1"`, `"vp9"`)
//! - Four-character code (`[u8; 4]`, e.g. `*b"AV01"`)
//! - [`CodecId`] enum variant
//! - Capability flags (encode / decode / lossless)
//! - Profile enumeration
//!
//! # Example
//!
//! ```rust
//! use oximedia_codec::codec_registry::{CodecRegistry, CodecDescriptor, CodecDirection};
//! use oximedia_core::CodecId;
//!
//! let mut registry = CodecRegistry::default_registry();
//! let desc = registry.lookup_by_id(CodecId::Av1).expect("AV1 should be registered");
//! assert!(desc.can_encode);
//! assert!(desc.can_decode);
//! ```

use std::collections::HashMap;

use oximedia_core::CodecId;

/// A four-character code (`FOURCC`) identifying a codec in container metadata.
pub type Fourcc = [u8; 4];

/// Direction a codec entry supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecDirection {
    /// Encoding (raw → compressed) only.
    EncodeOnly,
    /// Decoding (compressed → raw) only.
    DecodeOnly,
    /// Both encoding and decoding.
    Both,
}

/// A single named codec profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodecProfile {
    /// Profile name (e.g. `"Main"`, `"High"`, `"Profile 0"`).
    pub name: String,
    /// Numeric profile identifier (codec-specific).
    pub id: u32,
    /// Optional human-readable description.
    pub description: Option<String>,
}

impl CodecProfile {
    /// Create a new profile entry.
    pub fn new(name: impl Into<String>, id: u32) -> Self {
        Self {
            name: name.into(),
            id,
            description: None,
        }
    }

    /// Builder: set a description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }
}

/// Complete description of a codec as stored in the registry.
#[derive(Debug, Clone)]
pub struct CodecDescriptor {
    /// Canonical [`CodecId`] variant.
    pub codec_id: CodecId,
    /// Short lowercase name used for string-based lookup (e.g. `"av1"`).
    pub name: String,
    /// Optional human-readable long name (e.g. `"AOMedia Video 1"`).
    pub long_name: Option<String>,
    /// All `FOURCC` codes associated with this codec.
    pub fourccs: Vec<Fourcc>,
    /// Whether this codec entry supports encoding.
    pub can_encode: bool,
    /// Whether this codec entry supports decoding.
    pub can_decode: bool,
    /// Whether lossless mode is available.
    pub is_lossless: bool,
    /// Named profiles supported by this codec.
    pub profiles: Vec<CodecProfile>,
    /// Maximum bit depth (e.g. 8, 10, 12).
    pub max_bit_depth: u8,
}

impl CodecDescriptor {
    /// Create a minimal descriptor from a [`CodecId`].
    pub fn new(codec_id: CodecId) -> Self {
        Self {
            name: codec_id.name().to_string(),
            codec_id,
            long_name: None,
            fourccs: Vec::new(),
            can_encode: false,
            can_decode: false,
            is_lossless: codec_id.is_lossless(),
            profiles: Vec::new(),
            max_bit_depth: 8,
        }
    }

    /// Builder: set the long name.
    pub fn with_long_name(mut self, s: impl Into<String>) -> Self {
        self.long_name = Some(s.into());
        self
    }

    /// Builder: add a FOURCC.
    pub fn with_fourcc(mut self, fourcc: Fourcc) -> Self {
        self.fourccs.push(fourcc);
        self
    }

    /// Builder: set encode/decode direction.
    pub fn with_direction(mut self, dir: CodecDirection) -> Self {
        match dir {
            CodecDirection::EncodeOnly => {
                self.can_encode = true;
                self.can_decode = false;
            }
            CodecDirection::DecodeOnly => {
                self.can_encode = false;
                self.can_decode = true;
            }
            CodecDirection::Both => {
                self.can_encode = true;
                self.can_decode = true;
            }
        }
        self
    }

    /// Builder: override lossless flag (normally derived from [`CodecId::is_lossless`]).
    pub fn with_lossless(mut self, lossless: bool) -> Self {
        self.is_lossless = lossless;
        self
    }

    /// Builder: add a profile entry.
    pub fn with_profile(mut self, profile: CodecProfile) -> Self {
        self.profiles.push(profile);
        self
    }

    /// Builder: set maximum bit depth.
    pub fn with_max_bit_depth(mut self, depth: u8) -> Self {
        self.max_bit_depth = depth;
        self
    }

    /// Returns `true` if this descriptor supports a given FOURCC.
    pub fn has_fourcc(&self, fourcc: &Fourcc) -> bool {
        self.fourccs.contains(fourcc)
    }

    /// Look up a profile by name (case-insensitive).
    pub fn profile_by_name(&self, name: &str) -> Option<&CodecProfile> {
        let lower = name.to_lowercase();
        self.profiles
            .iter()
            .find(|p| p.name.to_lowercase() == lower)
    }

    /// Look up a profile by numeric id.
    pub fn profile_by_id(&self, id: u32) -> Option<&CodecProfile> {
        self.profiles.iter().find(|p| p.id == id)
    }
}

/// Central registry of codec descriptors.
///
/// Stores at most one descriptor per [`CodecId`]. Secondary indices on name and
/// FOURCC are rebuilt lazily on mutation.
#[derive(Debug, Default)]
pub struct CodecRegistry {
    /// Primary store — keyed by [`CodecId`].
    entries: HashMap<CodecId, CodecDescriptor>,
    /// Secondary index: short name → [`CodecId`].
    name_index: HashMap<String, CodecId>,
    /// Secondary index: FOURCC bytes → [`CodecId`].
    fourcc_index: HashMap<Fourcc, CodecId>,
}

impl CodecRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register or replace a codec descriptor.
    ///
    /// Updates all secondary indices automatically.
    pub fn register(&mut self, desc: CodecDescriptor) {
        let id = desc.codec_id;

        // Remove stale index entries if we're replacing an existing entry.
        if let Some(old) = self.entries.get(&id) {
            self.name_index.remove(&old.name);
            for fc in &old.fourccs {
                self.fourcc_index.remove(fc);
            }
        }

        // Build new index entries.
        self.name_index.insert(desc.name.clone(), id);
        for fc in &desc.fourccs {
            self.fourcc_index.insert(*fc, id);
        }

        self.entries.insert(id, desc);
    }

    /// Remove a codec by [`CodecId`].
    ///
    /// Returns the removed descriptor, or `None` if it was not registered.
    pub fn remove(&mut self, id: CodecId) -> Option<CodecDescriptor> {
        let desc = self.entries.remove(&id)?;
        self.name_index.remove(&desc.name);
        for fc in &desc.fourccs {
            self.fourcc_index.remove(fc);
        }
        Some(desc)
    }

    /// Look up a descriptor by [`CodecId`].
    pub fn lookup_by_id(&self, id: CodecId) -> Option<&CodecDescriptor> {
        self.entries.get(&id)
    }

    /// Look up a descriptor by short name (case-insensitive).
    pub fn lookup_by_name(&self, name: &str) -> Option<&CodecDescriptor> {
        let key = name.to_lowercase();
        let id = self.name_index.get(&key)?;
        self.entries.get(id)
    }

    /// Look up a descriptor by FOURCC bytes.
    pub fn lookup_by_fourcc(&self, fourcc: &Fourcc) -> Option<&CodecDescriptor> {
        let id = self.fourcc_index.get(fourcc)?;
        self.entries.get(id)
    }

    /// Return all descriptors that can encode.
    pub fn encoders(&self) -> Vec<&CodecDescriptor> {
        self.entries.values().filter(|d| d.can_encode).collect()
    }

    /// Return all descriptors that can decode.
    pub fn decoders(&self) -> Vec<&CodecDescriptor> {
        self.entries.values().filter(|d| d.can_decode).collect()
    }

    /// Return all lossless codec descriptors.
    pub fn lossless_codecs(&self) -> Vec<&CodecDescriptor> {
        self.entries.values().filter(|d| d.is_lossless).collect()
    }

    /// Return the total number of registered codecs.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` if no codecs are registered.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return all registered [`CodecId`]s.
    pub fn codec_ids(&self) -> Vec<CodecId> {
        self.entries.keys().copied().collect()
    }

    /// Build a default registry pre-populated with all OxiMedia-supported codecs.
    pub fn default_registry() -> Self {
        let mut reg = Self::new();

        // ── Video ────────────────────────────────────────────────────────────
        reg.register(
            CodecDescriptor::new(CodecId::Av1)
                .with_long_name("AOMedia Video 1")
                .with_fourcc(*b"AV01")
                .with_fourcc(*b"av01")
                .with_direction(CodecDirection::Both)
                .with_max_bit_depth(12)
                .with_profile(CodecProfile::new("Main", 0).with_description("8/10-bit 4:2:0"))
                .with_profile(CodecProfile::new("High", 1).with_description("8/10-bit 4:4:4"))
                .with_profile(
                    CodecProfile::new("Professional", 2)
                        .with_description("8/10/12-bit 4:0:0/4:2:2/4:4:4"),
                ),
        );

        reg.register(
            CodecDescriptor::new(CodecId::Vp9)
                .with_long_name("Google VP9")
                .with_fourcc(*b"VP90")
                .with_fourcc(*b"vp09")
                .with_direction(CodecDirection::Both)
                .with_max_bit_depth(12)
                .with_profile(CodecProfile::new("Profile 0", 0).with_description("8-bit 4:2:0"))
                .with_profile(
                    CodecProfile::new("Profile 1", 1).with_description("8-bit 4:2:2/4:4:4"),
                )
                .with_profile(CodecProfile::new("Profile 2", 2).with_description("10/12-bit 4:2:0"))
                .with_profile(
                    CodecProfile::new("Profile 3", 3).with_description("10/12-bit 4:2:2/4:4:4"),
                ),
        );

        reg.register(
            CodecDescriptor::new(CodecId::Vp8)
                .with_long_name("Google VP8")
                .with_fourcc(*b"VP80")
                .with_fourcc(*b"vp08")
                .with_direction(CodecDirection::Both)
                .with_max_bit_depth(8)
                .with_profile(CodecProfile::new("Baseline", 0)),
        );

        reg.register(
            CodecDescriptor::new(CodecId::Theora)
                .with_long_name("Xiph.org Theora")
                .with_fourcc(*b"theo")
                .with_direction(CodecDirection::Both)
                .with_max_bit_depth(8)
                .with_profile(CodecProfile::new("VP3 Compatible", 0)),
        );

        reg.register(
            CodecDescriptor::new(CodecId::Ffv1)
                .with_long_name("FFV1 Lossless Video Codec")
                .with_fourcc(*b"FFV1")
                .with_direction(CodecDirection::Both)
                .with_lossless(true)
                .with_max_bit_depth(16)
                .with_profile(CodecProfile::new("Version 0", 0))
                .with_profile(CodecProfile::new("Version 1", 1))
                .with_profile(CodecProfile::new("Version 3", 3).with_description("Multithreaded")),
        );

        reg.register(
            CodecDescriptor::new(CodecId::Png)
                .with_long_name("PNG Lossless Image")
                .with_fourcc(*b"png ")
                .with_direction(CodecDirection::Both)
                .with_lossless(true)
                .with_max_bit_depth(16),
        );

        // ── Audio ────────────────────────────────────────────────────────────
        reg.register(
            CodecDescriptor::new(CodecId::Opus)
                .with_long_name("Opus Interactive Audio Codec")
                .with_fourcc(*b"Opus")
                .with_direction(CodecDirection::Both)
                .with_max_bit_depth(16)
                .with_profile(CodecProfile::new("SILK", 0).with_description("Speech optimised"))
                .with_profile(CodecProfile::new("CELT", 1).with_description("Music/wideband"))
                .with_profile(
                    CodecProfile::new("Hybrid", 2).with_description("SILK+CELT combined"),
                ),
        );

        reg.register(
            CodecDescriptor::new(CodecId::Vorbis)
                .with_long_name("Xiph.org Vorbis")
                .with_fourcc(*b"vorb")
                .with_direction(CodecDirection::Both)
                .with_max_bit_depth(16),
        );

        reg.register(
            CodecDescriptor::new(CodecId::Flac)
                .with_long_name("Free Lossless Audio Codec")
                .with_fourcc(*b"fLaC")
                .with_direction(CodecDirection::Both)
                .with_lossless(true)
                .with_max_bit_depth(32),
        );

        reg.register(
            CodecDescriptor::new(CodecId::Pcm)
                .with_long_name("Raw PCM Audio")
                .with_fourcc(*b"pcm ")
                .with_direction(CodecDirection::Both)
                .with_lossless(true)
                .with_max_bit_depth(32),
        );

        reg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_registry() -> CodecRegistry {
        CodecRegistry::default_registry()
    }

    #[test]
    fn test_lookup_by_id_av1() {
        let reg = make_registry();
        let desc = reg.lookup_by_id(CodecId::Av1).expect("AV1 registered");
        assert_eq!(desc.name, "av1");
        assert!(desc.can_encode);
        assert!(desc.can_decode);
    }

    #[test]
    fn test_lookup_by_name_case_insensitive() {
        let reg = make_registry();
        assert!(reg.lookup_by_name("AV1").is_some());
        assert!(reg.lookup_by_name("av1").is_some());
        assert!(reg.lookup_by_name("Vp9").is_some());
    }

    #[test]
    fn test_lookup_by_fourcc() {
        let reg = make_registry();
        let desc = reg
            .lookup_by_fourcc(b"AV01")
            .expect("AV01 FOURCC registered");
        assert_eq!(desc.codec_id, CodecId::Av1);
    }

    #[test]
    fn test_lookup_missing_codec() {
        let reg = make_registry();
        assert!(reg.lookup_by_id(CodecId::H263).is_none());
        assert!(reg.lookup_by_name("nonexistent").is_none());
    }

    #[test]
    fn test_encoders_decoders() {
        let reg = make_registry();
        let encoders = reg.encoders();
        let decoders = reg.decoders();
        assert!(!encoders.is_empty(), "should have at least one encoder");
        assert!(!decoders.is_empty(), "should have at least one decoder");
        // All default entries have both directions
        assert_eq!(encoders.len(), decoders.len());
    }

    #[test]
    fn test_lossless_codecs() {
        let reg = make_registry();
        let lossless = reg.lossless_codecs();
        let ids: Vec<_> = lossless.iter().map(|d| d.codec_id).collect();
        assert!(ids.contains(&CodecId::Flac));
        assert!(ids.contains(&CodecId::Pcm));
        assert!(ids.contains(&CodecId::Ffv1));
    }

    #[test]
    fn test_register_and_remove() {
        let mut reg = CodecRegistry::new();
        reg.register(CodecDescriptor::new(CodecId::Av1).with_direction(CodecDirection::DecodeOnly));
        assert_eq!(reg.len(), 1);
        let removed = reg.remove(CodecId::Av1).expect("should remove");
        assert_eq!(removed.codec_id, CodecId::Av1);
        assert!(reg.is_empty());
        // Secondary indices should be cleaned up
        assert!(reg.lookup_by_name("av1").is_none());
    }

    #[test]
    fn test_replace_existing_entry() {
        let mut reg = CodecRegistry::new();
        reg.register(
            CodecDescriptor::new(CodecId::Vp9)
                .with_direction(CodecDirection::DecodeOnly)
                .with_fourcc(*b"VP90"),
        );
        // Re-register with encode enabled
        reg.register(
            CodecDescriptor::new(CodecId::Vp9)
                .with_direction(CodecDirection::Both)
                .with_fourcc(*b"VP90"),
        );
        let desc = reg.lookup_by_id(CodecId::Vp9).expect("should be present");
        assert!(desc.can_encode);
        assert_eq!(reg.len(), 1, "replacement should not duplicate");
    }

    #[test]
    fn test_profile_lookup() {
        let reg = make_registry();
        let desc = reg.lookup_by_id(CodecId::Av1).expect("AV1 registered");
        let main = desc.profile_by_name("main").expect("Main profile exists");
        assert_eq!(main.id, 0);
        let prof2 = desc.profile_by_id(2).expect("Profile 2 exists");
        assert_eq!(prof2.name, "Professional");
    }

    #[test]
    fn test_codec_ids_all_present() {
        let reg = make_registry();
        let ids = reg.codec_ids();
        assert!(ids.contains(&CodecId::Av1));
        assert!(ids.contains(&CodecId::Opus));
        assert!(ids.contains(&CodecId::Flac));
    }

    #[test]
    fn test_has_fourcc() {
        let desc = CodecDescriptor::new(CodecId::Vp8).with_fourcc(*b"VP80");
        assert!(desc.has_fourcc(b"VP80"));
        assert!(!desc.has_fourcc(b"VP90"));
    }
}
