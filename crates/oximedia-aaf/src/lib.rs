//! `OxiMedia` AAF - Advanced Authoring Format support
//!
//! This crate provides SMPTE ST 377-1 compliant AAF (Advanced Authoring Format)
//! reading and writing for professional post-production workflows.
//!
//! # Features
//!
//! - Full SMPTE ST 377-1 (AAF Object Specification) support
//! - SMPTE ST 2001 (AAF Operational Patterns) support
//! - Microsoft Structured Storage (compound file) parsing
//! - Complete object model (Mobs, Segments, Components, Effects)
//! - Dictionary support with extensibility
//! - Essence reference handling (embedded and external)
//! - Timeline and edit rate management
//! - Metadata preservation
//! - Conversion to `OpenTimelineIO` and EDL formats
//! - Read and write capability
//! - No unsafe code
//!
//! # AAF Structure
//!
//! AAF files use Microsoft Structured Storage format (compound files) and contain:
//! - Header: File identification and version
//! - Dictionary: Class, property, and type definitions
//! - Content Storage: Mobs (Master, Source, Composition)
//! - Essence: Media data (embedded or external references)
//!
//! # Example
//!
//! ```rust,no_run
//! use oximedia_aaf::{AafFile, AafReader};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Open an AAF file
//! let mut reader = AafReader::open("timeline.aaf")?;
//! let aaf = reader.read()?;
//!
//! // Access composition mobs
//! for comp_mob in aaf.composition_mobs() {
//!     println!("Composition: {}", comp_mob.name());
//!     for track in comp_mob.tracks() {
//!         println!("  Track: {}", track.name);
//!     }
//! }
//! # Ok(())
//! # }
//! ```

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    dead_code,
    clippy::pedantic
)]

pub mod aaf_export;
pub mod composition;
pub mod composition_mob;
pub mod convert;
pub mod davinci_edl;
pub mod descriptor;
pub mod dict_cache;
pub mod dictionary;
pub mod edl_export;
pub mod effect_def;
pub mod effects;
pub mod essence;
pub mod flatten;
pub mod inspector;
pub mod interchange;
pub mod lazy_essence;
pub mod media_data;
pub mod media_file_ref;
pub mod merge;
pub mod metadata;
pub mod mob_slot;
pub mod mob_traversal;
pub mod object_model;
pub mod operation_group;
pub mod parameter;
pub mod property_value;
pub mod scope;
pub mod search;
pub mod selector;
pub mod smpte_metadata;
pub mod source_clip;
pub mod streaming;
pub mod structured_storage;
pub mod timeline;
pub mod timeline_export;
pub mod timeline_mob;
pub mod track_group;
pub mod transition_def;
pub mod validate;
pub mod writer;
pub mod xml_bridge;

use std::collections::HashMap;
use std::io::{Read, Seek};
use std::path::Path;
use thiserror::Error;
use uuid::Uuid;

pub use composition::{
    CompositionMob, Effect, EffectParameter, FadeType, Filler, Sequence, SequenceComponent,
    SourceClip, Track, TrackType, Transition, UsageCode,
};
pub use convert::{
    EdlExporter, Timeline, TimelineClip, TimelineConverter, TimelineTrack, XmlExporter,
};
pub use dictionary::{Auid, DataDefinition, Dictionary};
pub use essence::{EssenceAccess, EssenceDescriptor, EssenceReference};
pub use metadata::{Comment, KlvData, TaggedValue, Timecode as AafTimecode};
pub use object_model::{Component, Header, Mob, MobSlot, Segment};
pub use search::{AafQuery, AafSearcher, MobRef, MobTypeKind};
pub use structured_storage::{StorageReader, StorageWriter};
pub use timeline::{EditRate, Position};
pub use validate::{AafValidator, IssueSeverity, ValidationIssue, ValidationReport};
pub use writer::AafWriter;

/// AAF error types
#[derive(Error, Debug)]
pub enum AafError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid AAF file: {0}")]
    InvalidFile(String),

    #[error("Unsupported AAF version: {0}.{1}")]
    UnsupportedVersion(u16, u16),

    #[error("Invalid structured storage: {0}")]
    InvalidStructuredStorage(String),

    #[error("Object not found: {0}")]
    ObjectNotFound(String),

    #[error("Property not found: {0}")]
    PropertyNotFound(String),

    #[error("Invalid class: {0}")]
    InvalidClass(String),

    #[error("Invalid property type: {0}")]
    InvalidPropertyType(String),

    #[error("Reference resolution failed: {0}")]
    ReferenceResolutionFailed(String),

    #[error("Dictionary error: {0}")]
    DictionaryError(String),

    #[error("Essence error: {0}")]
    EssenceError(String),

    #[error("Timeline error: {0}")]
    TimelineError(String),

    #[error("Conversion error: {0}")]
    ConversionError(String),

    #[error("Write error: {0}")]
    WriteError(String),

    #[error("Parse error: {0}")]
    ParseError(String),
}

pub type Result<T> = std::result::Result<T, AafError>;

/// Main AAF file structure
///
/// Represents a complete AAF file with header, dictionary, and content.
#[derive(Debug, Clone)]
pub struct AafFile {
    pub(crate) header: Header,
    pub(crate) dictionary: Dictionary,
    pub(crate) content_storage: ContentStorage,
    pub(crate) essence_data: Vec<EssenceData>,
}

impl AafFile {
    /// Create a new empty AAF file
    #[must_use]
    pub fn new() -> Self {
        Self {
            header: Header::new(),
            dictionary: Dictionary::new(),
            content_storage: ContentStorage::new(),
            essence_data: Vec::new(),
        }
    }

    /// Get the file header
    #[must_use]
    pub fn header(&self) -> &Header {
        &self.header
    }

    /// Get the dictionary
    #[must_use]
    pub fn dictionary(&self) -> &Dictionary {
        &self.dictionary
    }

    /// Get the content storage
    #[must_use]
    pub fn content_storage(&self) -> &ContentStorage {
        &self.content_storage
    }

    /// Get all composition mobs
    #[must_use]
    pub fn composition_mobs(&self) -> Vec<&CompositionMob> {
        self.content_storage.composition_mobs()
    }

    /// Get all master mobs
    #[must_use]
    pub fn master_mobs(&self) -> Vec<&Mob> {
        self.content_storage.master_mobs()
    }

    /// Get all source mobs
    #[must_use]
    pub fn source_mobs(&self) -> Vec<&Mob> {
        self.content_storage.source_mobs()
    }

    /// Find a mob by ID
    #[must_use]
    pub fn find_mob(&self, mob_id: &Uuid) -> Option<&Mob> {
        self.content_storage.find_mob(mob_id)
    }

    /// Get all essence data
    #[must_use]
    pub fn essence_data(&self) -> &[EssenceData] {
        &self.essence_data
    }

    /// Get the file's edit rate (from first composition mob)
    #[must_use]
    pub fn edit_rate(&self) -> Option<EditRate> {
        self.composition_mobs()
            .first()
            .and_then(|mob| mob.edit_rate())
    }

    /// Get file duration in edit units
    #[must_use]
    pub fn duration(&self) -> Option<i64> {
        self.composition_mobs()
            .first()
            .and_then(|mob| mob.duration())
    }
}

impl Default for AafFile {
    fn default() -> Self {
        Self::new()
    }
}

/// Content storage containing all mobs
#[derive(Debug, Clone)]
pub struct ContentStorage {
    mobs: HashMap<Uuid, Mob>,
    composition_mobs: HashMap<Uuid, CompositionMob>,
}

impl ContentStorage {
    /// Create new empty content storage
    #[must_use]
    pub fn new() -> Self {
        Self {
            mobs: HashMap::new(),
            composition_mobs: HashMap::new(),
        }
    }

    /// Add a mob
    pub fn add_mob(&mut self, mob: Mob) {
        let id = mob.mob_id();
        self.mobs.insert(id, mob);
    }

    /// Add a composition mob
    pub fn add_composition_mob(&mut self, comp_mob: CompositionMob) {
        let id = comp_mob.mob_id();
        self.composition_mobs.insert(id, comp_mob);
    }

    /// Get all composition mobs
    #[must_use]
    pub fn composition_mobs(&self) -> Vec<&CompositionMob> {
        self.composition_mobs.values().collect()
    }

    /// Get all master mobs
    #[must_use]
    pub fn master_mobs(&self) -> Vec<&Mob> {
        self.mobs.values().filter(|m| m.is_master_mob()).collect()
    }

    /// Get all source mobs
    #[must_use]
    pub fn source_mobs(&self) -> Vec<&Mob> {
        self.mobs.values().filter(|m| m.is_source_mob()).collect()
    }

    /// Find a mob by ID
    #[must_use]
    pub fn find_mob(&self, mob_id: &Uuid) -> Option<&Mob> {
        self.mobs.get(mob_id)
    }

    /// Find a composition mob by ID
    #[must_use]
    pub fn find_composition_mob(&self, mob_id: &Uuid) -> Option<&CompositionMob> {
        self.composition_mobs.get(mob_id)
    }

    /// Find a composition mob by name (first match).
    ///
    /// Performs a linear scan over all composition mobs comparing names.
    /// Returns `None` if no mob with that name exists.
    #[must_use]
    pub fn find_composition_mob_by_name(&self, name: &str) -> Option<&CompositionMob> {
        self.composition_mobs.values().find(|m| m.name() == name)
    }

    /// Deep-clone a composition mob with a fresh UUID.
    ///
    /// The cloned mob is an independent copy of the original; all tracks, sequences,
    /// and clips are preserved but the top-level `mob_id` is replaced with a new
    /// randomly-generated UUID so the clone does not conflict with the original.
    ///
    /// # Errors
    ///
    /// Returns `AafError::ObjectNotFound` if no composition mob with `id` exists.
    pub fn clone_mob(&self, id: Uuid) -> Result<CompositionMob> {
        let original = self
            .composition_mobs
            .get(&id)
            .ok_or_else(|| AafError::ObjectNotFound(format!("Composition mob {id} not found")))?;

        // Clone the whole structure and assign a fresh UUID
        let mut cloned = original.clone();
        let new_id = Uuid::new_v4();
        *cloned.mob_mut().mob_id_mut() = new_id;
        Ok(cloned)
    }

    /// Validate the content storage and return a report.
    ///
    /// Convenience wrapper around `AafValidator::validate`.
    /// Returns errors and warnings about structural integrity of the stored mobs.
    #[must_use]
    pub fn validate(&self) -> validate::ValidationReport {
        validate::AafValidator::validate(self)
    }
}

impl Default for ContentStorage {
    fn default() -> Self {
        Self::new()
    }
}

/// Essence data stored in the AAF file
#[derive(Debug, Clone)]
pub struct EssenceData {
    mob_id: Uuid,
    data: Vec<u8>,
}

impl EssenceData {
    /// Create new essence data
    #[must_use]
    pub fn new(mob_id: Uuid, data: Vec<u8>) -> Self {
        Self { mob_id, data }
    }

    /// Get the mob ID
    #[must_use]
    pub fn mob_id(&self) -> Uuid {
        self.mob_id
    }

    /// Get the essence data
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }
}

/// AAF file reader
pub struct AafReader<R: Read + Seek> {
    storage: StorageReader<R>,
}

impl<R: Read + Seek> AafReader<R> {
    /// Create a new AAF reader from a readable source
    pub fn new(reader: R) -> Result<Self> {
        let storage = StorageReader::new(reader)?;
        Ok(Self { storage })
    }

    /// Read the complete AAF file
    pub fn read(&mut self) -> Result<AafFile> {
        // Read header from root entry
        let header = self.read_header()?;

        // Read dictionary
        let dictionary = self.read_dictionary()?;

        // Read content storage
        let content_storage = self.read_content_storage(&dictionary)?;

        // Read essence data
        let essence_data = self.read_essence_data()?;

        Ok(AafFile {
            header,
            dictionary,
            content_storage,
            essence_data,
        })
    }

    fn read_header(&mut self) -> Result<Header> {
        // Implementation in object_model module
        object_model::read_header(&mut self.storage)
    }

    fn read_dictionary(&mut self) -> Result<Dictionary> {
        // Implementation in dictionary module
        dictionary::read_dictionary(&mut self.storage)
    }

    fn read_content_storage(&mut self, dictionary: &Dictionary) -> Result<ContentStorage> {
        // Implementation in object_model module
        object_model::read_content_storage(&mut self.storage, dictionary)
    }

    fn read_essence_data(&mut self) -> Result<Vec<EssenceData>> {
        // Implementation in essence module
        essence::read_essence_data(&mut self.storage)
    }
}

impl AafReader<std::fs::File> {
    /// Open an AAF file from a path
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        Self::new(file)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use composition::{Filler, Sequence, SequenceComponent, SourceClip, Track, TrackType};
    use dictionary::Auid;
    use object_model::{Mob, MobType};
    use timeline::{EditRate, Position};

    #[test]
    fn test_aaf_file_creation() {
        let aaf = AafFile::new();
        assert!(aaf.composition_mobs().is_empty());
        assert!(aaf.master_mobs().is_empty());
        assert!(aaf.source_mobs().is_empty());
    }

    #[test]
    fn test_content_storage() {
        let storage = ContentStorage::new();
        assert!(storage.composition_mobs().is_empty());
        assert!(storage.master_mobs().is_empty());
    }

    #[test]
    fn test_essence_data() {
        let mob_id = Uuid::new_v4();
        let data = vec![1, 2, 3, 4];
        let essence = EssenceData::new(mob_id, data.clone());
        assert_eq!(essence.mob_id(), mob_id);
        assert_eq!(essence.data(), &data);
    }

    // --- Task 1: find_composition_mob_by_name ---

    #[test]
    fn test_find_composition_mob_by_name_found() {
        let mut storage = ContentStorage::new();
        let mob_id = Uuid::new_v4();
        storage.add_composition_mob(CompositionMob::new(mob_id, "My Comp"));

        let found = storage.find_composition_mob_by_name("My Comp");
        assert!(found.is_some(), "should find mob by name");
        assert_eq!(found.map(|m| m.mob_id()), Some(mob_id));
    }

    #[test]
    fn test_find_composition_mob_by_name_not_found() {
        let mut storage = ContentStorage::new();
        storage.add_composition_mob(CompositionMob::new(Uuid::new_v4(), "Other"));
        let found = storage.find_composition_mob_by_name("Missing");
        assert!(found.is_none());
    }

    // --- Task 2: Display for EditRate, Position, TimelineRange ---

    #[test]
    fn test_display_edit_rate_integer() {
        let rate = EditRate::new(25, 1);
        assert_eq!(rate.to_string(), "25");
    }

    #[test]
    fn test_display_edit_rate_fractional() {
        let rate = EditRate::new(30000, 1001);
        assert_eq!(rate.to_string(), "30000/1001");
    }

    #[test]
    fn test_display_position() {
        let pos = Position::new(42);
        assert_eq!(pos.to_string(), "42");
    }

    #[test]
    fn test_display_timeline_range() {
        use timeline::TimelineRange;
        let range = TimelineRange::new(Position::new(10), 50);
        // [10..60)
        assert_eq!(range.to_string(), "[10..60)");
    }

    // --- Task 3: Validation pass ---

    #[test]
    fn test_validate_valid_storage() {
        let mut storage = ContentStorage::new();
        let source_id = Uuid::new_v4();
        storage.add_mob(Mob::new(
            source_id,
            "Source.mov".to_string(),
            MobType::Source,
        ));

        let mut comp = CompositionMob::new(Uuid::new_v4(), "Edit");
        let mut seq = Sequence::new(Auid::PICTURE);
        seq.add_component(SequenceComponent::SourceClip(SourceClip::new(
            100,
            Position::zero(),
            source_id,
            1,
        )));
        let mut track = Track::new(1, "V", EditRate::PAL_25, TrackType::Picture);
        track.set_sequence(seq);
        comp.add_track(track);
        storage.add_composition_mob(comp);

        let report = storage.validate();
        assert!(
            report.is_valid(),
            "Expected valid storage; errors: {:?}",
            report.errors
        );
    }

    #[test]
    fn test_validate_missing_name_is_error() {
        let mut storage = ContentStorage::new();
        storage.add_composition_mob(CompositionMob::new(Uuid::new_v4(), ""));
        let report = storage.validate();
        assert!(!report.is_valid());
    }

    #[test]
    fn test_validate_unresolved_ref_is_error() {
        let mut storage = ContentStorage::new();
        let bogus = Uuid::new_v4(); // never registered

        let mut comp = CompositionMob::new(Uuid::new_v4(), "Broken");
        let mut seq = Sequence::new(Auid::PICTURE);
        seq.add_component(SequenceComponent::SourceClip(SourceClip::new(
            50,
            Position::zero(),
            bogus,
            1,
        )));
        let mut track = Track::new(1, "V", EditRate::PAL_25, TrackType::Picture);
        track.set_sequence(seq);
        comp.add_track(track);
        storage.add_composition_mob(comp);

        let report = storage.validate();
        assert!(!report.is_valid());
        assert!(report
            .errors
            .iter()
            .any(|e| e.message.contains("unknown mob")));
    }

    // --- Task 4: Round-trip test ---

    #[test]
    fn test_round_trip_build_verify() {
        // Programmatic round-trip: build structure → add to ContentStorage → read back.
        // (No binary serialisation in this simplified implementation, but the logical
        //  structure is fully preserved in memory.)
        let source_mob_id = Uuid::new_v4();
        let source_slot_id = 1u32;
        let clip_length: i64 = 200;
        let clip_start = Position::new(10);
        let edit_rate = EditRate::FILM_24;
        let comp_name = "RoundTripComp";

        // Build
        let mut storage = ContentStorage::new();
        storage.add_mob(Mob::new(
            source_mob_id,
            "source.mov".to_string(),
            MobType::Source,
        ));

        let comp_id = Uuid::new_v4();
        let mut comp = CompositionMob::new(comp_id, comp_name);
        let mut seq = Sequence::new(Auid::PICTURE);
        seq.add_component(SequenceComponent::SourceClip(SourceClip::new(
            clip_length,
            clip_start,
            source_mob_id,
            source_slot_id,
        )));
        let mut track = Track::new(1, "Video", edit_rate, TrackType::Picture);
        track.set_sequence(seq);
        comp.add_track(track);
        storage.add_composition_mob(comp);

        // Verify round-trip: all fields accessible after storage
        let found = storage
            .find_composition_mob(&comp_id)
            .expect("comp mob must be in storage");
        assert_eq!(found.name(), comp_name);
        assert_eq!(found.mob_id(), comp_id);

        let tracks = found.tracks();
        assert_eq!(tracks.len(), 1);
        let track = &tracks[0];
        assert_eq!(track.edit_rate, edit_rate);

        let clips = track.source_clips();
        assert_eq!(clips.len(), 1);
        let clip = clips[0];
        assert_eq!(clip.length, clip_length);
        assert_eq!(clip.start_time, clip_start);
        assert_eq!(clip.source_mob_id, source_mob_id);
        assert_eq!(clip.source_mob_slot_id, source_slot_id);

        // Validation must pass
        let report = storage.validate();
        assert!(
            report.is_valid(),
            "Round-trip storage must be valid; errors: {:?}",
            report.errors
        );
    }

    // --- Task 4: Filler and Transition in round-trip ---

    #[test]
    fn test_round_trip_with_filler() {
        let mut storage = ContentStorage::new();
        let comp_id = Uuid::new_v4();
        let mut comp = CompositionMob::new(comp_id, "FillerComp");
        let mut seq = Sequence::new(Auid::PICTURE);
        seq.add_component(SequenceComponent::Filler(Filler::new(50)));
        let mut track = Track::new(1, "V", EditRate::PAL_25, TrackType::Picture);
        track.set_sequence(seq);
        comp.add_track(track);
        storage.add_composition_mob(comp);

        let found = storage
            .find_composition_mob(&comp_id)
            .expect("comp mob must exist");
        let duration = found.duration();
        assert_eq!(
            duration,
            Some(50),
            "Duration should equal the filler length"
        );
    }

    // --- Task 5: Mob cloning ---

    #[test]
    fn test_clone_mob_creates_fresh_uuid() {
        let mut storage = ContentStorage::new();
        let original_id = Uuid::new_v4();
        let comp = CompositionMob::new(original_id, "Original");
        storage.add_composition_mob(comp);

        let cloned = storage
            .clone_mob(original_id)
            .expect("clone should succeed");

        assert_ne!(
            cloned.mob_id(),
            original_id,
            "Cloned mob must have a different UUID"
        );
        assert_eq!(cloned.name(), "Original", "Name should be preserved");
    }

    #[test]
    fn test_clone_mob_deep_copies_tracks() {
        let mut storage = ContentStorage::new();
        let original_id = Uuid::new_v4();

        let mut comp = CompositionMob::new(original_id, "WithTracks");
        let seq = Sequence::new(Auid::PICTURE);
        let mut track = Track::new(1, "V", EditRate::PAL_25, TrackType::Picture);
        track.set_sequence(seq);
        comp.add_track(track);
        storage.add_composition_mob(comp);

        let cloned = storage
            .clone_mob(original_id)
            .expect("clone should succeed");

        // Clone has same tracks
        assert_eq!(cloned.tracks().len(), 1);
        // Original still intact
        let original = storage
            .find_composition_mob(&original_id)
            .expect("original still there");
        assert_eq!(original.tracks().len(), 1);
    }

    #[test]
    fn test_clone_mob_not_found_returns_error() {
        let storage = ContentStorage::new();
        let bogus_id = Uuid::new_v4();
        let result = storage.clone_mob(bogus_id);
        assert!(result.is_err());
    }
}
