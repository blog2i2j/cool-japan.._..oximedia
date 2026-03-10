//! Reel and source management for EDL operations.
//!
//! This module provides structures for managing source reels, file references,
//! and source metadata in EDL files.

use crate::error::{EdlError, EdlResult};
use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;

/// Reel or source identifier used in EDL events.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ReelId(String);

impl ReelId {
    /// Create a new reel ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the reel ID is invalid (empty or too long).
    pub fn new(id: impl Into<String>) -> EdlResult<Self> {
        let id = id.into();

        if id.is_empty() {
            return Err(EdlError::InvalidReelName(
                "Reel ID cannot be empty".to_string(),
            ));
        }

        // CMX 3600 reel names are typically limited to 8 characters
        if id.len() > 32 {
            return Err(EdlError::InvalidReelName(format!(
                "Reel ID too long: {} (max 32 characters)",
                id.len()
            )));
        }

        // Validate characters (alphanumeric, underscore, hyphen)
        if !id
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
        {
            return Err(EdlError::InvalidReelName(format!(
                "Invalid characters in reel ID: {id}"
            )));
        }

        Ok(Self(id))
    }

    /// Get the reel ID as a string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Check if this is a black/BL reel (used for black frames).
    #[must_use]
    pub fn is_black(&self) -> bool {
        self.0 == "BL" || self.0 == "BLACK"
    }

    /// Check if this is an auxiliary (AX) reel.
    #[must_use]
    pub fn is_auxiliary(&self) -> bool {
        self.0 == "AX" || self.0.starts_with("AX")
    }
}

impl fmt::Display for ReelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<ReelId> for String {
    fn from(id: ReelId) -> Self {
        id.0
    }
}

/// Source information for a reel.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceInfo {
    /// Reel ID.
    pub reel_id: ReelId,

    /// Optional file path to the source media.
    pub file_path: Option<PathBuf>,

    /// Optional tape/reel name (for physical media).
    pub tape_name: Option<String>,

    /// Optional source clip name.
    pub clip_name: Option<String>,

    /// Source duration in frames (if known).
    pub duration_frames: Option<u64>,

    /// Additional metadata.
    pub metadata: HashMap<String, String>,
}

impl SourceInfo {
    /// Create new source information with the given reel ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the reel ID is invalid.
    pub fn new(reel_id: impl Into<String>) -> EdlResult<Self> {
        Ok(Self {
            reel_id: ReelId::new(reel_id)?,
            file_path: None,
            tape_name: None,
            clip_name: None,
            duration_frames: None,
            metadata: HashMap::new(),
        })
    }

    /// Set the file path.
    pub fn set_file_path(&mut self, path: PathBuf) {
        self.file_path = Some(path);
    }

    /// Set the tape name.
    pub fn set_tape_name(&mut self, name: String) {
        self.tape_name = Some(name);
    }

    /// Set the clip name.
    pub fn set_clip_name(&mut self, name: String) {
        self.clip_name = Some(name);
    }

    /// Set the duration in frames.
    pub fn set_duration_frames(&mut self, frames: u64) {
        self.duration_frames = Some(frames);
    }

    /// Add metadata entry.
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Get metadata value by key.
    #[must_use]
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }

    /// Check if this source has a file path.
    #[must_use]
    pub const fn has_file_path(&self) -> bool {
        self.file_path.is_some()
    }

    /// Get the reel ID.
    #[must_use]
    pub const fn reel_id(&self) -> &ReelId {
        &self.reel_id
    }
}

/// Reel table managing all sources in an EDL.
#[derive(Debug, Clone, Default)]
pub struct ReelTable {
    /// Map of reel IDs to source information.
    sources: HashMap<ReelId, SourceInfo>,
}

impl ReelTable {
    /// Create a new empty reel table.
    #[must_use]
    pub fn new() -> Self {
        Self {
            sources: HashMap::new(),
        }
    }

    /// Add a source to the reel table.
    pub fn add_source(&mut self, source: SourceInfo) {
        self.sources.insert(source.reel_id.clone(), source);
    }

    /// Get source information by reel ID.
    #[must_use]
    pub fn get_source(&self, reel_id: &ReelId) -> Option<&SourceInfo> {
        self.sources.get(reel_id)
    }

    /// Get mutable source information by reel ID.
    pub fn get_source_mut(&mut self, reel_id: &ReelId) -> Option<&mut SourceInfo> {
        self.sources.get_mut(reel_id)
    }

    /// Check if a reel exists in the table.
    #[must_use]
    pub fn contains_reel(&self, reel_id: &ReelId) -> bool {
        self.sources.contains_key(reel_id)
    }

    /// Remove a source from the reel table.
    pub fn remove_source(&mut self, reel_id: &ReelId) -> Option<SourceInfo> {
        self.sources.remove(reel_id)
    }

    /// Get the number of sources in the table.
    #[must_use]
    pub fn len(&self) -> usize {
        self.sources.len()
    }

    /// Check if the reel table is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }

    /// Get an iterator over all sources.
    pub fn iter(&self) -> impl Iterator<Item = (&ReelId, &SourceInfo)> {
        self.sources.iter()
    }

    /// Get a list of all reel IDs.
    #[must_use]
    pub fn reel_ids(&self) -> Vec<&ReelId> {
        self.sources.keys().collect()
    }

    /// Find sources by file path.
    #[must_use]
    pub fn find_by_file_path(&self, path: &PathBuf) -> Vec<&SourceInfo> {
        self.sources
            .values()
            .filter(|s| s.file_path.as_ref() == Some(path))
            .collect()
    }

    /// Find sources by clip name.
    #[must_use]
    pub fn find_by_clip_name(&self, name: &str) -> Vec<&SourceInfo> {
        self.sources
            .values()
            .filter(|s| s.clip_name.as_ref().is_some_and(|n| n == name))
            .collect()
    }

    /// Validate all sources in the table.
    ///
    /// # Errors
    ///
    /// Returns an error if any source is invalid.
    pub fn validate(&self) -> EdlResult<()> {
        for (reel_id, source) in &self.sources {
            if reel_id != &source.reel_id {
                return Err(EdlError::ValidationError(format!(
                    "Reel ID mismatch: key={}, source={}",
                    reel_id, source.reel_id
                )));
            }
        }
        Ok(())
    }
}

/// Builder for creating reel tables.
#[derive(Debug, Default)]
pub struct ReelTableBuilder {
    sources: Vec<SourceInfo>,
}

impl ReelTableBuilder {
    /// Create a new reel table builder.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            sources: Vec::new(),
        }
    }

    /// Add a source to the builder.
    #[must_use]
    pub fn add_source(mut self, source: SourceInfo) -> Self {
        self.sources.push(source);
        self
    }

    /// Add a simple reel with just an ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the reel ID is invalid.
    pub fn add_reel(mut self, reel_id: impl Into<String>) -> EdlResult<Self> {
        self.sources.push(SourceInfo::new(reel_id)?);
        Ok(self)
    }

    /// Build the reel table.
    #[must_use]
    pub fn build(self) -> ReelTable {
        let mut table = ReelTable::new();
        for source in self.sources {
            table.add_source(source);
        }
        table
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reel_id_creation() {
        let reel = ReelId::new("A001").expect("failed to create");
        assert_eq!(reel.as_str(), "A001");
    }

    #[test]
    fn test_reel_id_validation() {
        assert!(ReelId::new("").is_err());
        assert!(ReelId::new("A001").is_ok());
        assert!(ReelId::new("TEST_REEL").is_ok());
        assert!(ReelId::new("REEL-123").is_ok());
        assert!(ReelId::new("invalid!name").is_err());
    }

    #[test]
    fn test_black_reel() {
        let reel = ReelId::new("BL").expect("failed to create");
        assert!(reel.is_black());

        let reel = ReelId::new("BLACK").expect("failed to create");
        assert!(reel.is_black());

        let reel = ReelId::new("A001").expect("failed to create");
        assert!(!reel.is_black());
    }

    #[test]
    fn test_auxiliary_reel() {
        let reel = ReelId::new("AX").expect("failed to create");
        assert!(reel.is_auxiliary());

        let reel = ReelId::new("AX001").expect("failed to create");
        assert!(reel.is_auxiliary());

        let reel = ReelId::new("A001").expect("failed to create");
        assert!(!reel.is_auxiliary());
    }

    #[test]
    fn test_source_info() {
        let mut source = SourceInfo::new("A001").expect("failed to create");
        source.set_file_path(PathBuf::from("/path/to/clip.mov"));
        source.set_clip_name("Clip 1".to_string());
        source.set_duration_frames(1000);

        assert_eq!(source.reel_id.as_str(), "A001");
        assert!(source.has_file_path());
        assert_eq!(source.clip_name, Some("Clip 1".to_string()));
        assert_eq!(source.duration_frames, Some(1000));
    }

    #[test]
    fn test_reel_table() {
        let mut table = ReelTable::new();

        let source1 = SourceInfo::new("A001").expect("failed to create");
        let source2 = SourceInfo::new("A002").expect("failed to create");

        table.add_source(source1.clone());
        table.add_source(source2.clone());

        assert_eq!(table.len(), 2);
        assert!(table.contains_reel(&source1.reel_id));
        assert!(table.contains_reel(&source2.reel_id));
    }

    #[test]
    fn test_reel_table_builder() {
        let table = ReelTableBuilder::new()
            .add_reel("A001")
            .expect("operation should succeed")
            .add_reel("A002")
            .expect("operation should succeed")
            .build();

        assert_eq!(table.len(), 2);
    }

    #[test]
    fn test_metadata() {
        let mut source = SourceInfo::new("A001").expect("failed to create");
        source.add_metadata("format".to_string(), "ProRes".to_string());
        source.add_metadata("resolution".to_string(), "1920x1080".to_string());

        assert_eq!(source.get_metadata("format"), Some(&"ProRes".to_string()));
        assert_eq!(
            source.get_metadata("resolution"),
            Some(&"1920x1080".to_string())
        );
    }

    #[test]
    fn test_find_by_file_path() {
        let mut table = ReelTable::new();

        let mut source1 = SourceInfo::new("A001").expect("failed to create");
        source1.set_file_path(PathBuf::from("/path/to/clip1.mov"));
        table.add_source(source1);

        let mut source2 = SourceInfo::new("A002").expect("failed to create");
        source2.set_file_path(PathBuf::from("/path/to/clip2.mov"));
        table.add_source(source2);

        let results = table.find_by_file_path(&PathBuf::from("/path/to/clip1.mov"));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].reel_id.as_str(), "A001");
    }
}
