//! Audio description script management.

use crate::error::{AccessError, AccessResult};
use serde::{Deserialize, Serialize};

/// Audio description script containing timed entries.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AudioDescriptionScript {
    entries: Vec<AudioDescriptionEntry>,
    metadata: ScriptMetadata,
}

/// Metadata for an audio description script.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScriptMetadata {
    /// Title of the content.
    pub title: Option<String>,
    /// Author of the audio description.
    pub author: Option<String>,
    /// Language code (e.g., "en", "es", "fr").
    pub language: String,
    /// Creation date.
    pub created_at: Option<String>,
    /// Last modified date.
    pub modified_at: Option<String>,
    /// Additional notes.
    pub notes: Option<String>,
}

impl AudioDescriptionScript {
    /// Create a new empty script.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with metadata.
    #[must_use]
    pub fn with_metadata(metadata: ScriptMetadata) -> Self {
        Self {
            entries: Vec::new(),
            metadata,
        }
    }

    /// Add an audio description entry.
    pub fn add_entry(&mut self, entry: AudioDescriptionEntry) {
        self.entries.push(entry);
        self.sort_entries();
    }

    /// Add multiple entries.
    pub fn add_entries(&mut self, entries: Vec<AudioDescriptionEntry>) {
        self.entries.extend(entries);
        self.sort_entries();
    }

    /// Remove an entry at the given index.
    pub fn remove_entry(&mut self, index: usize) -> Option<AudioDescriptionEntry> {
        if index < self.entries.len() {
            Some(self.entries.remove(index))
        } else {
            None
        }
    }

    /// Get all entries.
    #[must_use]
    pub fn entries(&self) -> &[AudioDescriptionEntry] {
        &self.entries
    }

    /// Get mutable entries.
    pub fn entries_mut(&mut self) -> &mut Vec<AudioDescriptionEntry> {
        &mut self.entries
    }

    /// Get entries active at the given timestamp.
    #[must_use]
    pub fn entries_at(&self, timestamp_ms: i64) -> Vec<&AudioDescriptionEntry> {
        self.entries
            .iter()
            .filter(|e| e.is_active_at(timestamp_ms))
            .collect()
    }

    /// Get entry at specific index.
    #[must_use]
    pub fn entry(&self, index: usize) -> Option<&AudioDescriptionEntry> {
        self.entries.get(index)
    }

    /// Get mutable entry at specific index.
    pub fn entry_mut(&mut self, index: usize) -> Option<&mut AudioDescriptionEntry> {
        self.entries.get_mut(index)
    }

    /// Number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if script is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Sort entries by start time.
    fn sort_entries(&mut self) {
        self.entries.sort_by_key(|e| e.start_time_ms);
    }

    /// Get metadata.
    #[must_use]
    pub const fn metadata(&self) -> &ScriptMetadata {
        &self.metadata
    }

    /// Set metadata.
    pub fn set_metadata(&mut self, metadata: ScriptMetadata) {
        self.metadata = metadata;
    }

    /// Validate script for timing conflicts.
    pub fn validate(&self) -> AccessResult<()> {
        for i in 0..self.entries.len() {
            let entry = &self.entries[i];

            // Check valid duration
            if entry.end_time_ms <= entry.start_time_ms {
                return Err(AccessError::InvalidTiming(format!(
                    "Entry {} has invalid duration: start={}, end={}",
                    i, entry.start_time_ms, entry.end_time_ms
                )));
            }

            // Check for overlaps with next entry
            if i + 1 < self.entries.len() {
                let next = &self.entries[i + 1];
                if entry.end_time_ms > next.start_time_ms {
                    return Err(AccessError::InvalidTiming(format!(
                        "Entries {} and {} overlap: {}..{} and {}..{}",
                        i,
                        i + 1,
                        entry.start_time_ms,
                        entry.end_time_ms,
                        next.start_time_ms,
                        next.end_time_ms
                    )));
                }
            }
        }

        Ok(())
    }

    /// Export to JSON format.
    pub fn to_json(&self) -> AccessResult<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| AccessError::Other(format!("JSON serialization failed: {e}")))
    }

    /// Import from JSON format.
    pub fn from_json(json: &str) -> AccessResult<Self> {
        serde_json::from_str(json)
            .map_err(|e| AccessError::Other(format!("JSON deserialization failed: {e}")))
    }

    /// Get total duration covered by all entries.
    #[must_use]
    pub fn total_duration_ms(&self) -> i64 {
        self.entries
            .iter()
            .map(AudioDescriptionEntry::duration_ms)
            .sum()
    }

    /// Get time range (first start to last end).
    #[must_use]
    pub fn time_range(&self) -> Option<(i64, i64)> {
        if self.entries.is_empty() {
            return None;
        }

        let start = self.entries.first()?.start_time_ms;
        let end = self.entries.last()?.end_time_ms;
        Some((start, end))
    }
}

/// A single audio description entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioDescriptionEntry {
    /// Start time in milliseconds.
    pub start_time_ms: i64,
    /// End time in milliseconds.
    pub end_time_ms: i64,
    /// Description text.
    pub text: String,
    /// Priority level (higher = more important).
    pub priority: u8,
    /// Category of description.
    pub category: DescriptionCategory,
    /// Additional metadata.
    pub metadata: EntryMetadata,
}

/// Category of audio description.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DescriptionCategory {
    /// Scene setting and environment.
    Scene,
    /// Character appearance and actions.
    Character,
    /// On-screen text or graphics.
    Text,
    /// Important visual effects or transitions.
    VisualEffect,
    /// Mood, tone, or atmosphere.
    Mood,
    /// Other or uncategorized.
    Other,
}

/// Additional metadata for an entry.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EntryMetadata {
    /// Speaker or character context.
    pub speaker: Option<String>,
    /// Location or scene identifier.
    pub location: Option<String>,
    /// Tags for organization.
    pub tags: Vec<String>,
}

impl AudioDescriptionEntry {
    /// Create a new audio description entry.
    #[must_use]
    pub fn new(start_time_ms: i64, end_time_ms: i64, text: String) -> Self {
        Self {
            start_time_ms,
            end_time_ms,
            text,
            priority: 5,
            category: DescriptionCategory::Other,
            metadata: EntryMetadata::default(),
        }
    }

    /// Set priority.
    #[must_use]
    pub const fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    /// Set category.
    #[must_use]
    pub const fn with_category(mut self, category: DescriptionCategory) -> Self {
        self.category = category;
        self
    }

    /// Set metadata.
    #[must_use]
    pub fn with_metadata(mut self, metadata: EntryMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Get duration in milliseconds.
    #[must_use]
    pub const fn duration_ms(&self) -> i64 {
        self.end_time_ms - self.start_time_ms
    }

    /// Check if entry is active at the given timestamp.
    #[must_use]
    pub const fn is_active_at(&self, timestamp_ms: i64) -> bool {
        timestamp_ms >= self.start_time_ms && timestamp_ms < self.end_time_ms
    }

    /// Check if entry overlaps with another.
    #[must_use]
    pub const fn overlaps_with(&self, other: &Self) -> bool {
        self.start_time_ms < other.end_time_ms && self.end_time_ms > other.start_time_ms
    }

    /// Get text length.
    #[must_use]
    pub fn text_length(&self) -> usize {
        self.text.len()
    }

    /// Estimate speaking duration (rough approximation: 150 words per minute).
    #[must_use]
    pub fn estimated_speaking_duration_ms(&self) -> i64 {
        let words = self.text.split_whitespace().count();
        (words as f64 * 60000.0 / 150.0) as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_script_creation() {
        let script = AudioDescriptionScript::new();
        assert!(script.is_empty());
        assert_eq!(script.len(), 0);
    }

    #[test]
    fn test_add_entries() {
        let mut script = AudioDescriptionScript::new();
        script.add_entry(AudioDescriptionEntry::new(
            1000,
            3000,
            "First description".to_string(),
        ));
        script.add_entry(AudioDescriptionEntry::new(
            5000,
            7000,
            "Second description".to_string(),
        ));

        assert_eq!(script.len(), 2);
    }

    #[test]
    fn test_entries_at_timestamp() {
        let mut script = AudioDescriptionScript::new();
        script.add_entry(AudioDescriptionEntry::new(
            1000,
            3000,
            "Description 1".to_string(),
        ));
        script.add_entry(AudioDescriptionEntry::new(
            5000,
            7000,
            "Description 2".to_string(),
        ));

        let active = script.entries_at(2000);
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].text, "Description 1");

        let active = script.entries_at(6000);
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].text, "Description 2");

        let active = script.entries_at(4000);
        assert_eq!(active.len(), 0);
    }

    #[test]
    fn test_validation_valid() {
        let mut script = AudioDescriptionScript::new();
        script.add_entry(AudioDescriptionEntry::new(
            1000,
            2000,
            "Entry 1".to_string(),
        ));
        script.add_entry(AudioDescriptionEntry::new(
            3000,
            4000,
            "Entry 2".to_string(),
        ));

        assert!(script.validate().is_ok());
    }

    #[test]
    fn test_validation_overlap() {
        let mut script = AudioDescriptionScript::new();
        script.add_entry(AudioDescriptionEntry::new(
            1000,
            3000,
            "Entry 1".to_string(),
        ));
        script.add_entry(AudioDescriptionEntry::new(
            2000,
            4000,
            "Entry 2".to_string(),
        ));

        assert!(script.validate().is_err());
    }

    #[test]
    fn test_entry_duration() {
        let entry = AudioDescriptionEntry::new(1000, 3500, "Test".to_string());
        assert_eq!(entry.duration_ms(), 2500);
    }

    #[test]
    fn test_entry_active_at() {
        let entry = AudioDescriptionEntry::new(1000, 3000, "Test".to_string());
        assert!(!entry.is_active_at(500));
        assert!(entry.is_active_at(1000));
        assert!(entry.is_active_at(2000));
        assert!(!entry.is_active_at(3000));
    }

    #[test]
    fn test_entry_overlap() {
        let entry1 = AudioDescriptionEntry::new(1000, 3000, "Entry 1".to_string());
        let entry2 = AudioDescriptionEntry::new(2000, 4000, "Entry 2".to_string());
        let entry3 = AudioDescriptionEntry::new(5000, 7000, "Entry 3".to_string());

        assert!(entry1.overlaps_with(&entry2));
        assert!(!entry1.overlaps_with(&entry3));
    }

    #[test]
    fn test_json_roundtrip() {
        let mut script = AudioDescriptionScript::new();
        script.add_entry(AudioDescriptionEntry::new(1000, 2000, "Test".to_string()));

        let json = script.to_json().expect("json should be valid");
        let restored = AudioDescriptionScript::from_json(&json).expect("restored should be valid");

        assert_eq!(restored.len(), 1);
        assert_eq!(restored.entries()[0].text, "Test");
    }
}
