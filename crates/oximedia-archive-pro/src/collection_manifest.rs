#![allow(dead_code)]

//! Collection-level manifest management for archival packages.
//!
//! This module provides tools for building, validating, and querying manifests
//! that describe entire collections of archived media objects. A collection
//! manifest aggregates metadata across multiple archival packages.

use std::collections::HashMap;

/// Unique identifier for a manifest entry.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ManifestEntryId(String);

impl ManifestEntryId {
    /// Creates a new manifest entry identifier.
    #[must_use]
    pub fn new(id: &str) -> Self {
        Self(id.to_string())
    }

    /// Returns the identifier string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ManifestEntryId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Status of an individual manifest entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryStatus {
    /// Entry is pending ingest.
    Pending,
    /// Entry has been ingested successfully.
    Ingested,
    /// Entry failed validation.
    Failed,
    /// Entry has been withdrawn from the collection.
    Withdrawn,
    /// Entry is under review.
    UnderReview,
}

impl EntryStatus {
    /// Returns true if the entry is in a terminal state.
    #[must_use]
    pub const fn is_terminal(&self) -> bool {
        matches!(self, Self::Ingested | Self::Failed | Self::Withdrawn)
    }

    /// Returns true if the entry is active in the collection.
    #[must_use]
    pub const fn is_active(&self) -> bool {
        matches!(self, Self::Ingested)
    }

    /// Returns a label string.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Ingested => "ingested",
            Self::Failed => "failed",
            Self::Withdrawn => "withdrawn",
            Self::UnderReview => "under_review",
        }
    }
}

/// A single entry in the collection manifest.
#[derive(Debug, Clone)]
pub struct ManifestEntry {
    /// Unique identifier.
    pub id: ManifestEntryId,
    /// Human-readable title.
    pub title: String,
    /// Current status.
    pub status: EntryStatus,
    /// Size in bytes of all associated files.
    pub total_bytes: u64,
    /// Number of files.
    pub file_count: u32,
    /// SHA-256 checksum of the package manifest.
    pub package_checksum: String,
    /// Arbitrary key-value metadata.
    pub metadata: HashMap<String, String>,
}

impl ManifestEntry {
    /// Creates a new manifest entry.
    #[must_use]
    pub fn new(id: &str, title: &str) -> Self {
        Self {
            id: ManifestEntryId::new(id),
            title: title.to_string(),
            status: EntryStatus::Pending,
            total_bytes: 0,
            file_count: 0,
            package_checksum: String::new(),
            metadata: HashMap::new(),
        }
    }

    /// Sets the status.
    pub fn set_status(&mut self, status: EntryStatus) {
        self.status = status;
    }

    /// Sets size information.
    pub fn set_size(&mut self, total_bytes: u64, file_count: u32) {
        self.total_bytes = total_bytes;
        self.file_count = file_count;
    }

    /// Adds a metadata key-value pair.
    pub fn add_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }
}

/// A collection manifest aggregating multiple archival packages.
#[derive(Debug, Clone)]
pub struct CollectionManifest {
    /// Collection identifier.
    collection_id: String,
    /// Collection title.
    title: String,
    /// Entries in the manifest, keyed by entry ID.
    entries: HashMap<ManifestEntryId, ManifestEntry>,
    /// Creation timestamp (seconds since epoch).
    created_epoch: u64,
}

impl CollectionManifest {
    /// Creates a new empty collection manifest.
    #[must_use]
    pub fn new(collection_id: &str, title: &str) -> Self {
        Self {
            collection_id: collection_id.to_string(),
            title: title.to_string(),
            entries: HashMap::new(),
            created_epoch: 0,
        }
    }

    /// Sets the creation timestamp.
    pub fn set_created_epoch(&mut self, epoch: u64) {
        self.created_epoch = epoch;
    }

    /// Returns the collection identifier.
    #[must_use]
    pub fn collection_id(&self) -> &str {
        &self.collection_id
    }

    /// Returns the collection title.
    #[must_use]
    pub fn title(&self) -> &str {
        &self.title
    }

    /// Returns the number of entries.
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Adds an entry to the manifest. Returns `true` if it was newly inserted.
    pub fn add_entry(&mut self, entry: ManifestEntry) -> bool {
        if self.entries.contains_key(&entry.id) {
            return false;
        }
        self.entries.insert(entry.id.clone(), entry);
        true
    }

    /// Removes an entry by its ID. Returns the removed entry, if any.
    pub fn remove_entry(&mut self, id: &ManifestEntryId) -> Option<ManifestEntry> {
        self.entries.remove(id)
    }

    /// Looks up an entry by ID.
    #[must_use]
    pub fn get_entry(&self, id: &ManifestEntryId) -> Option<&ManifestEntry> {
        self.entries.get(id)
    }

    /// Returns a mutable reference to an entry by ID.
    pub fn get_entry_mut(&mut self, id: &ManifestEntryId) -> Option<&mut ManifestEntry> {
        self.entries.get_mut(id)
    }

    /// Returns the total bytes across all entries.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.entries.values().map(|e| e.total_bytes).sum()
    }

    /// Returns the total file count across all entries.
    #[must_use]
    pub fn total_file_count(&self) -> u64 {
        self.entries.values().map(|e| u64::from(e.file_count)).sum()
    }

    /// Returns entries filtered by status.
    #[must_use]
    pub fn entries_by_status(&self, status: EntryStatus) -> Vec<&ManifestEntry> {
        self.entries
            .values()
            .filter(|e| e.status == status)
            .collect()
    }

    /// Returns the count of entries in each status.
    #[must_use]
    pub fn status_summary(&self) -> HashMap<&'static str, usize> {
        let mut summary = HashMap::new();
        for entry in self.entries.values() {
            *summary.entry(entry.status.label()).or_insert(0) += 1;
        }
        summary
    }

    /// Validates the manifest and returns a list of issues found.
    #[must_use]
    pub fn validate(&self) -> Vec<String> {
        let mut issues = Vec::new();

        if self.collection_id.is_empty() {
            issues.push("Collection ID is empty".to_string());
        }
        if self.title.is_empty() {
            issues.push("Collection title is empty".to_string());
        }

        for entry in self.entries.values() {
            if entry.title.is_empty() {
                issues.push(format!("Entry {} has empty title", entry.id));
            }
            if entry.status == EntryStatus::Ingested && entry.package_checksum.is_empty() {
                issues.push(format!(
                    "Ingested entry {} has no package checksum",
                    entry.id
                ));
            }
        }

        issues
    }

    /// Returns all entry IDs sorted alphabetically.
    #[must_use]
    pub fn sorted_entry_ids(&self) -> Vec<ManifestEntryId> {
        let mut ids: Vec<ManifestEntryId> = self.entries.keys().cloned().collect();
        ids.sort_by(|a, b| a.as_str().cmp(b.as_str()));
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_entry(id: &str, title: &str, status: EntryStatus) -> ManifestEntry {
        let mut e = ManifestEntry::new(id, title);
        e.set_status(status);
        e.set_size(1000, 5);
        if status == EntryStatus::Ingested {
            e.package_checksum = "abc123".to_string();
        }
        e
    }

    #[test]
    fn test_manifest_entry_id_display() {
        let id = ManifestEntryId::new("PKG-001");
        assert_eq!(id.to_string(), "PKG-001");
        assert_eq!(id.as_str(), "PKG-001");
    }

    #[test]
    fn test_entry_status_is_terminal() {
        assert!(!EntryStatus::Pending.is_terminal());
        assert!(EntryStatus::Ingested.is_terminal());
        assert!(EntryStatus::Failed.is_terminal());
        assert!(EntryStatus::Withdrawn.is_terminal());
        assert!(!EntryStatus::UnderReview.is_terminal());
    }

    #[test]
    fn test_entry_status_is_active() {
        assert!(EntryStatus::Ingested.is_active());
        assert!(!EntryStatus::Pending.is_active());
        assert!(!EntryStatus::Withdrawn.is_active());
    }

    #[test]
    fn test_entry_status_labels() {
        assert_eq!(EntryStatus::Pending.label(), "pending");
        assert_eq!(EntryStatus::Ingested.label(), "ingested");
        assert_eq!(EntryStatus::Failed.label(), "failed");
        assert_eq!(EntryStatus::Withdrawn.label(), "withdrawn");
        assert_eq!(EntryStatus::UnderReview.label(), "under_review");
    }

    #[test]
    fn test_manifest_entry_new() {
        let e = ManifestEntry::new("E-001", "My Entry");
        assert_eq!(e.id.as_str(), "E-001");
        assert_eq!(e.title, "My Entry");
        assert_eq!(e.status, EntryStatus::Pending);
        assert_eq!(e.total_bytes, 0);
    }

    #[test]
    fn test_manifest_entry_metadata() {
        let mut e = ManifestEntry::new("E-001", "My Entry");
        e.add_metadata("creator", "test");
        assert_eq!(e.metadata.get("creator").expect("operation should succeed"), "test");
    }

    #[test]
    fn test_collection_manifest_new() {
        let m = CollectionManifest::new("COL-001", "Test Collection");
        assert_eq!(m.collection_id(), "COL-001");
        assert_eq!(m.title(), "Test Collection");
        assert_eq!(m.entry_count(), 0);
    }

    #[test]
    fn test_add_and_get_entry() {
        let mut m = CollectionManifest::new("COL-001", "Test");
        let added = m.add_entry(sample_entry("E-001", "Entry 1", EntryStatus::Pending));
        assert!(added);
        assert_eq!(m.entry_count(), 1);

        let id = ManifestEntryId::new("E-001");
        assert!(m.get_entry(&id).is_some());
    }

    #[test]
    fn test_add_duplicate_entry() {
        let mut m = CollectionManifest::new("COL-001", "Test");
        m.add_entry(sample_entry("E-001", "Entry 1", EntryStatus::Pending));
        let dup = m.add_entry(sample_entry("E-001", "Duplicate", EntryStatus::Ingested));
        assert!(!dup);
        assert_eq!(m.entry_count(), 1);
    }

    #[test]
    fn test_remove_entry() {
        let mut m = CollectionManifest::new("COL-001", "Test");
        m.add_entry(sample_entry("E-001", "Entry 1", EntryStatus::Pending));
        let id = ManifestEntryId::new("E-001");
        let removed = m.remove_entry(&id);
        assert!(removed.is_some());
        assert_eq!(m.entry_count(), 0);
    }

    #[test]
    fn test_total_bytes_and_files() {
        let mut m = CollectionManifest::new("COL-001", "Test");
        m.add_entry(sample_entry("E-001", "A", EntryStatus::Pending));
        m.add_entry(sample_entry("E-002", "B", EntryStatus::Ingested));
        assert_eq!(m.total_bytes(), 2000);
        assert_eq!(m.total_file_count(), 10);
    }

    #[test]
    fn test_entries_by_status() {
        let mut m = CollectionManifest::new("COL-001", "Test");
        m.add_entry(sample_entry("E-001", "A", EntryStatus::Pending));
        m.add_entry(sample_entry("E-002", "B", EntryStatus::Ingested));
        m.add_entry(sample_entry("E-003", "C", EntryStatus::Pending));

        let pending = m.entries_by_status(EntryStatus::Pending);
        assert_eq!(pending.len(), 2);

        let ingested = m.entries_by_status(EntryStatus::Ingested);
        assert_eq!(ingested.len(), 1);
    }

    #[test]
    fn test_status_summary() {
        let mut m = CollectionManifest::new("COL-001", "Test");
        m.add_entry(sample_entry("E-001", "A", EntryStatus::Pending));
        m.add_entry(sample_entry("E-002", "B", EntryStatus::Ingested));
        m.add_entry(sample_entry("E-003", "C", EntryStatus::Pending));

        let summary = m.status_summary();
        assert_eq!(*summary.get("pending").unwrap_or(&0), 2);
        assert_eq!(*summary.get("ingested").unwrap_or(&0), 1);
    }

    #[test]
    fn test_validate_ok() {
        let mut m = CollectionManifest::new("COL-001", "Test");
        m.add_entry(sample_entry("E-001", "A", EntryStatus::Ingested));
        let issues = m.validate();
        assert!(issues.is_empty(), "unexpected: {issues:?}");
    }

    #[test]
    fn test_validate_empty_title() {
        let mut m = CollectionManifest::new("COL-001", "Test");
        m.add_entry(sample_entry("E-001", "", EntryStatus::Pending));
        let issues = m.validate();
        assert!(issues.iter().any(|i| i.contains("empty title")));
    }

    #[test]
    fn test_sorted_entry_ids() {
        let mut m = CollectionManifest::new("COL-001", "Test");
        m.add_entry(sample_entry("C", "c", EntryStatus::Pending));
        m.add_entry(sample_entry("A", "a", EntryStatus::Pending));
        m.add_entry(sample_entry("B", "b", EntryStatus::Pending));

        let ids = m.sorted_entry_ids();
        assert_eq!(ids[0].as_str(), "A");
        assert_eq!(ids[1].as_str(), "B");
        assert_eq!(ids[2].as_str(), "C");
    }
}
