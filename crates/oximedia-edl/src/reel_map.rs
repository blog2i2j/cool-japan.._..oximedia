//! Reel/tape name mapping for EDL conform workflows.
//!
//! Maps reel identifiers to physical tape names, volume labels and
//! file-system paths for online/offline media management.

#![allow(dead_code)]

/// A single reel mapping entry.
#[derive(Debug, Clone)]
pub struct ReelEntry {
    /// Short reel identifier (e.g. "A001").
    pub reel_id: String,
    /// Human-readable tape name (e.g. "TAPE_001").
    pub tape_name: String,
    /// Volume label of the storage medium.
    pub volume_label: String,
    /// Optional filesystem path to the reel's media directory.
    pub path: Option<String>,
}

impl ReelEntry {
    /// Create a new reel entry.
    #[must_use]
    pub fn new(
        reel_id: impl Into<String>,
        tape_name: impl Into<String>,
        volume_label: impl Into<String>,
        path: Option<String>,
    ) -> Self {
        Self {
            reel_id: reel_id.into(),
            tape_name: tape_name.into(),
            volume_label: volume_label.into(),
            path,
        }
    }

    /// Returns true if this reel has an online (accessible) path.
    #[must_use]
    pub fn is_online(&self) -> bool {
        self.path.is_some()
    }

    /// Returns the path if the reel is online.
    #[must_use]
    pub fn media_path(&self) -> Option<&str> {
        self.path.as_deref()
    }
}

/// A map of reel identifiers to their physical tape information.
#[derive(Debug, Default)]
pub struct ReelMap {
    /// All reel entries.
    pub entries: Vec<ReelEntry>,
}

impl ReelMap {
    /// Create a new empty reel map.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Add a reel entry to the map.
    ///
    /// If an entry with the same reel_id already exists it is replaced.
    pub fn add(&mut self, entry: ReelEntry) {
        if let Some(existing) = self.entries.iter_mut().find(|e| e.reel_id == entry.reel_id) {
            *existing = entry;
        } else {
            self.entries.push(entry);
        }
    }

    /// Find a reel entry by its reel identifier.
    #[must_use]
    pub fn find_by_id(&self, reel_id: &str) -> Option<&ReelEntry> {
        self.entries.iter().find(|e| e.reel_id == reel_id)
    }

    /// Find a reel entry by its tape name.
    #[must_use]
    pub fn find_by_tape(&self, tape_name: &str) -> Option<&ReelEntry> {
        self.entries.iter().find(|e| e.tape_name == tape_name)
    }

    /// Resolve the filesystem path for the given reel identifier.
    ///
    /// Returns `None` if the reel is not found or is offline.
    #[must_use]
    pub fn resolve_path(&self, reel_id: &str) -> Option<&str> {
        self.find_by_id(reel_id).and_then(|e| e.media_path())
    }

    /// Returns the number of online reels.
    #[must_use]
    pub fn online_count(&self) -> usize {
        self.entries.iter().filter(|e| e.is_online()).count()
    }

    /// Returns the total number of reel entries.
    #[must_use]
    pub fn total_count(&self) -> usize {
        self.entries.len()
    }

    /// Returns the number of offline reels.
    #[must_use]
    pub fn offline_count(&self) -> usize {
        self.entries.iter().filter(|e| !e.is_online()).count()
    }

    /// Returns all online reel entries.
    #[must_use]
    pub fn online_reels(&self) -> Vec<&ReelEntry> {
        self.entries.iter().filter(|e| e.is_online()).collect()
    }

    /// Returns all offline reel entries.
    #[must_use]
    pub fn offline_reels(&self) -> Vec<&ReelEntry> {
        self.entries.iter().filter(|e| !e.is_online()).collect()
    }

    /// Returns true if all reels are online.
    #[must_use]
    pub fn all_online(&self) -> bool {
        !self.entries.is_empty() && self.entries.iter().all(|e| e.is_online())
    }

    /// Remove a reel entry by its reel identifier.
    ///
    /// Returns true if an entry was removed.
    pub fn remove(&mut self, reel_id: &str) -> bool {
        let before = self.entries.len();
        self.entries.retain(|e| e.reel_id != reel_id);
        self.entries.len() < before
    }

    /// Set the path for a reel, making it online.
    ///
    /// Returns true if the reel was found and updated.
    pub fn set_path(&mut self, reel_id: &str, path: impl Into<String>) -> bool {
        if let Some(entry) = self.entries.iter_mut().find(|e| e.reel_id == reel_id) {
            entry.path = Some(path.into());
            true
        } else {
            false
        }
    }

    /// Mark a reel as offline by clearing its path.
    ///
    /// Returns true if the reel was found and updated.
    pub fn set_offline(&mut self, reel_id: &str) -> bool {
        if let Some(entry) = self.entries.iter_mut().find(|e| e.reel_id == reel_id) {
            entry.path = None;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_online(id: &str, tape: &str) -> ReelEntry {
        ReelEntry::new(id, tape, "VOL_01", Some(format!("/media/{}", id)))
    }

    fn make_offline(id: &str, tape: &str) -> ReelEntry {
        ReelEntry::new(id, tape, "VOL_02", None)
    }

    #[test]
    fn test_reel_entry_is_online_with_path() {
        let entry = make_online("A001", "TAPE_001");
        assert!(entry.is_online());
        assert_eq!(entry.media_path(), Some("/media/A001"));
    }

    #[test]
    fn test_reel_entry_is_offline_without_path() {
        let entry = make_offline("B001", "TAPE_002");
        assert!(!entry.is_online());
        assert!(entry.media_path().is_none());
    }

    #[test]
    fn test_reel_map_add_and_total_count() {
        let mut map = ReelMap::new();
        assert_eq!(map.total_count(), 0);

        map.add(make_online("A001", "TAPE_001"));
        map.add(make_offline("A002", "TAPE_002"));

        assert_eq!(map.total_count(), 2);
    }

    #[test]
    fn test_reel_map_add_replaces_existing() {
        let mut map = ReelMap::new();
        map.add(make_offline("A001", "TAPE_001"));
        assert_eq!(map.online_count(), 0);

        // Replace with an online entry
        map.add(make_online("A001", "TAPE_001"));
        assert_eq!(map.total_count(), 1);
        assert_eq!(map.online_count(), 1);
    }

    #[test]
    fn test_reel_map_find_by_id() {
        let mut map = ReelMap::new();
        map.add(make_online("A001", "TAPE_001"));
        map.add(make_online("A002", "TAPE_002"));

        let entry = map.find_by_id("A001");
        assert!(entry.is_some());
        assert_eq!(entry.expect("entry should be valid").tape_name, "TAPE_001");

        assert!(map.find_by_id("UNKNOWN").is_none());
    }

    #[test]
    fn test_reel_map_find_by_tape() {
        let mut map = ReelMap::new();
        map.add(make_online("A001", "TAPE_001"));
        map.add(make_online("A002", "TAPE_002"));

        let entry = map.find_by_tape("TAPE_002");
        assert!(entry.is_some());
        assert_eq!(entry.expect("entry should be valid").reel_id, "A002");

        assert!(map.find_by_tape("NONEXISTENT").is_none());
    }

    #[test]
    fn test_reel_map_resolve_path() {
        let mut map = ReelMap::new();
        map.add(make_online("A001", "TAPE_001"));
        map.add(make_offline("A002", "TAPE_002"));

        assert_eq!(map.resolve_path("A001"), Some("/media/A001"));
        assert_eq!(map.resolve_path("A002"), None);
        assert_eq!(map.resolve_path("UNKNOWN"), None);
    }

    #[test]
    fn test_reel_map_online_count() {
        let mut map = ReelMap::new();
        map.add(make_online("A001", "TAPE_001"));
        map.add(make_online("A002", "TAPE_002"));
        map.add(make_offline("A003", "TAPE_003"));

        assert_eq!(map.online_count(), 2);
        assert_eq!(map.offline_count(), 1);
    }

    #[test]
    fn test_reel_map_online_reels() {
        let mut map = ReelMap::new();
        map.add(make_online("A001", "TAPE_001"));
        map.add(make_offline("A002", "TAPE_002"));

        let online = map.online_reels();
        assert_eq!(online.len(), 1);
        assert_eq!(online[0].reel_id, "A001");
    }

    #[test]
    fn test_reel_map_offline_reels() {
        let mut map = ReelMap::new();
        map.add(make_online("A001", "TAPE_001"));
        map.add(make_offline("A002", "TAPE_002"));
        map.add(make_offline("A003", "TAPE_003"));

        let offline = map.offline_reels();
        assert_eq!(offline.len(), 2);
    }

    #[test]
    fn test_reel_map_all_online() {
        let mut map = ReelMap::new();
        map.add(make_online("A001", "TAPE_001"));
        map.add(make_online("A002", "TAPE_002"));
        assert!(map.all_online());

        map.add(make_offline("A003", "TAPE_003"));
        assert!(!map.all_online());
    }

    #[test]
    fn test_reel_map_remove() {
        let mut map = ReelMap::new();
        map.add(make_online("A001", "TAPE_001"));
        map.add(make_online("A002", "TAPE_002"));

        let removed = map.remove("A001");
        assert!(removed);
        assert_eq!(map.total_count(), 1);

        let not_removed = map.remove("UNKNOWN");
        assert!(!not_removed);
    }

    #[test]
    fn test_reel_map_set_path() {
        let mut map = ReelMap::new();
        map.add(make_offline("A001", "TAPE_001"));
        assert_eq!(map.online_count(), 0);

        let updated = map.set_path("A001", "/media/new/A001");
        assert!(updated);
        assert_eq!(map.online_count(), 1);
        assert_eq!(map.resolve_path("A001"), Some("/media/new/A001"));
    }

    #[test]
    fn test_reel_map_set_offline() {
        let mut map = ReelMap::new();
        map.add(make_online("A001", "TAPE_001"));
        assert_eq!(map.online_count(), 1);

        let updated = map.set_offline("A001");
        assert!(updated);
        assert_eq!(map.online_count(), 0);
        assert!(map.resolve_path("A001").is_none());
    }
}
