#![allow(dead_code)]
//! Preset version tracking, migration, and compatibility management.
//!
//! Provides tools for tracking preset schema versions, migrating presets
//! between format revisions, and checking cross-version compatibility.

use std::collections::HashMap;
use std::fmt;

/// Semantic version representation for preset schemas.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PresetVersion {
    /// Major version (breaking changes).
    pub major: u32,
    /// Minor version (backward-compatible additions).
    pub minor: u32,
    /// Patch version (backward-compatible fixes).
    pub patch: u32,
}

impl PresetVersion {
    /// Create a new preset version.
    #[must_use]
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Parse a version string of the form "major.minor.patch".
    pub fn parse(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 3 {
            return None;
        }
        let major = parts[0].parse().ok()?;
        let minor = parts[1].parse().ok()?;
        let patch = parts[2].parse().ok()?;
        Some(Self {
            major,
            minor,
            patch,
        })
    }

    /// Check if this version is compatible with another (same major version).
    #[must_use]
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        self.major == other.major
    }

    /// Check if this version is newer than another.
    #[must_use]
    pub fn is_newer_than(&self, other: &Self) -> bool {
        if self.major != other.major {
            return self.major > other.major;
        }
        if self.minor != other.minor {
            return self.minor > other.minor;
        }
        self.patch > other.patch
    }

    /// Return the next major version.
    #[must_use]
    pub fn next_major(&self) -> Self {
        Self {
            major: self.major + 1,
            minor: 0,
            patch: 0,
        }
    }

    /// Return the next minor version.
    #[must_use]
    pub fn next_minor(&self) -> Self {
        Self {
            major: self.major,
            minor: self.minor + 1,
            patch: 0,
        }
    }

    /// Return the next patch version.
    #[must_use]
    pub fn next_patch(&self) -> Self {
        Self {
            major: self.major,
            minor: self.minor,
            patch: self.patch + 1,
        }
    }

    /// Convert to a tuple for comparison purposes.
    #[must_use]
    pub fn as_tuple(&self) -> (u32, u32, u32) {
        (self.major, self.minor, self.patch)
    }
}

impl fmt::Display for PresetVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl PartialOrd for PresetVersion {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PresetVersion {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.as_tuple().cmp(&other.as_tuple())
    }
}

/// Describes a single migration step between two versions.
#[derive(Debug, Clone)]
pub struct MigrationStep {
    /// Source version.
    pub from: PresetVersion,
    /// Target version.
    pub to: PresetVersion,
    /// Human-readable description of changes.
    pub description: String,
    /// Fields added in this migration.
    pub added_fields: Vec<String>,
    /// Fields removed in this migration.
    pub removed_fields: Vec<String>,
    /// Fields renamed (old_name -> new_name).
    pub renamed_fields: Vec<(String, String)>,
}

impl MigrationStep {
    /// Create a new migration step.
    #[must_use]
    pub fn new(from: PresetVersion, to: PresetVersion, description: &str) -> Self {
        Self {
            from,
            to,
            description: description.to_string(),
            added_fields: Vec::new(),
            removed_fields: Vec::new(),
            renamed_fields: Vec::new(),
        }
    }

    /// Add an added field to this migration.
    #[must_use]
    pub fn with_added_field(mut self, field: &str) -> Self {
        self.added_fields.push(field.to_string());
        self
    }

    /// Add a removed field to this migration.
    #[must_use]
    pub fn with_removed_field(mut self, field: &str) -> Self {
        self.removed_fields.push(field.to_string());
        self
    }

    /// Add a renamed field to this migration.
    #[must_use]
    pub fn with_renamed_field(mut self, old: &str, new: &str) -> Self {
        self.renamed_fields.push((old.to_string(), new.to_string()));
        self
    }

    /// Check if this migration is a breaking change (crosses major version).
    #[must_use]
    pub fn is_breaking(&self) -> bool {
        self.from.major != self.to.major
    }

    /// Count the total number of field changes.
    #[must_use]
    pub fn change_count(&self) -> usize {
        self.added_fields.len() + self.removed_fields.len() + self.renamed_fields.len()
    }
}

/// Registry of migration steps for building upgrade paths.
pub struct MigrationRegistry {
    /// All registered migration steps.
    steps: Vec<MigrationStep>,
}

impl MigrationRegistry {
    /// Create a new empty migration registry.
    #[must_use]
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Register a migration step.
    pub fn register(&mut self, step: MigrationStep) {
        self.steps.push(step);
    }

    /// Find a direct migration from one version to another.
    #[must_use]
    pub fn find_direct(&self, from: &PresetVersion, to: &PresetVersion) -> Option<&MigrationStep> {
        self.steps.iter().find(|s| &s.from == from && &s.to == to)
    }

    /// Build an ordered migration path from `from` to `to`.
    ///
    /// Returns `None` if no path exists.
    #[must_use]
    pub fn find_path(
        &self,
        from: &PresetVersion,
        to: &PresetVersion,
    ) -> Option<Vec<&MigrationStep>> {
        if from == to {
            return Some(Vec::new());
        }

        // BFS over versions
        let mut visited: HashMap<PresetVersion, Option<usize>> = HashMap::new();
        let mut queue = std::collections::VecDeque::new();
        visited.insert(from.clone(), None);
        queue.push_back(from.clone());

        while let Some(current) = queue.pop_front() {
            if &current == to {
                break;
            }
            for (idx, step) in self.steps.iter().enumerate() {
                if step.from == current && !visited.contains_key(&step.to) {
                    visited.insert(step.to.clone(), Some(idx));
                    queue.push_back(step.to.clone());
                }
            }
        }

        if !visited.contains_key(to) {
            return None;
        }

        // Reconstruct path
        let mut path = Vec::new();
        let mut current = to.clone();
        while &current != from {
            let idx = visited.get(&current)?.as_ref()?;
            path.push(&self.steps[*idx]);
            current = self.steps[*idx].from.clone();
        }
        path.reverse();
        Some(path)
    }

    /// Get the number of registered migrations.
    #[must_use]
    pub fn count(&self) -> usize {
        self.steps.len()
    }

    /// List all registered source versions.
    #[must_use]
    pub fn source_versions(&self) -> Vec<&PresetVersion> {
        let mut versions: Vec<&PresetVersion> = self.steps.iter().map(|s| &s.from).collect();
        versions.sort();
        versions.dedup();
        versions
    }

    /// List all registered target versions.
    #[must_use]
    pub fn target_versions(&self) -> Vec<&PresetVersion> {
        let mut versions: Vec<&PresetVersion> = self.steps.iter().map(|s| &s.to).collect();
        versions.sort();
        versions.dedup();
        versions
    }
}

impl Default for MigrationRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Compatibility check result between two preset versions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompatibilityStatus {
    /// Fully compatible, no migration needed.
    FullyCompatible,
    /// Compatible with minor adjustments.
    BackwardCompatible,
    /// Migration required but possible.
    MigrationRequired,
    /// Incompatible versions.
    Incompatible,
}

impl CompatibilityStatus {
    /// Check if the status allows usage without migration.
    #[must_use]
    pub fn is_usable(&self) -> bool {
        matches!(
            self,
            CompatibilityStatus::FullyCompatible | CompatibilityStatus::BackwardCompatible
        )
    }
}

/// Check compatibility between two preset versions.
#[must_use]
pub fn check_compatibility(current: &PresetVersion, target: &PresetVersion) -> CompatibilityStatus {
    if current == target {
        return CompatibilityStatus::FullyCompatible;
    }
    if current.major == target.major && current.minor == target.minor {
        return CompatibilityStatus::FullyCompatible;
    }
    if current.major == target.major {
        return CompatibilityStatus::BackwardCompatible;
    }
    if current.major + 1 == target.major || target.major + 1 == current.major {
        return CompatibilityStatus::MigrationRequired;
    }
    CompatibilityStatus::Incompatible
}

/// Version history entry for tracking a preset's evolution.
#[derive(Debug, Clone)]
pub struct VersionHistoryEntry {
    /// The version at this point.
    pub version: PresetVersion,
    /// Timestamp as ISO 8601 string.
    pub timestamp: String,
    /// Author of this version change.
    pub author: String,
    /// Description of changes.
    pub changelog: String,
}

impl VersionHistoryEntry {
    /// Create a new version history entry.
    #[must_use]
    pub fn new(version: PresetVersion, author: &str, changelog: &str) -> Self {
        Self {
            version,
            timestamp: String::new(),
            author: author.to_string(),
            changelog: changelog.to_string(),
        }
    }

    /// Set the timestamp.
    #[must_use]
    pub fn with_timestamp(mut self, ts: &str) -> Self {
        self.timestamp = ts.to_string();
        self
    }
}

/// Tracks the version history for a single preset.
pub struct VersionHistory {
    /// Preset identifier.
    pub preset_id: String,
    /// Ordered list of version entries (oldest first).
    entries: Vec<VersionHistoryEntry>,
}

impl VersionHistory {
    /// Create a new version history for a preset.
    #[must_use]
    pub fn new(preset_id: &str) -> Self {
        Self {
            preset_id: preset_id.to_string(),
            entries: Vec::new(),
        }
    }

    /// Add a new version entry.
    pub fn add_entry(&mut self, entry: VersionHistoryEntry) {
        self.entries.push(entry);
    }

    /// Get the latest version, if any.
    #[must_use]
    pub fn latest(&self) -> Option<&VersionHistoryEntry> {
        self.entries.last()
    }

    /// Get the first version, if any.
    #[must_use]
    pub fn first(&self) -> Option<&VersionHistoryEntry> {
        self.entries.first()
    }

    /// Get all entries.
    #[must_use]
    pub fn entries(&self) -> &[VersionHistoryEntry] {
        &self.entries
    }

    /// Get the number of version entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the history is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_parse() {
        let v = PresetVersion::parse("1.2.3").expect("v should be valid");
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 3);
    }

    #[test]
    fn test_version_parse_invalid() {
        assert!(PresetVersion::parse("1.2").is_none());
        assert!(PresetVersion::parse("abc").is_none());
        assert!(PresetVersion::parse("1.2.3.4").is_none());
    }

    #[test]
    fn test_version_display() {
        let v = PresetVersion::new(2, 5, 1);
        assert_eq!(v.to_string(), "2.5.1");
    }

    #[test]
    fn test_version_compatibility() {
        let v1 = PresetVersion::new(1, 0, 0);
        let v2 = PresetVersion::new(1, 5, 0);
        assert!(v1.is_compatible_with(&v2));

        let v3 = PresetVersion::new(2, 0, 0);
        assert!(!v1.is_compatible_with(&v3));
    }

    #[test]
    fn test_version_ordering() {
        let v1 = PresetVersion::new(1, 0, 0);
        let v2 = PresetVersion::new(1, 1, 0);
        let v3 = PresetVersion::new(2, 0, 0);
        assert!(v1 < v2);
        assert!(v2 < v3);
        assert!(v1.is_newer_than(&PresetVersion::new(0, 9, 9)));
    }

    #[test]
    fn test_version_next() {
        let v = PresetVersion::new(1, 2, 3);
        assert_eq!(v.next_major(), PresetVersion::new(2, 0, 0));
        assert_eq!(v.next_minor(), PresetVersion::new(1, 3, 0));
        assert_eq!(v.next_patch(), PresetVersion::new(1, 2, 4));
    }

    #[test]
    fn test_migration_step() {
        let step = MigrationStep::new(
            PresetVersion::new(1, 0, 0),
            PresetVersion::new(1, 1, 0),
            "Add HDR fields",
        )
        .with_added_field("hdr_mode")
        .with_renamed_field("bitrate", "video_bitrate");

        assert!(!step.is_breaking());
        assert_eq!(step.change_count(), 2);
    }

    #[test]
    fn test_migration_step_breaking() {
        let step = MigrationStep::new(
            PresetVersion::new(1, 5, 0),
            PresetVersion::new(2, 0, 0),
            "Major schema rewrite",
        )
        .with_removed_field("legacy_mode");

        assert!(step.is_breaking());
        assert_eq!(step.change_count(), 1);
    }

    #[test]
    fn test_migration_registry_direct() {
        let mut reg = MigrationRegistry::new();
        reg.register(MigrationStep::new(
            PresetVersion::new(1, 0, 0),
            PresetVersion::new(1, 1, 0),
            "Minor update",
        ));
        assert_eq!(reg.count(), 1);

        let found = reg.find_direct(&PresetVersion::new(1, 0, 0), &PresetVersion::new(1, 1, 0));
        assert!(found.is_some());
    }

    #[test]
    fn test_migration_registry_path() {
        let mut reg = MigrationRegistry::new();
        reg.register(MigrationStep::new(
            PresetVersion::new(1, 0, 0),
            PresetVersion::new(1, 1, 0),
            "Step 1",
        ));
        reg.register(MigrationStep::new(
            PresetVersion::new(1, 1, 0),
            PresetVersion::new(1, 2, 0),
            "Step 2",
        ));

        let path = reg
            .find_path(&PresetVersion::new(1, 0, 0), &PresetVersion::new(1, 2, 0))
            .expect("test expectation failed");
        assert_eq!(path.len(), 2);
    }

    #[test]
    fn test_migration_registry_no_path() {
        let reg = MigrationRegistry::new();
        let path = reg.find_path(&PresetVersion::new(1, 0, 0), &PresetVersion::new(2, 0, 0));
        assert!(path.is_none());
    }

    #[test]
    fn test_compatibility_check() {
        let v1 = PresetVersion::new(1, 0, 0);
        assert_eq!(
            check_compatibility(&v1, &PresetVersion::new(1, 0, 0)),
            CompatibilityStatus::FullyCompatible
        );
        assert_eq!(
            check_compatibility(&v1, &PresetVersion::new(1, 0, 5)),
            CompatibilityStatus::FullyCompatible
        );
        assert_eq!(
            check_compatibility(&v1, &PresetVersion::new(1, 3, 0)),
            CompatibilityStatus::BackwardCompatible
        );
        assert_eq!(
            check_compatibility(&v1, &PresetVersion::new(2, 0, 0)),
            CompatibilityStatus::MigrationRequired
        );
        assert_eq!(
            check_compatibility(&v1, &PresetVersion::new(5, 0, 0)),
            CompatibilityStatus::Incompatible
        );
    }

    #[test]
    fn test_compatibility_usable() {
        assert!(CompatibilityStatus::FullyCompatible.is_usable());
        assert!(CompatibilityStatus::BackwardCompatible.is_usable());
        assert!(!CompatibilityStatus::MigrationRequired.is_usable());
        assert!(!CompatibilityStatus::Incompatible.is_usable());
    }

    #[test]
    fn test_version_history() {
        let mut history = VersionHistory::new("my-preset");
        assert!(history.is_empty());

        history.add_entry(
            VersionHistoryEntry::new(PresetVersion::new(1, 0, 0), "admin", "Initial release")
                .with_timestamp("2024-01-01T00:00:00Z"),
        );
        history.add_entry(VersionHistoryEntry::new(
            PresetVersion::new(1, 1, 0),
            "admin",
            "Add HDR",
        ));

        assert_eq!(history.len(), 2);
        assert_eq!(
            history.first().expect("first should succeed").version,
            PresetVersion::new(1, 0, 0)
        );
        assert_eq!(
            history.latest().expect("latest should succeed").version,
            PresetVersion::new(1, 1, 0)
        );
    }

    #[test]
    fn test_migration_registry_versions_listing() {
        let mut reg = MigrationRegistry::new();
        reg.register(MigrationStep::new(
            PresetVersion::new(1, 0, 0),
            PresetVersion::new(1, 1, 0),
            "Step A",
        ));
        reg.register(MigrationStep::new(
            PresetVersion::new(1, 1, 0),
            PresetVersion::new(2, 0, 0),
            "Step B",
        ));

        let sources = reg.source_versions();
        assert_eq!(sources.len(), 2);
        let targets = reg.target_versions();
        assert_eq!(targets.len(), 2);
    }
}
