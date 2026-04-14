//! Archive manifest versioning.
//!
//! Tracks the schema / format version of an archive manifest so that readers
//! can refuse to open incompatible archives and writers can bump the version
//! when the format changes.
//!
//! # Compatibility policy
//!
//! A version `v` is considered **compatible** with another version `other` when
//! `other <= v` (i.e., a newer reader can always open an older manifest, but an
//! older reader should refuse to open a newer manifest).
//!
//! # Example
//! ```rust
//! use oximedia_archive::versioning::ArchiveManifestVersion;
//!
//! let mut ver = ArchiveManifestVersion::new(1);
//! assert!(ver.is_compatible(1));
//! ver.bump();
//! assert_eq!(ver.version(), 2);
//! assert!(!ver.is_compatible(3)); // cannot open a future version
//! ```

#![allow(dead_code)]

/// Version number attached to an archive manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ArchiveManifestVersion {
    version: u32,
}

impl ArchiveManifestVersion {
    /// Create a new manifest version with the given version number.
    ///
    /// `v = 0` is reserved for "unversioned / legacy"; the minimum meaningful
    /// version is `1`.
    #[must_use]
    pub const fn new(v: u32) -> Self {
        Self { version: v }
    }

    /// Return the current version number.
    #[must_use]
    pub const fn version(self) -> u32 {
        self.version
    }

    /// Increment the version number by 1.
    pub fn bump(&mut self) {
        self.version = self.version.saturating_add(1);
    }

    /// Return `true` when `other` is compatible with this version.
    ///
    /// An archive written at version `other` can be read by a reader at version
    /// `self` if and only if `other <= self.version`.  In other words, a reader
    /// must be at least as new as the archive.
    #[must_use]
    pub fn is_compatible(self, other: u32) -> bool {
        other <= self.version
    }

    /// Minimum version number this reader supports (constant `1`).
    #[must_use]
    pub const fn min_supported() -> u32 {
        1
    }
}

impl std::fmt::Display for ArchiveManifestVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "v{}", self.version)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_stores_version() {
        let v = ArchiveManifestVersion::new(3);
        assert_eq!(v.version(), 3);
    }

    #[test]
    fn test_bump_increments() {
        let mut v = ArchiveManifestVersion::new(1);
        v.bump();
        assert_eq!(v.version(), 2);
        v.bump();
        assert_eq!(v.version(), 3);
    }

    #[test]
    fn test_bump_saturation_at_u32_max() {
        let mut v = ArchiveManifestVersion::new(u32::MAX);
        v.bump(); // Must not overflow
        assert_eq!(v.version(), u32::MAX);
    }

    #[test]
    fn test_is_compatible_same_version() {
        let v = ArchiveManifestVersion::new(2);
        assert!(v.is_compatible(2));
    }

    #[test]
    fn test_is_compatible_older_archive() {
        let v = ArchiveManifestVersion::new(5);
        assert!(v.is_compatible(1));
        assert!(v.is_compatible(4));
    }

    #[test]
    fn test_is_compatible_newer_archive_rejected() {
        let v = ArchiveManifestVersion::new(2);
        assert!(!v.is_compatible(3));
        assert!(!v.is_compatible(100));
    }

    #[test]
    fn test_display() {
        let v = ArchiveManifestVersion::new(7);
        assert_eq!(v.to_string(), "v7");
    }

    #[test]
    fn test_ordering() {
        let v1 = ArchiveManifestVersion::new(1);
        let v3 = ArchiveManifestVersion::new(3);
        assert!(v1 < v3);
        assert!(v3 > v1);
    }
}
