//! IMF Content Version tracking.
//!
//! Implements SMPTE ST 2067-3 content version labels used inside
//! Composition Playlists to identify and compare content revisions.

use std::fmt;
use uuid::Uuid;

/// A content version label as defined in SMPTE ST 2067-3.
///
/// Combines a UUID that is stable across versions with a human-readable label.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContentVersion {
    /// Unique identifier stable across re-deliveries of the same content
    pub id: Uuid,
    /// Human-readable version label (e.g. "2024-01-15T10:00:00Z")
    pub label: String,
}

impl ContentVersion {
    /// Create a new content version with a generated UUID.
    #[allow(dead_code)]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            label: label.into(),
        }
    }

    /// Create a content version with an explicit UUID.
    #[allow(dead_code)]
    pub fn with_id(id: Uuid, label: impl Into<String>) -> Self {
        Self {
            id,
            label: label.into(),
        }
    }

    /// Parse a content version from XML attribute strings.
    ///
    /// `id_str` must be a valid URN UUID: `urn:uuid:<uuid>`.
    #[allow(dead_code)]
    pub fn from_xml(id_str: &str, label: &str) -> Result<Self, String> {
        let uuid_str = id_str
            .strip_prefix("urn:uuid:")
            .ok_or_else(|| format!("Expected urn:uuid: prefix, got: {id_str}"))?;
        let id =
            Uuid::parse_str(uuid_str).map_err(|e| format!("Invalid UUID '{uuid_str}': {e}"))?;
        Ok(Self {
            id,
            label: label.to_owned(),
        })
    }

    /// Serialize to a `urn:uuid:` URN string.
    #[allow(dead_code)]
    pub fn to_urn(&self) -> String {
        format!("urn:uuid:{}", self.id)
    }
}

impl fmt::Display for ContentVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.to_urn(), self.label)
    }
}

/// A list of content versions associated with a CPL.
///
/// The list is ordered; the last entry represents the most recent version.
#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct ContentVersionList {
    versions: Vec<ContentVersion>,
}

impl ContentVersionList {
    /// Create an empty list.
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a content version to the list.
    #[allow(dead_code)]
    pub fn push(&mut self, version: ContentVersion) {
        self.versions.push(version);
    }

    /// Return the most recent version, if any.
    #[allow(dead_code)]
    pub fn latest(&self) -> Option<&ContentVersion> {
        self.versions.last()
    }

    /// Find a version by its UUID.
    #[allow(dead_code)]
    pub fn find_by_id(&self, id: &Uuid) -> Option<&ContentVersion> {
        self.versions.iter().find(|v| &v.id == id)
    }

    /// Return all versions.
    #[allow(dead_code)]
    pub fn versions(&self) -> &[ContentVersion] {
        &self.versions
    }

    /// Number of versions in the list.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.versions.len()
    }

    /// True if no versions have been added.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.versions.is_empty()
    }

    /// Check if any version matches a given UUID.
    #[allow(dead_code)]
    pub fn contains_id(&self, id: &Uuid) -> bool {
        self.versions.iter().any(|v| &v.id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_version_new_generates_uuid() {
        let v1 = ContentVersion::new("v1");
        let v2 = ContentVersion::new("v1");
        assert_ne!(v1.id, v2.id);
        assert_eq!(v1.label, "v1");
    }

    #[test]
    fn test_content_version_with_explicit_id() {
        let id = Uuid::new_v4();
        let v = ContentVersion::with_id(id, "label");
        assert_eq!(v.id, id);
    }

    #[test]
    fn test_to_urn_format() {
        let id =
            Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").expect("id should be valid");
        let v = ContentVersion::with_id(id, "test");
        assert_eq!(v.to_urn(), "urn:uuid:550e8400-e29b-41d4-a716-446655440000");
    }

    #[test]
    fn test_display_format() {
        let id =
            Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").expect("id should be valid");
        let v = ContentVersion::with_id(id, "v2");
        let s = v.to_string();
        assert!(s.contains("urn:uuid:"));
        assert!(s.contains("v2"));
    }

    #[test]
    fn test_from_xml_valid() {
        let result =
            ContentVersion::from_xml("urn:uuid:550e8400-e29b-41d4-a716-446655440000", "version-1");
        assert!(result.is_ok());
        let v = result.expect("v should be valid");
        assert_eq!(v.label, "version-1");
    }

    #[test]
    fn test_from_xml_missing_prefix() {
        let result = ContentVersion::from_xml("550e8400-e29b-41d4-a716-446655440000", "v");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("urn:uuid:"));
    }

    #[test]
    fn test_from_xml_invalid_uuid() {
        let result = ContentVersion::from_xml("urn:uuid:not-a-uuid", "v");
        assert!(result.is_err());
    }

    #[test]
    fn test_content_version_list_empty() {
        let list = ContentVersionList::new();
        assert!(list.is_empty());
        assert!(list.latest().is_none());
    }

    #[test]
    fn test_content_version_list_push_and_latest() {
        let mut list = ContentVersionList::new();
        list.push(ContentVersion::new("v1"));
        list.push(ContentVersion::new("v2"));
        assert_eq!(list.len(), 2);
        assert_eq!(list.latest().expect("latest should succeed").label, "v2");
    }

    #[test]
    fn test_find_by_id() {
        let mut list = ContentVersionList::new();
        let v = ContentVersion::new("findme");
        let id = v.id;
        list.push(v);
        list.push(ContentVersion::new("other"));
        let found = list.find_by_id(&id).expect("found should be valid");
        assert_eq!(found.label, "findme");
    }

    #[test]
    fn test_find_by_id_missing() {
        let list = ContentVersionList::new();
        assert!(list.find_by_id(&Uuid::new_v4()).is_none());
    }

    #[test]
    fn test_contains_id_true_and_false() {
        let mut list = ContentVersionList::new();
        let v = ContentVersion::new("x");
        let id = v.id;
        list.push(v);
        assert!(list.contains_id(&id));
        assert!(!list.contains_id(&Uuid::new_v4()));
    }

    #[test]
    fn test_versions_slice() {
        let mut list = ContentVersionList::new();
        list.push(ContentVersion::new("a"));
        list.push(ContentVersion::new("b"));
        assert_eq!(list.versions().len(), 2);
    }
}
