#![allow(dead_code)]
//! In-memory object store abstraction — keys, metadata, and basic CRUD operations.

use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// A validated object key (path-like, non-empty, no leading slash).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ObjectKey(String);

impl ObjectKey {
    /// Attempt to create an `ObjectKey` from a raw string.
    ///
    /// Returns `None` if the key is empty, contains a leading `/`, or contains `..`.
    pub fn new(raw: impl Into<String>) -> Option<Self> {
        let s = raw.into();
        if Self::is_valid_str(&s) {
            Some(Self(s))
        } else {
            None
        }
    }

    /// Validate without constructing.
    pub fn is_valid(raw: &str) -> bool {
        Self::is_valid_str(raw)
    }

    fn is_valid_str(s: &str) -> bool {
        if s.is_empty() {
            return false;
        }
        if s.starts_with('/') {
            return false;
        }
        if s.contains("..") {
            return false;
        }
        true
    }

    /// Return the underlying string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Returns true if this key starts with the given prefix.
    pub fn has_prefix(&self, prefix: &str) -> bool {
        self.0.starts_with(prefix)
    }
}

impl std::fmt::Display for ObjectKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Lightweight metadata associated with a stored object.
#[derive(Debug, Clone)]
pub struct ObjectMetadata {
    /// The object's key.
    pub key: ObjectKey,
    /// Size in bytes.
    pub size: u64,
    /// MIME content type.
    pub content_type: Option<String>,
    /// When the object was first created.
    pub created_at: SystemTime,
    /// When the object was last modified.
    pub last_modified: SystemTime,
    /// Arbitrary user-defined tags.
    pub tags: HashMap<String, String>,
    /// ETag / checksum.
    pub etag: Option<String>,
}

impl ObjectMetadata {
    /// Create new metadata for a freshly stored object.
    pub fn new(key: ObjectKey, size: u64) -> Self {
        let now = SystemTime::now();
        Self {
            key,
            size,
            content_type: None,
            created_at: now,
            last_modified: now,
            tags: HashMap::new(),
            etag: None,
        }
    }

    /// Returns the age of this object in seconds (since creation), or 0 on clock error.
    pub fn age_secs(&self, now: SystemTime) -> u64 {
        now.duration_since(self.created_at)
            .unwrap_or(Duration::ZERO)
            .as_secs()
    }

    /// Set the content type.
    pub fn with_content_type(mut self, ct: impl Into<String>) -> Self {
        self.content_type = Some(ct.into());
        self
    }

    /// Add a tag.
    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.insert(key.into(), value.into());
        self
    }

    /// Set the etag.
    pub fn with_etag(mut self, etag: impl Into<String>) -> Self {
        self.etag = Some(etag.into());
        self
    }
}

/// Errors from object store operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObjectStoreError {
    /// The key does not exist.
    NotFound(String),
    /// The key already exists and overwrite is not permitted.
    AlreadyExists(String),
    /// Key validation failed.
    InvalidKey(String),
}

impl std::fmt::Display for ObjectStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(k) => write!(f, "object not found: {k}"),
            Self::AlreadyExists(k) => write!(f, "object already exists: {k}"),
            Self::InvalidKey(k) => write!(f, "invalid object key: {k}"),
        }
    }
}

/// Payload stored alongside metadata.
#[derive(Debug, Clone)]
struct StoredObject {
    meta: ObjectMetadata,
    data: Vec<u8>,
}

/// An in-memory object store suitable for testing and lightweight usage.
#[derive(Debug, Default)]
pub struct ObjectStore {
    objects: HashMap<String, StoredObject>,
}

impl ObjectStore {
    /// Create a new empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Store an object. Overwrites any existing object with the same key.
    pub fn put(
        &mut self,
        key: &str,
        data: Vec<u8>,
        content_type: Option<&str>,
    ) -> Result<ObjectMetadata, ObjectStoreError> {
        let ok =
            ObjectKey::new(key).ok_or_else(|| ObjectStoreError::InvalidKey(key.to_string()))?;

        let size = data.len() as u64;
        let mut meta = ObjectMetadata::new(ok.clone(), size);
        if let Some(ct) = content_type {
            meta.content_type = Some(ct.to_string());
        }

        // Generate a trivial etag from size
        meta.etag = Some(format!("{size:x}"));

        self.objects.insert(
            ok.as_str().to_string(),
            StoredObject {
                meta: meta.clone(),
                data,
            },
        );
        Ok(meta)
    }

    /// Retrieve raw bytes for an object.
    pub fn get(&self, key: &str) -> Result<&[u8], ObjectStoreError> {
        self.objects
            .get(key)
            .map(|o| o.data.as_slice())
            .ok_or_else(|| ObjectStoreError::NotFound(key.to_string()))
    }

    /// Retrieve metadata for an object without fetching the data.
    pub fn get_metadata(&self, key: &str) -> Result<&ObjectMetadata, ObjectStoreError> {
        self.objects
            .get(key)
            .map(|o| &o.meta)
            .ok_or_else(|| ObjectStoreError::NotFound(key.to_string()))
    }

    /// Delete an object. Returns metadata of the deleted object.
    pub fn delete(&mut self, key: &str) -> Result<ObjectMetadata, ObjectStoreError> {
        self.objects
            .remove(key)
            .map(|o| o.meta)
            .ok_or_else(|| ObjectStoreError::NotFound(key.to_string()))
    }

    /// List all object keys that start with `prefix`. Empty prefix returns all keys.
    pub fn list_prefix(&self, prefix: &str) -> Vec<&str> {
        let mut keys: Vec<&str> = self
            .objects
            .keys()
            .filter(|k| prefix.is_empty() || k.starts_with(prefix))
            .map(std::string::String::as_str)
            .collect();
        keys.sort_unstable();
        keys
    }

    /// Returns the total number of stored objects.
    pub fn object_count(&self) -> usize {
        self.objects.len()
    }

    /// Returns true if an object with the given key exists.
    pub fn contains(&self, key: &str) -> bool {
        self.objects.contains_key(key)
    }

    /// Total bytes across all stored objects.
    pub fn total_bytes(&self) -> u64 {
        self.objects.values().map(|o| o.meta.size).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, SystemTime};

    #[test]
    fn test_object_key_valid() {
        assert!(ObjectKey::is_valid("videos/clip.mp4"));
        assert!(ObjectKey::is_valid("a"));
    }

    #[test]
    fn test_object_key_invalid_empty() {
        assert!(!ObjectKey::is_valid(""));
    }

    #[test]
    fn test_object_key_invalid_leading_slash() {
        assert!(!ObjectKey::is_valid("/bad/key"));
    }

    #[test]
    fn test_object_key_invalid_dotdot() {
        assert!(!ObjectKey::is_valid("a/../b"));
    }

    #[test]
    fn test_object_key_has_prefix() {
        let key = ObjectKey::new("videos/clip.mp4").expect("valid object key");
        assert!(key.has_prefix("videos/"));
        assert!(!key.has_prefix("audio/"));
    }

    #[test]
    fn test_object_metadata_age_secs() {
        let key = ObjectKey::new("test/obj").expect("valid object key");
        let mut meta = ObjectMetadata::new(key, 1024);
        // Backdate created_at by 10 seconds
        meta.created_at = SystemTime::now() - Duration::from_secs(10);
        let age = meta.age_secs(SystemTime::now());
        assert!(age >= 10, "expected age >= 10, got {age}");
    }

    #[test]
    fn test_object_metadata_zero_age() {
        let key = ObjectKey::new("test/obj2").expect("valid object key");
        let meta = ObjectMetadata::new(key, 512);
        // Should be nearly zero age
        let age = meta.age_secs(SystemTime::now());
        assert!(age < 5);
    }

    #[test]
    fn test_object_store_put_and_get() {
        let mut store = ObjectStore::new();
        let meta = store
            .put("videos/a.mp4", b"hello".to_vec(), Some("video/mp4"))
            .expect("should succeed");
        assert_eq!(meta.size, 5);
        assert_eq!(meta.content_type.as_deref(), Some("video/mp4"));
        assert_eq!(
            store.get("videos/a.mp4").expect("get should succeed"),
            b"hello"
        );
    }

    #[test]
    fn test_object_store_get_not_found() {
        let store = ObjectStore::new();
        let err = store.get("missing/key").unwrap_err();
        assert_eq!(err, ObjectStoreError::NotFound("missing/key".to_string()));
    }

    #[test]
    fn test_object_store_get_metadata() {
        let mut store = ObjectStore::new();
        store
            .put("img/photo.jpg", vec![1, 2, 3], Some("image/jpeg"))
            .expect("should succeed");
        let meta = store
            .get_metadata("img/photo.jpg")
            .expect("get metadata should succeed");
        assert_eq!(meta.size, 3);
    }

    #[test]
    fn test_object_store_delete() {
        let mut store = ObjectStore::new();
        store
            .put("tmp/file", vec![0u8; 64], None)
            .expect("put should succeed");
        assert!(store.contains("tmp/file"));
        store.delete("tmp/file").expect("delete should succeed");
        assert!(!store.contains("tmp/file"));
    }

    #[test]
    fn test_object_store_delete_not_found() {
        let mut store = ObjectStore::new();
        let err = store.delete("ghost/key").unwrap_err();
        assert_eq!(err, ObjectStoreError::NotFound("ghost/key".to_string()));
    }

    #[test]
    fn test_object_store_list_prefix() {
        let mut store = ObjectStore::new();
        store
            .put("a/1.mp4", vec![], None)
            .expect("put should succeed");
        store
            .put("a/2.mp4", vec![], None)
            .expect("put should succeed");
        store
            .put("b/3.mp4", vec![], None)
            .expect("put should succeed");

        let a_keys = store.list_prefix("a/");
        assert_eq!(a_keys.len(), 2);

        let all_keys = store.list_prefix("");
        assert_eq!(all_keys.len(), 3);
    }

    #[test]
    fn test_object_store_total_bytes() {
        let mut store = ObjectStore::new();
        store
            .put("f1", vec![0u8; 100], None)
            .expect("put should succeed");
        store
            .put("f2", vec![0u8; 200], None)
            .expect("put should succeed");
        assert_eq!(store.total_bytes(), 300);
    }

    #[test]
    fn test_object_store_invalid_key_rejected() {
        let mut store = ObjectStore::new();
        let err = store.put("/bad/key", vec![], None).unwrap_err();
        assert_eq!(err, ObjectStoreError::InvalidKey("/bad/key".to_string()));
    }
}
