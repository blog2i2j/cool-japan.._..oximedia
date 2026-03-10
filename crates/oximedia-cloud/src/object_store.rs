//! Cloud object storage abstraction
//!
//! Provides a simple in-memory object store abstraction for cloud storage,
//! independent of provider-specific implementations:
//! - Storage class classification with latency/cost metadata
//! - Object key parsing (bucket, key, extension, filename)
//! - Object metadata tracking
//! - Basic CRUD operations on an in-memory store

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// The storage class (tier) assigned to a cloud object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObjectStorageClass {
    /// Frequently accessed objects — lowest latency, highest cost
    Standard,
    /// Automatically moves objects between access tiers
    IntelligentTiering,
    /// Archival storage with minutes-level retrieval
    Glacier,
    /// Long-term archival with hours-level retrieval
    Archive,
    /// Lowest-cost archival; retrieval takes 12–48 h
    DeepArchive,
}

impl ObjectStorageClass {
    /// Returns the typical retrieval latency for this storage class in milliseconds.
    #[must_use]
    pub fn retrieval_latency_ms(&self) -> u32 {
        match self {
            ObjectStorageClass::Standard => 10,
            ObjectStorageClass::IntelligentTiering => 10,
            ObjectStorageClass::Glacier => 300_000, // 5 min (expedited)
            ObjectStorageClass::Archive => 3_600_000, // 1 h
            ObjectStorageClass::DeepArchive => 43_200_000, // 12 h
        }
    }

    /// Returns the approximate cost per GB per month in USD for this storage class.
    #[must_use]
    pub fn cost_per_gb(&self) -> f32 {
        match self {
            ObjectStorageClass::Standard => 0.023,
            ObjectStorageClass::IntelligentTiering => 0.023,
            ObjectStorageClass::Glacier => 0.004,
            ObjectStorageClass::Archive => 0.002,
            ObjectStorageClass::DeepArchive => 0.001,
        }
    }
}

/// A fully-qualified reference to a cloud object.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ObjectKey {
    /// The bucket (container) name
    pub bucket: String,
    /// The object key (path within the bucket)
    pub key: String,
}

impl ObjectKey {
    /// Construct a new `ObjectKey`.
    pub fn new(bucket: impl Into<String>, key: impl Into<String>) -> Self {
        Self {
            bucket: bucket.into(),
            key: key.into(),
        }
    }

    /// Returns the canonical `bucket/key` representation.
    #[must_use]
    pub fn full_path(&self) -> String {
        format!("{}/{}", self.bucket, self.key)
    }

    /// Returns the file extension of the object key, if any.
    ///
    /// The extension is the portion after the final `.` in the filename segment.
    #[must_use]
    pub fn extension(&self) -> Option<String> {
        let filename = self.filename();
        let dot_pos = filename.rfind('.')?;
        let ext = &filename[dot_pos + 1..];
        if ext.is_empty() {
            None
        } else {
            Some(ext.to_string())
        }
    }

    /// Returns the final path segment (filename) of the object key.
    #[must_use]
    pub fn filename(&self) -> &str {
        self.key.rsplit('/').next().unwrap_or(self.key.as_str())
    }
}

/// Metadata describing a single cloud object.
#[derive(Debug, Clone)]
pub struct CloudObjectMetadata {
    /// Fully-qualified object key
    pub key: ObjectKey,
    /// Size of the object in bytes
    pub size_bytes: u64,
    /// MIME content type (e.g. `"video/mp4"`)
    pub content_type: String,
    /// Provider-assigned entity tag (used for change detection)
    pub etag: String,
    /// Unix epoch timestamp (seconds) of the last modification
    pub last_modified_epoch: u64,
    /// Assigned storage class
    pub storage_class: ObjectStorageClass,
}

impl CloudObjectMetadata {
    /// Returns `true` if the content type indicates a media file.
    #[must_use]
    pub fn is_media(&self) -> bool {
        self.content_type.starts_with("video/")
            || self.content_type.starts_with("audio/")
            || self.content_type.starts_with("image/")
    }

    /// Returns the age of the object in fractional days given a `now` epoch.
    #[must_use]
    pub fn age_days(&self, now: u64) -> f32 {
        let secs = now.saturating_sub(self.last_modified_epoch);
        secs as f32 / 86_400.0
    }
}

/// A simple in-memory cloud object store.
#[derive(Debug, Default)]
pub struct ObjectStore {
    /// All stored objects
    pub objects: Vec<CloudObjectMetadata>,
}

impl ObjectStore {
    /// Create a new, empty object store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace an object in the store.
    ///
    /// If an object with the same key already exists it is replaced.
    pub fn put(&mut self, obj: CloudObjectMetadata) {
        if let Some(existing) = self.objects.iter_mut().find(|o| o.key == obj.key) {
            *existing = obj;
        } else {
            self.objects.push(obj);
        }
    }

    /// Retrieve a reference to an object by its key.
    #[must_use]
    pub fn get(&self, key: &ObjectKey) -> Option<&CloudObjectMetadata> {
        self.objects.iter().find(|o| &o.key == key)
    }

    /// List all objects whose key starts with `prefix` (matched against `bucket/key`).
    #[must_use]
    pub fn list(&self, prefix: &str) -> Vec<&CloudObjectMetadata> {
        self.objects
            .iter()
            .filter(|o| o.key.full_path().starts_with(prefix))
            .collect()
    }

    /// Remove the object identified by `key` and return `true` if it existed.
    pub fn delete(&mut self, key: &ObjectKey) -> bool {
        let before = self.objects.len();
        self.objects.retain(|o| &o.key != key);
        self.objects.len() < before
    }

    /// Returns the sum of all object sizes in bytes.
    #[must_use]
    pub fn total_size_bytes(&self) -> u64 {
        self.objects.iter().map(|o| o.size_bytes).sum()
    }

    /// Returns the total number of objects in the store.
    #[must_use]
    pub fn count(&self) -> usize {
        self.objects.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_obj(bucket: &str, key: &str, content_type: &str, size: u64) -> CloudObjectMetadata {
        CloudObjectMetadata {
            key: ObjectKey::new(bucket, key),
            size_bytes: size,
            content_type: content_type.to_string(),
            etag: "abc123".to_string(),
            last_modified_epoch: 1_000_000,
            storage_class: ObjectStorageClass::Standard,
        }
    }

    #[test]
    fn test_storage_class_standard_latency() {
        assert_eq!(ObjectStorageClass::Standard.retrieval_latency_ms(), 10);
    }

    #[test]
    fn test_storage_class_deep_archive_latency() {
        assert!(ObjectStorageClass::DeepArchive.retrieval_latency_ms() > 1_000_000);
    }

    #[test]
    fn test_storage_class_cost_ordering() {
        assert!(
            ObjectStorageClass::Standard.cost_per_gb()
                > ObjectStorageClass::DeepArchive.cost_per_gb()
        );
    }

    #[test]
    fn test_object_key_full_path() {
        let key = ObjectKey::new("my-bucket", "videos/clip.mp4");
        assert_eq!(key.full_path(), "my-bucket/videos/clip.mp4");
    }

    #[test]
    fn test_object_key_filename() {
        let key = ObjectKey::new("b", "a/b/c/file.mov");
        assert_eq!(key.filename(), "file.mov");
    }

    #[test]
    fn test_object_key_extension_present() {
        let key = ObjectKey::new("b", "clip.mp4");
        assert_eq!(key.extension(), Some("mp4".to_string()));
    }

    #[test]
    fn test_object_key_extension_absent() {
        let key = ObjectKey::new("b", "noextension");
        assert_eq!(key.extension(), None);
    }

    #[test]
    fn test_is_media_video() {
        let obj = make_obj("b", "clip.mp4", "video/mp4", 1024);
        assert!(obj.is_media());
    }

    #[test]
    fn test_is_media_audio() {
        let obj = make_obj("b", "track.wav", "audio/wav", 512);
        assert!(obj.is_media());
    }

    #[test]
    fn test_is_not_media_json() {
        let obj = make_obj("b", "meta.json", "application/json", 256);
        assert!(!obj.is_media());
    }

    #[test]
    fn test_age_days() {
        let obj = make_obj("b", "x.mp4", "video/mp4", 100);
        // last_modified = 1_000_000 s, now = 1_000_000 + 86400 s → 1 day
        let age = obj.age_days(1_000_000 + 86_400);
        assert!((age - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_put_and_get() {
        let mut store = ObjectStore::new();
        let obj = make_obj("bucket", "a/b.mp4", "video/mp4", 1024);
        let key = obj.key.clone();
        store.put(obj);
        assert!(store.get(&key).is_some());
    }

    #[test]
    fn test_put_replaces_existing() {
        let mut store = ObjectStore::new();
        let obj1 = make_obj("bucket", "a.mp4", "video/mp4", 100);
        let obj2 = make_obj("bucket", "a.mp4", "video/mp4", 200);
        store.put(obj1);
        store.put(obj2);
        assert_eq!(store.count(), 1);
        assert_eq!(
            store
                .get(&ObjectKey::new("bucket", "a.mp4"))
                .expect("test expectation failed")
                .size_bytes,
            200
        );
    }

    #[test]
    fn test_delete_existing() {
        let mut store = ObjectStore::new();
        let obj = make_obj("b", "x.mp4", "video/mp4", 10);
        let key = obj.key.clone();
        store.put(obj);
        assert!(store.delete(&key));
        assert_eq!(store.count(), 0);
    }

    #[test]
    fn test_delete_missing_returns_false() {
        let mut store = ObjectStore::new();
        let key = ObjectKey::new("b", "missing.mp4");
        assert!(!store.delete(&key));
    }

    #[test]
    fn test_list_with_prefix() {
        let mut store = ObjectStore::new();
        store.put(make_obj("videos", "2024/clip1.mp4", "video/mp4", 100));
        store.put(make_obj("videos", "2024/clip2.mp4", "video/mp4", 200));
        store.put(make_obj("audio", "track.wav", "audio/wav", 50));
        let results = store.list("videos/2024/");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_total_size_bytes() {
        let mut store = ObjectStore::new();
        store.put(make_obj("b", "a.mp4", "video/mp4", 100));
        store.put(make_obj("b", "b.mp4", "video/mp4", 200));
        assert_eq!(store.total_size_bytes(), 300);
    }
}
