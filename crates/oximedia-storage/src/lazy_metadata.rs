//! Lazy metadata loading — defers per-object HEAD requests until accessed.
//!
//! When listing objects from a cloud provider, the initial `list_objects` call
//! returns minimal metadata (key, size, last-modified).  Full metadata (content
//! type, custom headers, storage class) typically requires a separate per-object
//! HEAD request, which is expensive at scale.
//!
//! This module provides `LazyObject` and `LazyObjectList` which defer the
//! HEAD call until the caller actually reads the extended metadata, using an
//! interior-mutability pattern backed by `std::sync::OnceLock`.
//!
//! # Design
//!
//! ```text
//! LazyObjectList
//!   └── Vec<LazyObject>
//!         ├── BasicMeta (always populated from list response)
//!         └── ExtendedMeta (loaded on first access via MetadataLoader)
//! ```
//!
//! `MetadataLoader` is a pluggable trait that abstracts the provider-specific
//! HEAD request.  For testing, an in-memory loader is provided.

#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use chrono::{DateTime, Utc};

// ─── BasicMeta ──────────────────────────────────────────────────────────────

/// Minimal metadata returned by a list operation (no HEAD required).
#[derive(Debug, Clone)]
pub struct BasicMeta {
    /// Object key.
    pub key: String,
    /// Object size in bytes.
    pub size: u64,
    /// Last-modified timestamp.
    pub last_modified: DateTime<Utc>,
    /// ETag (often available from list, but not always).
    pub etag: Option<String>,
}

// ─── ExtendedMeta ───────────────────────────────────────────────────────────

/// Extended metadata obtained from a per-object HEAD request.
#[derive(Debug, Clone)]
pub struct ExtendedMeta {
    /// MIME content type.
    pub content_type: Option<String>,
    /// Storage class / tier.
    pub storage_class: Option<String>,
    /// Cache-Control header.
    pub cache_control: Option<String>,
    /// Content-Encoding header.
    pub content_encoding: Option<String>,
    /// Content-Disposition header.
    pub content_disposition: Option<String>,
    /// Custom user metadata (x-amz-meta-*, x-ms-meta-*, etc.).
    pub custom_metadata: HashMap<String, String>,
    /// Server-side encryption algorithm, if any.
    pub encryption_algorithm: Option<String>,
    /// Version ID (for versioned buckets).
    pub version_id: Option<String>,
}

impl Default for ExtendedMeta {
    fn default() -> Self {
        Self {
            content_type: None,
            storage_class: None,
            cache_control: None,
            content_encoding: None,
            content_disposition: None,
            custom_metadata: HashMap::new(),
            encryption_algorithm: None,
            version_id: None,
        }
    }
}

// ─── MetadataLoader ─────────────────────────────────────────────────────────

/// Trait for loading extended metadata for a given key.
///
/// Implementors should perform the provider-specific HEAD request.
pub trait MetadataLoader: Send + Sync {
    /// Load extended metadata for the given object key.
    ///
    /// Returns `Ok(ExtendedMeta)` on success, or an error string on failure.
    fn load_extended(&self, key: &str) -> std::result::Result<ExtendedMeta, String>;
}

// ─── InMemoryLoader ─────────────────────────────────────────────────────────

/// In-memory metadata loader for testing purposes.
pub struct InMemoryLoader {
    store: HashMap<String, ExtendedMeta>,
}

impl InMemoryLoader {
    /// Create a new in-memory loader with pre-populated metadata.
    pub fn new(store: HashMap<String, ExtendedMeta>) -> Self {
        Self { store }
    }
}

impl MetadataLoader for InMemoryLoader {
    fn load_extended(&self, key: &str) -> std::result::Result<ExtendedMeta, String> {
        self.store
            .get(key)
            .cloned()
            .ok_or_else(|| format!("metadata not found for key: {key}"))
    }
}

// ─── CountingLoader ─────────────────────────────────────────────────────────

/// Loader that counts how many times load_extended is called (for verifying laziness).
pub struct CountingLoader {
    inner: InMemoryLoader,
    call_count: std::sync::atomic::AtomicU64,
}

impl CountingLoader {
    /// Create a counting loader wrapping an in-memory store.
    pub fn new(store: HashMap<String, ExtendedMeta>) -> Self {
        Self {
            inner: InMemoryLoader::new(store),
            call_count: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Number of times `load_extended` has been called.
    pub fn call_count(&self) -> u64 {
        self.call_count.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl MetadataLoader for CountingLoader {
    fn load_extended(&self, key: &str) -> std::result::Result<ExtendedMeta, String> {
        self.call_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.inner.load_extended(key)
    }
}

// ─── LazyObject ─────────────────────────────────────────────────────────────

/// An object with lazily loaded extended metadata.
///
/// Basic metadata is always available; extended metadata is loaded on first
/// access and cached thereafter.
pub struct LazyObject {
    /// Always-available basic metadata.
    basic: BasicMeta,
    /// Lazily loaded extended metadata.
    extended: OnceLock<std::result::Result<ExtendedMeta, String>>,
    /// Shared reference to the metadata loader.
    loader: Arc<dyn MetadataLoader>,
}

impl std::fmt::Debug for LazyObject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LazyObject")
            .field("basic", &self.basic)
            .field("extended_loaded", &self.extended.get().is_some())
            .finish()
    }
}

impl LazyObject {
    /// Create a new lazy object.
    pub fn new(basic: BasicMeta, loader: Arc<dyn MetadataLoader>) -> Self {
        Self {
            basic,
            extended: OnceLock::new(),
            loader,
        }
    }

    /// Access the basic metadata (always available, no HEAD request).
    pub fn basic(&self) -> &BasicMeta {
        &self.basic
    }

    /// Access the object key.
    pub fn key(&self) -> &str {
        &self.basic.key
    }

    /// Access the object size.
    pub fn size(&self) -> u64 {
        self.basic.size
    }

    /// Access the last-modified timestamp.
    pub fn last_modified(&self) -> DateTime<Utc> {
        self.basic.last_modified
    }

    /// Access the ETag, if available from the list response.
    pub fn etag(&self) -> Option<&str> {
        self.basic.etag.as_deref()
    }

    /// Access extended metadata, triggering a HEAD request on first call.
    ///
    /// The result is cached: subsequent calls return the same result without
    /// another HEAD request.
    pub fn extended(&self) -> std::result::Result<&ExtendedMeta, &str> {
        let result = self
            .extended
            .get_or_init(|| self.loader.load_extended(&self.basic.key));
        match result {
            Ok(meta) => Ok(meta),
            Err(e) => Err(e.as_str()),
        }
    }

    /// Check whether extended metadata has already been loaded.
    pub fn is_extended_loaded(&self) -> bool {
        self.extended.get().is_some()
    }

    /// Get the content type, loading extended metadata if needed.
    pub fn content_type(&self) -> Option<&str> {
        self.extended().ok().and_then(|m| m.content_type.as_deref())
    }

    /// Get the storage class, loading extended metadata if needed.
    pub fn storage_class(&self) -> Option<&str> {
        self.extended()
            .ok()
            .and_then(|m| m.storage_class.as_deref())
    }

    /// Get custom metadata, loading extended metadata if needed.
    pub fn custom_metadata(&self) -> Option<&HashMap<String, String>> {
        self.extended().ok().map(|m| &m.custom_metadata)
    }
}

// ─── LazyObjectList ─────────────────────────────────────────────────────────

/// A list of lazily-loaded objects.
pub struct LazyObjectList {
    objects: Vec<LazyObject>,
    /// Continuation token for fetching the next page.
    pub next_token: Option<String>,
    /// Whether there are more pages.
    pub has_more: bool,
}

impl LazyObjectList {
    /// Create a new lazy object list from basic metadata entries.
    pub fn new(
        basic_entries: Vec<BasicMeta>,
        loader: Arc<dyn MetadataLoader>,
        next_token: Option<String>,
        has_more: bool,
    ) -> Self {
        let objects = basic_entries
            .into_iter()
            .map(|b| LazyObject::new(b, Arc::clone(&loader)))
            .collect();
        Self {
            objects,
            next_token,
            has_more,
        }
    }

    /// Number of objects in this page.
    pub fn len(&self) -> usize {
        self.objects.len()
    }

    /// Whether the list is empty.
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }

    /// Get a reference to an object by index.
    pub fn get(&self, index: usize) -> Option<&LazyObject> {
        self.objects.get(index)
    }

    /// Iterate over the lazy objects.
    pub fn iter(&self) -> impl Iterator<Item = &LazyObject> {
        self.objects.iter()
    }

    /// Total size of all objects (from basic metadata, no HEAD required).
    pub fn total_size(&self) -> u64 {
        self.objects.iter().map(|o| o.size()).sum()
    }

    /// How many objects have had their extended metadata loaded.
    pub fn loaded_count(&self) -> usize {
        self.objects
            .iter()
            .filter(|o| o.is_extended_loaded())
            .count()
    }

    /// Eagerly load extended metadata for all objects.
    ///
    /// Returns the number of successfully loaded entries.
    pub fn prefetch_all(&self) -> usize {
        self.objects.iter().filter(|o| o.extended().is_ok()).count()
    }

    /// Filter objects by a predicate on basic metadata (no HEAD required).
    pub fn filter_by_basic<F>(&self, predicate: F) -> Vec<&LazyObject>
    where
        F: Fn(&BasicMeta) -> bool,
    {
        self.objects
            .iter()
            .filter(|o| predicate(o.basic()))
            .collect()
    }

    /// Filter objects by minimum size (no HEAD required).
    pub fn filter_by_min_size(&self, min_bytes: u64) -> Vec<&LazyObject> {
        self.filter_by_basic(|b| b.size >= min_bytes)
    }

    /// Filter objects by key prefix (no HEAD required).
    pub fn filter_by_prefix(&self, prefix: &str) -> Vec<&LazyObject> {
        self.filter_by_basic(|b| b.key.starts_with(prefix))
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_basic(key: &str, size: u64) -> BasicMeta {
        BasicMeta {
            key: key.to_string(),
            size,
            last_modified: Utc::now(),
            etag: Some(format!("etag-{key}")),
        }
    }

    fn make_extended(content_type: &str) -> ExtendedMeta {
        ExtendedMeta {
            content_type: Some(content_type.to_string()),
            ..Default::default()
        }
    }

    fn make_store() -> HashMap<String, ExtendedMeta> {
        let mut store = HashMap::new();
        store.insert("video/clip.mp4".to_string(), make_extended("video/mp4"));
        store.insert("audio/track.flac".to_string(), make_extended("audio/flac"));
        store.insert(
            "data/file.bin".to_string(),
            make_extended("application/octet-stream"),
        );
        store
    }

    #[test]
    fn test_basic_meta_always_available() {
        let loader = Arc::new(InMemoryLoader::new(make_store()));
        let obj = LazyObject::new(make_basic("video/clip.mp4", 1024), loader);

        assert_eq!(obj.key(), "video/clip.mp4");
        assert_eq!(obj.size(), 1024);
        assert!(!obj.is_extended_loaded());
    }

    #[test]
    fn test_extended_loaded_on_access() {
        let loader = Arc::new(CountingLoader::new(make_store()));
        let obj = LazyObject::new(
            make_basic("video/clip.mp4", 1024),
            Arc::clone(&loader) as Arc<dyn MetadataLoader>,
        );

        assert!(!obj.is_extended_loaded());
        assert_eq!(loader.call_count(), 0);

        let ext = obj.extended().expect("should load");
        assert_eq!(ext.content_type.as_deref(), Some("video/mp4"));
        assert!(obj.is_extended_loaded());
        assert_eq!(loader.call_count(), 1);

        // Second access should NOT trigger another load
        let _ext2 = obj.extended().expect("cached");
        assert_eq!(loader.call_count(), 1);
    }

    #[test]
    fn test_extended_error_for_missing_key() {
        let loader = Arc::new(InMemoryLoader::new(HashMap::new()));
        let obj = LazyObject::new(make_basic("missing.txt", 0), loader);

        let result = obj.extended();
        assert!(result.is_err());
    }

    #[test]
    fn test_content_type_convenience() {
        let loader = Arc::new(InMemoryLoader::new(make_store()));
        let obj = LazyObject::new(make_basic("audio/track.flac", 500), loader);

        assert_eq!(obj.content_type(), Some("audio/flac"));
    }

    #[test]
    fn test_lazy_object_list_basic_ops() {
        let loader = Arc::new(InMemoryLoader::new(make_store()));
        let entries = vec![
            make_basic("video/clip.mp4", 1000),
            make_basic("audio/track.flac", 500),
            make_basic("data/file.bin", 200),
        ];
        let list = LazyObjectList::new(entries, loader, None, false);

        assert_eq!(list.len(), 3);
        assert!(!list.is_empty());
        assert_eq!(list.total_size(), 1700);
        assert_eq!(list.loaded_count(), 0);
    }

    #[test]
    fn test_lazy_object_list_filter_by_prefix() {
        let loader = Arc::new(InMemoryLoader::new(make_store()));
        let entries = vec![
            make_basic("video/clip.mp4", 1000),
            make_basic("video/other.mp4", 800),
            make_basic("audio/track.flac", 500),
        ];
        let list = LazyObjectList::new(entries, loader, None, false);

        let video_only = list.filter_by_prefix("video/");
        assert_eq!(video_only.len(), 2);
    }

    #[test]
    fn test_lazy_object_list_filter_by_min_size() {
        let loader = Arc::new(InMemoryLoader::new(make_store()));
        let entries = vec![
            make_basic("a.bin", 100),
            make_basic("b.bin", 500),
            make_basic("c.bin", 1000),
        ];
        let list = LazyObjectList::new(entries, loader, None, false);

        let large = list.filter_by_min_size(500);
        assert_eq!(large.len(), 2);
    }

    #[test]
    fn test_lazy_object_list_prefetch_all() {
        let loader = Arc::new(CountingLoader::new(make_store()));
        let entries = vec![
            make_basic("video/clip.mp4", 1000),
            make_basic("audio/track.flac", 500),
        ];
        let list = LazyObjectList::new(
            entries,
            Arc::clone(&loader) as Arc<dyn MetadataLoader>,
            None,
            false,
        );

        assert_eq!(list.loaded_count(), 0);
        let loaded = list.prefetch_all();
        assert_eq!(loaded, 2);
        assert_eq!(list.loaded_count(), 2);
        assert_eq!(loader.call_count(), 2);
    }

    #[test]
    fn test_lazy_object_list_pagination() {
        let loader = Arc::new(InMemoryLoader::new(HashMap::new()));
        let list = LazyObjectList::new(
            vec![make_basic("a.bin", 10)],
            loader,
            Some("token-abc".to_string()),
            true,
        );

        assert!(list.has_more);
        assert_eq!(list.next_token.as_deref(), Some("token-abc"));
    }

    #[test]
    fn test_lazy_object_list_empty() {
        let loader = Arc::new(InMemoryLoader::new(HashMap::new()));
        let list = LazyObjectList::new(Vec::new(), loader, None, false);

        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
        assert_eq!(list.total_size(), 0);
        assert_eq!(list.loaded_count(), 0);
    }

    #[test]
    fn test_counting_loader_tracks_calls() {
        let loader = CountingLoader::new(make_store());
        assert_eq!(loader.call_count(), 0);

        let _ = loader.load_extended("video/clip.mp4");
        assert_eq!(loader.call_count(), 1);

        let _ = loader.load_extended("missing");
        assert_eq!(loader.call_count(), 2);
    }

    #[test]
    fn test_custom_metadata_access() {
        let mut store = HashMap::new();
        let mut meta = ExtendedMeta::default();
        meta.custom_metadata
            .insert("author".to_string(), "kitasan".to_string());
        store.insert("doc.pdf".to_string(), meta);

        let loader = Arc::new(InMemoryLoader::new(store));
        let obj = LazyObject::new(make_basic("doc.pdf", 999), loader);

        let custom = obj.custom_metadata().expect("should load");
        assert_eq!(custom.get("author").map(|s| s.as_str()), Some("kitasan"));
    }

    /// Concurrent access to the same `LazyObject` must invoke the loader
    /// exactly once, even when 4 threads race to call `extended()` simultaneously.
    #[test]
    fn test_concurrent_access_loader_called_once() {
        let loader = Arc::new(CountingLoader::new(make_store()));
        let obj = Arc::new(LazyObject::new(
            make_basic("video/clip.mp4", 1024),
            Arc::clone(&loader) as Arc<dyn MetadataLoader>,
        ));

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let obj_clone = Arc::clone(&obj);
                std::thread::spawn(move || {
                    // Each thread accesses extended metadata independently.
                    let result = obj_clone.extended();
                    result.is_ok()
                })
            })
            .collect();

        let all_ok = handles
            .into_iter()
            .all(|h| h.join().expect("thread should not panic"));

        assert!(all_ok, "all threads should see Ok extended metadata");
        // OnceLock ensures the loader is invoked exactly once despite the race.
        assert_eq!(
            loader.call_count(),
            1,
            "loader must be called exactly once under concurrent access"
        );
    }

    /// Verify that `LazyObjectList::get` returns the correct entry by index.
    #[test]
    fn test_lazy_object_list_get_by_index() {
        let loader = Arc::new(InMemoryLoader::new(make_store()));
        let entries = vec![
            make_basic("video/clip.mp4", 1000),
            make_basic("audio/track.flac", 500),
        ];
        let list = LazyObjectList::new(entries, loader, None, false);

        let obj = list.get(1).expect("index 1 should exist");
        assert_eq!(obj.key(), "audio/track.flac");
        assert!(list.get(99).is_none());
    }

    /// Verify that `loaded_count()` increases exactly as entries are accessed.
    #[test]
    fn test_lazy_object_list_loaded_count_increments() {
        let loader = Arc::new(CountingLoader::new(make_store()));
        let entries = vec![
            make_basic("video/clip.mp4", 1000),
            make_basic("audio/track.flac", 500),
            make_basic("data/file.bin", 200),
        ];
        let list = LazyObjectList::new(
            entries,
            Arc::clone(&loader) as Arc<dyn MetadataLoader>,
            None,
            false,
        );

        assert_eq!(list.loaded_count(), 0);

        // Access first entry
        let _ = list.get(0).and_then(|o| o.extended().ok());
        assert_eq!(list.loaded_count(), 1);

        // Access second entry
        let _ = list.get(1).and_then(|o| o.extended().ok());
        assert_eq!(list.loaded_count(), 2);
    }

    /// storage_class() convenience accessor loads extended metadata.
    #[test]
    fn test_storage_class_convenience() {
        let mut store = HashMap::new();
        let mut meta = ExtendedMeta::default();
        meta.storage_class = Some("STANDARD_IA".to_string());
        store.insert("cold/data.bin".to_string(), meta);

        let loader = Arc::new(InMemoryLoader::new(store));
        let obj = LazyObject::new(make_basic("cold/data.bin", 8192), loader);

        assert_eq!(obj.storage_class(), Some("STANDARD_IA"));
    }

    /// etag() returns the value baked into BasicMeta without loading extended.
    #[test]
    fn test_etag_from_basic_no_load() {
        let loader = Arc::new(CountingLoader::new(make_store()));
        let obj = LazyObject::new(
            make_basic("video/clip.mp4", 100),
            Arc::clone(&loader) as Arc<dyn MetadataLoader>,
        );

        // etag comes from BasicMeta — no HEAD request needed
        let etag = obj.etag();
        assert!(
            etag.is_some(),
            "etag should be populated from list response"
        );
        assert_eq!(loader.call_count(), 0, "no load triggered for etag access");
    }
}
