//! Lazy (deferred) essence data loading for large AAF files.
//!
//! AAF files can embed hundreds of gigabytes of raw essence (media) data.
//! Loading it all at parse time is impractical.  This module provides two
//! types:
//!
//! * [`LazyEssence`] — a single essence stream descriptor whose bytes are
//!   loaded from the source file the first time they are needed, then cached
//!   behind `Arc<Mutex<Option<Arc<Vec<u8>>>>>`.  Callers receive an
//!   `Arc<Vec<u8>>` — a shared, reference-counted pointer — so multiple
//!   callers can hold the data simultaneously without any extra copies.
//! * [`EssenceCollection`] — an indexed bag of `LazyEssence` items with
//!   aggregate statistics and a bulk-preload helper.
//!
//! # Thread safety
//!
//! Both types are `Send + Sync`.  The `Mutex` serialises the first load; once
//! populated the `Arc<Vec<u8>>` can be cloned cheaply by any number of
//! concurrent readers.
//!
//! # Example
//!
//! ```no_run
//! use oximedia_aaf::lazy_essence::{LazyEssence, EssenceCollection};
//! use std::path::PathBuf;
//!
//! let essence = LazyEssence::new(
//!     PathBuf::from("media.aaf"),
//!     1024,          // byte offset inside the file
//!     4096,          // byte length
//!     "Picture".to_string(),
//! );
//!
//! // Data is loaded on first access; returns Arc<Vec<u8>> (zero-copy share)
//! let bytes = essence.data().expect("load essence");
//! assert_eq!(bytes.len(), 4096);
//! assert!(essence.is_loaded());
//! ```

use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

// ─── LazyEssence ─────────────────────────────────────────────────────────────

/// A single essence stream that is loaded from disk only on first access.
///
/// Clone is `O(1)` — all clones share the same underlying cache.  Callers
/// receive an `Arc<Vec<u8>>` from [`data()`](Self::data), so the bytes are
/// shared rather than copied on every access.
#[derive(Clone)]
pub struct LazyEssence {
    /// Path to the file containing the essence (may be the same .aaf or an
    /// external media file).
    pub source_path: PathBuf,
    /// Byte offset within `source_path` where the essence begins.
    pub offset: u64,
    /// Byte length of the essence block.
    pub length: u64,
    /// Human-readable data-definition name (e.g. "Picture", "Sound").
    pub descriptor: String,
    /// Shared cache — `None` until first load.  The inner `Arc<Vec<u8>>`
    /// allows multiple callers to hold the data simultaneously without copying.
    data_cache: Arc<Mutex<Option<Arc<Vec<u8>>>>>,
}

impl std::fmt::Debug for LazyEssence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let loaded = self.is_loaded();
        f.debug_struct("LazyEssence")
            .field("source_path", &self.source_path)
            .field("offset", &self.offset)
            .field("length", &self.length)
            .field("descriptor", &self.descriptor)
            .field("loaded", &loaded)
            .finish()
    }
}

impl LazyEssence {
    /// Create a new (unloaded) essence descriptor.
    #[must_use]
    pub fn new(source_path: PathBuf, offset: u64, length: u64, descriptor: String) -> Self {
        Self {
            source_path,
            offset,
            length,
            descriptor,
            data_cache: Arc::new(Mutex::new(None)),
        }
    }

    /// Load and return all essence bytes as a shared `Arc<Vec<u8>>`.
    ///
    /// On the first call the bytes are read from `source_path` and cached;
    /// subsequent calls clone the `Arc` — no extra I/O or buffer copies.
    ///
    /// # Errors
    ///
    /// Returns an [`std::io::Error`] if the file cannot be opened or the read
    /// fails.  A poisoned mutex is reported as an error with kind `Other`.
    pub fn data(&self) -> std::io::Result<Arc<Vec<u8>>> {
        let mut guard = self.data_cache.lock().map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::Other, "lazy essence mutex poisoned")
        })?;
        if let Some(ref cached) = *guard {
            return Ok(Arc::clone(cached));
        }
        let loaded = Arc::new(self.load_from_disk()?);
        *guard = Some(Arc::clone(&loaded));
        Ok(loaded)
    }

    /// Return `true` if the data has already been loaded into the cache.
    #[must_use]
    pub fn is_loaded(&self) -> bool {
        self.data_cache.lock().map(|g| g.is_some()).unwrap_or(false)
    }

    /// Return the declared byte length (not necessarily the number of bytes
    /// actually cached — use [`Self::data()`] for that).
    #[must_use]
    pub fn byte_length(&self) -> u64 {
        self.length
    }

    /// Return the data-definition descriptor string.
    #[must_use]
    pub fn descriptor(&self) -> &str {
        &self.descriptor
    }

    /// Return bytes `data[start..end]` as a new `Vec<u8>`.
    ///
    /// Unlike [`data()`](Self::data) this always copies the slice, but slices
    /// are typically small compared to the full essence buffer.  `end` is
    /// clamped to the actual data length; an empty `Vec` is returned when
    /// `start >= end` after clamping.
    ///
    /// # Errors
    ///
    /// Propagates any I/O error from the initial load.
    pub fn read_range(&self, start: u64, end: u64) -> std::io::Result<Vec<u8>> {
        let all = self.data()?;
        let len = all.len() as u64;
        let clamped_start = start.min(len) as usize;
        let clamped_end = end.min(len) as usize;
        if clamped_start >= clamped_end {
            return Ok(Vec::new());
        }
        Ok(all[clamped_start..clamped_end].to_vec())
    }

    /// Internal: read `self.length` bytes from `self.source_path` starting at
    /// `self.offset`.
    fn load_from_disk(&self) -> std::io::Result<Vec<u8>> {
        let mut file = std::fs::File::open(&self.source_path)?;
        file.seek(SeekFrom::Start(self.offset))?;
        let mut buf = vec![0u8; self.length as usize];
        file.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Load essence from an already-open `Read + Seek` source.
    ///
    /// This is useful when the caller already holds an open handle to the AAF
    /// compound file and wants to avoid a second `open()`.  If the cache is
    /// already populated this is a no-op.
    ///
    /// # Errors
    ///
    /// Returns any I/O error from seeking or reading.
    pub fn load_from_reader<R: Read + Seek>(&self, reader: &mut R) -> std::io::Result<()> {
        let mut guard = self.data_cache.lock().map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::Other, "lazy essence mutex poisoned")
        })?;
        if guard.is_some() {
            return Ok(());
        }
        reader.seek(SeekFrom::Start(self.offset))?;
        let mut buf = vec![0u8; self.length as usize];
        reader.read_exact(&mut buf)?;
        *guard = Some(Arc::new(buf));
        Ok(())
    }

    /// Pre-populate the cache with externally supplied bytes.
    ///
    /// Provided for testing and for cases where the bytes were obtained
    /// through a different mechanism (e.g. network).  If the cache is already
    /// populated this replaces the existing data.
    pub fn inject(&self, data: Vec<u8>) {
        if let Ok(mut guard) = self.data_cache.lock() {
            *guard = Some(Arc::new(data));
        }
    }
}

// ─── EssenceCollection ───────────────────────────────────────────────────────

/// An ordered collection of [`LazyEssence`] items.
///
/// Items are appended with [`add`](Self::add) and retrieved by zero-based
/// index with [`get`](Self::get).
#[derive(Debug, Clone, Default)]
pub struct EssenceCollection {
    items: Vec<LazyEssence>,
}

impl EssenceCollection {
    /// Create an empty collection.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append an essence item.
    pub fn add(&mut self, essence: LazyEssence) {
        self.items.push(essence);
    }

    /// Return a reference to the item at `index`, or `None`.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&LazyEssence> {
        self.items.get(index)
    }

    /// Number of items in the collection.
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Return `true` if the collection has no items.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Number of items whose data has been loaded into memory.
    #[must_use]
    pub fn loaded_count(&self) -> usize {
        self.items.iter().filter(|e| e.is_loaded()).count()
    }

    /// Sum of the declared byte lengths of all items.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.items.iter().map(|e| e.length).sum()
    }

    /// Eagerly load all unloaded items.
    ///
    /// Returns the first error encountered, but items that were already loaded
    /// before the failure remain cached.
    ///
    /// # Errors
    ///
    /// Returns an [`std::io::Error`] from the first item that fails to load.
    pub fn preload_all(&self) -> std::io::Result<()> {
        for item in &self.items {
            // Discard the Arc — we only care that the data is now cached.
            let _ = item.data()?;
        }
        Ok(())
    }

    /// Iterate over all items.
    pub fn iter(&self) -> std::slice::Iter<'_, LazyEssence> {
        self.items.iter()
    }

    /// Find the first item matching the given descriptor string.
    #[must_use]
    pub fn find_by_descriptor(&self, descriptor: &str) -> Option<&LazyEssence> {
        self.items.iter().find(|e| e.descriptor == descriptor)
    }

    /// Return a slice of all items.
    #[must_use]
    pub fn as_slice(&self) -> &[LazyEssence] {
        &self.items
    }
}

impl<'a> IntoIterator for &'a EssenceCollection {
    type Item = &'a LazyEssence;
    type IntoIter = std::slice::Iter<'a, LazyEssence>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.iter()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::thread;

    /// Write `data` to a temp file and return its path.
    fn make_temp_file(data: &[u8]) -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let mut path = std::env::temp_dir();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
        let name = format!("oximedia_aaf_lazy_test_{now}_{seq}.bin");
        path.push(name);
        let mut f = std::fs::File::create(&path).expect("create temp file");
        f.write_all(data).expect("write temp data");
        path
    }

    // ── LazyEssence basic ────────────────────────────────────────────────────

    #[test]
    fn test_lazy_essence_not_loaded_on_creation() {
        let path = make_temp_file(b"hello");
        let e = LazyEssence::new(path.clone(), 0, 5, "Picture".to_string());
        assert!(!e.is_loaded());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_lazy_essence_loads_on_data_call() {
        let payload = b"LAZY_LOAD_TEST";
        let path = make_temp_file(payload);
        let e = LazyEssence::new(path.clone(), 0, payload.len() as u64, "Sound".to_string());
        let data = e.data().expect("data must load");
        assert_eq!(data.as_slice(), payload.as_ref());
        assert!(e.is_loaded());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_lazy_essence_data_cached_after_first_load() {
        let payload = b"CACHED";
        let path = make_temp_file(payload);
        let e = LazyEssence::new(path.clone(), 0, payload.len() as u64, "Picture".to_string());
        let _ = e.data().expect("first load");
        // Remove the file — second call must still succeed from cache
        let _ = std::fs::remove_file(&path);
        let data2 = e.data().expect("second load must use cache");
        assert_eq!(data2.as_slice(), payload.as_ref());
    }

    #[test]
    fn test_lazy_essence_byte_length() {
        let e = LazyEssence::new(PathBuf::from("/dev/null"), 0, 42, "Data".to_string());
        assert_eq!(e.byte_length(), 42);
    }

    #[test]
    fn test_lazy_essence_descriptor() {
        let e = LazyEssence::new(PathBuf::from("/dev/null"), 0, 0, "Sound".to_string());
        assert_eq!(e.descriptor(), "Sound");
    }

    #[test]
    fn test_lazy_essence_offset_load() {
        // File: 0..5 = "AAAAA", 5..10 = "BBBBB"
        let mut data = vec![0u8; 10];
        data[..5].copy_from_slice(b"AAAAA");
        data[5..].copy_from_slice(b"BBBBB");
        let path = make_temp_file(&data);
        let e = LazyEssence::new(path.clone(), 5, 5, "Sound".to_string());
        let loaded = e.data().expect("load from offset");
        assert_eq!(loaded.as_slice(), b"BBBBB");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_lazy_essence_read_range_full() {
        let payload = b"ABCDEFGH";
        let path = make_temp_file(payload);
        let e = LazyEssence::new(path.clone(), 0, payload.len() as u64, "Data".to_string());
        let slice = e.read_range(2, 5).expect("read_range");
        assert_eq!(slice, b"CDE");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_lazy_essence_read_range_clamps_to_length() {
        let payload = b"XYZ";
        let path = make_temp_file(payload);
        let e = LazyEssence::new(path.clone(), 0, 3, "Data".to_string());
        let slice = e.read_range(1, 9999).expect("read_range clamped");
        assert_eq!(slice, b"YZ");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_lazy_essence_read_range_empty_when_start_ge_end() {
        let payload = b"HELLO";
        let path = make_temp_file(payload);
        let e = LazyEssence::new(path.clone(), 0, 5, "Data".to_string());
        let slice = e.read_range(3, 3).expect("empty range");
        assert!(slice.is_empty());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_lazy_essence_inject_skips_disk_io() {
        // non-existent path — must not be opened because inject pre-populates
        let e = LazyEssence::new(
            PathBuf::from("/no/such/file.aaf"),
            0,
            4,
            "Picture".to_string(),
        );
        e.inject(b"FAKE".to_vec());
        let data = e.data().expect("injected data");
        assert_eq!(data.as_slice(), b"FAKE");
    }

    #[test]
    fn test_lazy_essence_clone_shares_cache() {
        let payload = b"SHARED";
        let path = make_temp_file(payload);
        let e1 = LazyEssence::new(path.clone(), 0, payload.len() as u64, "Sound".to_string());
        let e2 = e1.clone();
        assert!(!e2.is_loaded());
        let _ = e1.data().expect("load via e1");
        // e2 must now be loaded as well (shared Arc<Mutex<Option<…>>>)
        assert!(e2.is_loaded());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_lazy_essence_error_on_missing_file() {
        let e = LazyEssence::new(
            PathBuf::from("/no/such/file_oximedia.aaf"),
            0,
            8,
            "Picture".to_string(),
        );
        assert!(e.data().is_err());
    }

    #[test]
    fn test_lazy_essence_load_from_reader() {
        let payload = b"READER_LOAD";
        let path = make_temp_file(payload);
        let e = LazyEssence::new(path.clone(), 0, payload.len() as u64, "Data".to_string());
        let mut file = std::fs::File::open(&path).expect("open");
        e.load_from_reader(&mut file).expect("load from reader");
        assert!(e.is_loaded());
        assert_eq!(e.data().expect("cached").as_slice(), payload.as_ref());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_lazy_essence_debug_repr() {
        let e = LazyEssence::new(PathBuf::from("/tmp/x.aaf"), 0, 10, "Sound".to_string());
        let s = format!("{e:?}");
        assert!(s.contains("LazyEssence"));
        assert!(s.contains("loaded: false"));
    }

    // ── EssenceCollection ────────────────────────────────────────────────────

    #[test]
    fn test_collection_empty_on_new() {
        let c = EssenceCollection::new();
        assert!(c.is_empty());
        assert_eq!(c.len(), 0);
        assert_eq!(c.loaded_count(), 0);
        assert_eq!(c.total_bytes(), 0);
    }

    #[test]
    fn test_collection_add_and_get() {
        let mut c = EssenceCollection::new();
        let e = LazyEssence::new(PathBuf::from("/tmp/a.aaf"), 0, 100, "Picture".to_string());
        c.add(e);
        assert_eq!(c.len(), 1);
        assert!(c.get(0).is_some());
        assert!(c.get(1).is_none());
    }

    #[test]
    fn test_collection_total_bytes() {
        let mut c = EssenceCollection::new();
        c.add(LazyEssence::new(
            PathBuf::from("/x"),
            0,
            50,
            "A".to_string(),
        ));
        c.add(LazyEssence::new(
            PathBuf::from("/y"),
            0,
            75,
            "B".to_string(),
        ));
        assert_eq!(c.total_bytes(), 125);
    }

    #[test]
    fn test_collection_loaded_count() {
        let mut c = EssenceCollection::new();
        let e1 = LazyEssence::new(PathBuf::from("/x"), 0, 5, "A".to_string());
        let e2 = LazyEssence::new(PathBuf::from("/y"), 0, 3, "B".to_string());
        e1.inject(b"hello".to_vec());
        c.add(e1);
        c.add(e2);
        assert_eq!(c.loaded_count(), 1);
    }

    #[test]
    fn test_collection_preload_all() {
        let payload = b"PRELOAD";
        let path = make_temp_file(payload);
        let mut c = EssenceCollection::new();
        c.add(LazyEssence::new(
            path.clone(),
            0,
            payload.len() as u64,
            "Sound".to_string(),
        ));
        assert_eq!(c.loaded_count(), 0);
        c.preload_all().expect("preload must succeed");
        assert_eq!(c.loaded_count(), 1);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_collection_find_by_descriptor() {
        let mut c = EssenceCollection::new();
        c.add(LazyEssence::new(
            PathBuf::from("/a"),
            0,
            1,
            "Picture".to_string(),
        ));
        c.add(LazyEssence::new(
            PathBuf::from("/b"),
            0,
            2,
            "Sound".to_string(),
        ));
        assert!(c.find_by_descriptor("Sound").is_some());
        assert!(c.find_by_descriptor("Missing").is_none());
    }

    #[test]
    fn test_collection_iter() {
        let mut c = EssenceCollection::new();
        c.add(LazyEssence::new(PathBuf::from("/a"), 0, 1, "A".to_string()));
        c.add(LazyEssence::new(PathBuf::from("/b"), 0, 2, "B".to_string()));
        let total: u64 = c.iter().map(|e| e.byte_length()).sum();
        assert_eq!(total, 3);
    }

    #[test]
    fn test_collection_into_iter() {
        let mut c = EssenceCollection::new();
        c.add(LazyEssence::new(
            PathBuf::from("/a"),
            0,
            10,
            "X".to_string(),
        ));
        let mut count = 0;
        for _ in &c {
            count += 1;
        }
        assert_eq!(count, 1);
    }

    // ── Concurrency ──────────────────────────────────────────────────────────

    #[test]
    fn test_concurrent_first_load_is_safe() {
        let payload: Vec<u8> = (0u8..=255).collect();
        let path = make_temp_file(&payload);
        let e = Arc::new(LazyEssence::new(
            path.clone(),
            0,
            payload.len() as u64,
            "Picture".to_string(),
        ));

        let handles: Vec<_> = (0..8)
            .map(|_| {
                let ec = Arc::clone(&e);
                thread::spawn(move || ec.data().map(|d| d.len()))
            })
            .collect();

        for h in handles {
            let len = h.join().expect("thread join").expect("load ok");
            assert_eq!(len, 256);
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_as_slice() {
        let mut c = EssenceCollection::new();
        c.add(LazyEssence::new(PathBuf::from("/a"), 0, 1, "A".to_string()));
        assert_eq!(c.as_slice().len(), 1);
    }
}
