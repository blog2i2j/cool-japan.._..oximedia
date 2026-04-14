//! GPU shader cache management.
//!
//! This module provides compiled shader storage, cache eviction, version
//! tracking, and **disk-persistent** caching to avoid redundant shader
//! compilation work across process restarts.
//!
//! # Disk cache layout
//!
//! The persistent cache stores each entry as a pair of files inside the
//! configured directory:
//!
//! ```text
//! <cache_dir>/<hex_hash>_<backend>_<flags>.shd   – raw bytecode
//! <cache_dir>/<hex_hash>_<backend>_<flags>.meta  – metadata (JSON-like text)
//! ```
//!
//! The text metadata file contains a single line:
//! `<source_hash> <backend> <feature_flags> <created_unix_secs>`.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Version identifier for a compiled shader.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ShaderVersion {
    /// Source code hash (simple string identifier).
    pub source_hash: u64,
    /// Backend name (e.g. "vulkan", "metal", "dx12").
    pub backend: String,
    /// Optional feature flags bitmask.
    pub feature_flags: u32,
}

impl ShaderVersion {
    /// Create a new `ShaderVersion`.
    #[allow(dead_code)]
    #[must_use]
    pub fn new(source_hash: u64, backend: impl Into<String>, feature_flags: u32) -> Self {
        Self {
            source_hash,
            backend: backend.into(),
            feature_flags,
        }
    }
}

/// A compiled shader blob stored in the cache.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct CompiledShader {
    /// The shader byte code or SPIR-V blob.
    pub bytecode: Vec<u8>,
    /// Version information.
    pub version: ShaderVersion,
    /// When this entry was inserted.
    pub created_at: SystemTime,
    /// Approximate size of the bytecode in bytes.
    pub size_bytes: usize,
    /// How many times this shader has been retrieved from the cache.
    pub hit_count: u64,
}

impl CompiledShader {
    /// Create a new `CompiledShader`.
    #[allow(dead_code)]
    #[must_use]
    pub fn new(bytecode: Vec<u8>, version: ShaderVersion) -> Self {
        let size_bytes = bytecode.len();
        Self {
            bytecode,
            version,
            created_at: SystemTime::now(),
            size_bytes,
            hit_count: 0,
        }
    }
}

/// Statistics for the shader cache.
#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct ShaderCacheStats {
    /// Total number of entries currently held.
    pub entry_count: usize,
    /// Total bytes occupied by all cached bytecodes.
    pub total_bytes: usize,
    /// Total number of cache hits.
    pub hits: u64,
    /// Total number of cache misses.
    pub misses: u64,
    /// Total number of evictions performed.
    pub evictions: u64,
}

/// Eviction policy for the shader cache.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EvictionPolicy {
    /// Least-recently-used eviction.
    #[default]
    Lru,
    /// Least-frequently-used eviction.
    Lfu,
    /// Oldest-first eviction.
    OldestFirst,
}

/// In-process GPU shader cache.
#[allow(dead_code)]
pub struct GpuShaderCache {
    entries: HashMap<ShaderVersion, CompiledShader>,
    /// Maximum total bytes stored before eviction triggers.
    max_bytes: usize,
    /// Maximum number of entries before eviction triggers.
    max_entries: usize,
    /// Selected eviction policy.
    policy: EvictionPolicy,
    /// Accumulated statistics.
    stats: ShaderCacheStats,
    /// Last-access timestamps (version → instant-as-duration-since-epoch).
    last_access: HashMap<ShaderVersion, SystemTime>,
}

impl GpuShaderCache {
    /// Create a new shader cache.
    ///
    /// * `max_bytes`   – evict when total payload exceeds this many bytes.
    /// * `max_entries` – evict when the number of entries exceeds this.
    /// * `policy`      – which eviction strategy to use.
    #[allow(dead_code)]
    #[must_use]
    pub fn new(max_bytes: usize, max_entries: usize, policy: EvictionPolicy) -> Self {
        Self {
            entries: HashMap::new(),
            max_bytes,
            max_entries,
            policy,
            stats: ShaderCacheStats::default(),
            last_access: HashMap::new(),
        }
    }

    /// Insert a compiled shader into the cache.
    ///
    /// If the cache is full, one entry is evicted according to the current
    /// policy before the new entry is stored.
    #[allow(dead_code)]
    pub fn insert(&mut self, shader: CompiledShader) {
        // Evict if needed before inserting.
        while self.needs_eviction(shader.size_bytes) {
            if !self.evict_one() {
                break; // Nothing left to evict.
            }
        }

        self.stats.total_bytes += shader.size_bytes;
        self.stats.entry_count += 1;
        self.last_access
            .insert(shader.version.clone(), SystemTime::now());
        self.entries.insert(shader.version.clone(), shader);
    }

    /// Retrieve a compiled shader by its version key.
    ///
    /// Returns `None` if the shader is not cached.
    #[allow(dead_code)]
    pub fn get(&mut self, version: &ShaderVersion) -> Option<&CompiledShader> {
        if self.entries.contains_key(version) {
            self.stats.hits += 1;
            // Update access time and hit count.
            self.last_access.insert(version.clone(), SystemTime::now());
            if let Some(e) = self.entries.get_mut(version) {
                e.hit_count += 1;
            }
            self.entries.get(version)
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Check whether a shader is present in the cache.
    #[allow(dead_code)]
    #[must_use]
    pub fn contains(&self, version: &ShaderVersion) -> bool {
        self.entries.contains_key(version)
    }

    /// Remove a specific shader from the cache.
    #[allow(dead_code)]
    pub fn remove(&mut self, version: &ShaderVersion) -> Option<CompiledShader> {
        if let Some(shader) = self.entries.remove(version) {
            self.stats.total_bytes = self.stats.total_bytes.saturating_sub(shader.size_bytes);
            self.stats.entry_count = self.stats.entry_count.saturating_sub(1);
            self.last_access.remove(version);
            Some(shader)
        } else {
            None
        }
    }

    /// Remove all shaders for a given backend.
    #[allow(dead_code)]
    pub fn invalidate_backend(&mut self, backend: &str) {
        let to_remove: Vec<ShaderVersion> = self
            .entries
            .keys()
            .filter(|v| v.backend == backend)
            .cloned()
            .collect();
        for key in to_remove {
            self.remove(&key);
        }
    }

    /// Clear all entries.
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.entries.clear();
        self.last_access.clear();
        self.stats.total_bytes = 0;
        self.stats.entry_count = 0;
    }

    /// Current statistics.
    #[allow(dead_code)]
    #[must_use]
    pub fn stats(&self) -> &ShaderCacheStats {
        &self.stats
    }

    /// Number of entries currently held.
    #[allow(dead_code)]
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if the cache is empty.
    #[allow(dead_code)]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn needs_eviction(&self, incoming_bytes: usize) -> bool {
        let bytes_after = self.stats.total_bytes + incoming_bytes;
        bytes_after > self.max_bytes || self.stats.entry_count >= self.max_entries
    }

    /// Evict one entry according to the current policy. Returns `true` if
    /// something was actually removed.
    fn evict_one(&mut self) -> bool {
        if self.entries.is_empty() {
            return false;
        }

        let victim_key: Option<ShaderVersion> = match self.policy {
            EvictionPolicy::Lru => {
                // Remove the entry with the oldest last-access time.
                self.last_access
                    .iter()
                    .min_by_key(|(_, t)| *t)
                    .map(|(k, _)| k.clone())
            }
            EvictionPolicy::Lfu => {
                // Remove the entry with the lowest hit_count.
                self.entries
                    .iter()
                    .min_by_key(|(_, v)| v.hit_count)
                    .map(|(k, _)| k.clone())
            }
            EvictionPolicy::OldestFirst => {
                // Remove the entry created earliest.
                self.entries
                    .iter()
                    .min_by_key(|(_, v)| v.created_at)
                    .map(|(k, _)| k.clone())
            }
        };

        if let Some(key) = victim_key {
            self.remove(&key);
            self.stats.evictions += 1;
            true
        } else {
            false
        }
    }
}

impl Default for GpuShaderCache {
    fn default() -> Self {
        // 64 MB default cache, up to 256 entries.
        Self::new(64 * 1024 * 1024, 256, EvictionPolicy::Lru)
    }
}

/// Compute a simple 64-bit FNV-1a hash of a byte slice.
#[allow(dead_code)]
#[must_use]
pub fn hash_source(data: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
    const FNV_PRIME: u64 = 1_099_511_628_211;
    let mut hash = FNV_OFFSET;
    for &byte in data {
        hash ^= u64(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Convenience: convert a `u8` to `u64` without a cast warning.
#[inline(always)]
fn u64(v: u8) -> u64 {
    u64::from(v)
}

/// Estimate the age of a `SystemTime` relative to now.
#[allow(dead_code)]
#[must_use]
pub fn age_of(t: SystemTime) -> Duration {
    SystemTime::now()
        .duration_since(t)
        .unwrap_or(Duration::ZERO)
}

// =============================================================================
// Disk-persistent shader cache (Task 3)
// =============================================================================

/// Errors that can occur during disk cache I/O.
#[derive(Debug)]
pub enum DiskCacheError {
    /// A filesystem I/O error occurred.
    Io(std::io::Error),
    /// The metadata file was malformed (could not be parsed).
    MalformedMetadata(String),
}

impl std::fmt::Display for DiskCacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "disk cache I/O error: {e}"),
            Self::MalformedMetadata(s) => write!(f, "malformed cache metadata: {s}"),
        }
    }
}

impl From<std::io::Error> for DiskCacheError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Statistics for the disk cache.
#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct DiskCacheStats {
    /// Number of times a bytecode blob was successfully read from disk.
    pub disk_hits: u64,
    /// Number of times a lookup fell through to compilation.
    pub disk_misses: u64,
    /// Number of bytecode blobs written to disk.
    pub disk_writes: u64,
    /// Number of I/O errors encountered (non-fatal; treated as misses).
    pub io_errors: u64,
}

/// Disk-persistent GPU shader cache.
///
/// Each entry is stored as two files in `cache_dir`:
/// - `<key>.shd` – the raw bytecode blob.
/// - `<key>.meta` – a single-line text file:
///   `<source_hash> <backend> <feature_flags> <unix_secs_since_epoch>`.
///
/// The cache directory is created automatically on first use.
#[allow(dead_code)]
pub struct DiskShaderCache {
    cache_dir: PathBuf,
    stats: DiskCacheStats,
}

impl DiskShaderCache {
    /// Open (or create) a disk shader cache rooted at `cache_dir`.
    ///
    /// The directory is created if it does not already exist.
    ///
    /// # Errors
    ///
    /// Returns a [`DiskCacheError`] if the directory cannot be created.
    #[allow(dead_code)]
    pub fn open(cache_dir: impl AsRef<Path>) -> Result<Self, DiskCacheError> {
        let cache_dir = cache_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&cache_dir)?;
        Ok(Self {
            cache_dir,
            stats: DiskCacheStats::default(),
        })
    }

    /// Look up a shader by its [`ShaderVersion`].
    ///
    /// Returns `Some(bytecode)` if both `.shd` and `.meta` files exist and
    /// the metadata matches the requested version.  Returns `None` on any
    /// mismatch or I/O error.
    #[allow(dead_code)]
    pub fn get(&mut self, version: &ShaderVersion) -> Option<Vec<u8>> {
        let key = self.entry_key(version);
        let shd_path = self.cache_dir.join(format!("{key}.shd"));
        let meta_path = self.cache_dir.join(format!("{key}.meta"));

        // Read and validate the metadata.
        match self.read_meta(&meta_path, version) {
            Err(_) => {
                self.stats.disk_misses += 1;
                return None;
            }
            Ok(false) => {
                self.stats.disk_misses += 1;
                return None;
            }
            Ok(true) => {}
        }

        // Read the bytecode blob.
        match std::fs::read(&shd_path) {
            Ok(bytes) => {
                self.stats.disk_hits += 1;
                Some(bytes)
            }
            Err(_) => {
                self.stats.disk_misses += 1;
                self.stats.io_errors += 1;
                None
            }
        }
    }

    /// Store a compiled shader on disk.
    ///
    /// On any I/O error the error is recorded in statistics but is **not**
    /// propagated — the in-memory cache remains the source of truth.
    #[allow(dead_code)]
    pub fn put(&mut self, shader: &CompiledShader) {
        let key = self.entry_key(&shader.version);
        let shd_path = self.cache_dir.join(format!("{key}.shd"));
        let meta_path = self.cache_dir.join(format!("{key}.meta"));

        // Write bytecode.
        if let Err(_e) = self.write_bytes(&shd_path, &shader.bytecode) {
            self.stats.io_errors += 1;
            return;
        }

        // Write metadata: "<source_hash> <backend> <feature_flags> <unix_secs>"
        let unix_secs = shader
            .created_at
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_secs();
        let meta_content = format!(
            "{} {} {} {}",
            shader.version.source_hash,
            shader.version.backend,
            shader.version.feature_flags,
            unix_secs
        );
        if let Err(_e) = self.write_str(&meta_path, &meta_content) {
            self.stats.io_errors += 1;
            // Remove orphaned .shd to avoid inconsistency.
            let _ = std::fs::remove_file(&shd_path);
            return;
        }

        self.stats.disk_writes += 1;
    }

    /// Invalidate (delete) all entries for a specific backend.
    ///
    /// Errors during directory listing or file deletion are silently ignored.
    #[allow(dead_code)]
    pub fn invalidate_backend(&mut self, backend: &str) {
        let Ok(entries) = std::fs::read_dir(&self.cache_dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                // The key encodes the backend: `<hash>_<backend>_<flags>.*`
                if name.contains(&format!("_{backend}_")) {
                    let _ = std::fs::remove_file(&path);
                }
            }
        }
    }

    /// Remove all cached entries from disk.
    ///
    /// Errors during directory listing or file deletion are silently ignored.
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        let Ok(entries) = std::fs::read_dir(&self.cache_dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                if ext == "shd" || ext == "meta" {
                    let _ = std::fs::remove_file(&path);
                }
            }
        }
    }

    /// Returns a snapshot of the accumulated disk-cache statistics.
    #[allow(dead_code)]
    #[must_use]
    pub fn stats(&self) -> &DiskCacheStats {
        &self.stats
    }

    // ── private helpers ──────────────────────────────────────────────────────

    /// Derive a filesystem-safe key from a [`ShaderVersion`].
    fn entry_key(&self, v: &ShaderVersion) -> String {
        // Sanitise the backend string (remove chars that are illegal on some
        // filesystems).  We allow alphanumerics and hyphens only.
        let safe_backend: String = v
            .backend
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        format!(
            "{:016x}_{}_{}",
            v.source_hash, safe_backend, v.feature_flags
        )
    }

    /// Read and validate a `.meta` file.
    ///
    /// Returns `Ok(true)` if the file matches `version`, `Ok(false)` if it
    /// does not match, and `Err` on I/O failure.
    fn read_meta(&mut self, path: &Path, version: &ShaderVersion) -> Result<bool, DiskCacheError> {
        let mut file = std::fs::File::open(path)?;
        let mut content = String::new();
        file.read_to_string(&mut content)?;
        let parts: Vec<&str> = content.trim().splitn(4, ' ').collect();
        if parts.len() < 3 {
            return Err(DiskCacheError::MalformedMetadata(content.clone()));
        }
        let stored_hash: u64 = parts[0]
            .parse()
            .map_err(|_| DiskCacheError::MalformedMetadata(parts[0].to_string()))?;
        let stored_backend = parts[1];
        let stored_flags: u32 = parts[2]
            .parse()
            .map_err(|_| DiskCacheError::MalformedMetadata(parts[2].to_string()))?;
        Ok(stored_hash == version.source_hash
            && stored_backend == version.backend
            && stored_flags == version.feature_flags)
    }

    fn write_bytes(&self, path: &Path, data: &[u8]) -> std::io::Result<()> {
        let mut f = std::fs::File::create(path)?;
        f.write_all(data)
    }

    fn write_str(&self, path: &Path, s: &str) -> std::io::Result<()> {
        let mut f = std::fs::File::create(path)?;
        f.write_all(s.as_bytes())
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_version(hash: u64) -> ShaderVersion {
        ShaderVersion::new(hash, "vulkan", 0)
    }

    fn make_shader(hash: u64, size: usize) -> CompiledShader {
        CompiledShader::new(vec![0u8; size], make_version(hash))
    }

    #[test]
    fn test_insert_and_get() {
        let mut cache = GpuShaderCache::default();
        let shader = make_shader(1, 100);
        let version = shader.version.clone();
        cache.insert(shader);
        assert!(cache.get(&version).is_some());
    }

    #[test]
    fn test_cache_miss() {
        let mut cache = GpuShaderCache::default();
        let v = make_version(42);
        assert!(cache.get(&v).is_none());
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn test_hit_count_increments() {
        let mut cache = GpuShaderCache::default();
        let shader = make_shader(7, 50);
        let version = shader.version.clone();
        cache.insert(shader);
        cache.get(&version);
        cache.get(&version);
        assert_eq!(
            cache
                .get(&version)
                .expect("cache get should return stored data")
                .hit_count,
            3
        );
    }

    #[test]
    fn test_remove() {
        let mut cache = GpuShaderCache::default();
        let shader = make_shader(99, 200);
        let version = shader.version.clone();
        cache.insert(shader);
        assert!(cache.remove(&version).is_some());
        assert!(cache.get(&version).is_none());
    }

    #[test]
    fn test_contains() {
        let mut cache = GpuShaderCache::default();
        let shader = make_shader(5, 10);
        let version = shader.version.clone();
        assert!(!cache.contains(&version));
        cache.insert(shader);
        assert!(cache.contains(&version));
    }

    #[test]
    fn test_clear() {
        let mut cache = GpuShaderCache::default();
        cache.insert(make_shader(1, 10));
        cache.insert(make_shader(2, 10));
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.stats().total_bytes, 0);
    }

    #[test]
    fn test_eviction_by_entry_count() {
        // Allow at most 2 entries.
        let mut cache = GpuShaderCache::new(usize::MAX, 2, EvictionPolicy::Lfu);
        cache.insert(make_shader(1, 10));
        cache.insert(make_shader(2, 10));
        // Hitting shader 2 raises its hit count so shader 1 gets evicted (LFU).
        cache.get(&make_version(2));
        // Insert a third shader – should evict the LFU entry.
        cache.insert(make_shader(3, 10));
        assert_eq!(cache.len(), 2);
        assert!(cache.stats().evictions >= 1);
    }

    #[test]
    fn test_eviction_by_bytes() {
        // Allow at most 30 bytes.
        let mut cache = GpuShaderCache::new(30, usize::MAX, EvictionPolicy::OldestFirst);
        cache.insert(make_shader(1, 15));
        cache.insert(make_shader(2, 15));
        // Third insert (15 bytes) exceeds the cap – one entry should be evicted.
        cache.insert(make_shader(3, 15));
        assert!(cache.stats().evictions >= 1);
    }

    #[test]
    fn test_invalidate_backend() {
        let mut cache = GpuShaderCache::default();
        let v1 = ShaderVersion::new(1, "vulkan", 0);
        let v2 = ShaderVersion::new(2, "metal", 0);
        cache.insert(CompiledShader::new(vec![0u8; 10], v1));
        cache.insert(CompiledShader::new(vec![0u8; 10], v2.clone()));
        cache.invalidate_backend("vulkan");
        assert!(!cache.contains(&ShaderVersion::new(1, "vulkan", 0)));
        assert!(cache.contains(&v2));
    }

    #[test]
    fn test_hash_source_deterministic() {
        let data = b"hello world shader";
        assert_eq!(hash_source(data), hash_source(data));
    }

    #[test]
    fn test_hash_source_differs_for_different_inputs() {
        assert_ne!(hash_source(b"shader_a"), hash_source(b"shader_b"));
    }

    #[test]
    fn test_default_cache_capacity() {
        let cache = GpuShaderCache::default();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_shader_version_equality() {
        let v1 = ShaderVersion::new(10, "dx12", 3);
        let v2 = ShaderVersion::new(10, "dx12", 3);
        let v3 = ShaderVersion::new(10, "dx12", 4);
        assert_eq!(v1, v2);
        assert_ne!(v1, v3);
    }

    #[test]
    fn test_age_of_is_non_negative() {
        let t = SystemTime::now();
        let age = age_of(t);
        // Age should be very small but non-negative.
        assert!(age < Duration::from_secs(5));
    }

    // ── DiskShaderCache tests ─────────────────────────────────────────────────

    #[test]
    fn test_disk_cache_put_and_get() {
        let dir = std::env::temp_dir().join("oximedia_gpu_disk_cache_test_pg");
        let _ = std::fs::remove_dir_all(&dir); // clean slate
        let mut cache = DiskShaderCache::open(&dir).expect("open disk cache");
        let version = ShaderVersion::new(0xDEAD_BEEF, "vulkan", 7);
        let shader = CompiledShader::new(vec![1, 2, 3, 4, 5], version.clone());
        cache.put(&shader);
        let bytes = cache.get(&version).expect("should find stored shader");
        assert_eq!(bytes, vec![1u8, 2, 3, 4, 5]);
        assert_eq!(cache.stats().disk_writes, 1);
        assert_eq!(cache.stats().disk_hits, 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_disk_cache_miss_unknown_version() {
        let dir = std::env::temp_dir().join("oximedia_gpu_disk_cache_test_miss");
        let _ = std::fs::remove_dir_all(&dir);
        let mut cache = DiskShaderCache::open(&dir).expect("open disk cache");
        let version = ShaderVersion::new(0x1234, "metal", 0);
        assert!(cache.get(&version).is_none());
        assert_eq!(cache.stats().disk_misses, 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_disk_cache_roundtrip_large_bytecode() {
        let dir = std::env::temp_dir().join("oximedia_gpu_disk_cache_test_large");
        let _ = std::fs::remove_dir_all(&dir);
        let mut cache = DiskShaderCache::open(&dir).expect("open disk cache");
        let version = ShaderVersion::new(0xABCD_1234, "dx12", 3);
        let bytecode: Vec<u8> = (0..=255u8).cycle().take(4096).collect();
        let shader = CompiledShader::new(bytecode.clone(), version.clone());
        cache.put(&shader);
        let result = cache.get(&version).expect("should retrieve large blob");
        assert_eq!(result, bytecode);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_disk_cache_version_mismatch_returns_none() {
        let dir = std::env::temp_dir().join("oximedia_gpu_disk_cache_test_mismatch");
        let _ = std::fs::remove_dir_all(&dir);
        let mut cache = DiskShaderCache::open(&dir).expect("open disk cache");
        let v1 = ShaderVersion::new(0xAAAA, "vulkan", 1);
        let v2 = ShaderVersion::new(0xBBBB, "vulkan", 1); // different hash
        cache.put(&CompiledShader::new(vec![0u8; 10], v1));
        // v2 was never written; looking it up must return None.
        assert!(cache.get(&v2).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_disk_cache_clear_removes_all_files() {
        let dir = std::env::temp_dir().join("oximedia_gpu_disk_cache_test_clear");
        let _ = std::fs::remove_dir_all(&dir);
        let mut cache = DiskShaderCache::open(&dir).expect("open disk cache");
        for i in 0u64..5 {
            cache.put(&CompiledShader::new(
                vec![i as u8; 8],
                ShaderVersion::new(i, "vulkan", 0),
            ));
        }
        cache.clear();
        // After clearing, no .shd or .meta files should remain.
        let file_count = std::fs::read_dir(&dir)
            .map(|it| it.flatten().count())
            .unwrap_or(0);
        assert_eq!(
            file_count, 0,
            "expected 0 files after clear, got {file_count}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_disk_cache_invalidate_backend() {
        let dir = std::env::temp_dir().join("oximedia_gpu_disk_cache_test_inval");
        let _ = std::fs::remove_dir_all(&dir);
        let mut cache = DiskShaderCache::open(&dir).expect("open disk cache");
        let v_vulkan = ShaderVersion::new(0x10, "vulkan", 0);
        let v_metal = ShaderVersion::new(0x20, "metal", 0);
        cache.put(&CompiledShader::new(vec![1u8; 8], v_vulkan.clone()));
        cache.put(&CompiledShader::new(vec![2u8; 8], v_metal.clone()));
        cache.invalidate_backend("vulkan");
        // Vulkan entry must be gone; metal must remain.
        assert!(
            cache.get(&v_vulkan).is_none(),
            "vulkan entry should be gone"
        );
        assert!(
            cache.get(&v_metal).is_some(),
            "metal entry should still exist"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_disk_cache_stats_accumulate() {
        let dir = std::env::temp_dir().join("oximedia_gpu_disk_cache_test_stats");
        let _ = std::fs::remove_dir_all(&dir);
        let mut cache = DiskShaderCache::open(&dir).expect("open disk cache");
        let v = ShaderVersion::new(0xFF, "dx12", 0);
        // Miss first.
        cache.get(&v);
        // Write then hit twice.
        cache.put(&CompiledShader::new(vec![7u8; 4], v.clone()));
        cache.get(&v);
        cache.get(&v);
        assert_eq!(cache.stats().disk_misses, 1);
        assert_eq!(cache.stats().disk_writes, 1);
        assert_eq!(cache.stats().disk_hits, 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Shader cache invalidation tests ──────────────────────────────────────

    /// Helper: build a `ShaderVersion` keyed by `source_hash`.
    fn versioned(source: &[u8]) -> ShaderVersion {
        ShaderVersion::new(hash_source(source), "vulkan", 0)
    }

    #[test]
    fn test_invalidation_initial_hit() {
        let mut cache = GpuShaderCache::default();
        let source_v1 = b"// shader version 1\nvoid main() {}";
        let version_v1 = versioned(source_v1);
        let shader = CompiledShader::new(vec![0xAA; 32], version_v1.clone());
        cache.insert(shader);
        // First retrieval must be a hit.
        assert!(cache.get(&version_v1).is_some(), "version 1 must hit");
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().misses, 0);
    }

    #[test]
    fn test_invalidation_different_source_is_miss() {
        let mut cache = GpuShaderCache::default();
        let source_v1 = b"// shader version 1\nvoid main() {}";
        let source_v2 = b"// shader version 2\nvoid main() { discard; }";
        let version_v1 = versioned(source_v1);
        let version_v2 = versioned(source_v2);
        // Insert version 1 only.
        cache.insert(CompiledShader::new(vec![0x11; 16], version_v1.clone()));
        // Looking up version 2 must be a miss.
        assert!(
            cache.get(&version_v2).is_none(),
            "different source hash must be a miss"
        );
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn test_invalidation_old_version_not_accessible_after_remove() {
        let mut cache = GpuShaderCache::default();
        let source_v1 = b"// version 1";
        let source_v2 = b"// version 2";
        let version_v1 = versioned(source_v1);
        let version_v2 = versioned(source_v2);
        // Cache version 1.
        cache.insert(CompiledShader::new(vec![0x01; 8], version_v1.clone()));
        assert!(cache.get(&version_v1).is_some(), "v1 hit");
        // Simulate invalidation: remove v1, insert v2.
        cache.remove(&version_v1);
        cache.insert(CompiledShader::new(vec![0x02; 8], version_v2.clone()));
        // v1 must be gone.
        assert!(
            cache.get(&version_v1).is_none(),
            "old version must not be accessible after remove"
        );
        // v2 must be present.
        assert!(cache.get(&version_v2).is_some(), "new version must hit");
    }

    #[test]
    fn test_invalidation_source_hash_changes_on_whitespace_edit() {
        // Even a single whitespace difference must produce a different hash.
        let source_a = b"void main(){}";
        let source_b = b"void main() {}";
        assert_ne!(
            hash_source(source_a),
            hash_source(source_b),
            "whitespace change must produce different hash"
        );
    }

    #[test]
    fn test_invalidation_disk_cache_version_change() {
        let dir = std::env::temp_dir().join("oximedia_gpu_shader_inval_test");
        let _ = std::fs::remove_dir_all(&dir);
        let mut disk = DiskShaderCache::open(&dir).expect("open disk cache");

        let source_v1 = b"// v1 source";
        let source_v2 = b"// v2 source -- recompiled";
        let version_v1 = ShaderVersion::new(hash_source(source_v1), "vulkan", 0);
        let version_v2 = ShaderVersion::new(hash_source(source_v2), "vulkan", 0);

        // Write version 1.
        disk.put(&CompiledShader::new(vec![0x01; 4], version_v1.clone()));
        // Version 1 must hit.
        assert!(disk.get(&version_v1).is_some(), "v1 disk hit");
        // Version 2 must miss (not yet written).
        assert!(disk.get(&version_v2).is_none(), "v2 disk miss before write");
        // Write version 2.
        disk.put(&CompiledShader::new(vec![0x02; 4], version_v2.clone()));
        // Now version 2 must hit.
        assert!(disk.get(&version_v2).is_some(), "v2 disk hit after write");
        // Version 1 still hits (two independent entries).
        assert!(disk.get(&version_v1).is_some(), "v1 still exists");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_invalidation_clear_invalidates_all() {
        let mut cache = GpuShaderCache::default();
        let v1 = versioned(b"shader A");
        let v2 = versioned(b"shader B");
        cache.insert(CompiledShader::new(vec![1u8; 8], v1.clone()));
        cache.insert(CompiledShader::new(vec![2u8; 8], v2.clone()));
        cache.clear();
        assert!(cache.get(&v1).is_none(), "v1 must be gone after clear");
        assert!(cache.get(&v2).is_none(), "v2 must be gone after clear");
        assert!(cache.is_empty());
    }
}
