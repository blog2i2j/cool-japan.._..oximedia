//! GPU shader cache management.
//!
//! This module provides compiled shader storage, cache eviction, and version
//! tracking to avoid redundant shader compilation work.

use std::collections::HashMap;
use std::time::{Duration, SystemTime};

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
}
