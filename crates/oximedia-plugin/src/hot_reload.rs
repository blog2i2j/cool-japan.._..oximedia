//! Hot-reload plugin management.
//!
//! Enables plugins to be updated at runtime without restarting the host process.
//! File changes are detected via FNV-1a content hashing; actual shared-library
//! unloading and reloading is delegated to the dynamic-loading layer (not performed
//! here — this module focuses on lifecycle management and change detection).
//!
//! # Design
//!
//! - [`HotReloadManager`] owns the set of loaded plugins and their metadata.
//! - [`WatchEntry`] associates a plugin ID with its source path and last known hash.
//! - `check_for_changes` compares hashes of in-memory bytes against stored hashes;
//!   callers supply the content (e.g. freshly read file bytes) for comparison.
//! - [`PluginLifecycle`] is a trait for plugin objects that want to be notified
//!   about load/unload/reload events.
//! - [`GracefulReload`] wraps a drain-then-reload sequence with a configurable timeout.

use crate::error::{PluginError, PluginResult};
use crate::version_resolver::SemVer;
use std::collections::HashMap;
use std::time::Instant;

// ── compute_hash ──────────────────────────────────────────────────────────────

/// Compute a 64-bit FNV-1a hash of `data`.
///
/// FNV-1a is a non-cryptographic hash suitable for change detection.
/// See <http://www.isthe.com/chongo/tech/comp/fnv/>.
pub fn compute_hash(data: &[u8]) -> u64 {
    const FNV_OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
    const FNV_PRIME: u64 = 1_099_511_628_211;

    let mut hash = FNV_OFFSET_BASIS;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Compute a 64-bit FNV-1a hash of the contents of the file at `path`.
///
/// Returns an `io::Error` if the file cannot be read.
pub fn compute_hash_file(path: &std::path::Path) -> std::io::Result<u64> {
    let data = std::fs::read(path)?;
    Ok(compute_hash(&data))
}

// ── ReloadPolicy ─────────────────────────────────────────────────────────────

/// When should the [`HotReloadManager`] attempt to reload a changed plugin?
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReloadPolicy {
    /// Reload as soon as a content change is detected.
    OnChange,
    /// Reload only when explicitly triggered by a signal or API call.
    OnSignal,
    /// Reload on a fixed schedule (millisecond interval).
    Scheduled { interval_ms: u64 },
    /// Hot-reload is disabled; plugins are only loaded once.
    Disabled,
}

// ── PluginVersion ─────────────────────────────────────────────────────────────

/// Metadata about a currently loaded plugin version.
#[derive(Debug, Clone)]
pub struct PluginVersion {
    /// Plugin identifier.
    pub id: String,
    /// Parsed semantic version.
    pub version: SemVer,
    /// FNV-1a hash of the plugin binary (for change detection).
    pub hash: u64,
    /// When this version was loaded.
    pub loaded_at: Instant,
}

impl PluginVersion {
    /// Create a new `PluginVersion`.
    pub fn new(id: impl Into<String>, version: SemVer, hash: u64) -> Self {
        Self {
            id: id.into(),
            version,
            hash,
            loaded_at: Instant::now(),
        }
    }
}

// ── WatchEntry ────────────────────────────────────────────────────────────────

/// Tracks a single plugin file for change detection.
#[derive(Debug, Clone)]
pub struct WatchEntry {
    /// Plugin identifier.
    pub plugin_id: String,
    /// Path (or logical name) of the plugin source.
    pub path: String,
    /// FNV-1a hash of the last-seen content.
    pub last_hash: u64,
}

impl WatchEntry {
    /// Construct a new watch entry for a plugin with the given initial hash.
    pub fn new(plugin_id: impl Into<String>, path: impl Into<String>, initial_hash: u64) -> Self {
        Self {
            plugin_id: plugin_id.into(),
            path: path.into(),
            last_hash: initial_hash,
        }
    }
}

// ── PluginLifecycle ───────────────────────────────────────────────────────────

/// Lifecycle hooks for hot-reloadable plugins.
///
/// Implementors receive notifications at key lifecycle moments so they can
/// flush state, release resources, or migrate data across versions.
pub trait PluginLifecycle {
    /// Called immediately after the plugin is loaded for the first time.
    fn on_load(&mut self);

    /// Called immediately before the plugin is unloaded.
    fn on_unload(&mut self);

    /// Called after the plugin has been reloaded.
    ///
    /// `old_version` carries the metadata of the previous instance so the
    /// plugin can decide whether to perform a data migration.
    fn on_reload(&mut self, old_version: &PluginVersion);
}

// ── HotReloadManager ─────────────────────────────────────────────────────────

/// Manages the hot-reload lifecycle for a set of plugins.
///
/// # Simulation note
///
/// This implementation does not perform actual dynamic library loading; it
/// manages the *metadata* and *change-detection* layer.  The actual dlopen /
/// dlclose operations are performed by the `loader` module (feature-gated on
/// `dynamic-loading`).
pub struct HotReloadManager {
    /// Currently loaded plugins by ID.
    pub loaded_plugins: HashMap<String, PluginVersion>,
    /// Active reload policy.
    pub policy: ReloadPolicy,
    /// File watchers for change detection.
    pub watchers: Vec<WatchEntry>,
    /// Timestamp of the last scheduled check (used with `Scheduled` policy).
    last_check: Instant,
}

impl HotReloadManager {
    /// Create a new manager with the given policy and no loaded plugins.
    pub fn new(policy: ReloadPolicy) -> Self {
        Self {
            loaded_plugins: HashMap::new(),
            policy,
            watchers: Vec::new(),
            last_check: Instant::now(),
        }
    }

    /// Register a plugin as loaded with the given metadata.
    pub fn register_loaded(&mut self, version: PluginVersion) {
        self.loaded_plugins.insert(version.id.clone(), version);
    }

    /// Add a watcher for a plugin path.
    ///
    /// `initial_content` should be the current bytes of the plugin binary so
    /// that the baseline hash is computed correctly.
    pub fn watch(
        &mut self,
        plugin_id: impl Into<String>,
        path: impl Into<String>,
        initial_content: &[u8],
    ) {
        let plugin_id = plugin_id.into();
        let path = path.into();
        let hash = compute_hash(initial_content);
        self.watchers.push(WatchEntry::new(plugin_id, path, hash));
    }

    /// Check for changes by comparing `current_content` (indexed by plugin_id)
    /// against stored hashes.
    ///
    /// Returns the IDs of plugins whose content has changed.  The internal
    /// `last_hash` values are **not** updated here; call `update_hash` after
    /// a successful reload.
    pub fn check_for_changes(&self, current_content: &HashMap<String, Vec<u8>>) -> Vec<String> {
        self.watchers
            .iter()
            .filter_map(|w| {
                let content = current_content.get(&w.plugin_id)?;
                let new_hash = compute_hash(content);
                if new_hash != w.last_hash {
                    Some(w.plugin_id.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Update the stored hash for a plugin after a successful reload.
    pub fn update_hash(&mut self, plugin_id: &str, new_content: &[u8]) {
        let new_hash = compute_hash(new_content);
        for w in &mut self.watchers {
            if w.plugin_id == plugin_id {
                w.last_hash = new_hash;
                return;
            }
        }
    }

    /// Simulate reloading a plugin (metadata update only — no actual dlopen).
    ///
    /// Updates the `loaded_at` timestamp and version hash in the manager's
    /// internal state.
    ///
    /// # Errors
    ///
    /// Returns [`PluginError::NotFound`] if the plugin ID is not currently loaded.
    pub fn reload_plugin(
        &mut self,
        plugin_id: &str,
        new_version: PluginVersion,
    ) -> PluginResult<()> {
        if !self.loaded_plugins.contains_key(plugin_id) {
            return Err(PluginError::NotFound(plugin_id.to_string()));
        }
        self.loaded_plugins
            .insert(plugin_id.to_string(), new_version);
        Ok(())
    }

    /// Unload a plugin, removing it from the manager.
    ///
    /// # Errors
    ///
    /// Returns [`PluginError::NotFound`] if the plugin is not loaded.
    pub fn unload_plugin(&mut self, plugin_id: &str) -> PluginResult<PluginVersion> {
        self.loaded_plugins
            .remove(plugin_id)
            .ok_or_else(|| PluginError::NotFound(plugin_id.to_string()))
    }

    /// Check whether a scheduled reload is due.
    ///
    /// Returns `true` only when the policy is [`ReloadPolicy::Scheduled`] and
    /// the interval has elapsed.  Resets the internal timer on a `true` return.
    pub fn is_scheduled_reload_due(&mut self) -> bool {
        if let ReloadPolicy::Scheduled { interval_ms } = &self.policy {
            let elapsed = self.last_check.elapsed().as_millis() as u64;
            if elapsed >= *interval_ms {
                self.last_check = Instant::now();
                return true;
            }
        }
        false
    }

    /// Return `true` if the policy allows automatic reloading.
    pub fn auto_reload_enabled(&self) -> bool {
        !matches!(self.policy, ReloadPolicy::Disabled)
    }
}

// ── GracefulReload ────────────────────────────────────────────────────────────

/// Performs a drain-then-reload sequence with a configurable drain timeout.
///
/// In a production system, "draining" means waiting for in-flight requests
/// to finish before swapping the plugin.  This struct models that timeout
/// and provides a helper that delegates the actual reload to
/// [`HotReloadManager::reload_plugin`].
pub struct GracefulReload {
    /// Maximum time (ms) to wait for in-flight operations to complete.
    pub drain_timeout_ms: u64,
}

impl GracefulReload {
    /// Create a new graceful-reload helper.
    pub fn new(drain_timeout_ms: u64) -> Self {
        Self { drain_timeout_ms }
    }

    /// Drain active operations (simulated) then reload the plugin.
    ///
    /// This implementation uses a simple busy-wait simulation:
    /// - If `drain_timeout_ms` is zero the drain is considered instant.
    /// - Otherwise the caller is assumed to have already drained; the method
    ///   just records the elapsed time and proceeds.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`HotReloadManager::reload_plugin`].
    pub fn drain_and_reload(
        &self,
        plugin_id: &str,
        manager: &mut HotReloadManager,
        new_version: PluginVersion,
    ) -> PluginResult<()> {
        // In a real implementation this would:
        //  1. Signal the plugin to stop accepting new work.
        //  2. Wait up to `drain_timeout_ms` for in-flight operations to finish.
        //  3. Force-terminate remaining operations if the deadline is exceeded.
        //
        // Here we simulate a successful drain (no actual I/O / threading).
        let _start = Instant::now();
        manager.reload_plugin(plugin_id, new_version)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_version(id: &str, major: u32, minor: u32, patch: u32) -> PluginVersion {
        PluginVersion::new(id, SemVer::new(major, minor, patch), 0)
    }

    // 1. compute_hash is deterministic
    #[test]
    fn test_hash_deterministic() {
        let h1 = compute_hash(b"hello");
        let h2 = compute_hash(b"hello");
        assert_eq!(h1, h2);
    }

    // 2. compute_hash differs for different content
    #[test]
    fn test_hash_distinct() {
        assert_ne!(compute_hash(b"foo"), compute_hash(b"bar"));
    }

    // 3. compute_hash of empty slice is the FNV offset basis
    #[test]
    fn test_hash_empty() {
        let h = compute_hash(&[]);
        assert_eq!(h, 14_695_981_039_346_656_037);
    }

    // 4. HotReloadManager::new has no loaded plugins
    #[test]
    fn test_manager_new_empty() {
        let m = HotReloadManager::new(ReloadPolicy::OnChange);
        assert!(m.loaded_plugins.is_empty());
        assert!(m.watchers.is_empty());
    }

    // 5. register_loaded stores a plugin
    #[test]
    fn test_register_loaded() {
        let mut m = HotReloadManager::new(ReloadPolicy::Disabled);
        m.register_loaded(make_version("codec-a", 1, 0, 0));
        assert!(m.loaded_plugins.contains_key("codec-a"));
    }

    // 6. watch creates a watcher with correct hash
    #[test]
    fn test_watch_hash() {
        let mut m = HotReloadManager::new(ReloadPolicy::OnChange);
        let content = b"binary data";
        m.watch("plug", "/lib/plug.so", content);
        assert_eq!(m.watchers.len(), 1);
        assert_eq!(m.watchers[0].last_hash, compute_hash(content));
    }

    // 7. check_for_changes: no change → empty list
    #[test]
    fn test_no_changes() {
        let mut m = HotReloadManager::new(ReloadPolicy::OnChange);
        let content = b"same content";
        m.watch("p", "/lib/p.so", content);

        let mut current = HashMap::new();
        current.insert("p".to_string(), content.to_vec());

        let changed = m.check_for_changes(&current);
        assert!(changed.is_empty());
    }

    // 8. check_for_changes: changed content → plugin ID returned
    #[test]
    fn test_change_detected() {
        let mut m = HotReloadManager::new(ReloadPolicy::OnChange);
        m.watch("p", "/lib/p.so", b"v1");

        let mut current = HashMap::new();
        current.insert("p".to_string(), b"v2".to_vec());

        let changed = m.check_for_changes(&current);
        assert_eq!(changed, vec!["p".to_string()]);
    }

    // 9. update_hash clears the changed flag
    #[test]
    fn test_update_hash_clears_change() {
        let mut m = HotReloadManager::new(ReloadPolicy::OnChange);
        m.watch("p", "/lib/p.so", b"v1");
        m.update_hash("p", b"v2");

        let mut current = HashMap::new();
        current.insert("p".to_string(), b"v2".to_vec());

        let changed = m.check_for_changes(&current);
        assert!(changed.is_empty());
    }

    // 10. reload_plugin updates the loaded version
    #[test]
    fn test_reload_plugin() {
        let mut m = HotReloadManager::new(ReloadPolicy::OnChange);
        m.register_loaded(make_version("p", 1, 0, 0));

        let new_v = make_version("p", 1, 1, 0);
        m.reload_plugin("p", new_v).expect("reload");

        assert_eq!(m.loaded_plugins["p"].version, SemVer::new(1, 1, 0));
    }

    // 11. reload_plugin on unknown ID → NotFound
    #[test]
    fn test_reload_unknown() {
        let mut m = HotReloadManager::new(ReloadPolicy::OnChange);
        let err = m.reload_plugin("ghost", make_version("ghost", 1, 0, 0));
        assert!(matches!(err, Err(PluginError::NotFound(_))));
    }

    // 12. unload_plugin removes the plugin
    #[test]
    fn test_unload_plugin() {
        let mut m = HotReloadManager::new(ReloadPolicy::Disabled);
        m.register_loaded(make_version("p", 1, 0, 0));
        m.unload_plugin("p").expect("unload");
        assert!(!m.loaded_plugins.contains_key("p"));
    }

    // 13. unload_plugin on unknown ID → NotFound
    #[test]
    fn test_unload_unknown() {
        let mut m = HotReloadManager::new(ReloadPolicy::Disabled);
        assert!(matches!(
            m.unload_plugin("ghost"),
            Err(PluginError::NotFound(_))
        ));
    }

    // 14. auto_reload_enabled: Disabled → false
    #[test]
    fn test_auto_reload_disabled() {
        let m = HotReloadManager::new(ReloadPolicy::Disabled);
        assert!(!m.auto_reload_enabled());
    }

    // 15. auto_reload_enabled: OnChange → true
    #[test]
    fn test_auto_reload_on_change() {
        let m = HotReloadManager::new(ReloadPolicy::OnChange);
        assert!(m.auto_reload_enabled());
    }

    // 16. is_scheduled_reload_due: not due with large interval
    #[test]
    fn test_scheduled_not_due() {
        let mut m = HotReloadManager::new(ReloadPolicy::Scheduled {
            interval_ms: 1_000_000,
        });
        assert!(!m.is_scheduled_reload_due());
    }

    // 17. GracefulReload::drain_and_reload succeeds
    #[test]
    fn test_graceful_reload_ok() {
        let mut m = HotReloadManager::new(ReloadPolicy::OnChange);
        m.register_loaded(make_version("p", 1, 0, 0));

        let gr = GracefulReload::new(100);
        let new_v = make_version("p", 1, 2, 0);
        gr.drain_and_reload("p", &mut m, new_v)
            .expect("graceful reload");

        assert_eq!(m.loaded_plugins["p"].version, SemVer::new(1, 2, 0));
    }

    // 18. GracefulReload on unknown plugin propagates NotFound
    #[test]
    fn test_graceful_reload_not_found() {
        let mut m = HotReloadManager::new(ReloadPolicy::OnChange);
        let gr = GracefulReload::new(0);
        let err = gr.drain_and_reload("ghost", &mut m, make_version("ghost", 1, 0, 0));
        assert!(matches!(err, Err(PluginError::NotFound(_))));
    }

    // 19. WatchEntry stores correct path
    #[test]
    fn test_watch_entry_path() {
        let w = WatchEntry::new("plug", "/some/path.so", 0xABCD);
        assert_eq!(w.plugin_id, "plug");
        assert_eq!(w.path, "/some/path.so");
        assert_eq!(w.last_hash, 0xABCD);
    }

    // 20. Multiple watchers, only changed ones returned
    #[test]
    fn test_multiple_watchers_selective() {
        let mut m = HotReloadManager::new(ReloadPolicy::OnChange);
        m.watch("a", "/lib/a.so", b"v1a");
        m.watch("b", "/lib/b.so", b"v1b");

        let mut current = HashMap::new();
        current.insert("a".to_string(), b"v2a".to_vec()); // changed
        current.insert("b".to_string(), b"v1b".to_vec()); // unchanged

        let changed = m.check_for_changes(&current);
        assert_eq!(changed.len(), 1);
        assert_eq!(changed[0], "a");
    }

    // ── compute_hash_mmap tests ────────────────────────────────────────────────

    use std::io::Write;

    fn write_temp_file(content: &[u8]) -> (tempfile::NamedTempFile, std::path::PathBuf) {
        let mut f = tempfile::NamedTempFile::new().expect("tempfile");
        f.write_all(content).expect("write");
        f.flush().expect("flush");
        let path = f.path().to_path_buf();
        (f, path)
    }

    // 21. compute_hash_mmap: small file matches compute_hash_file
    #[test]
    fn test_mmap_small_file_matches_file_hash() {
        let content = b"small plugin binary data for testing";
        let (_f, path) = write_temp_file(content);
        let mmap_hash = super::compute_hash_mmap(&path).expect("mmap hash");
        let file_hash = super::compute_hash_file(&path).expect("file hash");
        assert_eq!(mmap_hash, file_hash);
    }

    // 22. compute_hash_mmap: result matches compute_hash on same bytes
    #[test]
    fn test_mmap_hash_matches_in_memory_hash() {
        let content = b"reference content for hash comparison";
        let (_f, path) = write_temp_file(content);
        let mmap_hash = super::compute_hash_mmap(&path).expect("mmap hash");
        let direct_hash = super::compute_hash(content);
        assert_eq!(mmap_hash, direct_hash);
    }

    // 23. compute_hash_mmap: empty file returns FNV offset basis
    #[test]
    fn test_mmap_empty_file_returns_offset_basis() {
        let (_f, path) = write_temp_file(b"");
        let hash = super::compute_hash_mmap(&path).expect("empty mmap hash");
        assert_eq!(hash, super::compute_hash(&[]));
    }

    // 24. compute_hash_mmap: different content produces different hashes
    #[test]
    fn test_mmap_distinct_content_produces_distinct_hashes() {
        let (_f1, path1) = write_temp_file(b"content version one");
        let (_f2, path2) = write_temp_file(b"content version two");
        let h1 = super::compute_hash_mmap(&path1).expect("h1");
        let h2 = super::compute_hash_mmap(&path2).expect("h2");
        assert_ne!(h1, h2);
    }

    // 25. compute_hash_mmap: deterministic (same file twice produces same hash)
    #[test]
    fn test_mmap_deterministic() {
        let content = b"deterministic test content";
        let (_f, path) = write_temp_file(content);
        let h1 = super::compute_hash_mmap(&path).expect("first");
        let h2 = super::compute_hash_mmap(&path).expect("second");
        assert_eq!(h1, h2);
    }

    // 26. compute_hash_mmap: nonexistent file returns error
    #[test]
    fn test_mmap_nonexistent_file_returns_error() {
        let result = super::compute_hash_mmap(std::path::Path::new("/nonexistent/path/plugin.so"));
        assert!(result.is_err());
    }

    // 27. compute_hash_mmap: single-byte file
    #[test]
    fn test_mmap_single_byte_file() {
        let (_f, path) = write_temp_file(b"\xFF");
        let hash = super::compute_hash_mmap(&path).expect("single byte");
        assert_eq!(hash, super::compute_hash(b"\xFF"));
    }

    // 28. MMAP_THRESHOLD_BYTES is at least 1 MiB (meaningful threshold)
    #[test]
    fn test_mmap_threshold_is_reasonable() {
        assert!(super::MMAP_THRESHOLD_BYTES >= 1024 * 1024);
    }

    // 29. compute_hash_mmap: file below threshold matches direct hash
    #[test]
    fn test_mmap_below_threshold_matches_direct_hash() {
        let content: Vec<u8> = (0..1024).map(|i| (i % 251) as u8).collect();
        let (_f, path) = write_temp_file(&content);
        let mmap_hash = super::compute_hash_mmap(&path).expect("below threshold");
        let direct_hash = super::compute_hash(&content);
        assert_eq!(mmap_hash, direct_hash);
    }
}

// ── Streaming large-file hash (mmap-equivalent) ──────────────────────────────

/// Minimum file size (in bytes) at which [`compute_hash_mmap`] switches from
/// a full heap allocation to a streaming, page-sized I/O strategy.
///
/// Files smaller than this threshold are hashed via a single `fs::read`
/// (same as [`compute_hash_file`]).  Files at or above this threshold are
/// read in fixed-size chunks equal to the OS page size, so only one page
/// of data resides in heap memory at a time — analogous to what a memory
/// map would provide, but without requiring an `unsafe` block.
pub const MMAP_THRESHOLD_BYTES: u64 = 4 * 1024 * 1024; // 4 MiB

/// Size of each I/O chunk used by [`compute_hash_mmap`] for large files.
///
/// Matches a typical OS page size (4 KiB) so that each read corresponds
/// to one demand-paged physical page — the same granularity a true
/// memory-mapped implementation would process.
pub const MMAP_CHUNK_SIZE: usize = 4096; // 4 KiB (one OS page)

/// Compute a 64-bit FNV-1a hash of `path` using page-sized streaming I/O
/// for large files.
///
/// # Strategy
///
/// - If the file size is **below** [`MMAP_THRESHOLD_BYTES`], the entire
///   content is read into a heap buffer (same as [`compute_hash_file`]).
/// - If the file size is **at or above** the threshold, the file is read
///   in [`MMAP_CHUNK_SIZE`]-byte chunks.  Only one chunk is in memory at a
///   time, achieving the same low-memory-footprint goal as memory-mapped I/O
///   without requiring an `unsafe` block.  This is equivalent to
///   demand-paging one OS page at a time.
///
/// # Errors
///
/// Returns `std::io::Error` if the file cannot be opened, stat'd, or read.
pub fn compute_hash_mmap(path: &std::path::Path) -> std::io::Result<u64> {
    use std::io::Read;

    let metadata = std::fs::metadata(path)?;
    let file_size = metadata.len();

    if file_size == 0 {
        // Empty file — FNV offset basis (same as compute_hash(&[]))
        return Ok(14_695_981_039_346_656_037);
    }

    if file_size < MMAP_THRESHOLD_BYTES {
        // Small file: read into heap, no chunking overhead needed.
        let data = std::fs::read(path)?;
        return Ok(compute_hash(&data));
    }

    // Large file: stream in page-sized chunks, accumulating FNV-1a hash.
    // This mirrors memory-mapped page-fault semantics: only one page of data
    // occupies physical RAM at any given moment.
    const FNV_OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
    const FNV_PRIME: u64 = 1_099_511_628_211;

    let mut file = std::fs::File::open(path)?;
    let mut buf = vec![0u8; MMAP_CHUNK_SIZE];
    let mut hash = FNV_OFFSET_BASIS;

    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        for &byte in &buf[..n] {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }

    Ok(hash)
}
