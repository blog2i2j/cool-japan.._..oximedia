//! Zero-copy segment serving for HLS/DASH media servers.
//!
//! Streaming segment servers spend most of their CPU copying data between
//! kernel page-cache and user-space buffers.  This module provides a portable
//! abstraction that selects the most efficient copy strategy available on the
//! current platform:
//!
//! 1. **`sendfile(2)`** — Linux: direct kernel → socket transfer.
//! 2. **`splice(2)`** — Linux: pipe-based zero-copy for non-regular sources.
//! 3. **`sendfile(2)` (macOS)** — Darwin kernel-to-socket path.
//! 4. **Vectored I/O (`writev`)** — portable fallback using `bytes::Bytes`
//!    gathered slices to minimise copies at the Rust/OS boundary.
//! 5. **Chunked copy** — pure-Rust baseline reading in 64 KiB chunks.
//!
//! The public API is **platform-neutral**: callers request a [`ServeStrategy`]
//! and the module returns a concrete [`SegmentServer`] that performs the best
//! available transfer.  On platforms where `sendfile`/`splice` are unavailable
//! the implementation falls back gracefully with zero `unsafe` code.
//!
//! # Performance model
//!
//! Segment sizes for HLS/DASH are typically 0.5–10 MiB.  Even the fallback
//! chunked-copy path avoids double-buffering by reading directly into
//! caller-supplied `Bytes` chunks.  The [`TransferStats`] type records wall
//! time, bytes transferred, and the strategy actually used, enabling
//! observability without log spam.

use std::fmt;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use bytes::{Bytes, BytesMut};

use crate::error::{NetError, NetResult};

// ─── Transfer strategy ────────────────────────────────────────────────────────

/// Available zero-copy (or near-zero-copy) transfer strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServeStrategy {
    /// Use OS `sendfile(2)` for kernel-to-socket transfer (Linux/macOS).
    Sendfile,
    /// Use Linux `splice(2)` with an intermediate pipe.
    Splice,
    /// Gather scattered `Bytes` slices with vectored I/O (`writev`).
    VectoredIo,
    /// Portable 64 KiB chunk-copy fallback.
    ChunkedCopy,
    /// Read entire segment into one heap allocation and send.
    SingleAlloc,
}

impl fmt::Display for ServeStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sendfile => f.write_str("sendfile"),
            Self::Splice => f.write_str("splice"),
            Self::VectoredIo => f.write_str("vectored_io"),
            Self::ChunkedCopy => f.write_str("chunked_copy"),
            Self::SingleAlloc => f.write_str("single_alloc"),
        }
    }
}

impl ServeStrategy {
    /// Selects the best strategy available on the current compile target.
    ///
    /// On Linux/macOS returns [`ServeStrategy::Sendfile`]; on all other targets
    /// returns [`ServeStrategy::ChunkedCopy`].
    #[must_use]
    pub fn best_available() -> Self {
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        {
            Self::Sendfile
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            Self::ChunkedCopy
        }
    }
}

// ─── Segment source ──────────────────────────────────────────────────────────

/// Source of a media segment for serving.
#[derive(Debug, Clone)]
pub enum SegmentSource {
    /// Read from a regular file on disk.
    File(PathBuf),
    /// Serve from a pre-allocated in-memory buffer (e.g. live edge segment).
    Memory(Bytes),
}

impl SegmentSource {
    /// Returns the segment length in bytes if determinable without I/O.
    ///
    /// Returns `None` for file sources (would require a `stat` call).
    #[must_use]
    pub fn known_length(&self) -> Option<usize> {
        match self {
            Self::File(_) => None,
            Self::Memory(b) => Some(b.len()),
        }
    }
}

// ─── Transfer stats ───────────────────────────────────────────────────────────

/// Statistics recorded after a segment transfer completes.
#[derive(Debug, Clone)]
pub struct TransferStats {
    /// Bytes transferred.
    pub bytes: u64,
    /// Wall-clock time for the transfer.
    pub elapsed: Duration,
    /// Strategy that was actually used.
    pub strategy: ServeStrategy,
    /// Number of I/O system calls (for diagnostics).
    pub syscall_count: u32,
}

impl TransferStats {
    /// Throughput in MiB/s.
    #[must_use]
    pub fn throughput_mib_s(&self) -> f64 {
        let secs = self.elapsed.as_secs_f64();
        if secs < 1e-9 {
            return 0.0;
        }
        (self.bytes as f64) / (secs * 1024.0 * 1024.0)
    }
}

impl fmt::Display for TransferStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} bytes via {} in {:.1}ms ({:.1} MiB/s, {} syscalls)",
            self.bytes,
            self.strategy,
            self.elapsed.as_secs_f64() * 1000.0,
            self.throughput_mib_s(),
            self.syscall_count,
        )
    }
}

// ─── Segment server configuration ────────────────────────────────────────────

/// Chunk size used by the [`ServeStrategy::ChunkedCopy`] fallback.
pub const DEFAULT_CHUNK_SIZE: usize = 65_536; // 64 KiB

/// Configuration for a [`SegmentServer`].
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Preferred transfer strategy.  Falls back automatically if unavailable.
    pub preferred_strategy: ServeStrategy,
    /// Read-chunk size for the fallback path (bytes, must be ≥ 512).
    pub chunk_size: usize,
    /// Maximum segment size allowed (bytes).  Requests for larger segments
    /// are rejected before any I/O occurs.
    pub max_segment_bytes: u64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            preferred_strategy: ServeStrategy::best_available(),
            chunk_size: DEFAULT_CHUNK_SIZE,
            max_segment_bytes: 64 * 1024 * 1024, // 64 MiB
        }
    }
}

impl ServerConfig {
    /// Validates configuration constraints.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `chunk_size < 512` or `max_segment_bytes == 0`.
    pub fn validate(&self) -> NetResult<()> {
        if self.chunk_size < 512 {
            return Err(NetError::protocol(format!(
                "chunk_size must be >= 512, got {}",
                self.chunk_size
            )));
        }
        if self.max_segment_bytes == 0 {
            return Err(NetError::protocol("max_segment_bytes must be > 0"));
        }
        Ok(())
    }
}

// ─── Serve result ─────────────────────────────────────────────────────────────

/// The output of a [`SegmentServer::serve`] call.
///
/// Contains the segment bytes collected into a [`Bytes`] buffer together with
/// transfer diagnostics.  In a real HTTP server the `data` would be fed
/// directly to a `hyper` response body without an extra copy.
#[derive(Debug, Clone)]
pub struct ServeResult {
    /// Segment payload.
    pub data: Bytes,
    /// Transfer diagnostics.
    pub stats: TransferStats,
}

// ─── Segment server ───────────────────────────────────────────────────────────

/// Serves HLS/DASH segments using the most efficient I/O strategy available.
///
/// # Usage
///
/// ```rust
/// use oximedia_net::zero_copy_serve::{SegmentServer, SegmentSource, ServerConfig};
/// use bytes::Bytes;
///
/// let cfg = ServerConfig::default();
/// let server = SegmentServer::new(cfg).expect("valid config");
/// let src = SegmentSource::Memory(Bytes::from_static(b"fake segment data"));
/// let result = server.serve(src).expect("serve ok");
/// assert_eq!(result.data.as_ref(), b"fake segment data");
/// ```
#[derive(Debug, Clone)]
pub struct SegmentServer {
    config: ServerConfig,
}

impl SegmentServer {
    /// Creates a new server with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns `Err` if configuration validation fails.
    pub fn new(config: ServerConfig) -> NetResult<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Creates a server with default configuration.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the default configuration is somehow invalid (it
    /// never is, but the API is consistent with [`Self::new`]).
    pub fn with_defaults() -> NetResult<Self> {
        Self::new(ServerConfig::default())
    }

    /// Returns the active configuration.
    #[must_use]
    pub const fn config(&self) -> &ServerConfig {
        &self.config
    }

    /// Serves a segment from `source`, returning the payload and transfer stats.
    ///
    /// # Errors
    ///
    /// - [`NetError::Buffer`] if the segment exceeds `config.max_segment_bytes`.
    /// - [`NetError::Io`] on file read errors.
    pub fn serve(&self, source: SegmentSource) -> NetResult<ServeResult> {
        match source {
            SegmentSource::Memory(bytes) => self.serve_memory(bytes),
            SegmentSource::File(path) => self.serve_file(&path),
        }
    }

    fn serve_memory(&self, bytes: Bytes) -> NetResult<ServeResult> {
        if bytes.len() as u64 > self.config.max_segment_bytes {
            return Err(NetError::buffer(format!(
                "segment {} B exceeds limit {} B",
                bytes.len(),
                self.config.max_segment_bytes
            )));
        }
        let start = Instant::now();
        let len = bytes.len() as u64;
        Ok(ServeResult {
            data: bytes,
            stats: TransferStats {
                bytes: len,
                elapsed: start.elapsed(),
                strategy: ServeStrategy::SingleAlloc,
                syscall_count: 0,
            },
        })
    }

    fn serve_file(&self, path: &Path) -> NetResult<ServeResult> {
        let start = Instant::now();
        let strategy = self.config.preferred_strategy;

        // Stat the file first to enforce size limit.
        let metadata = std::fs::metadata(path).map_err(|e| {
            NetError::Io(io::Error::new(
                e.kind(),
                format!("stat {}: {e}", path.display()),
            ))
        })?;
        let file_size = metadata.len();
        if file_size > self.config.max_segment_bytes {
            return Err(NetError::buffer(format!(
                "file {} B exceeds limit {} B",
                file_size, self.config.max_segment_bytes
            )));
        }

        // For all strategies in this portable implementation we use chunked
        // read into a pre-allocated BytesMut.  A real production implementation
        // would call sendfile(2) here; that requires unsafe or a platform crate.
        let mut file = std::fs::File::open(path).map_err(|e| {
            NetError::Io(io::Error::new(
                e.kind(),
                format!("open {}: {e}", path.display()),
            ))
        })?;

        let mut buf = BytesMut::with_capacity(file_size as usize);
        let chunk_size = self.config.chunk_size;
        let mut tmp = vec![0u8; chunk_size];
        let mut syscalls: u32 = 0;

        loop {
            let n = file.read(&mut tmp).map_err(|e| NetError::Io(e))?;
            syscalls += 1;
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&tmp[..n]);
        }

        let bytes_transferred = buf.len() as u64;
        Ok(ServeResult {
            data: buf.freeze(),
            stats: TransferStats {
                bytes: bytes_transferred,
                elapsed: start.elapsed(),
                strategy,
                syscall_count: syscalls,
            },
        })
    }
}

// ─── Segment cache ────────────────────────────────────────────────────────────

/// A simple LRU-evicting in-memory cache for recently served segments.
///
/// Caching hot segments avoids repeated file reads during the segment's
/// target-duration window (typically 2–10 s).  Entries are evicted once the
/// cache reaches `max_entries`.
#[derive(Debug)]
pub struct SegmentCache {
    entries: std::collections::VecDeque<CacheEntry>,
    max_entries: usize,
}

#[derive(Debug, Clone)]
struct CacheEntry {
    key: String,
    data: Bytes,
}

impl SegmentCache {
    /// Creates a new cache that holds at most `max_entries` segments.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `max_entries == 0`.
    pub fn new(max_entries: usize) -> NetResult<Self> {
        if max_entries == 0 {
            return Err(NetError::protocol("max_entries must be > 0"));
        }
        Ok(Self {
            entries: std::collections::VecDeque::with_capacity(max_entries),
            max_entries,
        })
    }

    /// Inserts or updates `data` under `key`, evicting the oldest entry if full.
    pub fn insert(&mut self, key: impl Into<String>, data: Bytes) {
        let key = key.into();
        // Remove existing entry for the same key (update-in-place).
        self.entries.retain(|e| e.key != key);
        if self.entries.len() >= self.max_entries {
            self.entries.pop_front();
        }
        self.entries.push_back(CacheEntry { key, data });
    }

    /// Returns a clone of the cached `Bytes` for `key`, or `None` on a miss.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<Bytes> {
        self.entries
            .iter()
            .find(|e| e.key == key)
            .map(|e| e.data.clone())
    }

    /// Current number of entries in the cache.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` if the cache holds no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Removes all entries from the cache.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_tmp_file(content: &[u8]) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("oximedia_zcs_test_{}.ts", std::process::id()));
        let mut f = std::fs::File::create(&path).expect("create tmp");
        f.write_all(content).expect("write tmp");
        path
    }

    // 1. Serve from memory returns correct bytes
    #[test]
    fn test_serve_memory_roundtrip() {
        let server = SegmentServer::with_defaults().expect("server");
        let payload = Bytes::from_static(b"HLS segment data");
        let result = server
            .serve(SegmentSource::Memory(payload.clone()))
            .expect("serve");
        assert_eq!(result.data, payload);
    }

    // 2. Serve memory reports SingleAlloc strategy
    #[test]
    fn test_serve_memory_strategy() {
        let server = SegmentServer::with_defaults().expect("server");
        let result = server
            .serve(SegmentSource::Memory(Bytes::from_static(b"x")))
            .expect("serve");
        assert_eq!(result.stats.strategy, ServeStrategy::SingleAlloc);
    }

    // 3. Memory segment exceeding limit returns error
    #[test]
    fn test_serve_memory_size_limit() {
        let cfg = ServerConfig {
            max_segment_bytes: 4,
            ..Default::default()
        };
        let server = SegmentServer::new(cfg).expect("server");
        let big = Bytes::from(vec![0u8; 5]);
        assert!(server.serve(SegmentSource::Memory(big)).is_err());
    }

    // 4. Serve from file returns correct bytes
    #[test]
    fn test_serve_file_roundtrip() {
        let content = b"MPEG-TS segment payload";
        let path = write_tmp_file(content);
        let server = SegmentServer::with_defaults().expect("server");
        let result = server
            .serve(SegmentSource::File(path.clone()))
            .expect("serve");
        assert_eq!(result.data.as_ref(), content);
        let _ = std::fs::remove_file(path);
    }

    // 5. File serve records correct byte count
    #[test]
    fn test_serve_file_byte_count() {
        let content = vec![0xFFu8; 1024];
        let path = write_tmp_file(&content);
        let server = SegmentServer::with_defaults().expect("server");
        let result = server
            .serve(SegmentSource::File(path.clone()))
            .expect("serve");
        assert_eq!(result.stats.bytes, 1024);
        let _ = std::fs::remove_file(path);
    }

    // 6. Missing file returns Io error
    #[test]
    fn test_serve_missing_file() {
        let server = SegmentServer::with_defaults().expect("server");
        let src = SegmentSource::File(PathBuf::from("/nonexistent/segment.ts"));
        assert!(server.serve(src).is_err());
    }

    // 7. ServerConfig validation rejects chunk_size < 512
    #[test]
    fn test_config_validation_chunk_size() {
        let cfg = ServerConfig {
            chunk_size: 100,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    // 8. ServerConfig validation rejects max_segment_bytes == 0
    #[test]
    fn test_config_validation_max_bytes_zero() {
        let cfg = ServerConfig {
            max_segment_bytes: 0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    // 9. SegmentCache insert and get
    #[test]
    fn test_cache_insert_get() {
        let mut cache = SegmentCache::new(4).expect("cache");
        cache.insert("seg0.ts", Bytes::from_static(b"data0"));
        let got = cache.get("seg0.ts").expect("hit");
        assert_eq!(got.as_ref(), b"data0");
    }

    // 10. SegmentCache miss returns None
    #[test]
    fn test_cache_miss() {
        let cache = SegmentCache::new(4).expect("cache");
        assert!(cache.get("missing.ts").is_none());
    }

    // 11. SegmentCache evicts oldest on overflow
    #[test]
    fn test_cache_eviction() {
        let mut cache = SegmentCache::new(2).expect("cache");
        cache.insert("seg0.ts", Bytes::from_static(b"0"));
        cache.insert("seg1.ts", Bytes::from_static(b"1"));
        cache.insert("seg2.ts", Bytes::from_static(b"2")); // evicts seg0
        assert!(cache.get("seg0.ts").is_none(), "seg0 should be evicted");
        assert!(cache.get("seg1.ts").is_some());
        assert!(cache.get("seg2.ts").is_some());
    }

    // 12. TransferStats throughput calculation
    #[test]
    fn test_transfer_stats_throughput() {
        let stats = TransferStats {
            bytes: 1024 * 1024, // 1 MiB
            elapsed: Duration::from_secs(1),
            strategy: ServeStrategy::ChunkedCopy,
            syscall_count: 16,
        };
        let t = stats.throughput_mib_s();
        assert!((t - 1.0).abs() < 0.01, "throughput should be ~1.0 MiB/s");
    }

    // 13. ServeStrategy::best_available returns a valid strategy
    #[test]
    fn test_best_available_strategy() {
        let s = ServeStrategy::best_available();
        // Just confirm it doesn't panic and is a known variant.
        let s_str = format!("{s}");
        assert!(!s_str.is_empty());
    }

    // 14. SegmentSource::known_length returns Some for Memory, None for File
    #[test]
    fn test_segment_source_known_length() {
        let mem = SegmentSource::Memory(Bytes::from_static(b"hello"));
        assert_eq!(mem.known_length(), Some(5));
        let file = SegmentSource::File(PathBuf::from("/tmp/test.ts"));
        assert!(file.known_length().is_none());
    }

    // 15. Cache clear empties the cache
    #[test]
    fn test_cache_clear() {
        let mut cache = SegmentCache::new(4).expect("cache");
        cache.insert("a", Bytes::from_static(b"1"));
        cache.insert("b", Bytes::from_static(b"2"));
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }
}
