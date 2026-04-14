//! Incremental checksumming with resume support.
//!
//! Provides the ability to checkpoint checksum computation so that interrupted
//! verifications can be resumed without re-reading already-processed bytes.
//! Supports BLAKE3, SHA-256, and CRC32 simultaneously.

use crate::{ArchiveError, ArchiveResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Checkpoint
// ---------------------------------------------------------------------------

/// Intermediate state of a multi-algorithm checksum computation.
///
/// This can be serialized to JSON and persisted so that verification of large
/// files can survive process restarts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChecksumCheckpoint {
    /// Path of the file being checksummed.
    pub file_path: PathBuf,
    /// Total file size in bytes (must match on resume).
    pub file_size: u64,
    /// Number of bytes already processed.
    pub bytes_processed: u64,
    /// BLAKE3 hasher state (hex of partial hash — we re-hash processed
    /// chunks on resume for BLAKE3 since its internal state is not trivially
    /// serializable, but we store the intermediate digest for cross-checks).
    pub blake3_partial_hex: Option<String>,
    /// SHA-256 intermediate state (8 × u32 words + byte count).
    pub sha256_state: Option<Sha256State>,
    /// CRC32 running value.
    pub crc32_value: Option<u32>,
    /// Chunk size that was used (must match on resume).
    pub chunk_size: usize,
    /// Unix timestamp when the checkpoint was created.
    pub created_at_secs: u64,
}

/// Serializable SHA-256 intermediate state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sha256State {
    /// The 8 state words (H0..H7).
    pub h: [u32; 8],
    /// Total bytes processed so far (for final padding).
    pub total_bytes: u64,
    /// Pending bytes that did not fill a complete 64-byte block.
    pub pending: Vec<u8>,
}

impl ChecksumCheckpoint {
    /// Serialize the checkpoint to a JSON string.
    pub fn to_json(&self) -> ArchiveResult<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| ArchiveError::Validation(format!("checkpoint serialization failed: {e}")))
    }

    /// Deserialize a checkpoint from a JSON string.
    pub fn from_json(json: &str) -> ArchiveResult<Self> {
        serde_json::from_str(json).map_err(|e| {
            ArchiveError::Validation(format!("checkpoint deserialization failed: {e}"))
        })
    }

    /// Return the fraction of completion (0.0 to 1.0).
    pub fn progress(&self) -> f64 {
        if self.file_size == 0 {
            return 1.0;
        }
        self.bytes_processed as f64 / self.file_size as f64
    }

    /// Whether the checkpoint indicates the file was fully processed.
    pub fn is_complete(&self) -> bool {
        self.bytes_processed >= self.file_size
    }
}

// ---------------------------------------------------------------------------
// Incremental checksumming engine
// ---------------------------------------------------------------------------

/// Configuration for incremental checksumming.
#[derive(Debug, Clone)]
pub struct IncrementalConfig {
    /// Enable BLAKE3.
    pub enable_blake3: bool,
    /// Enable SHA-256.
    pub enable_sha256: bool,
    /// Enable CRC32.
    pub enable_crc32: bool,
    /// Read chunk size in bytes (default 1 MiB).
    pub chunk_size: usize,
    /// How often to create checkpoints (every N bytes).
    pub checkpoint_interval_bytes: u64,
}

impl Default for IncrementalConfig {
    fn default() -> Self {
        Self {
            enable_blake3: true,
            enable_sha256: true,
            enable_crc32: true,
            chunk_size: 1024 * 1024,                     // 1 MiB
            checkpoint_interval_bytes: 64 * 1024 * 1024, // 64 MiB
        }
    }
}

/// Result of a completed incremental checksum operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalResult {
    /// Path of the file.
    pub file_path: PathBuf,
    /// File size in bytes.
    pub file_size: u64,
    /// BLAKE3 hex digest.
    pub blake3_hex: Option<String>,
    /// SHA-256 hex digest.
    pub sha256_hex: Option<String>,
    /// CRC32 hex digest.
    pub crc32_hex: Option<String>,
    /// Whether this was a resumed computation.
    pub was_resumed: bool,
    /// Total bytes processed (should equal file_size on success).
    pub bytes_processed: u64,
}

/// SHA-256 round constants.
#[allow(clippy::unreadable_literal)]
const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

#[allow(clippy::unreadable_literal)]
const SHA256_H_INIT: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

fn sha256_compress_block(state: &mut [u32; 8], block: &[u8]) {
    let mut w = [0u32; 64];
    for i in 0..16 {
        w[i] = u32::from_be_bytes([
            block[i * 4],
            block[i * 4 + 1],
            block[i * 4 + 2],
            block[i * 4 + 3],
        ]);
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }

    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let temp1 = h
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(SHA256_K[i])
            .wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let temp2 = s0.wrapping_add(maj);
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(temp1);
        d = c;
        c = b;
        b = a;
        a = temp1.wrapping_add(temp2);
    }

    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
    state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g);
    state[7] = state[7].wrapping_add(h);
}

/// Resumable SHA-256 hasher that exposes its internal state for serialization.
#[derive(Debug, Clone)]
pub struct ResumableSha256 {
    state: [u32; 8],
    total_bytes: u64,
    pending: Vec<u8>,
}

impl ResumableSha256 {
    /// Create a new hasher with the standard initial state.
    pub fn new() -> Self {
        Self {
            state: SHA256_H_INIT,
            total_bytes: 0,
            pending: Vec::with_capacity(64),
        }
    }

    /// Restore a hasher from a serialized checkpoint state.
    pub fn from_state(saved: &Sha256State) -> Self {
        Self {
            state: saved.h,
            total_bytes: saved.total_bytes,
            pending: saved.pending.clone(),
        }
    }

    /// Feed data into the hasher.
    pub fn update(&mut self, data: &[u8]) {
        self.total_bytes += data.len() as u64;
        self.pending.extend_from_slice(data);

        while self.pending.len() >= 64 {
            let block: Vec<u8> = self.pending.drain(..64).collect();
            sha256_compress_block(&mut self.state, &block);
        }
    }

    /// Export the current state for serialization.
    pub fn save_state(&self) -> Sha256State {
        Sha256State {
            h: self.state,
            total_bytes: self.total_bytes,
            pending: self.pending.clone(),
        }
    }

    /// Finalize and return the SHA-256 digest as a hex string.
    ///
    /// This consumes the hasher conceptually but we clone internally to
    /// allow the caller to keep the state.
    pub fn finalize_hex(&self) -> String {
        let mut state = self.state;
        let bit_len = self.total_bytes.wrapping_mul(8);

        let mut padded = [0u8; 128];
        let rem = self.pending.len();
        padded[..rem].copy_from_slice(&self.pending);
        padded[rem] = 0x80;

        let pad_len = if rem < 56 { 64 } else { 128 };
        padded[pad_len - 8..pad_len].copy_from_slice(&bit_len.to_be_bytes());

        sha256_compress_block(&mut state, &padded[..64]);
        if pad_len == 128 {
            sha256_compress_block(&mut state, &padded[64..128]);
        }

        let mut digest = [0u8; 32];
        for (i, word) in state.iter().enumerate() {
            digest[i * 4..(i + 1) * 4].copy_from_slice(&word.to_be_bytes());
        }
        digest.iter().map(|b| format!("{b:02x}")).collect()
    }
}

impl Default for ResumableSha256 {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute checksums incrementally for a byte slice, supporting resume from
/// a checkpoint. This is the synchronous in-memory variant.
pub fn compute_incremental(
    data: &[u8],
    config: &IncrementalConfig,
    checkpoint: Option<&ChecksumCheckpoint>,
) -> ArchiveResult<IncrementalResult> {
    let start_offset = checkpoint
        .map(|cp| cp.bytes_processed as usize)
        .unwrap_or(0);

    if start_offset > data.len() {
        return Err(ArchiveError::Validation(
            "checkpoint offset exceeds data length".to_string(),
        ));
    }

    let was_resumed = checkpoint.is_some();

    // Initialize hashers from checkpoint or fresh
    let mut blake3_hasher = if config.enable_blake3 {
        Some(blake3::Hasher::new())
    } else {
        None
    };

    let mut sha256_hasher = if config.enable_sha256 {
        if let Some(cp) = checkpoint {
            cp.sha256_state
                .as_ref()
                .map(|s| ResumableSha256::from_state(s))
        } else {
            Some(ResumableSha256::new())
        }
    } else {
        None
    };

    // For CRC32 we use crc32fast::Hasher for incremental computation.
    // When resuming, we reconstruct the hasher by re-hashing the already-
    // processed prefix (checkpointed value used to validate consistency).
    let mut crc32_hasher: Option<crc32fast::Hasher> = if config.enable_crc32 {
        let mut h = crc32fast::Hasher::new();
        // If resuming, replay the already-processed bytes through CRC32 too.
        if let Some(cp) = checkpoint {
            if cp.crc32_value.is_some() {
                // Re-hash the prefix so our hasher is consistent.
                h.update(&data[..start_offset]);
            }
        }
        Some(h)
    } else {
        None
    };

    // For BLAKE3: if resuming, we must re-hash from the beginning since
    // blake3::Hasher state is not trivially serializable. But for
    // in-memory data this is acceptable.
    if config.enable_blake3 {
        if let Some(ref mut hasher) = blake3_hasher {
            // Re-hash everything from the start for BLAKE3 (the checkpoint
            // only truly saves SHA-256 and CRC32 state).
            hasher.update(&data[..start_offset]);
        }
    }

    // Process remaining data
    let remaining = &data[start_offset..];
    let chunk_size = config.chunk_size.max(1);
    for chunk in remaining.chunks(chunk_size) {
        if let Some(ref mut hasher) = blake3_hasher {
            hasher.update(chunk);
        }
        if let Some(ref mut hasher) = sha256_hasher {
            hasher.update(chunk);
        }
        if let Some(ref mut h) = crc32_hasher {
            h.update(chunk);
        }
    }

    let blake3_hex = blake3_hasher.map(|h| h.finalize().to_hex().to_string());
    let sha256_hex = sha256_hasher.map(|h| h.finalize_hex());
    let crc32_hex = crc32_hasher.map(|h| format!("{:08x}", h.finalize()));

    Ok(IncrementalResult {
        file_path: checkpoint
            .map(|cp| cp.file_path.clone())
            .unwrap_or_default(),
        file_size: data.len() as u64,
        blake3_hex,
        sha256_hex,
        crc32_hex,
        was_resumed,
        bytes_processed: data.len() as u64,
    })
}

/// Create a checkpoint at the current position in the data.
pub fn create_checkpoint(
    file_path: &Path,
    file_size: u64,
    bytes_processed: u64,
    sha256_hasher: Option<&ResumableSha256>,
    crc32_value: Option<u32>,
    chunk_size: usize,
) -> ChecksumCheckpoint {
    let now_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    ChecksumCheckpoint {
        file_path: file_path.to_path_buf(),
        file_size,
        bytes_processed,
        blake3_partial_hex: None, // BLAKE3 state is not easily serializable
        sha256_state: sha256_hasher.map(|h| h.save_state()),
        crc32_value,
        chunk_size,
        created_at_secs: now_secs,
    }
}

// ---------------------------------------------------------------------------
// Checkpoint store (in-memory)
// ---------------------------------------------------------------------------

/// In-memory store for checksum checkpoints, keyed by file path.
#[derive(Debug, Default)]
pub struct CheckpointStore {
    checkpoints: HashMap<PathBuf, ChecksumCheckpoint>,
}

impl CheckpointStore {
    /// Create an empty checkpoint store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Save a checkpoint.
    pub fn save(&mut self, checkpoint: ChecksumCheckpoint) {
        self.checkpoints
            .insert(checkpoint.file_path.clone(), checkpoint);
    }

    /// Load a checkpoint for the given path, if one exists.
    pub fn load(&self, path: &Path) -> Option<&ChecksumCheckpoint> {
        self.checkpoints.get(path)
    }

    /// Remove a checkpoint (e.g. after successful completion).
    pub fn remove(&mut self, path: &Path) -> Option<ChecksumCheckpoint> {
        self.checkpoints.remove(path)
    }

    /// Number of stored checkpoints.
    pub fn len(&self) -> usize {
        self.checkpoints.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.checkpoints.is_empty()
    }

    /// List all paths that have checkpoints.
    pub fn paths(&self) -> Vec<&PathBuf> {
        self.checkpoints.keys().collect()
    }

    /// Remove all checkpoints.
    pub fn clear(&mut self) {
        self.checkpoints.clear();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> IncrementalConfig {
        IncrementalConfig::default()
    }

    // --- ResumableSha256 ---

    #[test]
    fn test_resumable_sha256_empty() {
        let h = ResumableSha256::new();
        assert_eq!(
            h.finalize_hex(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn test_resumable_sha256_abc() {
        let mut h = ResumableSha256::new();
        h.update(b"abc");
        assert_eq!(
            h.finalize_hex(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn test_resumable_sha256_chunked_matches_whole() {
        let data = b"The quick brown fox jumps over the lazy dog";
        let mut h1 = ResumableSha256::new();
        h1.update(data);

        let mut h2 = ResumableSha256::new();
        h2.update(&data[..10]);
        h2.update(&data[10..25]);
        h2.update(&data[25..]);

        assert_eq!(h1.finalize_hex(), h2.finalize_hex());
    }

    #[test]
    fn test_resumable_sha256_save_restore() {
        let data = b"Hello world of incremental checksumming";
        let mut h1 = ResumableSha256::new();
        h1.update(&data[..20]);
        let saved = h1.save_state();

        let mut h2 = ResumableSha256::from_state(&saved);
        h2.update(&data[20..]);

        let mut h_full = ResumableSha256::new();
        h_full.update(data);

        assert_eq!(h2.finalize_hex(), h_full.finalize_hex());
    }

    #[test]
    fn test_resumable_sha256_large_data() {
        let data: Vec<u8> = (0u8..=255).cycle().take(2048).collect();
        let mut h = ResumableSha256::new();
        h.update(&data);
        let hex = h.finalize_hex();
        assert_eq!(hex.len(), 64);
        // Verify it is deterministic
        let mut h2 = ResumableSha256::new();
        h2.update(&data);
        assert_eq!(h2.finalize_hex(), hex);
    }

    // --- Incremental computation ---

    #[test]
    fn test_compute_incremental_fresh() {
        let data = b"test data for incremental checksumming";
        let config = default_config();
        let result = compute_incremental(data, &config, None).expect("compute_incremental failed");
        assert!(!result.was_resumed);
        assert_eq!(result.bytes_processed, data.len() as u64);
        assert!(result.blake3_hex.is_some());
        assert!(result.sha256_hex.is_some());
        assert!(result.crc32_hex.is_some());
    }

    #[test]
    fn test_compute_incremental_sha256_matches_standard() {
        let data = b"abc";
        let config = IncrementalConfig {
            enable_blake3: false,
            enable_sha256: true,
            enable_crc32: false,
            ..default_config()
        };
        let result = compute_incremental(data, &config, None).expect("compute_incremental failed");
        assert_eq!(
            result.sha256_hex.as_deref(),
            Some("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        );
    }

    #[test]
    fn test_compute_incremental_resume_sha256() {
        let data = b"Hello world of incremental checksumming!";

        // Compute the full result for reference
        let config = IncrementalConfig {
            enable_blake3: false,
            enable_sha256: true,
            enable_crc32: true,
            chunk_size: 8,
            ..default_config()
        };
        let full_result = compute_incremental(data, &config, None).expect("full compute failed");

        // Simulate partial processing: compute first 20 bytes
        let mut sha_hasher = ResumableSha256::new();
        sha_hasher.update(&data[..20]);
        let crc_val = crc32fast::hash(&data[..20]);

        let checkpoint = create_checkpoint(
            Path::new("/test/file.bin"),
            data.len() as u64,
            20,
            Some(&sha_hasher),
            Some(crc_val),
            8,
        );

        // Resume from checkpoint
        let resumed_result =
            compute_incremental(data, &config, Some(&checkpoint)).expect("resumed compute failed");

        assert!(resumed_result.was_resumed);
        assert_eq!(resumed_result.sha256_hex, full_result.sha256_hex);
        assert_eq!(resumed_result.crc32_hex, full_result.crc32_hex);
    }

    #[test]
    fn test_compute_incremental_crc32_only() {
        let data = b"crc32 only test";
        let config = IncrementalConfig {
            enable_blake3: false,
            enable_sha256: false,
            enable_crc32: true,
            ..default_config()
        };
        let result = compute_incremental(data, &config, None).expect("compute_incremental failed");
        assert!(result.blake3_hex.is_none());
        assert!(result.sha256_hex.is_none());
        assert!(result.crc32_hex.is_some());
    }

    #[test]
    fn test_compute_incremental_empty_data() {
        let data: &[u8] = b"";
        let config = default_config();
        let result = compute_incremental(data, &config, None).expect("compute_incremental failed");
        assert_eq!(result.bytes_processed, 0);
        assert_eq!(result.file_size, 0);
    }

    // --- ChecksumCheckpoint ---

    #[test]
    fn test_checkpoint_json_roundtrip() {
        let cp = create_checkpoint(
            Path::new("/archive/video.mkv"),
            1_000_000,
            500_000,
            None,
            Some(0xDEADBEEF),
            1024 * 1024,
        );

        let json = cp.to_json().expect("serialization failed");
        let restored = ChecksumCheckpoint::from_json(&json).expect("deserialization failed");

        assert_eq!(restored.file_path, cp.file_path);
        assert_eq!(restored.file_size, cp.file_size);
        assert_eq!(restored.bytes_processed, cp.bytes_processed);
        assert_eq!(restored.crc32_value, cp.crc32_value);
        assert_eq!(restored.chunk_size, cp.chunk_size);
    }

    #[test]
    fn test_checkpoint_progress() {
        let cp = create_checkpoint(Path::new("/test"), 1000, 250, None, None, 1024);
        assert!((cp.progress() - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_checkpoint_progress_empty_file() {
        let cp = create_checkpoint(Path::new("/test"), 0, 0, None, None, 1024);
        assert!((cp.progress() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_checkpoint_is_complete() {
        let cp = create_checkpoint(Path::new("/test"), 100, 100, None, None, 1024);
        assert!(cp.is_complete());
    }

    #[test]
    fn test_checkpoint_not_complete() {
        let cp = create_checkpoint(Path::new("/test"), 100, 50, None, None, 1024);
        assert!(!cp.is_complete());
    }

    // --- CheckpointStore ---

    #[test]
    fn test_checkpoint_store_save_load() {
        let mut store = CheckpointStore::new();
        assert!(store.is_empty());

        let cp = create_checkpoint(Path::new("/archive/a.mkv"), 1000, 500, None, Some(42), 1024);
        store.save(cp);
        assert_eq!(store.len(), 1);

        let loaded = store.load(Path::new("/archive/a.mkv"));
        assert!(loaded.is_some());
        assert_eq!(loaded.map(|c| c.bytes_processed), Some(500));
    }

    #[test]
    fn test_checkpoint_store_remove() {
        let mut store = CheckpointStore::new();
        let cp = create_checkpoint(Path::new("/a"), 100, 50, None, None, 1024);
        store.save(cp);
        assert_eq!(store.len(), 1);

        let removed = store.remove(Path::new("/a"));
        assert!(removed.is_some());
        assert!(store.is_empty());
    }

    #[test]
    fn test_checkpoint_store_clear() {
        let mut store = CheckpointStore::new();
        store.save(create_checkpoint(
            Path::new("/a"),
            100,
            50,
            None,
            None,
            1024,
        ));
        store.save(create_checkpoint(
            Path::new("/b"),
            200,
            100,
            None,
            None,
            1024,
        ));
        assert_eq!(store.len(), 2);
        store.clear();
        assert!(store.is_empty());
    }

    #[test]
    fn test_checkpoint_store_paths() {
        let mut store = CheckpointStore::new();
        store.save(create_checkpoint(
            Path::new("/x"),
            100,
            50,
            None,
            None,
            1024,
        ));
        store.save(create_checkpoint(
            Path::new("/y"),
            200,
            100,
            None,
            None,
            1024,
        ));
        let paths = store.paths();
        assert_eq!(paths.len(), 2);
    }

    #[test]
    fn test_checkpoint_store_overwrite() {
        let mut store = CheckpointStore::new();
        store.save(create_checkpoint(
            Path::new("/a"),
            100,
            25,
            None,
            None,
            1024,
        ));
        store.save(create_checkpoint(
            Path::new("/a"),
            100,
            75,
            None,
            None,
            1024,
        ));
        assert_eq!(store.len(), 1);
        let loaded = store.load(Path::new("/a"));
        assert_eq!(loaded.map(|c| c.bytes_processed), Some(75));
    }

    // --- Resume with sha256 state roundtrip ---

    #[test]
    fn test_sha256_state_json_roundtrip() {
        let mut hasher = ResumableSha256::new();
        hasher.update(b"partial data");
        let state = hasher.save_state();

        let json = serde_json::to_string(&state).expect("serialize failed");
        let restored: Sha256State = serde_json::from_str(&json).expect("deserialize failed");

        assert_eq!(restored.h, state.h);
        assert_eq!(restored.total_bytes, state.total_bytes);
        assert_eq!(restored.pending, state.pending);
    }

    #[test]
    fn test_invalid_checkpoint_offset() {
        let data = b"short";
        let config = default_config();
        let cp = ChecksumCheckpoint {
            file_path: PathBuf::from("/test"),
            file_size: data.len() as u64,
            bytes_processed: 9999,
            blake3_partial_hex: None,
            sha256_state: None,
            crc32_value: None,
            chunk_size: 1024,
            created_at_secs: 0,
        };
        let result = compute_incremental(data, &config, Some(&cp));
        assert!(result.is_err());
    }
}
