//! Pre-roll management and media verification for playlist playout.
//!
//! This module provides two complementary subsystems:
//!
//! 1. **`PrerollManager`** — timing calculations for when to begin the
//!    pre-roll ramp before a scheduled air time.
//! 2. **`MediaVerifier`** — lightweight, synchronous per-file validation that
//!    checks filesystem existence, readability, file size, and magic-byte
//!    format recognition.  Results are collected via
//!    `PrerollManager::verify_upcoming_items` for any playlist items scheduled
//!    within a configurable look-ahead window.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};
use tracing::info;

// ─────────────────────────────────────────────────────────────────────────────
// PrerollConfig / PrerollManager (original timing code, unchanged)
// ─────────────────────────────────────────────────────────────────────────────

/// Pre-roll configuration.
#[derive(Debug, Clone)]
pub struct PrerollConfig {
    /// Number of frames to pre-roll
    pub frames: u64,
    /// Frame rate for timing calculation
    pub frame_rate: f64,
}

impl Default for PrerollConfig {
    fn default() -> Self {
        Self {
            frames: 150, // 5 seconds at 30fps
            frame_rate: 30.0,
        }
    }
}

/// A playlist item with a scheduled air time, used for look-ahead verification.
#[derive(Debug, Clone)]
pub struct ScheduledItem {
    /// Filesystem path to the media file.
    pub path: PathBuf,
    /// Scheduled air time (wall clock).
    pub scheduled_air_time: SystemTime,
}

impl ScheduledItem {
    /// Create a new scheduled item.
    pub fn new(path: impl Into<PathBuf>, scheduled_air_time: SystemTime) -> Self {
        Self {
            path: path.into(),
            scheduled_air_time,
        }
    }
}

/// Pre-roll manager for playlist items.
pub struct PrerollManager {
    config: PrerollConfig,
}

impl PrerollManager {
    /// Create a new pre-roll manager.
    pub fn new() -> Self {
        Self {
            config: PrerollConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: PrerollConfig) -> Self {
        Self { config }
    }

    /// Calculate pre-roll duration.
    pub fn preroll_duration(&self) -> Duration {
        let seconds = self.config.frames as f64 / self.config.frame_rate;
        Duration::from_secs_f64(seconds)
    }

    /// Get pre-roll frames.
    pub fn preroll_frames(&self) -> u64 {
        self.config.frames
    }

    /// Calculate when to start pre-roll for a scheduled time.
    pub fn calculate_preroll_start(&self, scheduled_time: SystemTime) -> SystemTime {
        scheduled_time - self.preroll_duration()
    }

    /// Check if it's time to start pre-roll.
    pub fn should_start_preroll(&self, scheduled_time: SystemTime) -> bool {
        let now = SystemTime::now();
        let preroll_start = self.calculate_preroll_start(scheduled_time);
        now >= preroll_start
    }

    /// Set frame rate.
    pub fn set_frame_rate(&mut self, frame_rate: f64) {
        info!("Setting pre-roll frame rate to: {}", frame_rate);
        self.config.frame_rate = frame_rate;
    }

    /// Set pre-roll frames.
    pub fn set_preroll_frames(&mut self, frames: u64) {
        info!("Setting pre-roll frames to: {}", frames);
        self.config.frames = frames;
    }

    // ── Media verification integration ────────────────────────────────────────

    /// Verify a single playlist item at `path`.
    ///
    /// Delegates to `MediaVerifier::verify_media`.
    pub fn verify_playlist_item(&self, path: &Path) -> VerificationResult {
        MediaVerifier::verify_media(path)
    }

    /// Verify all `items` that are scheduled to air within the next
    /// `lookahead` duration from now.
    ///
    /// Items with an air time beyond `now + lookahead` are skipped.
    pub fn verify_upcoming_items(
        &self,
        items: &[ScheduledItem],
        lookahead: Duration,
    ) -> Vec<VerificationResult> {
        let now = SystemTime::now();
        let deadline = now + lookahead;

        items
            .iter()
            .filter(|item| item.scheduled_air_time <= deadline)
            .map(|item| MediaVerifier::verify_media(&item.path))
            .collect()
    }
}

impl Default for PrerollManager {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Magic-byte signatures
// ─────────────────────────────────────────────────────────────────────────────

/// A known media format with magic-byte signature and friendly name.
struct MagicSignature {
    /// Offset into the file where the magic bytes begin.
    offset: usize,
    /// Expected byte pattern.
    bytes: &'static [u8],
    /// Human-readable format name.
    name: &'static str,
}

/// Minimum number of header bytes to read for magic-byte detection.
const MAGIC_READ_BYTES: usize = 16;

/// Known media format signatures.
///
/// Listed in the order they are tried; earlier entries take precedence when
/// signatures overlap (e.g. RIFF appears before WAV-specific patterns).
static KNOWN_SIGNATURES: &[MagicSignature] = &[
    // MP4 / ISOBMFF — "ftyp" at offset 4
    MagicSignature {
        offset: 4,
        bytes: b"ftyp",
        name: "MP4/ISOBMFF",
    },
    // MKV / WebM — EBML magic (0x1a 0x45 0xdf 0xa3)
    MagicSignature {
        offset: 0,
        bytes: b"\x1a\x45\xdf\xa3",
        name: "MKV/WebM",
    },
    // WAV / AVI — RIFF container header
    MagicSignature {
        offset: 0,
        bytes: b"RIFF",
        name: "RIFF (WAV/AVI)",
    },
    // FLAC
    MagicSignature {
        offset: 0,
        bytes: b"fLaC",
        name: "FLAC",
    },
    // Ogg container (used by Opus/Vorbis)
    MagicSignature {
        offset: 0,
        bytes: b"OggS",
        name: "Ogg",
    },
    // MPEG transport stream (common sync byte 0x47)
    MagicSignature {
        offset: 0,
        bytes: &[0x47],
        name: "MPEG-TS",
    },
    // MXF — SMPTE 377M key (14 bytes)
    MagicSignature {
        offset: 0,
        bytes: &[0x06, 0x0e, 0x2b, 0x34],
        name: "MXF",
    },
    // AIFF — "FORM" + "AIFF"/"AIFC" — just detect "FORM" at offset 0
    MagicSignature {
        offset: 0,
        bytes: b"FORM",
        name: "AIFF",
    },
    // QuickTime MOV — wide-atom "moov" or free-atom or mdat at offset 4
    MagicSignature {
        offset: 4,
        bytes: b"moov",
        name: "QuickTime MOV",
    },
    MagicSignature {
        offset: 4,
        bytes: b"mdat",
        name: "QuickTime MOV (mdat)",
    },
    MagicSignature {
        offset: 4,
        bytes: b"free",
        name: "QuickTime MOV (free)",
    },
    MagicSignature {
        offset: 4,
        bytes: b"wide",
        name: "QuickTime MOV (wide)",
    },
];

// ─────────────────────────────────────────────────────────────────────────────
// VerificationResult
// ─────────────────────────────────────────────────────────────────────────────

/// Result of verifying a single media file.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Canonical path of the media file that was checked.
    pub path: PathBuf,
    /// `true` if the file exists in the filesystem.
    pub exists: bool,
    /// `true` if the file could be opened for reading.
    pub readable: bool,
    /// `true` if a recognised media format magic byte was found at the
    /// expected offset.
    pub format_valid: bool,
    /// Human-readable name of the detected format, if recognised.
    pub detected_format: Option<String>,
    /// File size in bytes, if the file exists and is readable.
    pub file_size_bytes: Option<u64>,
    /// Estimated duration in seconds — currently a rough heuristic derived
    /// from file size.  `None` if size could not be determined.
    pub estimated_duration_secs: Option<f64>,
    /// Non-empty if any check failed.
    pub errors: Vec<String>,
}

impl VerificationResult {
    /// `true` if all checks passed (exists, readable, format valid).
    pub fn is_valid(&self) -> bool {
        self.exists && self.readable && self.format_valid
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MediaVerifier
// ─────────────────────────────────────────────────────────────────────────────

/// Synchronous, allocation-light media file verifier.
///
/// All checks are performed on the calling thread without spawning tasks.
/// Callers requiring async integration should wrap with
/// `tokio::task::spawn_blocking`.
pub struct MediaVerifier;

impl MediaVerifier {
    /// Verify a media file at `path`.
    ///
    /// Checks performed (in order):
    ///
    /// 1. **Existence** — `path.exists()`.
    /// 2. **Readability** — `std::fs::File::open(path)`.
    /// 3. **Non-zero size** — `file.metadata().len() > 0`.
    /// 4. **Format** — read first `MAGIC_READ_BYTES` bytes and match against
    ///    known signatures in `KNOWN_SIGNATURES`.
    ///
    /// Any failed check is recorded in `VerificationResult::errors`.
    pub fn verify_media(path: &Path) -> VerificationResult {
        let mut result = VerificationResult {
            path: path.to_path_buf(),
            exists: false,
            readable: false,
            format_valid: false,
            detected_format: None,
            file_size_bytes: None,
            estimated_duration_secs: None,
            errors: Vec::new(),
        };

        // ── 1. Existence ──────────────────────────────────────────────────────
        if !path.exists() {
            result
                .errors
                .push(format!("file does not exist: {}", path.display()));
            return result;
        }
        result.exists = true;

        // ── 2. Readability ────────────────────────────────────────────────────
        let file = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(e) => {
                result
                    .errors
                    .push(format!("cannot open file for reading: {e}"));
                return result;
            }
        };
        result.readable = true;

        // ── 3. Non-zero size ─────────────────────────────────────────────────
        let size_bytes = match file.metadata() {
            Ok(meta) => meta.len(),
            Err(e) => {
                result
                    .errors
                    .push(format!("cannot read file metadata: {e}"));
                return result;
            }
        };

        if size_bytes == 0 {
            result.errors.push("file is empty (zero bytes)".to_string());
            return result;
        }
        result.file_size_bytes = Some(size_bytes);

        // ── 4. Magic-byte format detection ────────────────────────────────────
        use std::io::Read;
        let mut header = [0u8; MAGIC_READ_BYTES];
        let bytes_read = {
            let mut handle = &file;
            handle.read(&mut header).unwrap_or(0)
        };

        if bytes_read == 0 {
            result
                .errors
                .push("could not read file header bytes".to_string());
            return result;
        }

        let header_slice = &header[..bytes_read];
        let detected = Self::detect_format(header_slice);

        if let Some(fmt) = detected {
            result.format_valid = true;
            result.detected_format = Some(fmt.to_string());
        } else {
            result.errors.push(format!(
                "unrecognised media format (no matching magic bytes in first {} bytes)",
                bytes_read
            ));
        }

        // ── Estimated duration heuristic ──────────────────────────────────────
        // Assume average ~10 Mbit/s = 1.25 MB/s for video; audio typically
        // smaller but this is just a rough guide for scheduling.
        result.estimated_duration_secs = Some(size_bytes as f64 / 1_250_000.0);

        result
    }

    /// Match `header` bytes against known media format signatures.
    /// Returns the format name if recognised, otherwise `None`.
    fn detect_format(header: &[u8]) -> Option<&'static str> {
        for sig in KNOWN_SIGNATURES {
            let end = sig.offset + sig.bytes.len();
            if end <= header.len() && &header[sig.offset..end] == sig.bytes {
                return Some(sig.name);
            }
        }
        None
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    // ── PrerollManager timing tests (original) ────────────────────────────────

    #[test]
    fn test_preroll_manager_creation() {
        let manager = PrerollManager::new();
        assert_eq!(manager.preroll_frames(), 150);
    }

    #[test]
    fn test_preroll_duration() {
        let manager = PrerollManager::new();
        let duration = manager.preroll_duration();
        assert_eq!(duration.as_secs(), 5);
    }

    #[test]
    fn test_set_frame_rate() {
        let mut manager = PrerollManager::new();
        manager.set_frame_rate(60.0);

        let duration = manager.preroll_duration();
        assert_eq!(duration.as_millis(), 2500); // 150 frames at 60fps = 2.5 seconds
    }

    #[test]
    fn test_calculate_preroll_start() {
        let manager = PrerollManager::new();
        let scheduled = SystemTime::now() + Duration::from_secs(10);
        let preroll_start = manager.calculate_preroll_start(scheduled);

        let diff = scheduled
            .duration_since(preroll_start)
            .expect("duration_since should succeed");
        assert_eq!(diff.as_secs(), 5);
    }

    // ── MediaVerifier — helper to write a temp file ───────────────────────────

    /// Write bytes to a temp file in `std::env::temp_dir()` and return the path.
    fn write_temp_file(name: &str, content: &[u8]) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("oximedia_preroll_test_{}", name));
        let mut f = std::fs::File::create(&path).expect("create temp file should succeed");
        f.write_all(content).expect("write should succeed");
        path
    }

    fn cleanup(path: &Path) {
        let _ = std::fs::remove_file(path);
    }

    // ── Existence check ───────────────────────────────────────────────────────

    #[test]
    fn test_verify_nonexistent_file() {
        let path = std::env::temp_dir().join("oximedia_definitely_missing_abc123.mp4");
        let result = MediaVerifier::verify_media(&path);
        assert!(!result.exists);
        assert!(!result.readable);
        assert!(!result.format_valid);
        assert!(!result.errors.is_empty());
    }

    // ── Empty file check ──────────────────────────────────────────────────────

    #[test]
    fn test_verify_empty_file() {
        let path = write_temp_file("empty_test.mp4", &[]);
        let result = MediaVerifier::verify_media(&path);
        cleanup(&path);
        assert!(result.exists);
        assert!(result.readable);
        assert!(!result.format_valid, "empty file should fail format check");
        assert!(result.errors.iter().any(|e| e.contains("empty")));
    }

    // ── MP4/ISOBMFF ──────────────────────────────────────────────────────────

    #[test]
    fn test_verify_mp4_magic_bytes() {
        // Build 8-byte header: 4 bytes size + "ftyp"
        let mut header = vec![0u8, 0, 0, 28]; // size field
        header.extend_from_slice(b"ftyp");
        header.extend_from_slice(b"mp42");
        let path = write_temp_file("test_mp4.mp4", &header);
        let result = MediaVerifier::verify_media(&path);
        cleanup(&path);
        assert!(result.exists);
        assert!(result.readable);
        assert!(result.format_valid, "MP4 magic bytes should be recognised");
        assert_eq!(result.detected_format.as_deref(), Some("MP4/ISOBMFF"));
    }

    // ── MKV ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_verify_mkv_magic_bytes() {
        // EBML magic: 0x1a 0x45 0xdf 0xa3
        let mut header = vec![0x1a, 0x45, 0xdf, 0xa3];
        header.extend_from_slice(&[0u8; 12]);
        let path = write_temp_file("test_mkv.mkv", &header);
        let result = MediaVerifier::verify_media(&path);
        cleanup(&path);
        assert!(result.format_valid, "MKV magic bytes should be recognised");
        assert_eq!(result.detected_format.as_deref(), Some("MKV/WebM"));
    }

    // ── WAV / RIFF ────────────────────────────────────────────────────────────

    #[test]
    fn test_verify_wav_magic_bytes() {
        let mut header = b"RIFF".to_vec();
        header.extend_from_slice(&[0u8; 4]); // size
        header.extend_from_slice(b"WAVE");
        let path = write_temp_file("test_wav.wav", &header);
        let result = MediaVerifier::verify_media(&path);
        cleanup(&path);
        assert!(
            result.format_valid,
            "WAV/RIFF magic bytes should be recognised"
        );
        assert!(result
            .detected_format
            .as_deref()
            .map(|s| s.contains("RIFF"))
            .unwrap_or(false));
    }

    // ── FLAC ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_verify_flac_magic_bytes() {
        let mut header = b"fLaC".to_vec();
        header.extend_from_slice(&[0u8; 12]);
        let path = write_temp_file("test_flac.flac", &header);
        let result = MediaVerifier::verify_media(&path);
        cleanup(&path);
        assert!(result.format_valid, "FLAC magic bytes should be recognised");
        assert_eq!(result.detected_format.as_deref(), Some("FLAC"));
    }

    // ── MXF ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_verify_mxf_magic_bytes() {
        // SMPTE 377M key prefix: 06 0e 2b 34
        let mut header = vec![0x06, 0x0e, 0x2b, 0x34];
        header.extend_from_slice(&[0u8; 12]);
        let path = write_temp_file("test_mxf.mxf", &header);
        let result = MediaVerifier::verify_media(&path);
        cleanup(&path);
        assert!(result.format_valid, "MXF magic bytes should be recognised");
        assert_eq!(result.detected_format.as_deref(), Some("MXF"));
    }

    // ── Unknown format ────────────────────────────────────────────────────────

    #[test]
    fn test_verify_unknown_format() {
        // Write random bytes that don't match any signature
        let content = vec![
            0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00,
        ];
        let path = write_temp_file("test_unknown.bin", &content);
        let result = MediaVerifier::verify_media(&path);
        cleanup(&path);
        assert!(result.exists);
        assert!(result.readable);
        assert!(
            !result.format_valid,
            "unknown format should fail format check"
        );
        assert!(result.detected_format.is_none());
    }

    // ── is_valid helper ───────────────────────────────────────────────────────

    #[test]
    fn test_verification_result_is_valid() {
        let mut header = vec![0u8, 0, 0, 28];
        header.extend_from_slice(b"ftyp");
        header.extend_from_slice(b"mp42");
        let path = write_temp_file("test_valid_mp4_2.mp4", &header);
        let result = MediaVerifier::verify_media(&path);
        cleanup(&path);
        assert!(result.is_valid());
    }

    // ── PrerollManager::verify_playlist_item ─────────────────────────────────

    #[test]
    fn test_verify_playlist_item_delegates() {
        let manager = PrerollManager::new();
        let nonexistent = std::env::temp_dir().join("no_such_file_xyz_12345.mp4");
        let result = manager.verify_playlist_item(&nonexistent);
        assert!(!result.exists);
    }

    // ── PrerollManager::verify_upcoming_items ────────────────────────────────

    #[test]
    fn test_verify_upcoming_items_filters_by_lookahead() {
        let manager = PrerollManager::new();

        // Create one valid file (MP4 magic bytes)
        let mut header = vec![0u8, 0, 0, 28];
        header.extend_from_slice(b"ftyp");
        header.extend_from_slice(b"mp42");
        let path = write_temp_file("upcoming_mp4.mp4", &header);

        let now = SystemTime::now();
        let items = vec![
            // Within lookahead (airs in 30s)
            ScheduledItem::new(&path, now + Duration::from_secs(30)),
            // Beyond lookahead (airs in 3600s = 1 hour)
            ScheduledItem::new(&path, now + Duration::from_secs(3600)),
        ];

        let results = manager.verify_upcoming_items(&items, Duration::from_secs(60));
        cleanup(&path);

        assert_eq!(
            results.len(),
            1,
            "only the item within 60s lookahead should be verified"
        );
        assert!(results[0].is_valid());
    }

    #[test]
    fn test_verify_upcoming_items_empty_list() {
        let manager = PrerollManager::new();
        let results = manager.verify_upcoming_items(&[], Duration::from_secs(300));
        assert!(results.is_empty());
    }

    // ── Estimated duration heuristic ─────────────────────────────────────────

    #[test]
    fn test_estimated_duration_present_for_valid_file() {
        let mut header = b"fLaC".to_vec();
        header.extend_from_slice(&[0u8; 12]);
        let path = write_temp_file("test_flac_dur.flac", &header);
        let result = MediaVerifier::verify_media(&path);
        cleanup(&path);
        assert!(result.estimated_duration_secs.is_some());
        // 16 bytes at 1.25 MB/s ≈ 0 seconds (tiny file, non-zero)
        assert!(result.estimated_duration_secs.unwrap() >= 0.0);
    }
}
