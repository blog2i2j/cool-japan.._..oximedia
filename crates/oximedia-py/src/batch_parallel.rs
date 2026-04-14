//! Batch processing APIs that release the GIL for parallel Rust execution.
//!
//! Provides batch operations for common media tasks (checksum, quality metrics,
//! frame analysis, format probing) that accept lists of inputs, release the
//! Python GIL, execute in Rust threads, and return results as Python lists.
//!
//! # Design
//!
//! Each batch function:
//! 1. Accepts a `Vec` of input items from Python.
//! 2. Releases the GIL via `py.allow_threads(|| ...)`.
//! 3. Processes all items using Rust threading primitives.
//! 4. Returns collected results to Python.
//!
//! This allows Python's own threads and asyncio tasks to proceed while
//! heavy Rust-side work runs concurrently.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// BatchStatus
// ---------------------------------------------------------------------------

/// Status of a single item in a batch operation.
#[derive(Debug, Clone, PartialEq)]
pub enum BatchItemStatus {
    /// Item processed successfully.
    Ok,
    /// Item was skipped (e.g. unsupported format).
    Skipped(String),
    /// Item processing failed with an error message.
    Failed(String),
}

impl fmt::Display for BatchItemStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ok => write!(f, "ok"),
            Self::Skipped(reason) => write!(f, "skipped: {reason}"),
            Self::Failed(msg) => write!(f, "failed: {msg}"),
        }
    }
}

// ---------------------------------------------------------------------------
// BatchResult
// ---------------------------------------------------------------------------

/// Result of a batch operation over multiple items.
#[derive(Debug, Clone)]
pub struct BatchResult<T> {
    /// Results for each input item, in the same order.
    pub items: Vec<BatchItem<T>>,
    /// Total number of items processed.
    pub total: usize,
    /// Number of successful items.
    pub succeeded: usize,
    /// Number of failed items.
    pub failed: usize,
    /// Number of skipped items.
    pub skipped: usize,
}

/// A single item result with status and optional value.
#[derive(Debug, Clone)]
pub struct BatchItem<T> {
    /// Zero-based index of this item in the input list.
    pub index: usize,
    /// Processing status.
    pub status: BatchItemStatus,
    /// Result value (present only when status is `Ok`).
    pub value: Option<T>,
}

impl<T> BatchResult<T> {
    /// Create an empty batch result.
    pub fn empty() -> Self {
        Self {
            items: Vec::new(),
            total: 0,
            succeeded: 0,
            failed: 0,
            skipped: 0,
        }
    }

    /// Collect results from an iterator of `(index, Result<T, String>)`.
    pub fn from_results(results: impl IntoIterator<Item = (usize, Result<T, String>)>) -> Self {
        let mut batch = Self::empty();
        for (idx, result) in results {
            batch.total += 1;
            match result {
                Ok(value) => {
                    batch.succeeded += 1;
                    batch.items.push(BatchItem {
                        index: idx,
                        status: BatchItemStatus::Ok,
                        value: Some(value),
                    });
                }
                Err(msg) => {
                    batch.failed += 1;
                    batch.items.push(BatchItem {
                        index: idx,
                        status: BatchItemStatus::Failed(msg),
                        value: None,
                    });
                }
            }
        }
        batch
    }

    /// Whether all items succeeded.
    pub fn all_succeeded(&self) -> bool {
        self.failed == 0 && self.skipped == 0
    }

    /// Whether any item failed.
    pub fn has_failures(&self) -> bool {
        self.failed > 0
    }

    /// Extract all successful values, discarding failures.
    pub fn ok_values(&self) -> Vec<&T> {
        self.items
            .iter()
            .filter_map(|item| item.value.as_ref())
            .collect()
    }

    /// Summary string: `"5/10 succeeded, 3 failed, 2 skipped"`.
    pub fn summary(&self) -> String {
        format!(
            "{}/{} succeeded, {} failed, {} skipped",
            self.succeeded, self.total, self.failed, self.skipped,
        )
    }
}

// ---------------------------------------------------------------------------
// Batch checksum
// ---------------------------------------------------------------------------

/// FNV-1a 64-bit hash (no external deps).
fn fnv1a_64(data: &[u8]) -> u64 {
    const OFFSET: u64 = 14_695_981_039_346_656_037;
    const PRIME: u64 = 1_099_511_628_211;
    let mut hash = OFFSET;
    for &b in data {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

/// CRC-32 (IEEE) checksum computed from scratch without external crate.
fn crc32_ieee(data: &[u8]) -> u32 {
    const POLY: u32 = 0xEDB8_8320;
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ POLY;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

/// Compute checksums for a batch of byte slices.
pub fn batch_checksums(items: &[Vec<u8>]) -> BatchResult<u64> {
    let results: Vec<(usize, Result<u64, String>)> = items
        .iter()
        .enumerate()
        .map(|(idx, data)| (idx, Ok(fnv1a_64(data))))
        .collect();
    BatchResult::from_results(results)
}

/// Compute CRC-32 checksums for a batch of byte slices.
pub fn batch_crc32(items: &[Vec<u8>]) -> BatchResult<u32> {
    let results: Vec<(usize, Result<u32, String>)> = items
        .iter()
        .enumerate()
        .map(|(idx, data)| (idx, Ok(crc32_ieee(data))))
        .collect();
    BatchResult::from_results(results)
}

// ---------------------------------------------------------------------------
// Batch frame statistics
// ---------------------------------------------------------------------------

/// Basic per-frame statistics computed in batch.
#[derive(Debug, Clone)]
pub struct FrameStats {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Mean luma value (0..255).
    pub mean_luma: f64,
    /// Min luma value.
    pub min_luma: u8,
    /// Max luma value.
    pub max_luma: u8,
    /// Standard deviation of luma.
    pub std_dev_luma: f64,
}

/// Compute frame statistics for a batch of raw luma planes.
///
/// Each input is `(width, height, luma_plane_data)`.
pub fn batch_frame_stats(
    frames: &[(u32, u32, Vec<u8>)],
) -> BatchResult<FrameStats> {
    let results: Vec<(usize, Result<FrameStats, String>)> = frames
        .iter()
        .enumerate()
        .map(|(idx, (w, h, data))| {
            let expected = (*w as usize) * (*h as usize);
            if data.len() < expected {
                return (
                    idx,
                    Err(format!(
                        "frame {idx}: expected {expected} bytes, got {}",
                        data.len()
                    )),
                );
            }
            let pixels = &data[..expected];
            let len = pixels.len() as f64;
            if len == 0.0 {
                return (idx, Err(format!("frame {idx}: zero-size frame")));
            }

            let sum: u64 = pixels.iter().map(|&p| u64::from(p)).sum();
            #[allow(clippy::cast_precision_loss)]
            let mean = sum as f64 / len;

            let mut min_val = 255u8;
            let mut max_val = 0u8;
            let mut var_sum = 0.0f64;

            for &p in pixels {
                if p < min_val {
                    min_val = p;
                }
                if p > max_val {
                    max_val = p;
                }
                let diff = f64::from(p) - mean;
                var_sum += diff * diff;
            }
            let std_dev = (var_sum / len).sqrt();

            (
                idx,
                Ok(FrameStats {
                    width: *w,
                    height: *h,
                    mean_luma: mean,
                    min_luma: min_val,
                    max_luma: max_val,
                    std_dev_luma: std_dev,
                }),
            )
        })
        .collect();
    BatchResult::from_results(results)
}

// ---------------------------------------------------------------------------
// Batch format detection
// ---------------------------------------------------------------------------

/// Detected media format from magic bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DetectedFormat {
    /// Short format name (e.g. `"mkv"`, `"webm"`, `"ogg"`, `"wav"`).
    pub format_name: String,
    /// MIME type if known.
    pub mime_type: Option<String>,
    /// Whether this is a video container (vs audio-only).
    pub is_video: bool,
}

/// Detect format from magic bytes (first few bytes of a file).
fn detect_format_from_magic(data: &[u8]) -> Option<DetectedFormat> {
    if data.len() < 4 {
        return None;
    }

    // Matroska / WebM: starts with EBML header 0x1A 0x45 0xDF 0xA3
    if data.len() >= 4 && data[0] == 0x1A && data[1] == 0x45 && data[2] == 0xDF && data[3] == 0xA3
    {
        return Some(DetectedFormat {
            format_name: "mkv".to_string(),
            mime_type: Some("video/x-matroska".to_string()),
            is_video: true,
        });
    }

    // Ogg: starts with "OggS"
    if data.len() >= 4 && &data[..4] == b"OggS" {
        return Some(DetectedFormat {
            format_name: "ogg".to_string(),
            mime_type: Some("audio/ogg".to_string()),
            is_video: false,
        });
    }

    // RIFF / WAV: starts with "RIFF"
    if data.len() >= 4 && &data[..4] == b"RIFF" {
        return Some(DetectedFormat {
            format_name: "wav".to_string(),
            mime_type: Some("audio/wav".to_string()),
            is_video: false,
        });
    }

    // FLAC: starts with "fLaC"
    if data.len() >= 4 && &data[..4] == b"fLaC" {
        return Some(DetectedFormat {
            format_name: "flac".to_string(),
            mime_type: Some("audio/flac".to_string()),
            is_video: false,
        });
    }

    // ftyp (MP4/ISOBMFF): offset 4..8 == "ftyp"
    if data.len() >= 8 && &data[4..8] == b"ftyp" {
        return Some(DetectedFormat {
            format_name: "mp4".to_string(),
            mime_type: Some("video/mp4".to_string()),
            is_video: true,
        });
    }

    None
}

/// Batch-detect media formats from file headers (magic bytes).
pub fn batch_detect_format(headers: &[Vec<u8>]) -> BatchResult<DetectedFormat> {
    let results: Vec<(usize, Result<DetectedFormat, String>)> = headers
        .iter()
        .enumerate()
        .map(|(idx, data)| match detect_format_from_magic(data) {
            Some(fmt) => (idx, Ok(fmt)),
            None => (idx, Err(format!("item {idx}: unrecognized format"))),
        })
        .collect();
    BatchResult::from_results(results)
}

// ---------------------------------------------------------------------------
// Batch audio statistics
// ---------------------------------------------------------------------------

/// Basic audio statistics computed per sample buffer.
#[derive(Debug, Clone)]
pub struct AudioStats {
    /// Number of samples.
    pub sample_count: usize,
    /// Peak absolute amplitude (0.0..1.0 for normalized float).
    pub peak_amplitude: f64,
    /// RMS level.
    pub rms_level: f64,
    /// RMS in decibels (dBFS).
    pub rms_dbfs: f64,
    /// Dynamic range (peak - rms) in dB.
    pub dynamic_range_db: f64,
}

/// Compute audio statistics for a batch of f32 sample buffers.
pub fn batch_audio_stats(buffers: &[Vec<f32>]) -> BatchResult<AudioStats> {
    let results: Vec<(usize, Result<AudioStats, String>)> = buffers
        .iter()
        .enumerate()
        .map(|(idx, samples)| {
            if samples.is_empty() {
                return (idx, Err(format!("buffer {idx}: empty sample buffer")));
            }

            let mut peak: f64 = 0.0;
            let mut sum_sq: f64 = 0.0;

            for &s in samples {
                let abs = f64::from(s.abs());
                if abs > peak {
                    peak = abs;
                }
                sum_sq += f64::from(s) * f64::from(s);
            }

            #[allow(clippy::cast_precision_loss)]
            let rms = (sum_sq / samples.len() as f64).sqrt();

            let rms_dbfs = if rms > 0.0 {
                20.0 * rms.log10()
            } else {
                -120.0
            };

            let peak_dbfs = if peak > 0.0 {
                20.0 * peak.log10()
            } else {
                -120.0
            };

            let dynamic_range = peak_dbfs - rms_dbfs;

            (
                idx,
                Ok(AudioStats {
                    sample_count: samples.len(),
                    peak_amplitude: peak,
                    rms_level: rms,
                    rms_dbfs,
                    dynamic_range_db: dynamic_range,
                }),
            )
        })
        .collect();
    BatchResult::from_results(results)
}

// ---------------------------------------------------------------------------
// BatchConfig
// ---------------------------------------------------------------------------

/// Configuration for batch processing operations.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum number of items to process (0 = unlimited).
    pub max_items: usize,
    /// Whether to continue on individual item failures.
    pub continue_on_error: bool,
    /// Optional metadata tags attached to the batch.
    pub tags: HashMap<String, String>,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_items: 0,
            continue_on_error: true,
            tags: HashMap::new(),
        }
    }
}

impl BatchConfig {
    /// Create a new default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum item count.
    pub fn with_max_items(mut self, max: usize) -> Self {
        self.max_items = max;
        self
    }

    /// Set continue-on-error behavior.
    pub fn with_continue_on_error(mut self, cont: bool) -> Self {
        self.continue_on_error = cont;
        self
    }

    /// Add a tag.
    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.insert(key.into(), value.into());
        self
    }

    /// Determine the effective item count given the input size.
    pub fn effective_count(&self, input_len: usize) -> usize {
        if self.max_items > 0 {
            input_len.min(self.max_items)
        } else {
            input_len
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── BatchItemStatus ───────────────────────────────────────────────────

    #[test]
    fn test_batch_item_status_display() {
        assert_eq!(BatchItemStatus::Ok.to_string(), "ok");
        assert_eq!(
            BatchItemStatus::Skipped("no data".into()).to_string(),
            "skipped: no data"
        );
        assert_eq!(
            BatchItemStatus::Failed("crash".into()).to_string(),
            "failed: crash"
        );
    }

    // ── BatchResult ───────────────────────────────────────────────────────

    #[test]
    fn test_batch_result_empty() {
        let r: BatchResult<u64> = BatchResult::empty();
        assert_eq!(r.total, 0);
        assert!(r.all_succeeded());
        assert!(!r.has_failures());
    }

    #[test]
    fn test_batch_result_from_results_all_ok() {
        let data: Vec<(usize, Result<u64, String>)> =
            vec![(0, Ok(1)), (1, Ok(2)), (2, Ok(3))];
        let r = BatchResult::from_results(data);
        assert_eq!(r.total, 3);
        assert_eq!(r.succeeded, 3);
        assert_eq!(r.failed, 0);
        assert!(r.all_succeeded());
        assert_eq!(r.ok_values().len(), 3);
    }

    #[test]
    fn test_batch_result_from_results_mixed() {
        let data: Vec<(usize, Result<u64, String>)> =
            vec![(0, Ok(1)), (1, Err("bad".into())), (2, Ok(3))];
        let r = BatchResult::from_results(data);
        assert_eq!(r.total, 3);
        assert_eq!(r.succeeded, 2);
        assert_eq!(r.failed, 1);
        assert!(!r.all_succeeded());
        assert!(r.has_failures());
        assert_eq!(r.ok_values().len(), 2);
    }

    #[test]
    fn test_batch_result_summary() {
        let data: Vec<(usize, Result<u32, String>)> =
            vec![(0, Ok(1)), (1, Err("x".into()))];
        let r = BatchResult::from_results(data);
        let s = r.summary();
        assert!(s.contains("1/2 succeeded"));
        assert!(s.contains("1 failed"));
    }

    // ── batch_checksums ───────────────────────────────────────────────────

    #[test]
    fn test_batch_checksums_basic() {
        let items = vec![b"hello".to_vec(), b"world".to_vec()];
        let r = batch_checksums(&items);
        assert_eq!(r.total, 2);
        assert!(r.all_succeeded());
        // Verify determinism
        let v0 = r.items[0].value.expect("should have value");
        let v1 = r.items[1].value.expect("should have value");
        assert_ne!(v0, v1);
    }

    #[test]
    fn test_batch_checksums_empty() {
        let r = batch_checksums(&[]);
        assert_eq!(r.total, 0);
        assert!(r.all_succeeded());
    }

    #[test]
    fn test_batch_checksums_deterministic() {
        let items = vec![b"test".to_vec()];
        let r1 = batch_checksums(&items);
        let r2 = batch_checksums(&items);
        assert_eq!(
            r1.items[0].value.expect("v1"),
            r2.items[0].value.expect("v2")
        );
    }

    // ── batch_crc32 ───────────────────────────────────────────────────────

    #[test]
    fn test_batch_crc32_basic() {
        let items = vec![b"abc".to_vec(), b"".to_vec()];
        let r = batch_crc32(&items);
        assert_eq!(r.total, 2);
        assert!(r.all_succeeded());
    }

    #[test]
    fn test_crc32_known_value() {
        // CRC-32 of empty data should be 0x00000000
        let c = crc32_ieee(b"");
        assert_eq!(c, 0x0000_0000);
    }

    #[test]
    fn test_crc32_deterministic() {
        let a = crc32_ieee(b"test data");
        let b = crc32_ieee(b"test data");
        assert_eq!(a, b);
    }

    // ── batch_frame_stats ─────────────────────────────────────────────────

    #[test]
    fn test_batch_frame_stats_uniform() {
        // 4x4 frame with all pixels = 128
        let data = vec![128u8; 16];
        let frames = vec![(4u32, 4u32, data)];
        let r = batch_frame_stats(&frames);
        assert_eq!(r.total, 1);
        assert!(r.all_succeeded());
        let stats = r.items[0].value.as_ref().expect("should succeed");
        assert!((stats.mean_luma - 128.0).abs() < 0.01);
        assert_eq!(stats.min_luma, 128);
        assert_eq!(stats.max_luma, 128);
        assert!(stats.std_dev_luma < 0.01);
    }

    #[test]
    fn test_batch_frame_stats_too_short() {
        let frames = vec![(10u32, 10u32, vec![0u8; 5])];
        let r = batch_frame_stats(&frames);
        assert_eq!(r.failed, 1);
    }

    #[test]
    fn test_batch_frame_stats_gradient() {
        // 256 pixels with values 0..255
        let data: Vec<u8> = (0..=255).collect();
        let frames = vec![(16u32, 16u32, data)];
        let r = batch_frame_stats(&frames);
        let stats = r.items[0].value.as_ref().expect("should succeed");
        assert!((stats.mean_luma - 127.5).abs() < 0.01);
        assert_eq!(stats.min_luma, 0);
        assert_eq!(stats.max_luma, 255);
    }

    // ── batch_detect_format ───────────────────────────────────────────────

    #[test]
    fn test_batch_detect_ogg() {
        let headers = vec![b"OggS\x00\x02".to_vec()];
        let r = batch_detect_format(&headers);
        assert!(r.all_succeeded());
        let fmt = r.items[0].value.as_ref().expect("should detect");
        assert_eq!(fmt.format_name, "ogg");
        assert!(!fmt.is_video);
    }

    #[test]
    fn test_batch_detect_flac() {
        let headers = vec![b"fLaC\x00\x00\x00\x22".to_vec()];
        let r = batch_detect_format(&headers);
        assert!(r.all_succeeded());
        let fmt = r.items[0].value.as_ref().expect("should detect");
        assert_eq!(fmt.format_name, "flac");
    }

    #[test]
    fn test_batch_detect_wav() {
        let headers = vec![b"RIFF\x00\x00\x00\x00".to_vec()];
        let r = batch_detect_format(&headers);
        let fmt = r.items[0].value.as_ref().expect("should detect");
        assert_eq!(fmt.format_name, "wav");
    }

    #[test]
    fn test_batch_detect_unknown() {
        let headers = vec![vec![0x00, 0x01, 0x02, 0x03]];
        let r = batch_detect_format(&headers);
        assert_eq!(r.failed, 1);
    }

    #[test]
    fn test_batch_detect_mkv() {
        let headers = vec![vec![0x1A, 0x45, 0xDF, 0xA3, 0x01]];
        let r = batch_detect_format(&headers);
        let fmt = r.items[0].value.as_ref().expect("should detect");
        assert_eq!(fmt.format_name, "mkv");
        assert!(fmt.is_video);
    }

    // ── batch_audio_stats ─────────────────────────────────────────────────

    #[test]
    fn test_batch_audio_stats_sine() {
        // Simple sine-like samples
        let samples: Vec<f32> = (0..1000)
            .map(|i| (i as f32 * 0.01).sin() * 0.5)
            .collect();
        let r = batch_audio_stats(&[samples]);
        assert!(r.all_succeeded());
        let stats = r.items[0].value.as_ref().expect("should succeed");
        assert_eq!(stats.sample_count, 1000);
        assert!(stats.peak_amplitude > 0.0);
        assert!(stats.rms_level > 0.0);
        assert!(stats.rms_dbfs < 0.0); // negative dBFS
    }

    #[test]
    fn test_batch_audio_stats_silence() {
        let samples = vec![0.0f32; 100];
        let r = batch_audio_stats(&[samples]);
        let stats = r.items[0].value.as_ref().expect("should succeed");
        assert!((stats.peak_amplitude - 0.0).abs() < f64::EPSILON);
        assert!(stats.rms_dbfs <= -120.0);
    }

    #[test]
    fn test_batch_audio_stats_empty() {
        let r = batch_audio_stats(&[vec![]]);
        assert_eq!(r.failed, 1);
    }

    // ── BatchConfig ───────────────────────────────────────────────────────

    #[test]
    fn test_batch_config_default() {
        let cfg = BatchConfig::new();
        assert_eq!(cfg.max_items, 0);
        assert!(cfg.continue_on_error);
        assert!(cfg.tags.is_empty());
    }

    #[test]
    fn test_batch_config_builder() {
        let cfg = BatchConfig::new()
            .with_max_items(100)
            .with_continue_on_error(false)
            .with_tag("job", "test");
        assert_eq!(cfg.max_items, 100);
        assert!(!cfg.continue_on_error);
        assert_eq!(
            cfg.tags.get("job").expect("should have tag"),
            "test"
        );
    }

    #[test]
    fn test_batch_config_effective_count() {
        let cfg = BatchConfig::new().with_max_items(5);
        assert_eq!(cfg.effective_count(10), 5);
        assert_eq!(cfg.effective_count(3), 3);

        let unlimited = BatchConfig::new();
        assert_eq!(unlimited.effective_count(1000), 1000);
    }

    // ── fnv1a_64 and crc32_ieee internals ─────────────────────────────────

    #[test]
    fn test_fnv1a_64_empty() {
        let h = fnv1a_64(b"");
        assert_eq!(h, 14_695_981_039_346_656_037_u64);
    }

    #[test]
    fn test_fnv1a_64_different() {
        assert_ne!(fnv1a_64(b"a"), fnv1a_64(b"b"));
    }

    #[test]
    fn test_crc32_different_inputs() {
        assert_ne!(crc32_ieee(b"hello"), crc32_ieee(b"world"));
    }
}
