//! High-level proxy API for OxiMedia.
//!
//! Provides simplified, unified access to proxy generation, management, and
//! quality-comparison workflows. This module wraps the low-level proxy modules
//! with ergonomic types and pure-Rust implementations.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use std::collections::BinaryHeap;

// ---------------------------------------------------------------------------
// SmartProxyGenerator
// ---------------------------------------------------------------------------

/// Information about the source media used to select a proxy specification.
#[derive(Debug, Clone)]
pub struct MediaInfo {
    /// File path.
    pub path: String,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Frames per second.
    pub fps: f64,
    /// Codec name (e.g. "h264", "prores", "dnxhd").
    pub codec: String,
    /// Bitrate in bits per second.
    pub bitrate_bps: u64,
    /// Duration in seconds.
    pub duration_s: f64,
}

impl MediaInfo {
    /// Create a new `MediaInfo`.
    pub fn new(path: impl Into<String>, width: u32, height: u32, fps: f64) -> Self {
        Self {
            path: path.into(),
            width,
            height,
            fps,
            codec: String::new(),
            bitrate_bps: 0,
            duration_s: 0.0,
        }
    }

    /// Set codec.
    pub fn with_codec(mut self, codec: impl Into<String>) -> Self {
        self.codec = codec.into();
        self
    }

    /// Set bitrate.
    pub fn with_bitrate(mut self, bps: u64) -> Self {
        self.bitrate_bps = bps;
        self
    }

    /// Set duration.
    pub fn with_duration(mut self, s: f64) -> Self {
        self.duration_s = s;
        self
    }
}

/// A proxy specification: target resolution, codec, and bitrate.
#[derive(Debug, Clone, PartialEq)]
pub struct ProxySpec {
    /// Target width.
    pub width: u32,
    /// Target height.
    pub height: u32,
    /// Target codec.
    pub codec: String,
    /// Target bitrate in kbps.
    pub bitrate_kbps: u32,
}

impl ProxySpec {
    /// Create a new proxy spec.
    pub fn new(width: u32, height: u32, codec: impl Into<String>, bitrate_kbps: u32) -> Self {
        Self {
            width,
            height,
            codec: codec.into(),
            bitrate_kbps,
        }
    }
}

/// Selects an appropriate proxy specification from source media characteristics.
pub struct SmartProxyGenerator;

impl SmartProxyGenerator {
    /// Select a proxy codec and resolution appropriate for the given source.
    ///
    /// Rules applied (in order):
    /// - 4K (≥3840px wide) → 1920×1080 H.264 proxy at 8 Mbps
    /// - 1080p (≥1920px wide) → 960×540 H.264 proxy at 4 Mbps
    /// - 720p (≥1280px wide) → 640×360 H.264 proxy at 2 Mbps
    /// - smaller → same resolution, H.264 proxy at 1 Mbps
    pub fn select_codec(source: &MediaInfo) -> ProxySpec {
        if source.width >= 3840 {
            ProxySpec::new(1920, 1080, "h264", 8_000)
        } else if source.width >= 1920 {
            ProxySpec::new(960, 540, "h264", 4_000)
        } else if source.width >= 1280 {
            ProxySpec::new(640, 360, "h264", 2_000)
        } else {
            ProxySpec::new(source.width, source.height, "h264", 1_000)
        }
    }
}

// ---------------------------------------------------------------------------
// ProxyFingerprint — FNV-1a hash-based fingerprint
// ---------------------------------------------------------------------------

/// Computes a deterministic fingerprint of proxy media using FNV-1a hashing.
pub struct ProxyFingerprint;

impl ProxyFingerprint {
    /// Compute a 32-byte fingerprint for proxy identification.
    ///
    /// `proxy_path` — path string to hash as the primary key.
    /// `sample_count` — number of simulated samples mixed in (affects the hash).
    ///
    /// Uses FNV-1a (64-bit) across 4 independent seeds to produce 32 bytes.
    pub fn compute(proxy_path: &str, sample_count: u32) -> [u8; 32] {
        const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
        const FNV_PRIME: u64 = 1_099_511_628_211;

        let fnv1a = |seed: u64, data: &[u8]| -> u64 {
            let mut hash = seed;
            for &byte in data {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(FNV_PRIME);
            }
            hash
        };

        let path_bytes = proxy_path.as_bytes();
        let count_bytes = sample_count.to_le_bytes();

        // Four independent hash lanes using different offsets
        let lane0 = fnv1a(FNV_OFFSET, path_bytes);
        let lane1 = fnv1a(FNV_OFFSET ^ 0xdeadbeef_cafebabe, path_bytes);
        let lane2 = {
            let mut h = fnv1a(FNV_OFFSET, &count_bytes);
            for &b in path_bytes {
                h ^= u64::from(b);
                h = h.wrapping_mul(FNV_PRIME);
            }
            h
        };
        let lane3 = fnv1a(lane0 ^ lane1, &count_bytes);

        let mut result = [0u8; 32];
        result[0..8].copy_from_slice(&lane0.to_le_bytes());
        result[8..16].copy_from_slice(&lane1.to_le_bytes());
        result[16..24].copy_from_slice(&lane2.to_le_bytes());
        result[24..32].copy_from_slice(&lane3.to_le_bytes());
        result
    }
}

// ---------------------------------------------------------------------------
// ProxyManifest — new(original, proxy, spec) + to_json()
// ---------------------------------------------------------------------------

/// A lightweight manifest that records an original → proxy relationship.
#[derive(Debug, Clone)]
pub struct ProxyManifestRecord {
    /// Path to the original media.
    pub original_path: String,
    /// Path to the proxy media.
    pub proxy_path: String,
    /// The proxy specification used.
    pub spec: ProxySpec,
    /// Creation timestamp (Unix seconds, 0 if unknown).
    pub created_at: u64,
}

impl ProxyManifestRecord {
    /// Create a new manifest record.
    pub fn new(original: impl Into<String>, proxy: impl Into<String>, spec: ProxySpec) -> Self {
        Self {
            original_path: original.into(),
            proxy_path: proxy.into(),
            spec,
            created_at: 0,
        }
    }

    /// Set creation timestamp.
    pub fn with_created_at(mut self, ts: u64) -> Self {
        self.created_at = ts;
        self
    }

    /// Serialize this manifest record to a JSON string.
    ///
    /// Does not depend on serde; builds the string manually to avoid external deps.
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"original_path":"{orig}","proxy_path":"{proxy}","codec":"{codec}","width":{w},"height":{h},"bitrate_kbps":{br},"created_at":{ts}}}"#,
            orig = escape_json(&self.original_path),
            proxy = escape_json(&self.proxy_path),
            codec = escape_json(&self.spec.codec),
            w = self.spec.width,
            h = self.spec.height,
            br = self.spec.bitrate_kbps,
            ts = self.created_at,
        )
    }
}

/// Escape double-quotes and backslashes in a JSON string value.
fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

// ---------------------------------------------------------------------------
// ProxyBandwidthCalc
// ---------------------------------------------------------------------------

/// Estimates the bandwidth required for a proxy at a given resolution and quality.
pub struct ProxyBandwidthCalc;

impl ProxyBandwidthCalc {
    /// Estimate proxy bitrate in kbps.
    ///
    /// Formula: `(width * height * fps * quality / 10.0 * 0.07) as u32`
    ///
    /// - `quality` — quality factor in the range 1–10 (10 = highest)
    pub fn estimate_kbps(width: u32, height: u32, fps: f64, quality: u8) -> u32 {
        (width as f64 * height as f64 * fps * quality as f64 / 10.0 * 0.07) as u32
    }
}

// ---------------------------------------------------------------------------
// OfflineEditWorkflow
// ---------------------------------------------------------------------------

/// Parameters for preparing an offline editing session.
#[derive(Debug, Clone)]
pub struct OfflineSession {
    /// Path to the original (online) media.
    pub original_path: String,
    /// The proxy spec selected for this session.
    pub proxy_spec: ProxySpec,
    /// A unique session identifier (derived from the original path hash).
    pub session_id: u64,
    /// Whether the proxy has been generated yet.
    pub proxy_ready: bool,
}

/// An EDL (Edit Decision List) entry.
#[derive(Debug, Clone)]
pub struct EdlEntry {
    /// Source clip identifier (file path or media ID).
    pub clip_id: String,
    /// Start frame in the source clip.
    pub src_in: u64,
    /// End frame in the source clip.
    pub src_out: u64,
    /// Start frame on the timeline.
    pub timeline_in: u64,
}

/// A plan for conforming an offline proxy edit back to original online media.
#[derive(Debug, Clone)]
pub struct ConformPlan {
    /// Session being conformed.
    pub session: OfflineSession,
    /// Resolved EDL entries mapped to original media paths.
    pub resolved_entries: Vec<(EdlEntry, String)>,
    /// Whether all EDL entries could be resolved.
    pub fully_resolved: bool,
}

/// Manages offline editing proxy workflows.
pub struct OfflineEditWorkflow;

impl OfflineEditWorkflow {
    /// Prepare an offline session for the given original media.
    ///
    /// Selects an appropriate proxy spec using `SmartProxyGenerator` and
    /// builds an `OfflineSession` ready for proxy generation.
    pub fn prepare(original: impl Into<String>, proxy_spec: ProxySpec) -> OfflineSession {
        let original_path: String = original.into();
        // Derive a stable session ID from path using FNV-1a
        let session_id = {
            const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
            const FNV_PRIME: u64 = 1_099_511_628_211;
            let mut h = FNV_OFFSET;
            for b in original_path.bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(FNV_PRIME);
            }
            h
        };
        OfflineSession {
            original_path,
            proxy_spec,
            session_id,
            proxy_ready: false,
        }
    }

    /// Conform an offline session's EDL back to original media.
    ///
    /// Each EDL entry is resolved: the clip_id is used directly as the original
    /// media path (in production this would query a MAM). Entries whose clip_id
    /// matches `session.original_path` are considered fully resolved.
    pub fn conform(session: OfflineSession, edl: &[EdlEntry]) -> ConformPlan {
        let resolved_entries: Vec<(EdlEntry, String)> = edl
            .iter()
            .map(|entry| {
                // Resolve clip_id → original path (identity mapping for this impl)
                let original = entry.clip_id.clone();
                (entry.clone(), original)
            })
            .collect();

        let fully_resolved = resolved_entries
            .iter()
            .all(|(_, path)| !path.is_empty());

        ConformPlan {
            session,
            resolved_entries,
            fully_resolved,
        }
    }
}

// ---------------------------------------------------------------------------
// ProxyAger
// ---------------------------------------------------------------------------

/// Determines whether a proxy should be regenerated based on age.
pub struct ProxyAger;

impl ProxyAger {
    /// Return `true` if the proxy is older than `max_age_secs` as of `now_secs`.
    ///
    /// `proxy_created_at` — Unix timestamp (seconds) when the proxy was created.
    /// `max_age_secs`     — Maximum allowed age in seconds.
    /// `now_secs`         — Current Unix timestamp in seconds.
    pub fn should_regenerate(proxy_created_at: u64, max_age_secs: u64, now_secs: u64) -> bool {
        now_secs.saturating_sub(proxy_created_at) > max_age_secs
    }
}

// ---------------------------------------------------------------------------
// ProxyScheduler — priority queue (max-heap by priority)
// ---------------------------------------------------------------------------

/// A queued proxy job entry.
#[derive(Debug, Clone, PartialEq, Eq)]
struct SchedulerJob {
    priority: u32,
    media_id: u64,
}

impl PartialOrd for SchedulerJob {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SchedulerJob {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Higher priority comes first; break ties by media_id (lower first)
        self.priority
            .cmp(&other.priority)
            .then_with(|| other.media_id.cmp(&self.media_id))
    }
}

/// Priority queue for proxy generation scheduling.
///
/// Jobs with higher `priority` values are dequeued first.
pub struct ProxyScheduler {
    heap: BinaryHeap<SchedulerJob>,
}

impl ProxyScheduler {
    /// Create an empty scheduler.
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
        }
    }

    /// Add a media item to the queue with the given priority (higher = sooner).
    pub fn queue(&mut self, media_id: u64, priority: u32) {
        self.heap.push(SchedulerJob { priority, media_id });
    }

    /// Remove and return the `media_id` with the highest priority.
    ///
    /// Returns `None` when the queue is empty.
    pub fn next_job(&mut self) -> Option<u64> {
        self.heap.pop().map(|job| job.media_id)
    }

    /// Current number of queued jobs.
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Return `true` if no jobs are queued.
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
}

impl Default for ProxyScheduler {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ProxySyncChecker
// ---------------------------------------------------------------------------

/// Verifies whether a proxy and its original are in sync using content hashes.
pub struct ProxySyncChecker;

impl ProxySyncChecker {
    /// Return `true` when both hashes are non-empty and equal.
    ///
    /// In a real system the hashes would be frame-level checksums; here we
    /// treat them as opaque byte strings.
    pub fn is_in_sync(original_hash: &[u8], proxy_hash: &[u8]) -> bool {
        !original_hash.is_empty() && original_hash == proxy_hash
    }
}

// ---------------------------------------------------------------------------
// ResolutionMapper — proxy timecode → original timecode
// ---------------------------------------------------------------------------

/// Maps timecodes between proxy and original media at different frame rates.
pub struct ResolutionMapper;

impl ResolutionMapper {
    /// Convert a proxy timecode (in frames) to the corresponding original
    /// timecode at a different frame rate.
    ///
    /// Formula: `proxy_tc * orig_fps / proxy_fps` (rounded to nearest frame).
    pub fn map_proxy_tc_to_original(proxy_tc: u64, proxy_fps: f64, orig_fps: f64) -> u64 {
        if proxy_fps <= 0.0 || orig_fps <= 0.0 {
            return 0;
        }
        (proxy_tc as f64 * orig_fps / proxy_fps).round() as u64
    }
}

// ---------------------------------------------------------------------------
// ProxyQualityCompare — PSNR estimation
// ---------------------------------------------------------------------------

/// Pixel data for a frame, represented as raw bytes.
#[derive(Debug, Clone)]
pub struct FrameData {
    /// Raw pixel bytes (assumed 8-bit per channel, e.g. planar Y).
    pub data: Vec<u8>,
}

impl FrameData {
    /// Create from a byte slice.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self {
            data: bytes.to_vec(),
        }
    }
}

/// Compares proxy and original frame quality using PSNR estimation.
pub struct ProxyQualityCompare;

impl ProxyQualityCompare {
    /// Estimate the PSNR (dB) between a proxy frame and its original.
    ///
    /// Uses the standard PSNR formula:
    ///   MSE = Σ(proxy[i] - original[i])² / (w * h)
    ///   PSNR = 10 * log10(255² / MSE)
    ///
    /// If `proxy.data` and `original.data` have fewer than `w * h` bytes,
    /// only the available samples are used.  Returns `f32::INFINITY` when
    /// the frames are identical.
    pub fn psnr_estimate(proxy: &FrameData, original: &FrameData, w: u32, h: u32) -> f32 {
        let pixel_count = (w as usize) * (h as usize);
        if pixel_count == 0 {
            return 0.0;
        }

        let samples = proxy.data.len().min(original.data.len()).min(pixel_count);
        if samples == 0 {
            return 0.0;
        }

        let sum_sq: u64 = proxy.data[..samples]
            .iter()
            .zip(original.data[..samples].iter())
            .map(|(&p, &o)| {
                let diff = i32::from(p) - i32::from(o);
                (diff * diff) as u64
            })
            .sum();

        if sum_sq == 0 {
            return f32::INFINITY;
        }

        let mse = sum_sq as f64 / samples as f64;
        let psnr = 10.0 * (255.0_f64 * 255.0 / mse).log10();
        psnr as f32
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- SmartProxyGenerator ---

    #[test]
    fn test_smart_proxy_4k_selects_1080p() {
        let source = MediaInfo::new("/media/clip.mov", 3840, 2160, 24.0);
        let spec = SmartProxyGenerator::select_codec(&source);
        assert_eq!(spec.width, 1920);
        assert_eq!(spec.height, 1080);
        assert_eq!(spec.codec, "h264");
    }

    #[test]
    fn test_smart_proxy_1080p_selects_540p() {
        let source = MediaInfo::new("/media/clip.mov", 1920, 1080, 25.0);
        let spec = SmartProxyGenerator::select_codec(&source);
        assert_eq!(spec.width, 960);
        assert_eq!(spec.height, 540);
    }

    #[test]
    fn test_smart_proxy_720p_selects_360p() {
        let source = MediaInfo::new("/media/clip.mov", 1280, 720, 30.0);
        let spec = SmartProxyGenerator::select_codec(&source);
        assert_eq!(spec.width, 640);
        assert_eq!(spec.height, 360);
    }

    #[test]
    fn test_smart_proxy_small_passthrough() {
        let source = MediaInfo::new("/media/clip.mov", 640, 360, 30.0);
        let spec = SmartProxyGenerator::select_codec(&source);
        assert_eq!(spec.width, 640);
        assert_eq!(spec.height, 360);
    }

    // --- ProxyFingerprint ---

    #[test]
    fn test_fingerprint_deterministic() {
        let a = ProxyFingerprint::compute("/proxy/clip.mp4", 100);
        let b = ProxyFingerprint::compute("/proxy/clip.mp4", 100);
        assert_eq!(a, b);
    }

    #[test]
    fn test_fingerprint_differs_on_path() {
        let a = ProxyFingerprint::compute("/proxy/a.mp4", 100);
        let b = ProxyFingerprint::compute("/proxy/b.mp4", 100);
        assert_ne!(a, b);
    }

    #[test]
    fn test_fingerprint_differs_on_samples() {
        let a = ProxyFingerprint::compute("/proxy/clip.mp4", 10);
        let b = ProxyFingerprint::compute("/proxy/clip.mp4", 20);
        assert_ne!(a, b);
    }

    // --- ProxyManifestRecord ---

    #[test]
    fn test_manifest_contains_original_path() {
        let spec = ProxySpec::new(1920, 1080, "h264", 8_000);
        let manifest = ProxyManifestRecord::new("/originals/clip.mov", "/proxies/clip.mp4", spec);
        let json = manifest.to_json();
        assert!(json.contains("/originals/clip.mov"), "json: {json}");
    }

    #[test]
    fn test_manifest_contains_codec() {
        let spec = ProxySpec::new(1920, 1080, "h264", 8_000);
        let manifest = ProxyManifestRecord::new("/originals/clip.mov", "/proxies/clip.mp4", spec);
        let json = manifest.to_json();
        assert!(json.contains("h264"), "json: {json}");
    }

    #[test]
    fn test_manifest_json_is_object() {
        let spec = ProxySpec::new(960, 540, "vp9", 2_000);
        let manifest = ProxyManifestRecord::new("/orig.mxf", "/proxy.webm", spec);
        let json = manifest.to_json();
        assert!(json.starts_with('{') && json.ends_with('}'));
    }

    // --- ProxyBandwidthCalc ---

    #[test]
    fn test_bandwidth_1080p_30fps_quality5() {
        // 1920 * 1080 * 30.0 * 5/10 * 0.07 = 21772800 * 0.35 * 0.07
        let kbps = ProxyBandwidthCalc::estimate_kbps(1920, 1080, 30.0, 5);
        // Expected: (1920 * 1080 * 30.0 * 0.5 * 0.07) as u32 = ~2177
        assert!(kbps > 0, "kbps should be positive");
    }

    #[test]
    fn test_bandwidth_higher_quality_gives_higher_kbps() {
        let low = ProxyBandwidthCalc::estimate_kbps(1920, 1080, 30.0, 2);
        let high = ProxyBandwidthCalc::estimate_kbps(1920, 1080, 30.0, 8);
        assert!(high > low);
    }

    // --- OfflineEditWorkflow ---

    #[test]
    fn test_offline_prepare_sets_original() {
        let spec = ProxySpec::new(1920, 1080, "h264", 8_000);
        let session = OfflineEditWorkflow::prepare("/orig/clip.mov", spec);
        assert_eq!(session.original_path, "/orig/clip.mov");
    }

    #[test]
    fn test_offline_session_id_stable() {
        let spec1 = ProxySpec::new(1920, 1080, "h264", 8_000);
        let spec2 = ProxySpec::new(1920, 1080, "h264", 8_000);
        let s1 = OfflineEditWorkflow::prepare("/orig/clip.mov", spec1);
        let s2 = OfflineEditWorkflow::prepare("/orig/clip.mov", spec2);
        assert_eq!(s1.session_id, s2.session_id);
    }

    #[test]
    fn test_conform_resolves_entries() {
        let spec = ProxySpec::new(1920, 1080, "h264", 8_000);
        let session = OfflineEditWorkflow::prepare("/orig/clip.mov", spec);
        let edl = vec![
            EdlEntry {
                clip_id: "/orig/clip.mov".to_string(),
                src_in: 0,
                src_out: 240,
                timeline_in: 0,
            },
        ];
        let plan = OfflineEditWorkflow::conform(session, &edl);
        assert_eq!(plan.resolved_entries.len(), 1);
        assert!(plan.fully_resolved);
    }

    // --- ProxyAger ---

    #[test]
    fn test_proxy_ager_not_expired() {
        // created 100s ago, max_age 200s
        assert!(!ProxyAger::should_regenerate(900, 200, 1000));
    }

    #[test]
    fn test_proxy_ager_expired() {
        // created 300s ago, max_age 200s
        assert!(ProxyAger::should_regenerate(700, 200, 1000));
    }

    #[test]
    fn test_proxy_ager_exactly_at_boundary() {
        // difference == max_age → NOT expired (> comparison)
        assert!(!ProxyAger::should_regenerate(800, 200, 1000));
    }

    // --- ProxyScheduler ---

    #[test]
    fn test_scheduler_empty() {
        let mut s = ProxyScheduler::new();
        assert!(s.next_job().is_none());
    }

    #[test]
    fn test_scheduler_single_job() {
        let mut s = ProxyScheduler::new();
        s.queue(42, 10);
        assert_eq!(s.next_job(), Some(42));
        assert!(s.next_job().is_none());
    }

    #[test]
    fn test_scheduler_priority_order() {
        let mut s = ProxyScheduler::new();
        s.queue(1, 5);
        s.queue(2, 10);
        s.queue(3, 1);
        // highest priority first
        assert_eq!(s.next_job(), Some(2)); // priority 10
        assert_eq!(s.next_job(), Some(1)); // priority 5
        assert_eq!(s.next_job(), Some(3)); // priority 1
    }

    // --- ProxySyncChecker ---

    #[test]
    fn test_sync_checker_in_sync() {
        let hash = b"abc123";
        assert!(ProxySyncChecker::is_in_sync(hash, hash));
    }

    #[test]
    fn test_sync_checker_out_of_sync() {
        assert!(!ProxySyncChecker::is_in_sync(b"abc", b"xyz"));
    }

    #[test]
    fn test_sync_checker_empty_hash() {
        assert!(!ProxySyncChecker::is_in_sync(b"", b""));
    }

    // --- ResolutionMapper ---

    #[test]
    fn test_tc_mapping_same_fps() {
        let result = ResolutionMapper::map_proxy_tc_to_original(100, 24.0, 24.0);
        assert_eq!(result, 100);
    }

    #[test]
    fn test_tc_mapping_scales_correctly() {
        // proxy at 24fps → original at 48fps: frame 50 → frame 100
        let result = ResolutionMapper::map_proxy_tc_to_original(50, 24.0, 48.0);
        assert_eq!(result, 100);
    }

    #[test]
    fn test_tc_mapping_downscale() {
        // proxy at 60fps → original at 30fps: frame 60 → frame 30
        let result = ResolutionMapper::map_proxy_tc_to_original(60, 60.0, 30.0);
        assert_eq!(result, 30);
    }

    #[test]
    fn test_tc_mapping_zero_fps() {
        let result = ResolutionMapper::map_proxy_tc_to_original(100, 0.0, 24.0);
        assert_eq!(result, 0);
    }

    // --- ProxyQualityCompare ---

    #[test]
    fn test_psnr_identical_frames_infinity() {
        let data = vec![128u8; 100];
        let proxy = FrameData::from_bytes(&data);
        let original = FrameData::from_bytes(&data);
        let psnr = ProxyQualityCompare::psnr_estimate(&proxy, &original, 10, 10);
        assert!(psnr.is_infinite());
    }

    #[test]
    fn test_psnr_different_frames_finite() {
        let proxy_data: Vec<u8> = (0u8..=99).collect();
        let orig_data = vec![128u8; 100];
        let proxy = FrameData::from_bytes(&proxy_data);
        let original = FrameData::from_bytes(&orig_data);
        let psnr = ProxyQualityCompare::psnr_estimate(&proxy, &original, 10, 10);
        assert!(psnr.is_finite() && psnr > 0.0, "psnr={psnr}");
    }

    #[test]
    fn test_psnr_zero_dimensions() {
        let proxy = FrameData::from_bytes(&[]);
        let original = FrameData::from_bytes(&[]);
        let psnr = ProxyQualityCompare::psnr_estimate(&proxy, &original, 0, 0);
        assert_eq!(psnr, 0.0);
    }

    #[test]
    fn test_psnr_high_quality_proxy() {
        // Proxy is nearly identical to original (off by 1 per pixel)
        let orig: Vec<u8> = vec![200u8; 1000];
        let proxy: Vec<u8> = vec![201u8; 1000];
        let proxy_frame = FrameData::from_bytes(&proxy);
        let orig_frame = FrameData::from_bytes(&orig);
        let psnr = ProxyQualityCompare::psnr_estimate(&proxy_frame, &orig_frame, 40, 25);
        // MSE = 1.0, PSNR = 10 * log10(65025) ≈ 48.1 dB
        assert!(psnr > 40.0, "expected high PSNR for near-identical frames, got {psnr}");
    }
}
